import logging
import os
import sys
import random
import numpy as np
import torch
import evaluate
from typing import NoReturn

from ..config import DataTrainingArguments, ModelArguments
from datasets import DatasetDict, load_from_disk
from .trainer_qa import QuestionAnsweringTrainer
from transformers import (
    AutoConfig,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    TrainingArguments,
    set_seed,
)
from ..utils import check_no_error, postprocess_qa_predictions
from transformers.models.roberta.modeling_roberta import RobertaModel, RobertaPreTrainedModel
from torch.nn import CrossEntropyLoss
import torch.nn as nn
import wandb

# PyTorch 2.6+ 호환성: torch.load의 weights_only 기본값 변경 대응
# numpy 객체를 안전하게 로드할 수 있도록 설정
if hasattr(torch.serialization, 'add_safe_globals'):
    try:
        import numpy.core.multiarray
        torch.serialization.add_safe_globals([numpy.core.multiarray._reconstruct])
    except (AttributeError, ImportError):
        pass

seed = 2024
deterministic = False

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
if deterministic:
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False

logger = logging.getLogger(__name__)

def main():
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    if training_args.report_to and "wandb" in training_args.report_to:
        wandb.init(
            config={
                "learning_rate": training_args.learning_rate,
                "num_train_epochs": training_args.num_train_epochs,
                "per_device_train_batch_size": training_args.per_device_train_batch_size,
                "gradient_accumulation_steps": training_args.gradient_accumulation_steps,
                "lr_scheduler_type": training_args.lr_scheduler_type,
                "model_name": model_args.model_name_or_path,
            }
        )


    print(model_args.model_name_or_path)
    print(f"model is from {model_args.model_name_or_path}")
    print(f"data is from {data_args.dataset_name}")

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -    %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.info("Training/evaluation parameters %s", training_args)

    set_seed(training_args.seed)
    datasets = load_from_disk(data_args.dataset_name)
    print(datasets)

    model_config_path = model_args.config_name if model_args.config_name else model_args.model_name_or_path
    model_config = AutoConfig.from_pretrained(model_config_path)
    
    tokenizer_model_path = model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_model_path, use_fast=True)
    
    is_tensorflow_format = ".ckpt" in model_args.model_name_or_path
    model = RobertaCNNForQuestionAnswering.from_pretrained(
        model_args.model_name_or_path,
        from_tf=is_tensorflow_format,
        config=model_config,
    )

    print(type(training_args), type(model_args), type(datasets), type(tokenizer), type(model))

    should_run_mrc = training_args.do_train or training_args.do_eval
    if should_run_mrc:
        run_mrc(data_args, training_args, model_args, datasets, tokenizer, model)

def run_mrc(
    data_args: DataTrainingArguments,
    training_args: TrainingArguments,
    model_args: ModelArguments,
    datasets: DatasetDict,
    tokenizer,
    model,
) -> NoReturn:

    is_training = training_args.do_train
    if is_training:
        dataset_columns = datasets["train"].column_names
    else:
        dataset_columns = datasets["validation"].column_names

    has_question = "question" in dataset_columns
    has_context = "context" in dataset_columns
    has_answers = "answers" in dataset_columns
    
    q_col = "question" if has_question else dataset_columns[0]
    ctx_col = "context" if has_context else dataset_columns[1]
    ans_col = "answers" if has_answers else dataset_columns[2]

    padding_right = tokenizer.padding_side == "right"

    last_checkpoint, max_seq_length = check_no_error(
        data_args, training_args, datasets, tokenizer
    )

    def prepare_train_features(examples):
        first_sequence = examples[q_col if padding_right else ctx_col]
        second_sequence = examples[ctx_col if padding_right else q_col]
        trunc_strategy = "only_second" if padding_right else "only_first"
        pad_strategy = "max_length" if data_args.pad_to_max_length else False
        
        tokenized_examples = tokenizer(
            first_sequence,
            second_sequence,
            truncation=trunc_strategy,
            max_length=max_seq_length,
            stride=data_args.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            return_token_type_ids=False, # roberta모델을 사용할 경우 False, bert를 사용할 경우 True로 표기해야합니다.           
            padding=pad_strategy,
        )

        overflow_map = tokenized_examples.pop("overflow_to_sample_mapping")
        offset_maps = tokenized_examples.pop("offset_mapping")

        tokenized_examples["start_positions"] = []
        tokenized_examples["end_positions"] = []

        for example_idx, offsets in enumerate(offset_maps):
            input_token_ids = tokenized_examples["input_ids"][example_idx]
            cls_idx = input_token_ids.index(tokenizer.cls_token_id)
            sequence_id_list = tokenized_examples.sequence_ids(example_idx)
            original_idx = overflow_map[example_idx]
            answer_info = examples[ans_col][original_idx]

            answer_exists = len(answer_info["answer_start"]) > 0
            if not answer_exists:
                tokenized_examples["start_positions"].append(cls_idx)
                tokenized_examples["end_positions"].append(cls_idx)
            else:
                char_start = answer_info["answer_start"][0]
                answer_str = answer_info["text"][0]
                char_end = char_start + len(answer_str)

                ctx_start_pos = 0
                ctx_identifier = 1 if padding_right else 0
                while sequence_id_list[ctx_start_pos] != ctx_identifier:
                    ctx_start_pos += 1

                ctx_end_pos = len(input_token_ids) - 1
                while sequence_id_list[ctx_end_pos] != ctx_identifier:
                    ctx_end_pos -= 1

                answer_within_span = (
                    offsets[ctx_start_pos][0] <= char_start
                    and offsets[ctx_end_pos][1] >= char_end
                )
                
                if not answer_within_span:
                    tokenized_examples["start_positions"].append(cls_idx)
                    tokenized_examples["end_positions"].append(cls_idx)
                else:
                    while ctx_start_pos < len(offsets) and offsets[ctx_start_pos][0] <= char_start:
                        ctx_start_pos += 1
                    tokenized_examples["start_positions"].append(ctx_start_pos - 1)
                    while offsets[ctx_end_pos][1] >= char_end:
                        ctx_end_pos -= 1
                    tokenized_examples["end_positions"].append(ctx_end_pos + 1)

        return tokenized_examples

    processed_train_data = None
    if is_training:
        if "train" not in datasets:
            raise ValueError("--do_train requires a train dataset")
        training_data = datasets["train"]

        processed_train_data = training_data.map(
            prepare_train_features,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=dataset_columns,
            load_from_cache_file=not data_args.overwrite_cache,
        )

    def prepare_validation_features(examples):
        q_seq = examples[q_col if padding_right else ctx_col]
        c_seq = examples[ctx_col if padding_right else q_col]
        trunc_mode = "only_second" if padding_right else "only_first"
        pad_mode = "max_length" if data_args.pad_to_max_length else False
        
        tokenized_examples = tokenizer(
            q_seq,
            c_seq,
            truncation=trunc_mode,
            max_length=max_seq_length,
            stride=data_args.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            return_token_type_ids=False, # roberta모델을 사용할 경우 False, bert를 사용할 경우 True로 표기해야합니다.           
            padding=pad_mode,
        )

        overflow_to_sample = tokenized_examples.pop("overflow_to_sample_mapping")
        tokenized_examples["example_id"] = []

        total_examples = len(tokenized_examples["input_ids"])
        for example_index in range(total_examples):
            seq_id_list = tokenized_examples.sequence_ids(example_index)
            context_id = 1 if padding_right else 0
            orig_sample_index = overflow_to_sample[example_index]
            tokenized_examples["example_id"].append(examples["id"][orig_sample_index])
            
            current_offsets = tokenized_examples["offset_mapping"][example_index]
            updated_offsets = []
            for pos_idx, offset_value in enumerate(current_offsets):
                if seq_id_list[pos_idx] == context_id:
                    updated_offsets.append(offset_value)
                else:
                    updated_offsets.append(None)
            tokenized_examples["offset_mapping"][example_index] = updated_offsets
        return tokenized_examples

    is_evaluating = training_args.do_eval
    processed_val_data = None
    if is_evaluating:
        validation_data = datasets["validation"]

        processed_val_data = validation_data.map(
            prepare_validation_features,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=dataset_columns,
            load_from_cache_file=not data_args.overwrite_cache,
        )

    pad_multiple = 8 if training_args.fp16 else None
    data_collator = DataCollatorWithPadding(
        tokenizer, pad_to_multiple_of=pad_multiple
    )

    def post_processing_function(examples, features, predictions, training_args):
        processed_preds = postprocess_qa_predictions(
            examples=examples,
            features=features,
            predictions=predictions,
            max_answer_length=data_args.max_answer_length,
            output_dir=training_args.output_dir,
            version_2_with_negative=True,  # <--- [핵심] 이 옵션이 있어야 빈 문자열("")을 뱉습니다.
            null_score_diff_threshold=0.0, # [선택] 답 없음으로 판단할 기준점 (기본 0.0)
        )
        formatted_preds = []
        for prediction_id, prediction_text in processed_preds.items():
            formatted_preds.append({"id": prediction_id, "prediction_text": prediction_text})
        
        is_predicting = training_args.do_predict
        if is_predicting:
            return formatted_preds

        elif is_evaluating:
            ref_list = []         # 정답지 (References) - 기존 변수명 유지
            final_preds = []      # 예측값 (Predictions) - 짝을 맞추기 위해 새로 정의

            for val_example in datasets["validation"]:
                
                # -----------------------------------------------------------
                # 1. 예측값 (Prediction) 담기
                # -----------------------------------------------------------
                # 기존 processed_preds 딕셔너리에서 ID에 맞는 예측 텍스트를 가져옵니다.
                pred_text = processed_preds.get(val_example["id"], "")
                
                final_preds.append({
                    "id": val_example["id"], 
                    "prediction_text": pred_text
                })

                # -----------------------------------------------------------
                # 2. 정답지 (Reference) 담기 (로직 수정됨)
                # -----------------------------------------------------------
                # 기존 변수명(val_example[ans_col]) 활용
                original_answers = val_example[ans_col]

                # [수정] 정답 리스트가 비어있는 경우(Negative) 처리
                # 그냥 넘기면 max() 에러가 나므로, [""](빈 문자열)이 정답인 것으로 변환
                if len(original_answers["text"]) == 0:
                    formatted_answers = {
                        "text": [""],        # "정답은 빈 문자열이다"
                        "answer_start": [-1] # 형식 유지를 위한 더미 값
                    }
                else:
                    # 정답이 있는 경우(Positive)는 원본 그대로 사용
                    formatted_answers = original_answers
                
                # 기존 변수명(ref_list)에 추가
                ref_list.append({
                    "id": val_example["id"], 
                    "answers": formatted_answers
                })
            return EvalPrediction(
                predictions=formatted_preds, label_ids=ref_list
            )

    metric = evaluate.load("squad")

    def compute_metrics(p: EvalPrediction):
        metrics =  metric.compute(predictions=p.predictions, references=p.label_ids)

        metrics["eval_exact_match"] = metrics["exact_match"]
        metrics["eval_f1"] = metrics["f1"]

        return metrics

    qa_trainer = QuestionAnsweringTrainer(
        model=model,
        args=training_args,
        train_dataset=processed_train_data if is_training else None,
        eval_dataset=processed_val_data if is_evaluating else None,
        eval_examples=datasets["validation"] if is_evaluating else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        post_process_function=post_processing_function,
        compute_metrics=compute_metrics,
    )

    if is_training:
        # PyTorch 2.6+ 호환성: 체크포인트에서 RNG state 로드 시 오류 방지
        # 기존 체크포인트가 있으면 무시하고 처음부터 학습
        resume_checkpoint = None
        # 주의: PyTorch 2.6+에서는 체크포인트의 RNG state 로드 시 오류가 발생할 수 있습니다.
        # 체크포인트에서 재개하려면 transformers 라이브러리를 최신 버전으로 업그레이드하거나
        # PyTorch를 2.5 이하로 다운그레이드하세요.
        # if last_checkpoint is not None:
        #     resume_checkpoint = last_checkpoint
        # elif os.path.isdir(model_args.model_name_or_path):
        #     resume_checkpoint = model_args.model_name_or_path
        
        training_results = qa_trainer.train(resume_from_checkpoint=resume_checkpoint)
        # qa_trainer.save_model()

        train_metrics = training_results.metrics
        train_metrics["train_samples"] = len(processed_train_data)

        qa_trainer.log_metrics("train", train_metrics)
        qa_trainer.save_metrics("train", train_metrics)
        qa_trainer.save_state()

        results_file_path = os.path.join(training_args.output_dir, "train_results.txt")

        with open(results_file_path, "w") as file_writer:
            logger.info("***** Train results *****")
            for metric_key, metric_value in sorted(training_results.metrics.items()):
                logger.info(f"  {metric_key} = {metric_value}")
                file_writer.write(f"{metric_key} = {metric_value}\n")

        qa_trainer.state.save_to_json(
            os.path.join(training_args.output_dir, "trainer_state.json")
        )

    if is_evaluating:
        logger.info("*** Evaluate ***")
        eval_metrics = qa_trainer.evaluate()

        eval_metrics["eval_samples"] = len(processed_val_data)

        # wandb에 로깅하기 위해 log() 메서드 사용
        # qa_trainer.log(eval_metrics)
        # qa_trainer.log_metrics("eval", eval_metrics)
        # qa_trainer.save_metrics("eval", eval_metrics)

        wandb_logs = {}
        for key, value in eval_metrics.items():
            # "eval_"로 시작하는 키를 "eval/"로 변경 (예: eval_exact_match -> eval/exact_match)
            if key.startswith("eval_"):
                new_key = key.replace("eval_", "eval/", 1)
            elif key == "epoch":
                new_key = "epoch"
            else:
                # 그 외의 경우 (예: eval_samples -> eval/samples)
                new_key = f"eval/{key}" if not key.startswith("eval/") else key
            
            wandb_logs[new_key] = value

        wandb.log(wandb_logs)
        
        # 파일 저장용은 원본 키(eval_...) 유지
        qa_trainer.save_metrics("eval", eval_metrics)


class RobertaCNNForQuestionAnswering(RobertaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        # 1. RoBERTa 본체
        self.roberta = RobertaModel(config, add_pooling_layer=False)
        
        # 2. CNN 블록 (2개 층)
        self.cnn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(config.hidden_size, config.hidden_size, kernel_size=3, padding=1),
                nn.Conv1d(config.hidden_size, config.hidden_size, kernel_size=1),
                nn.ReLU(),
            ) for _ in range(2)
        ])
        
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(config.hidden_size) for _ in range(2)
        ])

        # 3. 출력층
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

        # 4. 중요: 가중치 초기화 적용
        self.post_init() 
        self._init_cnn_weights() # CNN 전용 초기화 별도 실행

    def _init_cnn_weights(self):
        # Conv1d 레이어들은 RoBERTa 기본 초기화에 포함되지 않을 수 있으므로 별도 초기화
        for module in self.cnn_layers.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.normal_(module.weight, mean=0.0, std=0.001)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
        
        for block in self.cnn_layers:
            # block 구조: [Conv, Conv, ReLU] -> 인덱스 1이 두 번째 Conv
            last_conv = block[1] 
            if isinstance(last_conv, nn.Conv1d):
                nn.init.constant_(last_conv.weight, 0.0)
                if last_conv.bias is not None:
                    nn.init.constant_(last_conv.bias, 0.0)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        start_positions=None,
        end_positions=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0] # (Batch, Seq_Len, Hidden)

        # # CNN 입력 전 NaN 체크 (RoBERTa 자체 발산 방지)
        # sequence_output = torch.nan_to_num(sequence_output, nan=0.0).float()

        # if attention_mask is not None:
        #      expanded_mask = attention_mask.unsqueeze(-1).float()
        #      sequence_output = sequence_output * expanded_mask

        # i = 0
        # for cnn_layer, layer_norm in zip(self.cnn_layers, self.layer_norms):
        #     residual = sequence_output
            
        #     # [수정 2] Transpose 후 contiguous() 필수!
        #     # 메모리 비연속성으로 인한 연산 오류 방지
        #     cnn_input = sequence_output.transpose(1, 2).contiguous()

        #     # [DEBUG] CNN 입력 확인
        #     if torch.isnan(cnn_input).any():
        #         print(f"🚨 [비상] CNN Layer {i} 입력 전 NaN 발견!")
            
        #     cnn_output = cnn_layer(cnn_input)
            
        #     # 다시 돌려놓기 + contiguous
        #     cnn_output = cnn_output.transpose(1, 2).contiguous()
            
        #     if attention_mask is not None:
        #         cnn_output = cnn_output * expanded_mask
            
        #     sequence_output = layer_norm(residual + cnn_output)

        #     i+=1

        with torch.amp.autocast('cuda', enabled=False):
            
            # 들어오자마자 FP32(float)로 옷을 갈아입힙니다.
            sequence_output = sequence_output.float()
            
            # 혹시 모를 NaN 제거
            sequence_output = torch.nan_to_num(sequence_output, nan=0.0, posinf=0.0, neginf=0.0)

            # 마스크 처리 (FP32 상태에서 안전하게)
            if attention_mask is not None:
                extended_mask = attention_mask.unsqueeze(-1).float()
                sequence_output = sequence_output * extended_mask
            
            i = 0
            # CNN 레이어 루프
            for cnn_layer, layer_norm in zip(self.cnn_layers, self.layer_norms):
                residual = sequence_output

                # # [디버깅] 가중치 자체가 NaN인지 확인 (이게 뜨면 이전 스텝 역전파에서 망가진 것)
                for name, param in cnn_layer.named_parameters():
                    # if torch.isnan(param).any() or torch.isinf(param).any():
                    if torch.isnan(param).any():
                        print(f"💀 [사망 신고] CNN Layer {i}의 가중치({name})가 이미 NaN입니다!")
                
                # Transpose + Contiguous
                cnn_input = sequence_output.transpose(1, 2).contiguous()

                # [DEBUG] CNN 입력 확인
                if torch.isnan(cnn_input).any():
                    print(f"🚨 [비상] CNN Layer {i} 입력 전 NaN 발견!")
                
                # CNN 연산 (이제 FP32라서 안 터짐!)
                cnn_output = cnn_layer(cnn_input)
                
                cnn_output = cnn_output.transpose(1, 2).contiguous()

                # [방어 1] CNN 출력값 소독 (여기서 무한대가 자주 나옵니다)
                cnn_output = torch.nan_to_num(cnn_output, nan=0.0, posinf=0.0, neginf=0.0)
                
                # [방어 2] 값 자르기 (Clamp) - Residual 더하기 전에 너무 큰 값 방지
                cnn_output = torch.clamp(cnn_output, min=-10.0, max=10.0)
                
                if attention_mask is not None:
                    cnn_output = cnn_output * extended_mask
                
                # Residual 더하기
                added_output = residual + cnn_output
                
                # LayerNorm 실행
                sequence_output = layer_norm(added_output)

                # [방어 3] LayerNorm 결과 소독 (분산 계산 중 NaN 발생 가능성 차단)
                sequence_output = torch.nan_to_num(sequence_output, nan=0.0, posinf=0.0, neginf=0.0)

                # [DEBUG] 생존 확인
                if torch.isnan(sequence_output).any():
                     print(f"🚨 [비상] CNN Layer {i} 통과 후 여전히 NaN 존재!")

                i += 1
            
            # 너무 큰 값 자르기 (Clamp)
            sequence_output = torch.clamp(sequence_output, min=-15, max=15)
            
            # 출력층 (FP32 상태에서 계산)
            logits = self.qa_outputs(sequence_output)

        
            # NaN이 있으면 0으로 치환하고, 너무 큰 값은 자릅니다.
            sequence_output = torch.nan_to_num(sequence_output, nan=0.0)
            sequence_output = torch.clamp(sequence_output, min=-20, max=20)
            
            logits = self.qa_outputs(sequence_output)
        
            # print(f"DEBUG: Final Logits - Max: {logits.max().item():.4f}, Min: {logits.min().item():.4f}")
        
            start_logits, end_logits = logits.split(1, dim=-1)
            start_logits = start_logits.squeeze(-1)
            end_logits = end_logits.squeeze(-1)

            total_loss = None
            if start_positions is not None and end_positions is not None:
                if len(start_positions.size()) > 1:
                    start_positions = start_positions.squeeze(-1)
                if len(end_positions.size()) > 1:
                    end_positions = end_positions.squeeze(-1)
            
                ignored_index = start_logits.size(1)
                start_positions = start_positions.clamp(0, ignored_index)
                end_positions = end_positions.clamp(0, ignored_index)

                loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
                start_loss = loss_fct(start_logits, start_positions)
                end_loss = loss_fct(end_logits, end_positions)
                total_loss = (start_loss + end_loss) / 2
            
            # Loss가 NaN이면 0이 아니라 에러를 띄우거나 처리가 필요하지만,
            # 보통 초기화만 잘 되면 해결됩니다.

        if not return_dict:
            output = (start_logits, end_logits) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        from transformers.modeling_outputs import QuestionAnsweringModelOutput
        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

if __name__ == "__main__":
    main()