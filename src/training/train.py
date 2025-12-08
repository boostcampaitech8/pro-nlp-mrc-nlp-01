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
            project="reader-parmeter-tuning",  # 프로젝트 이름
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
    model = AutoModelForQuestionAnswering.from_pretrained(
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
        )
        formatted_preds = []
        for prediction_id, prediction_text in processed_preds.items():
            formatted_preds.append({"id": prediction_id, "prediction_text": prediction_text})
        
        is_predicting = training_args.do_predict
        if is_predicting:
            return formatted_preds

        elif is_evaluating:
            ref_list = []
            for val_example in datasets["validation"]:
                ref_list.append({"id": val_example["id"], "answers": val_example[ans_col]})
            return EvalPrediction(
                predictions=formatted_preds, label_ids=ref_list
            )

    metric = evaluate.load("squad")

    def compute_metrics(p: EvalPrediction):
        return metric.compute(predictions=p.predictions, references=p.label_ids)

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
        qa_trainer.log(eval_metrics)
        qa_trainer.log_metrics("eval", eval_metrics)
        qa_trainer.save_metrics("eval", eval_metrics)

if __name__ == "__main__":
    main()
