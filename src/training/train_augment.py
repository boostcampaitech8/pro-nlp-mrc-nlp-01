import logging
import os
import sys
import random
import numpy as np
import torch
import evaluate
from typing import NoReturn, Optional

from ..config import DataTrainingArguments, ModelArguments
from datasets import DatasetDict, load_from_disk, concatenate_datasets, Dataset
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
    import argparse

    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments)
    )

    # argparse for extra augment arguments
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--augment_dataset_name", type=str, default=None, help="Augmented dataset disk path")
    arg_parser.add_argument("--augment_ratio", type=float, default=0.0, help="Proportion of augment dataset to use for training")
    args, unknown_args = arg_parser.parse_known_args()

    model_args, data_args, training_args = parser.parse_args_into_dataclasses(unknown_args)
    
    augment_dataset_name = args.augment_dataset_name
    augment_ratio = args.augment_ratio if args.augment_ratio is not None else 0.0

    if training_args.report_to and "wandb" in training_args.report_to:
        wandb.init(
            project="reader",  # 프로젝트 이름
            config={
                "learning_rate": training_args.learning_rate,
                "num_train_epochs": training_args.num_train_epochs,
                "per_device_train_batch_size": training_args.per_device_train_batch_size,
                "gradient_accumulation_steps": training_args.gradient_accumulation_steps,
                "lr_scheduler_type": training_args.lr_scheduler_type,
                "model_name": model_args.model_name_or_path,
                "augment_dataset_name": augment_dataset_name,
                "augment_ratio": augment_ratio,
            }
        )


    print(model_args.model_name_or_path)
    print(f"model is from {model_args.model_name_or_path}")
    print(f"data is from {data_args.dataset_name}")
    if augment_dataset_name:
        print(f"augment data is from {augment_dataset_name} (ratio: {augment_ratio})")

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -    %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.info("Training/evaluation parameters %s", training_args)

    set_seed(training_args.seed)
    datasets = load_from_disk(data_args.dataset_name)
    print(datasets)

    # augment_dataset 추가 로딩
    augment_datasets = None
    if augment_dataset_name:
        try:
            augment_datasets = load_from_disk(augment_dataset_name)
            print(f"Loaded augment dataset: {augment_datasets}")
        except FileNotFoundError as e:
            logger.warning(
                f"Could not load augment dataset from '{augment_dataset_name}': {e}\n"
                f"Will skip data augmentation and proceed with only the base training data."
            )
            augment_datasets = None
        except Exception as e:
            logger.warning(
                f"Unexpected error while loading augment dataset from '{augment_dataset_name}': {e}\n"
                f"Will skip data augmentation and proceed with only the base training data."
            )
            augment_datasets = None

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
        run_mrc(
            data_args, training_args, model_args,
            datasets, tokenizer, model,
            augment_datasets=augment_datasets,
            augment_ratio=augment_ratio
        )

def run_mrc(
    data_args: DataTrainingArguments,
    training_args: TrainingArguments,
    model_args: ModelArguments,
    datasets: DatasetDict,
    tokenizer,
    model,
    augment_datasets: Optional[DatasetDict] = None,
    augment_ratio: float = 0.0,
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
        
        # filter_overflow_chunks 속성이 있는지 확인 (하위 호환성)
        filter_overflow = getattr(data_args, 'filter_overflow_chunks', False)
        
        # overflow chunk 필터링을 위한 인덱스 추적
        valid_indices = []
        # 각 원본 샘플별로 첫 번째 chunk가 이미 처리되었는지 추적
        first_chunk_processed = set()

        for example_idx, offsets in enumerate(offset_maps):
            input_token_ids = tokenized_examples["input_ids"][example_idx]
            cls_idx = input_token_ids.index(tokenizer.cls_token_id)
            sequence_id_list = tokenized_examples.sequence_ids(example_idx)
            original_idx = overflow_map[example_idx]
            answer_info = examples[ans_col][original_idx]

            answer_exists = len(answer_info["answer_start"]) > 0
            if not answer_exists:
                # answer가 없는 경우 (negative samples)
                if not filter_overflow:
                    tokenized_examples["start_positions"].append(cls_idx)
                    tokenized_examples["end_positions"].append(cls_idx)
                    valid_indices.append(example_idx)
                else:
                    # 필터링 모드: 각 원본 샘플의 첫 번째 chunk만 유지
                    if original_idx not in first_chunk_processed:
                        tokenized_examples["start_positions"].append(cls_idx)
                        tokenized_examples["end_positions"].append(cls_idx)
                        valid_indices.append(example_idx)
                        first_chunk_processed.add(original_idx)
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
                    # answer가 span 밖에 있는 경우
                    if not filter_overflow:
                        # 필터링 안 함: cls_idx로 처리하되 chunk는 유지
                        tokenized_examples["start_positions"].append(cls_idx)
                        tokenized_examples["end_positions"].append(cls_idx)
                        valid_indices.append(example_idx)
                    else:
                        # 필터링 모드: answer가 없는 chunk는 제거
                        # 각 원본 샘플의 첫 번째 chunk만 유지
                        if original_idx not in first_chunk_processed:
                            tokenized_examples["start_positions"].append(cls_idx)
                            tokenized_examples["end_positions"].append(cls_idx)
                            valid_indices.append(example_idx)
                            first_chunk_processed.add(original_idx)
                else:
                    # answer가 span 안에 있는 경우: 항상 포함 (answer가 있는 chunk)
                    while ctx_start_pos < len(offsets) and offsets[ctx_start_pos][0] <= char_start:
                        ctx_start_pos += 1
                    tokenized_examples["start_positions"].append(ctx_start_pos - 1)
                    while offsets[ctx_end_pos][1] >= char_end:
                        ctx_end_pos -= 1
                    tokenized_examples["end_positions"].append(ctx_end_pos + 1)
                    valid_indices.append(example_idx)
                    # answer가 있는 chunk는 항상 유지하므로 first_chunk_processed에 추가하지 않음

        # 필터링 모드인 경우 valid_indices로 필터링
        if filter_overflow and len(valid_indices) < len(tokenized_examples["input_ids"]):
            for key in list(tokenized_examples.keys()):
                tokenized_examples[key] = [tokenized_examples[key][i] for i in valid_indices]
            logger.info(
                f"Filtered overflow chunks: {len(offset_maps)} -> {len(valid_indices)} "
                f"(removed {len(offset_maps) - len(valid_indices)} chunks without answers)"
            )

        return tokenized_examples

    processed_train_data = None
    if is_training:
        if "train" not in datasets:
            raise ValueError("--do_train requires a train dataset")
        training_data = datasets["train"]
        logger.info(f"Original training data size: {len(training_data)}")

        # augment_data와 비율 처리
        combined_train_data = training_data
        if augment_datasets is not None and hasattr(augment_datasets, "keys") and "train" in augment_datasets and augment_ratio > 0:
            augment_train_data = augment_datasets["train"]
            logger.info(f"Augment dataset size: {len(augment_train_data)}")
            num_aug = int(len(augment_train_data) * augment_ratio)
            if num_aug > 0:
                # 랜덤 샘플링
                indices = np.random.choice(len(augment_train_data), num_aug, replace=False)
                sampled_augment = augment_train_data.select(indices.tolist())
                combined_train_data = concatenate_datasets([training_data, sampled_augment])
                logger.info(
                    f"Using augment data: {num_aug} examples ({augment_ratio:.2f} of {len(augment_train_data)})"
                )
                logger.info(f"Combined dataset size (before shuffle): {len(combined_train_data)}")
            else:
                logger.info("Augment ratio set, but no samples selected from augment data.")
        elif augment_datasets is not None and hasattr(augment_datasets, "keys") and "train" in augment_datasets and augment_ratio <= 0:
            logger.info("Augment dataset is provided but ratio is 0 or negative. Only original train data will be used.")
        elif augment_datasets is not None and hasattr(augment_datasets, "keys") and "train" not in augment_datasets:
            logger.info("Augment dataset provided but no 'train' split found.")
        elif isinstance(augment_datasets, Dataset):
            # fallback if someone passed a single Dataset for augmentation
            logger.info("Augment dataset is a single Dataset object, not a DatasetDict. Treating as full augment set.")
            augment_train_data = augment_datasets
            num_aug = int(len(augment_train_data) * augment_ratio)
            if num_aug > 0:
                indices = np.random.choice(len(augment_train_data), num_aug, replace=False)
                sampled_augment = augment_train_data.select(indices.tolist())
                combined_train_data = concatenate_datasets([training_data, sampled_augment])
                logger.info(
                    f"Using augment data: {num_aug} examples ({augment_ratio:.2f} of {len(augment_train_data)})"
                )
            else:
                logger.info("Augment ratio set, but no samples selected from augment data.")
        elif augment_datasets is not None:
            logger.info("Augment dataset provided but could not identify format.")

        # 합친 데이터셋 셔플 (원본 데이터와 augment 데이터가 섞이도록)
        if augment_datasets is not None and augment_ratio > 0:
            combined_train_data = combined_train_data.shuffle(seed=training_args.seed)
            logger.info(
                f"Shuffled combined dataset: {len(combined_train_data)} total examples "
                f"(original: {len(training_data)}, augment: {len(combined_train_data) - len(training_data)})"
            )
        
        logger.info(f"Final combined dataset size (before tokenization): {len(combined_train_data)}")

        processed_train_data = combined_train_data.map(
            prepare_train_features,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=dataset_columns,
            load_from_cache_file=not data_args.overwrite_cache,
        )
        
        logger.info(f"Processed training data size (after tokenization): {len(processed_train_data)}")
        logger.info(
            f"Note: Due to return_overflowing_tokens=True, the processed data may have more samples "
            f"than the original ({len(combined_train_data)} -> {len(processed_train_data)})"
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
            version_2_with_negative=False,  # Negative passage는 평가에서 제외하므로 False
            null_score_diff_threshold=0.0, # [선택] 답 없음으로 판단할 기준점 (기본 0.0)
        )
        formatted_preds = []
        for prediction_id, prediction_text in processed_preds.items():
            formatted_preds.append({"id": prediction_id, "prediction_text": prediction_text})
        
        is_predicting = training_args.do_predict
        if is_predicting:
            return formatted_preds

        elif is_evaluating:
            ref_list = []         # 정답지 (References)
            final_preds = []      # 예측값 (Predictions)

            for val_example in datasets["validation"]:
                # 예측값 가져오기
                pred_text = processed_preds.get(val_example["id"], "")
                final_preds.append({
                    "id": val_example["id"], 
                    "prediction_text": pred_text
                })
                
                # 정답지 추가
                ref_list.append({
                    "id": val_example["id"], 
                    "answers": val_example[ans_col]
                })
                
            return EvalPrediction(
                predictions=final_preds, label_ids=ref_list
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
        #qa_trainer.save_model()

        train_metrics = training_results.metrics
        actual_train_samples = len(processed_train_data)
        train_metrics["train_samples"] = actual_train_samples
        logger.info(f"Actual training samples used: {actual_train_samples}")
        logger.info(f"Expected training samples: {len(combined_train_data)} (original) -> {actual_train_samples} (after tokenization with overflow)")

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

if __name__ == "__main__":
    main()
