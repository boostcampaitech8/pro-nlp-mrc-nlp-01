"""유틸리티 함수 모듈 - QA 작업을 위한 헬퍼 함수들."""

import collections
import json
import logging
import os
import csv
import random
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from ..config import DataTrainingArguments, ModelArguments
from datasets import DatasetDict
from tqdm.auto import tqdm
from transformers import PreTrainedTokenizerFast, TrainingArguments, is_torch_available
from transformers.trainer_utils import get_last_checkpoint

logger = logging.getLogger(__name__)


def set_seed(seed: int = 42) -> None:
    """시드를 고정하여 재현 가능한 결과를 보장합니다.
    
    Args:
        seed: 고정할 시드 값 (기본값: 42)
    """
    random.seed(seed)
    np.random.seed(seed)
    if is_torch_available():
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def postprocess_qa_predictions(
    examples: Dict[str, Any],
    features: List[Dict[str, Any]],
    predictions: Tuple[np.ndarray, np.ndarray],
    version_2_with_negative: bool = False,
    n_best_size: int = 20,
    max_answer_length: int = 30,
    null_score_diff_threshold: float = 0.0,
    output_dir: Optional[str] = None,
    prefix: Optional[str] = None,
    is_world_process_zero: bool = True,
    run_name: Optional[str] = None,
    ensemble_save_dir: Optional[str] = None,
) -> Dict[str, str]:
    """QA 예측 결과를 후처리하여 최종 답변을 생성합니다.
    
    Args:
        examples: 원본 예제 딕셔너리
        features: 토큰화된 피처 리스트
        predictions: (start_logits, end_logits) 튜플
        version_2_with_negative: SQuAD 2.0 형식 지원 여부
        n_best_size: 상위 n개 후보를 고려
        max_answer_length: 최대 답변 길이
        null_score_diff_threshold: null 답변 판단 임계값
        output_dir: 결과 저장 디렉토리
        prefix: 파일명 접두사
        is_world_process_zero: 메인 프로세스 여부
        run_name: 실행 이름 (앙상블 파일명에 사용)
        ensemble_save_dir: 앙상블 예측 저장 디렉토리 (None이면 output_dir 사용)
    
    Returns:
        예측 결과 딕셔너리 {example_id: answer_text}
    """
    assert (
        len(predictions) == 2
    ), "`predictions` should be a tuple with two elements (start_logits, end_logits)."
    all_start_logits, all_end_logits = predictions

    assert len(predictions[0]) == len(
        features
    ), f"Got {len(predictions[0])} predictions and {len(features)} features."

    example_id_to_index = {}
    for idx, example_id in enumerate(examples["id"]):
        example_id_to_index[example_id] = idx
    
    features_per_example = collections.defaultdict(list)
    for feat_idx, feature in enumerate(features):
        example_idx = example_id_to_index[feature["example_id"]]
        features_per_example[example_idx].append(feat_idx)

    all_predictions = collections.OrderedDict()
    all_nbest_json = collections.OrderedDict()
    all_ensemble_json = collections.OrderedDict()
    if version_2_with_negative:
        scores_diff_json = collections.OrderedDict()

    logger.setLevel(logging.INFO if is_world_process_zero else logging.WARN)
    logger.info(
        f"Post-processing {len(examples)} example predictions split into {len(features)} features."
    )

    for example_index, example in enumerate(tqdm(examples)):
        feature_indices = features_per_example[example_index]

        min_null_prediction = None
        prelim_predictions = []

        for feature_index in feature_indices:
            start_logits = all_start_logits[feature_index]
            end_logits = all_end_logits[feature_index]
            offset_mapping = features[feature_index]["offset_mapping"]
            token_is_max_context = features[feature_index].get(
                "token_is_max_context", None
            )

            feature_null_score = start_logits[0] + end_logits[0]
            if (
                min_null_prediction is None
                or min_null_prediction["score"] > feature_null_score
            ):
                min_null_prediction = {
                    "offsets": (0, 0),
                    "score": feature_null_score,
                    "start_logit": start_logits[0],
                    "end_logit": end_logits[0],
                }

            start_indexes = np.argsort(start_logits)[
                -1 : -n_best_size - 1 : -1
            ].tolist()

            end_indexes = np.argsort(end_logits)[-1 : -n_best_size - 1 : -1].tolist()

            for start_index in start_indexes:
                for end_index in end_indexes:
                    if (
                        start_index >= len(offset_mapping)
                        or end_index >= len(offset_mapping)
                        or offset_mapping[start_index] is None
                        or offset_mapping[end_index] is None
                    ):
                        continue
                    if (
                        end_index < start_index
                        or end_index - start_index + 1 > max_answer_length
                    ):
                        continue
                    if (
                        token_is_max_context is not None
                        and not token_is_max_context.get(str(start_index), False)
                    ):
                        continue
                    prelim_predictions.append(
                        {
                            "offsets": (
                                offset_mapping[start_index][0],
                                offset_mapping[end_index][1],
                            ),
                            "score": start_logits[start_index] + end_logits[end_index],
                            "start_logit": start_logits[start_index],
                            "end_logit": end_logits[end_index],
                        }
                    )

        if version_2_with_negative:
            prelim_predictions.append(min_null_prediction)
            null_score = min_null_prediction["score"]

        predictions = sorted(
            prelim_predictions, key=lambda x: x["score"], reverse=True
        )[:n_best_size]

        if version_2_with_negative and not any(
            p["offsets"] == (0, 0) for p in predictions
        ):
            predictions.append(min_null_prediction)

        context_text = example["context"]
        for pred in predictions:
            offset_tuple = pred.pop("offsets")
            start_pos, end_pos = offset_tuple
            pred["text"] = context_text[start_pos:end_pos]

        if len(predictions) == 0 or (
            len(predictions) == 1 and predictions[0]["text"] == ""
        ):

            predictions.insert(
                0, {"text": "empty", "start_logit": 0.0, "end_logit": 0.0, "score": 0.0}
            )

        score_values = [pred.pop("score") for pred in predictions]
        scores_array = np.array(score_values)
        max_score = np.max(scores_array)
        exp_scores = np.exp(scores_array - max_score)
        probs = exp_scores / exp_scores.sum()

        for idx, pred in enumerate(predictions):
            pred["probability"] = probs[idx]

        if not version_2_with_negative:
            all_predictions[example["id"]] = predictions[0]["text"]
        else:
            i = 0
            while predictions[i]["text"] == "":
                i += 1
            best_non_null_pred = predictions[i]

            score_diff = (
                null_score
                - best_non_null_pred["start_logit"]
                - best_non_null_pred["end_logit"]
            )
            scores_diff_json[example["id"]] = float(score_diff)
            if score_diff > null_score_diff_threshold:
                all_predictions[example["id"]] = ""
            else:
                all_predictions[example["id"]] = best_non_null_pred["text"]

        nbest_list = []
        for pred in predictions:
            formatted_pred = {}
            for key, value in pred.items():
                if isinstance(value, (np.float16, np.float32, np.float64)):
                    formatted_pred[key] = float(value)
                else:
                    formatted_pred[key] = value
            nbest_list.append(formatted_pred)
        all_nbest_json[example["id"]] = nbest_list

        # 앙상블용 json 생성: 상위 n_best_size개 내에서 동일한 텍스트의 확률을 합산
        ensemble_candidates = collections.defaultdict(float)
        for pred in nbest_list:
            if "text" in pred and "probability" in pred:
                ensemble_candidates[pred["text"]] += pred["probability"]
        
        # 딕셔너리를 리스트로 변환하고 확률 내림차순 정렬
        ensemble_list = [
            {"text": k, "probability": v} 
            for k, v in ensemble_candidates.items()
        ]
        ensemble_list.sort(key=lambda x: x["probability"], reverse=True)
        
        # 상위 10개만 선택하여 저장
        all_ensemble_json[example["id"]] = ensemble_list[:10]

    if output_dir is not None:
        assert os.path.isdir(output_dir), f"{output_dir} is not a directory."

        prediction_file = os.path.join(
            output_dir,
            "predictions.json" if prefix is None else f"predictions_{prefix}.json",
        )
        nbest_file = os.path.join(
            output_dir,
            "nbest_predictions.json"
            if prefix is None
            else f"nbest_predictions_{prefix}.json",
        )

        # 앙상블 저장 디렉토리 설정 (기본값: output_dir/predictions_for_ensemble)
        if ensemble_save_dir is None:
            ensemble_save_dir = os.path.join(output_dir, "predictions_for_ensemble")
        
        if not os.path.exists(ensemble_save_dir):
            os.makedirs(ensemble_save_dir, exist_ok=True)

        # 파일명 생성 (run_name이 없으면 타임스탬프 사용)
        if run_name:
            safe_run_name = run_name.replace("/", "_").replace("\\", "_")  # 경로 문자 제거
            file_name = f"prediction_for_ensemble_{safe_run_name}.json"
        else:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            file_name = f"prediction_for_ensemble_{timestamp}.json"

        if prefix is not None:
            file_name = file_name.replace(".json", f"_{prefix}.json")

        ensemble_file = os.path.join(ensemble_save_dir, file_name)

        prediction_csv_file = os.path.join(
            output_dir,
            "predictions_submit.csv" if prefix is None else f"predictions_submit_{prefix}.csv",
        )
        
        if version_2_with_negative:
            null_odds_file = os.path.join(
                output_dir,
                "null_odds.json" if prefix is None else f"null_odds_{prefix}.json",
            )

        logger.info(f"Saving predictions to {prediction_file}.")
        with open(prediction_file, "w", encoding="utf-8") as writer:
            writer.write(
                json.dumps(all_predictions, indent=4, ensure_ascii=False) + "\n"
            )
        
        # nbest 예측 저장
        with open(nbest_file, "w", encoding="utf-8") as writer:
            writer.write(
                json.dumps(all_nbest_json, indent=4, ensure_ascii=False) + "\n"
            )
        
        # 앙상블 예측 저장
        logger.info(f"Saving ensemble predictions to {ensemble_file}.")
        with open(ensemble_file, "w", encoding="utf-8") as writer:
            writer.write(
                json.dumps(all_ensemble_json, indent=4, ensure_ascii=False) + "\n"
            )
        
        # CSV 형식으로 제출용 파일 저장
        with open(prediction_csv_file, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f, delimiter="\t")
            for key, value in all_predictions.items():
                writer.writerow([key, value])
        
        # SQuAD 2.0 형식인 경우 null odds 저장
        if version_2_with_negative:
            with open(null_odds_file, "w", encoding="utf-8") as writer:
                writer.write(
                    json.dumps(scores_diff_json, indent=4, ensure_ascii=False) + "\n"
                )

    return all_predictions

def check_no_error(
    data_args: DataTrainingArguments,
    training_args: TrainingArguments,
    datasets: DatasetDict,
    tokenizer: PreTrainedTokenizerFast,
) -> Tuple[Optional[str], int]:
    """학습 전 설정 검증 및 체크포인트 확인.
    
    Args:
        data_args: 데이터 학습 인자
        training_args: 학습 인자
        datasets: 데이터셋 딕셔너리
        tokenizer: 토크나이저
    
    Returns:
        (마지막 체크포인트 경로, 최대 시퀀스 길이) 튜플
    
    Raises:
        ValueError: 검증 실패 시
    """

    last_checkpoint = None
    if (
        os.path.isdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    if not isinstance(tokenizer, PreTrainedTokenizerFast):
        raise ValueError(
            "This example script only works for models that have a fast tokenizer. Checkout the big table of models "
            "at https://huggingface.co/transformers/index.html#bigtable to find the model types that meet this "
            "requirement"
        )

    if data_args.max_seq_length > tokenizer.model_max_length:
        logger.warn(
            f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    if "validation" not in datasets:
        raise ValueError("--do_eval requires a validation dataset")
    return last_checkpoint, max_seq_length
