"""QA를 위한 커스텀 Trainer 클래스."""

from typing import Any, Dict, List, Optional

from transformers import Trainer, is_datasets_available
from transformers.utils import is_torch_xla_available
from transformers.trainer_utils import PredictionOutput

if is_datasets_available():
    import datasets

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met


class QuestionAnsweringTrainer(Trainer):
    """Question Answering 작업을 위한 커스텀 Trainer.
    
    표준 Trainer를 확장하여 QA 작업에 특화된 평가 및 예측 기능을 제공합니다.
    """

    def __init__(
        self,
        *args,
        eval_examples: Optional[datasets.Dataset] = None,
        post_process_function: Optional[callable] = None,
        **kwargs
    ):
        """QuestionAnsweringTrainer 초기화.
        
        Args:
            eval_examples: 평가용 원본 예제 데이터셋
            post_process_function: 예측 후처리 함수
            *args, **kwargs: Trainer의 기본 인자들
        """
        super().__init__(*args, **kwargs)
        self.eval_examples = eval_examples
        self.post_process_function = post_process_function


    def evaluate(
        self,
        eval_dataset: Optional[datasets.Dataset] = None,
        eval_examples: Optional[datasets.Dataset] = None,
        ignore_keys: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """QA 작업을 위한 평가 수행.
        
        Args:
            eval_dataset: 평가용 데이터셋 (None이면 self.eval_dataset 사용)
            eval_examples: 평가용 원본 예제 (None이면 self.eval_examples 사용)
            ignore_keys: 무시할 키 리스트
        
        Returns:
            평가 메트릭 딕셔너리
        """
        if eval_dataset is None:
            eval_dataset = self.eval_dataset
        if eval_examples is None:
            eval_examples = self.eval_examples
        
        eval_dataloader = self.get_eval_dataloader(eval_dataset)

        compute_metrics = self.compute_metrics
        self.compute_metrics = None
        try:
            output = self.prediction_loop(
                eval_dataloader,
                description="Evaluation",
                prediction_loss_only=True if compute_metrics is None else None,
                ignore_keys=ignore_keys,
            )
        finally:
            self.compute_metrics = compute_metrics

        if isinstance(eval_dataset, datasets.Dataset):
            eval_dataset.set_format(
                type=eval_dataset.format["type"],
                columns=list(eval_dataset.features.keys()),
            )

        has_post_process = self.post_process_function is not None
        has_compute_metrics = self.compute_metrics is not None
        
        if has_post_process and has_compute_metrics:
            eval_preds = self.post_process_function(
                eval_examples, eval_dataset, output.predictions, self.args
            )
            metrics = self.compute_metrics(eval_preds)
            self.log(metrics)
        else:
            metrics = {}

        if is_torch_xla_available() and (self.args.tpu_metrics_debug or self.args.debug):
            xm.master_print(met.metrics_report())

        self.control = self.callback_handler.on_evaluate(
            self.args, self.state, self.control, metrics
        )
        return metrics

    def predict(
        self,
        test_dataset: datasets.Dataset,
        test_examples: datasets.Dataset,
        ignore_keys: Optional[List[str]] = None
    ) -> Any:
        """QA 작업을 위한 예측 수행.
        
        Args:
            test_dataset: 테스트용 데이터셋
            test_examples: 테스트용 원본 예제
            ignore_keys: 무시할 키 리스트
        
        Returns:
            후처리된 예측 결과 또는 원본 출력
        """
        test_dataloader = self.get_test_dataloader(test_dataset)

        compute_metrics = self.compute_metrics
        self.compute_metrics = None
        try:
            output = self.prediction_loop(
                test_dataloader,
                description="Evaluation",
                prediction_loss_only=True if compute_metrics is None else None,
                ignore_keys=ignore_keys,
            )
        finally:
            self.compute_metrics = compute_metrics

        has_post_process = self.post_process_function is not None
        has_compute_metrics = self.compute_metrics is not None
        
        if not has_post_process or not has_compute_metrics:
            return output

        if isinstance(test_dataset, datasets.Dataset):
            test_dataset.set_format(
                type=test_dataset.format["type"],
                columns=list(test_dataset.features.keys()),
            )

        predictions = self.post_process_function(
            test_examples, test_dataset, output.predictions, self.args
        )
        return predictions
