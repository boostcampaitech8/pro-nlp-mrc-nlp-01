from transformers import Trainer, is_datasets_available
from transformers.utils import is_torch_xla_available
from transformers.trainer_utils import PredictionOutput

if is_datasets_available():
    import datasets

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met

class QuestionAnsweringTrainer(Trainer):
    def __init__(self, *args, eval_examples=None, post_process_function=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.eval_examples = eval_examples
        self.post_process_function = post_process_function

    def evaluate(self, eval_dataset=None, eval_examples=None, ignore_keys=None):
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

        if self.args.tpu_metrics_debug or self.args.debug:
            xm.master_print(met.metrics_report())

        self.control = self.callback_handler.on_evaluate(
            self.args, self.state, self.control, metrics
        )
        return metrics

    def predict(self, test_dataset, test_examples, ignore_keys=None):
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
