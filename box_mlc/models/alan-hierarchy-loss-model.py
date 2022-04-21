from typing import List, Union, Dict, Any, Optional
import logging
import torch
from allennlp.common import Lazy

from allennlp.models import Model
from allennlp.data import Vocabulary
from allennlp.modules import FeedForward
from allennlp.training.metrics import Average
from allennlp.nn import InitializerApplicator
from box_embeddings.parameterizations import BoxTensor

# imported from multilabel_learning repo
from box_mlc.metrics.f1 import F1
from box_mlc.metrics.macro_micro_f1 import MicroMacroF1
from box_mlc.metrics.mean_average_precision import MeanAvgPrecision
from box_mlc.metrics.micro_average_precision import MicroAvgPrecision
from box_mlc.modules.binary_nll_hierarchy_loss import BinaryNLLHierarchyLoss
from box_mlc.modules.box_embedding import BoxEmbeddingModule
from box_mlc.modules.vector2box import Vec2Box

from box_embeddings.modules.intersection import Intersection
from box_embeddings.modules.volume import Volume

logger = logging.getLogger(__name__)


@Model.register("alan-hierarchy-loss-model")
class AlanHierarchyLossModel(Model):

    def __init__(
        self,
        vocab: Vocabulary,
        label_embeddings: BoxEmbeddingModule,
        feedforward: FeedForward,
        vec2box: Vec2Box,
        intersect: Intersection,
        volume: Volume,
        loss_fn: Lazy[BinaryNLLHierarchyLoss],
        initializer: Optional[InitializerApplicator] = None,
        visualization_mode: bool = True,
    ) -> None:
        super().__init__(vocab=vocab)

        self.current_labels = None

        self._volume = volume
        self._intersect = intersect
        self._vec2box = vec2box
        self._label_embeddings = label_embeddings
        self._feedforward = feedforward
        self.loss_fn = loss_fn.construct(vocab=vocab)
        # loss_fn.construct_distance_matrix(vocab)
        self.map = MeanAvgPrecision()
        self.micro_map = MicroAvgPrecision()
        self.micro_macro_f1 = MicroMacroF1()
        self.f1 = F1()
        self.constraint_violation_metric = Average()
        self.visualization_mode = visualization_mode

        if initializer:
            initializer(self)



    def get_scores(
        self,
        x: torch.Tensor,
        labels: torch.Tensor,
        meta: List[Dict[str, Any]],
        results: Dict,
    ) -> torch.Tensor:
        self.current_labels = labels

        predicted_label_reps = self._feedforward(x).unsqueeze(-2)
        batch, _, hidden_dims = predicted_label_reps.shape

        predicted_label_boxes: BoxTensor = self._vec2box(predicted_label_reps)

        box_size = predicted_label_boxes.box_shape[-1]

        label_boxes: BoxTensor = self._label_embeddings.all_boxes

        # if self.visualization_mode:
        # in vis mode we assume that batch size is 1
        results["x_boxes_z"] = predicted_label_boxes.z.squeeze(
            -2
        )  # Shape: (batch=1, box_size)
        results["x_boxes_Z"] = predicted_label_boxes.Z.squeeze(-2)
        # we need to add one extra dim with size 1 in the begining to fool the
        # forward_on_instances() to think it is batch dim
        results["y_boxes_z"] = label_boxes.z.unsqueeze(
            0
        )  # Shape: (batch=1,labels, box_size)
        results["y_boxes_Z"] = label_boxes.Z.unsqueeze(0)

        total_labels, _ = label_boxes.box_shape
        assert label_boxes.box_shape[1] == box_size
        label_boxes = label_boxes.box_reshape(
            (1, *label_boxes.box_shape)
        )
        assert label_boxes.box_shape == (1, total_labels, box_size)

        intersection = self._intersect(
            predicted_label_boxes, label_boxes
        )  # shape (batch, total_labels, box_size)
        assert intersection.box_shape == (batch, total_labels, box_size)
        log_probabilities = self._volume(intersection) - self._volume(
            predicted_label_boxes
        )

        if (log_probabilities > 1e-4).any():
            logger.warning(f"{(log_probabilities> 0).sum()} log_probability values greater than 0")
        log_probabilities.clamp_(max=0.0)
        assert log_probabilities.shape == (batch, total_labels)

        return log_probabilities

    def forward(
        self,
        x: torch.Tensor,
        labels: torch.Tensor,
        meta: List[Dict[str, Any]],
    ) -> Dict[str, torch.Tensor]:

        results: Dict[str, Any] = {"meta": meta}
        log_probabilities = self.get_scores(x, labels, meta, results)
        # loss
        results["scores"] = log_probabilities
        results["positive_probs"] = torch.exp(log_probabilities)

        results["loss"] = self.loss_fn(log_probabilities, labels)

        self.compute_metrics(results, labels)

        return results

    def compute_metrics(self, results: Dict, labels: torch.Tensor) -> None:
        self.map(results["scores"], labels)
        self.f1(results["positive_probs"], labels)
        self.micro_map(results["scores"], labels)
        self.micro_macro_f1(results["positive_probs"], labels)

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {
            "MAP": self.map.get_metric(reset),
            "fixed_f1": self.f1.get_metric(reset),
            "micro_map": self.micro_map.get_metric(reset),
        }

    def make_output_human_readable(  # type: ignore
        self, output_dict: Dict[str, Union[torch.Tensor, Dict]]
    ) -> Dict[str, Union[torch.Tensor, List]]:
        threshold = 0.5
        batch_size = output_dict["positive_probs"].shape[0]  # type:ignore
        preds_list_idx: List[List[str]] = [
            (
                (output_dict["positive_probs"][sample_number] > threshold)
                .nonzero()
                .view(-1)
                .tolist()
            )
            for sample_number in range(batch_size)
        ]

        output_dict["predictions"] = [
            [  # type: ignore
                self.vocab.get_index_to_token_vocabulary(  # type: ignore
                    namespace="labels"
                ).get(pred_idx)
                for pred_idx in example_pred_list
            ]
            for example_pred_list in preds_list_idx
        ]

        return output_dict  # type: ignore

    # def make_output_human_readable(  # type: ignore
    #     self, output_dict: Dict[str, List[List]]
    # ) -> Dict[str, Union[torch.Tensor, List]]:
    #     threshold = 0.5
    #     batch_size = output_dict["positive_probs"].shape[0]  # type:ignore
    #     preds_list_idx: List[List[str]] = [
    #         (
    #             (output_dict["positive_probs"][sample_number] > threshold)
    #             .nonzero()
    #             .view(-1)
    #             .tolist()
    #         )
    #         for sample_number in range(batch_size)
    #     ]
    #
    #     output_dict["predictions"] = [
    #         [
    #             self.vocab.get_index_to_token_vocabulary(
    #                 namespace="labels"
    #             ).get(pred_idx)
    #             for pred_idx in example_pred_list
    #         ]
    #         for example_pred_list in preds_list_idx
    #     ]
    #
    #     return output_dict