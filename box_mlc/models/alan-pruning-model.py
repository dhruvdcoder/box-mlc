from typing import List, Union, Dict, Any, Optional
import logging
import torch

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
from box_mlc.modules.NearestNeighborFilter import Filter
from box_mlc.modules.box_embedding import BoxEmbeddingModule
from box_mlc.modules.binary_nll_loss import BinaryNLLLoss
from box_mlc.modules.vector2box import Vec2Box
from box_mlc.modules.multi_instance_typing_encoder import (
    MultiInstanceTypingEncoder,
)

from box_embeddings.modules.intersection import Intersection
from box_embeddings.modules.volume import Volume

# from box_search.NearestNeighborFilter import NearestNeighborFilter

logger = logging.getLogger(__name__)


@Model.register("alan-pruning-model")
class AlanBaselineModel(Model):

    def __init__(
        self,
        vocab: Vocabulary,
        label_embeddings: BoxEmbeddingModule,
        feedforward: FeedForward,
        vec2box: Vec2Box,
        intersect: Intersection,
        volume: Volume,
        nn_filter: Filter,
        encoder_stack: Optional[MultiInstanceTypingEncoder] = None,
        initializer: Optional[InitializerApplicator] = None,
        visualization_mode: bool = True,
    ) -> None:
        super().__init__(vocab=vocab)

        self.current_labels = None

        self._volume = volume
        self._intersect = intersect
        self._nn_filter = nn_filter
        self._vec2box = vec2box
        self._encoder = encoder_stack
        self._label_embeddings = label_embeddings
        self._feedforward = feedforward
        self.loss_fn = BinaryNLLLoss()
        self.map = MeanAvgPrecision()
        self.map_nn = MeanAvgPrecision()
        self.micro_map = MicroAvgPrecision()
        self.micro_map_nn = MicroAvgPrecision()
        self.micro_macro_f1 = MicroMacroF1()
        self.micro_macro_f1_nn = MicroMacroF1()
        self.f1 = F1()
        self.f1_nn = F1()
        self.f1_random_nn = F1()
        self.constraint_violation_metric = Average()
        self.visualization_mode = visualization_mode

        self._nn_filter.set_intersec_and_volume(self._intersect, self._volume)

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

        if self._encoder:
            x = self._encoder(x)

        predicted_label_reps = self._feedforward(x).unsqueeze(-2)
        batch, _, hidden_dims = predicted_label_reps.shape

        predicted_label_boxes: BoxTensor = self._vec2box(predicted_label_reps)

        box_size = predicted_label_boxes.box_shape[-1]

        label_boxes: BoxTensor = self._label_embeddings.all_boxes

        self._nn_filter.find_nearest_neighbors(predicted_label_boxes, label_boxes)

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

        results["loss"] = self.loss_fn(log_probabilities, labels) # + L1_on_theta_params

        self.compute_metrics(results, labels)

        return results

    def compute_metrics(self, results: Dict, labels: torch.Tensor) -> None:

        # multiply by negative infinity
        # nn_mask = torch.ones(results["scores"].shape[-1], dtype=float) * -float("inf")
        # nn_mask
        # nn_filtered_scores = results["scores"] * self._nn_filter.nn_indices

        nn_scores = torch.ones(results["scores"].shape[-1], dtype=torch.float32).cuda() * -float("inf")
        nn_scores[self._nn_filter.nn_indices] = results["scores"][0, self._nn_filter.nn_indices]
        nn_scores = nn_scores.unsqueeze(0)

        nn_positive_probs = torch.zeros(results["positive_probs"].shape[-1], dtype=torch.float32).cuda()
        nn_positive_probs[self._nn_filter.nn_indices] = results["positive_probs"][0, self._nn_filter.nn_indices]
        nn_positive_probs = nn_positive_probs.unsqueeze(0)

        random_nn_positive_probs = results["positive_probs"] * (
                torch.cuda.FloatTensor(results["positive_probs"].shape[-1]).uniform_()
                > (1 - self._nn_filter.nn_indices.shape[-1]/results["positive_probs"].shape[-1])
        )
        # random_nn_positive_probs = torch.zeros(results["positive_probs"].shape[-1], dtype=torch.float32).cuda() * -float("inf")
        # random_indices = torch.randint(results["positive_probs"].shape[-1], (self._nn_filter.nn_indices.shape[-1],))
        # random_nn_positive_probs[random_indices] = results["positive_probs"][0, random_indices]
        # random_nn_positive_probs = random_nn_positive_probs.unsqueeze(0)

        self.map(results["scores"], labels)
        # self.map_nn(nn_scores, labels)
        self.f1(results["positive_probs"], labels)
        self.f1_nn(nn_positive_probs, labels)
        self.f1_random_nn(random_nn_positive_probs, labels)
        self.micro_map(results["scores"], labels)
        # self.micro_map_nn(nn_scores, labels)
        self.micro_macro_f1(results["positive_probs"], labels)
        self.micro_macro_f1_nn(nn_positive_probs, labels)

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {
            "MAP": self.map.get_metric(reset),
            # "MAP_nn": self.map_nn.get_metric(reset),
            "fixed_f1": self.f1.get_metric(reset),
            "fixed_f1_nn": self.f1_nn.get_metric(reset),
            "fixed_f1_random_nn": self.f1_random_nn.get_metric(reset),
            "micro_map": self.micro_map.get_metric(reset),
            # "micro_map_nn": self.micro_map_nn.get_metric(reset),
            "nn_pruning_steps": self._nn_filter.pruning_steps,
            "nn_pruned_boxes": self._nn_filter.num_pruned,
            "nn_pruned_boxes_ratio": self._nn_filter.num_pruned,
            "nn_mean_pruning_steps": self._nn_filter.mean_pruning_steps,
            "nn_mean_pruned_boxes": self._nn_filter.mean_pruned_boxes,
            "nn_std_pruning_steps": self._nn_filter.std_pruning_steps,
            "nn_std_pruned_boxes": self._nn_filter.std_pruned_boxes,
            "nn_num_pruned_ratio": self._nn_filter.num_pruned_ratio,
            "nn_mean_num_pruned_ratio": self._nn_filter.mean_num_pruned_ratio,
            "nn_std_num_pruned_ratio": self._nn_filter.std_num_pruned_ratio
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