"""A box model for multi instance typing"""
from typing import List, Tuple, Union, Dict, Any, Optional
import numpy as np
from allennlp.models import Model
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.modules.seq2seq_encoders import (
    Seq2SeqEncoder,
    PassThroughEncoder,
)
from box_embeddings.parameterizations.box_tensor import BoxTensor
from box_embeddings.modules.regularization import BoxRegularizer
from allennlp.modules.seq2vec_encoders import Seq2VecEncoder
from allennlp.modules.feedforward import FeedForward
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.data.fields.text_field import TextFieldTensors
from allennlp.training.metrics import Average
from box_embeddings.modules.intersection import Intersection
from box_embeddings.modules.volume import Volume
from box_embeddings.modules.pooling.pooling import BoxPooler
from box_mlc.metrics.hierarchy_constraint_violation import (
    ConstraintViolationMetric,
)
from box_mlc.metrics.f1 import F1
from box_mlc.metrics.macro_micro_f1 import MicroMacroF1
from box_mlc.metrics.micro_average_precision import (
    MicroAvgPrecision,
)
from box_embeddings.modules.intersection.gumbel_intersection import (
    gumbel_intersection,
)
from box_mlc.modules.hierarchy_constraint_violation import (
    ConstraintViolation,
)
from box_embeddings.modules.volume.bessel_volume import (
    log_bessel_volume_approx,
)

from box_mlc.modules.multi_instance_typing_encoder import (
    MultiInstanceTypingEncoder,
)
from box_mlc.modules.hierarchy_regularizer import (
    HierarchyRegularizer,
)
from box_mlc.modules.binary_nll_loss import (
    BinaryNLLLoss,
)
from box_mlc.modules.box_embedding import (
    BoxEmbedding,
    BoxEmbeddingModule,
)
from box_mlc.modules.vector2box import Vec2Box
from box_mlc.metrics.mean_average_precision import (
    MeanAvgPrecision,
)
from box_mlc.metrics.f1 import F1
from box_mlc.metrics.micro_average_precision import (
    MicroAvgPrecision,
)

from box_embeddings.common.utils import log1mexp
from box_mlc.common.debug_utils import analyse_tensor
import torch
import logging

logger = logging.getLogger(__name__)


@Model.register("box-distance-box-model")
class BoxDistanceBoxModel(Model):
    """Does multi-instance entity typing as multilabel classification.
    Uses the same encoding stack as :class:`MultiInstanceTyping` model.
    But converts the vectors into boxes and uses box embedding for labels
    as well.
    """

    @torch.no_grad()
    def get_min_normalized_scores(self, scores: torch.Tensor) -> torch.Tensor:
        adj_T = (self.adj.transpose(0, 1))[None, :, :]

        return ((adj_T * scores[:, None, :]).min(dim=-1)[0]).detach()

    @torch.no_grad()
    def get_max_normalized_scores(self, scores: torch.Tensor) -> torch.Tensor:
        adj = (self.adj)[None, :, :]

        return ((adj * scores[:, None, :]).max(dim=-1)[0]).detach()

    def get_device(self):
        for p in self._label_embeddings.parameters():
            return p.device

    def __init__(
        self,
        vocab: Vocabulary,
        encoder_stack: MultiInstanceTypingEncoder,
        feedforward: FeedForward,
        vec2box: Vec2Box,
        intersect: Intersection,
        volume: Volume,
        label_embeddings: BoxEmbeddingModule,
        alpha: float,
        gamma: float,
        distance_type: str = 'l1',
        num_distance_dims: int = 0,
        dropout: float = None,
        constraint_violation: Optional[ConstraintViolationMetric] = None,
        regularizer: Optional[RegularizerApplicator] = None,
        label_box_regularizer: Optional[BoxRegularizer] = None,
        label_regularizer: Optional[HierarchyRegularizer] = None,
        label_sample_percent: int = 100,
        batch_regularization: bool = False,
        warmup_epochs: int = 0,
        debug_level: int = 0,
        visualization_mode: bool = False,
        avg_vector: bool = True,
        initializer: Optional[InitializerApplicator] = None,
        add_new_metrics: bool = True,
        **kwargs: Any,
    ) -> None:
        # TODO: look at baseline papers, decide if title is neccessary
        ## in that case get rid of mentions and concat title maybe
        """
        Following is the model architecture:
        .. aafig::
            :aspect: 60
            :scale: 150
            :proportional:
            +-------------------------------------------------------+
            |                       +---------+                     |
            |              +------->+  Loss   <--------------+      |
            |              |        +---------+              |      |
            |     +--------+--------+      +--------------+  |      |
            |     |    Box Volume   +<-----+ Label Embed  |  |      |
            |     +--------+--------+      +-------^------+  |      |
            |              ^                                 |      |
            |     +--------+---------+                       |      |
            |     |    Box pooler    |                       |      |
            |     +--------+---------+                       |      |
            |              ^                                 |      |
            |      +-------+------+                          |      |
            |      |Vector to box                            |      |
            |      +-------+------+                          |      |
            |              ^                           +-----+      |
            |      +-------+------+                    |            |
            |      | feedforward  |                    |            |
            |      +-------^------+                    |            |
            |              |                           |            |
            |        +-----+----+                      |            |
            |        | Dropout  |                      |            |
            |        +-----+----+                      |            |
            |              |                           |            |
            |      +-------+--------+                  |            |
            |      |  Concatenate   |                  |            |
            |      +-+------------+-+                  |            |
            |        ^            ^                    |            |
            |   +----+------------+------+             |            |
            |   |                        |             |            |
            |   |     Encoder Stack      |             |            |
            |   |                        |             |            |
            |   +---+-------------+------+             |            |
            |       ^             ^                    +            |
            |       +             +             Labels: 0,1,0,1,... |
            |   Mention_1       Sent_1                              | mention1+mention2 = one datapoint
            |   Mention_2       Sent_2                              |
            |   ...             ...                                 |
            |   Mention_bagsize Sent_bagsize                        |
            |                                                       |
            |                          Single datapoint             |
            |                                                       |
            +-------------------------------------------------------+
        Args:
            vocab: Vocabulary for the model. It will have the following namespaces: labels, tokens, positions
            encoder_stack: Collection of TextFieldEmbedder, Seq2SeqEncoder, Seq2VecEncoder forming a complete encoding stack.
            initializer: Init regexes for all weights in the model.
                See corresponding `AllenNLP doc <https://docs.allennlp.org/master/api/nn/initializers/#initializerapplicator>`_
            feedforward: Used at the end on either sentence endcoding or
                mention+sentence encoding based on the concat_mention argument.
            intersect: Box Intersection
            volume: Box Volume
            pooler: Pools the boxes in the sentences dimension
            concat_mention: Form the input to the FeedForward by concatenating mention
                representation and sentence representation.
                This would most likely be `False` if your encoder_stack uses
                a pretrained transformer as the in that case the sentence sequence
                itself will contain two sub-sequences [CLS] sentence [SEP]  mention [SEP].
            dropout: Dropout probability to use on the encoded representation
                before it is passed to the feedforward.
            regularizer: See corresponding
                `AllenNLP doc <https://docs.allennlp.org/master/api/nn/regularizers/regularizer_applicator/>`_
            label_box_regularizer: Applies some regularization to the label embeddings
            label_regularizer: Regularization for the DAG relationships in the label space.
            label_sample_percent: Percent of labels to sample for label-label regularization.
                               Default 100 implies include all labels in the regularization loss.
            warmup_epochs: Number of epochs to perform warmup training on labels.
                This is only useful when using `label_regularizer` that requires warm-up like `HierarchyRegularizer`. (default:0)
            debug_level: scale of 0 to 3.
                0 meaning no-debug (fastest) and 3 highest debugging possible (slowest).
            visualization_mode: False(default) or True
                Returns the x boxes and y boxes for visualization purposes.
            **kwargs: Unused
        Returns: (None)
        """
        super().__init__(vocab=vocab, regularizer=regularizer)
        self._encoder = encoder_stack
        # TODO: remove sentence output order property
        assert self._encoder.sentences_output_order == 2
        self._alpha = alpha
        self._gamma = gamma
        self._dropout = (
            torch.nn.Dropout(dropout) if dropout is not None else None
        )
        self._feedforward = feedforward
        self._volume = volume
        self._intersect = intersect
        self._vec2box = vec2box
        self._label_embeddings = label_embeddings
        self._label_box_regularizer = label_box_regularizer
        self._distance_type = distance_type
        self._num_distance_dims = num_distance_dims
        self.debug_level = debug_level
        self.visualization_mode = visualization_mode
        self.loss_fn = BinaryNLLLoss()
        self.map = MeanAvgPrecision()
        self.micro_map = MicroAvgPrecision()
        self.f1 = F1()
        self.micro_macro_f1 = MicroMacroF1()

        if add_new_metrics:
            self.f1_min_n = F1()
            self.micro_macro_f1_min_n = MicroMacroF1()
            self.map_min_n = MeanAvgPrecision()
        self.add_new_metrics = add_new_metrics
        self._label_regularizer = label_regularizer
        # TODO: uncommented for now
        # self.epoch = -1  #: completed epochs
        self.batch_regularization = batch_regularization
        self.label_sample_percent = label_sample_percent
        self.warmup_epochs = warmup_epochs
        self.constraint_violation_metric = constraint_violation
        self.avg_vector = avg_vector

        if warmup_epochs and (label_regularizer is None):
            logger.warning(
                f"Non-zero warmup_epochs ({warmup_epochs})"
                " is not useful when label_regularizer is None"
            )

        if label_regularizer is None and (
            label_sample_percent <= 0 or label_sample_percent > 100
        ):
            logger.error(
                f"Invalid label_sample_percent value: {label_sample_percent}"
            )
            raise ValueError(
                f"Invalid label_sample_percent value: {label_sample_percent}"
            )
        self.register_buffer(
            "current_labels", torch.empty(0), persistent=False
        )
        # self.register_buffer(
        #     "adj",
        #     torch.tensor(self.constraint_violation_metric.adjacency_matrix),
        #     persistent=False,
        # )

        if initializer is not None:
            initializer(self)

    def get_regularization_penalty(self) -> Optional[torch.Tensor]:
        """Applies all the regularization for this model.
        Returns:
            `torch.Tensor` containing the total regularization regularization penalty.
        """
        penalty = None

        if self._label_box_regularizer is not None:
            penalty = self._label_box_regularizer(
                self._label_embeddings.all_boxes
            )

        if self._label_embeddings.delta_penalty is not None:
            p = self._label_embeddings.get_delta_penalty()
            penalty = p if penalty is None else penalty + p

        if self._label_regularizer is not None:
            active_nodes: Optional[torch.BoolTensor] = None

            if self.batch_regularization:
                if self.current_labels.numel() == 0:
                    return penalty if penalty is not None else 0.0
                active_nodes = (
                    (self.current_labels).to(dtype=torch.bool).any(dim=0)
                )  # shape (active_nodes,)
                active_boxes = self._label_embeddings(
                    active_nodes.nonzero().flatten()
                )
                target_shape = list(active_boxes.box_shape)
                target_shape.insert(-1, 1)
                reshaped_boxes = active_boxes.box_reshape(
                    tuple(target_shape)
                )  # (active_boxes, -1, box_size)
                label_boxes = active_boxes  # (active_boxes, box_size)
            elif self.label_sample_percent != 100:
                # (label_sample_size, )
                label_size = self.vocab.get_vocab_size(namespace="labels")
                label_sample_size = int(
                    (label_size * self.label_sample_percent) / 100
                )
                device = self.get_device()
                nodes = torch.randperm(label_size, device=device)[
                    :label_sample_size
                ]
                active_nodes = torch.zeros(label_size, device=device).to(
                    dtype=torch.bool
                )
                active_nodes[nodes] = True
                active_boxes = self._label_embeddings(
                    active_nodes.nonzero().flatten()
                )
                target_shape = list(active_boxes.box_shape)
                target_shape.insert(-1, 1)
                reshaped_boxes = active_boxes.box_reshape(
                    tuple(target_shape)
                )  # (label_sample_size, -1, box_size)
                label_boxes = active_boxes  # (label_sample_size, box_size)
            else:
                target_shape = list(self._label_embeddings.all_boxes.box_shape)
                target_shape.insert(-1, 1)
                reshaped_boxes = self._label_embeddings.all_boxes.box_reshape(
                    tuple(target_shape)
                )  # (labels, 1, box_size)
                label_boxes = self._label_embeddings.all_boxes

            label_boxes = label_boxes.box_reshape(
                (1, *label_boxes.box_shape)
            )  # (1, labels, box_size)
            intersection_volume = self._volume(
                self._intersect(reshaped_boxes, label_boxes)
            )  # (labels, labels)

            log_probabilities = intersection_volume - self._volume(
                label_boxes
            )  # shape (labels, labels)

            if (log_probabilities > 0).any():
                logger.warning(
                    f"Label Regularization: {(log_probabilities> 0).sum()} log_probability values greater than 0"
                )
            log_probabilities.clamp_(max=0.0)
            hierarchy_penalty = self._label_regularizer(
                log_probabilities, active_nodes
            )
            penalty = (
                hierarchy_penalty
                if penalty is None
                else penalty + hierarchy_penalty
            )

        return penalty

    def get_scores(  # type:ignore
        self,
        text: TextFieldTensors,
        labels: torch.Tensor,
        meta: List[Dict[str, Any]],
        results: Dict,
        **kwargs: Any,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            text: Nested dict of Tensors corresponding to the text/sentences.
                Each value tensor will be of shape (batch, sentences, seq_len), where sentences is bag_size.
            mentions: Same as sentences but only contains tensors for surface mention
            labels: Tensor containing one-hot labels. Has shape (batch, label_set_size).
            meta: Contains raw text and other meta data for each datapoint.
            **kwargs: Unused
        Returns:
            Dict: Tensor dict containing the following keys: loss
        """

        if labels is not None:
            self.current_labels = labels  # hold on the the labels in the buffer for regularization

        if self.debug_level >= 2:
            torch.autograd.set_detect_anomaly(True)
        encoded_vec = self._encoder(
            text
        )  # shapes (batch, sentences, hidden_size)

        if self._dropout:
            encoded_vec = self._dropout(encoded_vec)

        predicted_label_reps = self._feedforward(encoded_vec)

        label_boxes: BoxTensor = (
            self._label_embeddings.all_boxes
        )  # box_shape (total_labels, box_size)

        num_labels, dims = label_boxes.box_shape

        # if self.visualization_mode:
        #     # in vis mode we assume that batch size is 1
        #     results[
        #         "x_boxes_z"
        #     ] = predicted_label_boxes.z  # Shape: (batch=1, box_size)
        #     results["x_boxes_Z"] = predicted_label_boxes.Z
        #     # we need to add one extra dim with size 1 in the begining to fool the
        #     # forward_on_instances() to think it is batch dim
        #     results["y_boxes_z"] = label_boxes.z.unsqueeze(
        #         0
        #     )  # Shape: (batch=1,labels, box_size)
        #     results["y_boxes_Z"] = label_boxes.Z.unsqueeze(0)

        # Stack z and Z tensors and take mean across newly added dimension

        def aggregate_distances(dist_input: torch.Tensor):

            dist = dist_input if not self._num_distance_dims else torch.topk(dist_input, k=self._num_distance_dims, dim=-1).values

            if self._distance_type == 'l1':
                return torch.sum(dist, dim=-1)
            if self._distance_type == 'l2':
                return torch.norm(dist, 2, dim=-1)

            raise ValueError(f"Unsupported distance_type {self._distance_type}")

        predicted_centers = predicted_label_reps.unsqueeze(1)

        batch, _, _ = predicted_centers.shape

        label_boxes_stack = torch.stack((label_boxes.z, label_boxes.Z), dim=1).unsqueeze(0)
        dist = torch.min(
            torch.abs(
                label_boxes_stack - predicted_centers.unsqueeze(1)
            ), dim=2
        ).values

        contained_mask = torch.logical_and(
            (label_boxes.z.unsqueeze(0) < predicted_centers), (predicted_centers < label_boxes.Z.unsqueeze(0))
        )  # .min(dim=-1).values

        dist_outside = aggregate_distances(torch.logical_not(contained_mask) * dist) / dims

        label_centers = torch.mean(label_boxes_stack, axis=2)
        dist_inside_contained = (label_centers*contained_mask) - (predicted_centers*contained_mask)

        dist_inside_not_contained = (
                label_centers*torch.logical_not(contained_mask)
        ) - (
                label_boxes.z*torch.logical_not(contained_mask)
        )

        dist_inside = aggregate_distances(dist_inside_contained + dist_inside_not_contained) / dims

        dist_final = dist_outside + self._alpha*dist_inside
        positives = labels
        negatives = torch.logical_not(labels)

        num_positives = torch.count_nonzero(positives, dim=-1)
        num_negatives = num_labels - num_positives

        loss = torch.mean(
            - (positives*torch.log(
                torch.sigmoid(self._gamma - dist_final)
            )).sum(-1) / num_positives
            - (negatives*torch.log(
                torch.sigmoid(dist_final - self._gamma)
            )).sum(-1) / num_negatives
        )

        log_probabilities = torch.log(
            torch.sigmoid(-dist_final)
        )

        assert not torch.isnan(loss)
        assert log_probabilities.shape == (batch, num_labels)

        return log_probabilities, loss

    def compute_metrics(self, results: Dict, labels: torch.Tensor) -> None:
        # metrics
        # Some numeric score as input (likely will be log probability)
        self.map(results["scores"], labels)
        # Just give positive probabilities
        self.f1(results["positive_probs"], labels)
        self.micro_macro_f1(results["positive_probs"], labels)
        self.micro_map(results["scores"], labels)

        if self.constraint_violation_metric is not None:
            self.constraint_violation_metric(results["scores"], labels)

        # TODO: remove these metrics and replace with the ones from the feature box model
        if self.add_new_metrics:
            s = self.get_min_normalized_scores(results["scores"])
            p = self.get_min_normalized_scores(results["positive_probs"])
            self.map_min_n(s, labels)
            self.f1_min_n(p, labels)
            self.micro_macro_f1_min_n(p, labels)

    def forward(  # type:ignore
        self,
        text: TextFieldTensors,
        labels: torch.Tensor,
        meta: List[Dict[str, Any]],
        **kwargs: Any,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            sentences: Nested dict of Tensors corresponding to the text/sentences.
                Each value tensor will be of shape (batch, sentences, seq_len), where sentences is bag_size.
            labels: Tensor containing one-hot labels. Has shape (batch, label_set_size).
            meta: Contains raw text and other meta data for each datapoint.
            **kwargs: Unused
        Returns:
            Dict: Tensor dict containing the following keys: loss
        """

        results: Dict[str, Any] = {"meta": meta}
        log_probabilities, loss = self.get_scores(
            text, labels, meta, results, **kwargs
        )

        # loss and metrics
        results["scores"] = log_probabilities
        results["positive_probs"] = torch.exp(log_probabilities)

        if self.debug_level > 0:
            analyse_tensor(
                log_probabilities,
                debug_level=self.debug_level,
                name="log_probabilities",
            )

        if labels is not None:
            # TODO: can just use returned probabilities
            # results["loss"] = self.loss_fn(log_probabilities, labels)
            results["loss"] = loss

            if self.debug_level > 0:
                analyse_tensor(results["loss"], self.debug_level, "loss")
            # metrics
            self.compute_metrics(results, labels)
        else:
            raise ValueError('No labels specified')

        return results

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = {
            "MAP": self.map.get_metric(reset),
            "fixed_f1": self.f1.get_metric(reset),
            **self.micro_macro_f1.get_metric(reset),
            "micro_map": self.micro_map.get_metric(reset),
        }

        if self.constraint_violation_metric is not None:
            metrics[
                "constraint_violation"
            ] = self.constraint_violation_metric.get_metric(reset)

        if self.add_new_metrics:
            metrics = {
                **metrics,
                **{
                    "MAP_min_n": self.map_min_n.get_metric(reset),
                    "fixed_f1_min_n": self.f1_min_n.get_metric(reset),
                    **{
                        f1_str + "_min_n": f1_
                        for f1_str, f1_ in self.micro_macro_f1_min_n.get_metric(
                            reset
                        ).items()
                    }
                    # "micro_map": self.micro_map.get_metric(reset),
                },
            }

        return metrics

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
        preds = (
            output_dict["positive_probs"] >= threshold  # type: ignore
        )  # shape (batch, label_set_size)

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

    @torch.no_grad()
    def get_edge_scores(self) -> np.ndarray:
        label_boxes = self._label_embeddings.all_boxes
        num_labels, hidden_dims = label_boxes.box_shape
        log_conditional_probs = (
            self._volume(
                self._intersect(
                    label_boxes.box_reshape((1, num_labels, hidden_dims)),
                    label_boxes.box_reshape((num_labels, 1, hidden_dims)),
                )
            )
            - self._volume(label_boxes).unsqueeze(0)
        )

        return log_conditional_probs.cpu().numpy()

    @torch.no_grad()
    def get_marginal(self, label: str) -> float:
        model = self
        try:
            idx = model.vocab.get_token_index(label, namespace="labels")
        except KeyError:
            return None
        box = model._label_embeddings(
            torch.tensor(idx, dtype=torch.long).to(device=self.get_device())
        )
        # compute volume for this model
        vol = model._volume(box)

        return float(vol)
