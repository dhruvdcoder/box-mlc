"""Implements Negative Log Likelihood Loss for multi instance typing model"""
from typing import List, Tuple, Union, Dict, Any, Optional
from allennlp.common import Registrable
from allennlp.data import Vocabulary
from allennlp.models import Model
from os.path import commonprefix
from box_mlc.common.debug_utils import analyse_tensor
from box_embeddings.common.utils import log1mexp
from box_mlc.modules.binary_nll_loss import BinaryNLLLoss
import logging
import torch

logger = logging.getLogger(__name__)


class Sampler(Registrable):
    """Given target binary vector of shape (batch, total_labels) and predicted log probabilities of shape (batch, total_labels)
    performs sampling to produce a sampled target of shape (batch, sample_size) and log probabilities of shape (batch, sample_size)
    """

    default_implementation = "identity"

    def sample(
        self, scores: torch.Tensor, targets: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        return scores, targets, None


Sampler.register("identity")(Sampler)


@Sampler.register("true-positive-negative-pairs")
class TruePositiveNegativePairSampler(Sampler):
    def __init__(
        self,
        number_of_pairs: int = 1,
        adversarial: bool = False
    ):
        self.number_of_pairs = number_of_pairs
        self.adversarial = adversarial

    def sample(
        self, scores: torch.Tensor, targets: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        targets_float = targets.to(dtype=torch.float)

        neg_distribution = 1.0 - targets_float
        positive_distribution = targets_float

        if self.adversarial:
            neg_distribution = torch.softmax(scores, dim=-1) * neg_distribution
            # positive_distribution = (
            # torch.softmax(-scores, dim=-1) * targets_float
            # )
        positives_sample_indices = torch.multinomial(
            positive_distribution, self.number_of_pairs
        )  # (batch, 1)
        negative_sample_indices = torch.multinomial(
            neg_distribution, self.number_of_pairs
        )  # (batch, number_of_negatives)
        positives_sample = torch.gather(
            scores, -1, positives_sample_indices
        )  # (batch, 1)
        negative_sample = torch.gather(
            scores, -1, negative_sample_indices
        )  # (batch, number_of_negative)
        sample_scores = torch.cat(
            (positives_sample, negative_sample), dim=-1
        )  # (batch, 1+num_negatives)
        sample_targets = torch.cat(
            (
                torch.gather(targets, -1, positives_sample_indices),
                torch.gather(targets, -1, negative_sample_indices),
            ),
            dim=-1,
        )
        indices = torch.stack((positives_sample_indices.flatten(), negative_sample_indices.flatten()))
        #     .to(
        #     device=sample_scores.device, dtype=int
        # )

        return sample_scores, sample_targets, indices


class BinaryNLLHierarchyLoss(torch.nn.Module, Registrable):
    """Given log P of the positive class, computes the NLLLoss using log1mexp for log 1-P."""

    default_implementation = "binary-nll-hierarchy-loss"

    def __init__(
        self,
        vocab: Vocabulary,
        debug_level: int = 0,
        reduction: "str" = "mean",
        sampler: Sampler = None,
        distance_weight: float = 1.0,
        exponential_scaling: bool = False
    ) -> None:
        """
        Args:
            debug_level: scale of 0 to 3.
                0 meaning no-debug (fastest) and 3 highest debugging possible (slowest).
            reduction: Same as `torch.NLLLoss`.
            **kwargs: Unused
        Returns: (None)
        """
        super().__init__()
        self.debug_level = debug_level
        self.sampler = sampler or TruePositiveNegativePairSampler()
        self.inference_loss = BinaryNLLLoss(debug_level=debug_level, reduction=reduction)
        self.distance_weight = distance_weight
        self.exponential_scaling = exponential_scaling
        # self.distance_matrix = self._construct_distance_matrix(vocab)
        self.construct_distance_matrix(vocab)

    def forward(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        # """
        # Args:
        #     input: (predicted_log_probabilities) log probabilities for each label predicted by the model.
        #                                  Size is (batch, total_labels)
        #     target: (true_labels) True labels for the data (batch, total_labels)
        # Returns:
        #     The negative log likelihood loss between the predicted and true labels.
        # """

        if not self.training:
            return self.inference_loss.forward(input, target)

        input, target, indices = self.sampler.sample(input, target)
        log1mexp_log_probabilities = log1mexp(input)

        if self.debug_level > 0:
            analyse_tensor(
                log1mexp_log_probabilities, self.debug_level, "log1mexp"
            )
        predicted_prob = torch.stack(
            [log1mexp_log_probabilities, input], -2
        )  # (batch, 2, total_labels)
        loss_values = torch.nn.functional.nll_loss(
            predicted_prob,
            target.to(dtype=torch.long),
            weight=None,
            reduction='none',
        )  # (batch, total_pairs)

        loss_sum_vector = torch.sum(loss_values, dim=1)

        # Index distance matrix with positive/negative sample indices
        distance_vector = torch.diag(
            self.distance_matrix
                .index_select(0, indices[0])
                .index_select(1, indices[1])
        ) # (batch, total_pairs)

        if self.exponential_scaling:
            distanced_weighted_loss_values = loss_sum_vector * torch.exp(distance_vector * self.distance_weight)
        else:
            distanced_weighted_loss_values = loss_sum_vector * (distance_vector*self.distance_weight)

        loss = torch.mean(distanced_weighted_loss_values)

        return loss

    def reduce_loss(self, loss_tensor: torch.Tensor) -> None:
        """
        Args:
            loss_tensor: Computed loss values (batch, total_labels)
        Returns:
            Reduced value by summing up the values across labels dimension
            and averaging across the batch dimension (torch.Tensor)
        """
        # return torch.mean(torch.sum(torch.topk(loss_tensor,500,dim=-1,sorted=False)[0], -1))

        return torch.mean(torch.sum(loss_tensor, -1))

    def construct_distance_matrix(self, vocab: Vocabulary) -> torch.Tensor:
        vocab_size = vocab.get_vocab_size('labels')
        distance_matrix = torch.zeros([vocab_size]*2).to(device='cuda')
        vocab_tokens = list(vocab.get_index_to_token_vocabulary('labels').items())

        for idx1, label1 in vocab_tokens[:-1]:
            for idx2, label2 in vocab_tokens[idx1+1:]:

                common_prefix = commonprefix([label1, label2])

                if common_prefix:
                    if common_prefix == label1 and common_prefix == label2:
                        distance = 0
                    else:
                        distance_from_prefix = lambda l: len(l[len(common_prefix):].split('.')) if len(l) > len(common_prefix) else 0
                        distance = sum(map(distance_from_prefix, [label1, label2]))
                else:
                    distance = -torch.inf

                distance_matrix[idx1, idx2] = distance
                distance_matrix[idx2, idx1] = distance

        # Replace negative infinities with maximum distance based value
        max_distance = int(distance_matrix.max() * 1.2)
        torch.nan_to_num(distance_matrix, neginf=max_distance, out=distance_matrix)

        self.distance_matrix = distance_matrix


BinaryNLLHierarchyLoss.register("binary-nll-hierarchy-loss")(BinaryNLLHierarchyLoss)
