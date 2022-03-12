"""Scorer for x->y i.e score(y|x) implemented as a module"""
from allennlp.common import Registrable
from allennlp.data.vocabulary import Vocabulary
from torch.nn.parameter import Parameter
from typing import List, Tuple, Union, Dict, Any, Optional
import logging
import torch
from torch import linalg as LA
from math import sqrt

logger = logging.getLogger(__name__)


class ImplicationScorer(torch.nn.Module, Registrable):
    """Base class to satisfy Registrable"""

    default_implementation = "dot"

    pass


@ImplicationScorer.register("dot")
class DotImplicationScorer(ImplicationScorer):
    def __init__(
        self,
        **kwargs: Any,
    ) -> None:
        """

        Args:
            **kwargs: TODO

        Returns: (None)

        """
        super().__init__(**kwargs)  # type:ignore

    def forward(self, y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Given the two tensors as input, returns the dot product as the score.

        Args:
            y: (out_features|num_labels, in_features|hidden_dim)
            x: Shape (batch, *, in_features|hidden_dim)

        Returns:
            scores: yx^T (batch, *, num_labels|out_features) where each scores_yx indicates P(y|x)

        """
        scores = torch.matmul(x, y.T)

        return scores


@ImplicationScorer.register("normalized-dot")
class NormalizedDotImplicationScorer(DotImplicationScorer):
    def __init__(
        self,
        vocab: Vocabulary,
        normalize_length: bool = True,
        scaled_threshold: bool = True,
        **kwargs: Any,
    ) -> None:
        """

        Args:
            **kwargs: TODO

        Returns: (None)

        """
        super().__init__(**kwargs)  # type:ignore
        num_labels = vocab.get_vocab_size(namespace="labels")

        if scaled_threshold:
            self.thresholds = torch.nn.Parameter(
                (torch.rand(num_labels) * 10.0)
            )
        else:
            self.thresholds = torch.nn.Parameter((torch.rand(num_labels)))

        self.normalize_length = normalize_length
        self.scaled_threshold = scaled_threshold

    def forward(self, y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Given the two tensors as input, returns the dot product as the score.

        Args:
            y: (out_features|num_labels, in_features|hidden_dim)
            x: Shape (batch, *, in_features|hidden_dim)

        Returns:
            scores: yx^T (batch, *, num_labels|out_features) where each scores_yx indicates P(y|x)

        """

        if self.normalize_length:
            x_n = x / (
                torch.norm(x, p=2, dim=-1, keepdim=True) + 1e-20
            )  # (batch, *, hidden)
            y_n = y / (torch.norm(y, p=2, dim=-1, keepdim=True) + 1e-20)
        else:
            x_n = x
            y_n = y
        dot = torch.matmul(x_n, y_n.T)  # (batch, *, num_labels)
        leading_dims = x_n.dim() - 1

        if self.scaled_threshold:
            scores = (10.0 * dot) * (
                torch.sigmoid(self.thresholds)[(None,) * leading_dims]
            )
        else:
            scores = self.thresholds[(None,) * leading_dims] * dot

        return scores


@ImplicationScorer.register("bilinear")
class BilinearImplicationScorer(ImplicationScorer):
    def __init__(
        self,
        size: int,
    ) -> None:
        """

        Args:
            size: size of the scorer_weight tensor. Should be hidden_dim.

        Returns: (None)

        """
        super().__init__()  # type:ignore
        self.size = size
        self.scorer_weight = Parameter(
            torch.Tensor(size, size)
        )  # type: torch.Tensor
        torch.nn.init.uniform_(self.scorer_weight, - sqrt(size), sqrt(size))

    def forward(self, y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Given the two tensors as input, returns the dot product as the score.

        Args:
            y: (out_features, size)
            x: Shape (batch, *, size)

        Returns:
            scores: xT A y (batch, *, out_features|num_labels), where A = scorer_weight and
                    where each scores_ij represent unnormalized score for P(i|j).

        """
        # weight : (size, size)
        # x: (batch, * , size)
        # y : (num_labels, size)
        scores = torch.matmul(
            torch.matmul(x, self.scorer_weight), y.T
        )  # xT A y

        return scores


@ImplicationScorer.register("hyperbolic")
class HyperbolicImplicationScorer(ImplicationScorer):
    def __init__(
        self,
        **kwargs: Any,
    ) -> None:
        """

        Args:
            **kwargs: TODO

        Returns: (None)

        """
        super().__init__(**kwargs)  # type:ignore
        self.make_asymmetric = False

    def distance(self, y, x):
        """Given the two tensors as input, returns the dot product as the score.

        Args:
            y: (out_features|num_labels, in_features|hidden_dim)
            x: Shape (batch, *, in_features|hidden_dim)

        Returns:
            scores: Eq (1) https://arxiv.org/pdf/2101.04997.pdf
                    (batch, *, num_labels|out_features)

        """
        # y.unsqueeze(0) x.unsqueeze(1) #(1, nl, h) (b, 1, h)
        d = (
            LA.norm(y.unsqueeze(0) - x.unsqueeze(1), dim=-1) ** 2
        )  # batch, num_labels
        denominator_x = 1 - LA.norm(x, dim=-1) ** 2  # batch, *
        denominator_y = 1 - LA.norm(y, dim=-1) ** 2  # num_labels
        denominator = (
            denominator_x.unsqueeze(1) * denominator_y.unsqueeze(0) + 1e-13
        )  # batch, * , num_labels
        d = torch.div(d, denominator)  # batch, *, num_labels
        d = 1 + (2 * d)
        d = torch.where(d > 1.0 + 1e-6, d, torch.ones_like(d) + 1e-6)

        return torch.arccosh(d)

    def forward(self, y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Given the two tensors as input, returns the dot product as the score.

        Args:
            y: (out_features|num_labels, in_features|hidden_dim)
            x: Shape (batch, *, in_features|hidden_dim)

        Returns:
            scores: Eq (1) https://arxiv.org/pdf/2101.04997.pdf
                    (batch, *, num_labels|out_features) where each scores_yx indicates P(y|x)
                    where y and x first mapped tp the hyperbolic space, pi(x)=x/(1+(1+||x||^2)^0.5)

        """

        y_denominator = 1 + torch.sqrt(
            1 + LA.norm(y, dim=-1) ** 2
        )  # num_labels
        pi_y = torch.div(
            y, y_denominator.unsqueeze(-1)
        )  # num_labels, hidden_dim
        x_denominator = 1 + torch.sqrt(1 + LA.norm(x, dim=-1) ** 2)  # batch, *
        pi_x = torch.div(
            x, x_denominator.unsqueeze(-1)
        )  # batch, *, hidden_dim
        scores = -self.distance(pi_y, pi_x)  # batch, *,  num_labels

        if self.make_asymmetric:
            general = pi_y
            specific = pi_x
            scores = (
                1
                + 1e-3
                * (
                    torch.linalg.norm(
                        general, ord=2, dim=-1, keepdim=True
                    )  # (num_labels, 1)
                    - torch.linalg.norm(
                        specific, ord=2, dim=-1, keepdim=True
                    ).transpose(0, 1)
                )  # (1, num_labels)
            ) * scores

        return scores
