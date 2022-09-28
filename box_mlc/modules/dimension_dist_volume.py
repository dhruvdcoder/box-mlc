import torch
from torch.nn.functional import softplus, softmax
from box_embeddings.modules.volume import Volume
from box_embeddings.parameterizations import BoxTensor
import numpy as np
from sparsemax.sparsemax import Sparsemax

eps = 1e-23


@Volume.register("dimension-dist")
class DimensionDistVolume(Volume):

    def __init__(
        self,
        dim: int,
        regularization: str,
        init: str = 'uniform',
        beta: float = 1.0,
        gumbel_beta: float = 1.0,
        scale: float = 1.0,
        log_scale: bool = True,
    ):

        self.beta = beta
        self.gumbel_beta = gumbel_beta
        self.scale = scale

        super().__init__(log_scale)

        if regularization == 'softmax':
            self.regularizer = softmax
        elif regularization == 'sparsemax':
            self._sparsemax = Sparsemax(-1)
            self.regularizer = self._sparsemax_regularizer
        else:
            raise ValueError(f"regularization parameter '{regularization}' invalid")

        if init == 'uniform':
            dimension_dist = torch.rand(dim)
        else:
            raise ValueError(f"init parameter '{init}' invalid")

        self._dimension_dist = torch.nn.Parameter(dimension_dist)

    def forward(self, box_tensor: BoxTensor) -> torch.Tensor:
        return torch.sum(
            torch.log(softplus(
                box_tensor.Z - box_tensor.z, beta=self.beta
            ) + eps) * self.gumbel_beta * self.regularizer(self._dimension_dist),
            dim=-1,
        ) + float(
            np.log(self.scale)
        )

    def _sparsemax_regularizer(self, dimension_dist: torch.Tensor):
        return self._sparsemax(dimension_dist.unsqueeze(0)).squeeze(0)
