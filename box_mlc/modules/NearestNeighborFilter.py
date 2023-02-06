import torch
from torch import Tensor
import numpy as np
from allennlp.common import Registrable
from box_embeddings.modules.intersection import Intersection
from box_embeddings.modules.volume import Volume
from box_embeddings.parameterizations import BoxTensor


class Filter(Registrable):
    pass


@Filter.register("nearest-neighbor-filter")
class NearestNeighborFilter:

    variance_by_dim = None
    top_k_scores = None
    nn_indices = None
    pruning_steps = None
    num_pruned = None
    num_boxes = None
    _pruning_steps_history = list()
    _pruned_boxes_history = list()
    
    def __init__(
            self,
            pruning_ratio: float,
            reverse_variance: bool = False,
            strict_containment: bool = False,
    ):
        self.pruning_ratio = pruning_ratio
        self.reverse_variance = reverse_variance
        self.strict_containment = strict_containment

    def set_intersec_and_volume(self, intersect: Intersection, volume: Volume):
        self.intersect = intersect
        self.volume = volume

    def find_nearest_neighbors(
            self,
            x: BoxTensor,
            label_boxes: BoxTensor
    ):
        if not self.intersect or not self.volume:
            raise RuntimeError("NearestNeighbor filter needs intersec and volume objects set first")
        self.num_boxes = label_boxes.box_shape[0]

        x_reshaped = BoxTensor((x.z[0, 0], x.Z[0, 0]))
        self._get_variance_by_dim(label_boxes)

        # k = label_boxes.box_shape[0] * (1-self.pruning_ratio)
        self.num_pruned = 0  # alg: initialize counter to keep track of number of pruned boxes
        sort_rank = 0  # alg: initialize counter to keep track of current rank position of current dimension in sorted dimensions used for pruning
        target_pruned = round(self.pruning_ratio * label_boxes.box_shape[0])  # alg: compute target number of boxes to prune based on specified pruning ratio

        containment_mask = torch.ones(label_boxes.box_shape[0],
                                      dtype=torch.bool).cuda()  # alg: initialize containment mask (represents which boxes are contained within target box in current dimension)

        # containment_mask_history = list()
        self.pruning_steps = 0
        self.num_pruned_boxes = 0

        while self.num_pruned < target_pruned and sort_rank < self.variance_by_dim.indices.shape[
            0]:  # alg: while we haven't reached our pruning target and we don't use more dimensions to prune than the model's dimensionality

            self.pruning_steps += 1

            filter_dims = self.variance_by_dim.indices[sort_rank]  # alg: select dimension to filter on
            sort_rank += 1  # alg: increment filter dim rank counter

            # strict_containment
            _containment_mask = torch.logical_and(  # completely contained
                label_boxes.z[:, filter_dims] <= x_reshaped.z[filter_dims],
                label_boxes.Z[:, filter_dims] >= x_reshaped.Z[filter_dims]
            )

            if not self.strict_containment:
                # alg: determine containment mask of target box within label boxes only in current filter dimension
                _containment_mask = torch.logical_or(
                    _containment_mask,
                    torch.logical_or(
                        torch.logical_and(  # intersects on left side
                            label_boxes.z[:, filter_dims] >= x_reshaped.z[filter_dims],
                            label_boxes.z[:, filter_dims] <= x_reshaped.Z[filter_dims]
                        ),
                        torch.logical_and(  # intersects on right side
                            label_boxes.Z[:, filter_dims] >= x_reshaped.z[filter_dims],
                            label_boxes.Z[:, filter_dims] <= x_reshaped.Z[filter_dims]
                        )
                    )
                )

            _containment_mask = torch.logical_and(
                # alg: logical and current containment mask with mask from previous iterations
                _containment_mask, containment_mask
            )
            _num_pruned = torch.logical_not(containment_mask).sum().item()  # alg: calculate number of pruned boxes

            # Next pruning step would remove too many dimensions
            if _num_pruned > int(1.5 * target_pruned):  # alg: do not include results of this step if the number of pruned boxes goes over the target pruning number by 50%
                continue

            # containment_mask_history.append(_containment_mask)
            self.num_pruned = _num_pruned
            containment_mask = _containment_mask  # alg: store current containment mask

        self._pruned_boxes_history.append(self.num_pruned)
        self._pruning_steps_history.append(self.pruning_steps)

        """ Score Calculation """

        # Select non-pruned boxes
        contained_index = containment_mask \
            .nonzero() \
            .flatten()  # alg: get indices of non-pruned boxes
        # contained_boxes = BoxTensor(
        #     (label_boxes.z[contained_index], label_boxes.Z[contained_index])
        # )  # alg: get non-pruned boxes

        # scores: Tensor = self.volume(
        #     self.intersect(
        #         contained_boxes,
        #         BoxTensor((x_reshaped.z, x_reshaped.Z))
        #     )  # alg: compute intersection scores of target box with non-pruned boxes
        # )

        # top_k_scores = torch.topk(scores, min(k or scores.shape[-1], scores.shape[-1]))  # alg: get top-k scores

        # remapped_indices = contained_index[
        #     top_k_scores.indices]  # alg: get indices of top k boxes in sorted order according to scores

        # self.top_k_scores = top_k_scores.values
        # self.nn_indices = remapped_indices
        self.nn_indices = contained_index

    @property
    def num_pruned_ratio(self):
        return self.num_pruned / self.num_boxes

    @property
    def mean_pruning_steps(self):
        return np.mean(self._pruning_steps_history)

    @property
    def mean_pruned_boxes(self):
        return np.mean(self._pruned_boxes_history)

    @property
    def mean_num_pruned_ratio(self):
        return self.mean_pruned_boxes / self.num_boxes

    @property
    def std_pruning_steps(self):
        return np.std(self._pruning_steps_history)

    @property
    def std_pruned_boxes(self):
        return np.std(self._pruned_boxes_history)

    @property
    def std_num_pruned_ratio(self):
        return self.std_pruned_boxes / self.num_boxes

    def _get_variance_by_dim(
            self,
            label_boxes: BoxTensor
    ) -> torch.sort:

        intersection_all_boxes: BoxTensor = self.intersect(
            BoxTensor((label_boxes.z.unsqueeze(0), label_boxes.Z.unsqueeze(0))),
            BoxTensor((label_boxes.z.unsqueeze(1), label_boxes.Z.unsqueeze(1)))
        )

        # Unsqueezing extra dimension to retain box dimensions in subsequent volume computation
        intersection_all_boxes_unsqueezed = BoxTensor.from_zZ(
            intersection_all_boxes.z.unsqueeze(-1),
            intersection_all_boxes.Z.unsqueeze(-1)
        )

        intersection_volume_by_dim = self.volume(
            intersection_all_boxes_unsqueezed
        )

        var = torch.var(
            intersection_volume_by_dim,
            (0, 1)
        )

        var_sort = torch.sort(var, descending=(not self.reverse_variance))
        self.variance_by_dim = var_sort