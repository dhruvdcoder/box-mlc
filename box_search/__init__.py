import json
import logging
import pickle
from typing import List, Dict, Tuple, Iterable, Optional
import os
import time

import ast
import numpy as np
import wandb
from allennlp.models.archival import load_archive
from allennlp.predictors.predictor import Predictor
from allennlp.common.util import import_module_and_submodules
import torch
from torch import Tensor
from tqdm import tqdm
import matplotlib.pyplot as plt
from box_embeddings.initializations.initializer import BoxTensor
from box_embeddings.modules.intersection.gumbel_intersection import GumbelIntersection
from box_embeddings.modules.volume.soft_volume import SoftVolume
from sklearn.metrics import f1_score, average_precision_score, ndcg_score
import_module_and_submodules("box_mlc")
import argparse

logger = logging.getLogger("box_search")

DEFAULT_MODEL_PATH = 'model.tar.gz'
DEFAULT_CONFIG_PATH = 'config.json'
DEFAULT_VOCAB_FILE = 'vocabulary'


class BoxSearcherBase:

    name: str = None
    predictions: List[dict] = None
    vocab: List[str]
    vocab_idx: Dict[str, int]
    y_boxes: Dict[str, BoxTensor] = None    # Maps label to y_box
    y_vecs: Dict[str, Tensor] = None
    total_dims: int                                      # Dimensionality of boxes
    variance_by_dim: torch.sort = None
    pruning_ratio = 0.8

    def __init__(
            self,
            rundir: str,
            name: str = None,
            model_path: Optional[str] = None,
            config_path: Optional[str] = None,
            vocab_file: Optional[str] = None,
            get_predictions: Optional[bool] = True
    ) -> None:
        self.rundir: str = rundir
        self.name = name or rundir.split('/')[-1]

        model_path = f"{rundir}/{DEFAULT_MODEL_PATH}" if model_path is None else model_path
        config_path = f"{rundir}/{DEFAULT_CONFIG_PATH}" if config_path is None else config_path
        vocab_file = f"{rundir}/{DEFAULT_VOCAB_FILE}" if vocab_file is None else vocab_file

        # Load config
        with open(config_path) as f:
            self.config = json.load(f)

        # Load labels
        with open(vocab_file+"/labels.txt") as f:
            self.vocab = [label.strip() for label in f]
            self.vocab_idx = {label: idx for idx, label in enumerate(self.vocab)}
            # self.vocab = {label.strip(): idx for idx, label in enumerate(f)}

        # Load archive
        archive = load_archive(model_path)

        # Get predictor
        self.predictor = Predictor.from_archive(
            archive,
            # args.predictor,
            dataset_reader_to_load="validation",
            frozen=True
        )

        # Get prediction data
        self.prediction_data = list(self.predictor._dataset_reader.read(self.config["validation_data_path"]))

        if get_predictions:
            self.get_predictions()

    def get_predictions(self) -> None:
        self.predictor._model.visualization_mode = True

        data = self.prediction_data
        is_box_model = "y_boxes_z" in self.predictor.predict_instance(data[0])

        logger.info("Starting to predict")
        predictions = []
        y_boxes: List[BoxTensor] = []
        y_vecs: List[Tensor] = []
        for x in tqdm(data, desc="predicting"):
            r = self.predictor.predict_instance(x)

            prediction = {
                "label_scores": r["scores"],
                "predicted_labels": r["predictions"],
                "true_labels": r["meta"]["labels"]
            }

            if 'x_boxes_z' in r and "x_boxes_Z" in r:
                x_box_z = Tensor(r["x_boxes_z"]).squeeze()
                x_box_Z = Tensor(r["x_boxes_Z"]).squeeze()
                prediction["x_box"] = BoxTensor((x_box_z, x_box_Z))

            if 'x_vecs' in r:
                prediction['x_vec'] = Tensor(r['x_vecs'])

            predictions.append(prediction)

            if not y_boxes and "y_boxes_z" in r and "y_boxes_Z" in r:
                for y_box_z, y_box_Z in zip(r["y_boxes_z"], r["y_boxes_Z"]):
                    y_boxes.append(
                        BoxTensor(
                            (Tensor(y_box_z).squeeze(), Tensor(y_box_Z).squeeze())
                        )
                    )

            if not y_vecs and "y_vecs" in r:
                for y_vec in r["y_vecs"]:
                    y_vecs.append(Tensor(y_vec))

        with open(self.rundir + "/box_vis_predictions.pkl", 'wb') as f:
            pickle.dump(predictions, f)
        if y_boxes:
            with open(self.rundir + "/box_vis_y_boxes.pkl", 'wb') as f:
                pickle.dump(y_boxes, f)
        logger.info("Wrote prediction data to file")

        assert len(predictions) > 0
        self.predictions = predictions
        if y_boxes:
            self.y_boxes = {self.vocab[idx]: y_box for idx, y_box in enumerate(y_boxes)}
        if y_vecs:
            self.y_vecs = {self.vocab[idx]: y_vec for idx, y_vec in enumerate(y_vecs)}
        if "x_box" in predictions[0]:
            self.total_dims = predictions[0]["x_box"].box_shape[0]

    def get_variance_by_dim(self, print_graphs=False) -> torch.sort:

        if self.variance_by_dim:
            return self.variance_by_dim

        all_boxes = BoxTensor(
            torch.stack([
                torch.stack((box.z, box.Z)) for box in self.y_boxes.values()
            ])
        )

        intersec = self.predictor._model._intersect
        volume = self.predictor._model._volume
        # volume = SoftVolume()

        intersection_all_boxes: BoxTensor = intersec(
            BoxTensor(all_boxes.data.unsqueeze(0)),
            BoxTensor(all_boxes.data.unsqueeze(1))
        )

        # Unsqueezing extra dimension to retain box dimensions in subsequent volume computation
        intersection_all_boxes_unsqueezed = BoxTensor.from_zZ(
            intersection_all_boxes.z.unsqueeze(-1),
            intersection_all_boxes.Z.unsqueeze(-1)
        )

        intersection_volume_by_dim = volume(
            intersection_all_boxes_unsqueezed
        )

        var = torch.var(
            intersection_volume_by_dim,
            (0, 1)
        )

        var_sort = torch.sort(var, descending=(not args.rev_var))
        self.variance_by_dim = var_sort

        """Analysis"""
        # Intersection per box per dimension
        if print_graphs:
            plt.imshow(
                torch.sort(
                    torch.var(intersection_volume_by_dim, [0]),
                    -1,
                    descending=True
                ).values,
                cmap='autumn',
                interpolation='none'
            )
            plt.show()

            # Overall variance per dimension
            plt.bar(
                range(len(var_sort.values)),
                var_sort.values
            )
            plt.show()

        return self.variance_by_dim

    def search(
            self,
            target: BoxTensor,
            k: int,
            all_boxes,
            include_pruning_history=False
    ):

        if args.dist and hasattr(self.predictor._model, '_volume') and hasattr(self.predictor._model._volume, '_dimension_dist'):
            dim_dist_vector = self.predictor._model._volume._dimension_dist.detach()
            var_argsort = torch.sort(dim_dist_vector, descending=(not args.rev_var))
        else:
            var_argsort = self.get_variance_by_dim()  # alg: determine variance of intersection of label boxes and obtain argsort

        """ Containment Mask """

        num_pruned = 0  # alg: initialize counter to keep track of number of pruned boxes
        sort_rank = 0  # alg: initialize counter to keep track of current rank position of current dimension in sorted dimensions used for pruning
        target_pruned = round(self.pruning_ratio*all_boxes.box_shape[0])  # alg: compute target number of boxes to prune based on specified pruning ratio

        containment_mask = torch.ones(all_boxes.box_shape[0], dtype=torch.bool)  # alg: initialize containment mask (represents which boxes are contained within target box in current dimension)

        containment_mask_history = list()

        while num_pruned < target_pruned and sort_rank < var_argsort.indices.shape[0]:  # alg: while we haven't reached our pruning target and we don't use more dimensions to prune than the model's dimensionality

            filter_dims = var_argsort.indices[sort_rank]  # alg: select dimension to filter on
            sort_rank += 1  # alg: increment filter dim rank counter

            # Save
            # target_slices = torch.stack((
            #     target.z[filter_dims],
            #     # -1 * target.Z[filter_dims[1]]
            #     target.Z[filter_dims]
            # )) # .unsqueeze(-1)

            # torch.searchsorted(sorted, target_slices)

            # TODO: contained mask should reflect containment of box
            ## in target box slices across ALL dimensions

            # strict_containment
            _containment_mask = torch.logical_and(  # completely contained
                all_boxes.z[:, filter_dims] <= target.z[filter_dims],
                all_boxes.Z[:, filter_dims] >= target.Z[filter_dims]
            )

            if not args.strict_containment:
                # alg: determine containment mask of target box within label boxes only in current filter dimension
                _containment_mask = torch.logical_or(
                    _containment_mask,
                    torch.logical_or(
                        torch.logical_and(  # intersects on left side
                            all_boxes.z[:, filter_dims] >= target.z[filter_dims],
                            all_boxes.z[:, filter_dims] <= target.Z[filter_dims]
                        ),
                        torch.logical_and(  # intersects on right side
                            all_boxes.Z[:, filter_dims] >= target.z[filter_dims],
                            all_boxes.Z[:, filter_dims] <= target.Z[filter_dims]
                        )
                    )
                )

            _containment_mask = torch.logical_and(  # alg: logical and current containment mask with mask from previous iterations
                _containment_mask, containment_mask
            )
            _num_pruned = torch.logical_not(containment_mask).sum()  # alg: calculate number of pruned boxes

            # Next pruning step would remove too many dimensions
            if _num_pruned > int(1.5 * target_pruned):  # alg: do not include results of this step if the number of pruned boxes goes over the target pruning number by 50%
                break

            containment_mask_history.append(_containment_mask)
            num_pruned = _num_pruned
            containment_mask = _containment_mask  # alg: store current containment mask

        """ Score Calculation """

        # Select non-pruned boxes
        contained_index = containment_mask \
            .nonzero() \
            .flatten()  # alg: get indices of non-pruned boxes
        contained_boxes = BoxTensor(
            all_boxes.data[contained_index]
        )  # alg: get non-pruned boxes

        volume = self.predictor._model._volume
        # volume = SoftVolume()
        intersec = self.predictor._model._intersect

        scores: Tensor = volume(
            intersec(
                contained_boxes,
                BoxTensor((target.z.clone(), target.Z.clone()))
            )  # alg: compute intersection scores of target box with non-pruned boxes
        )

        top_k_scores = torch.topk(scores, min(k or scores.shape[-1], scores.shape[-1]))  # alg: get top-k scores

        remapped_indices = contained_index[top_k_scores.indices]  # alg: get indices of top k boxes in sorted order according to scores

        result = top_k_scores.values, remapped_indices  # alg: return score values and coressponding box indices

        if include_pruning_history:
            return result, containment_mask_history

        return result

        """
        Take highest variance dim from all boxes
        Two sorted lists of z and Zs. Each has to be mapped back to its original box
        When target box comes, take it's highest dim z and Z, and do binary search over sorted
        z and Zs to find containing boxes. Work inwards first and if nothing there work outwards.
        
        Consider how you can scale this to multiple target boxes so that you can do this in parallel.
        """

def evaluate_search(
    searcher: BoxSearcherBase,
    k: int,
    # all_boxes: BoxTensor,
):
    local_score_history = {
        'label': list(),
        'scores': list(),
        'indices': list(),
        'containment_history': list(),
        'f1': list(),
        'positive_overlap': list(),
        'ndcg': list(),
        'ap': list()
    }

    # for pred_data in searcher.predictions:  # alg: iterate over all x boxes treating each as target box
    #     target = pred_data['x_box']
    targets = searcher.y_boxes.items() if args.target == 'y' else [(idx, pred_data['x_box']) for (idx, pred_data) in enumerate(searcher.predictions)]
    for label, target in targets:
        all_boxes = BoxTensor(
            torch.stack([
                torch.stack((box.z, box.Z)) for _label, box in searcher.y_boxes.items() if (args.target == 'x' or label != _label)
            ])
        )

        (scores, indices) = searcher.search(
            target, k,
            all_boxes,
            include_pruning_history=False
        )  # alg: get top-k scores and indices of boxes

        local_score_history['label'].append(label)
        local_score_history['scores'].append(scores)
        local_score_history['indices'].append(indices)
        # all_scores['containment_history'].append(containment_history)

        volume = searcher.predictor._model._volume
        # volume = SoftVolume()
        intersec = searcher.predictor._model._intersect

        ground_truth = torch.topk(
            volume(
                intersec(
                    all_boxes,
                    BoxTensor((target.z.clone(), target.Z.clone()))
                )
            ), min(k or all_boxes.box_shape[-1], all_boxes.box_shape[-1])
        )  # alg: compute ground truth top-k scores and indices

        pred_mask = torch.zeros(all_boxes.box_shape[0], dtype=bool)
        pred_mask[indices] = True  # alg: get predicted containment mask

        true_mask = torch.zeros(all_boxes.box_shape[0], dtype=bool)
        true_mask[ground_truth.indices] = True  # alg: get ground truth containment mask

        positive_overlap = torch.logical_and(
            true_mask, pred_mask
        ).sum() / k  # alg: compute true positive ratio

        ap = average_precision_score(true_mask, pred_mask)  # alg: compute average precision

        true_rank_scores = torch.zeros(all_boxes.box_shape[0])
        true_rank_scores[ground_truth.indices] = torch.flip(torch.arange(k, dtype=torch.float32), [0])  # alg: generate ground truth ranking scores

        pred_rank_scores = torch.zeros(all_boxes.box_shape[0])
        pred_rank_scores[indices] = torch.flip(torch.arange(min(k, indices.shape[0]), dtype=torch.float32), [0])  # alg: generate predicted ranking scores

        ndcg = ndcg_score(true_rank_scores.unsqueeze(0), pred_rank_scores.unsqueeze(0))  # alg: compute ndcg using ranking scores

        f1 = f1_score(true_mask, pred_mask)  # alg: compute f1

        local_score_history['f1'].append(f1)
        local_score_history['positive_overlap'].append(positive_overlap.item())
        local_score_history['ndcg'].append(ndcg)
        local_score_history['ap'].append(ap)

    return local_score_history

args = None

if __name__ == '__main__':

    wandb.init()

    # score_history = {}
    score_keys = [
        'f1',
        # 'positive_overlap',
        'ndcg',
        'ap'
    ]

    # score_history_file = f'history/{"rev_var" if args.rev_var else ""}_{"dist" if args.dist else ""}_{time.strftime("%y%m%d_%H%M%S")}.pkl'

    def evaluate(model_name):
        if not os.path.isdir(f'plots/{model_name}'):
            os.mkdir(f'plots/{model_name}')

        searcher = BoxSearcherBase(
            f'/work/asempruch_umass_edu/saved_models/{model_name}',
            name=model_name,
            get_predictions=False,
        )

        if args.dist and not (hasattr(searcher.predictor._model, '_volume') and hasattr(searcher.predictor._model._volume, '_dimension_dist')):
            logger.error(f"Model {model_name} not compatible for dimension distribution evaluation. Skipping")
            return

        searcher.get_predictions()

        k = min(int(np.ceil(int(model_name.split('_')[0]) * 0.2)), 50)

        results = dict()

        for pruning_ratio in tqdm(np.arange(0.80, step=0.05)):
            searcher.pruning_ratio = pruning_ratio

            results[pruning_ratio] = evaluate_search(
                searcher,
                k
            )

        score_file_name = f'scores/{searcher.name}_{args.target}_k{k}_{"sc" if args.strict_containment else "-"}_{"revvar" if args.rev_var else "-"}_{"dist" if args.dist else "-"}.pkl'

        with open(score_file_name, 'wb') as f:
            pickle.dump(results, f)

        avg_scores = {
            ratio : { score_key : np.mean(results[ratio][score_key]) for score_key in score_keys } for ratio in results.keys()
        }

        dimensionality = searcher.total_dims
        model_type = model_name.split('_')[-1]

        # if dimensionality not in score_history:
        #     score_history[dimensionality] = {}
        # if model_type not in score_history[dimensionality]:
        #     score_history[dimensionality][model_type] = {}

        # score_history_entry = score_history[dimensionality][model_type]

        for score_key in score_keys:
            # if score_key not in score_history_entry:
            #     score_history_entry[score_key] = []
            x = avg_scores.keys()
            y = [scores[score_key] for scores in avg_scores.values()]
            # score_history_entry[score_key] = (x, y)
            plt.scatter(x, y)
            plt.title(f'{searcher.name} - {score_key} k={k} {"rev_var" if args.rev_var else ""} {"dist" if args.dist else ""}')
            plt.ylim(0, 1)
            plt.xlabel('pruning threshold')
            plt.savefig(f'plots/{searcher.name}/{searcher.name}_{score_key}_k{k}_{"rev_var" if args.rev_var else ""}_{"dist" if args.dist else ""}.jpg')
            plt.show()
            plt.clf()

        # with open(score_history_file, mode='w') as f:
        #     pickle.dump(score_history, f)

        # avg_pruning_steps = {
        #     ratio: np.mean([len(containment_history) for containment_history in results[ratio]['containment_history']]) for ratio in results
        # }
        #
        # plt.scatter(list(avg_pruning_steps.keys()), avg_pruning_steps.values())
        # plt.title(f"{searcher.name} - Avg pruning steps for pruning threshold")
        # plt.savefig(f'plots/{searcher.name}/{searcher.name}_{"rev_var" if args.rev_var else ""}_{"dist" if args.dist else ""}_avg_pruning_steps.jpg')
        # plt.show()
        # plt.clf()

        if hasattr(searcher.predictor._model, '_volume')\
                and hasattr(searcher.predictor._model._volume, '_dimension_dist'):
            regularizer = searcher.predictor._model._volume.regularizer
            regularized_dist = regularizer(searcher.predictor._model._volume._dimension_dist).detach().numpy()

            if not os.path.isdir('dist_vectors'):
                os.mkdir('dist_vectors')

            dist_vector_file_name = f'dist_vectors/{searcher.name}_distvec.pkl'
            with open(dist_vector_file_name, 'wb') as f:
                pickle.dump(regularized_dist, f)

            plt.plot(range(regularized_dist.shape[0]), regularized_dist)
            plt.title(f"{searcher.name} - dimension vector")
            plt.xlabel('dim')
            plt.ylabel('weight')
            plt.savefig(f"plots/{searcher.name}/{searcher.name}_dimension_vector.jpg")
            plt.show()
            plt.clf()
        del searcher

    # for model in ['5_dist1', '10_dist1', '50_dist1', '500_dist1', '5_softmax', '10_softmax', '50_softmax', '100_softmax', '500_softmax', '5_sparsemax', '10_sparsemax', '50_sparsemax']:
    #     evaluate(model)
    dir = '/work/asempruch_umass_edu/saved_models'
    start_time = time.time()
    # evaluate('1000_q2b5')

    models = args.models or reversed(os.listdir(dir))

    for model in models:
        if not args.force and os.path.isdir(f'plots/{model}'):
            logger.info(f"Skipping model {model} as it already has a plot directory")
            continue
        if not args.models and args.newer:
            mtime = os.stat(os.path.join(dir, model)).st_mtime
            hours_since_mod = (start_time - mtime) / 3600
            if hours_since_mod > args.newer:
                logger.debug("skipping model:", model)
                continue
        logger.debug("evaluating model:", model)
        try:
            with torch.no_grad():
                evaluate(model)
        except KeyboardInterrupt:
            break
        except Exception as e:
            wandb.log({'exception': e})
            logger.error(f"failed to evaluate model: {model}")
            # logger.error(e)
            raise e

    # if not os.path.isdir('plots/all'):
    #     os.mkdir('plots/all')

    # for dimensionality in score_history:
    #     fig_dir = f'plots/all/{time.strftime("%y%m%d_%H%M%S")}'
    #     if not os.path.isdir(fig_dir):
    #         os.mkdir(fig_dir)
    #
    #     for score_key in score_keys:
    #         for model_type in score_history[dimensionality]:
    #             scores = score_history[dimensionality][model_type]
    #             plt.scatter(scores[score_key][0], scores[score_key][1], label=model_type)
    #         plt.title(f'{dimensionality} - {score_key} {"rev_var" if args.rev_var else ""} {"dist" if args.dist else ""}')
    #         plt.ylim(0, 1)
    #         plt.legend()
    #         plt.xlabel('pruning threshold')
    #         if not os.path.isdir(f'{fig_dir}/{dimensionality}'):
    #             os.mkdir(f'{fig_dir}/{dimensionality}')
    #         plt.savefig(f'{fig_dir}/{dimensionality}/{score_key}_{"rev_var" if args.rev_var else ""}_{"dist" if args.dist else ""}.jpg')
    #         plt.show()
    #         plt.clf()


    # avg_boxes_pruned = {
    #     ratio: np.mean([torch.logical_not(containment_history).sum() for containment_history in results[ratio]['containment_history']]) for ratio in results
    # }

    print()
