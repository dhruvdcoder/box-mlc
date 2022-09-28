import logging
from os.path import exists
from pprint import pprint
from typing import List, Dict, Iterable
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from enum import Enum
import pickle as pkl

from tqdm import tqdm
import cupy as cp
import numpy as np

from box_vis import BoxVisualizer

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", level=logging.DEBUG)
logger = logging.getLogger("box_vis ErrorAnalyzer")


class METRIC(Enum):
    f1 = 'F1'
    hfpr = 'HFPR'


ERROR_FILE_NAME = 'box_vis_errors_v1.pkl'

class ErrorAnalyzer(BoxVisualizer):

    adjacency_matrix: NDArray = None
    ground_truth_matrix: NDArray = None
    errors: Dict = None
    all_thresholds: NDArray = None

    def _construct_adjacency_matrix(self):
        adjacency_matrix = cp.zeros([len(self.vocab)] * 2, dtype=bool)
        with open('/gypsum/scratch1/dhruveshpate/multilabel_classification/multilabel-learning/.data/expr_FUN/hierarchy_tc.edgelist', mode='r') as f:
            for line in f:
                a_label, b_label, _ = line.split(' ')
                try:
                    a_idx, b_idx = self.vocab_idx[a_label], self.vocab_idx[b_label]
                except KeyError:
                    logger.debug(f"Skipping labels {a_label} {b_label}")
                    continue
                adjacency_matrix[b_idx][a_idx] = True
        self.adjacency_matrix = adjacency_matrix

    def _construct_ground_truth_matrix(self):
        ground_truth_matrix = cp.zeros((len(self.predictions), len(self.vocab)), dtype=bool)

        for pred_idx, pred in enumerate(self.predictions):
            for true_label in pred['true_labels']:
                ground_truth_matrix[pred_idx][self.vocab_idx[true_label]] = True

        self.ground_truth_matrix = ground_truth_matrix

    def compute_error(
            self
    ) -> Dict[str, float]:

        score_matrix = cp.array([pred['label_scores'] for pred in self.predictions])

        all_thresholds = cp.unique(score_matrix)
        self.all_thresholds = all_thresholds

        # TODO: uncomment again
        # if exists(self.rundir + "/" + ERROR_FILE_NAME):
        #     with open(self.rundir + "/" + ERROR_FILE_NAME, 'rb') as f:
        #         self.errors = pkl.load(f)
        #         logger.info("Read errors from file")
        #     return self.errors

        # np.seterr(invalid='ignore')
        if not self.adjacency_matrix:
            self._construct_adjacency_matrix()

        if not self.ground_truth_matrix:
            self._construct_ground_truth_matrix()

        results = dict()

        num_batches = 200

        for thresholds in tqdm(cp.array_split(all_thresholds, num_batches), unit='batch'):
            positives_tensor = cp.broadcast_to(score_matrix, [len(thresholds), *score_matrix.shape]) > thresholds[:, None, None]
            tp_tensor = cp.logical_and(self.ground_truth_matrix, positives_tensor)
            fp_tensor = cp.logical_and(cp.logical_not(self.ground_truth_matrix), positives_tensor)
            fn_tensor = cp.logical_and(self.ground_truth_matrix, cp.logical_not(positives_tensor))
            # hierarchy_fp_tensor = fp_tensor*(self.ground_truth_matrix@self.adjacency_matrix.T > 0)
            hierarchy_fp_tensor = fp_tensor*(self.ground_truth_matrix@self.adjacency_matrix > 0)

            def store_result(key: str, value: NDArray):
                if key not in results:
                    results[key] = cp.asnumpy(value)
                else:
                    results[key] = np.append(results[key], cp.asnumpy(value), axis=0)
                del value

            for axis in (1, 2, (1, 2)):
                tp_sum_matrix = tp_tensor.sum(axis)
                precision_matrix = tp_sum_matrix / (tp_sum_matrix + fp_tensor.sum(axis))
                recall_matrix = tp_sum_matrix / (tp_sum_matrix + fn_tensor.sum(axis))
                f1_matrix = 2 * (precision_matrix * recall_matrix) / (precision_matrix + recall_matrix)

                if cp.isnan(f1_matrix.min()):
                    pass

                if axis == 1:
                    store_result('f1_per_label', f1_matrix)
                elif axis == 2:
                    store_result('f1_per_sample', f1_matrix)
                elif axis == (1, 2):
                    store_result('f1_per_threshold', f1_matrix)

        # TODO: put total hfpr too (threshold with maximum f1)
            store_result('hfpr', hierarchy_fp_tensor.sum((1, 2)) / fp_tensor.sum((1, 2)))
            store_result('hfpr_per_label', hierarchy_fp_tensor.sum(1) / fp_tensor.sum(1))

        with open(self.rundir + "/" + ERROR_FILE_NAME, 'wb') as f:
            pkl.dump(results, f)
            logger.info("Wrote errors to file")

        self.errors = results
        return results

    def plot_metric_per_label_depth(self, metric: Enum):
        errors = self.errors or self.compute_error()

        if metric == METRIC.f1:
            metric_vals = errors['f1_per_label']
            metric_per_label = np.nanmax(metric_vals, axis=0)
        elif metric == METRIC.hfpr:
            metric_vals = errors['hfpr_per_label']
            metric_per_label = np.nanmin(metric_vals, axis=0)
        else:
            raise ValueError(f"Invalid metric specified {metric}")

        f1_per_label_depth = dict()
        for idx, metric_val in enumerate(metric_per_label):
            if np.isnan(metric_val):
                continue
            label_depth = self._get_label_depth(self.vocab[idx])
            if label_depth not in f1_per_label_depth:
                f1_per_label_depth[label_depth] = list()
            f1_per_label_depth[label_depth].append(float(metric_val))

        logger.info(f"Filtered out {np.count_nonzero(np.isnan(metric_per_label))} nan {metric.value} scores for {len(metric_per_label)} labels")

        fig, ax = plt.subplots(1, 1)
        ax.violinplot(
            f1_per_label_depth.values(),
            positions=list(f1_per_label_depth.keys()),
            # widths=[len(labels)/120 for labels in f1_per_label_depth.values()]
        )
        ax.set_xticks(list(f1_per_label_depth.keys()))
        ax.set_xticklabels([f"{key} ({len(labels)})" for key, labels in f1_per_label_depth.items()])
        ax.set_title(f"{self.name}\n{metric.value} per label depth")
        ax.set_xlabel("Label depth (number of samples)")
        plt.show()

    def plot_metric_per_threshold(self, metric: Enum):
        errors = self.errors or self.compute_error()

        if metric == METRIC.hfpr:
            metric_per_threshold: NDArray = errors['hfpr']
        elif metric == METRIC.f1:
            metric_per_threshold: NDArray = errors['f1_per_threshold']
        else:
            raise ValueError(f"Invalid metric specified {metric}")
        # %%
        plt.plot(self.all_thresholds.tolist(), metric_per_threshold.tolist())
        # plt.bar(self.all_thresholds.tolist(), metric_per_threshold.tolist(), width=0.06)
        plt.xticks([])
        # plt.xticks(self.all_thresholds.tolist(), [t if idx % 100 == 0 else None for idx, t in enumerate(self.all_thresholds)])
        # plt.ylim(bottom=np.nanmin(hfpr_per_threshold[np.nonzero(hfpr_per_threshold)])-0.001)
        plt.title(f"{self.name}\n{metric.value} per threshold")
        plt.show()
        # %%

    @staticmethod
    def _get_label_depth(label: str):
        return len(label.split('.'))


if __name__ == "__main__":
    # %% Two class y_boxes
    box_analyzer = ErrorAnalyzer(
        '/gypsum/scratch1/asempruch/1750',
        name='Box Model',
        dims=range(0, 4)
    )
    vec_analyzer = ErrorAnalyzer(
        '/gypsum/scratch1/asempruch/boxem/mvm_best',
        name='Vector Model',
        dims=range(0, 4)
    )
    hierarchy_analyzer = ErrorAnalyzer(
        '/gypsum/scratch1/asempruch/boxem/hierarchy_loss',
        name='Hierarchy Loss',
        dims=range(0, 4)
    )
    cone_analyzer = ErrorAnalyzer(
        '/gypsum/scratch1/asempruch/boxem/cone',
        name='Cones',
        dims=range(0, 4)
    )

    for analyzer in (box_analyzer, hierarchy_analyzer, vec_analyzer, cone_analyzer):
        analyzer.compute_error()
        errors = analyzer.errors
        print(analyzer.name)
        print('f1:', np.nanmax(errors['f1_per_threshold']))
        print('hfpr:', np.nanmin(errors['hfpr']))

    box_analyzer.errors['f1_per_threshold']


    for metric in (METRIC.f1, METRIC.hfpr):
        for analyzer in (box_analyzer, hierarchy_analyzer, vec_analyzer, cone_analyzer):
            analyzer.plot_metric_per_label_depth(metric)
            # analyzer.plot_metric_per_threshold(metric)

    # analyzer.plot_metric_per_label_depth('f1')
    # analyzer.plot_metric_per_label_depth('hfpr')
    # vec_analyzer.plot_metric_per_threshold(METRIC.f1)
    # vec_analyzer.plot_metric_per_threshold(METRIC.hfpr)

    pass