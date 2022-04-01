import logging
from pprint import pprint
from typing import List, Dict, Iterable
from numpy.typing import NDArray

from tqdm import tqdm
import numpy as np

from box_vis import BoxVisualizer

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", level=logging.DEBUG)
logger = logging.getLogger("box_vis ErrorAnalyzer")

class ErrorAnalyzer(BoxVisualizer):

    adjacency_matrix: NDArray = None
    ground_truth_matrix: NDArray = None

    def _construct_adjacency_matrix(self):
        adjacency_matrix = np.zeros([len(self.vocab)]*2)
        with open('/mnt/nfs/scratch1/dhruveshpate/multilabel_classification/multilabel-learning/.data/expr_FUN/hierarchy_tc.edgelist', mode='r') as f:
            for line in f:
                a_label, b_label, _ = line.split(' ')
                try:
                    a_idx, b_idx = self.vocab_idx[a_label], self.vocab_idx[b_label]
                except KeyError:
                    logger.debug(f"Skipping labels {a_label} {b_label}")
                    continue
                adjacency_matrix[b_idx][a_idx] = 1
        self.adjacency_matrix = adjacency_matrix

    def _construct_ground_truth_matrix(self):
        ground_truth_matrix = np.zeros((len(self.predictions), len(self.vocab)))

        for pred_idx, pred in enumerate(self.predictions):
            for true_label in pred['true_labels']:
                ground_truth_matrix[pred_idx][self.vocab_idx[true_label]] = 1

        self.ground_truth_matrix = ground_truth_matrix

    def compute_error(
            self,
            thresholds: Iterable[float]
    ) -> Dict[float, Dict[str, float]]:

        # TODO: possibly tensorize computations across multiple thresholds
        # TODO: compute F1 over all thresholds and determine best threshold
        np.seterr(invalid='ignore')
        if not self.adjacency_matrix:
            self._construct_adjacency_matrix()

        if not self.ground_truth_matrix:
            self._construct_ground_truth_matrix()

        score_matrix = np.array([pred['label_scores'] for pred in self.predictions])

        results = dict()
        for threshold in tqdm(thresholds, unit='thresholds', desc='Calculating errors'):
            result_entry = dict()
            fp_matrix = np.logical_and(
                    1-self.ground_truth_matrix,
                    np.where(score_matrix > threshold, 1, 0))

            result = fp_matrix*(self.ground_truth_matrix@self.adjacency_matrix > 0)
            result_entry['hfpr'] = result.sum()/fp_matrix.sum()
            # result_entry['hfpr_labels'] = result.sum(axis=0)/fp_matrix.sum(axis=0)

            results[threshold] = result_entry
            # TODO: vectorize FP matrix construction
            # for pred_idx, pred in enumerate(self.predictions):
            #     threshold_labels_idx = [idx for idx, label_score in enumerate(pred['label_scores']) if label_score >= threshold]
            #     for threshold_labels_idx
            # inter_path_errors = 0
            # intra_path_errors = 0
            #
            # true_positives = 0
            # false_positives = 0
            # total_labels_predicted = 0
            # for pred in self.predictions:
            #     total_labels_predicted += len(pred['true_labels'])
            #     threshold_labels = [self.vocab[idx] for idx, label_score in enumerate(pred['label_scores']) if
            #                         label_score >= threshold]
            #
            #     for threshold_label in threshold_labels:
            #         if threshold_label not in pred['true_labels']:
            #             false_positives += 1
            #             identified_intra_path_error = False
            #             # Detect intra error
            #             for true_label in pred['true_labels']:
            #                 if len(threshold_label) > len(true_label) and threshold_label.startswith(true_label):
            #                     intra_path_errors += 1
            #                     identified_intra_path_error = True
            #                     break
            #             # Count as inter error
            #             if not identified_intra_path_error:
            #                 inter_path_errors += 1
            #
            # results[threshold] = {
            #     'inter_path_errors': inter_path_errors,
            #     'intra_path_errors': intra_path_errors,
            #     'intra_inter_ratio': intra_path_errors / inter_path_errors,
            #     'HFPR': intra_path_errors / false_positives,
            #     'false_positives': false_positives
            # }

        # pprint(results)
        return results


if __name__ == "__main__":
    # %% Two class y_boxes
    analyzer = ErrorAnalyzer(
        '/mnt/nfs/scratch1/asempruch/1750',
        dims=range(0, 4)
    )

    error_results = analyzer.compute_error(np.arange(-15, 0, 0.1))

    pass