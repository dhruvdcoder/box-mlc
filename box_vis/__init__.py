import json
import logging
from typing import List, Tuple, Iterable, Set, Dict, Any, Generator, Optional

from allennlp.models.archival import load_archive
from allennlp.predictors.predictor import Predictor
from allennlp.common.util import import_module_and_submodules
from matplotlib.patches import Rectangle
from tqdm import tqdm
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import numpy as np
from os.path import exists
import pickle

import_module_and_submodules("box_mlc")

DEFAULT_MODEL_PATH = 'model.tar.gz'
DEFAULT_CONFIG_PATH = 'config.json'
DEFAULT_VOCAB_FILE = 'vocabulary'

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", level=logging.INFO)
logger = logging.getLogger("box_vis")


def box_to_rectangle(box: tuple, x_dim: int, y_dim: int) -> Rectangle:
    x_z, y_z = box[0][x_dim], box[0][y_dim]
    width = box[1][x_dim] - x_z
    height = box[1][y_dim] - y_z
    return Rectangle(
        (x_z, y_z),
        width,
        height,
        fill=False,
        alpha=0.1,
        linewidth=1
    )


def box_to_point(box: tuple, x_dim: int, y_dim: int) -> Tuple[float, float]:
    x_z, y_z = box[0][x_dim], box[0][y_dim]
    x_Z, y_Z = box[1][x_dim], box[1][y_dim]

    return np.mean((x_z, x_Z)), np.mean((y_z, y_Z))


# TODO: allow any iterable of dims as parameter
def iter_dim_pairs(
        start_dim: int,
        stop_dim: int,
        axes: Optional[List[Axes]] = None
):  # -> Generator[Tuple[int, int], None, None] | Generator[Tuple[int, int, Axes], None, None]:
    for dim in range(start_dim, stop_dim+1, 2):
        x_dim = dim
        y_dim = dim+1 if x_dim < stop_dim else dim-1
        if axes is not None:
            yield x_dim, y_dim, axes[x_dim] if len(axes) > 1 else axes[0]
        else:
            yield x_dim, y_dim


class BoxVisualizer:

    predictions: List[dict] = None
    vocab: List[str]
    y_boxes: Dict[str, Tuple[List[float], List[float]]]     # Maps label to y_box
    total_dims: int                                         # Dimensionality of boxes

    def __init__(
            self,
            rundir: str,
            model_path: Optional[str] = None,
            config_path: Optional[str] = None,
            vocab_file: Optional[str] = None,
            get_predictions: Optional[bool] = True
    ) -> None:
        self.rundir: str = rundir
        model_path = f"{rundir}/{DEFAULT_MODEL_PATH}" if model_path is None else model_path
        config_path = f"{rundir}/{DEFAULT_CONFIG_PATH}" if config_path is None else config_path
        vocab_file = f"{rundir}/{DEFAULT_VOCAB_FILE}" if vocab_file is None else vocab_file

        # Load config
        with open(config_path) as f:
            self.config = json.load(f)

        # Load labels
        with open(vocab_file+"/labels.txt") as f:
            self.vocab = [label.strip() for label in f]
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

        if exists(self.rundir + "/box_vis_predictions.pkl") and exists(self.rundir + "/box_vis_y_boxes.pkl"):
            with open(self.rundir + "/box_vis_predictions.pkl", 'rb') as f:
                predictions = pickle.load(f)
            with open(self.rundir + "/box_vis_y_boxes.pkl", 'rb') as f:
                y_boxes = pickle.load(f)
            logger.info("Done reading data...")
        else:
            logger.info("Starting to predict")
            predictions = []
            y_boxes: List[Tuple[List, List]] = []
            for x in tqdm(data, desc="predicting"):
                r = self.predictor.predict_instance(x)
                x_box_z = r["x_boxes_z"]
                x_box_Z = r["x_boxes_Z"]
                predictions.append(
                    {
                        "label_scores": r["scores"],
                        "predicted_labels": r["predictions"],
                        "true_labels": r["meta"]["labels"],
                        "x_box": (x_box_z, x_box_Z)
                    }
                )
                if not y_boxes:
                    for y_box_z, y_box_Z in zip(r["y_boxes_z"], r["y_boxes_Z"]):
                        y_boxes.append((y_box_z, y_box_Z))
            with open(self.rundir + "/box_vis_predictions.pkl", 'wb') as f:
                pickle.dump(predictions, f)
            with open(self.rundir + "/box_vis_y_boxes.pkl", 'wb') as f:
                pickle.dump(y_boxes, f)
            logger.info("Wrote prediction and y_box data to file")

        assert len(predictions) > 0
        self.predictions = predictions
        self.y_boxes = {self.vocab[idx]: y_box for idx, y_box in enumerate(y_boxes)}
        self.total_dims = len(predictions[0]["x_box"][0])

    def visualize(
            self,
            dimensions: Tuple[int, int] = None,
            x_lim: Optional[int] = None,
            label_recursive: Optional[str] = None,
    ) -> None:
        """

        Args:
            dimensions: Pair of start and end dimensions
            x_lim: maximum number of x boxes to plot
            label_recursive: label filter where all child labels are also considered
                eg: "01.01.03" => ["01", "01.01", "01.01.03"]

        """
        logger.info("Visualizing...")
        fig: Figure
        axes: List[Axes]
        dims = dimensions if dimensions else (0, self.total_dims)
        num_graphs = dims[1] - dims[0]
        fig, axes = plt.subplots(num_graphs, 1, figsize=(10, num_graphs*8))


        if num_graphs == 1:
            axes = [axes]
        # else:
        #     print("Alternate case")
        #     axes = axes[0]

        # ax = axes[0]

        max_x, max_y, min_x, min_y = None, None, None, None
        y_rects_generated = False
        y_rects = list()
        if label_recursive:
            labels = filter(lambda l: label_recursive.startswith(l), self.vocab)
            preds = self._filter_by_recursive_label(self.predictions, label_recursive, x_lim)
            fig.canvas.set_window_title(f"Recursive label {label_recursive}\ndims: {dims}")
        else:
            labels = self.vocab
            preds = self.predictions

        for pred in tqdm(preds, unit='x_boxes'):
            x_box = pred["x_box"]
            for x_dim, y_dim, ax in iter_dim_pairs(*dims, axes=axes):
            # for x_dim, y_dim in iter_dim_pairs(*dims):

                # if not y_rects_generated:
                #     y_rects += [box_to_rectangle(y_box, x_dim, y_dim) for y_box in self.y_boxes.values()]
                x, y = box_to_point(x_box, x_dim, y_dim)
                ax.scatter(x, y)

                if max_x is None or x > max_x:
                    max_x = x
                elif min_x is None or x < min_x:
                    min_x = x
                if max_y is None or y > max_y:
                    max_y = y
                elif min_y is None or y < min_y:
                    min_y = y

            y_rects_generated = True

        for label in tqdm(labels, unit='y_boxes'):
            for x_dim, y_dim, ax in iter_dim_pairs(*dims, axes=axes):
            # for x_dim, y_dim in iter_dim_pairs(*dims):
                y_box = self.y_boxes[label]
                ax.add_patch(box_to_rectangle(y_box, x_dim, y_dim))

                if y_box[0][x_dim] < min_x:
                    min_x = y_box[0][x_dim]
                elif y_box[1][x_dim] > max_x:
                    max_x = y_box[1][x_dim]
                if y_box[0][y_dim] < min_y:
                    min_y = y_box[0][y_dim]
                elif y_box[1][y_dim] > max_y:
                    max_y = y_box[1][y_dim]

        for x_dim, y_dim, ax in iter_dim_pairs(*dims, axes=axes):
        # for x_dim, y_dim in iter_dim_pairs(*dims):
            ax.title.set_text(f"({x_dim}, {y_dim})")

        for ax in axes:
            ax.set_xlim([min_x-0.1, max_x+0.1])
            ax.set_ylim([min_y-0.1, max_y+0.1])
        plt.show()

    @staticmethod
    def _filter_by_recursive_label(
            preds: List[Dict[str, Any]],
            label_recursive: str,
            limit: Optional[int] = None
    ) -> Generator[List[Dict[str, Any]], None, None]:
        yield_count = 0
        for pred in preds:
            for true_label in pred['true_labels']:
                if label_recursive.startswith(true_label):
                    yield pred

                    if limit:
                        yield_count += 1
                        if yield_count >= limit:
                            return


# %%
visualizer = BoxVisualizer('/home/asempruch/boxem/box-mlc/temp_', get_predictions=True)
# %%
import pickle

# with open('_temp_predictions.pkl', 'rb') as f:
#     _predictions = pickle.load(f)
# visualizer.predictions = _predictions
#
# visualizer.get_predictions()
#
# with open('_temp_y_boxes.pkl', 'rb') as f:
#     _y_boxes = pickle.load(f)
# visualizer.y_boxes = _y_boxes
#
# visualizer.total_dims = 1750

visualizer.visualize(
    dimensions=(0, 5),
    x_lim=30,
    label_recursive='01.01.03.05.02',
    # TODO: allow specifying another label to contrast with
    # TODO: make coloring dependent on label depth, look into color maps
)

# visualizer.visualize(
#     dimensions=(100, 101),
#     x_lim=30,
#     label_recursive='01.01.03.05.02',
# )
#
# visualizer.visualize(
#     dimensions=(200, 201),
#     x_lim=30,
#     label_recursive='01.01.03.05.02',
# )
# visualizer.visualize(label_recursive='01.01.03.05.02', x_lim=5)

# Find target samples where labels belong to what we want, multiple on each subplot
# Use supblot that contains each pair of dimensions, 2 per figure