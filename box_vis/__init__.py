import json
import logging
from typing import List, Tuple, Iterable, Set, Dict, Any, Generator, Optional, Callable, Union
from numpy.typing import NDArray

from box_embeddings.parameterizations.box_tensor import BoxTensor
from box_embeddings.modules.intersection import Intersection, GumbelIntersection
from box_embeddings.modules.volume import Volume, HardVolume, SoftVolume, BesselApproxVolume
from torch import Tensor
import torch
from allennlp.models.archival import load_archive
from allennlp.predictors.predictor import Predictor
from allennlp.common.util import import_module_and_submodules
from matplotlib.patches import Rectangle, Wedge
from tqdm import tqdm
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import numpy as np
import cupy as cp
from os.path import exists
import pickle

import_module_and_submodules("box_mlc")

DEFAULT_MODEL_PATH = 'model.tar.gz'
DEFAULT_CONFIG_PATH = 'config.json'
DEFAULT_VOCAB_FILE = 'vocabulary'

DEFAULT_COLOR_MAPS = ['Purples', 'Blues', 'Greens', 'Reds',
                      'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
                      'GnBu', 'PuBu', 'PuBuGn', 'BuGn']

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", level=logging.INFO)
logger = logging.getLogger("box_vis BoxVisualizer")


def box_to_rectangle(box: tuple, x_dim: int, y_dim: int) -> Rectangle:
    x_z, y_z = box[0][x_dim], box[0][y_dim]
    width = box[1][x_dim] - x_z
    height = box[1][y_dim] - y_z
    return Rectangle(
        (x_z, y_z),
        width,
        height,
        fill=False,
        linewidth=1.5
    )


def box_to_point(box: tuple, x_dim: int, y_dim: int) -> Tuple[float, float]:
    x_z, y_z = box[0][x_dim], box[0][y_dim]
    x_Z, y_Z = box[1][x_dim], box[1][y_dim]

    # Return center of box
    return np.mean((x_z, x_Z)), np.mean((y_z, y_Z))


def iter_dim_pairs(
        dims: List[int],
        axes: Optional[List[Axes]] = None
):  # -> Generator[Tuple[int, int], None, None] | Generator[Tuple[int, int, Axes], None, None]:
    axes_idx = 0
    for idx in range(0, len(dims), 2):
        x_dim = dims[idx]
        y_dim = dims[idx+1] if idx+1 < len(dims) else dims[idx-1]
        if axes is not None:
            yield x_dim, y_dim, axes[axes_idx]
        else:
            yield x_dim, y_dim
        axes_idx += 1


def get_intersection_volume_by_dim(
        x_boxes: List[Tuple[Tensor, Tensor]],
        y_boxes: List[Tuple[Tensor, Tensor]],
        intersection: Intersection = GumbelIntersection(),
        volume: Volume = SoftVolume()
) -> Tensor:
    total_dims = y_boxes[0][0].shape[0]
    # variance_by_dim = cp.zeros(total_dims)

    # TODO: vectorize computation
    ## torch.log(softplus(box_tensor.Z - box_tensor.z, beta=beta)
    x_box = BoxTensor.from_zZ(
        torch.stack([box[0] for box in x_boxes]),
        torch.stack([box[1] for box in x_boxes])
    )  # (I, D)

    y_box = BoxTensor.from_zZ(
        torch.stack([box[0] for box in y_boxes]),
        torch.stack([box[0] for box in y_boxes])
    )  # (L, D)

    x_box = x_box.box_reshape([x_box.box_shape[0], 1, x_box.box_shape[1]])
    y_box = y_box.box_reshape([1, *y_box.box_shape])

    # x_z_tensor = torch.stack([box[0] for box in x_boxes]).unsqueeze(1)
    # x_Z_tensor = torch.stack([box[1] for box in x_boxes]).unsqueeze(1)
    # y_z_tensor = torch.stack([box[0] for box in y_boxes]).unsqueeze(0)
    # y_Z_tensor = torch.stack([box[0] for box in y_boxes]).unsqueeze(0)

    intersection_by_dim: BoxTensor = intersection(x_box, y_box)
    intersection_1d_boxes = BoxTensor.from_zZ(intersection_by_dim.z.unsqueeze(-1), intersection_by_dim.Z.unsqueeze(-1))
    volume_by_dim: Tensor = volume(intersection_1d_boxes)

    return volume_by_dim

    # TODO: make heatmap out of these tensor, pass in all x and y boxes
    # variance_by_dim = torch.var(volume_by_dim, dim=0)
    # mean_by_dim = torch.mean(volume_by_dim, dim=0)
    #
    # # TODO: decide whether to take mean across label boxes for variance or to take the variance for the child label
    #
    # # x_z_tensor = cp.asarray([cp.asarray(box[0]) for box in x_boxes])
    # # x_Z_tensor = cp.asarray([cp.asarray(box[1]) for box in x_boxes])
    # #
    # # y_z_tensor = cp.asarray([cp.asarray(box[0]) for box in y_boxes])
    # # y_Z_tensor = cp.asarray([cp.asarray(box[1]) for box in y_boxes])
    #
    #
    # # for dim in tqdm(range(total_dims), desc='Calculating intersection volume variance by dim', unit='dims'):
    # #     variance_by_x_box = np.zeros(len(x_boxes))
    # #     for x_idx, (x_box_z, x_box_Z) in enumerate(x_boxes):
    # #         score_per_label = np.zeros(len(y_boxes))
    # #         for y_idx, (y_box_z, y_box_Z) in enumerate(y_boxes):
    # #             x_box = BoxTensor((x_box_z[dim:dim + 1], x_box_Z[dim:dim + 1]))
    # #             y_box = BoxTensor((y_box_z[dim:dim + 1], y_box_Z[dim:dim + 1]))
    # #             inter_box = intersection(x_box, y_box)
    # #             score_per_label[y_idx] = volume(inter_box)
    # #             # if score_per_label[y_idx].item() != -29.93360710144043:
    # #             #     print("Got different volume:", score_per_label[y_idx].item())
    # #
    # #             # z_distance = np.abs(x_box_z[dim] - y_box_z[dim])
    # #             # Z_distance = np.abs(x_box_Z[dim] - y_box_Z[dim])
    # #             # score_per_label[y_idx] = np.mean((z_distance.item(), Z_distance.item()))
    # #
    # #         variance_by_x_box[x_idx] = score_per_label.var()
    # #     variance_by_dim[dim] = variance_by_x_box.mean()
    #
    # return variance_by_dim


def get_cone_score_by_dim(
        x_vecs: List[Tensor],
        y_vecs: List[Tensor]
) -> Tensor:
    x, y = torch.stack(x_vecs), torch.stack(y_vecs)

    z = torch.max(y.unsqueeze(0), x.unsqueeze(1))
    diff = y.unsqueeze(0) - z

    return diff


class BoxVisualizer:

    name: str = None
    predictions: List[dict] = None
    axes: np.typing.NDArray = None
    dims: List[int] = None
    vocab: List[str]
    vocab_idx: Dict[str, int]
    y_boxes: Dict[str, Tuple[Tensor, Tensor]] = None    # Maps label to y_box
    y_vecs: Dict[str, Tensor] = None
    total_dims: int                                      # Dimensionality of boxes
    ax_lims: List[float] = [None] * 4

    def __init__(
            self,
            rundir: str,
            dims: Iterable[int],
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

        self.set_dims(dims)

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

        # if exists(self.rundir + "/box_vis_predictions.pkl") and exists(self.rundir + "/box_vis_y_boxes.pkl"):
        #     with open(self.rundir + "/box_vis_predictions.pkl", 'rb') as f:
        #         predictions = pickle.load(f)
        #     with open(self.rundir + "/box_vis_y_boxes.pkl", 'rb') as f:
        #         y_boxes = pickle.load(f)
        #     logger.info("Done reading data...")
        # else:
        logger.info("Starting to predict")
        predictions = []
        y_boxes: List[Tuple[Tensor, Tensor]] = []
        y_vecs: List[Tensor] = []
        for x in tqdm(data, desc="predicting"):
            r = self.predictor.predict_instance(x)

            prediction = {
                "label_scores": r["scores"],
                "predicted_labels": r["predictions"],
                "true_labels": r["meta"]["labels"]
            }

            if 'x_boxes_z' in r and "x_boxes_Z" in r:
                x_box_z = Tensor(r["x_boxes_z"])
                x_box_Z = Tensor(r["x_boxes_Z"])
                prediction["x_box"] = (x_box_z, x_box_Z)

            if 'x_vecs' in r:
                prediction['x_vec'] = Tensor(r['x_vecs'])

            predictions.append(prediction)

            if not y_boxes and "y_boxes_z" in r and "y_boxes_Z" in r:
                for y_box_z, y_box_Z in zip(r["y_boxes_z"], r["y_boxes_Z"]):
                    y_boxes.append((Tensor(y_box_z), Tensor(y_box_Z)))

            if not y_vecs and "y_vecs" in r:
                for y_vec in r["y_vecs"]:
                    y_vecs.append(Tensor(y_vec))

        with open(self.rundir + "/box_vis_predictions.pkl", 'wb') as f:
            pickle.dump(predictions, f)
        if y_boxes:
            with open(self.rundir + "/box_vis_y_boxes.pkl", 'wb') as f:
                pickle.dump(y_boxes, f)
        logger.info("Wrote prediction data to file")
        ## \t
        assert len(predictions) > 0
        self.predictions = predictions
        if y_boxes:
            self.y_boxes = {self.vocab[idx]: y_box for idx, y_box in enumerate(y_boxes)}
        if y_vecs:
            self.y_vecs = {self.vocab[idx]: y_vec for idx, y_vec in enumerate(y_vecs)}
        if "x_box" in predictions[0]:
            self.total_dims = len(predictions[0]["x_box"][0])

    def set_dims(
            self,
            dims: Iterable[int]
    ):
        self.clear()
        self.dims = list(dims)

    def visualize(
            self,
            x_lim: Optional[int] = None,
            plot_y_boxes: Optional[bool] = False,
            label_recursive: Optional[str] = None,
            show_plot: Optional[bool] = False,
            color_map: Optional[str] = None,
            auto_dims: Optional[int] = None
    ) -> None:
        """

        Args:
            x_lim: maximum number of x boxes to plot
            label_recursive: label filter where all child labels are also considered
                eg: "01.01.03" => ["01", "01.01", "01.01.03"]

        """
        logger.info("Visualizing...")

        if label_recursive:
            labels = list(filter(lambda l: label_recursive.startswith(l), self.vocab))

            if not color_map:  # Select colormap based on hash of label
                color_map = DEFAULT_COLOR_MAPS[hash(label_recursive) % len(DEFAULT_COLOR_MAPS)]

            _color_map = plt.cm.get_cmap(color_map, max([len(label) for label in labels]))

            def get_color(label: str):
                return _color_map(len(label))

            def get_x_label(pred):
                max_overlap_label = str()
                for true_label in pred['true_labels']:
                    if label_recursive.startswith(true_label) and len(true_label) > len(max_overlap_label):
                        max_overlap_label = true_label
                return max_overlap_label

            get_preds = lambda: self._filter_by_recursive_label(self.predictions, label_recursive, x_lim)
        else:
            labels = self.vocab
            get_preds = lambda: self.predictions[:x_lim] if x_lim is not None else self.predictions
            max_label_len = max([len(label) for label in labels])

            def get_color(label: str):
                if color_map:
                    map_name = color_map
                else:
                    map_name = DEFAULT_COLOR_MAPS[hash(label.split('.')[0]) % len(DEFAULT_COLOR_MAPS)]
                _color_map = plt.cm.get_cmap(map_name, max_label_len)
                return _color_map(len(label))

            def get_x_label(pred):
                return pred['true_labels'][0]

        if auto_dims:
            if self.y_boxes:
                volumes_by_dim = get_intersection_volume_by_dim(
                    [pred['x_box'] for pred in get_preds()],
                    [self.y_boxes[label] for label in labels],
                    self.predictor._model._intersect,
                    self.predictor._model._volume
                )
            elif self.y_vecs:
                volumes_by_dim = get_cone_score_by_dim(
                    [pred['x_vec'] for pred in get_preds()],
                    [self.y_vecs[label] for label in labels]
                )
            top_dims = self.find_highest_variance_dims(volumes_by_dim, top_k=auto_dims)
            self.set_dims(top_dims)

        num_graphs = int(np.ceil((len(self.dims) + 1)/2))

        if self.axes is None:
            fig: Figure
            axes: NDArray
            fig, axes = plt.subplots(num_graphs, 1, figsize=(6, num_graphs*6), squeeze=False)
            fig.suptitle(self.name)
            axes = axes.flatten()
            self.axes = axes

        # Plot y boxes
        if plot_y_boxes:
            for label in tqdm(labels, unit='y_boxes'):
                for x_dim, y_dim, ax in iter_dim_pairs(self.dims, axes=self.axes):
                    self._plot_y_entity(ax, label, x_dim, y_dim, get_color)

        # Plot x boxes as points
        for pred in tqdm(get_preds(), unit='x_boxes'):
            for x_dim, y_dim, ax in iter_dim_pairs(self.dims, axes=self.axes):
                self._plot_x_entity(pred, ax, get_x_label, x_dim, y_dim, get_color)

        for x_dim, y_dim, ax in iter_dim_pairs(self.dims, axes=self.axes):
            ax.title.set_text(f"({x_dim}, {y_dim})")

        if show_plot:
            self.show_plot()

    def clear(self):
        plt.clf()
        self.ax_lims = [None] * 4
        if self.axes is not None:
            del self.axes
            self.axes = None

    def show_plot(self):
        ax_lims = self.ax_lims
        for ax in self.axes:
            ax.set_xlim([ax_lims[0] - 0.1, ax_lims[1] + 0.1])
            ax.set_ylim([ax_lims[2] - 0.1, ax_lims[3] + 0.1])
        plt.show()
        self.clear()

    def _plot_x_entity(
            self,
            pred,
            ax: Axes,
            get_x_label: Callable,
            x_dim: int,
            y_dim: int,
            get_color: Callable
    ):
        ax_lims = self.ax_lims

        if 'x_box' in pred:
            x_box = pred['x_box']
            x, y = box_to_point(x_box, x_dim, y_dim)

        elif 'x_vec' in pred:
            x_vec = pred['x_vec']
            x, y = x_vec[x_dim], x_vec[y_dim]

        else:
            raise ValueError('No x_box or x_vec to plot')

        ax.scatter(x, y, color=get_color(get_x_label(pred)))

        if ax_lims[0] is None or x < ax_lims[0]:
            ax_lims[0] = x
        elif ax_lims[1] is None or x > ax_lims[1]:
            ax_lims[1] = x
        if ax_lims[2] is None or y < ax_lims[2]:
            ax_lims[2] = y
        elif ax_lims[3] is None or y > ax_lims[3]:
            ax_lims[3] = y

    def _plot_y_entity(
            self,
            ax: Axes,
            label: str,
            x_dim: int,
            y_dim: int,
            get_color: Callable
    ):
        ax_lims = self.ax_lims

        if self.y_boxes:
            y_box = self.y_boxes[label]
            rect: Rectangle = box_to_rectangle(y_box, x_dim, y_dim)

            rect.set_edgecolor(get_color(label))
            ax.add_patch(rect)

            if ax_lims[0] is None or y_box[0][x_dim] < ax_lims[0]:
                ax_lims[0] = y_box[0][x_dim]
            if ax_lims[1] is None or y_box[1][x_dim] > ax_lims[1]:
                ax_lims[1] = y_box[1][x_dim]
            if ax_lims[2] is None or y_box[0][y_dim] < ax_lims[2]:
                ax_lims[2] = y_box[0][y_dim]
            if ax_lims[3] is None or y_box[1][y_dim] > ax_lims[3]:
                ax_lims[3] = y_box[1][y_dim]

        if self.y_vecs:
            y_vec = self.y_vecs[label]

            wedge = Wedge(
                [y_vec[x_dim], y_vec[y_dim]],
                1e10, 0, 90,
                alpha=1,
                linewidth=2,
                fill=False,
                edgecolor=get_color(label)
            )
            ax.add_patch(wedge)

            if ax_lims[0] is None or y_vec[x_dim] < ax_lims[0]:
                ax_lims[0] = y_vec[x_dim]
            if ax_lims[1] is None or y_vec[x_dim] > ax_lims[1]:
                ax_lims[1] = y_vec[x_dim]
            if ax_lims[2] is None or y_vec[y_dim] < ax_lims[2]:
                ax_lims[2] = y_vec[y_dim]
            if ax_lims[3] is None or y_vec[y_dim] > ax_lims[3]:
                ax_lims[3] = y_vec[y_dim]

    def plot_intersection_heatmap(
            self,
            x_entities: List[Tuple[Tensor, Tensor]] = None,
            y_entities: List[Tuple[Tensor, Tensor]] = None,
            intersection: Intersection = None,
            volume: Volume = None,
            title: str = None
    ) -> None:
        title = title or self.name

        if self.y_boxes:
            x_boxes = x_entities or [pred['x_box'] for pred in self.predictions]
            y_boxes = y_entities or list(self.y_boxes.values())
            intersection = intersection or self.predictor._model._intersect
            volume = volume or self.predictor._model._volume

            intersection_tensor = get_intersection_volume_by_dim(
                x_boxes,
                y_boxes,
                intersection,
                volume
            )

        elif self.y_vecs:
            x_vecs = x_entities or [pred['x_vec'] for pred in self.predictions]
            y_vecs = y_entities or list(self.y_vecs.values())

            intersection_tensor = get_cone_score_by_dim(x_vecs, y_vecs)

        else:
            raise ValueError('No boxes or vectors to calculate scores for')

        variance_matrix = torch.var(intersection_tensor, dim=0)
        mean_matrix = torch.mean(intersection_tensor, dim=0)

        fig: Figure
        axes: List[Axes]
        fig, axes = plt.subplots(2, 1)

        fig.suptitle(f'{self.name} intersection heatmap by label and dimension')

        for idx, (title, data) in enumerate((
                ('variance (sqrt)', torch.sqrt(variance_matrix)),  # take sqrt of variance to increase contrast
                ('mean', mean_matrix)
        )):
            axes[idx].imshow(data, cmap='autumn', interpolation='none')
            axes[idx].set_title(title)

        plt.show()

        for title, data in (
                ('variance', torch.mean(variance_matrix, dim=0)),
                ('mean', torch.mean(mean_matrix, dim=0))
        ):
            fig, ax = plt.subplots(1, 1)
            fig.suptitle(f'{self.name} intersection by dimension')

            ax.plot(range(data.shape[0]), torch.sort(data, descending=True).values)
            ax.set_title(title)
            plt.show()

    @staticmethod
    def find_highest_variance_dims(
            volumes_by_dim: Tensor,
            top_k: Optional[int] = 0
            # x_entities: Union[List[Tuple[Tensor, Tensor]], List[Tensor]],
            # y_entities: Union[List[Tuple[Tensor, Tensor]], List[Tensor]],
            # intersection: Intersection = GumbelIntersection(),
            # volume: Volume = BesselApproxVolume(),
    ):
        variance_by_dim = torch.mean(
            torch.var(volumes_by_dim, dim=0),
            dim=0
        )
        top_k = variance_by_dim.argsort()[-top_k:]
        return torch.flip(top_k, dims=[0])

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
                    if limit:
                        yield_count += 1
                        if yield_count > limit:
                            return

                    yield pred


if __name__ == "__main__":

    cone_visualizer = BoxVisualizer(
        '/mnt/nfs/scratch1/asempruch/boxem/cone',
        name='Cone',
        dims=range(1000, 1004)
    )

    # cone_visualizer.visualize(
    #     x_lim=30,
    #     plot_y_boxes=True,
    #     label_recursive='20.03.01.01',
    #     color_map='Greens',
    #     show_plot=True,
    #     auto_dims=6
    # )

    cone_visualizer.plot_intersection_heatmap()

    # %% Two class y_boxes
    box_visualizer = BoxVisualizer(
        '/mnt/nfs/scratch1/asempruch/1750',
        name='Box Model',
        dims=range(0, 4)
    )

    # %% Plot intersection variance and mean heatmap

    # visualizer.plot_intersection_heatmap()


    # %% Plot label scores
    # buckets = {val: 0 for val in np.arange(-16, 0, 0.1)}
    # all_label_scores = list()
    # for pred in visualizer.predictions:
    #     for label_score in pred['label_scores']:
    #         for key in reversed(buckets.keys()):
    #             if label_score > key:
    #                 buckets[key] += 1
    #                 break
    # %%
    # plt.clf()
    # plt.rcParams['figure.dpi'] = 500
    # plt.bar(buckets.keys(), buckets.values(), width=0.05)
    # plt.xticks(
    #     [round(val, 2) for val in buckets.keys()],
    #     rotation=90,
    #     fontsize=2
    # )
    # # all_label_scores += pred['label_scores']
    # # print(len(all_label_scores))
    # plt.bar(range(len(all_label_scores)), sorted(all_label_scores))
    # plt.show()

    # %%

    # hierarchy_visualizer = BoxVisualizer(
    #     '/mnt/nfs/scratch1/asempruch/boxem/hierarchy_loss',
    #     name='Hierarchy Loss Box Model',
    #     dims=range(0, 4)
    # )

    # %%
    visualizer = cone_visualizer
    # Disjoint labels
    disjoint = [
        ('01.01.03.01.01', 'Greens'),
        ('10.03.05.01', 'Oranges')
    ]

    joint = [
        ('20.03.01.01', 'Blues'),
        ('20.09.07.03', 'Reds')
    ]

    for idx, (label, colors) in enumerate(disjoint):
        visualizer.visualize(
            x_lim=30,
            plot_y_boxes=True,
            label_recursive=label,
            color_map=colors,
            auto_dims=None if idx else 6
        )

    visualizer.show_plot()

    # %%

    # %% all y_boxes
    # visualizer.visualize(
    #     x_lim=None,
    #     # label_recursive='02.16.03',
    #     plot_y_boxes=True
    # )

    # visualizer.show_plot()

    # %% Two class y_boxes
    # visualizer = BoxVisualizer(
    #     '/home/asempruch/boxem/box-mlc/temp_',
    #     dims=range(1500, 1503)
    # )
    # visualizer.visualize(
    #     x_lim=30,
    #     label_recursive='10.03.02',
    #     plot_y_boxes=True,
    #     auto_dims=6,
    #     color_map='Reds'
    # )
    #
    # visualizer.visualize(
    #     x_lim=30,
    #     label_recursive='11.04.01',
    #     plot_y_boxes=True,
    #     color_map='Greens'
    # )
    #
    # visualizer.show_plot()
    #
    # # %% Three class one y_box
    # visualizer = BoxVisualizer(
    #     '/home/asempruch/boxem/box-mlc/temp_',
    #     dims=range(1500, 1503)
    # )
    # visualizer.visualize(
    #     x_lim=30,
    #     label_recursive='20.01.01.01.01.02',
    #     plot_y_boxes=True,
    #     auto_dims=6,
    #     color_map='Reds'
    # )
    #
    # visualizer.visualize(
    #     x_lim=30,
    #     label_recursive='02.16.03',
    #     color_map='Greens'
    # )
    #
    # visualizer.visualize(
    #     x_lim=30,
    #     label_recursive='01.01.03.05.02',
    #     color_map='Blues'
    # )
    #
    # visualizer.show_plot()

    # TODO: intelligently select dimensions based on variance in x box score (intersection)
    # you'll need a custom intersection function that allows you to compute in single dimsions, this will give you the
    # score and the metric you should be measuring variance in

    # %%
    # x = visualizer.predictions[100]['x_box']
    # y = visualizer.y_boxes['14']
    # xb = BoxTensor(x)
    # yb = BoxTensor(y)
    #
    # intersection = GumbelIntersection()(xb, yb)
    # HardVolume()(intersection)
    # SoftVolume()(intersection)
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


"""
TODO: Seperate errors into 
    * inter-path error - points not belonging to class given high score for it
    * itra-path error - parent point classified as child label
    
Iterate through all x points. Per x, set some threshold to classify whether it belongs to a label or not
Use formula from slides to compute ratio.

for x:
    for y:
        false_positives = ...
        true_positives = ...
"""


"""
TODO: <dataset_folder>/hierarchy_tc_edgelist
Construct adjacency matrix from file ignoring: root, GO00...

Construct false positive matrix using threshold and apply matrix multiplication

Elementwise(FP, Matmul(G, A))

Experiment with tensor batching

"""