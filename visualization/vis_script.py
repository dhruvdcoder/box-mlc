import json
import os
from typing import List, Tuple, Any
from pathlib import Path
import jsonlines
from allennlp.data.vocabulary import Vocabulary
from allennlp.models.archival import load_archive
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
import networkx as nx
import torch
from allennlp.common.util import import_module_and_submodules
from allennlp.predictors.predictor import Predictor, JsonDict
import sys
sys.path.append("../")
import logging
from typing import List
# import seaborn as sns
import ipywidgets as widgets
import matplotlib
from IPython.display import display
from box_embeddings.parameterizations.box_tensor import BoxTensor
import tqdm

from box_vis import BoxVisualizer

# import pydevd_pycharm
# pydevd_pycharm.settrace('amh.sempruch.com', port=41778, stdoutToServer=True, stderrToServer=True, suspend=False)

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", level=logging.INFO)
logger = logging.getLogger("notebook")

# top level variables
# os.chdir('../')
rundir = Path('/home/asempruch/boxem/box-mlc/temp_')
model_path = rundir/'model.tar.gz'
config_path = rundir/'config.json'
vocab_file = rundir/'vocabulary'

# load config
with open(config_path) as f:
    config = json.load(f)

import_module_and_submodules("box_mlc")

#%% Step 1: Load the model
# import pdb; pdb.set_trace()
archive = load_archive(
    model_path,
)
#%% Step 2: Get the predictor
predictor = Predictor.from_archive(
        archive,
        #args.predictor,
        dataset_reader_to_load="validation",
        frozen = True
    )
# Switch on the vis mode on model
predictor._model.visualization_mode = True

#%% Step 3: Perform predictions
data = list(predictor._dataset_reader.read(config["validation_data_path"]))
logger.info("Done reading data...")
logger.info("Starting to predict")
predictions = []
y_boxes: List[Tuple[List, List]] = []
for x in tqdm.tqdm(data, desc="predicting"):
    r = predictor.predict_instance(x)
    print(r.keys())
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

# print(predictions[0])

# %%
visualizer = BoxVisualizer('/home/asempruch/boxem/box-mlc/temp_')

# %%
def convert_to_rectangle(box: tuple, x_dim: int, y_dim: int) -> Rectangle:
    x_z, y_z = box[0][x_dim], box[0][y_dim]
    width = box[1][x_dim] - x_z
    height = box[1][y_dim] - y_z
    return Rectangle(
        (x_z, y_z),
        width,
        height,
        fill=False,
        alpha=0.1
    )

x_box = predictions[0]["x_box"]
dimensions = len(x_box[0])
assert dimensions == len(y_boxes[0][0])
plt.clf()
num_invisible_x_boxes = 0
y_rects = []
y_rects_generated = False
fig, ax = plt.subplots()
ax.set_xlim([-3, 3])
ax.set_ylim([-3, 3])
max_predictions = 100
max_dimensions = 1
for pred in predictions[:max_predictions]:
    x_box = pred["x_box"]
    for dim in range(400,401): # range(dimensions):
        x_dim = dim
        y_dim = dim+1 if x_dim < dimensions-1 else dim-1
        dim += 1

        x_rect = convert_to_rectangle(x_box, x_dim, y_dim)
        if not y_rects_generated:
            y_rects += [convert_to_rectangle(y_box, x_dim, y_dim) for y_box in y_boxes]

        if x_rect.get_width() == 0 or x_rect.get_height() == 0:
            num_invisible_x_boxes += 1
        else:
            ax.add_patch(x_rect)

    y_rects_generated = True

num_invisible_y_boxes = 0
for i, y_rect in enumerate(y_rects):  # TODO: remove limit
    if y_rect.get_width() == 0 or y_rect.get_height() == 0:
        num_invisible_y_boxes += 1
    else:
        ax.add_patch(y_rect)
    if i > 5:
        break
plt.show()

# TODO: software GEPHI
# TODO: add check if x box size is bigger than expected and throw warning

print("Invisible x_box:", 100 * num_invisible_x_boxes / max_dimensions*max_predictions, "%")
print("Invisible y_box:", 100 * num_invisible_y_boxes / max_dimensions*max_predictions, "%")
