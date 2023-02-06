import pickle
from argparse import ArgumentParser
import time
from typing import Dict
from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np

import os

import ast

argparser = ArgumentParser()
argparser.add_argument('--dims', nargs='*', help='Dimesnionality')
argparser.add_argument('--model', nargs='*', help='Model types')
argparser.add_argument('--target', nargs='*', type=str)
argparser.add_argument('--dist', nargs='*', type=ast.literal_eval, help='Dimension distribution vector')
argparser.add_argument('--revvar', nargs='*', type=ast.literal_eval, help='Reverse variance vector')
argparser.add_argument('--sc', nargs='*', type=ast.literal_eval, help='Strict containment')
args = argparser.parse_args()

models = list()

score_keys = [
    'f1',
    'ndcg',
    'ap'
]

# plot_file_name = ' '.join(args.dims) +
plot_file_name = time.strftime("%y%m%d_%H%M%S")

plots = {}

for file_name in tqdm(os.listdir('scores')):
    # split = os.path.splitext(file_name)[0].split('_')
    dims, model_type, target, k, strict_containment, revvar, dist = os.path.splitext(file_name)[0].split('_')
    # query = {
    #     'dims': split[0],
    #     'model': split[1],
    # }
    dims = False if dims == '-' else dims
    revvar = revvar == 'revvar'
    dist = dist == 'dist'
    strict_containment = strict_containment == 'sc'

    if not (
        (not args.dims or dims in args.dims) and
        (not args.model or model_type in args.model) and
        (not args.target or target in args.target) and
        (not args.revvar or revvar in args.revvar) and
        (not args.sc or strict_containment in args.sc) and
        (not args.dist or dist in args.dist)
    ):
        continue

    with open(os.path.join('scores', file_name), mode='rb') as f:
        scores = pickle.load(f)

    # print('test')
    avg_scores = {
        ratio: {score_key: np.mean(scores[ratio][score_key]) for score_key in score_keys} for ratio in scores.keys()
    }

    for score_key in score_keys:
        if score_key not in plots:
            fig, ax = plt.subplots()
            plots[score_key] = (fig, ax)
            ax.set_title(score_key)
            ax.set_ylim(0, 1)
        else:
            _, ax = plots[score_key]

        ratios = list(scores.keys())
        score_data = [avg_scores[ratio][score_key] for ratio in scores.keys()]

        ax.scatter(ratios, score_data, label=f'{dims}_{model_type}_{target}_{k}_{"sc" if strict_containment else "-"}_{"revvar" if revvar else "-"}_{"dist" if dist else "-"}')

for score_key, (fig, ax) in plots.items():
    ax.legend(loc='lower left', prop={'size': 6})
    fig.show()
    fig.savefig(os.path.join('plots', f'{plot_file_name}_{score_key}.png'))
