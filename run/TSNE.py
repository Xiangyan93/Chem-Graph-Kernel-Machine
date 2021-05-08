#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
CWD = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(CWD, '..'))
from chemml.args import KernelArgs
from chemml.data.data import Dataset
from chemml.kernels.utils import get_kernel_config
from sklearn.manifold import TSNE
from chemml.evaluator import Evaluator


def plotmap(ax, X, Y, c, cmap='viridis', size=1, min=None, max=None):
    if min is None:
        min = c.min()
    if max is None:
        max = c.max()
    style = dict(s=size, cmap=cmap)
    sc = ax.scatter(X, Y, c=c, vmin=min, vmax=max, **style)
    return sc


def main(args: KernelArgs) -> None:
    dataset = Dataset.load(args.save_dir)
    dataset.graph_kernel_type = args.graph_kernel_type
    kernel_config = get_kernel_config(args, dataset)
    R = kernel_config.kernel(dataset.X)
    d = R.diagonal() ** -0.5
    K = d[:, None] * R * d[None, :]
    D = np.sqrt(np.maximum(0, 2-2 * K**2))
    embed = TSNE(n_components=2, n_iter=10000).fit_transform(D)
    num_tasks = dataset.y.shape[1]
    embed_dict = {
        'embed_X': embed[:, 0],
        'embed_Y': embed[:, 1]
    }
    for i in range(num_tasks):
        embed_dict['target_%d' % i] = dataset.y[:, i]
    df = pd.DataFrame(embed_dict)

    for i in range(num_tasks):
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_axes([0, 0, 1, 1])
        p = 'target_%d' % i
        ymin = df[p].min()
        ymax = df[p].max()
        sc = plotmap(ax, df['embed_X'], df['embed_Y'], df[p], size=5, min=ymin,
                     max=ymax)
        fig.savefig('%s.png' % p, dpi=300)
    print(df)
    #df_out['embed_X'] = embed[:, 0]
    #df_out['embed_Y'] = embed[:, 1]
    #df_out.to_csv('%s_embed_tSNE.log' % properties[0], sep=' ', index=False)


if __name__ == '__main__':
    main(args=KernelArgs().parse_args())