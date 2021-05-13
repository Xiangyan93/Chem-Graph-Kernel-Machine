#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
CWD = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(CWD, '..'))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from chemml.args import EmbeddingArgs
from chemml.data.data import Dataset
from chemml.kernels.utils import get_kernel_config
from sklearn.manifold import TSNE
from sklearn.decomposition import KernelPCA


def plotmap(ax, X, Y, c, cmap='viridis', size=1, min=None, max=None):
    if min is None:
        min = c.min()
    if max is None:
        max = c.max()
    style = dict(s=size, cmap=cmap)
    sc = ax.scatter(X, Y, c=c, vmin=min, vmax=max, **style)
    return sc


def main(args: EmbeddingArgs) -> None:
    dataset = Dataset.load(args.save_dir, args=args).copy()
    kernel_config = get_kernel_config(args, dataset)
    if args.embedding_algorithm == 'tSNE':
        # compute data embedding.
        R = kernel_config.kernel(dataset.X)
        d = R.diagonal() ** -0.5
        K = d[:, None] * R * d[None, :]
        D = np.sqrt(np.maximum(0, 2-2 * K**2))
        embed = TSNE(n_components=args.n_components,
                     perplexity=args.perplexity,
                     n_iter=args.n_iter,
                     n_jobs=args.n_jobs).fit_transform(D)
    else:
        R = kernel_config.kernel(dataset.X)
        embed = KernelPCA(n_components=args.n_components,
                          kernel='precomputed',
                          n_jobs=args.n_jobs).fit_transform(R)
    # embedding dataframe.
    df = pd.DataFrame({'repr': dataset.repr.ravel()})
    for i in range(args.n_components):
        df['embedding_%d' % i] = embed[:, i]
    num_tasks = dataset.N_tasks
    if num_tasks == 1:
        df['y_0'] = dataset.y
    else:
        for i in range(num_tasks):
            df['y_%d' % i] = dataset.y[:, i]
    df.to_csv('%s/%s.csv' % (args.save_dir, args.embedding_algorithm), sep='\t', index=False)

    if args.save_png:
        for i in range(num_tasks):
            fig = plt.figure(figsize=(8, 8))
            ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
            p = 'y_%d' % i
            ymin = df[p].min()
            ymax = df[p].max()
            sc = plotmap(ax, df['embedding_0'], df['embedding_1'], df[p], size=5, min=ymin, max=ymax)
            ax.set_xlabel('Embedding 1')
            ax.set_ylabel('Embedding 2')
            ax.set_title(args.embedding_algorithm)
            fig.savefig('%s/%s_%s.png' % (args.save_dir, args.embedding_algorithm, p), dpi=300)


if __name__ == '__main__':
    main(args=EmbeddingArgs().parse_args())
