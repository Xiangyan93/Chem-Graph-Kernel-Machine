#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count


class ConvolutionKernel:
    def __call__(self, X, Y=None, eval_gradient=False, n_process=cpu_count(),
                 graph=None, K_graph=None, K_gradient_graph=None, theta=None):
        # format X, Y and check input.
        assert (graph is not None and K_graph is not None)
        assert (not eval_gradient or K_gradient_graph is not None)
        assert (len(theta) > 0)
        X = self._format_X(X)
        Y = self._format_X(Y)
        graph_ = self.get_graph(X, Y)
        assert (len(graph) == len(graph_))
        assert (False not in (graph == graph_))
        # set X and Y index.
        if Y is None:
            trivial_Xidx = np.arange(0, len(X))[
                list(map(lambda x: bool(1 - bool(x)), X))]
            trivial_Yidx = trivial_Xidx
            Xidx, Yidx = np.triu_indices(len(X))
            Xidx, Yidx = Xidx.astype(np.uint32), Yidx.astype(np.uint32)
            Y = X
            symmetric = True
        else:
            trivial_Xidx = np.arange(0, len(X))[
                list(map(lambda x: bool(1 - bool(x)), X))]
            trivial_Yidx = np.arange(0, len(Y))[
                list(map(lambda x: bool(1 - bool(x)), Y))]
            Xidx, Yidx = np.indices((len(X), len(Y)), dtype=np.uint32)
            Xidx = Xidx.ravel()
            Yidx = Yidx.ravel()
            symmetric = False
        # find and seperate empty trivial input.
        df_idx = pd.DataFrame({'Xidx': Xidx, 'Yidx': Yidx})
        df_one = df_idx[(df_idx.Xidx.isin(trivial_Xidx)) &
                        (df_idx.Yidx.isin(trivial_Yidx))]
        df_zero = df_idx[(df_idx.Xidx.isin(trivial_Xidx)) ^
                         (df_idx.Yidx.isin(trivial_Yidx))]
        df = df_idx[(~df_idx.Xidx.isin(trivial_Xidx)) &
                    (~df_idx.Yidx.isin(trivial_Yidx))]
        df_parts = np.array_split(df, n_process)
        if eval_gradient:
            K = np.zeros((len(X), len(Y)))
            K_gradient = np.zeros((len(X), len(Y), theta.shape[0]))
            with Pool(processes=n_process) as pool:
                result_parts = pool.map(
                    self.compute_k_gradient, [(df_part, X, Y, graph, K_graph,
                                               K_gradient_graph)
                                              for df_part in df_parts])
            result = np.concatenate(result_parts)
            K[df['Xidx'], df['Yidx']] = list(map(lambda x: x[0], result))
            K[df_one['Xidx'], df_one['Yidx']] = 1.0
            K[df_zero['Xidx'], df_zero['Yidx']] = 0.0
            K_gradient[df['Xidx'], df['Yidx']] = \
                list(map(lambda x: x[1], result))
            K_gradient[df_one['Xidx'], df_one['Yidx']] = 0.0
            K_gradient[df_zero['Xidx'], df_zero['Yidx']] = 0.0
        else:
            K = np.zeros((len(X), len(Y)))
            with Pool(processes=n_process) as pool:
                result_parts = pool.map(
                    self.compute_k, [(df_part, X, Y, graph, K_graph)
                                     for df_part in df_parts])
            K[df['Xidx'], df['Yidx']] = np.concatenate(result_parts)
            K[df_one['Xidx'], df_one['Yidx']] = 1.0
            K[df_zero['Xidx'], df_zero['Yidx']] = 0.0

        if symmetric:
            K = K + K.T
            K[np.diag_indices(len(X))] /= 2
            if eval_gradient:
                K_gradient = K_gradient + K_gradient.transpose([1, 0, 2])
                K_gradient[np.diag_indices(len(X))] /= 2
        if eval_gradient:
            return K, K_gradient
        else:
            return K

    def diag(self, X, eval_gradient=False, n_process=cpu_count(), graph=None,
             K_graph=None, K_gradient_graph=None, theta=None):
        # format X and check input.
        assert (graph is not None and K_graph is not None)
        assert (not eval_gradient or K_gradient_graph is not None)
        assert (len(theta) > 0)
        X = self._format_X(X)
        graph_ = self.get_graph(X, None)
        assert (len(graph) == len(graph_))
        assert (False not in (graph == graph_))
        # set X index.
        Xidx = np.arange(0, len(X))
        trivial_Xidx = np.arange(0, len(X))[
            list(map(lambda x: bool(1 - bool(x)), X))]
        # find and seperate empty trivial input.
        df_idx = pd.DataFrame({'Xidx': np.arange(0, len(X), dtype=np.uint32),
                               'Yidx': np.arange(0, len(X), dtype=np.uint32)})
        df_one = df_idx[(df_idx.Xidx.isin(trivial_Xidx))]
        df = df_idx[(~df_idx.Xidx.isin(trivial_Xidx))]
        df_parts = np.array_split(df, n_process)
        if eval_gradient:
            D = np.zeros(len(X))
            D_gradient = np.zeros((len(X), theta.shape[0]))
            with Pool(processes=n_process) as pool:
                result_parts = pool.map(
                    self.compute_k_gradient, [(df_part, X, X, graph, K_graph,
                                               K_gradient_graph)
                                              for df_part in df_parts])
            result = np.concatenate(result_parts)
            D[df['Xidx']] = list(map(lambda x: x[0], result))
            D[df_one['Xidx']] = 1.0
            D_gradient[df['Xidx']] = list(map(lambda x: x[1], result))
            D_gradient[df_one['Xidx']] = 0.0
            return D, D_gradient
        else:
            D = np.zeros(len(X))
            with Pool(processes=n_process) as pool:
                result_parts = pool.map(
                    self.compute_k, [(df_part, X, X, graph, K_graph)
                                     for df_part in df_parts])
            D[df['Xidx']] = np.concatenate(result_parts)
            D[df_one['Xidx']] = 1.0
            return D

    @staticmethod
    def _format_X(X):
        if X.__class__ == np.ndarray:
            return X.ravel()  # .tolist()
        else:
            return X

    @staticmethod
    def x2graph(x):
        return x[::2]

    @staticmethod
    def x2weight(x):
        return np.asarray(x[1::2])

    @staticmethod
    def get_graph(X, Y=None):
        graphs = []
        for x in X:
            graphs += ConvolutionKernel.x2graph(x)
        if Y is not None:
            for y in Y:
                graphs += ConvolutionKernel.x2graph(y)
        return np.sort(np.unique(graphs))

    @staticmethod
    def Kxy(x, y, eval_gradient=False, graph=None, K_graph=None,
            K_gradient_graph=None):
        def get_reaction_smarts(g, g_weight):
            reactants = []
            products = []
            for i, weight in enumerate(g_weight):
                if weight > 0:
                    reactants.append(g.smiles)
                elif weight < 0:
                    products.append(g.smiles)
            return '.'.join(reactants) + '>>' '.'.join(products)

        x, x_weight = ConvolutionKernel.x2graph(x), ConvolutionKernel.x2weight(x)
        y, y_weight = ConvolutionKernel.x2graph(y), ConvolutionKernel.x2weight(y)
        x_idx = np.searchsorted(graph, x)
        y_idx = np.searchsorted(graph, y)
        Kxy = K_graph[x_idx][:, y_idx]
        Kxx = K_graph[x_idx][:, x_idx]
        Kyy = K_graph[y_idx][:, y_idx]
        Fxy = x_weight.dot(Kxy).dot(y_weight)
        Fxx = x_weight.dot(Kxx).dot(x_weight)
        Fyy = y_weight.dot(Kyy).dot(y_weight)
        if Fxx <= 0.:
            raise Exception('trivial reaction: ',
                            get_reaction_smarts(x, x_weight))
        if Fyy == 0.:
            raise Exception('trivial reaction: ',
                            get_reaction_smarts(y, y_weight))
        if eval_gradient:
            dKxy = K_gradient_graph[x_idx][:, y_idx][:]
            dKxx = K_gradient_graph[x_idx][:, x_idx][:]
            dKyy = K_gradient_graph[y_idx][:, y_idx][:]
            dFxy = np.einsum("i,j,ijk->k", x_weight, y_weight, dKxy)
            dFxx = np.einsum("i,j,ijk->k", x_weight, x_weight, dKxx)
            dFyy = np.einsum("i,j,ijk->k", y_weight, y_weight, dKyy)
            sqrtFxxFyy = np.sqrt(Fxx * Fyy)
            return Fxy / sqrtFxxFyy, (
                    dFxy - 0.5 * dFxx / Fxx - 0.5 * dFyy / Fyy) / sqrtFxxFyy
        else:
            return Fxy / np.sqrt(Fxx * Fyy)

    @staticmethod
    def compute_k(args_kwargs):
        df, X, Y, graph, K_graph = args_kwargs
        return df.progress_apply(
            lambda x: ConvolutionKernel.Kxy(
                X[x['Xidx']], Y[x['Yidx']], graph=graph, K_graph=K_graph),
            axis=1)

    @staticmethod
    def compute_k_gradient(args_kwargs):
        df, X, Y, graph, K_graph, K_gradient_graph = args_kwargs
        return df.progress_apply(
            lambda x: ConvolutionKernel.Kxy(
                X[x['Xidx']], Y[x['Yidx']], graph=graph, K_graph=K_graph,
                K_gradient_graph=K_gradient_graph, eval_gradient=True),
            axis=1)
