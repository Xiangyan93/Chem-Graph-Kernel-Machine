#!/usr/bin/env python
# -*- coding: utf-8 -*-
from typing import Dict, Iterator, List, Optional, Union, Literal, Tuple
import os
import json
from graphdot.kernel.marginalized import MarginalizedGraphKernel
from graphdot.util.pretty_tuple import pretty_tuple
from graphdot.kernel.fix import Normalization
from graphdot.microkernel import (
    Additive,
    Constant as Const,
    TensorProduct,
    SquareExponential as sExp,
    KroneckerDelta as kDelta,
    Convolution as kConv,
    Normalize
)
from graphdot.microprobability import (
    Additive as Additive_p,
    Constant,
    UniformProbability,
    AssignProbability
)
from .BaseKernelConfig import BaseKernelConfig
from .HybridKernel import *
from .ConvKernel import *
from .utils import get_kernel_config
from ..data import Dataset
from ..args import KernelArgs


class Norm(Normalization):
    @property
    def requires_vector_input(self):
        return False

    def get_params(self, deep=False):
        return dict(
            kernel=self.kernel,
        )

    @property
    def n_dims(self):
        """Returns the number of non-fixed hyperparameters of the kernel."""
        return self.theta.shape[0]


class NormalizationMolSize(Norm):
    def __init__(self, kernel, s=100.0, s_bounds=(1e2, 1e3)):
        super().__init__(kernel)
        self.s = s
        self.s_bounds = s_bounds

    def __diag(self, K, l, r, K_gradient=None):
        l_lr = np.repeat(l, len(r)).reshape(len(l), len(r))
        r_rl = np.repeat(r, len(l)).reshape(len(r), len(l))
        se = np.exp(-((1 / l_lr ** 2 - 1 / r_rl.T ** 2) / self.s) ** 2)
        K = np.einsum("i,ij,j,ij->ij", l, K, r, se)
        if K_gradient is None:
            return K
        else:
            K_gradient = np.einsum("ijk,i,j,ij->ijk", K_gradient, l, r, se)
            if self.s_bounds == "fixed":
                return K, K_gradient
            else:
                dK_s = 2 * (l_lr - r_rl.T) ** 2 / self.s ** 3 * K
                dK_s = dK_s.reshape(len(l), len(r), 1)
                return K, np.concatenate([K_gradient, dK_s], axis=2)
            # return K, np.einsum("ijk,i,j,ij->ijk", K_gradient, l, r, se)

    def __call__(self, X, Y=None, eval_gradient=False, **options):
        """Normalized outcome of
        :py:`self.kernel(X, Y, eval_gradient, **options)`.

        Parameters
        ----------
        Inherits that of the graph kernel object.

        Returns
        -------
        Inherits that of the graph kernel object.
        """
        if eval_gradient is True:
            R, dR = self.kernel(X, Y, eval_gradient=True, **options)
            if Y is None:
                ldiag = rdiag = R.diagonal()
            else:
                ldiag, ldDiag = self.kernel.diag(X, True, **options)
                rdiag, rdDiag = self.kernel.diag(Y, True, **options)
            ldiag_inv = 1 / ldiag
            rdiag_inv = 1 / rdiag
            ldiag_rsqrt = np.sqrt(ldiag_inv)
            rdiag_rsqrt = np.sqrt(rdiag_inv)
            return self.__diag(R, ldiag_rsqrt, rdiag_rsqrt, dR)
        else:
            R = self.kernel(X, Y, **options)
            if Y is None:
                ldiag = rdiag = R.diagonal()
            else:
                ldiag = self.kernel.diag(X, **options)
                rdiag = self.kernel.diag(Y, **options)
            ldiag_inv = 1 / ldiag
            rdiag_inv = 1 / rdiag
            ldiag_rsqrt = np.sqrt(ldiag_inv)
            rdiag_rsqrt = np.sqrt(rdiag_inv)
            # K = ldiag_rsqrt[:, None] * R * rdiag_rsqrt[None, :]
            return self.__diag(R, ldiag_rsqrt, rdiag_rsqrt)

    @property
    def n_dims(self):
        if self.s_bounds == "fixed":
            return len(self.kernel.theta)
        else:
            return len(self.kernel.theta) + 1

    @property
    def hyperparameters(self):
        if self.s_bounds == "fixed":
            return self.kernel.hyperparameters
        else:
            return pretty_tuple(
                'MarginalizedGraphKernel',
                ['starting_probability', 'stopping_probability', 'node_kernel',
                 'edge_kernel', 'normalize_size']
            )(self.kernel.p.theta,
              self.kernel.q,
              self.kernel.node_kernel.theta,
              self.kernel.edge_kernel.theta,
              np.log(self.s))

    @property
    def hyperparameter_bounds(self):
        if self.s_bounds == "fixed":
            return self.kernel.hyperparameter_bounds
        else:
            return pretty_tuple(
                'GraphKernelHyperparameterBounds',
                ['starting_probability', 'stopping_probability', 'node_kernel',
                 'edge_kernel', 'normalize_size']
            )(self.kernel.p.bounds,
              self.kernel.q_bounds,
              self.kernel.node_kernel.bounds,
              self.kernel.edge_kernel.bounds,
              self.s_bounds)

    @property
    def theta(self):
        if self.s_bounds == "fixed":
            return self.kernel.theta
        else:
            return np.r_[self.kernel.theta, np.log(self.s)]

    @theta.setter
    def theta(self, value):
        if self.s_bounds == "fixed":
            self.kernel.theta = value
        else:
            self.kernel.theta = value[:-1]
            self.s = np.exp(value[-1])

    @property
    def bounds(self):
        if self.s_bounds == "fixed":
            return self.kernel.bounds
        else:
            return np.r_[self.kernel.bounds, np.log(np.reshape(self.s_bounds, (1, 2)))]

    def clone_with_theta(self, theta):
        clone = copy.deepcopy(self)
        clone.theta = theta
        return clone

    def get_params(self, deep=False):
        return dict(
            kernel=self.kernel,
            s=self.s,
            s_bounds=self.s_bounds,
        )


class MGK(MarginalizedGraphKernel):
    """
    X and Y could be 2-d numpy array.
    make it compatible with sklearn.
    remove repeated kernel calculations, if set unique=True.
    """

    def __init__(self, unique=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.unique = unique

    @staticmethod
    def _unique(X):
        X_unique = np.sort(np.unique(X))
        X_idx = np.searchsorted(X_unique, X)
        return X_unique, X_idx

    @staticmethod
    def _format_X(X):
        if X.__class__ == np.ndarray:
            return X.ravel()  # .tolist()
        else:
            return X

    def __call__(self, X, Y=None, eval_gradient=False, *args, **kwargs):
        X = self._format_X(X)
        Y = self._format_X(Y)
        if self.unique:
            X_unique, X_idx = self._unique(X)
            if Y is None:
                Y_unique, Y_idx = X_unique, X_idx
            else:
                Y_unique, Y_idx = self._unique(Y)
            if eval_gradient:
                K, K_gradient = super().__call__(
                    X_unique, Y_unique, eval_gradient=True, *args, **kwargs
                )
                return K[X_idx][:, Y_idx], K_gradient[X_idx][:, Y_idx][:]
            else:
                K = super().__call__(X_unique, Y_unique, eval_gradient=False,
                                     *args, **kwargs)
                return K[X_idx][:, Y_idx]
        else:
            return super().__call__(X, Y, eval_gradient=eval_gradient, *args,
                                    **kwargs)

    def diag(self, X, *args, **kwargs):
        X = self._format_X(X)
        if self.unique:
            X_unique, X_idx = self._unique(X)
            diag = super().diag(X_unique, *args, **kwargs)
            return diag[X_idx]
        else:
            return super().diag(X, *args, **kwargs)

    def get_params(self, deep=False):
        return dict(
            node_kernel=self.node_kernel,
            edge_kernel=self.edge_kernel,
            p=self.p,
            q=self.q,
            q_bounds=self.q_bounds,
            backend=self.backend
        )


class ConvolutionGraphKernel(MGK):
    def __call__(self, X, Y=None, eval_gradient=False, *args, **kwargs):
        graph, K_graph, K_gradient_graph = self.__get_K_dK(
            X, Y, eval_gradient)
        return ConvolutionKernel()(
            X, Y=Y, eval_gradient=eval_gradient, graph=graph, K_graph=K_graph,
            K_gradient_graph=K_gradient_graph, theta=self.theta)

    def diag(self, X, eval_gradient=False, n_process=cpu_count(),
                 K_graph=None, K_gradient_graph=None, *args, **kwargs):
        graph, K_graph, K_gradient_graph = self.__get_K_dK(
            X, None, eval_gradient)
        return ConvolutionKernel().diag(
            X, eval_gradient=eval_gradient, graph=graph, K_graph=K_graph,
            K_gradient_graph=K_gradient_graph, theta=self.theta)

    def __get_K_dK(self, X, Y, eval_gradient):
        X = ConvolutionKernel._format_X(X)
        Y = ConvolutionKernel._format_X(Y)
        graph = ConvolutionKernel.get_graph(X, Y)
        if eval_gradient:
            K_graph, K_gradient_graph = super().__call__(
                graph, eval_gradient=True)
        else:
            K_graph = super().__call__(graph)
            K_gradient_graph = None
        return graph, K_graph, K_gradient_graph


class GraphKernelConfig(BaseKernelConfig):
    def __init__(self, N_MGK: int = 0, N_conv_MGK: int = 0,
                 graph_hyperparameters: List[Dict] = [],
                 unique: bool = False,
                 N_RBF: int = 0,
                 sigma_RBF: List[float] = None,
                 sigma_RBF_bounds: List[Tuple[float, float]] = None):
        super().__init__(N_RBF, sigma_RBF, sigma_RBF_bounds)
        self.N_MGK = N_MGK
        self.N_conv_MGK = N_conv_MGK
        self.graph_hyperparameters = graph_hyperparameters
        self.unique = unique
        assert (len(self.graph_hyperparameters) == N_MGK + N_conv_MGK)
        self._update_kernel()

    def _get_single_graph_kernel(self, hyperdict: Dict):
        knode, kedge, p = self._get_knode_kedge_p(hyperdict)
        kernel = MGK(
            node_kernel=knode,
            edge_kernel=kedge,
            q=hyperdict['q'][0],
            q_bounds=hyperdict['q'][1],
            p=p,
            unique=self.unique
        )
        if hyperdict['Normalization'][0] == True:
            return Norm(kernel)
        elif hyperdict['Normalization'][0] == False:
            return kernel
        else:
            return NormalizationMolSize(
                kernel, s=hyperdict['Normalization'][0],
                s_bounds=hyperdict['Normalization'][1])

    def _get_conv_graph_kernel(self, hyperdict: Dict):
        knode, kedge, p = self._get_knode_kedge_p(hyperdict)
        kernel = ConvolutionGraphKernel(
            node_kernel=knode,
            edge_kernel=kedge,
            q=hyperdict['q'][0],
            q_bounds=hyperdict['q'][1],
            p=p,
            unique=self.unique
        )
        if hyperdict['Normalization'] == True:
            return Norm(kernel)
        elif hyperdict['Normalization'] == False:
            return kernel
        elif hyperdict['Normalization'][0]:
            return NormalizationMolSize(
                kernel, s=hyperdict['Normalization'][1],
                s_bounds=hyperdict['Normalization'][2])

    def _get_knode_kedge_p(self, hyperdict: Dict[str, Union[List, Dict]]):
        knode_dict = {}
        kedge_dict = {}
        p_dict = {}
        for key, microk_dict in hyperdict.items():
            if key.startswith('atom_'):
                microk = [self._get_microk(k, mk)
                          for k, mk in microk_dict.items()]
                knode_dict.update({key[5:]: np.product(microk)})
            elif key.startswith('bond_'):
                microk = [self._get_microk(k, mk)
                          for k, mk in microk_dict.items()]
                kedge_dict.update({key[5:]: np.product(microk)})
            elif key.startswith('probability_'):
                microp = [self._get_microk(k, mk)
                          for k, mk in microk_dict.items()]
                p_dict.update({key[12:]: np.product(microp)})

        return self._combine_microk(hyperdict['a_type'][0], knode_dict), \
               self._combine_microk(hyperdict['b_type'][0], kedge_dict), \
               self._combine_microk(hyperdict['p_type'][0], p_dict)

    @staticmethod
    def _get_microk(microk: str, value: List):
        # For fixed hyperparameter
        if value[1] != 'fixed':
            value[1] = tuple(value[1])
        # microkernels
        if microk == 'kDelta':
            return kDelta(value[0], value[1])
        elif microk == 'sExp':
            return sExp(value[0], length_scale_bounds=value[1])
        elif microk == 'kConv':
            return kConv(kDelta(value[0], value[1]))
        elif microk == 'Const':
            return Const(value[0], value[1])
        # microprobability
        elif microk == 'Uniform_p':
            return UniformProbability(value[0], value[1])
        elif microk == 'Const_p':
            return Constant(value[0], value[1])
        elif microk == 'Assign_p':
            return AssignProbability(value[0], value[1])
        else:
            raise RuntimeError(f'Unknown microkernel type {microk}')

    @staticmethod
    def _combine_microk(rule: Literal['Tensorproduct', 'Additive', 'Additive_p'],
                        microk_dict: Dict):
        if rule == 'Tensorproduct':
            return TensorProduct(**microk_dict)
        elif rule == 'Additive':
            return Normalize(Additive(**microk_dict))
        elif rule == 'Additive_p':
            return Additive_p(**microk_dict)
        else:
            raise RuntimeError(f'Unknown type: {rule}')

    def get_preCalc_kernel_config(self, args: KernelArgs, dataset: Dataset):
        dataset.set_ignore_features_add(True)
        N_RBF = self.N_RBF
        self.N_RBF = 0
        self._update_kernel()
        kernel_dict = self.get_kernel_dict(dataset.X_mol, dataset.X_repr.ravel())
        dataset.set_ignore_features_add(False)
        self.N_RBF = N_RBF
        args.graph_kernel_type = 'preCalc'
        return get_kernel_config(args, dataset, kernel_dict)

    # save hyperparameters files and kernel.pkl.
    def save_hyperparameters(self, path: str):
        for i, hyperdict in enumerate(self.graph_hyperparameters):
            open(os.path.join(path, 'hyperparameters_%d.json' % i), 'w').write(
                json.dumps(hyperdict, indent=1, sort_keys=False))
        super().save_hyperparameters(path)

    # functions for Bayesian optimization of hyperparameters.
    def get_space(self):
        SPACE = dict()
        for i, hyperdict in enumerate(self.graph_hyperparameters):
            for key, value in hyperdict.items():
                if value.__class__ == list:
                    hp_key = '%d:%s:' % (i, key)
                    hp_ = self._get_hp(hp_key, value)
                    if hp_ is not None:
                        SPACE[hp_key] = hp_
                else:
                    for micro_key, micro_value in value.items():
                        hp_key = '%d:%s:%s' % (i, key, micro_key)
                        hp_ = self._get_hp(hp_key, micro_value)
                        if hp_ is not None:
                            SPACE[hp_key] = hp_

        SPACE.update(super().get_space())
        return SPACE

    def update_from_space(self, hyperdict: Dict[str, Union[int, float]]):
        for key, value in hyperdict.items():
            n, term, microterm = key.split(':')
            # RBF kernels
            if n == 'RBF':
                n_rbf = int(term)
                self.sigma_RBF[n_rbf] = value
            else:
                n = int(n)
                if term in ['Normalization', 'q', 'a_type', 'b_type']:
                    self.graph_hyperparameters[n][term][0] = value
                else:
                    self.graph_hyperparameters[n][term][microterm][0] = value
        self._update_kernel()

    def update_from_theta(self):
        hyperparameters = np.exp(self.kernel.theta).tolist()
        for hyperdict in self.graph_hyperparameters:
            for key in self._get_order_keys(hyperdict):
                if key.__class__ == str:
                    hyperdict[key][0] = hyperparameters.pop(0)
                else:
                    hyperdict[key[0]][key[1]][0] = hyperparameters.pop(0)

        if self.N_RBF == 0 or self.sigma_RBF_bounds == 'fixed':
            assert len(hyperparameters) == 0
        else:
            assert len(hyperparameters) == len(self.sigma_RBF)
            self.sigma_RBF = hyperparameters

    @staticmethod
    def _get_order_keys(hyperdict: Dict) -> List[Union[List[str], str]]:
        sort_keys = []
        # starting probability
        for key, d in hyperdict.items():
            if key.startswith('probability_'):
                for key_, value in d.items():
                    if value[1] != 'fixed':
                        sort_keys.append([key, key_])
        # stop probability
        if hyperdict['q'][1] != 'fixed':
            sort_keys.append('q')
        # atom features
        for key, d in hyperdict.items():
            if key.startswith('atom_'):
                for key_, value in d.items():
                    if value[1] != 'fixed':
                        sort_keys.append([key, key_])
        # bond features
        for key, d in hyperdict.items():
            if key.startswith('bond_'):
                for key_, value in d.items():
                    if value[1] != 'fixed':
                        sort_keys.append([key, key_])
        # normalization
        if hyperdict['Normalization'] in [True, False]:
            pass
        elif hyperdict['Normalization'][1] != 'fixed':
            sort_keys.append('Normalization')
        return sort_keys

    def _update_kernel(self):
        N_MGK = self.N_MGK
        N_conv_MGK = self.N_conv_MGK
        N_RBF = self.N_RBF
        if N_MGK == 1 and N_conv_MGK == 0 and N_RBF == 0:
            self.kernel = self._get_single_graph_kernel(
                self.graph_hyperparameters[0])
        elif N_MGK == 0 and N_conv_MGK == 1 and N_RBF == 0:
            self.kernel = self._get_conv_graph_kernel(
                self.graph_hyperparameters[0])
        else:
            kernels = []
            for i in range(N_MGK):
                kernels.append(
                    self._get_single_graph_kernel(self.graph_hyperparameters[i]))
            for i in range(N_MGK, N_MGK + N_conv_MGK):
                kernels.append(
                    self._get_conv_graph_kernel(self.graph_hyperparameters[i]))
            kernels += self._get_rbf_kernel()
            composition = [(i,) for i in range(N_MGK + N_conv_MGK)] + \
                          [tuple(np.arange(N_MGK + N_conv_MGK,
                                           N_RBF + N_MGK + N_conv_MGK))]
            self.kernel = HybridKernel(
                kernel_list=kernels,
                composition=composition,
                hybrid_rule='product',
            )
