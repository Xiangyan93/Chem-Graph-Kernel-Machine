import os
import json
from tqdm import tqdm
tqdm.pandas()
from graphdot.kernel.marginalized import MarginalizedGraphKernel
from graphdot.kernel.fix import Normalization
from graphdot.microkernel import (
    Additive,
    Constant as kC,
    TensorProduct,
    SquareExponential as sExp,
    KroneckerDelta as kDelta,
    Convolution as kConv,
    Normalize
)
from graphdot.microprobability import (
    Additive as Additive_p,
    Constant,
    UniformProbability
)
from chemml.kernels.KernelConfig import KernelConfig
from chemml.kernels.MultipleKernel import *
from chemml.kernels.ConvKernel import *


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


class GraphKernelConfig(KernelConfig):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.type = 'graph'
        self.single_graph = self.params['single_graph']
        self.multi_graph = self.params['multi_graph']
        self.hyperdict = self.params['hyperdict']
        ns = len(self.single_graph)
        nm = len(self.multi_graph)
        assert (len(self.hyperdict) == ns + nm)
        if ns == 1 and nm == 0 and self.add_features is None:
            self.kernel = self.get_single_graph_kernel(self.hyperdict[0])
        elif ns == 0 and nm == 1 and self.add_features is None:
            self.kernel = self.get_conv_graph_kernel(self.hyperdict[0])
        else:
            na = 0 if self.add_features is None else len(self.add_features)
            kernels = []
            for i in range(ns):
                kernels += [self.get_single_graph_kernel(self.hyperdict[i])]
            for i in range(ns, ns + nm):
                kernels += [self.get_conv_graph_kernel(self.hyperdict[i])]
            kernels += self.get_rbf_kernel()
            composition = [(i,) for i in range(ns + nm)] + \
                          [tuple(np.arange(ns + nm, na + ns + nm))]
            self.kernel = MultipleKernel(
                kernel_list=kernels,
                composition=composition,
                combined_rule='product',
            )
        if self.hyperdict[0].get('theta') is not None:
            theta = []
            for hyperdict in self.hyperdict:
                theta += hyperdict['theta']
            if theta is not None:
                print('Reading Existed kernel parameter %s' % theta)
                self.kernel = self.kernel.clone_with_theta(theta)

    def get_single_graph_kernel(self, hyperdict):  # dont delete kernel_pkl
        knode, kedge, p = self.get_knode_kedge_p(hyperdict)
        kernel = MGK(
            node_kernel=knode,
            edge_kernel=kedge,
            q=hyperdict['q'][0],
            q_bounds=hyperdict['q'][1],
            p=p,
            unique=self.add_features is not None
        )
        if hyperdict['normalization']:
            kernel = Normalization(kernel)
        return kernel

    def get_conv_graph_kernel(self, hyperdict):  # dont delete kernel_pkl
        knode, kedge, p = self.get_knode_kedge_p(hyperdict)
        kernel = ConvolutionGraphKernel(
            node_kernel=knode,
            edge_kernel=kedge,
            q=hyperdict['q'][0],
            q_bounds=hyperdict['q'][1],
            p=p,
            unique=self.add_features is not None
        )
        if hyperdict['normalization']:
            kernel = Normalization(kernel)
        return kernel

    @staticmethod
    def get_knode_kedge_p(hyperdict):
        def get_microk(microk):
            if microk[2] != 'fixed':
                microk[2] = tuple(microk[2])
            if microk[0] == 'kDelta':
                return kDelta(microk[1], microk[2])
            elif microk[0] == 'sExp':
                # if microk[2] == 'fixed':
                # microk[2] = (microk[1], microk[1])
                return sExp(microk[1], length_scale_bounds=microk[2])
            elif microk[0] == 'kConv':
                return kConv(kDelta(microk[1], microk[2]))
            elif microk[0] == 'kC':
                return kC(microk[1], microk[2])
            elif microk[0] == 'Uniform_p':
                return UniformProbability(microk[1], microk[2])
            elif microk[0] == 'Const_p':
                return Constant(microk[1], microk[2])
            else:
                raise Exception('unknown microkernel type')

        knode_dict = {}
        kedge_dict = {}
        p_dict = {}
        for key, microk_list in hyperdict.items():
            if key.startswith('atom_'):
                microk = [get_microk(mk) for mk in microk_list]
                knode_dict.update({key[5:]: np.product(microk)})
            elif key.startswith('bond_'):
                microk = [get_microk(mk) for mk in microk_list]
                kedge_dict.update({key[5:]: np.product(microk)})
            elif key.startswith('probability_'):
                microp = [get_microk(mk) for mk in microk_list]
                p_dict.update({key[12:]: np.product(microp)})

        def fun(type, dict):
            if type == 'Tensorproduct':
                return TensorProduct(**dict)
            elif type == 'Additive':
                return Normalize(Additive(**dict))
            elif type == 'Additive_p':
                return Additive_p(**dict)
            else:
                raise Exception('unknown type:', type)

        return fun(hyperdict['a_type'], knode_dict), \
               fun(hyperdict['b_type'], kedge_dict), \
               fun(hyperdict['p_type'], p_dict)

    def save(self, path, model):
        kernel = model.kernel_ if hasattr(model, 'kernel_') \
            else model.kernel
        if hasattr(kernel, 'kernel_list'):
            for i, hyperdict in enumerate(self.hyperdict):
                theta = kernel.kernel_list[i].theta
                hyperdict.update({'theta': theta.tolist()})
                open(os.path.join(path, 'hyperparameters_%d.json' % i), 'w') \
                    .write(json.dumps(self.hyperdict))
        else:
            theta = kernel.theta
            self.hyperdict[0].update({'theta': theta.tolist()})
            open(os.path.join(path, 'hyperparameters.json'), 'w') \
                .write(json.dumps(self.hyperdict))
