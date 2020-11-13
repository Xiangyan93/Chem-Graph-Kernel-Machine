import os
import json
from tqdm import tqdm
tqdm.pandas()
from graphdot.kernel.marginalized import MarginalizedGraphKernel
from graphdot.util.pretty_tuple import pretty_tuple
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


class NormalizationMolSize(Normalization):
    def __init__(self, kernel, s=100.0, s_bounds=(1e2, 1e3)):
        super().__init__(kernel)
        self.s = s
        self.s_bounds = s_bounds

    def __diag(self, K, l, r, K_gradient=None):
        l_lr = np.repeat(l, len(r)).reshape(len(l), len(r))
        r_rl = np.repeat(r, len(l)).reshape(len(r), len(l))
        se = np.exp(-((l_lr - r_rl.T) / self.s) ** 2)
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
                print(K.max(), K.min())
                print(dK_s.max(), dK_s.min())
                print(self.theta)
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
              self.s)

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
            return np.r_[self.kernel.theta, self.s]

    @theta.setter
    def theta(self, value):
        if self.s_bounds == "fixed":
            self.kernel.theta = value
        else:
            self.kernel.theta = value[:-1]
            self.s = value[-1]

    @property
    def bounds(self):
        if self.s_bounds == "fixed":
            return self.kernel.bounds
        else:
            return np.r_[self.kernel.bounds, np.reshape(self.s_bounds, (1, 2))]

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
        if hyperdict['normalization'] == True:
            kernel = Normalization(kernel)
        elif hyperdict['normalization'][0]:
            kernel = NormalizationMolSize(
                kernel, s=hyperdict['normalization'][1],
                s_bounds=hyperdict['normalization'][2])
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
        if hyperdict['normalization'] == True:
            kernel = Normalization(kernel)
        elif hyperdict['normalization'][0]:
            kernel = NormalizationMolSize(
                kernel, s=hyperdict['normalization'][1],
                s_bounds=hyperdict['normalization'][2])
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
                .write(json.dumps(self.hyperdict[0]))
