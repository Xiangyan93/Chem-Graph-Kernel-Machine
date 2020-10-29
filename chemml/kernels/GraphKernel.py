import os
import json
import pickle
from graphdot.kernel.marginalized import MarginalizedGraphKernel
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
from chemml.kernels.PreCalcKernel import (
    ConvolutionPreCalcKernel as CPCK,
    _Kc,
)
from chemml.kernels.MultipleKernel import _get_uniX
from chemml.kernels.KernelConfig import KernelConfig
from chemml.kernels.MultipleKernel import *
from chemml.graph.hashgraph import HashGraph


class MGK(MarginalizedGraphKernel):
    """
    X and Y could be 2-d numpy array.
    make it compatible with sklearn.
    remove repeated kernel calculations, if set unique=True.
    """

    def __init__(self, unique=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.unique = unique

    def __unique(self, X):
        X_unique = np.sort(np.unique(X))
        X_idx = np.searchsorted(X_unique, X)
        return X_unique, X_idx

    def _graph(self, X):
        if X.__class__ == np.ndarray:
            return X.ravel()
        else:
            return X

    def __call__(self, X, Y=None, eval_gradient=False, *args, **kwargs):
        X = self._graph(X)
        Y = self._graph(Y)
        X = HashGraph.unify_datatype(X)
        Y = HashGraph.unify_datatype(Y) if Y is not None else Y
        if self.unique:
            X_unique, X_idx = self.__unique(X)
            if Y is None:
                Y_unique, Y_idx = X_unique, X_idx
            else:
                Y_unique, Y_idx = self.__unique(Y)
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
        X = self._graph(X)
        X = HashGraph.unify_datatype(X)
        if self.unique:
            X_unique, X_idx = self.__unique(X)
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


class NormalizedGraphKernel(MGK):
    def __diag2K(self, K, dx, dy, l, K_gradient=None):
        a = np.repeat(dx ** -2, len(dy)).reshape(len(dx), len(dy))
        b = np.repeat(dy ** -2, len(dx)).reshape(len(dy), len(dx))
        c = np.exp(-((a - b.T) / l) ** 2)
        K = np.einsum("i,ij,j,ij->ij", dx, K, dy, c)
        if K_gradient is None:
            return K
        else:
            return K, np.einsum("ijk,i,j,ij->ijk", K_gradient, dx, dy, c)

    def __normalize(self, X, Y, R, length=50000):
        if Y is None:
            # square matrix
            if type(R) is tuple:
                d = np.diag(R[0]) ** -0.5
                return self.__diag2K(R[0], d, d, length, K_gradient=R[1])
            else:
                d = np.diag(R) ** -0.5
                return self.__diag2K(R, d, d, length)
        else:
            # rectangular matrix, must have X and Y
            # diag_X = (super().diag(X) ** -0.5).flatten()
            # diag_Y = (super().diag(Y) ** -0.5).flatten()
            dx = super().diag(X) ** -0.5
            dy = super().diag(Y) ** -0.5
            if type(R) is tuple:
                return self.__diag2K(R[0], dx, dy, length, K_gradient=R[1])
            else:
                return self.__diag2K(R, dx, dy, length)

    '''
    def __normalize_old(self, X, Y, R):
        if Y is None:
            # square matrix
            if type(R) is tuple:
                d = np.diag(R[0]) ** -0.5
                K = np.diag(d).dot(R[0]).dot(np.diag(d))
                K_gradient = np.einsum("ijk,i,j->ijk", R[1], d, d)
                return K, K_gradient
            else:
                d = np.diag(R) ** -0.5
                K = np.diag(d).dot(R).dot(np.diag(d))
                return K
        else:
            # rectangular matrix, must have X and Y
            if type(R) is tuple:
                diag_X = super().diag(X) ** -0.5
                diag_Y = super().diag(Y) ** -0.5
                K = np.diag(diag_X).dot(R[0]).dot(np.diag(diag_Y))
                K_gradient = np.einsum("ijk,i,j->ijk", R[1], diag_X, diag_Y)
                return K, K_gradient
            else:
                diag_X = super().diag(X) ** -0.5
                diag_Y = super().diag(Y) ** -0.5
                K = np.einsum("ij,i,j->ij", R, diag_X, diag_Y)
                return K
    '''

    def __call__(self, X, Y=None, *args, **kwargs):
        R = super().__call__(X, Y, *args, **kwargs)
        return self.__normalize(X, Y, R)

    def diag(self, X, *args, **kwargs):
        return np.ones(len(X))


def _PreCalculate(self, X, result_dir, id=None):
    self.graphs = X
    self.K, self.K_gradient = self(self.graphs, eval_gradient=True)
    self.save(result_dir, id=id)


def _save(self, result_dir, id=None):
    if id is None:
        f_kernel = os.path.join(result_dir, 'kernel.pkl')
    else:
        f_kernel = os.path.join(result_dir, 'kernel_%i.pkl' % id)
    store_dict = self.__dict__.copy()
    for key in ['node_kernel', 'edge_kernel', 'p', 'q', 'q_bounds',
                'element_dtype', 'backend']:
        store_dict.pop(key, None)
    store_dict['theta'] = self.theta
    pickle.dump(store_dict, open(f_kernel, 'wb'), protocol=4)


def _load(self, result_dir):
    f_kernel = os.path.join(result_dir, 'kernel.pkl')
    store_dict = pickle.load(open(f_kernel, 'rb'))
    self.__dict__.update(**store_dict)


def _get_params(self, super, deep=False):
    params = super.get_params(deep=deep)
    params.update(dict(
        graphs=self.graphs,
        K=self.K,
        K_gradient=self.K_gradient,
    ))
    return params


def _call(self, super, X, Y=None, eval_gradient=False, *args, **kwargs):
    if self.K is None or self.K_gradient is None:
        return super.__call__(X, Y=Y, eval_gradient=eval_gradient, *args,
                              **kwargs)
    else:
        X_idx = np.searchsorted(self.graphs, X).ravel()
        Y_idx = np.searchsorted(self.graphs, Y).ravel() if Y is not None \
            else X_idx
    if eval_gradient:
        return self.K[X_idx][:, Y_idx], self.K_gradient[X_idx][:, Y_idx][:]
    else:
        return self.K[X_idx][:, Y_idx]


class PreCalcMarginalizedGraphKernel(MGK):
    def __init__(self, graphs=None, K=None, K_gradient=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.graphs = graphs
        self.K = K
        self.K_gradient = K_gradient

    def get_uniX(self, X):
        return _get_uniX(X)

    def PreCalculate(self, X, result_dir, id=None):
        _PreCalculate(self, self.get_uniX(X), result_dir, id=id)

    def save(self, result_dir, id=None):
        _save(self, result_dir, id=id)

    def load(self, result_dir):
        _load(self, result_dir)

    def __call__(self, X, Y=None, eval_gradient=False, *args, **kwargs):
        return _call(self, super(), X, Y=Y, eval_gradient=eval_gradient,
                     *args, **kwargs)

    def get_params(self, deep=False):
        return _get_params(self, deep)


class PreCalcNormalizedGraphKernel(NormalizedGraphKernel):
    def __init__(self, graphs=None, K=None, K_gradient=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.graphs = graphs
        self.K = K
        self.K_gradient = K_gradient

    def get_uniX(self, X):
        return _get_uniX(X)

    def PreCalculate(self, X, result_dir, id=None):
        _PreCalculate(self, self.get_uniX(X), result_dir, id=id)

    def save(self, result_dir, id=None):
        _save(self, result_dir, id=id)

    def load(self, result_dir):
        _load(self, result_dir)

    def __call__(self, X, Y=None, eval_gradient=False, *args, **kwargs):
        return _call(self, super(), X, Y=Y, eval_gradient=eval_gradient,
                     *args, **kwargs)

    def get_params(self, deep=False):
        return _get_params(self, super(), deep)


class ConvolutionNormalizedGraphKernel(PreCalcNormalizedGraphKernel):
    def __call__(self, X, Y=None, eval_gradient=False, *args, **kwargs):
        from chemml.kernels.PreCalcKernel import _call
        X = self._graph(X)
        Y = self._graph(Y)
        return _call(self, X, Y=Y, eval_gradient=eval_gradient,
                     *args, **kwargs)

    def Kc(self, x, y, eval_gradient=False):
        return _Kc(CPCK, super(), x, y, eval_gradient=eval_gradient)

    def get_uniX(self, X):
        graphs = []
        for x in X:
            graphs += CPCK.x2graph(x)
        return _get_uniX(graphs)

    def PreCalculate(self, X, result_dir, id=None):
        X = self._graph(X)
        X = self.get_uniX(X)
        self.graphs = np.sort(X)
        self.K, self.K_gradient = super().__call__(self.graphs,
                                                   eval_gradient=True)
        self.save(result_dir, id=id)


class GraphKernelConfig(KernelConfig):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
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
            for i in range(ns, ns+nm):
                kernels += [self.get_conv_graph_kernel(self.hyperdict[i])]
            kernels += self.get_rbf_kernel()
            composition = [(i,) for i in range(ns+nm)] + \
                [tuple(np.arange(ns+nm, na+ns+nm))]
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
        params = self.params
        self.type = 'graph'
        if params['NORMALIZED']:
            KernelObject = PreCalcNormalizedGraphKernel
        else:
            KernelObject = PreCalcMarginalizedGraphKernel
        knode, kedge, p = self.get_knode_kedge_p(hyperdict)
        return KernelObject(
            node_kernel=knode,
            edge_kernel=kedge,
            q=hyperdict['q'][0],
            q_bounds=hyperdict['q'][1],
            p=p,
            unique=self.add_features is not None
        )

    def get_conv_graph_kernel(self, hyperdict):  # dont delete kernel_pkl
        params = self.params
        self.type = 'graph'
        if params['NORMALIZED']:
            KernelObject = ConvolutionNormalizedGraphKernel
        else:
            raise Exception('not supported option')
        knode, kedge, p = self.get_knode_kedge_p(hyperdict)
        return KernelObject(
            node_kernel=knode,
            edge_kernel=kedge,
            q=hyperdict['q'][0],
            q_bounds=hyperdict['q'][1],
            p=p,
            unique=self.add_features is not None
        )

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
                open(os.path.join(path, 'hyperparameters_%d.json' % i), 'w')\
                    .write(json.dumps(self.hyperdict))
        else:
            theta = kernel.theta
            self.hyperdict[0].update({'theta': theta.tolist()})
            open(os.path.join(path, 'hyperparameters.json'), 'w') \
                .write(json.dumps(self.hyperdict))
