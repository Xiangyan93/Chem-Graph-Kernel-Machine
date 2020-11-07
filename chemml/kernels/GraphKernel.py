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

    def diag(self, X, eval_gradient=False, n_process=mp.cpu_count(),
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

'''
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
'''

"""
class ConvolutionNormalizedGraphKernel(MGK):
    def __call__(self, X, Y=None, eval_gradient=False, *args, **kwargs):
        X = self._format_X(X)
        Y = self._format_X(Y)
        X_ = self.get_uniX(X, Y)

        if Y is None:
            Xidx, Yidx = np.triu_indices(len(X))
            Xidx, Yidx = Xidx.astype(np.uint32), Yidx.astype(np.uint32)
            Y = X
            symmetric = True
        else:
            Xidx, Yidx = np.indices((len(X), len(Y)), dtype=np.uint32)
            Xidx = Xidx.ravel()
            Yidx = Yidx.ravel()
            symmetric = False

        K = np.zeros((len(X), len(Y)))
        if eval_gradient:
            K_gradient = np.zeros((len(X), len(Y), self.theta.shape[0]))
            K_, K_gradient_ = super().__call__(X_, eval_gradient=True)
            df = pd.DataFrame({'Xidx': Xidx, 'Yidx': Yidx})
            K[Xidx, Yidx], K_gradient[Xidx, Yidx] = df.progress_apply(
                lambda x: self._kxy(X[x['Xidx']], Y[x['Yidx']], True,
                                    X_, K_, K_gradient_),
                axis=1
            )
        else:
            K_ = super().__call__(X_)
            df = pd.DataFrame({'Xidx': Xidx, 'Yidx': Yidx})
            n_process = 8
            df_parts = np.array_split(df, n_process)
            def apply_f(x, y, A, B):
                k = []
                for i in range(len(x)):
                    xidx = x[i]
                    yidx = y[i]
                    k.append(self.Kc(X[xidx], Y[yidx], X=A, K=B))
                return np.asarray(k)
            with Pool(processes=n_process) as pool:
                result_parts = pool.map(apply_f, [(df_part['Xidx'], df_part['Yidx'], np.copy(X_), np.copy(K_)) for df_part in df_parts])
            K[Xidx, Yidx] = pd.concat(result_parts)

            '''
            for i in range(len(Xidx)):
                if i % 100 == 0:
                    print(i, len(Xidx))
                xidx = Xidx[i]
                yidx = Yidx[i]
                K[xidx][yidx] = self.Kc(X[xidx], Y[yidx], X=X_, K=K_)


            df = pd.DataFrame({'Xidx': Xidx, 'Yidx': Yidx, 'X_': [X_] * len(Xidx), 'K_': [K_] * len(Xidx)})
            print(df)
            print(cpu_count())
            def apply_f(df):
                return df.progress_apply(
                    lambda x: self._kxy(X[x['Xidx']], Y[x['Yidx']], False, x['X_'], x['K_']),
                    axis=1
                )
            K[Xidx, Yidx] = dd.from_pandas(df, npartitions=4).map_partitions(
                apply_f)# .compute(scheduler='threads')


            df_parts = np.array_split(df, 20)
            def apply_f(df):
                return df.apply(
                    lambda x: self._kxy(X[x['Xidx']], Y[x['Yidx']], False, X_, K_),
                    axis=1
                )

            with Pool(processes=8) as pool:
                result_parts = pool.map(apply_f, df_parts)
            K[Xidx, Yidx] = pd.concat(result_parts)
            K[Xidx, Yidx] = df.progress_apply(
                lambda x: self._kxy(X[x['Xidx']], Y[x['Yidx']], False, X_, K_),
                axis=1
            )
            '''

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

    def diag(self, X, eval_gradient=False, *args, **kwargs):
        X = self._format_X(X)
        X_ = self.get_uniX(X)
        D = np.zeros(len(X))
        if eval_gradient:
            D_gradient = np.zeros((len(X), self.theta.shape[0]))
            K_, K_gradient_ = super().__call__(X_, eval_gradient=True)
            for i, x in enumerate(X):
                D[i], D_gradient[i] = self.Kc(
                    x, x, eval_gradient=True,
                    X=X_, K=K_, K_gradient=K_gradient_)
            return D, D_gradient
        else:
            K_ = super().__call__(X_)
            for i, x in enumerate(X):
                D[i] = self.Kc(x, x, X=X_, K=K_)
            return D

    def _kxy(self, x, y, eval_gradient=False, X=None, K=None, K_gradient=None):
        def get_reaction_smarts(g, g_weight):
            reactants = []
            products = []
            for i, weight in enumerate(g_weight):
                if weight > 0:
                    reactants.append(g.smiles)
                elif weight < 0:
                    products.append(g.smiles)
            return '.'.join(reactants) + '>>' '.'.join(products)

        x, x_weight = self.x2graph(x), self.x2weight(x)
        y, y_weight = self.x2graph(y), self.x2weight(y)
        if not x and not y:
            k = 1.0
            k_gradient = np.zeros(len(self.theta))
        elif not x or not y:
            k = 0.0
            k_gradient = np.zeros(len(self.theta))
        else:
            if eval_gradient:
                x_idx = np.searchsorted(X, x)
                y_idx = np.searchsorted(X, y)
                Kxy, dKxy = K[x_idx][:, y_idx], \
                            K_gradient[x_idx][:, y_idx][:]
                Kxx, dKxx = K[x_idx][:, x_idx], \
                            K_gradient[x_idx][:, x_idx][:]
                Kyy, dKyy = K[y_idx][:, y_idx], \
                            K_gradient[y_idx][:, y_idx][:]
                Fxy = x_weight.dot(Kxy).dot(y_weight)
                Fxx = x_weight.dot(Kxx).dot(x_weight)
                Fyy = y_weight.dot(Kyy).dot(y_weight)
                dFxy = np.einsum("i,j,ijk->k", x_weight, y_weight, dKxy)
                dFxx = np.einsum("i,j,ijk->k", x_weight, x_weight, dKxx)
                dFyy = np.einsum("i,j,ijk->k", y_weight, y_weight, dKyy)
                '''
                if Fxx <= 0.:
                    raise Exception('trivial reaction: ',
                                    get_reaction_smarts(x, x_weight))
                if Fyy == 0.:
                    raise Exception('trivial reaction: ',
                                    get_reaction_smarts(y, y_weight))
                '''
                sqrtFxxFyy = np.sqrt(Fxx * Fyy)
                k = Fxy / sqrtFxxFyy
                k_gradient = (dFxy-0.5*dFxx/Fxx-0.5*dFyy/Fyy)/sqrtFxxFyy
            else:
                x_idx = np.searchsorted(X, x)
                y_idx = np.searchsorted(X, y)
                Kxy = K[x_idx][:, y_idx]
                Kxx = K[x_idx][:, x_idx]
                Kyy = K[y_idx][:, y_idx]
                Fxy = x_weight.dot(Kxy).dot(y_weight)
                Fxx = x_weight.dot(Kxx).dot(x_weight)
                Fyy = y_weight.dot(Kyy).dot(y_weight)
                if Fxx <= 0.:
                    raise Exception('trivial reaction: ',
                                    get_reaction_smarts(x, x_weight))
                if Fyy == 0.:
                    raise Exception('trivial reaction: ',
                                    get_reaction_smarts(y, y_weight))
                sqrtFxxFyy = np.sqrt(Fxx * Fyy)
                k = Fxy / sqrtFxxFyy

        if eval_gradient:
            return k, k_gradient
        else:
            return k

    def Kc(self, x, y, eval_gradient=False, X=None, K=None, K_gradient=None):
        def get_reaction_smarts(g, g_weight):
            reactants = []
            products = []
            for i, weight in enumerate(g_weight):
                if weight > 0:
                    reactants.append(g.smiles)
                elif weight < 0:
                    products.append(g.smiles)
            return '.'.join(reactants) + '>>' '.'.join(products)

        x, x_weight = self.x2graph(x), self.x2weight(x)
        y, y_weight = self.x2graph(y), self.x2weight(y)
        if not x and not y:
            k = 1.0
            k_gradient = np.zeros(len(self.theta))
        elif not x or not y:
            k = 0.0
            k_gradient = np.zeros(len(self.theta))
        else:
            if eval_gradient:
                if K is not None and K_gradient is not None:
                    x_idx = np.searchsorted(X, x)
                    y_idx = np.searchsorted(X, y)
                    Kxy, dKxy = K[x_idx][:, y_idx], \
                                K_gradient[x_idx][:, y_idx][:]
                    Kxx, dKxx = K[x_idx][:, x_idx], \
                                K_gradient[x_idx][:, x_idx][:]
                    Kyy, dKyy = K[y_idx][:, y_idx], \
                                K_gradient[y_idx][:, y_idx][:]
                else:
                    Kxy, dKxy = super().__call__(x, y, eval_gradient=True)
                    Kxx, dKxx = super().__call__(x, x, eval_gradient=True)
                    Kyy, dKyy = super().__call__(y, y, eval_gradient=True)
                Fxy = np.einsum("i,j,ij", x_weight, y_weight, Kxy)
                dFxy = np.einsum("i,j,ijk->k", x_weight, y_weight, dKxy)
                Fxx = np.einsum("i,j,ij", x_weight, x_weight, Kxx)
                dFxx = np.einsum("i,j,ijk->k", x_weight, x_weight, dKxx)
                Fyy = np.einsum("i,j,ij", y_weight, y_weight, Kyy)
                dFyy = np.einsum("i,j,ijk->k", y_weight, y_weight, dKyy)
                if Fxx <= 0.:
                    raise Exception('trivial reaction: ',
                                    get_reaction_smarts(x, x_weight))
                if Fyy == 0.:
                    raise Exception('trivial reaction: ',
                                    get_reaction_smarts(y, y_weight))
                sqrtFxxFyy = np.sqrt(Fxx * Fyy)
                k = Fxy / sqrtFxxFyy
                k_gradient = (dFxy-0.5*dFxx/Fxx-0.5*dFyy/Fyy)/sqrtFxxFyy
            else:
                if K is not None:
                    x_idx = np.searchsorted(X, x)
                    y_idx = np.searchsorted(X, y)
                    Kxy = K[x_idx][:, y_idx]
                    Kxx = K[x_idx][:, x_idx]
                    Kyy = K[y_idx][:, y_idx]
                else:
                    Kxy = super().__call__(x, y)
                    Kxx = super().__call__(x, x)
                    Kyy = super().__call__(y, y)
                # print(x_weight.shape, y_weight.shape, Kxy.shape)
                Fxy = np.einsum("i,j,ij", x_weight, y_weight, Kxy)
                Fxx = np.einsum("i,j,ij", x_weight, x_weight, Kxx)
                Fyy = np.einsum("i,j,ij", y_weight, y_weight, Kyy)
                Fxy = x_weight.dot(Kxy).dot(y_weight)
                Fxx = x_weight.dot(Kxx).dot(x_weight)
                Fyy = y_weight.dot(Kyy).dot(y_weight)
                if Fxx <= 0.:
                    raise Exception('trivial reaction: ',
                                    get_reaction_smarts(x, x_weight))
                if Fyy == 0.:
                    raise Exception('trivial reaction: ',
                                    get_reaction_smarts(y, y_weight))
                sqrtFxxFyy = np.sqrt(Fxx * Fyy)
                k = Fxy / sqrtFxxFyy

        if eval_gradient:
            return k, k_gradient
        else:
            return k

    @staticmethod
    def x2graph(x):
        return x[::2]

    @staticmethod
    def x2weight(x):
        return np.asarray(x[1::2])

    @staticmethod
    def get_uniX(X, Y=None):
        graphs = []
        for x in X:
            graphs += ConvolutionNormalizedGraphKernel.x2graph(x)
        if Y is not None:
            for y in Y:
                graphs += ConvolutionNormalizedGraphKernel.x2graph(y)
        return np.sort(np.unique(graphs))

    '''

    def PreCalculate(self, X, result_dir, id=None):
        X = self._format_X(X)
        X = self.get_uniX(X)
        self.graphs = np.sort(X)
        self.K, self.K_gradient = super().__call__(self.graphs,
                                                   eval_gradient=True)
        self.save(result_dir, id=id)
    '''
"""


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
