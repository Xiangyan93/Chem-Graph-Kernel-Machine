import os
import pickle
import threading
from collections import OrderedDict
from random import Random
from typing import Dict, Iterator, List, Optional, Union, Literal, Tuple

import numpy as np
import pandas as pd
import rdkit.Chem.AllChem as Chem
from rxntools.reaction import ChemicalReaction
from joblib import Parallel, delayed
from sklearn.utils.fixes import _joblib_parallel_args
from sklearn.preprocessing import StandardScaler
import networkx as nx
from graphdot.graph._from_networkx import _from_networkx

# from .scaler import StandardScaler
from chemml.features import get_features_generator, FeaturesGenerator
from chemml.graph.hashgraph import HashGraph
from chemml.args import CommonArgs, KernelArgs, TrainArgs
from .scaffold import scaffold_split


def remove_none(X: List):
    X_ = []
    for x in X:
        if x is not None:
            X_.append(x)
    return X_


def concatenate(X: List, axis: int = 0):
    X_ = remove_none(X)
    if X_:
        return np.concatenate(X_, axis=axis)
    else:
        return None


class SingleMolDatapoint:
    """
    SingleMolDatapoint: Object of single molecule.
    """

    def __init__(self, input: str, molfeatures: np.ndarray = None):
        self.input = input
        self.molfeatures = molfeatures
        self.set_mol()
        self.graph = HashGraph.from_rdkit(self.mol, self.input)

    def set_molfeatures(self, features_generator: List[str]):
        if features_generator is None:
            return
        self.molfeatures = []
        for fg in features_generator:
            features_generator_ = get_features_generator(fg)
            self.molfeatures.append(
                self.calc_molfeatures(self.mol, features_generator_))
        # print(self.features[0])
        self.molfeatures = np.concatenate(self.molfeatures)
        # Fix nans in features
        replace_token = 0
        if self.molfeatures is not None:
            self.molfeatures = np.where(
                np.isnan(self.molfeatures), replace_token, self.molfeatures)

    def calc_molfeatures(self, mol: Chem.Mol,
                         features_generator: FeaturesGenerator):
        if mol is not None and mol.GetNumHeavyAtoms() > 0:
            molfeatures = features_generator(mol)
        # for H2
        elif mol is not None and mol.GetNumHeavyAtoms() == 0:
            # not all features are equally long, so use methane as dummy
            # molecule to determine length
            molfeatures = np.zeros(
                len(features_generator(Chem.MolFromSmiles('C'))))
        else:
            molfeatures = None
        return np.asarray(molfeatures)

    def set_mol(self):
        if self.input.startswith('InChI'):
            self.mol = Chem.MolFromInchi(self.input)
        else:
            self.mol = Chem.MolFromSmiles(self.input)


class SingleReactionDatapoint:
    """
    SingleReactionDatapoint： Object of single chemical reaction.
    """

    def __init__(self, reaction_smarts: str,
                 reaction_type: Literal['reaction', 'agent', 'reaction+agent']
                 = 'reaction'):
        # get chemical reaction object
        self.reaction_smarts = reaction_smarts
        self.chemical_reaction = ChemicalReaction(reaction_smarts)
        self.rxn = self.chemical_reaction.rxn
        # get graph
        self.reaction_type = reaction_type
        if self.reaction_type in ['reaction', 'reaction+agent']:
            self.graph = HashGraph.from_cr(
                self.chemical_reaction, self.reaction_smarts)
        if self.reaction_type in ['agent', 'reaction+agent']:
            self.graph_agent = HashGraph.agent_from_cr(
                self.chemical_reaction, self.reaction_smarts)

    @property
    def X_single_graph(self) -> np.ndarray:
        if self.reaction_type == 'reaction':
            return np.asarray([self.graph])
        elif self.reaction_type == 'agent':
            return np.asarray([self.graph_agent])
        else:
            return np.asarray([self.graph, self.graph_agent])

    @property
    def X_multi_graph(self) -> Optional[np.ndarray]:
        return None

    @property
    def X_molfeatures(self) -> Optional[np.ndarray]:
        return None


class MultiMolDatapoint:
    def __init__(self, data: List[SingleMolDatapoint],
                 concentration: List[float] = None,
                 graph_type: Literal[
                     'single_graph', 'multi_graph'] = 'single_graph'):
        # read data point
        self.data = data
        # features set None
        self.molfeatures = None
        # set concentration
        if concentration is None:
            self.concentration = [1.0] * len(data)
        else:
            self.concentration = concentration
        graphs = [d.graph for d in self.data]
        map(lambda x, y: x.update_concentration(y), graphs, self.concentration)
        self.graph_type = graph_type
        if graph_type == 'single_graph':
            # combine several graphs into a disconnected graph
            self.graph = nx.disjoint_union_all(
                [g.to_networkx() for g in graphs])
            self.graph = _from_networkx(HashGraph, self.graph)
        else:
            self.graph = [
                rv for r in zip(graphs, self.concentration) for rv in r]

    @property
    def mol(self) -> Chem.Mol:
        assert self.graph_type == 'single_graph'
        assert len(self.data) == 1
        return self.data[0].mol

    @property
    def X_single_graph(self) -> Optional[np.ndarray]:
        if self.graph_type == 'single_graph':
            return np.asarray([self.graph])
        else:
            return None

    @property
    def X_multi_graph(self) -> Optional[np.ndarray]:
        if self.graph_type == 'multi_graph':
            return np.asarray([self.graph])
        else:
            return None

    @property
    def X_molfeatures(self) -> np.ndarray:
        return None if self.molfeatures is None else self.molfeatures

    def set_concentration(self, concentration: List[float] = None) -> None:
        if concentration is None:
            return
        else:
            self.concentration = concentration

    def set_molfeatures(self, features_generator: List[str] = None) \
            -> Optional[np.ndarray]:
        if features_generator == None:
            self.molfeatures = None
            return None

        self.molfeatures = []
        for i, d in enumerate(self.data):
            d.set_molfeatures(features_generator)
            self.molfeatures.append(d.molfeatures * self.concentration[i])
        self.molfeatures = np.asarray(self.molfeatures).mean(axis=0)
        return self.molfeatures

    @classmethod
    def from_smiles_or_inchi(cls, input: str):
        return cls([SingleMolDatapoint(input)])

    @classmethod
    def from_smiles_or_inchi_list(
            cls, input: List[str],
            concentration: List[float] = None,
            type: Literal['single_graph', 'multi_graph'] = 'single_graph'):
        return cls([SingleMolDatapoint(s) for s in input], concentration, type)


class CompositeDatapoint:
    def __init__(self, data_p: List[MultiMolDatapoint],
                 data_m: List[MultiMolDatapoint],
                 data_cr: List[SingleReactionDatapoint]):
        self.data_p = data_p
        self.data_m = data_m
        self.data_cr = data_cr

    def set_molfeatures(self, features_generator):
        for d in self.data_p:
            d.set_molfeatures(features_generator)

    @property
    def mol(self) -> Chem.Mol:
        assert len(self.data_p) == 1
        assert len(self.data_m) == 0
        assert len(self.data_cr) == 0
        return self.data_p[0].mol

    @property
    def X(self) -> np.ndarray:
        return concatenate([self.X_single_graph, self.X_multi_graph,
                            self.X_molfeatures])

    @property
    def X_single_graph(self) -> np.ndarray:
        return concatenate([d.X_single_graph for d in
                            self.data_p + self.data_m + self.data_cr])

    @property
    def X_multi_graph(self) -> np.ndarray:
        return concatenate([d.X_multi_graph for d in
                            self.data_p + self.data_m + self.data_cr])

    @property
    def X_molfeatures(self) -> np.ndarray:
        return concatenate([d.X_molfeatures for d in
                            self.data_p + self.data_m + self.data_cr])


class SubDataset:
    def __init__(self, data: CompositeDatapoint,
                 targets: np.ndarray,
                 addfeatures: Optional[np.ndarray] = None,
                 gid: int = None):
        self.data = data
        # set targets
        if targets.ndim == 1:
            targets = targets.reshape(1, -1)
        self.targets = targets
        # set features
        if addfeatures is not None and addfeatures.ndim == 1:
            addfeatures = addfeatures.reshape(1, -1)
        self.addfeatures = addfeatures
        # set group id
        self.gid = gid

    def expand_addfeatures(self, X, concat=False):
        if X is None:
            return None
        if self.addfeatures is None:
            return X
        else:
            if concat:
                return np.c_[X.repeat(len(self.addfeatures), axis=0),
                             self.addfeatures]
            else:
                return X.repeat(len(self.addfeatures), axis=0)

    @property
    def mol(self) -> Chem.Mol:
        return self.data.mol

    @property
    def X(self):
        return self.expand_addfeatures(self._X_mol, concat=True)

    @property
    def _X_mol(self):
        return self.data.X.reshape(1, -1)

    @property
    def X_gid(self):
        return self.expand_addfeatures(self._X_gid)

    @property
    def _X_gid(self):
        return np.asarray([[self.gid]])

    @property
    def X_graph(self):
        return self.expand_addfeatures(self._X_graph)

    @property
    def _X_graph(self):
        return np.concatenate(remove_none([self._X_single_graph,
                                           self._X_multi_graph]), axis=1)

    @property
    def X_single_graph(self) -> np.ndarray:
        return self.expand_addfeatures(self._X_single_graph)

    @property
    def _X_single_graph(self) -> np.ndarray:
        return None if self.data.X_single_graph is None \
            else self.data.X_single_graph.reshape(1, -1)

    @property
    def X_multi_graph(self) -> np.ndarray:
        return self.expand_addfeatures(self._X_multi_graph)

    @property
    def _X_multi_graph(self) -> Optional[np.ndarray]:
        return None if self.data.X_multi_graph is None \
            else self.data.X_multi_graph.reshape(1, -1)

    @property
    def X_molfeatures(self) -> np.ndarray:
        return self.expand_addfeatures(self._X_molfeatures)

    @property
    def _X_molfeatures(self) -> np.ndarray:
        return None if self.data.X_molfeatures is None \
            else self.data.X_molfeatures.reshape(1, -1)

    @property
    def y(self):
        self.targets = np.asarray(self.targets, dtype=float)
        return self.targets

    def set_features(self, features_generator: List[str] = None):
        self.data.set_molfeatures(features_generator)


class Dataset:
    def __init__(self, data: List[SubDataset] = None,
                 molfeatures_scaler: StandardScaler = None,
                 addfeatures_scaler: StandardScaler = None,
                 kernel_type: Literal['graph', 'preCalc'] = 'graph'):
        self.data = data
        self.unify_datatype()
        self.molfeatures_scaler = molfeatures_scaler
        self.addfeatures_scaler = addfeatures_scaler
        self.kernel_type = kernel_type
        self.ignore_addfeatures = False

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, item) -> Union[SubDataset, List[SubDataset]]:
        return self.data[item]

    def split(self, split_type: str = 'random',
              sizes: Tuple[float, float] = (0.8, 0.2),
              seed: int = 0):
        random = Random(seed)
        if split_type == 'random':
            indices = list(range(len(self.data)))
            random.shuffle(indices)
            train_size = int(sizes[0] * len(self.data))
            train = [self.data[i] for i in indices[:train_size]]
            test = [self.data[i] for i in indices[train_size:]]
        elif split_type == 'scaffold_balanced':
            train, test = scaffold_split(
                self, sizes=sizes, balanced=True, seed=seed)
        else:
            raise RuntimeError(f'Unsupported split_type {split_type}')
        return Dataset(train, self.molfeatures_scaler,
                       self.addfeatures_scaler, self.kernel_type), \
               Dataset(test, self.molfeatures_scaler,
                       self.addfeatures_scaler, self.kernel_type)

    def update_args(self, args: KernelArgs):
        if args.feature_columns is None:
            self.ignore_addfeatures = True
        self.kernel_type = args.kernel_type
        self.molfeatures_normalize = args.molfeatures_normalize
        self.addfeatures_normalize = args.addfeatures_normalize

    def normalize_features(self):
        if self.X_raw_molfeatures is not None and self.molfeatures_normalize:
            self.molfeatures_scaler = StandardScaler().fit(
                self.X_raw_molfeatures)
        else:
            self.molfeatures_scaler = None
        if self.X_raw_addfeatures is not None and self.addfeatures_normalize:
            self.addfeatures_scaler = StandardScaler().fit(
                self.X_raw_addfeatures)
        else:
            self.addfeatures_scaler = None

    def unify_datatype(self):
        if self.data is None:
            return
        X = np.concatenate([d._X_graph for d in self.data])
        for i in range(X.shape[1]):
            self._unify_datatype(X[:, i])

    @staticmethod
    def _unify_datatype(X):
        if X[0].__class__ == list:
            graphs = []
            for x in X:
                graphs += x[::2]
            HashGraph.unify_datatype(graphs, inplace=True)
        else:
            HashGraph.unify_datatype(X, inplace=True)

    @property
    def mols(self) -> List[Chem.Mol]:
        return [d.mol for d in self.data]

    @property
    def X_raw_molfeatures(self) -> Optional[np.ndarray]:
        assert self.kernel_type == 'graph'
        if self.ignore_addfeatures:
            return concatenate([d._X_molfeatures for d in self.data])
        else:
            return concatenate([d.X_molfeatures for d in self.data])

    @property
    def X_molfeatures(self) -> np.ndarray:
        molfeatures = self.X_raw_molfeatures
        if self.molfeatures_scaler is not None:
            molfeatures = self.molfeatures_scaler.transform(molfeatures)
        return molfeatures

    @property
    def X_raw_addfeatures(self) -> Optional[np.ndarray]:
        if self.ignore_addfeatures:
            return None
        else:
            return concatenate([d.addfeatures for d in self.data])

    @property
    def X_addfeatures(self) -> np.ndarray:
        addfeatures = self.X_raw_addfeatures
        if self.addfeatures_scaler is not None:
            addfeatures = self.addfeatures_scaler.transform(addfeatures)
        return addfeatures

    @property
    def X_features(self) -> np.ndarray:
        return concatenate([self.X_molfeatures, self.X_addfeatures])

    @property
    def X_graph(self) -> np.ndarray:
        if self.ignore_addfeatures:
            return np.concatenate([d._X_graph for d in self.data])
        else:
            return np.concatenate([d.X_graph for d in self.data])

    # This is used for graph kernel
    @property
    def X(self) -> np.ndarray:
        if self.kernel_type == 'graph':
            return concatenate([self.X_graph, self.X_features], axis=1)
        else:
            return concatenate([self.X_gid, self.X_addfeatures], axis=1)

    @property
    def X_gid(self) -> np.ndarray:
        if self.ignore_addfeatures:
            return np.concatenate([d._X_gid for d in self.data])
        else:
            return np.concatenate([d.X_gid for d in self.data])

    @property
    def y(self):
        y = [d.y for d in self.data]
        return np.concatenate(y, dtype=float)

    def save(self, path, filename='dataset.pkl', overwrite=False):
        f_dataset = os.path.join(path, filename)
        if os.path.isfile(f_dataset) and not overwrite:
            raise RuntimeError(
                f'Path {f_dataset} already exists. To overwrite, set '
                '`overwrite=True`.'
            )
        store = self.__dict__.copy()
        pickle.dump(store, open(f_dataset, 'wb'), protocol=4)

    @classmethod
    def load(cls, path, filename='dataset.pkl'):
        f_dataset = os.path.join(path, filename)
        store = pickle.load(open(f_dataset, 'rb'))
        dataset = cls()
        dataset.__dict__.update(**store)
        return dataset

    @staticmethod
    def get_subDataset(
            pure: List[str],
            mixture: List[Optional[str]],
            mixture_type: Literal['single_graph', 'multi_graph'],
            d_reaction_graph: List[str],
            reaction_type: Literal['reaction', 'agent', 'reaction+agent'],
            targets: np.ndarray,
            features: Optional[np.ndarray] = None,
            features_generator: List[str] = None,
            gid: int = None,
    ) -> SubDataset:
        data_p = []
        data_m = []
        data_r = []
        pure = [] if pure is None else list(pure)
        for s_or_i in pure:
            data_p.append(MultiMolDatapoint.from_smiles_or_inchi(s_or_i))
        for m in mixture:
            data_m.append(MultiMolDatapoint.from_smiles_or_inchi_list(
                m[0::2], concentration=m[1::2], type=mixture_type))
        for rg in d_reaction_graph:
            data_r.append(SingleReactionDatapoint(rg, reaction_type))
        data = SubDataset(CompositeDatapoint(data_p, data_m, data_r),
                          targets, features, gid)
        data.set_features(features_generator)
        return data

    @classmethod
    def from_csv(cls, args: CommonArgs):
        df = pd.read_csv(args.data_path)
        if args.unique_reading:
            n1 = len(args.pure_columns)
            n2 = len(args.mixture_columns)
            n3 = len(args.reaction_columns)
            groups = df.groupby(args.pure_columns + args.mixture_columns
                                + args.reaction_columns)
            data = Parallel(
                n_jobs=args.n_jobs, verbose=True,
                **_joblib_parallel_args(prefer='processes'))(
                delayed(cls.get_subDataset)(
                    list(g[0][:n1]),
                    list(g[0][n1:n2]),
                    args.mixture_type,
                    list(g[0][n2:n3]),
                    args.reaction_type,
                    to_numpy(g[1][args.feature_columns]),
                    to_numpy(g[1][args.target_columns]),
                    i
                )
                for i, g in groups)
        else:
            data = Parallel(
                n_jobs=args.n_jobs, verbose=True,
                **_joblib_parallel_args(prefer='processes'))(
                delayed(cls.get_subDataset)(
                    tolist(df.iloc[i].get(args.pure_columns)),
                    tolist(df.iloc[i].get(args.mixture_columns)),
                    args.mixture_type,
                    tolist(df.iloc[i].get(args.reaction_columns)),
                    args.reaction_type,
                    to_numpy(df.iloc[i][args.target_columns]),
                    to_numpy(df.iloc[i].get(args.feature_columns)),
                    args.features_generator,
                    i
                )
                for i in df.index)
        return cls(data)


def tolist(l: pd.Series) -> List[str]:
    if l is None:
        return []
    else:
        return list(l)


def to_numpy(l: pd.Series) -> Optional[np.ndarray]:
    if l is None:
        return None
    else:
        return l.to_numpy()
