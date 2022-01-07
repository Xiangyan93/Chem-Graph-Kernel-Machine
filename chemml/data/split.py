#!/usr/bin/env python
# -*- coding: utf-8 -*-
from typing import Dict, Iterator, List, Optional, Union, Literal, Tuple
from collections import defaultdict
from random import Random
from typing import Dict, List, Set, Tuple, Union
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from tqdm import tqdm
from .data import Dataset


def generate_scaffold(mol: Union[str, Chem.Mol],
                      include_chirality: bool = False) -> str:
    """ Computes the Bemis-Murcko scaffold for a SMILES string.

    Parameters
    ----------
    mol: A SMILES string or an RDKit molecule.
    include_chirality: bool
        Whether to include chirality in the computed scaffold..

    Returns
    -------
    The Bemis-Murcko scaffold for the molecule.
    """
    mol = Chem.MolFromSmiles(mol) if type(mol) == str else mol
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=include_chirality)
    return scaffold


def scaffold_to_smiles(mols: Union[List[str], List[Chem.Mol]],
                       use_indices: bool = False) -> Dict[str, Union[Set[str], Set[int]]]:
    """ Computes the scaffold for each SMILES and returns a mapping from scaffolds to sets of smiles (or indices).

    Parameters
    ----------
    mols: A list of SMILES strings or RDKit molecules.
    use_indices:
        Whether to map to the SMILES's index in :code:`mols` rather than mapping to the smiles string itself.
        This is necessary if there are duplicate smiles.

    Returns
    -------
    A dictionary mapping each unique scaffold to all SMILES (or indices) which have that scaffold.
    """
    scaffolds = defaultdict(set)
    for i, mol in tqdm(enumerate(mols), total=len(mols)):
        scaffold = generate_scaffold(mol)
        if use_indices:
            scaffolds[scaffold].add(i)
        else:
            scaffolds[scaffold].add(mol)
    return scaffolds


def scaffold_split(dataset: Dataset,
                   sizes: Tuple[float, float] = (0.8, 0.2),
                   balanced: bool = False,
                   seed: int = 0) -> List:
    """ Split a class 'Dataset' into training and test sets.

    Parameters
    ----------
    dataset: class ‘Dataset’
    sizes: [float, float].
        sizes are the percentages of molecules in training and test sets.
    balanced:
        Whether to balance the sizes of scaffolds in each set rather than putting the smallest in test set.
    seed: int
        Random seed for shuffling when doing balanced splitting.
    Returns
    -------
    [Dataset, Dataset]
    """
    assert sum(sizes) == 1

    # Split
    train_size, test_size = sizes[0] * len(dataset), sizes[1] * len(dataset)
    train, test = [], []
    train_scaffold_count, test_scaffold_count = 0, 0

    # Map from scaffold to index in the data
    scaffold_to_indices = scaffold_to_smiles(dataset.mols, use_indices=True)

    # Seed randomness
    random = Random(seed)

    if balanced:  # Put stuff that's bigger than half the val/test size into train, rest just order randomly
        index_sets = list(scaffold_to_indices.values())
        big_index_sets = []
        small_index_sets = []
        for index_set in index_sets:
            if len(index_set) > test_size / 2:
                big_index_sets.append(index_set)
            else:
                small_index_sets.append(index_set)
        random.seed(seed)
        random.shuffle(big_index_sets)
        random.shuffle(small_index_sets)
        index_sets = big_index_sets + small_index_sets
    else:  # Sort from largest to smallest scaffold sets
        index_sets = sorted(list(scaffold_to_indices.values()),
                            key=lambda index_set: len(index_set),
                            reverse=True)

    for index_set in index_sets:
        if len(train) + len(index_set) <= train_size:
            train += index_set
            train_scaffold_count += 1
        else:
            test += index_set
            test_scaffold_count += 1

    # Map from indices to data
    train = [dataset[i] for i in train]
    test = [dataset[i] for i in test]

    dataset_train, dataset_test = dataset.copy(), dataset.copy()
    dataset_train.data = train
    dataset_test.data = test

    return [dataset_train, dataset_test]


def dataset_split(dataset, split_type: Literal['random', 'scaffold_balanced', 'n_heavy'] = 'random',
                  sizes: Tuple[float, float] = (0.8, 0.2), n_heavy: int = 15,
                  seed: int = 0) -> List:
    """ Split the data set into two data sets: training set and test set.

    Parameters
    ----------
    split_type: The algorithm used for data splitting.
    sizes: [float, float].
        If split_type == 'random' or 'scaffold_balanced'.
        sizes are the percentages of molecules in training and test sets.
    n_heavy: int
        If split_type == 'n_heavy'.
        training set contains molecules with heavy atoms < n_heavy.
        test set contains molecules with heavy atoms >= n_heavy.
    seed

    Returns
    -------
    [Dataset, Dataset]
    """
    random = Random(seed)
    data = []
    if split_type == 'random':
        assert sum(sizes) == 1
        indices = list(range(len(dataset.data)))
        random.shuffle(indices)
        end = 0
        for size in sizes:
            start = end
            end = start + int(size * len(dataset.data))
            dataset_ = dataset.copy()
            dataset_.data = [dataset.data[i] for i in indices[start:end]]
            data.append(dataset_)
        return data
    elif split_type == 'scaffold_balanced':
        return scaffold_split(dataset, sizes=sizes, balanced=True, seed=seed)
    elif split_type == 'n_heavy':
        train = []
        test = []
        for d in dataset.data:
            if d.data.n_heavy < n_heavy:
                train.append(d)
            else:
                test.append(d)
        dataset_train, dataset_test = dataset.copy(), dataset.copy()
        dataset_train.data = train
        dataset_test.data = test
        return [dataset_train, dataset_test]
    else:
        raise RuntimeError(f'Unsupported split_type {split_type}')
