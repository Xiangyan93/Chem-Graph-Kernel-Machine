#!/usr/bin/env python
# -*- coding: utf-8 -*-
import csv
import os
import pickle
from typing import List

import numpy as np
import pandas as pd
from rdkit.Chem import PandasTools


def save_features(path: str, features: List[np.ndarray]) -> None:
    """
    Saves features_mol to a compressed :code:`.npz` file with array name "features_mol".

    :param path: Path to a :code:`.npz` file where the features_mol will be saved.
    :param features: A list of 1D numpy arrays containing the features_mol for molecules.
    """
    np.savez_compressed(path, features=features)


def load_features(path: str) -> np.ndarray:
    """
    Loads features_mol saved in a variety of formats.

    Supported formats:

    * :code:`.npz` compressed (assumes features_mol are saved with name "features_mol")
    * .npy
    * :code:`.csv` / :code:`.txt` (assumes comma-separated features_mol with a header and with one line per molecule)
    * :code:`.pkl` / :code:`.pckl` / :code:`.pickle` containing a sparse numpy array

    .. note::

       All formats assume that the SMILES loaded elsewhere in the code are in the same
       order as the features_mol loaded here.

    :param path: Path to a file containing features_mol.
    :return: A 2D numpy array of size :code:`(num_molecules, features_size)` containing the features_mol.
    """
    extension = os.path.splitext(path)[1]

    if extension == '.npz':
        features = np.load(path)['features_mol']
    elif extension == '.npy':
        features = np.load(path)
    elif extension in ['.csv', '.txt']:
        with open(path) as f:
            reader = csv.reader(f)
            next(reader)  # skip header
            features = np.array([[float(value) for value in row] for row in reader])
    elif extension in ['.pkl', '.pckl', '.pickle']:
        with open(path, 'rb') as f:
            features = np.array([np.squeeze(np.array(feat.todense())) for feat in pickle.load(f)])
    else:
        raise ValueError(f'Features path extension {extension} not supported.')

    return features

