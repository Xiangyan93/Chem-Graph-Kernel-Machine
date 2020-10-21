import pickle
import numpy as np
import sklearn.gaussian_process as gp
from rdkit.Chem import AllChem as Chem
from rdkit.Chem.AtomPairs import Pairs
from rdkit.Chem.AtomPairs import Torsions
from chemml.smiles import inchi2smiles


def get_fingerprint(rdk_mol, type='morgan', nBits=0,
                    radius=1, useFeatures=False,  # morgan
                    minPath=1, maxPath=7,  # rdk
                    hash=False  # torsion
                    ):
    if type == 'rdk':
        if nBits == 0:  # output a dict :{identifier: occurance}
            return Chem.UnfoldedRDKFingerprintCountBased(
                rdk_mol,
                minPath=minPath,
                maxPath=maxPath
            ).GetNonzeroElements()
        else:  # output a string: '01010101'
            return Chem.RDKFingerprint(
                rdk_mol,
                minPath=minPath,
                maxPath=maxPath,
                fpSize=nBits
            ).ToBitString()
    elif type == 'morgan':
        if nBits == 0:
            info = dict()
            Chem.GetMorganFingerprint(
                rdk_mol,
                radius,
                bitInfo=info,
                useFeatures=useFeatures
            )
            for key in info:
                info[key] = len(info[key])
            return info
        else:
            return Chem.GetMorganFingerprintAsBitVect(
                rdk_mol,
                radius,
                nBits=nBits,
                useFeatures=useFeatures
            ).ToBitString()
    elif type == 'pair':
        if nBits == 0:
            return Pairs.GetAtomPairFingerprintAsIntVect(
                rdk_mol
            ).GetNonzeroElements()
        else:
            return Pairs.GetAtomPairFingerprintAsBitVect(
                rdk_mol
            ).ToBitString()
    elif type == 'torsion':
        if nBits == 0:
            if hash:
                return Torsions.GetHashedTopologicalTorsionFingerprint(
                    rdk_mol
                ).GetNonzeroElements()
            else:
                return Torsions.GetTopologicalTorsionFingerprintAsIntVect(
                    rdk_mol
                ).GetNonzeroElements()
        else:
            return None


class SubstructureFingerprint:
    def __init__(self, type='rdk', nBits=0, radius=1, minPath=1, maxPath=7):
        self.type = type
        self.nBits = nBits
        self.radius = radius
        self.minPath = minPath
        self.maxPath = maxPath

    def get_fp_list(self, inchi_list, size=None):
        fp_list = []
        if self.nBits == 0:
            hash_list = []
            _fp_list = []
            for inchi in inchi_list:
                rdk_mol = Chem.MolFromInchi(inchi)
                fp = get_fingerprint(rdk_mol, type=self.type, nBits=self.nBits,
                                     radius=self.radius,
                                     minPath=self.minPath, maxPath=self.maxPath)
                _fp_list.append(fp)
                for key in fp.keys():
                    if key not in hash_list:
                        hash_list.append(key)
            hash_list.sort()

            for _fp in _fp_list:
                fp = []
                for hash in hash_list:
                    if hash in _fp.keys():
                        fp.append(_fp[hash])
                    else:
                        fp.append(0)
                fp_list.append(fp)
            fp = np.array(fp_list)
            if size is not None and size < fp.shape[1]:
                idx = np.argsort((fp < 0.5).astype(int).sum(axis=0))[:size]
                return np.array(fp_list)[:, idx]
            else:
                return np.array(fp_list)
        else:
            for inchi in inchi_list:
                rdk_mol = Chem.MolFromInchi(inchi)
                fp = get_fingerprint(rdk_mol, type=self.type, nBits=self.nBits,
                                     radius=self.radius,
                                     minPath=self.minPath, maxPath=self.maxPath)
                fp = list(map(int, list(fp)))
                fp_list.append(fp)
            return np.array(fp_list)


class VectorFPConfig:
    def __init__(self, type,
                 nBits=0, size=0,
                 radius=2,  # parameters when type = 'morgan'
                 minPath=1, maxPath=7,  # parameters when type = 'topol'
                 add_features=None, add_hyperparameters=None, theta=None
                 ):
        self.fp = SubstructureFingerprint(
            type=type,
            nBits=nBits,
            radius=radius,
            minPath=minPath,
            maxPath=maxPath
        )
        self.size = size
        self.features = add_features
        self.hyperparameters = add_hyperparameters
        self.theta = theta

    def get_kernel(self, inchi_list):
        self.X = self.fp.get_fp_list(inchi_list, size=self.size)
        kernel_size = self.X.shape[1]
        if self.features is not None:
            kernel_size += len(self.features)
        self.kernel = gp.kernels.ConstantKernel(1.0, (1e-3, 1e3)) *\
                      gp.kernels.RBF(length_scale=np.ones(kernel_size))
        if self.theta is not None:
            print('Reading Existed kernel parameter %s' % self.theta)
            with open(self.theta, 'rb') as file:
                theta = pickle.load(file)
            self.kernel = self.kernel.clone_with_theta(theta)


def get_XY_from_df(df, kernel_config, properties=None):
    if df.size == 0:
        return None, None, None

    kernel_config.get_kernel(df.inchi.to_list())
    X = kernel_config.X
    if kernel_config.features is not None:
        X = np.concatenate([X, df[kernel_config.features].to_numpy()], axis=1)
    smiles = df.inchi.apply(inchi2smiles).to_numpy()
    Y = df[properties].to_numpy()
    if len(properties) == 1:
        Y = Y.ravel()
    return [X, Y, smiles]