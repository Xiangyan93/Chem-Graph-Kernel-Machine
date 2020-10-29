import os
CWD = os.path.dirname(os.path.abspath(__file__))
import pandas as pd
from rdkit.Chem import AllChem as Chem
from graphdot import Graph
from graphdot.graph.reorder import rcm
from chemml.graph.from_rdkit import _from_rdkit


class HashGraph(Graph):
    def __init__(self, smiles=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.smiles = smiles

    def __eq__(self, other):
        if self.smiles == other.smiles:
            return True
        else:
            return False

    def __lt__(self, other):
        if self.smiles < other.smiles:
            return True
        else:
            return False

    def __gt__(self, other):
        if self.smiles > other.smiles:
            return True
        else:
            return False

    def __hash__(self):
        return hash(self.smiles)

    @classmethod
    def from_inchi(cls, inchi, rdkit_config):
        mol = Chem.MolFromInchi(inchi)
        g = cls.from_rdkit(mol, rdkit_config)
        return g

    @classmethod
    def from_smiles(self, smiles, rdkit_config):
        mol = Chem.MolFromSmiles(smiles)
        g = self.from_rdkit(mol, rdkit_config)
        return g

    @classmethod
    def from_inchi_or_smiles(cls, input, rdkit_config):
        if input.startswith('InChI'):
            return cls.from_inchi(input, rdkit_config)
        else:
            return cls.from_smiles(input, rdkit_config)

    @classmethod
    def from_rdkit(cls, mol, rdkit_config):
        rdkit_config.preprocess(mol)
        g = _from_rdkit(cls, mol, rdkit_config)
        g.smiles = Chem.MolToSmiles(mol)
        g = g.permute(rcm(g))
        return g
