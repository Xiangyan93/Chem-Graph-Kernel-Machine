import os
CWD = os.path.dirname(os.path.abspath(__file__))
from rdkit.Chem import AllChem as Chem
from graphdot import Graph
from graphdot.graph.reorder import rcm
from chemml.graph.from_rdkit import _from_rdkit

class HashGraph(Graph):
    def __init__(self, hash=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hash = hash

    def __eq__(self, other):
        if self.hash == other.hash:
            return True
        else:
            return False

    def __lt__(self, other):
        if self.hash < other.hash:
            return True
        else:
            return False

    def __gt__(self, other):
        if self.hash > other.hash:
            return True
        else:
            return False

    def __hash__(self):
        return hash(self.hash)

    @classmethod
    def from_inchi(cls, inchi, rdkit_config, hash):
        mol = Chem.MolFromInchi(inchi)
        g = cls.from_rdkit(mol, rdkit_config, hash)
        return g

    @classmethod
    def from_smiles(self, smiles, rdkit_config, hash):
        mol = Chem.MolFromSmiles(smiles)
        g = self.from_rdkit(mol, rdkit_config, hash)
        return g

    @classmethod
    def from_inchi_or_smiles(cls, input, rdkit_config, hash):
        if input.startswith('InChI'):
            return cls.from_inchi(input, rdkit_config, hash)
        else:
            return cls.from_smiles(input, rdkit_config, hash)

    @classmethod
    def from_rdkit(cls, mol, rdkit_config, hash):
        rdkit_config.preprocess(mol)
        g = _from_rdkit(cls, mol, rdkit_config)
        g.hash = hash
        # g = g.permute(rcm(g))
        return g
