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
    def from_inchi(cls, inchi):
        mol = Chem.MolFromInchi(inchi)
        g = cls.from_rdkit(mol)
        g = g.permute(rcm(g))
        g.smiles = Chem.MolToSmiles(mol)
        return g

    @classmethod
    def from_smiles(cls, smiles):
        mol = Chem.MolFromSmiles(smiles)
        g = cls.from_rdkit(mol)
        g = g.permute(rcm(g))
        g.smiles = Chem.MolToSmiles(mol)
        return g

    @classmethod
    def from_rdkit(cls, mol, bond_type='order', set_ring_list=True,
                   set_ring_stereo=True):
        return _from_rdkit(cls, mol,
                           bond_type=bond_type,
                           set_ring_list=set_ring_list,
                           set_ring_stereo=set_ring_stereo,
                           morgan_radius=3,
                           depth=5)
