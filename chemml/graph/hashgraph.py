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
        g = cls.from_rdkit(mol, set_morgan_identifier=True,
                           set_elemental_mode=True,
                           set_ring_membership=True,
                           set_ring_stereo=True,
                           set_hydrogen=False,
                           set_group=True, set_group_rule='element')
        g = g.permute(rcm(g))
        g.smiles = Chem.MolToSmiles(mol)
        return g

    @classmethod
    def from_smiles(cls, smiles):
        mol = Chem.MolFromSmiles(smiles)
        g = cls.from_rdkit(mol, set_morgan_identifier=True,
                           set_elemental_mode=True,
                           set_ring_membership=True,
                           set_ring_stereo=True,
                           set_hydrogen=False,
                           set_group=True, set_group_rule='element')
        g = g.permute(rcm(g))
        g.smiles = Chem.MolToSmiles(mol)
        return g

    @classmethod
    def from_inchi_or_smiles(cls, input):
        if input.startswith('InChI'):
            return cls.from_inchi(input)
        else:
            return cls.from_smiles(input)

    @classmethod
    def from_rdkit(cls, mol, bond_type='order',
        set_morgan_identifier=False, morgan_radius=3,
        set_elemental_mode=False,
        set_ring_membership=False,
        set_ring_stereo=False, depth=5,
        set_hydrogen=False,
        set_group=False, set_group_rule='element', reaction_center=None
    ):
        return _from_rdkit(cls, mol, bond_type=bond_type,
                           set_morgan_identifier=set_morgan_identifier,
                           morgan_radius=morgan_radius,
                           set_elemental_mode=set_elemental_mode,
                           set_ring_membership=set_ring_membership,
                           set_ring_stereo=set_ring_stereo, depth=depth,
                           set_hydrogen=set_hydrogen,
                           set_group=set_group, set_group_rule=set_group_rule,
                           reaction_center=reaction_center)
