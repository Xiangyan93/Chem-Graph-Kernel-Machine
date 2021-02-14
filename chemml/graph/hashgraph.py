import os
CWD = os.path.dirname(os.path.abspath(__file__))
import pandas as pd
from rdkit.Chem import AllChem as Chem
from graphdot import Graph
from graphdot.graph.reorder import rcm
from rdkit.Chem import rdChemReactions
from graphdot.graph._from_networkx import _from_networkx
import networkx as nx
from chemml.graph.from_rdkit import _from_rdkit, rdkit_config
from .reaction import *


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
    def from_inchi(cls, inchi, HASH, _rdkit_config=rdkit_config()):
        mol = Chem.MolFromInchi(inchi)
        g = cls.from_rdkit(mol, _rdkit_config, HASH)
        return g

    @classmethod
    def from_smiles(self, smiles, HASH, _rdkit_config=rdkit_config()):
        mol = Chem.MolFromSmiles(smiles)
        g = self.from_rdkit(mol, HASH, _rdkit_config)
        return g

    @classmethod
    def from_inchi_or_smiles(cls, inchi_or_smiles, HASH,
                             _rdkit_config=rdkit_config()):
        if inchi_or_smiles.startswith('InChI'):
            return cls.from_inchi(inchi_or_smiles, HASH, _rdkit_config)
        else:
            return cls.from_smiles(inchi_or_smiles, HASH, _rdkit_config)

    @classmethod
    def from_rdkit(cls, mol, HASH, _rdkit_config=rdkit_config()):
        _rdkit_config.preprocess(mol)
        g = _from_rdkit(cls, mol, _rdkit_config)
        g.hash = HASH
        # g = g.permute(rcm(g))
        return g

    @classmethod
    def from_atom_list(cls, atom_list, HASH):
        emode = pd.read_csv(os.path.join(CWD, 'emodes.dat'), sep='\s+')
        g = nx.Graph()
        for i, an in enumerate(atom_list):
            g.add_node(i)
            g.nodes[i]['ElementalMode1'] = emode[emode.an == an].em1.ravel()[0]
            g.nodes[i]['ElementalMode2'] = emode[emode.an == an].em2.ravel()[0]
            for j in range(i + 1, len(atom_list)):
                ij = (i, j)
                g.add_edge(*ij)
                g.edges[ij]['Order'] = 1.
        g = _from_networkx(cls, g)
        g.hash = HASH
        return g

    @classmethod
    def agent_from_reaction_smarts(cls, reaction_smarts, HASH,
                                   _rdkit_config=rdkit_config()):
        try:
            rxn = reaction_from_smarts(reaction_smarts)
            for agent in rxn.GetAgents():
                Chem.SanitizeMol(agent)

            if len(rxn.GetAgents()) == 0:
                return HashGraph.from_smiles('[He]', HASH, _rdkit_config)

            agents = HashGraph.from_rdkit(
                rxn.GetAgents()[0], '1', _rdkit_config).to_networkx()
            for mol in rxn.GetAgents()[1:]:
                g = HashGraph.from_rdkit(mol, '1', _rdkit_config).to_networkx()
                agents = nx.disjoint_union(agents, g)
            g = _from_networkx(cls, agents)
            g.hash = HASH
        except:
            return 'Parsing Error'
        return g

    @classmethod
    def from_reaction_smarts(cls, reaction_smarts, HASH):
        try:
            rxn = reaction_from_smarts(reaction_smarts)
            ReactingAtoms = getReactingAtoms(rxn, depth=1)

            _rdkit_config = rdkit_config(reaction_center=ReactingAtoms,
                                         reactant_or_product='reactant')
            reaction = HashGraph.from_rdkit(
                rxn.GetReactants()[0], '1', _rdkit_config).to_networkx()
            for reactant in rxn.GetReactants()[1:]:
                g = HashGraph.from_rdkit(reactant, '1', _rdkit_config).\
                    to_networkx()
                reaction = nx.disjoint_union(reaction, g)

            _rdkit_config = rdkit_config(reaction_center=ReactingAtoms,
                                         reactant_or_product='product')
            for product in rxn.GetProducts():
                g = HashGraph.from_rdkit(product, '1', _rdkit_config).\
                    to_networkx()
                reaction = nx.disjoint_union(reaction, g)

            g = _from_networkx(cls, reaction)
            g.hash = HASH
        except:
            return 'Parsing Error'
        if g.nodes.to_pandas()['ReactingCenter'].max() <= 0:
            raise RuntimeError(f'No reacting atoms are found in reactants:　'
                               f'{reaction_smarts}')
        if g.nodes.to_pandas()['ReactingCenter'].min() >= 0:
            raise RuntimeError(f'No reacting atoms are found in products:　'
                               f'{reaction_smarts}')
        return g
