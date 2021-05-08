#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
CWD = os.path.dirname(os.path.abspath(__file__))
from typing import Dict, List, Tuple, Optional
import networkx as nx
import pandas as pd
import numpy as np
from rdkit.Chem import rdMolDescriptors
from graphdot.graph._from_networkx import _from_networkx
from rxntools.smiles import *
from rxntools.substructure import (
    FunctionalGroup,
    AtomEnvironment
)


def get_bond_orientation_dict(mol: Chem.Mol) -> Dict[Tuple[int, int], int]:
    bond_orientation_dict = {}
    mb = Chem.MolToMolBlock(mol, includeStereo=True, kekulize=False)
    for info in re.findall(r'^\s+\d+\s+\d+\s+\d+\s+\d+$', mb, re.MULTILINE):
        _, i, j, _, d = re.split(r'\s+', info)
        i, j, d = int(i) - 1, int(j) - 1, int(d)
        i, j = min(i, j), max(i, j)
        bond_orientation_dict[(i, j)] = d
    return bond_orientation_dict


def get_atom_ring_stereo(
        mol: Chem.Mol,
        atom: Chem.Atom,
        ring_idx : Tuple[int],
        depth: int = 5,
        bond_orientation_dict: Optional[Dict[Tuple[int, int], int]] = None):
    """Return an atom is upward or downward refer to a ring plane.

    For atom in a ring. If it has 4 bonds. Two of them are included in the
    ring. Other two connecting 2 functional groups, has opposite orientation
    reference to the ring plane. Assuming the ring is in a plane, then the 2
    functional groups are assigned as upward and downward.

    Parameters
    ----------
    mol : molecule object in RDKit

    atom : atom object in RDKit

    ring_idx : a tuple of all index of atoms in the ring

    depth : the depth of the functional group tree

    bond_orientation_dict : a dictionary contains the all bond orientation
        information in the molecule

    Returns
    -------
    0 : No ring stereo.

    1 : The upward functional group is larger

    -1 : The downward functional group is larger

    """
    if bond_orientation_dict is None:
        bond_orientation_dict = get_bond_orientation_dict(mol)

    up_atom = down_atom = None
    updown_tag = None
    if len(atom.GetNeighbors()) + atom.GetTotalNumHs() != 4:
        return 0
    elif atom.GetTotalNumHs() >= 2:
        return 0  # bond to more than 2 hydrogens
    else:  # if len(atom.GetNeighbors()) == 3:
        bonds_out_ring = []
        bonds_idx = []
        bonds_in_ring = []
        for bond in atom.GetBonds():
            i = bond.GetBeginAtom().GetIdx()
            j = bond.GetEndAtom().GetIdx()
            ij = (i, j)
            ij = (min(ij), max(ij))
            if i in ring_idx and j in ring_idx:
                # in RDKit, the orientation information may saved in ring bond
                # for multi-ring molecules. The information is saved.
                bonds_in_ring.append(bond_orientation_dict.get(ij))
            else:
                bonds_out_ring.append(bond_orientation_dict.get(ij))
                temp = list(ij)
                temp.remove(atom.GetIdx())
                bonds_idx.append(temp[0])
    if bonds_out_ring == [1]:
        return 1
    elif bonds_out_ring == [6]:
        return -1
    elif bonds_out_ring in [[0, 1], [6, 0], [6, 1]]:
        fg_up = FunctionalGroup(
            mol, atom, mol.GetAtomWithIdx(bonds_idx[1]), depth
        )
        fg_down = FunctionalGroup(
            mol, atom, mol.GetAtomWithIdx(bonds_idx[0]), depth
        )
    elif bonds_out_ring in [[1, 0], [0, 6], [1, 6]]:
        fg_up = FunctionalGroup(
            mol, atom, mol.GetAtomWithIdx(bonds_idx[0]), depth
        )
        fg_down = FunctionalGroup(
            mol, atom, mol.GetAtomWithIdx(bonds_idx[1]), depth
        )
    elif bonds_out_ring == [0]:
        if bonds_in_ring == [0, 0]:
            fg = FunctionalGroup(mol, atom, mol.GetAtomWithIdx(bonds_idx[0]), 1)
            for ij in fg.get_bonds_list():
                if bond_orientation_dict.get(ij) == 1:
                    return 1
                elif bond_orientation_dict.get(ij) == 6:
                    return -1
            return 0
        elif bonds_in_ring in [[1, 0], [0, 1], [6, 6]]:
            return 1
        elif bonds_in_ring in [[0, 6], [6, 0], [1, 1]]:
            return -1
        elif bonds_in_ring in [[6, 1], [1, 6]]:
            return -1
        else:
            return 0
    elif bonds_out_ring == [0, 0]:
        if bonds_in_ring == [0, 0]:
            return 0
        elif bonds_in_ring in [[1, 0], [0, 1]]:
            return 1
        elif bonds_in_ring in [[0, 6], [6, 0]]:
            return -1
        elif bonds_in_ring in [[1, 1], [6, 6]]:
            return 0
        else:
            return 0
    else:
        return 0

    if fg_up > fg_down:
        return 1
    elif fg_up < fg_down:
        return -1
    else:
        return 0


def IsSymmetric(mol: Chem.Mol, ij: Tuple[int, int], depth: int = 2) -> bool:
    atom0 = mol.GetAtomWithIdx(ij[0])
    atom1 = mol.GetAtomWithIdx(ij[1])
    fg_1 = FunctionalGroup(mol, atom0, atom1, depth)
    fg_2 = FunctionalGroup(mol, atom1, atom0, depth)
    if fg_1 == fg_2:
        return True
    else:
        return False


def get_chiral_tag(mol, atom, depth=5):
    """Get the chiral information of an atom in a molecule.

    Parameters
    ----------
    mol:
    atom
    depth

    Returns
    -------
    0: chiral atom with undefined chirality.
    1: non-chiral.
    2: clockwise, CW.
    3: anticlockwise, CCW.
    """
    if atom.GetHybridization() == 4 and atom.GetDegree() >= 3:
        fg = []
        for a in atom.GetNeighbors():
            fg_ = FunctionalGroup(mol, atom, a, depth=depth)
            if fg_ in fg:
                return 1
            else:
                fg.append(fg_)
        if atom.GetChiralTag() == 1:
            return 2
        elif atom.GetChiralTag() == 2:
            return 3
        else:
            return 0
    else:
        assert (atom.GetChiralTag() == 0)
        return 1


def get_group_id(atom):
    return [atom.GetAtomicNum()]


class rdkit_config:
    def __init__(self, bond_type='order',
                 set_morgan_identifier=True, morgan_radius=3,
                 set_ring_membership=True,
                 set_ring_stereo=True, depth=5,
                 set_elemental_mode=False,
                 set_hydrogen_explicit=False,
                 reaction_center=None, reactant_or_product='reactant',
                 concentration=1.0,
                 IsSanitized=True,
                 set_group=True,
                 set_TPSA=False,
                 set_partial_charge=False):
        self.bond_type = bond_type
        self.set_morgan_identifier = set_morgan_identifier
        self.morgan_radius = morgan_radius
        self.set_elemental_mode = set_elemental_mode
        self.set_ring_membership = set_ring_membership
        self.set_ring_stereo = set_ring_stereo
        self.depth = depth
        self.set_hydrogen_explicit = set_hydrogen_explicit
        self.reaction_center = reaction_center
        self.reactant_or_product = reactant_or_product
        self.concentration = concentration
        self.IsSanitized = IsSanitized
        self.set_group = set_group
        self.set_TPSA = set_TPSA
        self.set_partial_charge = set_partial_charge
        if self.set_elemental_mode:
            # read elemental modes.
            self.emode = pd.read_csv(os.path.join(CWD, 'emodes.dat'), sep='\s+')
        if self.set_group:
            an_list = [0, 1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 35, 53]
            self.group_dict = dict()
            for an in an_list:
                self.group_dict[an] = 'group_an%d' % an

    @staticmethod
    def get_list_hash(l):
        return hash(','.join(list(map(str, np.sort(l)))))

    def preprocess(self, mol):
        if self.set_hydrogen_explicit:
            mol = Chem.AddHs(mol)

        if self.set_morgan_identifier:
            # calculate morgan substrcutre hasing value
            morgan_info = dict()
            atomidx_hash_dict = dict()
            radius = self.morgan_radius
            Chem.GetMorganFingerprint(mol, radius, bitInfo=morgan_info,
                                      useChirality=False)
            while len(atomidx_hash_dict) != mol.GetNumAtoms():
                for key in morgan_info.keys():
                    if morgan_info[key][0][1] != radius:
                        continue
                    for a in morgan_info[key]:
                        if a[0] not in atomidx_hash_dict:
                            atomidx_hash_dict[a[0]] = key
                radius -= 1
            self.atomidx_hash_dict = atomidx_hash_dict
        if self.set_ring_membership:
            self.ringlist_atom = self.get_ringlist(mol)
            self.ringlist_bond = self.get_ringlist(mol, type='bond')
        if self.set_TPSA:
            self.TPSA = rdMolDescriptors._CalcTPSAContribs(mol)
        if self.set_partial_charge:
            Chem.ComputeGasteigerCharges(mol)

    @staticmethod
    def get_ringlist(mol, type='atom'):
        if type == 'atom':
            ringlist = [[] for _ in range(mol.GetNumAtoms())]
            for ring in mol.GetRingInfo().AtomRings():
                for i in ring:
                    ringlist[i].append(len(ring))
            return [sorted(rings) if len(rings) else [0] for rings in ringlist]
        else:
            ringlist = [[] for _ in range(mol.GetNumBonds())]
            for ring in mol.GetRingInfo().BondRings():
                for i in ring:
                    ringlist[i].append(len(ring))
            return [sorted(rings) if len(rings) else [0] for rings in ringlist]

    def set_node(self, node, atom, mol):
        an = atom.GetAtomicNum()
        node['AtomicNumber'] = an
        if self.IsSanitized:
            node['Charge'] = atom.GetFormalCharge()
            node['Hcount'] = atom.GetTotalNumHs()
        else:
            node['Charge'] = get_Charge_from_atom_smarts(atom.GetSmarts())
            node['Hcount'] = get_Hcount_from_atom_smarts(atom.GetSmarts())
        node['Hybridization'] = atom.GetHybridization()
        node['Aromatic'] = atom.GetIsAromatic()
        node['Chiral'] = get_chiral_tag(mol, atom)
        node['InRing'] = atom.IsInRing()
        if mol.GetNumAtoms() == 1:
            node['Concentration'] = self.concentration / 2
            node['SingleAtom'] = True
        else:
            node['Concentration'] = self.concentration / mol.GetNumAtoms()
            node['SingleAtom'] = False
        if self.set_elemental_mode:
            emode = self.emode
            node['ElementalMode1'] = emode[emode.an == an].em1.ravel()[0]
            node['ElementalMode2'] = emode[emode.an == an].em2.ravel()[0]

        if self.set_morgan_identifier:
            node['MorganHash'] = self.atomidx_hash_dict[atom.GetIdx()]

        if self.reaction_center is not None:
            if atom.GetPropsAsDict().get('molAtomMapNumber') in \
                    self.reaction_center:
                if self.reactant_or_product == 'reactant':
                    node['ReactingCenter'] = 1.0
                elif self.reactant_or_product == 'product':
                    node['ReactingCenter'] = - 1.0
                else:
                    raise RuntimeError(
                        f'You need to sepcify reactant or product '
                        f'{self.reactant_or_product}')
            else:
                node['ReactingCenter'] = 0.0
        if self.set_group:
            node['GroupID'] = get_group_id(atom)
            # assert node['GroupID'][0] in self.group_dict
            for key, value in self.group_dict.items():
                node[value] = 1.0 if key in node['GroupID'] else 0.0
        # set ring information
        if self.set_ring_membership:
            node['RingSize_list'] = np.asarray(
                self.ringlist_atom[atom.GetIdx()])
            node['RingSize_hash'] = self.get_list_hash(node['RingSize_list'])
            if self.ringlist_atom[atom.GetIdx()] == [0]:
                node['Ring_count'] = 0
            else:
                node['Ring_count'] = len(self.ringlist_atom[atom.GetIdx()])
        if self.set_TPSA:
            node['TPSA'] = self.TPSA[atom.GetIdx()]
        if self.set_partial_charge:
            node['GasteigerCharge'] = float(atom.GetProp('_GasteigerCharge'))
            node['GasteigerHCharge'] = float(atom.GetProp('_GasteigerHCharge'))

    def set_edge(self, edge, bond):
        if self.bond_type == 'order':
            edge['Order'] = bond.GetBondTypeAsDouble()
        else:
            edge['Type'] = bond.GetBondType()
        edge['Aromatic'] = bond.GetIsAromatic()
        edge['Conjugated'] = bond.GetIsConjugated()
        edge['Stereo'] = bond.GetStereo()
        edge['InRing'] = bond.IsInRing()
        # edge['Symmetry'] = False
        if self.set_ring_stereo:
            edge['RingStereo'] = 0.
        if self.set_ring_membership:
            edge['RingSize_list'] = np.asarray(
                self.ringlist_bond[bond.GetIdx()])
            edge['RingSize_hash'] = self.get_list_hash(edge['RingSize_list'])
            if self.ringlist_bond[bond.GetIdx()] == [0]:
                edge['Ring_count'] = 0
            else:
                edge['Ring_count'] = len(self.ringlist_bond[bond.GetIdx()])

    def set_node_propogation(self, graph, mol, attribute, depth=1, count=True,
                             usehash=True, sum=True):
        if mol.GetNumBonds() == 0:
            for i, atom in enumerate(mol.GetAtoms()):
                for depth_ in range(1, depth+1):
                    graph.nodes[i][attribute + '_list_%i' % depth_] = \
                        np.asarray([0])
                    if count:
                        graph.nodes[i][attribute + '_count_%i' % depth_] = 1
                    if usehash:
                        graph.nodes[i][attribute + '_hash_%i' % depth_] = hash('0')
                    if sum:
                        graph.nodes[i][attribute + '_sum_%i' % depth_] = 0
        else:
            for i, atom in enumerate(mol.GetAtoms()):
                assert (attribute in graph.nodes[i])
                AE = AtomEnvironment(mol, atom, depth=depth,
                                     IsSanitized=self.IsSanitized)
                for depth_ in range(1, depth+1):
                    neighbors = AE.get_nth_neighbors(depth_)
                    if neighbors:
                        graph.nodes[i][attribute + '_list_%i' % depth_] = \
                            np.asarray([graph.nodes[a.GetIdx()][attribute]
                                        for a in neighbors])
                    else:
                        graph.nodes[i][attribute + '_list_%i' % depth_] = \
                            np.asarray([0])
                    if count:
                        graph.nodes[i][attribute + '_count_%i' % depth_] = \
                            len(graph.nodes[i][attribute + '_list_%i' % depth_])
                    if usehash:
                        graph.nodes[i][attribute + '_hash_%i' % depth_] = \
                            self.get_list_hash(
                                graph.nodes[i][attribute + '_list_%i' % depth_])
                    if sum:
                        graph.nodes[i][attribute + '_sum_%i' % depth_] = \
                            np.sum(graph.nodes[i]
                                   [attribute + '_list_%i' % depth_])

    def set_ghost_edge(self, edge):
        if self.bond_type == 'order':
            edge['Order'] = 0.
        else:
            edge['Type'] = 0
        edge['Aromatic'] = False
        edge['Conjugated'] = False
        edge['Stereo'] = 0
        edge['InRing'] = False
        # edge['symmetry'] = False
        if self.set_ring_stereo:
            edge['RingStereo'] = 0.
        if self.set_ring_membership:
            edge['RingSize_list'] = np.asarray([0])
            edge['RingSize_hash'] = hash('0')
            edge['Ring_count'] = 0


def _from_rdkit(cls, mol, rdkit_config):
    if rdkit_config.set_hydrogen_explicit:
        mol = Chem.AddHs(mol)
    g = nx.Graph()
    # For single heavy-atom molecules, such as water, methane and metalic ion.
    # A ghost atom is created and bond to it, because there must be at least
    # two nodes and one edge in graph kernel.
    if mol.GetNumBonds() == 0:
        for i, atom in enumerate(mol.GetAtoms()):
            assert (atom.GetIdx() == i)
            g.add_node(i)
            rdkit_config.set_node(g.nodes[i], atom, mol)

        if mol.GetNumAtoms() == 1:
            ij = (0, 0)
            g.add_edge(*ij)
            rdkit_config.set_ghost_edge(g.edges[ij])
        else:
            I, J = np.triu_indices(mol.GetNumAtoms(), k=1)
            for i in range(len(I)):
                ij = (I[i], J[i])
                g.add_edge(*ij)
                rdkit_config.set_ghost_edge(g.edges[ij])
    else:
        for i, atom in enumerate(mol.GetAtoms()):
            assert (atom.GetIdx() == i)
            g.add_node(i)
            rdkit_config.set_node(g.nodes[i], atom, mol)
        for bond in mol.GetBonds():
            ij = (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
            g.add_edge(*ij)
            rdkit_config.set_edge(g.edges[ij], bond)

        # set ring stereo
        if rdkit_config.set_ring_stereo:
            bond_orientation_dict = get_bond_orientation_dict(mol)
            for ring_idx in mol.GetRingInfo().AtomRings():
                atom_updown = []
                for idx in ring_idx:
                    if g.nodes[idx]['Ring_count'] != 1:
                        atom_updown.append(0)
                    else:
                        atom = mol.GetAtomWithIdx(idx)
                        atom_updown.append(
                            get_atom_ring_stereo(
                                mol,
                                atom,
                                ring_idx,
                                depth=rdkit_config.depth,
                                bond_orientation_dict=bond_orientation_dict
                            )
                        )
                atom_updown = np.array(atom_updown)
                for j in range(len(ring_idx)):
                    b = j
                    e = j + 1 if j != len(ring_idx) - 1 else 0
                    StereoOfRingBond = float(atom_updown[b] * atom_updown[e] *
                                             len(ring_idx))
                    if ring_idx[b] < ring_idx[e]:
                        ij = (ring_idx[b], ring_idx[e])
                    else:
                        ij = (ring_idx[e], ring_idx[b])
                    if g.edges[ij]['RingStereo'] != 0.:
                        raise Exception(ij, g.edges[ij]['RingStereo'],
                                        StereoOfRingBond)
                    else:
                        g.edges[ij]['RingStereo'] = StereoOfRingBond
    #rdkit_config.set_node_propogation(g, mol, 'Chiral', depth=1)
    rdkit_config.set_node_propogation(g, mol, 'AtomicNumber', depth=5, sum=False)
    rdkit_config.set_node_propogation(g, mol, 'Hcount', depth=1, sum=True)
    #rdkit_config.set_node_propogation(g, mol, 'FirstNeighbors', depth=4)
    #rdkit_config.set_node_propogation(g, mol, 'Aromatic', depth=4)
    return _from_networkx(cls, g)
