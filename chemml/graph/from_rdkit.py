#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Adaptor for RDKit's Molecule objects"""
import os

CWD = os.path.dirname(os.path.abspath(__file__))
import re
import networkx as nx
import pandas as pd
import numpy as np
from rdkit.Chem import AllChem as Chem
from graphdot.graph._from_networkx import _from_networkx
from .substructure import (
    FunctionalGroup,
    AtomEnvironment
)


def get_bond_orientation_dict(mol):
    bond_orientation_dict = {}
    mb = Chem.MolToMolBlock(mol, includeStereo=True, kekulize=False)
    for info in re.findall(r'^\s+\d+\s+\d+\s+\d+\s+\d+$', mb, re.MULTILINE):
        _, i, j, _, d = re.split(r'\s+', info)
        i, j, d = int(i) - 1, int(j) - 1, int(d)
        i, j = min(i, j), max(i, j)
        bond_orientation_dict[(i, j)] = d
    return bond_orientation_dict


def get_atom_ring_stereo(mol, atom, ring_idx, depth=5,
                         bond_orientation_dict=None):
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
                # in RDKit, the orientation information may saved in ring bond for
                # multi-ring molecules. The information is saved.
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


def IsSymmetric(mol, ij, depth=2):
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


def get_group_id(atom, rule):
    if rule == 'element':
        return [atom.GetAtomicNum()]
    else:
        return [0]


class rdkit_config:
    def __init__(self, bond_type='order',
                 set_morgan_identifier=True, morgan_radius=3,
                 set_elemental_mode=True,
                 set_ring_membership=True,
                 set_ring_stereo=True, depth=5,
                 set_hydrogen=False,
                 set_group=True, set_group_rule='element',
                 reaction_center=None):
        self.bond_type = bond_type
        self.set_morgan_identifier = set_morgan_identifier
        self.morgan_radius = morgan_radius
        self.set_elemental_mode = set_elemental_mode
        self.set_ring_membership = set_ring_membership
        self.set_ring_stereo = set_ring_stereo
        self.depth = depth
        self.set_hydrogen = set_hydrogen
        self.set_group = set_group
        self.set_group_rule = set_group_rule
        self.reaction_center = reaction_center
        if self.set_elemental_mode:
            # read elemental modes.
            self.emode = pd.read_csv(os.path.join(CWD, 'emodes.dat'), sep='\s+')
        if self.set_group:
            self.group_dict = {
                1: 'group_an1', 5: 'group_an5', 6: 'group_an6', 7: 'group_an7',
                8: 'group_an8', 9: 'group_an9', 14: 'group_an14',
                15: 'group_an15',
                16: 'group_an16', 17: 'group_an17', 35: 'group_an35',
                53: 'group_an53'
            }

    def preprocess(self, mol):
        if self.set_hydrogen:
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
            self.ringlist = self.get_ringlist(mol)

    @staticmethod
    def get_ringlist(mol):
        ringlist = [[] for _ in range(mol.GetNumAtoms())]
        for ring in mol.GetRingInfo().AtomRings():
            for i in ring:
                ringlist[i].append(len(ring))
        return [sorted(rings) if len(rings) else [0] for rings in ringlist]

    def set_node(self, node, atom, mol):
        an = atom.GetAtomicNum()
        node['atomic_number'] = an
        node['charge'] = atom.GetFormalCharge()
        node['hcount'] = atom.GetTotalNumHs()
        node['hybridization'] = atom.GetHybridization()
        node['aromatic'] = atom.GetIsAromatic()
        node['chiral'] = get_chiral_tag(mol, atom)
        node['chiral_1'] = [0]
        node['atomic_number_1'] = [0]
        node['atomic_number_2'] = [0]
        node['atomic_number_3'] = [0]
        node['atomic_number_4'] = [0]
        node['single_atom'] = True if mol.GetNumAtoms() == 1 else False
        if self.set_elemental_mode:
            emode = self.emode
            node['elemental_mode1'] = emode[emode.an == an].em1.ravel()[0]
            node['elemental_mode2'] = emode[emode.an == an].em2.ravel()[0]

        if self.set_morgan_identifier:
            node['morgan_hash'] = self.atomidx_hash_dict[atom.GetIdx()]

        if self.set_group:
            node['group_id'] = get_group_id(atom,
                                                  self.set_group_rule)
            for key, value in self.group_dict.items():
                node[value] = True if key in node['group_id'] else False

        if self.reaction_center is not None:
            node['group_reaction'] = True if atom.GetPropsAsDict().get(
                'molAtomMapNumber') in self.reaction_center else False

        # set ring information
        if self.set_ring_membership:
            node['ring_membership'] = self.ringlist[atom.GetIdx()]
            if self.ringlist[atom.GetIdx()] == [0]:
                node['ring_number'] = 0
            else:
                node['ring_number'] = len(self.ringlist[atom.GetIdx()])

    def set_edge(self, edge, bond):
        if self.bond_type == 'order':
            edge['order'] = bond.GetBondTypeAsDouble()
        else:
            edge['type'] = bond.GetBondType()
        edge['aromatic'] = bond.GetIsAromatic()
        edge['conjugated'] = bond.GetIsConjugated()
        edge['stereo'] = bond.GetStereo()
        # edge['symmetry'] = False
        if self.set_ring_stereo:
            edge['ring_stereo'] = 0.

    def set_node_propogation(self, graph, mol, attribute, depth=1):
        for i, atom in enumerate(mol.GetAtoms()):
            # print(graph.nodes[i], attribute)
            assert (attribute in graph.nodes[i])
            AE = AtomEnvironment(mol, atom, depth=depth)
            for depth_ in range(1, depth+1):
                neighbors = AE.get_nth_neighbors(depth_)
                if neighbors:
                    graph.nodes[i][attribute + '_%i' % depth_] = [
                        graph.nodes[a.GetIdx()][attribute] for a in neighbors]
                else:
                    graph.nodes[i][attribute + '_%i' % depth_] = [0]


def _from_rdkit(cls, mol, rdkit_config):
    g = nx.Graph()
    # For single heavy-atom molecules, such as water, methane and metalic ion.
    # A ghost atom is created and bond to it, because there must be at least
    # two nodes and one edge in graph kernel.
    if mol.GetNumAtoms() == 1:
        g.add_node(0)
        g.add_node(1)
        rdkit_config.set_node(g.nodes[0], mol.GetAtomWithIdx(0), mol)
        rdkit_config.set_node(g.nodes[1], mol.GetAtomWithIdx(0), mol)
        ij = (0, 1)
        g.add_edge(*ij)
        if rdkit_config.bond_type == 'order':
            g.edges[ij]['order'] = 0.
        else:
            g.edges[ij]['type'] = 0
        g.edges[ij]['aromatic'] = False
        g.edges[ij]['conjugated'] = False
        g.edges[ij]['stereo'] = 0
        # g.edges[ij]['symmetry'] = False
        if rdkit_config.set_ring_stereo:
            g.edges[ij]['ring_stereo'] = 0.

    else:
        for i, atom in enumerate(mol.GetAtoms()):
            assert (atom.GetIdx() == i)
            g.add_node(i)
            rdkit_config.set_node(g.nodes[i], atom, mol)
        rdkit_config.set_node_propogation(g, mol, 'chiral', depth=1)
        rdkit_config.set_node_propogation(g, mol, 'atomic_number', depth=4)
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
                    if g.nodes[idx]['ring_number'] != 1:
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
                    if g.edges[ij]['ring_stereo'] != 0.:
                        raise Exception(ij, g.edges[ij]['ring_stereo'],
                                        StereoOfRingBond)
                    else:
                        g.edges[ij]['ring_stereo'] = StereoOfRingBond
    return _from_networkx(cls, g)


def _from_rdkit_reaction(cls, rxn, rdkit_config):
    pass
