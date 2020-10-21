#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Adaptor for RDKit's Molecule objects"""
import networkx as nx
import numpy as np
from treelib import Tree
from rdkit.Chem import AllChem as Chem
from graphdot.graph._from_networkx import _from_networkx


class FunctionalGroup:
    """Functional Group.

    atom0 -> atom1 define a directed bond in the molecule. Then the bond is
    removed and the functional group is defined as a multitree. atom1 is the
    root node of the tree.

    Parameters
    ----------
    mol : molecule object in RDKit

    atom0, atom1 : atom object in RDKit

    depth: the depth of the multitree.

    Attributes
    ----------
    tree : multitree represent the functional group
        each node has 3 important attributes: tag: [atomic number, bond order
        with its parent], identifier: atom index defined in RDKit molecule
        object, data: RDKit atom object.

    """

    def __init__(self, mol, atom0, atom1, depth=5):
        self.mol = mol
        tree = Tree()
        bond_order = mol.GetBondBetweenAtoms(
            atom0.GetIdx(),
            atom1.GetIdx()
        ).GetBondTypeAsDouble()
        tree.create_node(
            tag=[atom0.GetAtomicNum(), bond_order],
            identifier=atom0.GetIdx(), data=atom0
        )
        tree.create_node(
            tag=[atom1.GetAtomicNum(), bond_order],
            identifier=atom1.GetIdx(),
            data=atom1,
            parent=atom0.GetIdx()
        )
        for _ in range(depth):
            for node in tree.all_nodes():
                if node.is_leaf():
                    for atom in node.data.GetNeighbors():
                        tree_id = tree._identifier
                        if atom.GetIdx() != node.predecessor(tree_id=tree_id):
                            order = mol.GetBondBetweenAtoms(
                                atom.GetIdx(),
                                node.data.GetIdx()
                            ).GetBondTypeAsDouble()
                            identifier = atom.GetIdx()
                            while tree.get_node(identifier) is not None:
                                identifier += len(mol.GetAtoms())
                            tree.create_node(
                                tag=[atom.GetAtomicNum(), order],
                                identifier=identifier,
                                data=atom,
                                parent=node.identifier
                            )
        self.tree = tree

    def __eq__(self, other):
        if self.get_rank_list() == other.get_rank_list():
            return True
        else:
            return False

    def __lt__(self, other):
        if self.get_rank_list() < other.get_rank_list():
            return True
        else:
            return False

    def __gt__(self, other):
        if self.get_rank_list() > other.get_rank_list():
            return True
        else:
            return False

    def get_rank_list(self):
        rank_list = []
        expand_tree = self.tree.expand_tree(mode=Tree.WIDTH, reverse=True)
        for identifier in expand_tree:
            rank_list += self.tree.get_node(identifier).tag
        return rank_list

    def get_bonds_list(self):
        bonds_list = []
        expand_tree = self.tree.expand_tree(mode=Tree.WIDTH, reverse=True)
        for identifier in expand_tree:
            i = identifier
            j = self.tree.get_node(identifier).predecessor(tree_id=self.tree._identifier)
            if j is None:
                continue
            ij = (min(i, j), max(i, j))
            bonds_list.append(ij)
        return bonds_list


def get_bond_orientation_dict(mol):
    bond_orientation_dict = {}
    for line in Chem.MolToMolBlock(mol).split('\n'):
        if len(line.split()) == 4:
            a, b, _, d = line.split()
            ij = (int(a) - 1, int(b) - 1)
            ij = (min(ij), max(ij))
            bond_orientation_dict[ij] = int(d)
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
    else:#if len(atom.GetNeighbors()) == 3:
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
    elif bonds_out_ring == [0, 0]:
        if bonds_in_ring == [0, 0]:
            return 0
        elif bonds_in_ring in [[1, 0], [0, 1]]:
            return 1
        elif bonds_in_ring in [[0, 6], [6, 0]]:
            return -1
        elif bonds_in_ring in [[1, 1], [6, 6]]:
            return 0

    if fg_up > fg_down:
        return 1
    elif fg_up < fg_down:
        return -1
    else:
        return 0


def get_ringlist(mol):
    ringlist = [[] for _ in range(mol.GetNumAtoms())]
    for ring in mol.GetRingInfo().AtomRings():
        for i in ring:
            ringlist[i].append(len(ring))
    return [sorted(rings) if len(rings) else [0] for rings in ringlist]


def IsSymmetric(mol, ij, depth=2):
    atom0 = mol.GetAtomWithIdx(ij[0])
    atom1 = mol.GetAtomWithIdx(ij[1])
    fg_1 = FunctionalGroup(mol, atom0, atom1, depth)
    fg_2 = FunctionalGroup(mol, atom1, atom0, depth)
    if fg_1 == fg_2:
        return True
    else:
        return False


def _from_rdkit(cls, mol, bond_type='order', set_ring_list=True,
                set_ring_stereo=True, morgan_radius=3, depth=5):
    g = nx.Graph()
    morgan_info = dict()
    atomidx_hash_dict = dict()
    radius = morgan_radius
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

    for i, atom in enumerate(mol.GetAtoms()):
        g.add_node(i)
        g.nodes[i]['atomic_number'] = atom.GetAtomicNum()
        g.nodes[i]['charge'] = atom.GetFormalCharge()
        g.nodes[i]['hcount'] = atom.GetTotalNumHs()
        g.nodes[i]['hybridization'] = atom.GetHybridization()
        g.nodes[i]['aromatic'] = atom.GetIsAromatic()
        g.nodes[i]['chiral'] = 0 if atom.IsInRing() else atom.GetChiralTag()
        g.nodes[i]['morgan_hash'] = atomidx_hash_dict[atom.GetIdx()]

    if set_ring_list:
        for i, rings in enumerate(get_ringlist(mol)):
            g.nodes[i]['ring_list'] = rings
            if rings == [0]:
                g.nodes[i]['ring_number'] = 0
            else:
                g.nodes[i]['ring_number'] = len(rings)

    for bond in mol.GetBonds():
        ij = (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
        g.add_edge(*ij)
        if bond_type == 'order':
            g.edges[ij]['order'] = bond.GetBondTypeAsDouble()
        else:
            g.edges[ij]['type'] = bond.GetBondType()
        g.edges[ij]['aromatic'] = bond.GetIsAromatic()
        g.edges[ij]['conjugated'] = bond.GetIsConjugated()
        g.edges[ij]['stereo'] = bond.GetStereo()
        g.edges[ij]['symmetry'] = IsSymmetric(mol, ij)
        if set_ring_stereo is True:
            g.edges[ij]['ring_stereo'] = 0.

    if set_ring_stereo is True:
        bond_orientation_dict = get_bond_orientation_dict(mol)
        for ring_idx in mol.GetRingInfo().AtomRings():
            atom_updown = []
            for idx in ring_idx:
                if len(g.nodes[idx]['ring_list']) != 1:
                    atom_updown.append(0)
                else:
                    atom = mol.GetAtomWithIdx(idx)
                    atom_updown.append(
                        get_atom_ring_stereo(
                            mol,
                            atom,
                            ring_idx,
                            depth=depth,
                            bond_orientation_dict=bond_orientation_dict
                        )
                    )
            atom_updown = np.array(atom_updown)
            for j in range(len(ring_idx)):
                b = j
                e = j + 1 if j != len(ring_idx) - 1 else 0
                StereoOfRingBond = float(atom_updown[b] * atom_updown[e] /
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
