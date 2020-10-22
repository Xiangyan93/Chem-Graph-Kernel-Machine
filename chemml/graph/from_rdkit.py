#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Adaptor for RDKit's Molecule objects"""
import os

CWD = os.path.dirname(os.path.abspath(__file__))
import re
import networkx as nx
import pandas as pd
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
            j = self.tree.get_node(identifier).predecessor(
                tree_id=self.tree._identifier)
            if j is None:
                continue
            ij = (min(i, j), max(i, j))
            bonds_list.append(ij)
        return bonds_list


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


def get_chiral_tag(mol, atom, depth=5):
    """

    Parameters
    ----------
    mol:
    atom
    depth

    Returns
    -------
    0: non-chiral
    1: clockwise, CW
    -1: anticlockwise, CCW
    """
    if atom.GetHybridization() == 4 and atom.GetDegree() >= 3:
        fg = []
        for a in atom.GetNeighbors():
            fg_ = FunctionalGroup(mol, atom, a, depth=depth)
            if fg_ in fg:
                return 0
            else:
                fg.append(fg_)
        if atom.GetChiralTag() == 1:
            return 1
        elif atom.GetChiralTag() == 2:
            return -1
        else:
            return 0
    else:
        if atom.GetChiralTag() == 0:
            return 0
        else:
            raise Exception('chiral tag error')


def get_group_id(atom, rule):
    if rule == 'element':
        return [atom.GetAtomicNum()]
    else:
        return [0]


def _from_rdkit(cls, mol, bond_type='order',
    set_morgan_identifier=False, morgan_radius=3,
    set_elemental_mode=False,
    set_ring_membership=False,
    set_ring_stereo=False, depth=5,
    set_hydrogen=False,
    set_group=False, set_group_rule='element', reaction_center=None
):
    g = nx.Graph()

    if set_hydrogen:
        mol = Chem.AddHs(mol)

    if set_morgan_identifier:
        # calculate morgan substrcutre hasing value
        def get_morgan_identifier(mol, morgan_radius):
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
            return atomidx_hash_dict

        atomidx_hash_dict = get_morgan_identifier(mol, morgan_radius)

    if set_elemental_mode:
        # read elemental modes.
        emode = pd.read_csv(os.path.join(CWD, 'emodes.dat'), sep='\s+')

    if set_group:
        group_dict = {
            1: 'an1', 5: 'an5', 6: 'an6', 7: 'an7', 8: 'an8', 9: 'an9',
            14: 'an14', 15: 'an15', 16: 'an16', 17: 'an17', 35: 'an35',
            53: 'an53'
        }  # group_id -> group name

    # set atomic attributes
    for i, atom in enumerate(mol.GetAtoms()):
        assert (i == atom.GetIdx())
        g.add_node(i)
        an = atom.GetAtomicNum()
        g.nodes[i]['atomic_number'] = an
        g.nodes[i]['charge'] = atom.GetFormalCharge()
        g.nodes[i]['hcount'] = atom.GetTotalNumHs()
        g.nodes[i]['hybridization'] = atom.GetHybridization()
        g.nodes[i]['aromatic'] = atom.GetIsAromatic()
        g.nodes[i]['chiral'] = get_chiral_tag(mol, atom)
        if g.nodes[i]['chiral'] == 0:
            g.nodes[i]['an_chiral'] = False
        else:
            g.nodes[i]['an_chiral'] = True
        if set_elemental_mode:
            g.nodes[i]['elemental_mode1'] = emode[emode.an == an].em1.ravel()[0]
            g.nodes[i]['elemental_mode2'] = emode[emode.an == an].em2.ravel()[0]
            #  g.nodes[i]['elemental_mode3'] = emode[emode.an == an].em3.ravel()[0]
            #  g.nodes[i]['elemental_mode4'] = emode[emode.an == an].em4.ravel()[0]
        if set_morgan_identifier:
            g.nodes[i]['morgan_hash'] = atomidx_hash_dict[atom.GetIdx()]

        if set_group:
            g.nodes[i]['group_id'] = get_group_id(atom, set_group_rule)
            for key, value in group_dict.items():
                g.nodes[i][value] = True if key in g.nodes[i]['group_id'] \
                    else False

        if reaction_center is not None:
            g.nodes[i]['an_reaction'] = True if i in reaction_center \
                else False

    # set ring information
    if set_ring_membership:
        for i, rings in enumerate(get_ringlist(mol)):
            g.nodes[i]['ring_membership'] = rings
            if rings == [0]:
                g.nodes[i]['ring_number'] = 0
            else:
                g.nodes[i]['ring_number'] = len(rings)

    # set bond attributes
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

    # set ring stereo
    if set_ring_stereo:
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
                            depth=depth,
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
                    if StereoOfRingBond != 0.:
                        g.nodes[ij[0]]['an_chiral'] = True
                        g.nodes[ij[1]]['an_chiral'] = True

    return _from_networkx(cls, g)
