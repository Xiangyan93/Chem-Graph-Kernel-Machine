#!/usr/bin/env python
# -*- coding: utf-8 -*-
from rdkit.Chem import AllChem as Chem
import re


def smiles2inchi(smiles):
    rdk_mol = Chem.MolFromSmiles(smiles)
    return mol2inchi(rdk_mol)


def inchi2smiles(inchi):
    rdk_mol = Chem.MolFromInchi(inchi)
    return mol2smiles(rdk_mol)


def mol2inchi(rdk_mol):
    return Chem.MolToInchi(rdk_mol)


def mol2smiles(rdk_mol):
    return Chem.MolToSmiles(rdk_mol)


def get_rdkit_smiles(smiles):
    rdk_mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(rdk_mol)


def CombineMols(mols):
    mol = mols[0]
    for m in mols[1:]:
        mol = Chem.CombineMols(mol, m)
    return mol


def get_Hcount_from_atom_smarts(atom_smarts):
    s = re.search('&H[0-9]+(:|&|])', atom_smarts)
    if s is None:
        return 0
    else:
        return int(s[0][2:-1])


def get_Charge_from_atom_smarts(atom_smarts):
    s = re.search('&(\+|-)\d{0,1}(:|&|])', atom_smarts)
    if s is None:
        return 0
    else:
        s = s[0][1:-1]
        if s == '-':
            s = '-1'
        if s == '+':
            s = '+1'
        return int(s)
