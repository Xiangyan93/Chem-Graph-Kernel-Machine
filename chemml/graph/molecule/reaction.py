#!/usr/bin/env python
# -*- coding: utf-8 -*-
import re
import numpy as np
import rdkit.Chem.AllChem as Chem
from rdkit.Chem import rdChemReactions
from .substructure import *

correction_dict = {
    '[Cl:1][c:2]1[cH:3][c:4]2[c:5]([n:6][se:7][n:8]2)[cH:9][cH:10]1.[OH2:20].[OH:11][N+:12]([O-:13])=[O:14].[S:15](=[O:16])(=[O:17])([OH:18])[OH:19]>>[Cl:1][c:2]1[c:3]([N+:12](=[O:11])[O-:13])[c:4]2[c:5]([n:6][se:7][n:8]2)[cH:9][cH:10]1': '[Cl:1][c:2]1[cH:3][c:4]2[c:5]([nH:6][se:7][nH:8]2)[cH:9][cH:10]1.[OH2:20].[OH:11][N+:12]([O-:13])=[O:14].[S:15](=[O:16])(=[O:17])([OH:18])[OH:19]>>[Cl:1][c:2]1[c:3]([N+:12](=[O:11])[O-:13])[c:4]2[c:5]([nH:6][se:7][nH:8]2)[cH:9][cH:10]1'
}


def RxnFromSmarts(reaction_smarts, Assign=False, canonical=False,
                  IsTemplate=False, DeleteIon=True):
    """ Get correct RDKit reaction object.
    This function will:
        1. Sanitize all involved molecules if set IsTemplate=False
        2. Reassign atom labeling if set Assign=True
        3. Rearrange the atoms in all molecules to make the subsequent output
            string canonical if set canonical=True.
        4. Rearrange molecules into reactants, products and reagents correctly.

    Parameters
    ----------
    reaction_smarts: reaction smarts string.
    Assign: Reassigned the atom mapping number based on atomic order.
    canonical: Canonical all molecules in the reaction.
    IsTemplate: Is input a reaction template.
    Returns
    -------
    RDKit reaction object.
    """
    if not IsTemplate:
        assert (canonical == False)
    if DeleteIon:
        reaction_smarts = DeleteReactingIons(reaction_smarts)
    rxn = Chem.ReactionFromSmarts(reaction_smarts)
    if not IsTemplate:
        SanitizeRxn(rxn)
    if Assign:
        ReassignMappingNumber(rxn)
    if canonical:
        CanonicalizeRxn(rxn, IsTemplate=IsTemplate)
    ReactingAtoms = getReactingAtoms(rxn, depth=1, IsTemplate=IsTemplate)
    for reactant in rxn.GetReactants():
        if not IsReactMol(reactant, ReactingAtoms):
            RemoveAtomMap(reactant)
            rxn.AddAgentTemplate(reactant)
    for product in rxn.GetProducts():
        if not IsReactMol(product, ReactingAtoms):
            RemoveAtomMap(product)
            rxn.AddAgentTemplate(product)
    rxn.RemoveUnmappedReactantTemplates(thresholdUnmappedAtoms=1e-5)
    rxn.RemoveUnmappedProductTemplates(thresholdUnmappedAtoms=1e-5)
    return rxn


def IsTrivialReaction(reaction_smarts):
    rxn = RxnFromSmarts(reaction_smarts, Assign=False, canonical=False)
    rxn.RemoveAgentTemplates()
    for reactant in rxn.GetReactants():
        RemoveAtomMap(reactant)
    for product in rxn.GetProducts():
        RemoveAtomMap(product)
    r, p = rdChemReactions.ReactionToSmiles(rxn).split('>>')
    r = '.'.join(sorted(r.split('.')))
    p = '.'.join(sorted(p.split('.')))
    if r == p:
        return True
    else:
        return False
    # return unmapped_reaction_string


def SanitizeRxn(rxn):
    # The effect of this line is not sure.
    rdChemReactions.SanitizeRxn(rxn)
    # Sanitize all molecules.
    for reactant in rxn.GetReactants():
        Chem.SanitizeMol(reactant)
    for product in rxn.GetProducts():
        Chem.SanitizeMol(product)
    for agent in rxn.GetAgents():
        Chem.SanitizeMol(agent)
    return rxn


def ReassignMappingNumber(rxn):
    # Delete map number that exists only in reactants or products.
    reactantsAtomMapList = getAtomMapList(rxn.GetReactants())
    productsAtomMapList = getAtomMapList(rxn.GetProducts())
    for reactant in rxn.GetReactants():
        for atom in reactant.GetAtoms():
            AMN = atom.GetPropsAsDict().get('molAtomMapNumber')
            if AMN is not None and AMN not in productsAtomMapList:
                atom.ClearProp('molAtomMapNumber')
    for product in rxn.GetProducts():
        for atom in product.GetAtoms():
            AMN = atom.GetPropsAsDict().get('molAtomMapNumber')
            if AMN is not None and AMN not in reactantsAtomMapList:
                atom.ClearProp('molAtomMapNumber')
    # sort labeled atoms in reactant based on atomic environment
    atoms = []
    AEs = []
    for reactant in rxn.GetReactants():
        for atom in reactant.GetAtoms():
            map_number = atom.GetPropsAsDict().get('molAtomMapNumber')
            if map_number is not None:
                AE = AtomEnvironment(reactant, atom, depth=10,
                                     IsSanitized=False)
                AEs.append(AE)
                atoms.append(atom)
    atoms = np.asarray(atoms)[np.argsort(AEs)]
    # relabel molAtomMapNumber from 1 to N based on atom order.
    atoms_map_change_dict = dict()
    label = 1
    for atom in atoms:
        map_number = atom.GetPropsAsDict()['molAtomMapNumber']
        assert (map_number not in atoms_map_change_dict)
        atoms_map_change_dict[map_number] = label
        atom.SetAtomMapNum(label)
        label += 1
    for product in rxn.GetProducts():
        for atom in product.GetAtoms():
            AMN = getAtomMapNumber(atom)
            if AMN in atoms_map_change_dict:
                atom.SetAtomMapNum(atoms_map_change_dict[AMN])
    return rxn


def CanonicalizeRxn(rxn, IsTemplate=False):
    """ Canonicalize RDKit reaction object. Remember to run
    ReassignMappingNumber first.

    Parameters
    ----------
    rxn: RDKit reaction object.
    IsTemplate: Is reaction template.

    Returns
    -------
    Canonicalized RDKit reaction object.
    """
    can_agents = []
    for agent in rxn.GetAgents():
        can_agents.append(CanonicalizeMol(agent, IsSanitized=True))

    can_reactants = []
    for reactant in rxn.GetReactants():
        can_reactants.append(
            CanonicalizeMol(reactant, IsSanitized=not IsTemplate))
        RemoveAtomMap(reactant)
    rxn.RemoveUnmappedReactantTemplates(thresholdUnmappedAtoms=1e-5)
    for reactant in can_reactants:
        rxn.AddReactantTemplate(reactant)

    can_products = []
    for product in rxn.GetProducts():
        can_products.append(
            CanonicalizeMol(product, IsSanitized=not IsTemplate))
        RemoveAtomMap(product)
    rxn.RemoveUnmappedProductTemplates(thresholdUnmappedAtoms=1e-5)
    for product in can_products:
        rxn.AddProductTemplate(product)
    rxn.RemoveAgentTemplates()
    map(rxn.AddAgentTemplate, can_agents)
    return rxn


def CanonicalizeMol(mol, IsSanitized=True):
    """ Reorder the atoms sequence in mol, this is helpful to make the output
    SMILES or SMARTS string canonical.

    Parameters
    ----------
    mol: RDKit molecule object
    IsSanitized: Set False only for molecular fragments in reaction template.

    Returns
    -------
    Canonicalzed RDKit molecule object.
    """
    # Get sorted labeled atoms and unlabeled atoms
    labeled_atoms = []
    labeled_AEs = []
    unlabeled_atoms = []
    unlabeled_AEs = []
    for atom in mol.GetAtoms():
        map_number = atom.GetPropsAsDict().get('molAtomMapNumber')
        if map_number is None:
            AE = AtomEnvironment(mol, atom, depth=50, IsSanitized=IsSanitized)
            unlabeled_AEs.append(AE)
            unlabeled_atoms.append(atom)
        else:
            AE = AtomEnvironment(mol, atom, depth=50, IsSanitized=IsSanitized)
            labeled_AEs.append(AE)
            labeled_atoms.append(atom)
    labeled_atoms = list(np.asarray(labeled_atoms)[np.argsort(labeled_AEs)])
    unlabeled_atoms = list(
        np.asarray(unlabeled_atoms)[np.argsort(unlabeled_AEs)])
    N_atoms = len(labeled_atoms) + len(unlabeled_atoms)

    # create an atom-sorted molecule
    m = Chem.MolFromSmiles('.'.join(['C'] * N_atoms))
    mw = Chem.RWMol(m)
    idx_change = dict()
    idx = 0
    for atom in labeled_atoms + unlabeled_atoms:
        idx_change[atom.GetIdx()] = idx
        mw.ReplaceAtom(idx, atom)
        idx += 1
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        mw.AddBond(idx_change[i], idx_change[j], bond.GetBondType())
    can_mol = mw.GetMol()
    if IsSanitized:
        Chem.SanitizeMol(can_mol)
    return can_mol


def getAtomMapDict(mols, depth=1, IsTemplate=False):
    AtomMapDict = dict()
    for mol in mols:
        for atom in mol.GetAtoms():
            AMN = atom.GetPropsAsDict().get('molAtomMapNumber')
            if AMN is not None:
                if not IsTemplate:
                    AtomMapDict[AMN] = AtomEnvironment(
                        mol, atom, depth=depth, order_by_labeling=True)
                else:
                    AtomMapDict[AMN] = AtomEnvironment(
                        mol, atom, depth=depth, order_by_labeling=True,
                        IsSanitized=False)
    return AtomMapDict


def getAtomMapList(mols):
    AtomMapList = []
    for mol in mols:
        for atom in mol.GetAtoms():
            AMN = atom.GetPropsAsDict().get('molAtomMapNumber')
            if AMN is not None:
                assert (AMN not in AtomMapList)
                AtomMapList.append(AMN)
    return AtomMapList


def getReactingAtoms(rxn, depth=1, IsTemplate=False):
    ReactingAtoms = []
    reactantAtomMap = getAtomMapDict(rxn.GetReactants(), depth=depth,
                                     IsTemplate=IsTemplate)
    productAtomMap = getAtomMapDict(rxn.GetProducts(), depth=depth,
                                    IsTemplate=IsTemplate)
    for idx, AEr in reactantAtomMap.items():
        AEp = productAtomMap.get(idx)
        if AEp is None:
            continue
        atom_r = AEr.tree.all_nodes()[0].data
        atom_p = AEp.tree.all_nodes()[0].data
        if AEr != AEp or \
                (
                        not IsTemplate and atom_r.GetTotalNumHs() != atom_p.GetTotalNumHs()) or \
                (
                        IsTemplate and atom_r.GetNumExplicitHs() != atom_p.GetNumExplicitHs()) or \
                atom_r.GetFormalCharge() != atom_p.GetFormalCharge():
            ReactingAtoms.append(idx)
    return ReactingAtoms


def IsReactMol(mol, ReactingAtoms):
    for atom in mol.GetAtoms():
        if atom.GetPropsAsDict().get('molAtomMapNumber') in ReactingAtoms:
            return True
    else:
        return False


def RemoveAtomMap(mol):
    for atom in mol.GetAtoms():
        atom.ClearProp('molAtomMapNumber')


def DeleteReactingIons(reaction_smarts):
    # Change some ions into ionic format.
    # Sometimes a reaction [Mg+2]>>[Mg] may exist, I think this information is
    # useless and should be deleted.
    def sub_ion(rs, ion, charge):
        if re.search('(^|\.|>>)\[%s:\d{1,2}]' % ion, rs) and \
                re.search('(^|\.|>>)\[%s\+2:\d{1,2}]' % ion, rs):
            rs = re.sub('^\[%s:' % ion, '[%s%s:' % (ion, charge), rs)
            rs = re.sub('\.\[%s:' % ion, '.[%s%s:' % (ion, charge), rs)
            rs = re.sub('>>\[%s:' % ion, '>>[%s%s:' % (ion, charge), rs)
        return rs

    ions = ['Ca', 'Mg', 'Ba', 'Sr', 'Al']
    charges = ['+2', '+2', '+2', '+2', '+3']
    for i in range(len(ions)):
        reaction_smarts = sub_ion(reaction_smarts, ions[i], charge=charges[i])
    return reaction_smarts


def GetUnmappedReactionSmarts(reaction_smarts):
    rxn = rdChemReactions.ReactionFromSmarts(reaction_smarts)
    for reagent in rxn.GetAgents():
        rxn.AddReactantTemplate(reagent)
    rxn.RemoveAgentTemplates()
    for reactant in rxn.GetReactants():
        RemoveAtomMap(reactant)
    for product in rxn.GetProducts():
        RemoveAtomMap(product)
    return rdChemReactions.ReactionToSmiles(rxn)
