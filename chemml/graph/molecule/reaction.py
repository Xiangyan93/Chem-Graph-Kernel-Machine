#!/usr/bin/env python
# -*- coding: utf-8 -*-
import re
import rdkit.Chem.AllChem as Chem
from rdkit.Chem import rdChemReactions
from .substructure import *


correction_dict ={
    '[Cl:1][c:2]1[cH:3][c:4]2[c:5]([n:6][se:7][n:8]2)[cH:9][cH:10]1.[OH2:20].[OH:11][N+:12]([O-:13])=[O:14].[S:15](=[O:16])(=[O:17])([OH:18])[OH:19]>>[Cl:1][c:2]1[c:3]([N+:12](=[O:11])[O-:13])[c:4]2[c:5]([n:6][se:7][n:8]2)[cH:9][cH:10]1': '[Cl:1][c:2]1[cH:3][c:4]2[c:5]([nH:6][se:7][nH:8]2)[cH:9][cH:10]1.[OH2:20].[OH:11][N+:12]([O-:13])=[O:14].[S:15](=[O:16])(=[O:17])([OH:18])[OH:19]>>[Cl:1][c:2]1[c:3]([N+:12](=[O:11])[O-:13])[c:4]2[c:5]([nH:6][se:7][nH:8]2)[cH:9][cH:10]1'
}


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
                (not IsTemplate and atom_r.GetTotalNumHs() != atom_p.GetTotalNumHs()) or \
                (IsTemplate and atom_r.GetNumExplicitHs() != atom_p.GetNumExplicitHs()) or \
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
    return rxn


def RxnFromSmarts(reaction_smarts):
    """ Get correct RDKit reaction object.
    This function will:
        1. Sanitize all involved molecules.
        2. Rearrange molecules into reactants, products and reagents correctly.

    Parameters
    ----------
    reaction_smarts: reaction smarts string

    Returns
    -------
    RDKit reaction object
    """
    # Change some ions into ionic format.
    # print(reaction_smarts, '\n')
    def sub_ion(rs, ion, charge):
        if re.search('(^|\.|>>)\[%s:\d{1,2}]' % ion, rs) and \
                re.search('(^|\.|>>)\[%s\+2:\d{1,2}]' % ion, rs):
            rs = re.sub('^\[%s:' % ion, '[%s%s:' % (ion, charge), rs)
            rs = re.sub('\.\[%s:' % ion, '.[%s%s:' % (ion, charge), rs)
            rs = re.sub('>>\[%s:' % ion, '>>[%s%s:' % (ion, charge), rs)
        return rs
    reaction_smarts = sub_ion(reaction_smarts, 'Ca', charge='+2')
    reaction_smarts = sub_ion(reaction_smarts, 'Mg', charge='+2')
    reaction_smarts = sub_ion(reaction_smarts, 'Ba', charge='+2')
    reaction_smarts = sub_ion(reaction_smarts, 'Sr', charge='+2')
    reaction_smarts = sub_ion(reaction_smarts, 'Al', charge='+2')

    # print(reaction_smarts, '\n')
    rxn = Chem.ReactionFromSmarts(reaction_smarts)
    SanitizeRxn(rxn)
    ReactingAtoms = getReactingAtoms(rxn, depth=1)
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


def IsTrivialReaction(reaction_smarts):
    rxn = RxnFromSmarts(reaction_smarts)
    rxn.RemoveAgentTemplates()
    for reactant in rxn.GetReactants():
        RemoveAtomMap(reactant)
    for product in rxn.GetProducts():
        RemoveAtomMap(product)
    r, p = rdChemReactions.ReactionToSmiles(rxn).split('>>')
    if r == p:
        return True
    else:
        return False
