import rdkit.Chem.AllChem as Chem
from rdkit.Chem import rdChemReactions
from .substructure import AtomEnvironment


def getAtomMapDict(mols, depth=1):
    AtomMapDict = dict()
    for mol in mols:
        Chem.SanitizeMol(mol)
        for atom in mol.GetAtoms():
            AMN = atom.GetPropsAsDict().get('molAtomMapNumber')
            if AMN is not None:
                AtomMapDict[AMN] = AtomEnvironment(
                    mol, atom, depth=depth)
    return AtomMapDict


def getReactingAtoms(rxn, depth=1):
    ReactingAtoms = []
    reactantAtomMap = getAtomMapDict(rxn.GetReactants(), depth=depth)
    productAtomMap = getAtomMapDict(rxn.GetProducts(), depth=depth)
    for idx, AE in reactantAtomMap.items():
        if AE != productAtomMap.get(idx):
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


def reaction_from_smarts(reaction_smarts):
    rxn = rdChemReactions.ReactionFromSmarts(reaction_smarts)
    rdChemReactions.SanitizeRxn(rxn)
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
