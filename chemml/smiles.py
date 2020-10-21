from rdkit.Chem import AllChem as Chem


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
