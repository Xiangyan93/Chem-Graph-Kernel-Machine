#!/usr/bin/env python
# -*- coding: utf-8 -*-
import re
from .reaction import *
from .smiles import *


def ExtractReactionTemplate(reaction_smarts, validate=True):
    rxn = RxnFromSmarts(reaction_smarts)
    reactants = rxn.GetReactants()
    products = rxn.GetProducts()
    ReactingAtoms = getReactingAtoms(rxn, depth=1)
    changed_atom_tags = list(map(str, ReactingAtoms))
    # Get fragments for reactants
    reactant_fragments = get_fragments_for_changed_atoms(
        reactants, changed_atom_tags, radius=1, category='reactants'
    )
    # Get fragments for products
    # (WITHOUT matching groups but WITH the addition of reactant fragments)
    product_fragments = get_fragments_for_changed_atoms(
        products, changed_atom_tags, radius=1, category='products'
    )
    rxn_string = '{}>>{}'.format(reactant_fragments, product_fragments)
    template = Chem.ReactionFromSmarts(rxn_string)

    if validate:
        # Make sure that applying the extracted template on the reactants could
        # obtain the products. Notice that you may get more than 1 products
        # sets. Only one of them is correct and others are fake reactions.
        if template.Validate()[1] != 0:
            raise RuntimeError(f'Could not validate reaction successfully. \n'
                               f'Input reaction smarts: {reaction_smarts}.\n'
                               f'Extracted reaction templates: {rxn_string}')

        map(RemoveAtomMap, reactants)
        SMILES_r = list(map(Chem.MolToSmiles, reactants))
        reactants = list(map(Chem.MolFromSmiles, SMILES_r))
        map(RemoveAtomMap, products)
        SMILES_p = Chem.MolToSmiles(CombineMols(products))
        template_products = template.RunReactants(reactants)
        SMILES_template_p = [Chem.MolToSmiles(CombineMols(products))
                             for products in template_products]
        if not SMILES_p in SMILES_template_p:
            print(reaction_smarts)
            print(SMILES_r)
            print(SMILES_p, SMILES_template_p)
        assert (SMILES_p in SMILES_template_p)
    return rxn_string


def get_fragments_for_changed_atoms(mols, changed_atom_tags, radius=0,
                                    category='reactants', expansion=[],
                                    USE_STEREOCHEMISTRY=True, USE_GROUP=False,
                                    verbose=False):
    '''Given a list of RDKit mols and a list of changed atom tags, this function
    computes the SMILES string of molecular fragments using MolFragmentToSmiles
    for all changed fragments'''

    fragments = ''
    for mol in mols:
        # Initialize list of replacement symbols (updated during expansion)
        symbol_replacements = []

        # Are we looking for groups? (reactants only)
        if USE_GROUP and category == 'reactants' and radius != 0:
            groups = get_special_groups(mol)
        else:
            groups = []

        # Build list of atoms to use
        atoms_to_use = []
        for atom in mol.GetAtoms():
            # Check self (only tagged atoms)
            if str(getAtomMapNumber(atom)) in changed_atom_tags:
                atoms_to_use.append(atom.GetIdx())
                symbol = get_detailed_smarts(atom)
                if symbol != atom.GetSmarts():
                    symbol_replacements.append((atom.GetIdx(), symbol))
                continue
        # print('stage1: ', atoms_to_use, symbol_replacements)
        # Check neighbors (any atom) and special groups in reactants
        # Inactive when set radius=0
        for k in range(radius):
            atoms_to_use, symbol_replacements = expand_atoms_to_use(
                mol,
                atoms_to_use,
                groups=groups,
                symbol_replacements=symbol_replacements
            )
        # print('stage2: ', atoms_to_use, symbol_replacements)

        '''
        # Add all unlabeled atoms that connected to the reaction center
        for atom in mol.GetAtoms():
            if str(getAtomMapNumber(atom)) in changed_atom_tags:
                for atom_ in atom.GetNeighbors():
                    if getAtomMapNumber(atom_) == -1:
                        frag = UnmappedFragment(mol, atom, atom_)
                        atoms_to_use += frag.get_atoms_idx()


        # I think the following part is useless.
        if category == 'products':
            # Add extra labels to include (for products only)
            if expansion:
                for atom in mol.GetAtoms():
                    if ':' not in atom.GetSmarts():
                        continue
                    label = atom.GetSmarts().split(':')[1][:-1]
                    if label in expansion and label not in changed_atom_tags:
                        atoms_to_use.append(atom.GetIdx())
                        # Make the expansion a wildcard
                        symbol_replacements.append(
                            (atom.GetIdx(), convert_atom_to_wildcard(atom)))
                        if verbose: print(
                            'expanded label {} to wildcard in products'.format(
                                label))
            # Make sure unmapped atoms are included (from products)
            for atom in mol.GetAtoms():
                print(atom.GetIdx(), getAtomMapNumber(atom))
                if not atom.HasProp('molAtomMapNumber'):
                    atoms_to_use.append(atom.GetIdx())
        '''
        # Define new symbols to replace terminal species with wildcards
        # (don't want to restrict templates too strictly)
        symbols = [atom.GetSmarts() for atom in mol.GetAtoms()]
        for (i, symbol) in symbol_replacements:
            symbols[i] = symbol

        if not atoms_to_use:
            continue
        # if v:
        # 	print('~~ replacement for this ' + category[:-1])
        # 	print('{} -> {}'.format([mol.GetAtomWithIdx(x).GetSmarts() for (x, s) in symbol_replacements],
        # 		                    [s for (x, s) in symbol_replacements]))
        # Remove molAtomMapNumber before canonicalization
        [x.ClearProp('molAtomMapNumber') for x in mol.GetAtoms()]
        fragments += '(' + Chem.MolFragmentToSmiles(mol, atoms_to_use,
                                                    atomSymbols=symbols,
                                                    allHsExplicit=True,
                                                    isomericSmiles=USE_STEREOCHEMISTRY,
                                                    allBondsExplicit=True) + ').'
    return fragments[:-1]


def get_detailed_smarts(atom, USE_STEREOCHEMISTRY=True):
    symbol = atom.GetSmarts()
    # Sometimes the Aromaticity of a atom is changed after sanitize, but its
    # Smarts didnt updated.
    if not atom.GetIsAromatic() and symbol[1].islower():
        symbol = symbol[0] + symbol[1].upper() + symbol[2:]
    # CUSTOM SYMBOL CHANGES
    if atom.GetTotalNumHs() == 0:
        # Be explicit when there are no hydrogens
        if ':' in symbol:  # stick H0 before label
            symbol = symbol.replace(':', ';H0:')
        else:  # stick before end
            symbol = symbol.replace(']', ';H0]')

    # print('Being explicit about H0!!!!')
    if atom.GetFormalCharge() == 0:
        # Also be explicit when there is no charge
        if ':' in symbol:
            symbol = symbol.replace(':', ';+0:')
        else:
            symbol = symbol.replace(']', ';+0]')
    if USE_STEREOCHEMISTRY:
        if atom.GetChiralTag() != Chem.rdchem.ChiralType.CHI_UNSPECIFIED:
            # Be explicit when there is a tetrahedral chiral tag
            if atom.GetChiralTag() == Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW:
                tag = '@'
            elif atom.GetChiralTag() == Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW:
                tag = '@@'
            if ':' in symbol:
                symbol = symbol.replace(':', ';{}:'.format(tag))
            else:
                symbol = symbol.replace(']', ';{}]'.format(tag))
    return symbol


def get_special_groups(mol, SUPER_GENERAL_TEMPLATES=False):
    '''Given an RDKit molecule, this function returns a list of tuples, where
    each tuple contains the AtomIdx's for a special group of atoms which should
    be included in a fragment all together. This should only be done for the
    reactants, otherwise the products might end up with mapping mismatches'''

    if SUPER_GENERAL_TEMPLATES:
        return []

    # Define templates, based on Functional_Group_Hierarchy.txt from Greg
    # Laandrum
    group_templates = [
        'C(=O)Cl',  # acid chloride
        'C(=O)[O;H,-]',  # carboxylic acid
        '[$(S-!@[#6])](=O)(=O)(Cl)',  # sulfonyl chloride
        '[$(B-!@[#6])](O)(O)',  # boronic acid
        '[$(N-!@[#6])](=!@C=!@O)',  # isocyanate
        '[N;H0;$(N-[#6]);D2]=[N;D2]=[N;D1]',  # azide
        'O=C1N(Br)C(=O)CC1',  # NBS brominating agent
        'C=O',  # carbonyl
        'ClS(Cl)=O',  # thionyl chloride
        '[Mg][Br,Cl]',  # grinard (non-disassociated)
        '[#6]S(=O)(=O)[O]',  # RSO3 leaving group
        '[O]S(=O)(=O)[O]',  # SO4 group
        '[N-]=[N+]=[C]',  # diazo-alkyl
    ]

    # Build list
    groups = []
    for template in group_templates:
        matches = mol.GetSubstructMatches(Chem.MolFromSmarts(template))
        groups.extend(list(matches))
    return groups


def expand_atoms_to_use(mol, atoms_to_use, groups=[], symbol_replacements=[]):
    '''Given an RDKit molecule and a list of AtomIdX which should be included
    in the reaction, this function expands the list of AtomIdXs to include one
    nearest neighbor with special consideration of (a) unimportant neighbors and
    (b) important functional groupings'''

    # Copy
    new_atoms_to_use = atoms_to_use[:]
    # Look for all atoms in the current list of atoms to use
    for atom in mol.GetAtoms():
        if atom.GetIdx() not in atoms_to_use:
            continue
        # Look for all nearest neighbors of the currently-included atoms
        for neighbor in atom.GetNeighbors():
            # Evaluate nearest neighbor atom to determine what should be included
            new_atoms_to_use, symbol_replacements = \
                expand_atoms_to_use_atom(mol, new_atoms_to_use,
                                         neighbor,
                                         groups=groups,
                                         symbol_replacements=symbol_replacements)

    return new_atoms_to_use, symbol_replacements


def expand_atoms_to_use_atom(mol, atoms_to_use, atom, groups=[],
                             symbol_replacements=[], verbose=False):
    '''Given an RDKit molecule and a list of AtomIdx which should be included
    in the reaction, this function extends the list of atoms_to_use by considering
    a candidate atom extension, atom_idx'''
    # Skip current candidate atom if it is already included
    if atom.GetIdx() in atoms_to_use:
        return atoms_to_use, symbol_replacements

    # See if this atom belongs to any groups
    found_in_group = False
    for group in groups:
        if int(atom.GetIdx()) in group:  # int correction
            if verbose:
                print('added group centered at {}'.format(atom_idx))
            # Add the whole list, redundancies don't matter
            atoms_to_use.extend(list(group))
            found_in_group = True
    if found_in_group:
        return atoms_to_use, symbol_replacements

    # How do we add an atom that wasn't in an identified important functional group?
    # Develop special SMARTS symbol

    # Include this atom
    atoms_to_use.append(atom.GetIdx())
    #symbol = get_detailed_smarts(atom)
    #if symbol != atom.GetSmarts():
    #    symbol_replacements.append((atom.GetIdx(), symbol))
    # Look for replacements
    symbol_replacements.append(
        (atom.GetIdx(), convert_atom_to_wildcard(atom)))

    return atoms_to_use, symbol_replacements


def convert_atom_to_wildcard(atom, verbose=False, SUPER_GENERAL_TEMPLATES=False):
    '''This function takes an RDKit atom and turns it into a wildcard
    using hard-coded generalization rules. This function should be used
    when candidate atoms are used to extend the reaction core for higher
    generalizability'''
    if SUPER_GENERAL_TEMPLATES:
        if ':' in atom.GetSmarts():
            # Fully generalize atom-mappped neighbors because they are aren't a leaving group
            label = re.search('\:[0-9]+\]', atom.GetSmarts())
            return '[*{}'.format(label.group())

    if not SUPER_GENERAL_TEMPLATES:

        # Is this a terminal atom? We can tell if the degree is one
        if atom.GetDegree() == 1:
            return atom.GetSmarts()

    # Initialize
    symbol = '['

    # Add atom primitive (don't use COMPLETE wildcards)
    if atom.GetAtomicNum() != 6:
        symbol += '#{};'.format(atom.GetAtomicNum())
    elif atom.GetIsAromatic():
        symbol += 'c;'
    else:
        symbol += 'C;'

    if not SUPER_GENERAL_TEMPLATES:
        # Charge is important
        if atom.GetFormalCharge() != 0:
            charges = re.search('([-+]+[1-9]?)', atom.GetSmarts())
            if charges: symbol += charges.group() + ';'

    # Strip extra semicolon
    if symbol[-1] == ';': symbol = symbol[:-1]

    # Close with label or with bracket
    label = re.search('\:[0-9]+\]', atom.GetSmarts())
    if label:
        symbol += label.group()
    else:
        symbol += ']'

    if verbose:
        if symbol != atom.GetSmarts():
            print('Improved generality of atom SMARTS {} -> {}'.format(
                atom.GetSmarts(), symbol))

    return symbol


def expand_changed_atom_tags(changed_atom_tags, reactant_fragments,
                             verbose=False):
    '''Given a list of changed atom tags (numbers as strings) and a string consisting
    of the reactant_fragments to include in the reaction transform, this function
    adds any tagged atoms found in the reactant side of the template to the
    changed_atom_tags list so that those tagged atoms are included in the products'''

    expansion = []
    atom_tags_in_reactant_fragments = re.findall('\:([[0-9]+)\]',
                                                 reactant_fragments)
    for atom_tag in atom_tags_in_reactant_fragments:
        if atom_tag not in changed_atom_tags:
            expansion.append(atom_tag)
    if verbose: print(
        'after building reactant fragments, additional labels included: {}'.format(
            expansion))
    return expansion
