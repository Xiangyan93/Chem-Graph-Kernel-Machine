import json

tensorproduct = {
    'a_type': 'Tensorproduct',
    'b_type': 'Tensorproduct',
    'p_type': 'Additive_p',
    'q': [0.01, [0.0001, 1.0]],
    'atom_atomic_number': [['kDelta', 0.75, [0.1, 1.0]]],
    'bond_order': [['sExp', 1.5, [0.1, 10.0]]],
    'probability_atomic_number': [['Const_p', 1.0, [1e-2, 1e2]]]
}
open('tensorproduct-basis.json', 'w').write(json.dumps(tensorproduct))
tensorproduct.update({
    'atom_aromatic': [['kDelta', 0.9, [0.1, 1.0]]],
    'atom_ring_number': [['kDelta', 0.9, [0.1, 1.0]]],
    'atom_chiral': [['kDelta', 0.5, [0.1, 1.0]]],
    'atom_hybridization': [['kDelta', 0.9, [0.1, 1.0]]],
    'atom_ring_membership': [['kConv', 0.9, [0.1, 1.0]]],
    'atom_charge': [['sExp', 2.5, [0.1, 10.0]]],
    # 'atom_hcount': [['sExp', 2.5, [0.1, 10.0]]],
    # 'bond_aromatic': [['kDelta', 0.9, [0.1, 1.0]]],
    'bond_stereo': [['kDelta', 0.5, [0.1, 1.0]]],
    # 'bond_conjugated': [['kDelta', 0.9, [0.1, 1.0]]],
    'bond_ring_stereo': [['kDelta', 0.5, [0.1, 1.0]]],
})
open('tensorproduct.json', 'w').write(json.dumps(tensorproduct))

def reaction(hyper):
    t = hyper.copy()
    t.pop('probability_atomic_number')
    t.update({
        'probability_group_reaction': [['Uniform_p', 1.0, [1e-2, 1e2]]]
    })
    open('tensorproduct-reaction.json', 'w').write(json.dumps(t))

reaction(tensorproduct)
'''

    'atom_aromatic': [['kDelta', 0.9, [0.1, 1.0]]],
    'atom_ring_number': [['kDelta', 0.9, [0.1, 1.0]]],
    # 'atom_morgan_hash': [['kDelta', 0.9, [0.1, 1.0]]],
    # 'atom_elemental_mode1': [['sExp', 2.0, [0.1, 10.0]]],
    # 'atom_elemental_mode2': [['sExp', 3.0, [0.1, 10.0]]],
    'atom_hcount': [['sExp', 2.5, [0.1, 10.0]]],
    'atom_ring_membership': [['kConv', 0.9, [0.1, 1.0]]],
    
    'bond_stereo': [['kDelta', 0.5, [0.1, 1.0]]],
    # 'bond_conjugated': [['kDelta', 0.9, [0.1, 1.0]]],
    'bond_ring_stereo': [['kDelta', 0.5, [0.1, 1.0]]],
}



additive1 = {
    'a_type': 'Additive',
    'b_type': 'Additive',
    'q': [0.01, [0.0001, 1.0]],
    'atom_atomic_number': [['kC', 0.5, [0.0001, 1.0]], ['kDelta', 0.75, [0.1, 1.0]]],
    'atom_aromatic': [['kC', 0.5, [0.0001, 1.0]], ['kDelta', 0.9, [0.1, 1.0]]],
    'atom_ring_number': [['kC', 0.5, [0.0001, 1.0]], ['kDelta', 0.9, [0.1, 1.0]]],
    # 'atom_hybridization': [['kC', 0.5, [0.0001, 1.0]], ['kDelta', 0.9, [0.1, 1.0]]],
    'atom_chiral': [['kC', 0.5, [0.0001, 1.0]], ['kDelta', 0.5, [0.1, 1.0]]],
    'atom_morgan_hash': [['kC', 0.5, [0.0001, 1.0]], ['kDelta', 0.9, [0.1, 1.0]]],
    # 'atom_elemental_mode1': [['kC', 0.5, [0.0001, 1.0]],  ['sExp', 2.0, [0.1, 10.0]]],
    # 'atom_elemental_mode2': [['kC', 0.5, [0.0001, 1.0]], ['sExp', 3.0, [0.1, 10.0]]],
    # 'atom_charge': [['kC', 0.5, [0.0001, 1.0]], ['sExp', 2.5, [0.1, 10.0]]],
    'atom_hcount': [['kC', 0.5, [0.0001, 1.0]], ['sExp', 2.5, [0.1, 10.0]]],
    'atom_ring_membership': [['kC', 0.5, [0.0001, 1.0]], ['kConv', 0.9, [0.1, 1.0]]],
    # 'bond_aromatic': [['kC', 0.5, [0.0001, 1.0]], ['kDelta', 0.9, [0.1, 1.0]]],
    'bond_stereo': [['kC', 0.5, [0.0001, 1.0]], ['kDelta', 0.5, [0.1, 1.0]]],
    # 'bond_conjugated': [['kC', 0.5, [0.0001, 1.0]], ['kDelta', 0.9, [0.1, 1.0]]],
    'bond_ring_stereo': [['kC', 0.5, [0.0001, 1.0]], ['kDelta', 0.5, [0.1, 1.0]]],
    'bond_order': [['kC', 0.5, [0.0001, 1.0]], ['sExp', 1.5, [0.1, 10.0]]],
}

open('tensorproduct-simple.json', 'w').write(json.dumps(tensorproduct1))
open('addtive-simple.json', 'w').write(json.dumps(additive1))
'''
