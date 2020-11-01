import json


class HyperJsonGenerator:
    def __init__(self, k_bounds=[0.5, 0.99], s_bounds=[0.5, 3.0]):
        self.k_bounds = k_bounds
        self.s_bounds = s_bounds
        self.tensorproduct_basis = {
            'normalization': True,
            'a_type': 'Tensorproduct',
            'b_type': 'Tensorproduct',
            'p_type': 'Additive_p',
            'q': [0.01, [0.0001, 1.0]],
            'atom_atomic_number': [['kDelta', 0.75, k_bounds]],
            'bond_order': [['sExp', 1.5, self.s_bounds]],
            'probability_atomic_number': [['Const_p', 1.0, "fixed"]]
        }
        self.additive_basis = {
            'normalization': True,
            'a_type': 'Additive',
            'b_type': 'Additive',
            'p_type': 'Additive_p',
            'q': [0.01, [0.0001, 1.0]],
            'atom_atomic_number': [
                ['kC', 0.5, [0.0001, 1.0]], ['kDelta', 0.75, self.k_bounds]
            ],
            'bond_order': [
                ['kC', 0.5, [0.0001, 1.0]], ['sExp', 1.5, self.s_bounds]
            ],
            'probability_atomic_number': [['Const_p', 1.0, "fixed"]]
        }

    def tensorproduct(self, elemental_mode=False, reaction=False,
                      normalization=True):
        tp = self.tensorproduct_basis.copy()
        tp.update({
            'atom_aromatic': [['kDelta', 0.9, self.k_bounds]],
            'atom_ring_number': [['kDelta', 0.9, self.k_bounds]],
            'atom_chiral': [['kDelta', 0.9, self.k_bounds]],
            # 'atom_hybridization': [['kDelta', 0.9, self.k_bounds]],
            'atom_ring_membership': [['kConv', 0.9, self.k_bounds]],
            'atom_charge': [['sExp', 2.5, self.s_bounds]],
            'atom_morgan_hash': [['kDelta', 0.9, self.k_bounds]],
            'atom_hcount': [['sExp', 2.5, self.s_bounds]],
            # 'bond_aromatic': [['kDelta', 0.9, self.k_bounds]],
            'bond_stereo': [['kDelta', 0.9, self.k_bounds]],
            'bond_conjugated': [['kDelta', 0.9, self.k_bounds]],
            'bond_ring_stereo': [['kDelta', 0.9, self.k_bounds]],
        })
        if elemental_mode:
            tp.pop('atom_atomic_number')
            tp.update({
                'atom_elemental_mode1': [['sExp', 2.0, self.s_bounds]],
                'atom_elemental_mode2': [['sExp', 3.0, self.s_bounds]],
            })
        if reaction:
            tp.pop('probability_atomic_number')
            tp.update({
                'probability_group_reaction': [['Uniform_p', 1.0, "fixed"]]
            })
        if not normalization:
            tp['normalization'] = False
        return tp

    def additive(self, elemental_mode=False, reaction=False,
                 normalization=True):
        ad = self.additive_basis.copy()
        ad.update({
            'atom_aromatic': [
                ['kC', 0.5, [0.01, 1.0]], ['kDelta', 0.9, self.k_bounds]
            ],
            'atom_ring_number': [
                ['kC', 0.5, [0.01, 1.0]], ['kDelta', 0.9, self.k_bounds]
            ],
            'atom_chiral': [
                ['kC', 0.5, [0.01, 1.0]], ['kDelta', 0.5, self.k_bounds]
            ],
            'atom_hybridization': [
                ['kC', 0.5, [0.01, 1.0]], ['kDelta', 0.9, self.k_bounds]
            ],
            'atom_ring_membership': [
                ['kC', 0.5, [0.01, 1.0]], ['kConv', 0.9, self.k_bounds]
            ],
            'atom_charge': [
                ['kC', 0.5, [0.01, 1.0]], ['sExp', 2.5, self.s_bounds]
            ],
            'atom_morgan_hash': [
                ['kC', 0.5, [0.01, 1.0]], ['kDelta', 0.9, self.k_bounds]
            ],
            # 'atom_hcount': [['kC', 0.5, [0.0001, 1.0]],
            #                 ['sExp', 2.5, self.s_bounds]],
            # 'bond_aromatic': [['kC', 0.5, [0.0001, 1.0]], ['kDelta', 0.9, self.k_bounds]],
            # 'bond_conjugated': [['kC', 0.5, [0.0001, 1.0]], ['kDelta', 0.9, self.k_bounds]],
            'bond_stereo': [
                ['kC', 0.5, [0.0001, 1.0]], ['kDelta', 0.5, self.k_bounds]
            ],
            'bond_ring_stereo': [
                ['kC', 0.5, [0.0001, 1.0]], ['kDelta', 0.5, self.k_bounds]
            ],
        })
        if elemental_mode:
            ad.pop('atom_atomic_number')
            ad.update({
                'atom_elemental_mode1': [
                    ['kC', 0.5, [0.0001, 1.0]],  ['sExp', 2.0, self.s_bounds]
                ],
                'atom_elemental_mode2': [
                    ['kC', 0.5, [0.0001, 1.0]], ['sExp', 3.0, self.s_bounds]
                ],
            })
        if reaction:
            ad.pop('probability_atomic_number')
            ad.update({
                'probability_group_reaction': [['Uniform_p', 1.0, "fixed"]]
            })
        if not normalization:
            ad['normalization'] = False
        return ad


hyper_json = HyperJsonGenerator()
open('tensorproduct-basis-NMGK.json', 'w').write(
    json.dumps(hyper_json.tensorproduct_basis))
open('tensorproduct-NMGK.json', 'w').write(
    json.dumps(hyper_json.tensorproduct()))
open('tensorproduct-MGK.json', 'w').write(
    json.dumps(hyper_json.tensorproduct(normalization=False)))
open('tensorproduct-reaction-NMGK.json', 'w').write(
    json.dumps(hyper_json.tensorproduct(reaction=True)))

open('additive-basis-NMGK.json', 'w').write(
    json.dumps(hyper_json.additive_basis))
open('additive-NMGK.json', 'w').write(
    json.dumps(hyper_json.additive()))
open('additive-MGK.json', 'w').write(
    json.dumps(hyper_json.additive(normalization=False)))
open('additive-reaction-NMGK.json', 'w').write(
    json.dumps(hyper_json.additive(reaction=True)))