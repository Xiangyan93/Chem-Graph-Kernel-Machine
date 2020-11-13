import json


class HyperJsonGenerator:
    def __init__(self, k_bounds=[0.5, 0.99], s_bounds=[0.5, 3.0]):
        self.k_bounds = k_bounds
        self.s_bounds = s_bounds
        self.tensorproduct_basis = {
            'Normalization': False,  # the normalization factor
            'a_type': 'Tensorproduct',
            'b_type': 'Tensorproduct',
            'p_type': 'Additive_p',
            'q': [0.01, [0.0001, 1.0]],
            'atom_atomic_number': [['kDelta', 0.75, k_bounds]],
            'bond_order': [['kDelta', 0.75, k_bounds]],
            'probability_atomic_number': [['Const_p', 1.0, "fixed"]]
        }
        self.additive_basis = {
            'Normalization': False,
            'a_type': 'Additive',
            'b_type': 'Additive',
            'p_type': 'Additive_p',
            'q': [0.01, [0.0001, 1.0]],
            'atom_atomic_number': [
                ['kC', 0.5, [0.0001, 1.0]], ['kDelta', 0.75, self.k_bounds]
            ],
            'bond_order': [
                ['kC', 0.5, [0.0001, 1.0]], ['kDelta', 0.75, self.k_bounds]
            ],
            'probability_atomic_number': [['Const_p', 1.0, "fixed"]]
        }

    def tensorproduct(self, elemental_mode=False, reaction=0,
                      inhomogeneous_start_probability=False,
                      normalization=False, normalizationMolSize=True):
        tp = self.tensorproduct_basis.copy()
        tp.update({# 40.96
            # 'atom_atomic_number_1': [['kConv', 0.9, self.k_bounds]],  # 39.7
            # 'atom_atomic_number_2': [['kConv', 0.9, self.k_bounds]],  # 39.7
            # 'atom_atomic_number_3': [['kConv', 0.9, self.k_bounds]],  # 39.7
            # 'atom_atomic_number_4': [['kConv', 0.9, self.k_bounds]],  # 39.7
            # 'atom_aromatic': [['kDelta', 0.9, self.k_bounds]],  # 39.7
            # 'atom_ring_number': [['kDelta', 0.9, self.k_bounds]],  #37.02
            # 'atom_chiral': [['kDelta', 0.9, self.k_bounds]],  # 40.42
            # 'atom_chiral_1': [['kConv', 0.9, self.k_bounds]],  # 40.42
            # 'atom_hybridization': [['kDelta', 0.9, self.k_bounds]],  #40.13
            # 'atom_ring_membership': [['kConv', 0.9, self.k_bounds]],  # 39.36
            # 'atom_charge': [['sExp', 2.5, self.s_bounds]],  # 40.67
            # 'atom_morgan_hash': [['kDelta', 0.9, self.k_bounds]],  # 35.42
            # 'atom_hcount': [['kDelta', 0.9, self.k_bounds]],  # 32.3
            # 'atom_hcount': [['sExp', 2.5, self.s_bounds]],  # 32.49
            # 'bond_aromatic': [['kDelta', 0.9, self.k_bounds]],  # 40.26
            # 'bond_stereo': [['kDelta', 0.9, self.k_bounds]],  # 41.07
            # 'bond_conjugated': [['kDelta', 0.9, self.k_bounds]],  # 40.67
            # 'bond_ring_stereo': [['kDelta', 0.9, self.k_bounds]],  # 40.98
        })
        if elemental_mode:
            tp.pop('atom_atomic_number')
            tp.update({
                'atom_elemental_mode1': [['sExp', 2.0, self.s_bounds]],
                'atom_elemental_mode2': [['sExp', 3.0, self.s_bounds]],
            })
        if reaction == 1:
            tp.pop('probability_atomic_number')
            tp.update({
                'probability_group_reaction': [['Uniform_p', 1.0, "fixed"]]
            })
        elif reaction == 2:
            tp.update({
                'atom_group_reaction': [['kDelta', 0.5, self.k_bounds]]
            })
        if normalizationMolSize:
            tp['normalization'] = [True, 100, "fixed"]
        elif normalization:
            tp['normalization'] = True
        if inhomogeneous_start_probability:
            tp.pop('probability_atomic_number')
            tp.update({
                'probability_group_an5': [['Uniform_p', 1.0, self.s_bounds]],
                'probability_group_an6': [['Uniform_p', 1.0, self.s_bounds]],
                'probability_group_an7': [['Uniform_p', 1.0, self.s_bounds]],
                'probability_group_an8': [['Uniform_p', 1.0, self.s_bounds]],
                'probability_group_an9': [['Uniform_p', 1.0, self.s_bounds]],
                'probability_group_an14': [['Uniform_p', 1.0, self.s_bounds]],
                'probability_group_an15': [['Uniform_p', 1.0, self.s_bounds]],
                'probability_group_an16': [['Uniform_p', 1.0, self.s_bounds]],
                'probability_group_an17': [['Uniform_p', 1.0, self.s_bounds]],
                'probability_group_an35': [['Uniform_p', 1.0, self.s_bounds]],
                'probability_group_an53': [['Uniform_p', 1.0, self.s_bounds]],
            })
        return tp

    def additive(self, elemental_mode=False, reaction=False,
                 normalization=True):
        ad = self.additive_basis.copy()
        ad.update({
            'atom_aromatic': [['kC', 0.5, [0.01, 1.0]], ['kDelta', 0.9, self.k_bounds]],
            'atom_ring_number': [['kC', 0.5, [0.01, 1.0]], ['kDelta', 0.9, self.k_bounds]],
            'atom_chiral': [['kC', 0.5, [0.01, 1.0]], ['kDelta', 0.5, self.k_bounds]],
            'atom_hybridization': [['kC', 0.5, [0.01, 1.0]], ['kDelta', 0.9, self.k_bounds]],
            'atom_ring_membership': [['kC', 0.5, [0.01, 1.0]], ['kConv', 0.9, self.k_bounds]],
            'atom_charge': [['kC', 0.5, [0.01, 1.0]], ['sExp', 2.5, self.s_bounds]],
            'atom_morgan_hash': [['kC', 0.5, [0.01, 1.0]], ['kDelta', 0.9, self.k_bounds]],
            'atom_hcount': [['kC', 0.5, [0.0001, 1.0]], ['sExp', 2.5, self.s_bounds]],
            # 'bond_aromatic': [['kC', 0.5, [0.0001, 1.0]], ['kDelta', 0.9, self.k_bounds]],
            # 'bond_conjugated': [['kC', 0.5, [0.0001, 1.0]], ['kDelta', 0.9, self.k_bounds]],
            'bond_stereo': [['kC', 0.5, [0.0001, 1.0]], ['kDelta', 0.5, self.k_bounds]],
            'bond_ring_stereo': [['kC', 0.5, [0.0001, 1.0]], ['kDelta', 0.5, self.k_bounds]],
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
open('tensorproduct-inhomo-NMGK.json', 'w').write(
    json.dumps(hyper_json.tensorproduct(inhomogeneous_start_probability=True)))
open('tensorproduct-em-NMGK.json', 'w').write(
    json.dumps(hyper_json.tensorproduct(elemental_mode=True)))
open('tensorproduct-reaction1-NMGK.json', 'w').write(
    json.dumps(hyper_json.tensorproduct(reaction=1)))
open('tensorproduct-reaction2-NMGK.json', 'w').write(
    json.dumps(hyper_json.tensorproduct(reaction=2)))

open('additive-basis-NMGK.json', 'w').write(
    json.dumps(hyper_json.additive_basis))
open('additive-NMGK.json', 'w').write(
    json.dumps(hyper_json.additive()))
open('additive-MGK.json', 'w').write(
    json.dumps(hyper_json.additive(normalization=False)))
open('additive-reaction-NMGK.json', 'w').write(
    json.dumps(hyper_json.additive(reaction=True)))