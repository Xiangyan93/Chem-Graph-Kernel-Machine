import json
from sklearn import svm

class HyperJsonGenerator:
    def __init__(self, k=0.9, k_bounds=[0.5, 0.99], s_bounds=[0.5, 3.0]):
        self.k = k
        self.k_bounds = k_bounds
        self.s_bounds = s_bounds
        self.tensorproduct_basis = {
            'Normalization': False,  # the normalization factor
            'a_type': 'Tensorproduct',
            'b_type': 'Tensorproduct',
            'p_type': 'Additive_p',
            'q': [0.01, [0.0001, 1.0]],
            'atom_AtomicNumber': [['kDelta', 0.75, k_bounds]],
            'bond_Order': [['kDelta', self.k, k_bounds]],
            # 'bond_Order': [['sExp', 1.5, s_bounds]],
            'probability_AtomicNumber': [['Const_p', 1.0, "fixed"]]
        }
        self.additive_basis = {
            'Normalization': False,
            'a_type': 'Additive',
            'b_type': 'Additive',
            'p_type': 'Additive_p',
            'q': [0.01, [0.0001, 1.0]],
            'atom_AtomicNumber': [
                ['kC', 0.5, [0.0001, 1.0]], ['kDelta', 0.75, self.k_bounds]
            ],
            'bond_Order': [
                ['kC', 0.5, [0.0001, 1.0]], ['kDelta', 0.75, self.k_bounds]
            ],
            'probability_AtomicNumber': [['Const_p', 1.0, "fixed"]]
        }

    def tensorproduct(self, elemental_mode=False, reaction=0,
                      inhomogeneous_start_probability=False,
                      normalization=False, normalizationMolSize=False):
        tp = self.tensorproduct_basis.copy()
        tp.update({
            'atom_AtomicNumber_1': [['kDelta', self.k, self.k_bounds]],
            'atom_AtomicNumber_2': [['kDelta', self.k, self.k_bounds]],
            'atom_AtomicNumber_3': [['kDelta', self.k, self.k_bounds]],
            'atom_AtomicNumber_4': [['kDelta', self.k, self.k_bounds]],
            #'atom_AtomicNumber_5': [['kConv', self.k, self.k_bounds]],
            #'atom_AtomicNumber_6': [['kConv', self.k, self.k_bounds]],
            #'atom_AtomicNumber_7': [['kConv', self.k, self.k_bounds]],
            #'atom_AtomicNumber_8': [['kConv', self.k, self.k_bounds]],
            #'atom_AtomicNumber_9': [['kConv', self.k, self.k_bounds]],
            #'atom_AtomicNumber_10': [['kConv', self.k, self.k_bounds]],
            # 'atom_Aromatic': [['kDelta', self.k, self.k_bounds]],
            #'atom_Aromatic_1': [['kConv', self.k, self.k_bounds]],
            #'atom_Aromatic_2': [['kConv', self.k, self.k_bounds]],
            #'atom_Aromatic_3': [['kConv', self.k, self.k_bounds]],
            #'atom_Aromatic_4': [['kConv', self.k, self.k_bounds]],
            'atom_RingNumber': [['kDelta', self.k, self.k_bounds]],
            'atom_Chiral': [['kDelta', self.k, self.k_bounds]],
            # 'atom_chiral_1': [['kConv', self.k, self.k_bounds]],
            #'atom_hybridization': [['kDelta', self.k, self.k_bounds]],
            'atom_RingMembership': [['kConv', self.k, self.k_bounds]],
            # 'atom_Charge': [['kDelta', self.k, self.k_bounds]],
            #'atom_Charge': [['sExp', 2.5, self.k_bounds]],
            'atom_MorganHash': [['kDelta', self.k, self.k_bounds]],
            #'atom_InRing': [['kDelta', self.k, self.k_bounds]],
            #'bond_InRing': [['kDelta', self.k, self.k_bounds]],
            'atom_FirstNeighbors': [['kDelta', self.k, self.k_bounds]],
            'atom_SecondNeighbors': [['kDelta', self.k, self.k_bounds]],
            #'atom_third_neighbors': [['kDelta', self.k, self.k_bounds]],
            #'atom_fourth_neighbors': [['kDelta', self.k, self.k_bounds]],
            #'atom_hcount_1': [['kConv', self.k, self.k_bounds]],
            # 'atom_hcount_2': [['kConv', self.k, self.k_bounds]],
            #'atom_hcount_3': [['kConv', self.k, self.k_bounds]],
            #'atom_hcount_4': [['kConv', self.k, self.k_bounds]],
            'atom_Hcount': [['kDelta', self.k, self.k_bounds]],
            # 'atom_FirstNeighbors_1': [['kConv', self.k, self.k_bounds]],
            # 'atom_FirstNeighbors_2': [['kConv', self.k, self.k_bounds]],
            #'bond_Aromatic': [['kDelta', self.k, self.k_bounds]],
            'bond_Stereo': [['kDelta', self.k, self.k_bounds]],
            # 'bond_Conjugated': [['kDelta', self.k, self.k_bounds]],
            'bond_RingStereo': [['kDelta', self.k, self.k_bounds]],
            #'bond_RingMembership': [['kConv', self.k, self.k_bounds]],
            #'bond_RingNumber': [['kDelta', self.k, self.k_bounds]],
            # 'atom_TPSA': [['kDelta', self.k, self.k_bounds]],
        })

        if elemental_mode:
            tp.pop('atom_AtomicNumber')
            tp.update({
                'atom_elemental_mode1': [['sExp', 2.0, self.s_bounds]],
                'atom_elemental_mode2': [['sExp', 3.0, self.s_bounds]],
            })
        if reaction == 1:
            tp.pop('probability_AtomicNumber')
            tp.update({
                'probability_group_reaction': [['Uniform_p', 1.0, "fixed"]]
            })
        elif reaction == 2:
            tp.update({
                'atom_group_reaction': [['kDelta', 0.5, self.k_bounds]]
            })
        if normalizationMolSize:
            tp['Normalization'] = [True, 1, "fixed"]
        elif normalization:
            tp['Normalization'] = True
        if inhomogeneous_start_probability:
            tp.pop('probability_AtomicNumber')
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
            'atom_Aromatic': [['kC', 0.5, [0.01, 1.0]], ['kDelta', self.k, self.k_bounds]],
            'atom_RingNumber': [['kC', 0.5, [0.01, 1.0]], ['kDelta', self.k, self.k_bounds]],
            'atom_chiral': [['kC', 0.5, [0.01, 1.0]], ['kDelta', 0.5, self.k_bounds]],
            'atom_hybridization': [['kC', 0.5, [0.01, 1.0]], ['kDelta', self.k, self.k_bounds]],
            'atom_RingMembership': [['kC', 0.5, [0.01, 1.0]], ['kConv', self.k, self.k_bounds]],
            'atom_charge': [['kC', 0.5, [0.01, 1.0]], ['sExp', 2.5, self.s_bounds]],
            'atom_morgan_hash': [['kC', 0.5, [0.01, 1.0]], ['kDelta', self.k, self.k_bounds]],
            'atom_hcount': [['kC', 0.5, [0.0001, 1.0]], ['sExp', 2.5, self.s_bounds]],
            # 'bond_Aromatic': [['kC', 0.5, [0.0001, 1.0]], ['kDelta', self.k, self.k_bounds]],
            # 'bond_conjugated': [['kC', 0.5, [0.0001, 1.0]], ['kDelta', self.k, self.k_bounds]],
            'bond_stereo': [['kC', 0.5, [0.0001, 1.0]], ['kDelta', 0.5, self.k_bounds]],
            'bond_ring_stereo': [['kC', 0.5, [0.0001, 1.0]], ['kDelta', 0.5, self.k_bounds]],
        })
        if elemental_mode:
            ad.pop('atom_AtomicNumber')
            ad.update({
                'atom_elemental_mode1': [
                    ['kC', 0.5, [0.0001, 1.0]],  ['sExp', 2.0, self.s_bounds]
                ],
                'atom_elemental_mode2': [
                    ['kC', 0.5, [0.0001, 1.0]], ['sExp', 3.0, self.s_bounds]
                ],
            })
        if reaction:
            ad.pop('probability_AtomicNumber')
            ad.update({
                'probability_group_reaction': [['Uniform_p', 1.0, "fixed"]]
            })
        if not normalization:
            ad['normalization'] = False
        return ad


hyper_json = HyperJsonGenerator()
open('tensorproduct-basis-NMGK.json', 'w').write(
    json.dumps(hyper_json.tensorproduct_basis))
open('tensorproduct-MSNMGK.json', 'w').write(
    json.dumps(hyper_json.tensorproduct(normalizationMolSize=True)))
open('tensorproduct-NMGK.json', 'w').write(
    json.dumps(hyper_json.tensorproduct(normalization=True)))
open('tensorproduct-MGK.json', 'w').write(
    json.dumps(hyper_json.tensorproduct()))
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