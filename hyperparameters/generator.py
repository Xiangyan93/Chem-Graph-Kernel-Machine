#!/usr/bin/env python
# -*- coding: utf-8 -*-
import json
from typing import Dict, Iterator, List, Optional, Union, Literal, Tuple


class HyperJsonGenerator:
    def tMGR(self, k: float = 0.9, k_bounds: Tuple[float, float] = (0.5, 1.0),
             k_an=0.75):
        return {
            'Normalization': [10000, (1000, 30000)],
            'a_type': ['Tensorproduct', 'fixed'],
            'atom_AtomicNumber': {'kDelta': [k_an, k_bounds, 0.05]},
            'atom_AtomicNumber_list_1': {'kConv': [k, k_bounds, 0.05]},
            'atom_AtomicNumber_list_2': {'kConv': [k, k_bounds, 0.05]},
            'atom_AtomicNumber_list_3': {'kConv': [k, k_bounds, 0.05]},
            'atom_AtomicNumber_list_4': {'kConv': [k, k_bounds, 0.05]},
            'atom_MorganHash': {'kDelta': [k, k_bounds, 0.05]},
            'atom_Ring_count': {'kDelta': [k, k_bounds, 0.05]},
            'atom_RingSize_list': {'kConv': [k, k_bounds, 0.05]},
            'atom_Hcount': {'kDelta': [k, k_bounds]},
            'atom_AtomicNumber_count_1': {'kDelta': [k, k_bounds, 0.05]},
            'atom_AtomicNumber_count_2': {'kDelta': [k, k_bounds, 0.05]},
            'atom_Chiral': {'kDelta': [k, k_bounds, 0.05]},
            'b_type': ['Tensorproduct', 'fixed'],
            'bond_Order': {'kDelta': [k, k_bounds, 0.05]},
            'bond_Stereo': {'kDelta': [k, k_bounds, 0.05]},
            'bond_RingStereo': {'kDelta': [k, k_bounds, 0.05]},
            'p_type': ['Additive_p', 'fixed'],
            'probability_AtomicNumber': {'Const_p': [1.0, 'fixed']},
            'q': [0.01, [0.01, 0.5], 0.01],
        }

    def add(self, k: float = 0.9, k_bounds: Tuple[float, float] = (0.5, 1.0),
            c: float = 1.0, c_bounds: Tuple[float, float] = (1.0, 10.0)):
        return {
            'Normalization': [10000, (1000, 30000)],
            'a_type': ['Additive', 'fixed'],
            'atom_AtomicNumber': {'Const': [c, c_bounds, 1.0], 'kDelta': [k, k_bounds]},
            'atom_AtomicNumber_list_1': {'Const': [c, c_bounds], 'kConv': [k, k_bounds]},
            'atom_AtomicNumber_list_2': {'Const': [c, c_bounds], 'kConv': [k, k_bounds]},
            'atom_AtomicNumber_list_3': {'Const': [c, c_bounds], 'kConv': [k, k_bounds]},
            'atom_AtomicNumber_list_4': {'Const': [c, c_bounds], 'kConv': [k, k_bounds]},
            'atom_MorganHash': {'Const': [c, c_bounds], 'kDelta': [k, k_bounds]},
            'atom_Ring_count': {'Const': [c, c_bounds], 'kDelta': [k, k_bounds]},
            'atom_RingSize_list': {'Const': [c, c_bounds], 'kConv': [k, k_bounds]},
            'atom_Hcount': {'Const': [c, c_bounds], 'kDelta': [k, k_bounds]},
            'atom_AtomicNumber_count_1': {'Const': [c, c_bounds], 'kDelta': [k, k_bounds]},
            'atom_AtomicNumber_count_2': {'Const': [c, c_bounds], 'kDelta': [k, k_bounds]},
            'atom_Chiral': {'Const': [c, c_bounds], 'kDelta': [k, k_bounds]},
            'b_type': ['Additive', 'fixed'],
            'bond_Order': {'Const': [c, c_bounds], 'kDelta': [k, k_bounds]},
            'bond_Stereo': {'Const': [c, c_bounds], 'kDelta': [k, k_bounds]},
            'bond_RingStereo': {'Const': [c, c_bounds], 'kDelta': [k, k_bounds]},
            'p_type': ['Additive_p', 'fixed'],
            'probability_AtomicNumber': {'Const_p': [1.0, "fixed"]},
            'q': [0.01, (0.001, 0.5)],
        }

    def general(self, k: float = 0.9, k_bounds: Tuple[float, float] = (0.75, 1.0),
                c: float = 1.0, c_bounds: Tuple[float, float] = (1.0, 10.0),
                p: float = 1.0, p_bounds: Tuple[float, float] = (1.0, 10.0)):
        return {
            'Normalization': [10000, (1000, 30000), 1000],
            'a_type': ['Additive', 'fixed'],
            'atom_AtomicNumber': {'Const': [c, c_bounds, 1.0], 'kDelta': [k, k_bounds, 0.05]},
            'atom_AtomicNumber_list_1': {'Const': [c, c_bounds, 1.0], 'kConv': [k, k_bounds, 0.05]},
            'atom_AtomicNumber_list_2': {'Const': [c, c_bounds, 1.0], 'kConv': [k, k_bounds, 0.05]},
            'atom_AtomicNumber_list_3': {'Const': [c, c_bounds, 1.0], 'kConv': [k, k_bounds, 0.05]},
            'atom_AtomicNumber_list_4': {'Const': [c, c_bounds, 1.0], 'kConv': [k, k_bounds, 0.05]},
            'atom_MorganHash': {'Const': [c, c_bounds, 1.0], 'kDelta': [k, k_bounds, 0.05]},
            'atom_Ring_count': {'Const': [c, c_bounds, 1.0], 'kDelta': [k, k_bounds, 0.05]},
            'atom_RingSize_list': {'Const': [c, c_bounds, 1.0], 'kConv': [k, k_bounds, 0.05]},
            'atom_Hcount': {'Const': [c, c_bounds, 1.0], 'kDelta': [k, k_bounds, 0.05]},
            'atom_AtomicNumber_count_1': {'Const': [c, c_bounds, 1.0], 'kDelta': [k, k_bounds, 0.05]},
            'atom_AtomicNumber_count_2': {'Const': [c, c_bounds, 1.0], 'kDelta': [k, k_bounds, 0.05]},
            'atom_Chiral': {'Const': [c, c_bounds, 1.0], 'kDelta': [k, k_bounds, 0.05]},
            'b_type': ['Additive', 'fixed'],
            'bond_Order': {'Const': [c, c_bounds, 1.0], 'kDelta': [k, k_bounds, 0.05]},
            'bond_Stereo': {'Const': [c, c_bounds, 1.0], 'kDelta': [k, k_bounds, 0.05]},
            'bond_RingStereo': {'Const': [c, c_bounds, 1.0], 'kDelta': [k, k_bounds, 0.05]},
            'p_type': ['Additive_p', 'fixed'],
            'probability_AtomicNumber': {'Const_p': [1.0, "fixed"]},
            'probability_group_an5': {'Assign_p': [p, p_bounds, 1.0]},
            'probability_group_an6': {'Assign_p': [p, p_bounds, 1.0]},
            'probability_group_an7': {'Assign_p': [p, p_bounds, 1.0]},
            'probability_group_an8': {'Assign_p': [p, p_bounds, 1.0]},
            'probability_group_an9': {'Assign_p': [p, p_bounds, 1.0]},
            'probability_group_an14': {'Assign_p': [p, p_bounds, 1.0]},
            'probability_group_an15': {'Assign_p': [p, p_bounds, 1.0]},
            'probability_group_an16': {'Assign_p': [p, p_bounds, 1.0]},
            'probability_group_an17': {'Assign_p': [p, p_bounds, 1.0]},
            'probability_group_an35': {'Assign_p': [p, p_bounds, 1.0]},
            'probability_group_an53': {'Assign_p': [p, p_bounds, 1.0]},
            'q': [0.01, (0.01, 0.5), 0.01],
        }

hyper_json = HyperJsonGenerator()
open('tMGR.json', 'w').write(
    json.dumps(hyper_json.tMGR(), indent=1, sort_keys=False))
open('additive.json', 'w').write(
    json.dumps(hyper_json.add(), indent=1, sort_keys=False))
open('general.json', 'w').write(
    json.dumps(hyper_json.general(), indent=1, sort_keys=False))
