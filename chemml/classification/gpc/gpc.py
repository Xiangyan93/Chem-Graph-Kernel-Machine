#!/usr/bin/env python
# -*- coding: utf-8 -*-
from sklearn.gaussian_process._gpc import GaussianProcessClassifier as GPC
import os
import pickle

'''
class GPC(GaussianProcessClassifier):
    def save(self, dir):
        f_model = os.path.join(dir, 'model.pkl')
        store_dict = self.__dict__.copy()
        store_dict['theta'] = self.kernel.theta
        store_dict.pop('kernel', None)
        store_dict['theta_'] = self.kernel_.theta
        store_dict.pop('kernel_', None)
        pickle.dump(store_dict, open(f_model, 'wb'), protocol=4)

    def load(self, dir):
        f_model = os.path.join(dir, 'model.pkl')
        return self.load_cls(f_model, self.kernel)

    @classmethod
    def load_cls(cls, f_model, kernel):
        store_dict = pickle.load(open(f_model, 'rb'))
        kernel = kernel.clone_with_theta(store_dict.pop('theta'))
        kernel_ = kernel.clone_with_theta(store_dict.pop('theta_'))
        model = cls(kernel=kernel)
        model.kernel_ = kernel_
        model.__dict__.update(**store_dict)
        return model
'''