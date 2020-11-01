import sklearn.gaussian_process as gp
import numpy as np
import pandas as pd


class KernelConfig:
    def __init__(self, add_features, add_hyperparameters, params):
        assert (self.__class__ != KernelConfig)
        self.add_features = add_features
        self.add_hyperparameters = add_hyperparameters
        self.params = params

    def get_rbf_kernel(self):
        if None not in [self.add_features, self.add_hyperparameters]:
            if len(self.add_features) != len(self.add_hyperparameters):
                raise Exception('features and hyperparameters must be the same '
                                'length')
            add_kernel = gp.kernels.ConstantKernel(1.0, (1e-3, 1e3)) * \
                         gp.kernels.RBF(length_scale=self.add_hyperparameters)
            return [add_kernel]
        else:
            return []


def get_XYid_from_df(df, kernel_config, properties=None):
    if df.size == 0:
        return None, None, None
    if kernel_config.type == 'graph':
        X_name = kernel_config.single_graph + kernel_config.multi_graph
    elif kernel_config.type == 'preCalc':
        X_name = ['group_id']
    else:
        raise Exception('unknown kernel type:', kernel_config.type)
    if kernel_config.add_features is not None:
        X_name += kernel_config.add_features
    X = df[X_name].to_numpy()
    if properties is None:
        return X, None, None
    Y = df[properties].to_numpy()
    if len(properties) == 1:
        Y = Y.ravel()
    return X, Y, df['id'].to_numpy()


def get_Xgroupid_from_df(df, kernel_config):
    if df.size == 0:
        return None, None
    assert (kernel_config.type == 'graph')
    X_name = kernel_config.single_graph + kernel_config.multi_graph
    df_ = []
    for x in df.groupby('group_id'):
        for name in X_name:
            assert (len(np.unique(x[1][name])) == 1)
        df_.append(x[1].sample(1))
    df_ = pd.concat(df_)
    return df_[X_name].to_numpy(), df_['group_id'].to_numpy()
