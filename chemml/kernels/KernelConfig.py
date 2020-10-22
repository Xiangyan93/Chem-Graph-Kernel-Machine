import sklearn.gaussian_process as gp


class KernelConfig:
    def __init__(self, single_graph, multi_graph, add_features,
                 add_hyperparameters, params):
        assert(self.__class__ != KernelConfig)
        self.single_graph = single_graph
        self.multi_graph = multi_graph
        self.params = params
        self.add_features = add_features
        self.add_hyperparameters = add_hyperparameters
        self.hyperdict = params.get('hyperdict')

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
    X_name = kernel_config.single_graph + kernel_config.multi_graph
    if kernel_config.add_features is not None:
        X_name += kernel_config.add_features
    X = df[X_name].to_numpy()
    if properties is None:
        return X, None, None
    Y = df[properties].to_numpy()
    if len(properties) == 1:
        Y = Y.ravel()
    return X, Y, df['id'].to_numpy()
