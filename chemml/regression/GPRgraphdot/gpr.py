import os
import pickle
from graphdot.model.gaussian_process.gpr import GaussianProcessRegressor


class GPR(GaussianProcessRegressor):
    def save(self, dir):
        """Save the GaussianProcessRegressor: dir/model.pkl.

        Parameters
        ----------
        dir: string
            The directory of saved model.

        """
        f_model = os.path.join(dir, 'model.pkl')
        store_dict = self.__dict__.copy()
        store_dict['theta'] = self.kernel.theta
        store_dict.pop('kernel', None)
        pickle.dump(store_dict, open(f_model, 'wb'), protocol=4)

    def load(self, dir):
        """Load the GaussianProcessRegressor: dir/model.pkl.

        Parameters
        ----------
        dir: string
            The directory of saved model.

        """
        f_model = os.path.join(dir, 'model.pkl')
        return self.load_cls(f_model, self.kernel)

    @classmethod
    def load_cls(cls, f_model, kernel):
        store_dict = pickle.load(open(f_model, 'rb'))
        kernel = kernel.clone_with_theta(store_dict.pop('theta'))
        model = cls(kernel)
        model.__dict__.update(**store_dict)
        return model
