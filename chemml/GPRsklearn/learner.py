from .gpr import RobustFitGaussianProcessRegressor
from chemml.learner import BaseLearner


class Learner(BaseLearner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = RobustFitGaussianProcessRegressor(kernel=self.kernel,
                                                       optimizer=self.optimizer,
                                                       normalize_y=True,
                                                       alpha=self.alpha)

    def train(self, verbose=True):
        self.model.fit_robust(self.train_X, self.train_Y)
        if verbose:
            print('hyperparameter: ', self.model.kernel_.hyperparameters)
