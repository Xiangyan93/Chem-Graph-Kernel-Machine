from .gpr import RobustFitGaussianProcessRegressor
from chemml.learner import BaseLearner


class Learner(BaseLearner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = RobustFitGaussianProcessRegressor(kernel=self.kernel,
                                                       optimizer=self.optimizer,
                                                       normalize_y=True,
                                                       alpha=self.alpha)

    def train(self, train_X=None, train_Y=None):
        if train_X is None:
            train_X = self.train_X
        if train_Y is None:
            train_Y = self.train_Y
        self.model.fit_robust(train_X, train_Y)
        print('hyperparameter: ', self.model.kernel_.hyperparameters)
