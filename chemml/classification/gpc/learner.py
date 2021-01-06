from .gpc import GPC
from ..gpc_learner import GPCLearner


class Learner(GPCLearner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = GPC(kernel=self.kernel, optimizer=self.optimizer)

    def train(self, train_X=None, train_y=None):
        if train_X is None:
            train_X = self.train_X
        if train_y is None:
            train_y = self.train_Y
        self.model.fit(train_X, train_y)
        print('hyperparameter: ', self.kernel_.hyperparameters)
