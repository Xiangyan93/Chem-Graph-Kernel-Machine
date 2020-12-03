from .gpc import GPC
from ..gpc_learner import GPCLearner


class Learner(GPCLearner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = GPC(kernel=self.kernel, optimizer=self.optimizer)

    def train(self, train_X=None, train_Y=None):
        if train_X is None:
            train_X = self.train_X
        if train_Y is None:
            train_Y = self.train_Y
        self.model.fit(train_X, train_Y)
        print('hyperparameter: ', self.kernel_.hyperparameters)
