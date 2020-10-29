from .gpr import GPR
from chemml.learner import BaseLearner


class Learner(BaseLearner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = GPR(kernel=self.kernel,
                         optimizer=self.optimizer,
                         normalize_y=True,
                         alpha=self.alpha)

    def train(self):
        self.model.fit_loocv(self.train_X, self.train_Y, verbose=True, repeat=1)
        print('hyperparameter: ', self.model.kernel.hyperparameters)
