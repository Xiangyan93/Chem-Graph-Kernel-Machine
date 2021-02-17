from sklearn.svm import SVC as SVMClassifier


class SVC(SVMClassifier):
    @property
    def kernel_(self):
        return self.kernel
