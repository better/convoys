import abc
import six


@six.add_metaclass(abc.ABCMeta)
class Model():
    def __init__(self, params={}):
        self.params = params

    @abc.abstractmethod
    def fit(self, C, N, B):
        pass

    @abc.abstractmethod
    def predict(self, ts, ci=None):
        pass

    @abc.abstractmethod
    def predict_final(self, ci=None):
        pass

    @abc.abstractmethod
    def predict_time(self, ci=None):
        pass
