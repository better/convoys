import abc
import six


@six.add_metaclass(abc.ABCMeta)
class Model():
    @abc.abstractmethod
    def fit(self, X, B, T):
        pass

    @abc.abstractmethod
    def predict(self, x, ts, ci=None):
        pass

    @abc.abstractmethod
    def predict_final(self, x, ci=None):
        pass

    @abc.abstractmethod
    def predict_time(self, x, ci=None):
        pass
