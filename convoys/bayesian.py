import numpy
import pymc3
import random
from scipy.special import expit
from pymc3.math import dot, sigmoid, log, exp

from convoys import Model

class WeibullRegression(Model):
    def fit(self, X, B, T):
        n, k = X.shape
        with pymc3.Model() as m:
            beta_sd = pymc3.Exponential('beta_sd', 1.0)  # Weak prior for the regression coefficients
            beta = pymc3.Normal('beta', mu=0, sd=beta_sd, shape=(k,))  # Regression coefficients
            c = sigmoid(dot(X, beta))  # Conversion rates for each example
            k = pymc3.Lognormal('k', mu=0, sd=1.0)  # Weak prior around k=1
            lambd = pymc3.Exponential('lambd', 0.1)  # Weak prior

            # PDF of Weibull: k * lambda * (x * lambda)^(k-1) * exp(-(t * lambda)^k)
            LL_observed = log(c) + log(k) + log(lambd) + (k-1)*(log(T) + log(lambd)) - (T*lambd)**k
            # CDF of Weibull: 1 - exp(-(t * lambda)^k)
            LL_censored = log((1-c) + c * exp(-(T*lambd)**k))

            # We need to implement the likelihood using pymc3.Potential (custom likelihood)
            # https://github.com/pymc-devs/pymc3/issues/826
            logp = B * LL_observed + (1 - B) * LL_censored
            logpvar = pymc3.Potential('logpvar', logp.sum())

            self.trace = pymc3.sample(n_simulations=500, tune=500, discard_tuned_samples=True, njobs=1)
            print('done')
        print('done 2')

    def predict(self):
        pass  # TODO: implement

    def predict_final(self, x):
        return numpy.mean(expit(numpy.dot(self.trace['beta'], x)))

    def predict_time(self):
        pass  # TODO: implement
