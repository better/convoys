import numpy
import scipy.optimize
from autograd import jacobian, hessian, grad
from autograd.scipy.special import expit, gamma, gammainc, gammaincc, gammaln
from autograd.numpy import exp, log, sum, dot

from convoys import Model

class WeibullRegression(Model):
    # This will replace the Weibull model in __init__.py soon.
    def fit(self, X, B, T):
        n, k = X.shape
        X = X.astype(numpy.float32)
        def f(x):
            lambd, k = exp(x[0]), exp(x[1])
            beta = x[2:]
            c = expit(dot(X, beta.T))  # Conversion rates for each example

            # PDF of Weibull: k * lambda * (x * lambda)^(k-1) * exp(-(t * lambda)^k)
            LL_observed = log(c) + log(k) + log(lambd) + (k-1)*(log(T) + log(lambd)) - (T*lambd)**k
            # CDF of Weibull: 1 - exp(-(t * lambda)^k)
            LL_censored = log((1-c) + c * exp(-(T*lambd)**k))

            LL = sum(B * LL_observed + (1 - B) * LL_censored)
            return -LL

        res = scipy.optimize.minimize(
            fun=f,
            jac=jacobian(f),
            hess=hessian(f),
            x0=numpy.zeros(k+2),
            method='trust-ncg')
        log_lambd, log_k = res.x[0], res.x[1]
        beta = res.x[2:]
        # Compute hessian of betas
        beta_hessian = hessian(f)(res.x)[2:,2:]
        self.params = dict(
            lambd=exp(log_lambd),
            k=exp(log_k),
            beta=beta,
            beta_hessian=beta_hessian
        )

    def predict(self):
        pass  # TODO: implement

    def predict_final(self, x, ci=None):
        x = numpy.array(x)
        def f(x, d=0):
            return expit(dot(x, self.params['beta']) + d)
        if ci:
            inv_var = dot(dot(x.T, self.params['beta_hessian']), x)
            lo, hi = (scipy.stats.norm.ppf(p, scale=inv_var**-0.5) for p in ((1 - ci)/2, (1 + ci)/2))
            return f(x), f(x, lo), f(x, hi)
        else:
            return f(x)

    def predict_time(self):
        pass  # TODO: implement
