import autograd
import emcee
import numpy
from scipy.special import gammainc, gammaincinv
from autograd.scipy.special import expit, gammaln # , gammainc
from autograd.numpy import isnan, exp, dot, log, sum
import scipy.stats
import tensorflow as tf
import warnings
from convoys import tf_utils


def my_gammainc(k, x):
    if k == 1:
        return 1 - exp(-x)  # This is true for k=1
    else:
        return gammainc(k, x)


class RegressionModel(object):
    pass


class GeneralizedGamma(RegressionModel):
    # https://en.wikipedia.org/wiki/Generalized_gamma_distribution
    # Note however that lambda is a^-1 in WP's notation
    # Note also that k = d/p so d = k*p
    def __init__(self, method='MCMC'):
        self._method = method

    def fit(self, X, B, T, W=None, k=None, p=None):
        # Note on using Powell: tf.igamma returns the wrong gradient wrt k
        # https://github.com/tensorflow/tensorflow/issues/17995
        # Sanity check input:
        if W is None:
            W = [1] * len(X)
        XBTW = [(x, b, t, w) for x, b, t, w in zip(X, B, T, W)
                if t > 0 or float(t) not in [0, 1] or w < 0]
        if len(XBTW) < len(X):
            n_removed = len(X) - len(XBTW)
            warnings.warn('Warning! Removed %d entries from inputs where' +
                          'T <= 0 or B not 0/1 or W < 0' % n_removed)
        X, B, T, W = (numpy.array([z[i] for z in XBTW], dtype=numpy.float32)
                      for i in range(4))
        n_features = X.shape[1]

        # Define model
        # Note that scipy.optimize and emcee forces the the parameters to be a vector:
        # (log k, log p, log sigma_alpha, log sigma_beta, a, b, alpha_1...alpha_k, beta_1...beta_k)
        fix_k, fix_p = k, p

        def log_likelihood(x):
            k = exp(x[0]) if fix_k is None else fix_k
            p = exp(x[1]) if fix_p is None else fix_p
            log_sigma_alpha = x[2] if n_features > 1 else 0
            log_sigma_beta = x[3] if n_features > 1 else 0
            a = x[4]
            b = x[5]
            alpha = x[6:6+n_features]
            beta = x[6+n_features:6+2*n_features]
            lambd = exp(dot(X, alpha)+a)
            c = expit(dot(X, beta)+b)

            # PDF: p*lambda^(k*p) / gamma(k) * t^(k*p-1) * exp(-(x*lambda)^p)
            log_pdf = \
                      log(p) + (k*p) * log(lambd) \
                      - gammaln(k) + (k*p-1) * log(T) \
                      - (T*lambd)**p
            cdf = my_gammainc(k, (T*lambd)**p)

            LL_observed = log(c) + log_pdf
            LL_censored = log((1-c) + c * (1 - cdf))

            LL_data = sum(
                W * B * LL_observed +
                W * (1 - B) * LL_censored, 0)
            LL_prior_a = -dot(alpha, alpha) / (2*exp(log_sigma_alpha)**2) - n_features*log_sigma_alpha
            LL_prior_b = -dot(beta, beta) / (2*exp(log_sigma_beta)**2) - n_features*log_sigma_beta

            LL = LL_prior_a + LL_prior_b + LL_data
            if isnan(LL):
                return -numpy.inf
            else:
                if isinstance(x, numpy.ndarray):
                    print('%9.6f %9.6f %9.6f %9.6f -> %9.6f %30s' % (k, p, exp(log_sigma_alpha), exp(log_sigma_beta), LL, '')) #, end='\r')
                return LL

        x0 = numpy.zeros(6+2*n_features)
        print('\nFinding MAP:')
        neg_log_likelihood = lambda x: -log_likelihood(x)
        fix_k = 1.0
        res = scipy.optimize.minimize(
            neg_log_likelihood,
            x0,
            jac=autograd.grad(neg_log_likelihood),
            method='SLSQP',
        )
        x0 = res.x
        print('\nFinding MAP (again, letting k vary):')
        fix_k = k
        res = scipy.optimize.minimize(
            neg_log_likelihood,
            x0,
            method='Powell',
        )
        x0 = res.x
        print('\nStarting MCMC:')
        if self._method == 'MCMC':
            nwalkers = 100
            dim, = x0.shape
            sampler = emcee.EnsembleSampler(
                nwalkers=nwalkers,
                dim=dim,
                lnpostfn=log_likelihood)
            p0 = [x0 + 1e-4 * numpy.random.randn(dim) for i in range(nwalkers)]
            sampler.run_mcmc(p0, 500)
            print('\nDumping results:')
            print(sampler.chain.shape)
            samples = sampler.chain[:,100:,].reshape((-1, dim))
            print(samples.shape)
            import corner
            fig = corner.corner(samples)
            fig.savefig('samples.png')

    def cdf(self, x, t, ci=None, n=1000):
        t = numpy.array(t)
        a = LinearCombination.sample(self.params['a'], x, ci, n)
        b = LinearCombination.sample(self.params['b'], x, ci, n)
        return tf_utils.predict(
            expit(b) * gammainc(
                self.params['k'],
                numpy.multiply.outer(t, numpy.exp(a))**self.params['p']),
            ci)

    def rvs(self, x, n_curves=1, n_samples=1, T=None):
        # Samples values from this distribution
        # T is optional and means we already observed non-conversion until T
        if T is None:
            T = numpy.zeros((n_curves, n_samples))
        else:
            assert T.shape == (n_curves, n_samples)
        a = LinearCombination.sample(self.params['a'], x, 1, n_curves)
        b = LinearCombination.sample(self.params['b'], x, 1, n_curves)
        B = numpy.zeros((n_curves, n_samples), dtype=numpy.bool)
        C = numpy.zeros((n_curves, n_samples))
        for i, (a, b) in enumerate(zip(a, b)):
            z = numpy.random.uniform(size=(n_samples,))
            cdf_now = expit(b) * gammainc(
                self.params['k'],
                numpy.multiply.outer(T[i], numpy.exp(a))**self.params['p'])
            cdf_final = expit(b)
            adjusted_z = cdf_now + (1 - cdf_now) * z
            B[i] = (adjusted_z < cdf_final)
            y = adjusted_z / cdf_final
            x = gammaincinv(self.params['k'], y)
            # x = (t * exp(a))**p
            C[i] = x**(1./self.params['p']) / numpy.exp(a)
            C[i][~B[i]] = 0

        return B, C


class Exponential(GeneralizedGamma):
    def fit(self, X, B, T, W=None):
        super(Exponential, self).fit(X, B, T, W, k=1, p=1)


class Weibull(GeneralizedGamma):
    def fit(self, X, B, T, W=None):
        super(Weibull, self).fit(X, B, T, W, k=1)


class Gamma(GeneralizedGamma):
    def fit(self, X, B, T, W=None):
        super(Gamma, self).fit(X, B, T, W, p=1)
