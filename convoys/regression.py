from convoys import autograd_scipy_monkeypatch  # NOQA
import autograd
from autograd_gamma import gammainc
from deprecated.sphinx import deprecated
import emcee
import numpy
from scipy.special import gammaincinv
from autograd.scipy.special import expit, gammaln
from autograd.numpy import isnan, exp, dot, log, sum
import progressbar
import scipy.optimize
import warnings


__all__ = ['Exponential',
           'Weibull',
           'Gamma',
           'GeneralizedGamma']


def generalized_gamma_loss(x, X, B, T, W, fix_k, fix_p,
                           hierarchical, flavor, callback=None):
    k = exp(x[0]) if fix_k is None else fix_k
    p = exp(x[1]) if fix_p is None else fix_p
    log_sigma_alpha = x[2]
    log_sigma_beta = x[3]
    a = x[4]
    b = x[5]
    n_features = int((len(x)-6)/2)
    alpha = x[6:6+n_features]
    beta = x[6+n_features:6+2*n_features]
    lambd = exp(dot(X, alpha)+a)

    # PDF: p*lambda^(k*p) / gamma(k) * t^(k*p-1) * exp(-(x*lambda)^p)
    log_pdf = log(p) + (k*p) * log(lambd) - gammaln(k) \
              + (k*p-1) * log(T) - (T*lambd)**p
    cdf = gammainc(k, (T*lambd)**p)

    if flavor == 'logistic':  # Log-likelihood with sigmoid
        c = expit(dot(X, beta)+b)
        LL_observed = log(c) + log_pdf
        LL_censored = log((1 - c) + c * (1 - cdf))
    elif flavor == 'linear':  # L2 loss, linear
        c = dot(X, beta)+b
        LL_observed = -(1 - c)**2 + log_pdf
        LL_censored = -(c*cdf)**2

    LL_data = sum(
        W * B * LL_observed +
        W * (1 - B) * LL_censored, 0)

    if hierarchical:
        # Hierarchical model with sigmas ~ invgamma(1, 1)
        LL_prior_a = -4*log_sigma_alpha - 1/exp(log_sigma_alpha)**2 \
                     - dot(alpha, alpha) / (2*exp(log_sigma_alpha)**2) \
                     - n_features*log_sigma_alpha
        LL_prior_b = -4*log_sigma_beta - 1/exp(log_sigma_beta)**2 \
                     - dot(beta, beta) / (2*exp(log_sigma_beta)**2) \
                     - n_features*log_sigma_beta
        LL = LL_prior_a + LL_prior_b + LL_data
    else:
        LL = LL_data

    if isnan(LL):
        return -numpy.inf
    if callback is not None:
        callback(LL)
    return LL


class RegressionModel(object):
    pass


class GeneralizedGamma(RegressionModel):
    ''' Generalization of Gamma, Weibull, and Exponential

    :param ci: boolean, defaults to False. Whether to use MCMC to
        sample from the posterior so that a confidence interval can be
        estimated later (see :meth:`predict`).
    :param hierarchical: boolean denoting whether we have a (Normal) prior
        on the alpha and beta parameters to regularize. The variance of
        the normal distribution is in itself assumed to be an inverse
        gamma distribution (1, 1).
    :param flavor: defaults to logistic. If set to 'linear', then an
        linear model is fit, where the beta params will be completely
        additive. This creates a much more interpretable model, with some
        minor loss of accuracy.

    This mostly follows the `Wikipedia article
    <https://en.wikipedia.org/wiki/Generalized_gamma_distribution>`_, although
    our notation is slightly different. Also see `this paper
    <http://data.princeton.edu/pop509/ParametricSurvival.pdf>`_ for an overview.

    **Shape of the probability function**

    The cumulative density function is:

    :math:`F(t) = P(k, (t\\lambda)^p)`

    where :math:`P(a, x) = \\gamma(a, x) / \\Gamma(a)` is the lower regularized
    incomplete gamma function.
    :math:`\\gamma(a, x)` is the incomplete gamma function and :math:`\\Gamma(a)`
    is the standard gamma function.

    The probability density function is:

    :math:`f(t) = p\\lambda^{kp} t^{kp-1} \\exp(-(t\\lambda)^p) / \\Gamma(k)`

    **Modeling conversion rate**

    Since our goal is to model the conversion rate, we assume the conversion
    rate converges to a final value

    :math:`c = \\sigma(\\mathbf{\\beta^Tx} + b)`

    where :math:`\\sigma(z) = 1/(1+e^{-z})` is the sigmoid function,
    :math:`\\mathbf{\\beta}` is an unknown vector we are solving for (with
    corresponding  intercept :math:`b`), and :math:`\\mathbf{x}` are the
    feature vector (inputs).

    We also assume that the rate parameter :math:`\\lambda` is determined by

    :math:`\\lambda = exp(\\mathbf{\\alpha^Tx} + a)`

    where :math:`\\mathrm{\\alpha}` is another unknown vector we are
    trying to solve for (with corresponding intercept :math:`a`).

    We also assume that the :math:`\\mathbf{\\alpha}, \\mathbf{\\beta}`
    vectors have a normal distribution

    :math:`\\alpha_i \\sim \\mathcal{N}(0, \\sigma_{\\alpha})`,
    :math:`\\beta_i \\sim \\mathcal{N}(0, \\sigma_{\\beta})`

    where hyperparameters :math:`\\sigma_{\\alpha}^2, \\sigma_{\\beta}^2`
    are drawn from an inverse gamma distribution

    :math:`\\sigma_{\\alpha}^2 \\sim \\text{inv-gamma}(1, 1)`,
    :math:`\\sigma_{\\beta}^2 \\sim \\text{inv-gamma}(1, 1)`

    **List of parameters**

    The full model fits vectors :math:`\\mathbf{\\alpha, \\beta}` and scalars
    :math:`a, b, k, p, \\sigma_{\\alpha}, \\sigma_{\\beta}`.

    **Likelihood and censorship**

    For entries that convert, the contribution to the likelihood is simply
    the probability density given by the probability distribution function
    :math:`f(t)` times the final conversion rate :math:`c`.

    For entries that *did not* convert, there is two options. Either the
    entry will never convert, which has probability :math:`1-c`. Or,
    it will convert at some later point that we have not observed yet,
    with probability given by the cumulative density function
    :math:`F(t)`.

    **Solving the optimization problem**

    To find the MAP (max a posteriori), `scipy.optimize.minimize
    <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html#scipy.optimize.minimize>`_
    with the SLSQP method.

    If `ci == True`, then `emcee <http://dfm.io/emcee/current/>`_ is used
    to sample from the full posterior in order to generate uncertainty
    estimates for all parameters.
    '''
    def __init__(self, ci=False, fix_k=None, fix_p=None, hierarchical=True,
                 flavor='logistic'):
        self._ci = ci
        self._fix_k = fix_k
        self._fix_p = fix_p
        self._hierarchical = hierarchical
        self._flavor = flavor

    def fit(self, X, B, T, W=None):
        '''Fits the model.

        :param X: numpy matrix of shape :math:`k \\cdot n`
        :param B: numpy vector of shape :math:`n`
        :param T: numpy vector of shape :math:`n`
        :param W: (optional) numpy vector of shape :math:`n`
        '''

        if W is None:
            W = numpy.ones(len(X))
        X, B, T, W = (Z if type(Z) == numpy.ndarray else numpy.array(Z)
                      for Z in (X, B, T, W))
        keep_indexes = (T > 0) & (B >= 0) & (B <= 1) & (W >= 0)
        if sum(keep_indexes) < X.shape[0]:
            n_removed = X.shape[0] - sum(keep_indexes)
            warnings.warn('Warning! Removed %d/%d entries from inputs where '
                          'T <= 0 or B not 0/1 or W < 0' % (n_removed, len(X)))
            X, B, T, W = (Z[keep_indexes] for Z in (X, B, T, W))
        n_features = X.shape[1]

        # scipy.optimize and emcee forces the the parameters to be a vector:
        # (log k, log p, log sigma_alpha, log sigma_beta,
        #  a, b, alpha_1...alpha_k, beta_1...beta_k)
        # Generalized Gamma is a bit sensitive to the starting point!
        x0 = numpy.zeros(6+2*n_features)
        x0[0] = +1 if self._fix_k is None else log(self._fix_k)
        x0[1] = -1 if self._fix_p is None else log(self._fix_p)
        args = (X, B, T, W, self._fix_k, self._fix_p,
                self._hierarchical, self._flavor)

        # Set up progressbar and callback
        bar = progressbar.ProgressBar(widgets=[
                progressbar.Variable('loss', width=15, precision=9), ' ',
                progressbar.BouncingBar(), ' ',
                progressbar.Counter(width=6),
                ' [', progressbar.Timer(), ']'])

        def callback(LL, value_history=[]):
            value_history.append(LL)
            bar.update(len(value_history), loss=LL)

        # Define objective and use automatic differentiation
        f = lambda x: -generalized_gamma_loss(x, *args, callback=callback)
        jac = autograd.grad(lambda x: -generalized_gamma_loss(x, *args))

        # Find the maximum a posteriori of the distribution
        res = scipy.optimize.minimize(f, x0, jac=jac, method='SLSQP',
                                      options={'maxiter': 9999})
        if not res.success:
            raise Exception('Optimization failed with message: %s' %
                            res.message)
        result = {'map': res.x}

        # TODO: should not use fixed k/p as search parameters
        if self._fix_k:
            result['map'][0] = log(self._fix_k)
        if self._fix_p:
            result['map'][1] = log(self._fix_p)

        # Make sure we're in a local minimum
        gradient = jac(result['map'])
        gradient_norm = numpy.dot(gradient, gradient)
        if gradient_norm >= 1e-2 * len(X):
            warnings.warn('Might not have found a local minimum! '
                          'Norm of gradient is %f' % gradient_norm)

        # Let's sample from the posterior to compute uncertainties
        if self._ci:
            dim, = res.x.shape
            n_walkers = 5*dim
            sampler = emcee.EnsembleSampler(
                nwalkers=n_walkers,
                ndim=dim,
                log_prob_fn=generalized_gamma_loss,
                args=args,
            )
            mcmc_initial_noise = 1e-3
            p0 = [result['map'] + mcmc_initial_noise * numpy.random.randn(dim)
                  for i in range(n_walkers)]
            n_burnin = 100
            n_steps = numpy.ceil(2000. / n_walkers)
            n_iterations = n_burnin + n_steps

            bar = progressbar.ProgressBar(max_value=n_iterations, widgets=[
                    progressbar.Percentage(), ' ', progressbar.Bar(),
                    ' %d walkers [' % n_walkers,
                    progressbar.AdaptiveETA(), ']'])
            for i, _ in enumerate(sampler.sample(p0, iterations=n_iterations)):
                bar.update(i+1)
            result['samples'] = sampler.chain[:, n_burnin:, :] \
                                       .reshape((-1, dim)).T
            if self._fix_k:
                result['samples'][0, :] = log(self._fix_k)
            if self._fix_p:
                result['samples'][1, :] = log(self._fix_p)

        self.params = {k: {
            'k': exp(data[0]),
            'p': exp(data[1]),
            'a': data[4],
            'b': data[5],
            'alpha': data[6:6+n_features].T,
            'beta': data[6+n_features:6+2*n_features].T,
        } for k, data in result.items()}

    def _predict(self, params, x, t):
        lambd = exp(dot(x, params['alpha'].T) + params['a'])
        if self._flavor == 'logistic':
            c = expit(dot(x, params['beta'].T) + params['b'])
        elif self._flavor == 'linear':
            c = dot(x, params['beta'].T) + params['b']
        M = c * gammainc(
            params['k'],
            (t*lambd)**params['p'])

        return M        

    def predict_posteriori(self, x, t):
        ''' Returns the trace samples generated via the MCMC steps.

        Requires the model to be fit with `ci = True`.'''
        x = numpy.array(x)
        t = numpy.array(t)
        assert self._ci
        params = self.params['samples']
        t = numpy.expand_dims(t, -1)
        return self._predict(params, x, t)

    def predict_ci(self, x, t, ci=0.8):
        '''Works like :meth:`predict` but produces a confidence interval.

        Requires the model to be fit with `ci = True`. The return value
        will contain one more dimension than for :meth:`predict`, and
        the last dimension will have size 3, containing the mean, the
        lower bound of the confidence interval, and the upper bound of
        the confidence interval.
        '''
        M = self.predict_posteriori(x, t)
        y = numpy.mean(M, axis=-1)
        y_lo = numpy.percentile(M, (1-ci)*50, axis=-1)
        y_hi = numpy.percentile(M, (1+ci)*50, axis=-1)
        return numpy.stack((y, y_lo, y_hi), axis=-1)

    def predict(self, x, t):
        '''Returns the value of the cumulative distribution function
        for a fitted model.

        :param x: feature vector (or matrix)
        :param t: time
        '''
        params = self.params['map']
        x = numpy.array(x)
        t = numpy.array(t)
        return self._predict(params, x, t)

    def rvs(self, x, n_curves=1, n_samples=1, T=None):
        ''' Samples values from this distribution

        T is optional and means we already observed non-conversion until T
        '''
        assert self._ci  # Need to be fit with MCMC
        if T is None:
            T = numpy.zeros((n_curves, n_samples))
        else:
            assert T.shape == (n_curves, n_samples)
        B = numpy.zeros((n_curves, n_samples), dtype=numpy.bool)
        C = numpy.zeros((n_curves, n_samples))
        params = self.params['samples']
        for i, j in enumerate(numpy.random.randint(len(params['k']),
                                                   size=n_curves)):
            k = params['k'][j]
            p = params['p'][j]
            lambd = exp(dot(x, params['alpha'][j]) + params['a'][j])
            c = expit(dot(x, params['beta'][j]) + params['b'][j])
            z = numpy.random.uniform(size=(n_samples,))
            cdf_now = c * gammainc(
                k,
                numpy.multiply.outer(T[i], lambd)**p)  # why is this outer?
            adjusted_z = cdf_now + (1 - cdf_now) * z
            B[i] = (adjusted_z < c)
            y = adjusted_z / c
            w = gammaincinv(k, y)
            # x = (t * lambd)**p
            C[i] = w**(1./p) / lambd
            C[i][~B[i]] = 0

        return B, C

    @deprecated(version='0.1.8',
                reason='Use :meth:`predict` or :meth:`predict_ci` instead.')
    def cdf(self, x, t, ci=False):
        '''Returns the predicted values.'''
        if ci:
            return self.predict_ci(x, t)
        else:
            return self.predict(x, t)

    @deprecated(version='0.1.8', reason='Use :meth:`predict_posteriori` instead.')
    def cdf_posteriori(self, x, t):
        '''Returns the a posterior distribution of the predicted values.'''
        return self.predict_posteriori(x, t)


class Exponential(GeneralizedGamma):
    ''' Specialization of :class:`.GeneralizedGamma` where :math:`k=1, p=1`.

    The cumulative density function is:

    :math:`F(t) = 1 - \\exp(-t\\lambda)`

    The probability density function is:

    :math:`f(t) = \\lambda\\exp(-t\\lambda)`

    The exponential distribution is the most simple distribution.
    From a conversion perspective, you can interpret it as having
    two competing final states where the probability of transitioning
    from the initial state to converted or dead is constant.

    See documentation for :class:`GeneralizedGamma`.'''
    def __init__(self, *args, **kwargs):
        kwargs.update(dict(fix_k=1, fix_p=1))
        super(Exponential, self).__init__(*args, **kwargs)


class Weibull(GeneralizedGamma):
    ''' Specialization of :class:`.GeneralizedGamma` where :math:`k=1`.

    The cumulative density function is:

    :math:`F(t) = 1 - \\exp(-(t\\lambda)^p)`

    The probability density function is:

    :math:`f(t) = p\\lambda(t\\lambda)^{p-1}\\exp(-(t\\lambda)^p)`

    See documentation for :class:`GeneralizedGamma`.'''
    def __init__(self, *args, **kwargs):
        kwargs.update(dict(fix_k=1))
        super(Weibull, self).__init__(*args, **kwargs)


class Gamma(GeneralizedGamma):
    ''' Specialization of :class:`.GeneralizedGamma` where :math:`p=1`.

    The cumulative density function is:

    :math:`F(t) = P(k, t\\lambda)`

    where :math:`P(a, x) = \\gamma(a, x) / \\Gamma(a)` is the lower regularized
    incomplete gamma function.

    The probability density function is:

    :math:`f(t) = \\lambda^k t^{k-1} \\exp(-x\\lambda) / \\Gamma(k)`

    See documentation for :class:`GeneralizedGamma`.'''
    def __init__(self, *args, **kwargs):
        kwargs.update(dict(fix_p=1))
        super(Gamma, self).__init__(*args, **kwargs)
