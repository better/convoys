from convoys import autograd_scipy_monkeypatch  # NOQA
import autograd
from autograd_gamma import gammainc
from deprecated.sphinx import deprecated
import emcee
from autograd import numpy
from scipy.special import gammaincinv
from autograd.scipy.special import expit, gammaln
from autograd.numpy import isnan, exp, dot, log, sum, array, repeat
import progressbar
import scipy.optimize
import warnings

def generalized_gamma_loss(x, X, B, T, W, fix_k, fix_p,
                           hierarchical, flavor, callback=None):
    # parameters for this distribution is p, k, lambd
    k = exp(x[0]) if fix_k is None else fix_k # x[0], x[1], x
    p = exp(x[1]) if fix_p is None else fix_p
    log_sigma_alpha = x[2]
    log_sigma_beta = x[3]
    a = x[4]
    b = x[5]
    n_features = int((len(x)-6)/2)
    alpha = x[6:6+n_features]
    beta = x[6+n_features:6+2*n_features]
    lambd = exp(dot(X, alpha)+a) # lambda = exp(\alpha+a),  X shape is N * n_groups, alpha is \n_features * 1 
    # PDF: p*lambda^(k*p) / gamma(k) * t^(k*p-1) * exp(-(x*lambda)^p), log pdf has shape (N,)
    log_pdf = log(p) + (k*p) * log(lambd) - gammaln(k) \
              + (k*p-1) * log(T) - (T*lambd)**p
    cdf = gammainc(k, (T*lambd)**p)
    # cdf has shape (N,)

    if flavor == 'logistic':  # Log-likelihood with sigmoid
        c = expit(dot(X, beta)+b) # fit one beta for each group , x has shape(N, n_group), beta has shape (n_group, ), c has shape (N,)
        # dot product shape is (N,)
        # beta has shape (n_group,)
        LL_observed = log(c) + log_pdf
        LL_censored = log((1 - c) + c * (1 - cdf))
    elif flavor == 'linear':  # L2 loss, linear
        c = dot(X, beta)+b
        LL_observed = -(1 - c)**2 + log_pdf
        LL_censored = -(c*cdf)**2
    print("shape of result before sum", (W * B * LL_observed + W * (1 - B) * LL_censored).shape)
    LL_data = numpy.sum(
        W * B * LL_observed +
        W * (1 - B) * LL_censored, 0)   #
    if hierarchical:
        # Hierarchical model with sigmas ~ invgamma(1, 1)
        LL_prior_a = -4*log_sigma_alpha - 1/exp(numpy.clip(log_sigma_alpha, -tol_upper,tol_upper))**2 \
                     - dot(alpha, alpha) / (2*exp(numpy.clip(log_sigma_alpha, -tol_upper, tol_upper))**2) \
                     - n_features*log_sigma_alpha
        LL_prior_b = -4*log_sigma_beta - 1/exp(numpy.clip(log_sigma_beta,-tol_upper,tol_upper))**2 \
                     - dot(beta, beta) / (2*exp(numpy.clip(log_sigma_beta,-tol_upper,tol_upper))**2) \
                     - n_features*log_sigma_beta
        LL = LL_prior_a + LL_prior_b + LL_data
    else:
        LL = LL_data

    if isnan(LL):
        return -numpy.inf
    if callback is not None:
        callback(LL)
    # loss is a constant, not a vector
    return LL


    

def double_hierarchy_weibull_loss(x, X, B, T, W, fix_k, fix_p,
                           hierarchical, max_std_alpha=0.5, max_std_beta=0.5, tol_lower=1e-2,tol_upper = 1e1,callback=None):
    # parameters for this distribution is p, k, lambd
    k = exp(x[0]) if fix_k is None else fix_k # x[0], x[1], x
    p = exp(x[1]) if fix_p is None else fix_p
    _,n_group1,n_group2 = X.shape
    log_sigma_alpha_1 = x[2] # TODO, sample from invGamma(1,1)
    log_sigma_beta_1 = x[3]
    # log sigma < = k
    # sigma <= exp(k)
    # I want sigma <= 0.1,
    # so log sigma <= -1
    log_sigma_alpha_2 = min(x[4], log(max_std_alpha))# restricting the variance to make sure it works 
    log_sigma_beta_2= min(x[5], log(max_std_beta))
    a = x[6]
    b = x[7]
    p = numpy.clip(tol_lower,tol_upper)
    lambd = numpy.clip(lambd,tol_lower,tol_upper)
    k = numpy.clip(k,tol_lower,tol_upper)
    alpha_1 = x[8:8+n_group1] # length=n_group1
    beta_1 = x[8+n_group1:8+2*n_group1] # length=n_group1
    n_group=n_group1*n_group2
    alpha_2 = x[8+2*n_group1:8+2*n_group1+n_group].reshape(n_group1,n_group2) # length = n_group1*n_group2
    beta_2 = x[8+2*n_group1+n_group:8+2*n_group1+2*n_group].reshape(n_group1,n_group2) # length = n_group1*n_group2
    # If X is (N, G1,G2), alpha_2 is (G1,G2), then X2 *alpha_2 is (N, G1,G2), call this Z_2
    
    # prevent overflow
    lambd = exp(numpy.clip(numpy.sum(X * alpha_2 , axis=(1,2))+a, -tol_upper,tol_upper))
    

    # PDF: p*lambda^(k*p) / gamma(k) * t^(k*p-1) * exp(-(x*lambda)^p)
    log_pdf = log(p) + (k*p) * log(lambd) - gammaln(k) + (k*p-1) * log(T) - (T*lambd)**p
    cdf = gammainc(k, numpy.clip((T*lambd)**p,-tol_upper, tol_upper))
    # logistic part 
    c = expit(numpy.sum(X * beta_2, axis=(1,2))+b)
    LL_observed = log(c) + log_pdf
    LL_censored = log((1 - c) + c * (1 - cdf))
    # W should be sample weights, it is not implemented right now, let's delete it 
    LL_data = sum(B * LL_observed +(1 - B) * LL_censored)
    
    #TODO: rewrite this loss
    if hierarchical:
        # Hierarchical model with sigmas ~ invgamma(1, 1)
        # alpha_2 is (G1,G2) -> 
        # alpha_1 is (G1,) -> make this (n_group1,1)
        # diff_alpha = alpha_2 - alpha_1 -> this is therefore (n_group1,n_group2)
        diff_alpha_sq = numpy.sum((alpha_2-numpy.expand_dims(alpha_1,axis=-1))**2)
        diff_beta_sq = numpy.sum((beta_2-numpy.expand_dims(beta_1,axis=-1))**2)
        log_sigma_alpha_1 = numpy.clip(log_sigma_alpha_1,-tol_upper,tol_upper)
        log_sigma_alpha_2 = numpy.clip(log_sigma_alpha_2,-tol_upper,tol_upper)
        log_sigma_beta_1 = numpy.clip(log_sigma_beta_1,-tol_upper,tol_upper)
        log_sigma_beta_2 = numpy.clip(log_sigma_beta_2,-tol_upper,tol_upper)

        LL_prior_a = -4*log_sigma_alpha_1 - 1/exp(log_sigma_alpha_1)**2 \
                     - dot(alpha_1, alpha_1) / (2*exp(log_sigma_alpha_1)**2) \
                     - n_group1*log_sigma_alpha_1 - \
                     diff_alpha_sq/(2*exp(log_sigma_alpha_2)**2) \
                     -n_group*log_sigma_alpha_2
        
        LL_prior_b = -4*log_sigma_beta_1 - 1/exp(log_sigma_beta_1)**2 \
                     - dot(beta_1, beta_1) / (2*exp(log_sigma_beta_1)**2) \
                     - n_group1*log_sigma_beta_1 - \
                     diff_beta_sq/(2*exp(log_sigma_beta_2)**2) -n_group*log_sigma_beta_2
        LL = LL_prior_a + LL_prior_b + LL_data
    else:
        LL = LL_data
    if isnan(LL):
        return -numpy.inf
    if callback is not None:
        callback(LL)
    return LL