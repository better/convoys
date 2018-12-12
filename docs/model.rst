Model
-----

In this section we'll walk through the entire model specification. Convoys tries to optimize the total likelihood of observing all the data given the model, optionally also using `Markov chain Monte Carlo <https://en.wikipedia.org/wiki/Markov_chain_Monte_Carlo>`_ to sample from the posterior in order to generate uncertainty estimate.

Let's say :math:`f(t, x)` is a probability density function over time of when a conversion happen, where :math:`t > 0` and :math:`x` is some feature vector. Note that :math:`f(t, x)` denote the probability density of conversion *conditional on the conversion happening*. This means that :math:`\int_0^\infty f(t, x) dt = 1`.

We use :math:`F(t, x)` to denote the cumulative density function of :math:`f(t, x)`. The definitions of :math:`f, F` depends on which model we use, so we can plug in a Weibull distribution or a Gamma distribution or something else.

We now introduce :math:`g(t, x) = c(x)f(t, x)` and :math:`G(t, x) = c(x)F(t, x)` where :math:`c(x)` denotes the probability of conversion at all. This means that :math:`\lim_{t \rightarrow \infty} G(t, x) = c(x)`.

For each data point, we have three different cases:

1. We observed conversion. In that case, the probability density is :math:`g(t, x)`
2. We didn't observe conversion yet and the current time is :math:`t`. There are two sub-cases

   a. No conversion will ever happen. The probability density is :math:`1 - c(x)`.
   b. Conversion will happen at some point in the future. The probability is :math:`c(x)(1 - F(x, t))`.

Multiplying all these probabilities and taking the logarithm gives the total log-likelihood of the data given the model. See documentation for :class:`GeneralizedGamma` for some more information about the exact math. There is also some regularization that wasn't covered here.
