Convoys
=======

Convoys is a simple library that fits a few statistical model useful for modeling time-lagged conversion rates.
You can check out the `Github repo <https://github.com/better/convoys>`_ for source code and more things!

Installation
------------

The easiest way right now is to install it straight from Github using Pip:

::

    pip install -e git://github.com/better/convoys#egg=convoys


Motivation
----------

Predicting conversions is a really important problem for ecommerce, online advertising, and many other applications.
In many cases when conversions are relatively quick, you can measure the response (e.g. whether the user bought the product) and use models like `logistic regression <https://en.wikipedia.org/wiki/Logistic_regression>`_ to predict conversion.

If conversions have substantial time lag (which is often the case) it gets a bit trickier.
You know who converted, but if someone did not convert, they might still convert in the future.
In other words, conversions are observed, but non-conversions are not observed.

The "hacky" way to address this is to define conversion as *conversion at time X*.
This turns the problem into a simple binary classification problem, but the drawback is you are losing data by *binarizing* it.
First of all, you can not learn from users that are younger than X.
You also can not learn from users that convert *after* X.
For an excellent introduction to this problem (and distributions like the `Weibull distribution <https://en.wikipedia.org/wiki/Weibull_distribution>`_), here's a blog post about `implementing a recurrent neural network to predict churn <https://ragulpr.github.io/2016/12/22/WTTE-RNN-Hackless-churn-modeling/>`_.

Survival analysis to the rescue
-------------------------------

Luckily, there is a somewhat similar field called `survival analysis <https://en.wikipedia.org/wiki/Survival_analysis>`_.
It introduces the concept of *censored data*, which is data that we have not observed yet.
`Lifelines <http://lifelines.readthedocs.io/en/latest/>`_ is a great Python package with excellent documentation that implements many classic models for survival analysis.

Unfortunately, survival analysis assumes that *everyone dies* in the end.
This is not a realistic assumption when you model conversion rates since not everyone will convert, even given infinite amount of time.
Typically conversion rates stabilize at some fraction eventually.

Predicting lagged conversions
-----------------------------

It turns out we can model conversions by essentially thinking of conversions as a logistic regression model *multiplied by* a distribution over time which is usually a `Weibull distribution <https://en.wikipedia.org/wiki/Weibull_distribution>`_ or something similar.
Convoys implements a few of these models with some utility functions for data conversion and plotting.
