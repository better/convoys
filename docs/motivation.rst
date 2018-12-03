Motivation
==========

A perfectly valid question is: Convoys really only implements one model: a generalized Gamma distribution multiplied by basically logistic regression. That seems like a very specialized distribution. In order to justify this choice, let's first look at a handful of conversion charts from real data at `Better <https://better.com>`_:

.. image:: images/conversion.gif

The legend, labels of the axes, and title are all removed so that no business secrets are revealed. The solid lines with shaded area are all the generalized Gamma fits, whereas the dashed lines are the Kaplan-Meier fits. Note that the fit is very good! In fact, we have observed that almost any conversion metric can be modeled reasonably well with the generalized Gamma model (multiplied by logistic regression).

Empirically, this model seems to hold up pretty well.

Some more mathematical justification
------------------------------------

A simple toy problem also demonstrates why we would expect to get a time-dependent distribution (like the Exponential distribution) multiplied by a logistic function. Consider the `continuous-time Markov chain <https://en.wikipedia.org/wiki/Markov_chain#Continuous-time_Markov_chain>`_ with three states: undecided, converted, or died.

.. image:: images/convoys-markov-chain.png
   :height: 300px
   :width: 300px

Everyone starts out "undecided" but either converts or dies. However, we *only observe the conversions,* not the deaths.

We can solve for the distribution by thinking of this as a partial differential equation. The distribution over the three states will be given as :math:`P(t) = e^{tA}` where :math:`A` is the `generator matrix <https://en.wikipedia.org/wiki/Transition_rate_matrix>`_

.. math::
   A = \left( {
   \begin{array}{ccc}
   -(\lambda_1 + \lambda_2) & \lambda_1 & \lambda_2 \\
   0 & 0 & 0 \\
   0 & 0 & 0
   \end{array}
   } \right)
