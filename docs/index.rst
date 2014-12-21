.. currentmodule:: squmfit

Welcome to squmfit's documentation
=====================================

This is the documentation for `squmfit`_ (pronounced "squim-fit"), a
Python library for convenient non-linear least-squares fitting of
multiple analytical models to data. This library is similar to the
excellent `lmfit`_ package but is a fresh implementation designed with
support for global fitting of multiple curves with parameter tying.

You may want to start at :doc:`getting_started` or go directly to the
:mod:`squmfit` module documentation.

.. _squmfit: http://github.com/bgamari/squmfit
.. _lmfit: http://cars9.uchicago.edu/software/python/lmfit/

Contents:

.. toctree::
   :maxdepth: 2

   getting_started
   reference

Overview
---------

`squmfit` is a general-purpose library for non-linear least-squared
curve fitting. The library is designed to enable modular models which
can be easily composed to describe complex data sets. Moreover, the
library supports the ability to simultaneously fit multiple data sets,
each with its own model, over a common set of parameters.

In constrast to `lmfit`_, `squmfit` treats the free parameters of the
the fit as first-class objects.

A simple example
~~~~~~~~~~~~~~~~~

Let's say that we have a exponential decay with Poissonian noise,

    >>> import numpy as np
    >>> noise_std = 0.1
    >>> xs = np.arange(4000)
    >>> ys = np.random.poisson(400 * np.exp(-xs / 400.))

We first define the functional form which we believe describes our
data (or use one of the models provided in the :mod:`squmfit.models`
module),

    >>> from squmfit import Model
    >>> def exponential_model(t, amplitude, rate):
    >>>     return amplitude * np.exp(-t * rate)
    >>> ExpModel = Model(exponential_model)

We then create a :class:`Fit` object which we will use to define the
parameters and objective of our fit,

    >>> from squmfit import Fit
    >>> fit = Fit()

Say we want to fit this model to our generated ``ys``, allowing
both the ``amplitude`` and ``rate`` parameters to vary. This can be
defined as,

    >>> amp = fit.param('amp', initial=100)
    >>> tau = fit.param('tau', initial=100)
    >>> model = ExpModel(xs, amp, 1. / tau)

Note how we can write expressions involving parameters, such as ``1. /
tau``, greatly simplifying parameter specification.  Next we add our
curve to our ``Fit``, specifying the model to which we wish to fit
along with some weights (taking care to avoid division-by-zero, of
course),

    >>> weights = np.zeros_like(ys, dtype='f')
    >>> weights[ys > 0] = 1 / np.sqrt(ys[ys > 0])
    >>> fit.add_curve('a', model, ys, weights=weights)

Finally we can run our fit and poke around at the results,

    >>> res = fit.fit()
    >>> print res.params
    {'amp': 403.01725751512635, 'tau': 393.19866908823133}
    >>> print res.curves['a'].reduced_chi_sqr
    0.949579885697

Fitting
--------

Fitting begins with the :class:`Fit` class which is used to define
the data sets and models to be fit and ultimately

.. autoclass:: squmfit.Fit
   :noindex:

Inspecting fit results
----------------------

.. autoclass:: squmfit.FitResult
   :noindex:

Defining a Model
-----------------

Defining a model is simply a matter of passing the function to the
:class:`Model` constructor,

    >>> def exponential_model(t, amplitude, rate):
    >>>     return amplitude * np.exp(-t * rate)
    >>> ExpModel = Model(exponential_model)

.. autoclass:: squmfit.Model
   :noindex:
