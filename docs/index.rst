Welcome to squmfit's documentation
=====================================

This is the documentation for `squmfit`_, a Python library for
convenient non-linear least-squares fitting of multiple analytical
models to data. This library is similar to the excellent `lmfit`_
package but is a fresh implementation designed with support for global
fitting of multiple curves with parameter tying.

You may want to start at :doc:`getting_started` or go directly to the
:mod:`squmfit` module documentation.

.. _squmfit: http://github.com/bgamari/squmfit
.. _lmfit: http://cars9.uchicago.edu/software/python/lmfit/

Contents:

.. toctree::
   :maxdepth: 2

   getting_started


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

Let's say that we have a noisy exponential decay that we want model,

    >>> import numpy as np
    >>> noise_std = 0.1
    >>> xs = np.arange(4000)
    >>> ys = 4 * np.exp(-xs / 400) + np.random.normal(scale=noise_std, size=xs.shape)

To fit this we first define the model which we believes describes our
data (or use one of the models provided in the :mod:`squmfit.models`
module,

    >>> def exponential_model(t, amplitude, rate):
    >>>     return amplitude * np.exp(-t * rate)
    >>> ExpModel = Model(exponential_model)

We then create a :class:`Fit` object, which we can then add our curve
to,

    >>> from squmfit import Fit
    >>> fit = Fit()
    >>> fit

Fitting
--------

Fitting begins with the :class:`Fit` class which is used to define
the data sets and models to be fit and ultimately 


Defining a Model
-----------------

Defining a model is simply a matter of passing the function to the
:class:`Model` constructor,

    >>> def exponential_model(t, amplitude, rate):
    >>>     return amplitude * np.exp(-t * rate)
    >>> ExpModel = Model(exponential_model)

.. autoclass:: squmfit.Model
