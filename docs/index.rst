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

In contrast to `lmfit`_, `squmfit` treats the free parameters of the
the fit as first-class objects. This allows models to be built up,
added, multiplied, and generally treated as standard Python
expressions. Moreover, no assumptions are imposed regarding how
parameters interact, allowing fitting of a common set of parameters
over several data sets.

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

    >>> import squmfit
    >>> def exponential_model(t, amplitude, rate):
    >>>     return amplitude * np.exp(-t * rate)

We then create a :class:`Fit` object which we will use to define the
parameters and objective of our fit,

    >>> fit = squmfit.Fit()

Say we want to fit this model to our generated ``ys``, allowing
both the ``amplitude`` and ``rate`` parameters to vary. We first need
to tell ``squmfit`` about these parameters,

    >>> amp = fit.param('amp', initial=100)
    >>> tau = fit.param('tau', initial=100)

Now we can use ``amp`` and ``tau`` as normal Python variables,

    >>> model = exponential_model(xs, amp, 1. / tau)

Alternatively we could simply do away with the function altogether,

    >>> model = amp * np.exp(-t / tau)

Note how we can write expressions involving parameters with the usual
Python arithmetic operations, such as ``1. / tau``, greatly
simplifying model composition.

Next we add our curve to our ``Fit``,
specifying the model to which we wish to fit along with some weights
(taking care to avoid division-by-zero, of course),

    >>> weights = np.zeros_like(ys, dtype='f')
    >>> weights[ys > 0] = 1 / np.sqrt(ys[ys > 0])
    >>> fit.add_curve('curve1', model, ys, weights=weights)

Finally we can run our fit and poke around at the results,

    >>> res = fit.fit()
    >>> print res.params
    {'amp': 403.01725751512635, 'tau': 393.19866908823133}
    >>> print res.curves['curve1'].reduced_chi_sqr
    0.949579885697

``res`` is a :class:`FitResult` object, which contains a variety of
information about the fit, including the residuals, covariances, and
fit values.

``squmfit`` has a variety of options for presenting the results of a
fit. :mod:`squmfit.pretty` has several utilities for producing a
quantitative summary of the fit. For instance,
:func:`squmfit.pretty.markdown_fit_result` will produce a Markdown
document describing the fit parameters and various goodness-of-fit
metrics. If you use IPython Notebook,
:func:`squmfit.pretty.ipynb_fit_result` can be used to generate a
presentation that can be rendered in rich HTML within the notebook,

Finally, :mod:`squmfit.plot` can be used to plot fits and residuals.


How does it work?
~~~~~~~~~~~~~~~~~

Expressions in ``squmfit`` are represented by the :class:`Expr`
class. An ``Expr`` captures an abstract syntax tree that describes
*how* a result can be computed. The elementary expressions could
represent a fitted parameter (e.g. ``amp`` in the example above), or
an expression depending upon a fitted parameter. The ``Expr`` class
implements basic arithmetic operations (e.g. ``__add__``) and numpy's
ufuncs (e.g. ``sqrt``), allowing it to be treated as a scalar or an array.

Of course, some functions require more structure beyond the operations
supported by ``Expr`` evaluate their result. In this case, you can
tell ``squmfit`` that you want the function arguments to be evaluated
before they are provided to your function with the :func:`model`
decorator,

    >>> @squmfit.model
    >>> def sum_odds(vec):
    >>>     return vec[1::2].sum()

In this case, we could invoke ``sum_odds`` with an :class:`Expr`,
which ``squmfit`` would automatically evaluate. It would then evaluate
``sum_odds`` with the value of the expression, and pack the result back into an
:class:`Expr`.

.. autofunction:: squmfit.model
   :noindex:

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

Functional interpretation
-------------------------

The architecture of ``squmfit`` is inspired by patterns used widely in
Haskell and other functional languages. The :class:`Expr` class is an
applicative functor and reader monad having access to an environment
containing packed parameter values.
