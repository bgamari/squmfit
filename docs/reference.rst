.. currentmodule:: squmfit

API Reference
=============

Building models
---------------

The :class:`Expr` class
~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: Expr
   :members:
   :member-order: bysource


Evaluating Exprs
~~~~~~~~~~~~~~~~

.. autofunction:: model


The :class:`FittedParam` class
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: squmfit.parameter.FittedParam
   :members:

Performing fits
---------------

The :class:`Fit` class
~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: Fit
   :members:

The :class:`FitResult` class
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: FitResult
   :members:

The :class:`CurveResult` class
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: CurveResult
   :members:

Rendering fit results
---------------------

Text
~~~~

.. automodule:: squmfit.pretty
   :members:

Plotting
~~~~~~~~
.. automodule:: squmfit.plot
   :members:

Helper classes
--------------

These classes represent the various nodes of the expression tree. It is
generally not necessary to use these explicitly but they have been included for
completeness.

.. inheritance-diagram:: squmfit.Expr squmfit.expr.Constant squmfit.expr.FiniteDiffGrad squmfit.expr.FuncExpr squmfit.expr.OpExpr squmfit.expr.Argument squmfit.parameter.FittedParam

.. currentmodule:: squmfit

The :class:`Constant` class
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: squmfit.expr.Constant
   :members:

   .. automethod:: squmfit.expr.Constant.__init__

The :class:`FiniteDiffGrad` class
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: squmfit.expr.FiniteDiffGrad
   :members:

   .. automethod:: squmfit.expr.FiniteDiffGrad.__init__
