"""
squmfit --- Flexible non-linear least-squares fitting
"""

from .expr import model, Expr, FuncExpr, Argument
from .fit import Curve, Fit, BoundedFit, FitResult, CurveResult
from .parameter import FittedParam
