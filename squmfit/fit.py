from __future__ import division
from copy import deepcopy
import numpy as np
import scipy.optimize
from .parameter import ParameterSet

class Curve(object):
    def __init__(self, name, model, data, weights=None, **user_args):
        # These shalln't be mutated
        self.name = name
        self.model = model
        self.data = data
        self.weights = weights
        self.user_args = user_args

    def eval_packed(self, params, **user_args):
        """ Evaluate the model """
        args = self.user_args.copy()
        args.update(user_args)
        return self.model.evaluate(params, **args)

    def residuals_packed(self, params, **user_args):
        """ Compute the weighed residuals """
        residuals = self.eval_packed(params, **user_args) - self.data
        if self.weights is not None:
            residuals *= self.weights
        return residuals

class Fit(object):
    """
    This represents a fit configuration.
    """

    def __init__(self):
        self._curves = []
        self.param_set = ParameterSet()

    def param(self, name=None, initial=None):
        return self.param_set.param(name, initial=initial)

    def add_curve(self, name, model, data, weights=None, **user_args):
        curve = Curve(name, model, data, weights, **user_args)
        self._curves.append(curve)

    def eval_packed(self, params, **user_args):
        """ Evaluate the model against packed parameters values """
        return {curve.name: curve.eval_packed(params, **user_args)
                for curve in self._curves}

    def residuals_packed(self, params, **user_args):
        """ Compute the weighed residuals against packed paramater values """
        return {curve.name: curve.residuals_packed(params, **user_args)
                for curve in self._curves}

    def eval(self, params, **user_args):
        """ Evaluate the model against a dictionary of parameters """
        return self.eval_packed(self.param_set._pack(params), **user_args)

    def residuals(self, params, **user_args):
        """ Evaluate the weighted model residuals against a dictionary of parameters """
        return self.residuals_packed(self.param_set._pack(params), **user_args)

    def fit(self, params0=None, **user_args):
        unpacked = self.param_set.initial_params()
        if params0 is not None:
            unpacked.update(params0)
        packed0 = self.param_set._pack(unpacked)
        def fit_func(p):
            res = self.residuals_packed(p, **user_args)
            return np.hstack(res.values())
        packed, cov_x, info, mesg, ier = scipy.optimize.leastsq(fit_func, packed0, full_output=True)

        def unpack_covar(matrix):
            return {name: self.param_set._unpack(inner)
                    for name, inner in self.param_set._unpack(matrix).items()}
        if cov_x is None:
            cov_p = None
        else:
            nparams = len(self.param_set.params)
            npts = sum(len(curve.data) for curve in self._curves)
            red_chisq = np.sum(info['fvec']**2) / (npts - nparams)
            cov_p = unpack_covar(cov_x * red_chisq)
        params = self.param_set._unpack(packed)
        initial = self.param_set._unpack(packed0)
        fit0 = FitResult(deepcopy(self), initial)
        fit = FitResult(deepcopy(self), params, cov_p, initial_result=fit0)
        return fit

class CurveResult(object):
    """
    This embodies a set of parameter values, particularly describing the goodness-of-fit with
    respect to a given curve.

    Attributes
    ----------
    fit_result : FitResult
        The FitResult describing this result
    curve : Curve
        The Curve described by this result
    degrees_of_freedom : int
        The number of degrees-of-freedom of the fit. This is defined as the number of
        data points in the curve minus the number of free parameters
        in the fitting model.
    fit : ndarray
        The model evaluated with the parameter values.
    residuals : ndarray
        The residuals of the fit
    chi_sqr : float
        Chi-squared of the fit
    reduced_chi_sqr : float
        The reduced chi-squared of the fit. Namely, ``chi_sqr / degrees_of_freedom``.
    """

    def __init__(self, fit_result, curve):
        params = fit_result.params
        self.fit_result = fit_result
        self.curve = curve
        npoints = len(self.curve.data)
        self.degrees_of_freedom = npoints - len(self.curve.model.parameters())
        self.fit = self.curve.eval_packed(self.fit_result.fit.param_set._pack(params))
        self.residuals = self.curve.residuals_packed(self.fit_result.fit.param_set._pack(params))
        self.chi_sqr = sum(self.residuals**2)
        self.reduced_chi_sqr = self.chi_sqr / self.degrees_of_freedom

class FitResult(object):
    """
    This embodies a set of parameter values, possibly originating from a fit.

    Attributes
    -----------
    fit : Fit
        The Fit for which these parameter apply.
    initial : FitResult
        The FitResult used as the initial parameters for the Fit from
        which this result originated.
    params : array, shape = [n_params]
        The parameter values
    curves : dict, curve_name -> CurveResults
        Results for particular curves.
    covar : dict of dicts, param_name -> param_name -> float or ``None``
        The covariances between parameters, or ``None`` if not available
        (which may either be due to numerical trouble in calculation
        or simply not being provided when the FitResult was created).
    stderr : dict, param_name -> float or ``None``
        The standard error of the parameter estimate or ``None`` if not available.
    correl : dict, param_name -> param_name -> float or ``None``
        The correlation coefficient between parameters or ``None`` if not available.
    """

    def __init__(self, fit, params, covar_p=None, initial_result=None):
        self.fit = fit
        self.initial = initial_result
        self.params = params
        self.covar = covar_p
        self.curves = {curve.name: CurveResult(self, curve)
                       for curve in fit._curves}

        self.stderr = None
        self.correl = None
        if self.covar is not None:
            self.stderr = {name: np.sqrt(self.covar[name][name])
                           for name in params}
            self.correl = {name: {name2: self.covar[name][name2] / self.stderr[name] / self.stderr[name2]
                                  for name2 in self.params
                                  if name != name2}
                           for name in self.params}

    @property
    def total_chi_sqr(self):
        return sum(curve.chi_sqr for curve in self.curves.values())
