from __future__ import division
from copy import deepcopy
import numpy as np
import scipy.optimize
from .parameter import ParameterSet
from .expr import Expr

class Curve(object):
    def __init__(self, name, model, data, weights=None, **user_args):
        """
        A curve to be fit against.

        :param name: A friendly name for the curve.
        :type model: :class:`Expr`
        :param model: The model to be fitted.
        :type weights: array of shape `(nsamples,)`
        :param data: The values of the dependent variable to fit against.
        :param weights: Weighing factors.
        """
        self._name = name
        self._model = model
        self._data = data
        self._weights = weights
        self._user_args = user_args

    @property
    def name(self):
        """ The friendly name of the curve """
        return self._name

    @property
    def model(self):
        """ The model to be fitted """
        return self._model

    @property
    def data(self):
        """ The values of the dependent variable to be fitted against """
        return self._data

    @property
    def weights(self):
        """ The fitting weights """
        return self._weights

    @property
    def user_args(self):
        """ User arguments """
        return self._user_args

    def eval_packed(self, params, **user_args):
        """ Evaluate the model """
        args = self.user_args.copy()
        args.update(user_args)
        return self.model.evaluate(params, **args)

    def residuals_packed(self, params, weighted=True, **user_args):
        """
        Compute the residuals between the curve and its model (:math:`model - data`).

        :type params: array of shape (nparams,)
        :param params: Packed array of parameter values.
        :type weighted: bool
        :param weighted: If False weights are ignored.
        :rtype: array of shape (npoints,)
        """
        residuals = self.eval_packed(params, **user_args) - self.data
        if weighted and self.weights is not None:
            residuals *= self.weights
        return residuals

class Fit(object):
    """
    This represents a fit configuration.
    """

    def __init__(self):
        """ Create a new fit configuration. """
        self._curves = []
        self.param_set = ParameterSet()

    def param(self, name=None, initial=None):
        """
        Create a new parameter.

        :type name: str, optional
        :param name:
            The name of the new parameter. A name will be generated if this is omitted.

        :type initial: float, optional
        :param initial:
            The initial value of the parameter. If not provided then a
            value must be provided when :func:`fit` is called.
        :rtype: :class:`.parameter.FittedParam`
        """
        return self.param_set.param(name, initial=initial)

    def add_curve(self, name, model, data, weights=None, **user_args):
        """
        Add a curve to the fit.

        :type name: str
        :param name:
            A unique name for the curve.

        :type model: :class:`Expr`
        :param model:
            The analytical model which will be fit to the curve.

        :type data: array, shape = [n_samples]
        :param data:
            An array of the dependent values for each sample.

        :type weights: array, shape = [n_samples], optional
        :param weights:
            An array of the weight of each sample. Often this is
            :math:`1 / \sigma` where `\sigma` is the standard
            deviation of the dependent value. If not given uniform
            weights are used.

        :type user_args: kwargs
        :param user_args:
            Keyword arguments passed to the model during evaluation.
        """
        if not isinstance(model, Expr):
            raise ValueError('Given model (%s) must be instance of Expr' % model)
        curve = Curve(name, model, data, weights, **user_args)
        self._curves.append(curve)

    def eval_packed(self, params, **user_args):
        """
        Evaluate the model against packed parameters values

        :type params: array, shape = [n_params]
        :param params: A packed array of parameters.

        :type user_args: kwargs
        :param user_args: Keyword arguments passed to the model.

        :rtype: dict, curve_name -> array, shape = [n_samples]
        """
        return {curve.name: curve.eval_packed(params, **user_args)
                for curve in self._curves}

    def residuals_packed(self, params, weighted=True, **user_args):
        """
        Compute the weighed residuals against packed paramater
        values. See :func:`eval_packed` for parameter descriptions.
        """
        return {curve.name: curve.residuals_packed(params, weighted, **user_args)
                for curve in self._curves}

    def eval(self, params, **user_args):
        """
        Evaluate the model against a dictionary of parameters

        :type params: dict, param_name -> float
        :param params: A dictionary of parameters values.

        :type user_args: kwargs
        :param user_args: Keyword arguments passed to the model.

        :rtype: dict, curve_name -> array, shape = [n_samples]
        :returns: The value of the model evaluated with the given parameters.
        """
        return self.eval_packed(self.param_set._pack(params), **user_args)

    def residuals(self, params, weighted=True, **user_args):
        """
        Evaluate the weighted model residuals against a dictionary of
        parameters values. See :func:`eval` for parameter
        descriptions.
        """
        return self.residuals_packed(self.param_set._pack(params), weighted, **user_args)

    def fit(self, params0=None, **user_args):
        """
        Carry out the fit.

        :type params0: dict, param_name -> float
        :param params0: The initial parameter values.

        :type user_args: kwargs
        :param user_args: Keyword arguments passed to the model.

        :rtype: A :class:`FitResults` object.
        """
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
    This embodies a set of parameter values describing the
    goodness-of-fit with respect to a given curve.
    """

    def __init__(self, fit_result, curve):
        params = fit_result.params
        self._fit_result = fit_result
        self._curve = curve
        self._fit = self.eval()
        self._residuals = self.curve.residuals_packed(self.fit_result.fit.param_set._pack(params))
        self._chi_sqr = sum(self.residuals**2)

    @property
    def fit_result(self):
        """
        The :class:`FitResult` this :class:`CurveResult` was derived from.

        :rtype: :class:`FitResult`
        """
        return self._fit_result

    @property
    def curve(self):
        """
        The curve described by this result

        :rtype: :class:`Curve`
        """
        return self._curve

    @property
    def fit(self):
        """
        The model evaluated with the parameter values.

        :rtype: ndarray
        """
        return self._fit

    @property
    def degrees_of_freedom(self):
        """
        The number of degrees-of-freedom of the fit. This is defined as the number of
        data points in the curve minus the number of free parameters
        in the fitting model.

        :rtype: int
        """
        if self.curve.weights is not None:
            npoints = np.count_nonzero(self.curve.weights)
        else:
            npoints = len(self.curve.data)
        return npoints - len(self.curve.model.parameters())

    @property
    def residuals(self):
        """
        The weighted residuals of the fit.

        :rtype: ndarray
        """
        return self._residuals

    @property
    def chi_sqr(self):
        """
        Chi-squared of the fit.

        :rtype: float
        """
        return self._chi_sqr

    @property
    def reduced_chi_sqr(self):
        """
        Reduced chi-squared of the fit. Namely, ``chi_sqr / degrees_of_freedom``.

        :rtype: float
        """
        return self.chi_sqr / self.degrees_of_freedom

    def eval(self, **user_args):
        """
        Evaluate the curve's model with overridden user arguments.
        """
        packed = self.fit_result.fit.param_set._pack(self.fit_result.params)
        return self.curve.eval_packed(packed, **user_args)

class FitResult(object):
    """
    This embodies a set of parameter values, possibly originating from a fit.
    """

    def __init__(self, fit, params, covar_p=None, initial_result=None):
        self._fit = fit
        self._initial = initial_result
        self._params = params
        self._curves = {curve.name: CurveResult(self, curve)
                        for curve in fit._curves}
        self._covar_p = covar_p

    @property
    def fit(self):
        """
        The :class:`Fit` to which these parameter apply.

        :rtype: :class:`Fit`
        """
        return self._fit

    @property
    def initial(self):
        """
        The :class:`FitResult` used as the initial parameters for the
        :class:`Fit` from which this result originated.

        :rtype: :class:`FitResult`
        """
        return self._initial

    @property
    def params(self):
        """
        The fitted parameter values

        :rtype: array, shape = [n_params]
        """
        return self._params

    @property
    def curves(self):
        """
        Results for particular curves.

        :rtype: dict, curve_name -> :class:`CurveResult`
        """
        return self._curves

    @property
    def covar(self):
        """
        The covariances between parameters, or ``None`` if not available
        (which may either be due to numerical trouble in calculation
        or simply not being provided when the :class:`FitResult` was
        created).

        :rtype: dict of dicts, param_name -> param_name -> float or ``None``
        """
        return self._covar_p

    @property
    def stderr(self):
        """
        The standard error of the parameter estimate or ``None`` if not available.

        :rtype: dict, param_name -> float or ``None``
        """
        if self._covar_p is None:
            return None
        return {name: np.sqrt(self._covar_p[name][name])
                for name in self.params}

    @property
    def correl(self):
        """
        The correlation coefficient between parameters or ``None`` if not available.

        :rtype: dict, param_name -> param_name -> float or ``None``
        """
        if self._covar_p is None:
            return None
        stderr = self.stderr
        return {name: {name2: self._covar_p[name][name2] / stderr[name] / stderr[name2]
                       for name2 in self.params
                       if name != name2}
                for name in self.params}

    @property
    def total_chi_sqr(self):
        """
        The sum of the individual curves' chi-squared values. This can
        be useful as a global goodness-of-fit metric.

        :rtype: float
        """
        return sum(curve.chi_sqr for curve in self.curves.values())

    def _repr_html_(self):
        from .pretty import ipynb_fit_result
        return ipynb_fit_result(self)
