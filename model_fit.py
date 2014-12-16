from __future__ import division
import scipy.optimize
from copy import deepcopy

class Model(object):
    def __init__(self, eval, param_names=None):
        """
        Model(eval, param_names=None)
        
        @eval: a function over some arguments 
        """
        if param_names is None:
            raise NotImplemented
        elif isinstance(param_names, str):
            param_names = param_names.split()
        elif not isinstance(param_names, list):
            raise RuntimeError('Expected list of parameter names, found %s' % param_names)
        self.param_names = param_names

        if eval is not None:
            self.eval = eval

    def __call__(self, *args, **kwargs):
        """ 
        Produce a closure which will invoke the Model's eval function
        with the provided arguments, taking arguments from a parameters
        vector as necessary.
        """
        return ModelInst(self, *args, **kwargs)
       
class Term(object):
    def __call__(self, params, **user_args):
        raise NotImplemented

    def count_params(self):
        raise NotImplemented

class ModelInst(Term):
    """ An instance of a model """
    def __init__(self, model, *args, **kwargs):
        self.model = model
        self.args = args
        self.kwargs = kwargs

    def __call__(self, params, **user_args):
        def evaluate(value):
            if isinstance(value, FittedParam):
                return params[value.idx]
            else:
                return value
        eval_args = map(evaluate, self.args)
        eval_kwargs = user_args.copy()
        eval_kwargs.update({k: evaluate(v) for k,v in self.kwargs.iteritems()})
        for name, value in zip(self.model.param_names, eval_args):
            eval_kwargs[name] = value
        if eval_kwargs.viewkeys() != set(self.model.param_names):
            given = eval_kwargs.viewkeys()
            expected = set(self.param_names)
            raise RuntimeError('Saw parameters %s, expected parameters %s' % (given, expected))
        return self.model.eval(**eval_kwargs)
        
    def count_params(self):
        accum = 0
        for p in self.args:
            if isinstance(p, FittedParam):
                accum += 1
        for p in self.kwargs.values():
            if isinstance(p, FittedParam):
                accum += 1
        return accum

    def __add__(self, a, b):
        return OpModel(sum, a, b)
        
class OpModel(Term):
    def __init__(self, op, *operands):
        self.op = op
        self.operands = operands

    def __call__(params, **user_args):
        return self.op(*[model(params, **user_args) for model in self.operands])
        
    def count_params(self):
        return sum(a for a in self.operands)

class FittedParam(object):
    """ A fitted parameter """
    def __init__(self, idx, name=None):
        # these shalln't be mutated
        self.idx = idx
        self.name = name

class ParameterSet(object):
    def __init__(self):
        self.params = {}

    def __getitem__(self, name):
        return self.params[name]
            
    def param_names(self):
        return self.params.keys()

    def _unused_name(self):
        """ Generate an unused parameter name """
        i = 0
        used = self.params.keys()
        while True:
            name = 'param%d' % i
            if name not in used:
                return name

    def param(self, name=None):
        if name is not None:
            assert name not in self.params.keys()
        else:
            name = self._unused_name()
        idx = len(self.params)
        param = FittedParam(idx, name)
        self.params[name] = param
        return param
        
    def _pack(self, values):
        """
        Pack a set of parameter values (given as a dictionary) into a
        vector.
        """
        unset = set(self.params.keys())
        accum = np.empty(shape=len(self.params), dtype='f8')
        for name, param in self.params.iteritems():
            accum[param.idx] = values[name]
            unset.remove(name)
        if len(unset) > 0:
            raise RuntimeError("No values were provided for the following parameters: %s" %
                               (', '.join(unset)))
        return accum

    def _unpack(self, values):
        """
        Unpack a parameter vector into a dictionary of parameter values.
        """
        accum = {}
        if len(values) != len(self.params):
            raise RuntimeError("This parameter set has %d parameters, the given vector has %d." %
                               (len(self.params), len(values)))
        for name, param in self.params.iteritems():
            accum[name] = values[param.idx]
        return accum

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
        return self.model(params, **args)

    def residuals_packed(self, params, **user_args):
        """ Compute the weighed residuals """
        residuals = self.eval_packed(params, **user_args) - self.data
        if self.weights is not None:
            residuals *= self.weights
        return residuals

class Fit(object):
    def __init__(self):
        self._curves = []
        self.param_set = ParameterSet()
    
    def param(self, name=None):
        return self.param_set.param(name)

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

    def fit(self, params0, **user_args):
        packed0 = self.param_set._pack(params0)
        def fit_func(p):
            res = self.residuals_packed(p, **user_args)
            return np.hstack(res.values())
        packed, cov, info, mesg, ier = scipy.optimize.leastsq(fit_func, packed0, full_output=True)
        if cov is None:
            unpacked_cov = None
        else:
            unpacked_cov = {name: self.param_set._unpack(inner)
                            for name, inner in self.param_set._unpack(cov).items()}
        params = self.param_set._unpack(packed)
        fit = FitResult(deepcopy(self), params0, params, unpacked_cov)
        return fit

class CurveResult(object):
    def __init__(self, fit_result, curve):
        params = fit_result.params
        self.fit_result = fit_result
        self.curve = curve
        self.npoints = len(self.curve.data)
        self.degrees_of_freedom = self.npoints - self.curve.model.count_params()
        self.residuals = self.curve.residuals_packed(self.fit_result.fit.param_set._pack(params))
        self.chi_sqr = sum(self.residuals**2)
        self.reduced_chi_sqr = self.chi_sqr / self.degrees_of_freedom
        
class FitResult(object):
    def __init__(self, fit, initial_params, params, covar):
        self.fit = fit
        self.initial_params = initial_params
        self.params = params
        print self.params
        self.covar = covar
        self.curves = {curve.name: CurveResult(self, curve)
                       for curve in fit._curves}

def exponential_model(t, amplitude, rate):
    return amplitude * np.exp(-t * rate)

import numpy as np
ExpModel = Model(exponential_model, 't rate amplitude')

# example
xs = np.arange(4000)
ys = 4 * np.exp(-xs / 400)
fit = Fit()
model = ExpModel(amplitude=fit.param('amp'), rate=fit.param('rate'))
fit.add_curve('curve1', model, ys, t=xs)
res = fit.fit({'amp': 3, 'rate': 1./100})
print res.params
print res.covar
print res.curves['curve1'].reduced_chi_sqr

from matplotlib import pyplot as pl
pl.plot(xs, ys)
pl.plot(xs, fit.eval(res.params)['curve1'])
pl.show()
