from __future__ import division
import numpy as np
from .expr import Expr, Constant

class FittedParam(Expr):
    """ A parameter to be fitted to data. """
    def __init__(self, param_set, idx, name=None, initial=None):
        # these shalln't be mutated
        self.param_set = param_set
        self.idx = idx
        self.name = name
        self.initial = initial

    def evaluate(self, params, **user_args):
        return params[self.idx]

    def parameters(self):
        return set([self])

    def gradient(self):
        arr = np.zeros(len(self.param_set.params))
        arr[self.idx] = 1
        return Constant(arr)

    def __str__(self):
        return 'Parameter(%s)' % self.name

class ParameterSet(object):
    """
    This object represents a set of parameters and how they can be
    packed into/out of a parameter vector.
    """
    def __init__(self):
        self._params = {}

    @property
    def params(self):
        """
        The parameters of this parameter set.

        :rtype: dict from parameter name to :class:`FittedParam`
        """
        return self._params

    def __getitem__(self, name):
        return self._params[name]

    def param_names(self):
        return self._params.keys()

    def _unused_name(self):
        """ Generate an unused parameter name """
        i = 0
        used = self._params.keys()
        while True:
            name = 'param%d' % i
            if name not in used:
                return name
            i += 1

    def initial_params(self):
        return {name: param.initial
                for name, param in self._params.items()}

    def param(self, name=None, initial=None):
        if name is not None:
            assert name not in self._params.keys()
        else:
            name = self._unused_name()
        idx = len(self._params)
        param = FittedParam(self, idx, name, initial)
        self._params[name] = param
        return param

    def _pack(self, values):
        """
        Pack a set of parameter values (given as a dictionary) into a
        vector.
        """
        unset = set(self._params.keys())
        accum = np.empty(shape=len(self._params), dtype='f8')
        for name, param in self._params.iteritems():
            if values[name] is None:
                continue
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
        if len(values) != len(self._params):
            raise RuntimeError("This parameter set has %d parameters, the given vector has %d." %
                               (len(self._params), len(values)))
        return {name: values[param.idx]
                for name, param in self._params.iteritems()}
