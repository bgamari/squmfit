from __future__ import division
import numpy as np

class FittedParam(object):
    """ A parameter to be fitted to data. """
    def __init__(self, idx, name=None):
        # these shalln't be mutated
        self.idx = idx
        self.name = name

class ParameterSet(object):
    """
    This object represents a set of parameters and how they can be
    packed into/out of a parameter vector.
    """
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