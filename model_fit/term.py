from __future__ import division
import scipy.optimize
from .parameter import FittedParam

class Model(object):
    """
    A Model is a function which can take either fixed arguments or
    parameters from a parameter vector.
    """

    def __init__(self, eval, param_names=None):
        """
        Create a Model.
        
        Parameters:
        ------------
        eval : callable
            The function to evalulate.
        param_names : list or str, optional
            The names of the parameters expected by the model. These are passed to eval.
        """
        if param_names is None:
            import inspect
            param_names = inspect.getargspec(eval).args
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

