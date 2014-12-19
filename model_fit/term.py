from __future__ import division
import operator
import scipy.optimize

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

def lift_term(value):
    if isinstance(value, Term):
        return value
    else:
        return ConstTerm(value)

class Term(object):
    def evaluate(self, params, **user_args):
        raise NotImplemented

    def parameters(self):
        """ Return the set of parameters used by this term (and its sub-terms) """
        raise NotImplemented

    def __add__(self, other):
        return OpTerm(sum, self, lift_term(other))

    def __radd__(self, other):
        return OpTerm(sum, self, lift_term(other))

    def __multiply__(self, other):
        product = lambda args: reduce(operator.mul, args, 1)
        return OpTerm(product, self, lift_term(other))

    def __rmultiply__(self, other):
        product = lambda args: reduce(operator.mul, args, 1)
        return OpTerm(product, self, lift_term(other))

class ModelInst(Term):
    """ An instance of a model """
    def __init__(self, _model, *args, **kwargs):
        self.model = _model
        self.args = args
        self.kwargs = kwargs

    def evaluate(self, params, **user_args):
        def eval_term(value):
            if isinstance(value, Term):
                return value.evaluate(params, **user_args)
            else:
                return value
        eval_args = map(eval_term, self.args)
        eval_kwargs = user_args.copy()
        eval_kwargs.update({k: eval_term(v) for k,v in self.kwargs.iteritems()})
        for name, value in zip(self.model.param_names, eval_args):
            eval_kwargs[name] = value
        if eval_kwargs.viewkeys() != set(self.model.param_names):
            given = eval_kwargs.viewkeys()
            expected = set(self.model.param_names)
            raise RuntimeError('Saw parameters %s, expected parameters %s' % (given, expected))
        return self.model.eval(**eval_kwargs)
        
    def parameters(self):
        accum = set()
        for p in self.args:
            if isinstance(p, Term):
                accum.update(p.parameters())
        for p in self.kwargs.values():
            if isinstance(p, Term):
                accum.update(p.parameters())
        return accum
        
class OpTerm(Term):
    def __init__(self, op, *operands):
        self.op = op
        self.operands = operands

    def evaluate(self, params, **user_args):
        return self.op([model.evaluate(params, **user_args) for model in self.operands])
        
    def parameters(self):
        accum = set()
        accum.update(*[a.parameters() for a in self.operands])
        return accum

class ConstTerm(Term):
    def __init__(self, value):
        self.value = value

    def evaluate(self, params, **user_args):
        return self.value

    def parameters(self):
        return set()
