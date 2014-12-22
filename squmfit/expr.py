from __future__ import division
import operator
import numpy as np
import scipy.optimize

class Model(object):
    """
    A Model is a function which can take either fixed arguments or
    parameters from a parameter vector.
    """

    def __init__(self, eval, param_names=None):
        """
        Create a Model.

        :type eval: callable
        :param eval: The function to evalulate.
        :type param_names: list or str, optional
        :param param_names: The names of the parameters expected by the model. These are passed to eval.
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
        Produce a closure which will invoke the :class:`Model`'s eval function
        with the provided arguments, taking arguments from a parameters
        vector as necessary.
        """
        return FuncExpr(self, *args, **kwargs)

def lift_term(value):
    if isinstance(value, Expr):
        return value
    else:
        return Constant(value)

class Expr(object):
    """
    An expression capable of taking parameters from a packed parameter
    vector. All of the usual arithmetic operations are supported.
    """

    def evaluate(self, params, **user_args):
        """
        Evaluate the model with the given parameter values.

        :type params: array, shape = [n_params]
        :param params: Packed parameters vector

        :type user_args: kwargs
        :param user_args: Keyword arguments from user
        """
        raise NotImplemented

    def parameters(self):
        """
        Return the set of fitted parameters used by this expression.

        :rtype: :class:`set` of :class:`FittedParam`.
        """
        raise NotImplemented

    def __neg__(self):
        return OpExpr(operator.neg, self)

    def __add__(self, other):
        return OpExpr(operator.add, self, lift_term(other))

    def __radd__(self, other):
        return OpExpr(operator.add, lift_term(other), self)

    def __sub__(self, other):
        return OpExpr(operator.sub, self, lift_term(other))

    def __rsub__(self, other):
        return OpExpr(operator.sub, lift_term(other), self)

    def __mul__(self, other):
        return OpExpr(operator.mul, self, lift_term(other))

    def __rmul__(self, other):
        return OpExpr(operator.mul, lift_term(other), self)

    def __truediv__(self, other):
        return OpExpr(operator.div, self, lift_term(other))

    def __rtruediv__(self, other):
        return OpExpr(operator.div, lift_term(other), self)

    def __div__(self, other):
        return OpExpr(operator.div, self, lift_term(other))

    def __rdiv__(self, other):
        return OpExpr(operator.div, lift_term(other), self)

    def __pow__(self, other):
        return OpExpr(operator.pow, self, lift_term(other))

    def __rpow__(self, other):
        return OpExpr(operator.pow, lift_term(other), self)

    # Used by numpy
    def floor(self):
        return OpExpr(np.floor, self)

    def ceil(self):
        return OpExpr(np.ceil, self)

    def exp(self):
        return OpExpr(np.exp, self)

class FuncExpr(Expr):
    """
    An expression which calls a function.

    Any arguments which are :class:`Expr` objects will be evaluated
    with :class:`Expr.evaluate` before being passed to the function.
    """

    def __init__(self, func, *args, **kwargs):
        """
        Create an expression which calls the provided function with
        the provided arguments. Any arguments which are :class:`Expr`
        objects will be evaluated with :class:`Expr.evaluate` before
        being passed on to the function.

        :param func: The function to call.
        :param args: Ordered arguments to pass to the function.
        :param kwargs: Keyword arguments to pass to the function.
        """
        self.model = func
        self.args = args
        self.kwargs = kwargs

    def evaluate(self, params, **user_args):
        def eval_term(value):
            if isinstance(value, Expr):
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
            if isinstance(p, Expr):
                accum.update(p.parameters())
        for p in self.kwargs.values():
            if isinstance(p, Expr):
                accum.update(p.parameters())
        return accum

class OpExpr(Expr):
    """ A helper used by arithmetic operations """
    def __init__(self, op, *operands):
        self.op = op
        self.operands = operands

    def evaluate(self, params, **user_args):
        return self.op(*[model.evaluate(params, **user_args) for model in self.operands])

    def parameters(self):
        accum = set()
        accum.update(*[a.parameters() for a in self.operands])
        return accum

class Constant(Expr):
    """ An :class:`Expr` which always evaluates to the given value """
    def __init__(self, value):
        self.value = value

    def evaluate(self, params, **user_args):
        return self.value

    def parameters(self):
        return set()
