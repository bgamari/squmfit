from __future__ import division
import operator
import numpy as np
import scipy.optimize

class Model(object):
    """
    A Model is a function which can take either fixed arguments or
    parameters from a parameter vector.

    This is essentially an adapter lifting a function into an `Expr`.
    """

    def __init__(self, eval, param_names=None, defaults={}):
        """
        Create a Model.

        :type eval: callable
        :param eval: The function to evalulate.
        :type param_names: list or str, optional
        :param param_names: The names of the parameters expected by the model. This is to allow
            validation of saturation of the argument list when an `Expr` is instantiated.
        """
        if param_names is None:
            import inspect
            param_names = inspect.getargspec(eval).args
        elif isinstance(param_names, str):
            param_names = param_names.split()
        elif not isinstance(param_names, list):
            raise RuntimeError('Expected list of parameter names, found %s' % param_names)
        self.param_names = param_names
        self.defaults = defaults

        self.eval = eval

    def __call__(self, *args, **kwargs):
        """
        Produce an :class:`Expr` which will invoke the `eval` function
        with the provided arguments, taking arguments from the parameters
        as appropriate.
        """
        expected = set(self.param_names)
        given = set(kwargs.viewkeys())
        if given != expected:
            raise RuntimeError('Saw parameters %s, expected parameters %s' % (given, expected))

        kwargs = kwargs.copy()
        kwargs.update(self.defaults)
        return FuncExpr(self.eval, *args, **kwargs)

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
        raise NotImplementedError()

    def parameters(self):
        """
        Return the set of fitted parameters used by this expression.

        :rtype: :class:`set` of :class:`FittedParam`.
        """
        raise NotImplementedError()

    def map(self, f):
        """
        Lift a function into an :class:`Expr`.
        """
        return FuncExpr(f, self)

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

    This is essentially a wrapper lifting function application into
    the `Expr` functor.
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
        self.func = func
        self.args = args
        self.kwargs = kwargs

    def evaluate(self, params, **user_args):
        def eval_term(value):
            if isinstance(value, Expr):
                return value.evaluate(params, **user_args)
            else:
                return value
        eval_args = map(eval_term, self.args)
        eval_kwargs = {k: eval_term(v) for k,v in self.kwargs.iteritems()}
        return self.func(*eval_args, **eval_kwargs)

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

class Argument(Expr):
    """
    An :class:`Expr` which evaluates to a keyword argument passed at
    evaluation-time.
    """
    def __init__(self, name):
        self.name = name

    def evaluate(self, params, **user_args):
        return user_args[self.name]

    def parameters(self):
        return set()
