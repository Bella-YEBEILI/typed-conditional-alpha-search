"""The functions used to create programs.

The :mod:`gplearn.functions` module contains all of the functions used by
gplearn programs. It also contains helper methods for a user to define their
own custom functions.
"""

import numpy as np
from joblib import wrap_non_picklable_objects

__all__ = ['make_function']


class _Function(object):

    """A representation of a mathematical relationship, a node in a program.

    This object is able to be called with NumPy vectorized arguments and return
    a resulting vector based on a mathematical relationship.

    Parameters
    ----------
    function : callable
        A function with signature function(x1, *args) that returns a Numpy
        array of the same shape as its arguments.

    name : str
        The name for the function as it should be represented in the program
        and its visualizations.

    arity : int
        The number of arguments that the ``function`` takes.

    """

    def __init__(self, function, name, arity, isRandom=(False, (1,100)),
                 role=None, is_ts_op=False, complexity_weight=1.0):
        self.function = function
        self.name = name
        self.arity = arity
        
        # 随机窗口/参数配置（主要给 dynamic_ts_* 这类算子使用）
        self.isRandom = isRandom[0]
        self.RandRange = isRandom[1]
        if (not isinstance(self.RandRange,tuple)) or (not isinstance(self.RandRange[0], int)) or (not isinstance(self.RandRange[1], int)) or len(self.RandRange) != 2:
            raise TypeError('RandRange should be a tuple ranging from 1 to 100.')
        self.baseConst = -1

        # 语义标签：用于后续在 _Program 中做 grammar/复杂度控制
        # role: 'arith' / 'lag' / 'ts_agg' / 'ts_norm' / 'other' 等
        self.role = role or 'other'
        # 是否属于“时间窗口/时间序列聚合”类算子（dynamic_ts_* / ts_* 等）
        self.is_ts_op = bool(is_ts_op)
        # 算子级复杂度权重（用于 fine-grain complexity penalty）
        self.complexity_weight = float(complexity_weight)

    def __call__(self, *args):
        if self.isRandom and self.baseConst > 0:
            if len(args) > 1 and isinstance(args[-1], int):
                return self.function(*args)
            return self.function(*args, self.baseConst)
        else:
            return self.function(*args)


def make_function(*, function, name, arity, wrap=True):
    """Make a function node, a representation of a mathematical relationship.

    This factory function creates a function node, one of the core nodes in any
    program. The resulting object is able to be called with NumPy vectorized
    arguments and return a resulting vector based on a mathematical
    relationship.

    Parameters
    ----------
    function : callable
        A function with signature `function(x1, *args)` that returns a Numpy
        array of the same shape as its arguments.

    name : str
        The name for the function as it should be represented in the program
        and its visualizations.

    arity : int
        The number of arguments that the `function` takes.

    wrap : bool, optional (default=True)
        When running in parallel, pickling of custom functions is not supported
        by Python's default pickler. This option will wrap the function using
        cloudpickle allowing you to pickle your solution, but the evolution may
        run slightly more slowly. If you are running single-threaded in an
        interactive Python session or have no need to save the model, set to
        `False` for faster runs.

    """
    if not isinstance(arity, int):
        raise ValueError('arity must be an int, got %s' % type(arity))
    if not isinstance(function, np.ufunc):
        if function.__code__.co_argcount != arity:
            raise ValueError('arity %d does not match required number of '
                             'function arguments of %d.'
                             % (arity, function.__code__.co_argcount))
    if not isinstance(name, str):
        raise ValueError('name must be a string, got %s' % type(name))
    if not isinstance(wrap, bool):
        raise ValueError('wrap must be an bool, got %s' % type(wrap))

    # Check output shape
    args = [np.ones(10) for _ in range(arity)]
    try:
        function(*args)
    except (ValueError, TypeError):
        raise ValueError('supplied function %s does not support arity of %d.'
                         % (name, arity))
    if not hasattr(function(*args), 'shape'):
        raise ValueError('supplied function %s does not return a numpy array.'
                         % name)
    if function(*args).shape != (10,):
        raise ValueError('supplied function %s does not return same shape as '
                         'input vectors.' % name)

    # Check closure for zero & negative input arguments
    args = [np.zeros(10) for _ in range(arity)]
    if not np.all(np.isfinite(function(*args))):
        raise ValueError('supplied function %s does not have closure against '
                         'zeros in argument vectors.' % name)
    args = [-1 * np.ones(10) for _ in range(arity)]
    if not np.all(np.isfinite(function(*args))):
        raise ValueError('supplied function %s does not have closure against '
                         'negatives in argument vectors.' % name)

    if wrap:
        return _Function(function=wrap_non_picklable_objects(function),
                         name=name,
                         arity=arity)
    return _Function(function=function,
                     name=name,
                     arity=arity)


def _protected_division(x1, x2):
    """Closure of division (x1/x2) for zero denominator."""
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(np.abs(x2) > 0.001, np.divide(x1, x2), 1.)


def _protected_sqrt(x1):
    """Closure of square root for negative arguments."""
    return np.sqrt(np.abs(x1))


def _protected_log(x1):
    """Closure of log for zero and negative arguments."""
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(np.abs(x1) > 0.001, np.log(np.abs(x1)), 0.)


def _protected_inverse(x1):
    """Closure of inverse for zero arguments."""
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(np.abs(x1) > 0.001, 1. / x1, 0.)


def _sigmoid(x1):
    """Special case of logistic function to transform to probabilities."""
    with np.errstate(over='ignore', under='ignore'):
        return 1 / (1 + np.exp(-x1))


add2 = _Function(function=np.add, name='add', arity=2,
                 role='arith', is_ts_op=False, complexity_weight=1.0)
sub2 = _Function(function=np.subtract, name='sub', arity=2,
                 role='arith', is_ts_op=False, complexity_weight=1.0)
mul2 = _Function(function=np.multiply, name='mul', arity=2,
                 role='arith', is_ts_op=False, complexity_weight=1.5)
div2 = _Function(function=_protected_division, name='div', arity=2,
                 role='arith', is_ts_op=False, complexity_weight=2.0)
sqrt1 = _Function(function=_protected_sqrt, name='sqrt', arity=1,
                  role='arith', is_ts_op=False, complexity_weight=1.5)
log1 = _Function(function=_protected_log, name='log', arity=1,
                 role='arith', is_ts_op=False, complexity_weight=1.5)
neg1 = _Function(function=np.negative, name='neg', arity=1,
                 role='arith', is_ts_op=False, complexity_weight=0.5)
inv1 = _Function(function=_protected_inverse, name='inv', arity=1,
                 role='arith', is_ts_op=False, complexity_weight=2.0)
abs1 = _Function(function=np.abs, name='abs', arity=1,
                 role='arith', is_ts_op=False, complexity_weight=1.5)
max2 = _Function(function=np.maximum, name='max', arity=2,
                 role='arith', is_ts_op=False, complexity_weight=1.5)
min2 = _Function(function=np.minimum, name='min', arity=2,
                 role='arith', is_ts_op=False, complexity_weight=1.5)
sin1 = _Function(function=np.sin, name='sin', arity=1,
                 role='arith', is_ts_op=False, complexity_weight=1.5)
cos1 = _Function(function=np.cos, name='cos', arity=1,
                 role='arith', is_ts_op=False, complexity_weight=1.5)
tan1 = _Function(function=np.tan, name='tan', arity=1,
                 role='arith', is_ts_op=False, complexity_weight=2.0)
sig1 = _Function(function=_sigmoid, name='sig', arity=1,
                 role='arith', is_ts_op=False, complexity_weight=1.5)

_function_map = {'add': add2,
                 'sub': sub2,
                 'mul': mul2,
                 'div': div2,
                 'sqrt': sqrt1,
                 'log': log1,
                 'abs': abs1,
                 'neg': neg1,
                 'inv': inv1,
                 'max': max2,
                 'min': min2,
                 'sin': sin1,
                 'cos': cos1,
                 'tan': tan1}
