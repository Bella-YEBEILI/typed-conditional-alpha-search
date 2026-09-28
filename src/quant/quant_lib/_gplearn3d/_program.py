"""The underlying data structure used in gplearn.

The :mod:`gplearn._program` module contains the underlying representation of a
computer program. It is used for creating and evolving programs used in the
:mod:`gplearn.genetic` module.
"""
import pandas as pd
from copy import copy
import numpy as np
from sklearn.utils.random import sample_without_replacement
from .functions import _Function
from .utils import check_random_state
import warnings
from copy import deepcopy
warnings.filterwarnings("ignore")

class _Program(object):
    """A program-like representation of the evolved program.

    This is the underlying data-structure used by the public classes in the
    :mod:`gplearn.genetic` module. It should not be used directly by the user.

    Parameters
    ----------
    function_set : list
        A list of valid functions to use in the program.

    arities : dict
        A dictionary of the form `{arity: [functions]}`. The arity is the
        number of arguments that the function takes, the functions must match
        those in the `function_set` parameter.

    init_depth : tuple of two ints
        The range of tree depths for the initial population of naive formulas.
        Individual trees will randomly choose a maximum depth from this range.
        When combined with `init_method='half and half'` this yields the well-
        known 'ramped half and half' initialization method.

    init_method : str
        - 'grow' : Nodes are chosen at random from both functions and
          terminals, allowing for smaller trees than `init_depth` allows. Tends
          to grow asymmetrical trees.
        - 'full' : Functions are chosen until the `init_depth` is reached, and
          then terminals are selected. Tends to grow 'bushy' trees.
        - 'half and half' : Trees are grown through a 50/50 mix of 'full' and
          'grow', making for a mix of tree shapes in the initial population.

    n_features : int
        The number of features in `X`.

    const_range : tuple of two floats
        The range of constants to include in the formulas.

    metric : _Fitness object
        The raw fitness metric.

    p_point_replace : float
        The probability that any given node will be mutated during point
        mutation.

    parsimony_coefficient : float
        This constant penalizes large programs by adjusting their fitness to
        be less favorable for selection. Larger values penalize the program
        more which can control the phenomenon known as 'bloat'. Bloat is when
        evolution is increasing the size of programs without a significant
        increase in fitness, which is costly for computation time and makes for
        a less understandable final result. This parameter may need to be tuned
        over successive runs.

    random_state : RandomState instance
        The random number generator. Note that ints, or None are not allowed.
        The reason for this being passed is that during parallel evolution the
        same program object may be accessed by multiple parallel processes.

    transformer : _Function object, optional (default=None)
        The function to transform the output of the program to probabilities,
        only used for the SymbolicClassifier.

    feature_names : list, optional (default=None)
        Optional list of feature names, used purely for representations in
        the `print` operation or `export_graphviz`. If None, then X0, X1, etc
        will be used for representations.

    program : list, optional (default=None)
        The flattened tree representation of the program. If None, a new naive
        random tree will be grown. If provided, it will be validated.

    Attributes
    ----------
    program : list
        The flattened tree representation of the program.

    raw_fitness_ : float
        The raw fitness of the individual program.

    fitness_ : float
        The penalized fitness of the individual program.

    oob_fitness_ : float
        The out-of-bag raw fitness of the individual program for the held-out
        samples. Only present when sub-sampling was used in the estimator by
        specifying `max_samples` < 1.0.

    parents : dict, or None
        If None, this is a naive random program from the initial population.
        Otherwise it includes meta-data about the program's parent(s) as well
        as the genetic operations performed to yield the current program. This
        is set outside this class by the controlling evolution loops.

    depth_ : int
        The maximum depth of the program tree.

    length_ : int
        The number of functions and terminals in the program.

    """

    def __init__(self,
                 function_set,
                 arities,
                 init_depth,
                 init_method,
                 n_features,
                 const_range,
                 metric,
                 p_point_replace,
                 parsimony_coefficient,
                 random_state,
                 transformer=None,
                 feature_names=None,
                 program=None,
                 op_penalty_map=None,
                 feature_min_unique=0,
                 feature_penalty=0.0,
                 depth_threshold=None,
                 depth_penalty=0.0):

        self.function_set = function_set
        self.arities = arities
        self.init_depth = (init_depth[0], init_depth[1] + 1)
        self.init_method = init_method
        self.n_features = n_features
        self.const_range = const_range
        self.metric = metric
        self.p_point_replace = p_point_replace
        self.parsimony_coefficient = parsimony_coefficient
        self.transformer = transformer
        self.feature_names = feature_names
        self.program = program
        # Penalty configuration for diversity/structure
        self._penalty_cfg = {
            'op_penalty_map': op_penalty_map or {},
            'feature_min_unique': int(feature_min_unique) if feature_min_unique is not None else 0,
            'feature_penalty': float(feature_penalty) if feature_penalty is not None else 0.0,
            'depth_threshold': int(depth_threshold) if depth_threshold is not None else None,
            'depth_penalty': float(depth_penalty) if depth_penalty is not None else 0.0,
        }

        # 粗类型标注：根据特征名将叶子特征分为 PRICE / LIQ / SIZE / OTHER
        self._feature_types = None
        if self.feature_names is not None:
            try:
                types = []
                for name in self.feature_names:
                    if name in {'adj_close', 'adj_open', 'adj_high', 'adj_low', 'vwap'}:
                        f_type = 'PRICE'
                    elif name in {'amount', 'volume', 'turnover'}:
                        f_type = 'LIQ'
                    elif 'float_mkt_cap' in str(name):
                        f_type = 'SIZE'
                    else:
                        f_type = 'OTHER'
                    types.append(f_type)
                self._feature_types = types
            except Exception:
                self._feature_types = None

        if self.program is not None:
            # 对外部提供的 program 做基础合法性 + 语法检查，不通过则重建
            if (not self.validate_program()) or (not self._grammar_ok(self.program)):
                # 若给出的 program 不合法，则重新随机生成一个符合语法的 program
                self.program = self.build_program(random_state)
        else:
            # Create a naive random program, retry少数次直到通过语法检查
            max_tries = 5
            last_prog = None
            for _ in range(max_tries):
                last_prog = self.build_program(random_state)
                if self._grammar_ok(last_prog):
                    self.program = last_prog
                    break
            if self.program is None:
                # 兜底：即便语法检查失败，也至少保留最后一次构造结果，避免崩溃
                self.program = last_prog

        # Age of the individual (used for age-layered / protection logic in Beam)
        # 默认从0开始，后续在进化循环中按父代age_+1更新
        self.age_ = 0

        self.raw_fitness_ = None
        self.fitness_ = None
        self.parents = None
        self._n_samples = None
        self._max_samples = None
        self._indices_state = None


    def build_program(self, random_state):
        """
        构建原生的随机表达式

        Parameters
        ----------
        random_state : RandomState instance
            随机状态生成

        Returns
        -------
        program : list
            展平的树结构的表达式（具体的解读方式类似于编译器对栈元素的解读）.

        """
        if self.init_method == 'half and half':
            method = ('full' if random_state.randint(2) else 'grow')
        else:
            method = self.init_method
        # 确定了每次增加的最大深度
        max_depth = random_state.randint(*self.init_depth)

        # Start a program with a function to avoid degenerative programs
        # 随机选择第一个初始的函数算子（根节点允许是任意函数，包括时间窗算子）
        function_idx = random_state.randint(len(self.function_set))
        function = deepcopy(self.function_set[function_idx])
        if function.isRandom:
            current_window = random_state.randint(function.RandRange[0], function.RandRange[1])
            function.baseConst = current_window

        program = [function]
        # 增加该算子需要的参数值
        terminal_stack = [function.arity]
        # ts_depth_stack 跟踪当前路径上时间窗算子（is_ts_op）的使用深度
        ts_depth_stack = [1 if getattr(function, "is_ts_op", False) else 0]
        # in_ts_subtree 跟踪当前是否在某个 TS 算子的子树内部（用于规则6）
        in_ts_subtree = [getattr(function, "is_ts_op", False)]
        terminal_value_stack = []
        # 当terminal_stack 的参数值没有被填满，表达式不充实的时候
        while terminal_stack:
            # 查看栈的深度
            depth = len(terminal_stack)
            # 能选择的特征和其他算子总共有哪些
            choice = self.n_features + len(self.function_set)
            # 随机选择算子/终端的索引
            choice = random_state.randint(choice)
            # Determine if we are adding a function or terminal
            # 决定了我们是增加一个新的函数还是算子
            if (depth < max_depth) and (method == 'full' or
                                        choice <= len(self.function_set)):
                # 根据当前路径上的 ts_depth 和是否在 TS 子树内部限制可选的函数集合
                curr_ts_depth = ts_depth_stack[-1] if ts_depth_stack else 0
                is_in_ts = in_ts_subtree[-1] if in_ts_subtree else False
                candidates = self.function_set
                
                # 规则1：同一路径上时间窗类算子（is_ts_op）最多出现 1 次
                if curr_ts_depth >= 1:
                    candidates = [f for f in candidates
                                  if not getattr(f, "is_ts_op", False)]
                
                # 规则6：TS 子树内部只允许特定的局部加工算子
                if is_in_ts:
                    allowed_inner = {'delta', 'delay', 'pos', 'neg_part', 'abs', 'tanh'}
                    candidates = [f for f in candidates
                                  if getattr(f, "name", None) in allowed_inner]
                
                # 理论上 candidates 不会为空；若为空则退回全体以避免死循环
                if not candidates:
                    candidates = self.function_set

                func_idx = random_state.randint(len(candidates))
                function = deepcopy(candidates[func_idx])
                if function.isRandom:
                    current_window = random_state.randint(function.RandRange[0],function.RandRange[1])
                    function.baseConst = current_window
                program.append(function)
                terminal_stack.append(function.arity)
                # 子节点路径上的 ts_depth = 父节点 ts_depth + 当前节点是否为 ts_op
                child_ts_depth = curr_ts_depth + (1 if getattr(function, "is_ts_op", False) else 0)
                ts_depth_stack.append(child_ts_depth)
                # 子节点是否在 TS 子树内部 = 当前节点本身是 TS（不继承父节点状态）
                child_in_ts = getattr(function, "is_ts_op", False)
                in_ts_subtree.append(child_in_ts)
            else:
                # We need a terminal, add a variable or constant
                if self.const_range is not None:
                    # 生成特征还是常数
                    terminal = random_state.randint(self.n_features + 1)
                    # 如果说:
                    # 1. 当前的值是因子，且和之前的值相同，则重新生成因子
                    while True:
                        if (terminal in terminal_value_stack and terminal != self.n_features):
                            terminal = random_state.randint(self.n_features + 1)
                        else:
                            break

                else:
                    terminal = random_state.randint(self.n_features)

                if terminal == self.n_features:
                    terminal = round(random_state.uniform(*self.const_range),3)
                    while True:
                        if terminal==0:
                            terminal = random_state.uniform(*self.const_range)
                        else:
                            break
                    if self.const_range is None:
                        # We should never get here
                        raise ValueError('A constant was produced with '
                                         'const_range=None.')
                program.append(terminal)
                terminal_stack[-1] -= 1
                if terminal_stack[-1]>0:
                    terminal_value_stack.append(terminal)
                while terminal_stack[-1] == 0:
                    terminal_value_stack = []
                    terminal_stack.pop()
                    # 与 terminal_stack 保持同步弹出对应的 ts_depth 和 in_ts_subtree
                    if ts_depth_stack:
                        ts_depth_stack.pop()
                    if in_ts_subtree:
                        in_ts_subtree.pop()
                    if not terminal_stack:
                        return program
                    terminal_stack[-1] -= 1

        # We should never get here
        return None

    def _grammar_ok(self, program):
        """
        限制若干结构/金融语法规则：
        1) 任一路径上时间窗算子(is_ts_op=True)最多出现 1 次；
        2) dynamic_ts_sum 仅作用在 LIQ/SIZE 特征上；
        3) 其它 TS 算子仅作用在 PRICE/LIQ 特征上（避免 SIZE/OTHER）；
        4) 禁止明显无意义的模式：abs(abs(x)) / neg(neg(x)) / delta(delta(x))；
        5) delay 的连续链长度不超过 2；
        6) TS 子树内部仅允许一层“局部加工”算子：delta/delay/pos/neg_part/abs/tanh，
           不允许在 TS 子树内部再出现 add/sub/mul/safe_div 等复杂组合；
        7) 代数运算必须遵守维度一致性：
           - add/sub 只能作用在同类型特征上；
           - mul/safe_div 仅允许白名单组合（如 PRICE×LIQ→LIQ，LIQ/SIZE→LIQ 等），
             以避免产生没有金融含义的 cross-type ratio。
        """
        if program is None:
            return False

        # 若没有特征类型信息，退化为只检查 TS 深度和简单 pattern
        feature_types = self._feature_types

        add_ops = {'add', 'sub'}
        mul_ops = {'mul'}
        div_ops = {'div', 'safe_div'}
        neutral_types = {'CONST', 'OTHER', None}

        mul_allowed = {
            frozenset({'PRICE', 'LIQ'}): 'LIQ',      # price * volume ≈ amount
            frozenset({'PRICE', 'SIZE'}): 'SIZE',    # price * shares ≈ cap
            frozenset({'LIQ', 'SIZE'}): 'LIQ',       # liquidity scaled by size
            frozenset({'PRICE', 'PRICE'}): 'PRICE',  # price spread * price
            frozenset({'LIQ', 'LIQ'}): 'LIQ',
            frozenset({'SIZE', 'SIZE'}): 'SIZE',
            frozenset({'RET', 'PRICE'}): 'RET',
            frozenset({'RET', 'LIQ'}): 'RET',
        }

        div_allowed = {
            ('LIQ', 'SIZE'): 'LIQ',    # turnover / float_mkt_cap
            ('LIQ', 'PRICE'): 'LIQ',   # amount / price ≈ volume
            ('LIQ', 'LIQ'): 'LIQ',     # liquidity ratios
            ('PRICE', 'PRICE'): 'PRICE',  # amplitude / price
            ('PRICE', 'LIQ'): 'PRICE',    # price normalized by liquidity
            ('RET', 'PRICE'): 'RET',
        }

        def _pass_through(child_types, default='OTHER'):
            if not child_types:
                return default
            for c in child_types:
                if c not in {None}:
                    return c
            return default

        def _infer_type(func, child_types):
            if func is None:
                return 'OTHER'
            name = getattr(func, 'name', None)

            if name in add_ops:
                comparable = [c for c in child_types if c not in neutral_types]
                if len(set(comparable)) > 1:
                    return None
                if comparable:
                    base = comparable[0]
                else:
                    base = _pass_through(child_types)
                return base or 'OTHER'

            if name in mul_ops:
                comparable = [c for c in child_types if c not in neutral_types]
                if not comparable:
                    return _pass_through(child_types)
                if len(comparable) == 1:
                    return comparable[0]
                key = frozenset(comparable[:2])
                if key in mul_allowed:
                    return mul_allowed[key]
                if len(set(comparable[:2])) == 1:
                    return comparable[0]
                return None

            if name in div_ops:
                comparable = [c for c in child_types if c not in neutral_types]
                if not comparable:
                    return _pass_through(child_types)
                if len(comparable) == 1:
                    return comparable[0]
                numerator, denominator = comparable[0], comparable[1]
                if numerator == denominator:
                    return numerator
                if (numerator, denominator) in div_allowed:
                    return div_allowed[(numerator, denominator)]
                return None

            if name in {'neg', 'abs', 'tanh', 'pos', 'neg_part', 'delay'}:
                return _pass_through(child_types)
            if name == 'delta':
                return 'RET'
            if name in {'dynamic_ts_mean', 'dynamic_ts_std', 'dynamic_ts_sum',
                        'ts_zscore', 'ts_rank', 'ts_ema'}:
                return _pass_through(child_types)
            return _pass_through(child_types)

        # 栈元素：{'pending': 剩余子节点数, 'ts_depth': 当前路径TS深度,
        #        'ts_parent': 顶层TS算子名或None, 'delay_chain': delay深度, 'func': 当前函数节点}
        stack = []
        try:
            for node in program:
                if isinstance(node, _Function):
                    if not stack:
                        parent_ts_depth = 0
                        parent_ts_parent = None
                        parent_delay_chain = 0
                    else:
                        parent_ctx = stack[-1]
                        parent_ts_depth = parent_ctx.get('ts_depth', 0)
                        parent_ts_parent = parent_ctx.get('ts_parent', None)
                        parent_delay_chain = parent_ctx.get('delay_chain', 0)

                    is_ts = bool(getattr(node, "is_ts_op", False))
                    curr_ts_depth = parent_ts_depth + (1 if is_ts else 0)
                    # 规则1：TS深度不得超过1
                    if curr_ts_depth > 1:
                        return False

                    # 当前路径有效的 TS 顶层算子名
                    if parent_ts_parent is not None:
                        ts_parent = parent_ts_parent
                    elif is_ts:
                        ts_parent = getattr(node, "name", None)
                    else:
                        ts_parent = None

                    # delay 链长度
                    if getattr(node, "name", "") == "delay":
                        curr_delay_chain = parent_delay_chain + 1
                    else:
                        curr_delay_chain = 0
                    # 规则5：delay(delay(delay(x))) 禁止（>2）
                    if curr_delay_chain > 2:
                        return False

                    # 规则4：一元函数的明显冗余模式
                    if stack:
                        parent_func = stack[-1].get('func', None)
                        if (parent_func is not None and
                                isinstance(parent_func, _Function) and
                                parent_func.arity == 1 and node.arity == 1 and
                                parent_func.name in {'abs', 'neg', 'delta'} and
                                node.name == parent_func.name):
                            return False

                    # 规则6：TS 子树内部只能是简单的一层局部加工算子
                    # ts_parent 非空表示当前节点位于某个 TS 顶层算子之下（不含 TS 本身）
                    if parent_ts_parent is not None and (not is_ts):
                        # 允许的局部算子：delta / delay / pos / neg_part / abs / tanh
                        allowed_inner = {'delta', 'delay', 'pos', 'neg_part', 'abs', 'tanh'}
                        if getattr(node, "name", None) not in allowed_inner:
                            return False

                    stack.append({
                        'pending': node.arity,
                        'ts_depth': curr_ts_depth,
                        'ts_parent': ts_parent,
                        'delay_chain': curr_delay_chain,
                        'func': node,
                        'child_types': [],
                    })
                else:
                    # 终端节点：特征索引或常数
                    if not stack:
                        continue
                    ctx = stack[-1]
                    ts_parent = ctx.get('ts_parent', None)
                    node_value_type = 'CONST'

                    if isinstance(node, int) and feature_types is not None:
                        if 0 <= node < len(feature_types):
                            f_type = feature_types[node]
                            # 规则2：dynamic_ts_sum 只允许 LIQ/SIZE
                            if ts_parent == 'dynamic_ts_sum':
                                if f_type not in {'LIQ', 'SIZE'}:
                                    return False
                            # 规则3：其它 TS 算子只允许 PRICE/LIQ
                            elif ts_parent is not None:
                                if f_type not in {'PRICE', 'LIQ'}:
                                    return False
                            node_value_type = f_type
                        else:
                            node_value_type = 'OTHER'
                    elif isinstance(node, int):
                        node_value_type = 'OTHER'

                    # 更新 pending 计数和栈
                    ctx['pending'] -= 1
                    ctx.setdefault('child_types', []).append(node_value_type)
                    while stack and stack[-1]['pending'] == 0:
                        completed = stack.pop()
                        func_node = completed.get('func', None)
                        child_types = completed.get('child_types', [])
                        node_type = _infer_type(func_node, child_types)
                        if node_type is None:
                            return False
                        if stack:
                            stack[-1]['pending'] -= 1
                            stack[-1].setdefault('child_types', []).append(node_type)

            # 所有节点遍历完毕，若 validate_program 已通过，则 grammar 也算通过
            return True
        except Exception:
            # 任意异常视为不通过，以便上层重建
            return False

    def validate_program(self):
        """Rough check that the embedded program in the object is valid."""
        terminals = [0]
        for node in self.program:
            if isinstance(node, _Function):
                terminals.append(node.arity)
            else:
                terminals[-1] -= 1
                while terminals[-1] == 0:
                    terminals.pop()
                    terminals[-1] -= 1
        return terminals == [-1]

    def __str__(self):
        """Overloads `print` output of the object to resemble a LISP tree."""
        terminals = [0]
        output = ''
        isRandomFunction = 0
        RandomFunctionStack = []
        # RandomFunctionStack[0].arity = 0
        for i, node in enumerate(self.program):
            if isinstance(node, _Function):
                RandomFunctionStack.append(deepcopy(node))
                terminals.append(node.arity)
                output += node.name + '('
            else:
                if isinstance(node, int):
                    if self.feature_names is None:
                        output += 'X%s' % node
                    else:
                        output += self.feature_names[node]
                else:
                    output += '%.3f' % node
                terminals[-1] -= 1

                if len(RandomFunctionStack)>0:
                    RandomFunctionStack[-1].arity -= 1
                    if RandomFunctionStack[-1].isRandom and RandomFunctionStack[-1].arity==0:
                        output += ',' +str( RandomFunctionStack[-1].baseConst)
                    
                while terminals[-1] == 0:
                    RandomFunctionStack.pop()
                    terminals.pop()

                    terminals[-1] -= 1
                    if len(RandomFunctionStack)>0:
                        RandomFunctionStack[-1].arity -= 1
                        output += ')'
                        if len(RandomFunctionStack)>0 and RandomFunctionStack[-1].isRandom:
                            output += ',' + str( RandomFunctionStack[-1].baseConst)
                    else:
                        output += ')'
                if i != len(self.program) - 1:
                    output += ', '

        return output

    def export_graphviz(self, fade_nodes=None):
        """Returns a string, Graphviz script for visualizing the program.

        Parameters
        ----------
        fade_nodes : list, optional
            A list of node indices to fade out for showing which were removed
            during evolution.

        Returns
        -------
        output : string
            The Graphviz script to plot the tree representation of the program.

        """
        terminals = []
        if fade_nodes is None:
            fade_nodes = []
        output = 'digraph program {\nnode [style=filled]\n'
        for i, node in enumerate(self.program):
            fill = '#cecece'
            if isinstance(node, _Function):
                if i not in fade_nodes:
                    fill = '#136ed4'
                terminals.append([node.arity, i])
                output += ('%d [label="%s", fillcolor="%s"] ;\n'
                           % (i, node.name, fill))
            else:
                if i not in fade_nodes:
                    fill = '#60a6f6'
                if isinstance(node, int):
                    if self.feature_names is None:
                        feature_name = 'X%s' % node
                    else:
                        feature_name = self.feature_names[node]
                    output += ('%d [label="%s", fillcolor="%s"] ;\n'
                               % (i, feature_name, fill))
                else:
                    output += ('%d [label="%.3f", fillcolor="%s"] ;\n'
                               % (i, node, fill))
                if i == 0:
                    # A degenerative program of only one node
                    return output + '}'
                terminals[-1][0] -= 1
                terminals[-1].append(i)
                while terminals[-1][0] == 0:
                    output += '%d -> %d ;\n' % (terminals[-1][1],
                                                terminals[-1][-1])
                    terminals[-1].pop()
                    if len(terminals[-1]) == 2:
                        parent = terminals[-1][-1]
                        terminals.pop()
                        if not terminals:
                            return output + '}'
                        terminals[-1].append(parent)
                        terminals[-1][0] -= 1

        # We should never get here
        return None

    def _depth(self):
        """Calculates the maximum depth of the program tree."""
        terminals = [0]
        depth = 1
        for node in self.program:
            if isinstance(node, _Function):
                terminals.append(node.arity)
                depth = max(len(terminals), depth)
            else:
                terminals[-1] -= 1
                while terminals[-1] == 0:
                    terminals.pop()
                    terminals[-1] -= 1
        return depth - 1

    def _length(self):
        """Calculates the number of functions and terminals in the program."""
        return len(self.program)

    def execute(self, X):
        """Execute the program according to X.

        Parameters
        ----------
        X : {array-like}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.

        Returns
        -------
        y_hats : array-like, shape = [n_samples]
            The result of executing the program on X.

        """
        # Check for single-node programs
        node = self.program[0]
        if isinstance(node, float):
            return np.repeat(node, X.shape[0])
        if isinstance(node, int):
            return X[:, node]

        apply_stack = []

        for node in self.program:

            if isinstance(node, _Function):
                apply_stack.append([node])
            else:
                # Lazily evaluate later
                apply_stack[-1].append(node)

            while len(apply_stack[-1]) == apply_stack[-1][0].arity + 1:
                # Apply functions that have sufficient arguments
                function = apply_stack[-1][0]
                terminals = [np.repeat(t, X.shape[0]) if isinstance(t, float)
                             else X[:, t] if isinstance(t, int)
                             else t for t in apply_stack[-1][1:]]
                intermediate_result = function(*terminals)
                if len(apply_stack) != 1:
                    apply_stack.pop()
                    apply_stack[-1].append(intermediate_result)
                else:
                    return intermediate_result

        # We should never get here
        return None
    
    def execute_3D(self, X):
        
        # Check for single-node programs
        node = self.program[0]

        if isinstance(node, float):
            return np.tile(node, (X.shape[0], X.shape[2]))
        if isinstance(node, int):
            return X[:, node, :]

        apply_stack = []

        for node in self.program:

            if isinstance(node, _Function):
                apply_stack.append([node])
            else:
                # Lazily evaluate later
                apply_stack[-1].append(node)

            while len(apply_stack[-1]) == apply_stack[-1][0].arity + 1:
                # Apply functions that have sufficient arguments
                function = apply_stack[-1][0]
                terminals = [np.tile(t, (X.shape[0],X.shape[2])) if isinstance(t, float)
            else X[:,t,:] if isinstance(t, int)
                else t for t in apply_stack[-1][1:]]
                intermediate_result = function(*terminals)
                if len(apply_stack) != 1:
                    apply_stack.pop()
                    apply_stack[-1].append(intermediate_result)
                else:
                    return intermediate_result

        # We should never get here
        return None


    def get_all_indices(self, n_samples=None, max_samples=None,
                        random_state=None):
        """Get the indices on which to evaluate the fitness of a program.

        Parameters
        ----------
        n_samples : int
            The number of samples.

        max_samples : int
            The maximum number of samples to use.

        random_state : RandomState instance
            The random number generator.

        Returns
        -------
        indices : array-like, shape = [n_samples]
            The in-sample indices.

        not_indices : array-like, shape = [n_samples]
            The out-of-sample indices.

        """
        if self._indices_state is None and random_state is None:
            raise ValueError('The program has not been evaluated for fitness '
                             'yet, indices not available.')

        if n_samples is not None and self._n_samples is None:
            self._n_samples = n_samples
        if max_samples is not None and self._max_samples is None:
            self._max_samples = max_samples
        if random_state is not None and self._indices_state is None:
            self._indices_state = random_state.get_state()

        indices_state = check_random_state(None)
        indices_state.set_state(self._indices_state)

        not_indices = sample_without_replacement(
            self._n_samples,
            self._n_samples - self._max_samples,
            random_state=indices_state)
        sample_counts = np.bincount(not_indices, minlength=self._n_samples)
        indices = np.where(sample_counts == 0)[0]

        return indices, not_indices

    def _indices(self):
        """Get the indices used to measure the program's fitness."""
        return self.get_all_indices()[0]

    def raw_fitness(self, X, y, sample_weight):
        """Evaluate the raw fitness of the program according to X, y.

        Parameters
        ----------
        X : {array-like}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.

        y : array-like, shape = [n_samples]
            Target values.

        sample_weight : array-like, shape = [n_samples]
            Weights applied to individual samples.

        Returns
        -------
        raw_fitness : float
            The raw fitness of the program.

        """
        try:
            y_pred = self.execute(X)
            
            # 检查 execute 是否返回了有效结果
            if y_pred is None:
                return 0.0 if self.metric.greater_is_better else np.inf
            
            # 检查是否包含过多 NaN 或 Inf
            if not np.isfinite(y_pred).any():
                return 0.0 if self.metric.greater_is_better else np.inf
            
            # 替换 NaN 和 Inf 为 0（保守处理）
            y_pred = np.nan_to_num(y_pred, nan=0.0, posinf=0.0, neginf=0.0)
            
            if self.transformer:
                y_pred = self.transformer(y_pred)
            
            raw_fitness = self.metric(y, y_pred, sample_weight)
            
            # 确保 fitness 是有效数值
            if not np.isfinite(raw_fitness):
                return 0.0 if self.metric.greater_is_better else np.inf
            
            return raw_fitness
            
        except Exception as e:
            # 任何异常都返回最差的 fitness
            return 0.0 if self.metric.greater_is_better else np.inf

    def raw_fitness_3D(self, X, y, sample_weight):
        """Evaluate the raw fitness of the program according to X, y.

        Parameters
        ----------
        X : {array-like}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.

        y : array-like, shape = [n_samples]
            Target values.

        sample_weight : array-like, shape = [n_samples]
            Weights applied to individual samples.

        Returns
        -------
        raw_fitness : float
            The raw fitness of the program.

        """
        try:
            y_pred = self.execute_3D(X)
            
            # 检查 execute_3D 是否返回了有效结果
            if y_pred is None:
                return 0.0 if self.metric.greater_is_better else np.inf
            
            # 检查是否包含过多 NaN 或 Inf
            if not np.isfinite(y_pred).any():
                return 0.0 if self.metric.greater_is_better else np.inf
            
            # 替换 NaN 和 Inf 为 0（保守处理）
            y_pred = np.nan_to_num(y_pred, nan=0.0, posinf=0.0, neginf=0.0)
            
            if self.transformer:
                y_pred = self.transformer(y_pred)
            
            raw_fitness = self.metric(y, y_pred, sample_weight)
            
            # 确保 fitness 是有效数值
            if not np.isfinite(raw_fitness):
                return 0.0 if self.metric.greater_is_better else np.inf
            
            return raw_fitness
            
        except Exception as e:
            # 任何异常都返回最差的 fitness
            return 0.0 if self.metric.greater_is_better else np.inf


    def fitness(self, parsimony_coefficient=None, oob_weight=0.0):
        """Evaluate the blended fitness (IS/OOB) of the program according to X, y.

        Parameters
        ----------
        parsimony_coefficient : float, optional
            If automatic parsimony is being used, the computed value according
            to the population. Otherwise the initialized value is used.
        
        oob_weight : float, optional (default=0.0)
            Weight for OOB fitness blending (0.0 = use only training fitness,
            0.3 = 30% OOB + 70% training). Helps reduce overfitting.

        Returns
        -------
        fitness : float
            The penalized fitness of the program.

        """
        # 确保 raw_fitness_ 有效
        if self.raw_fitness_ is None or not np.isfinite(self.raw_fitness_):
            return 0.0 if self.metric.greater_is_better else np.inf
        
        # Blend OOB fitness if available and oob_weight > 0
        if oob_weight > 0 and hasattr(self, 'oob_fitness_'):
            oob_fit = self.oob_fitness_
            # 确保 oob_fitness_ 也有效
            if oob_fit is None or not np.isfinite(oob_fit):
                oob_fit = self.raw_fitness_
            base_fitness = (1 - oob_weight) * self.raw_fitness_ + oob_weight * oob_fit
        else:
            base_fitness = self.raw_fitness_

        # Legacy structural penalties are disabled; return blended fitness only.
        # 最后再次确保返回值有效
        if not np.isfinite(base_fitness):
            return 0.0 if self.metric.greater_is_better else np.inf
        
        return base_fitness

    def get_subtree(self, random_state, program=None):
        """Get a random subtree from the program.

        Parameters
        ----------
        random_state : RandomState instance
            The random number generator.

        program : list, optional (default=None)
            The flattened tree representation of the program. If None, the
            embedded tree in the object will be used.

        Returns
        -------
        start, end : tuple of two ints
            The indices of the start and end of the random subtree.

        """
        if program is None:
            program = self.program
        # Choice of crossover points follows Koza's (1992) widely used approach
        # of choosing functions 90% of the time and leaves 10% of the time.
        probs = np.array([0.9 if isinstance(node, _Function) else 0.1
                          for node in program])
        probs = np.cumsum(probs / probs.sum())
        start = np.searchsorted(probs, random_state.uniform())

        stack = 1
        end = start
        while stack > end - start:
            node = program[end]
            if isinstance(node, _Function):
                stack += node.arity
            end += 1

        return start, end

    def reproduce(self):
        """Return a copy of the embedded program."""
        return copy(self.program)

    def crossover(self, donor, random_state):
        """Perform the crossover genetic operation on the program.

        Crossover selects a random subtree from the embedded program to be
        replaced. A donor also has a subtree selected at random and this is
        inserted into the original parent to form an offspring.

        Parameters
        ----------
        donor : list
            The flattened tree representation of the donor program.

        random_state : RandomState instance
            The random number generator.

        Returns
        -------
        program : list
            The flattened tree representation of the program.

        """
        # Get a subtree to replace
        start, end = self.get_subtree(random_state)
        removed = range(start, end)
        # Get a subtree to donate
        donor_start, donor_end = self.get_subtree(random_state, donor)
        donor_removed = list(set(range(len(donor))) -
                             set(range(donor_start, donor_end)))
        # Insert genetic material from donor
        return (self.program[:start] +
                donor[donor_start:donor_end] +
                self.program[end:]), removed, donor_removed

    def subtree_mutation(self, random_state):
        """Perform the subtree mutation operation on the program.

        Subtree mutation selects a random subtree from the embedded program to
        be replaced. A donor subtree is generated at random and this is
        inserted into the original parent to form an offspring. This
        implementation uses the "headless chicken" method where the donor
        subtree is grown using the initialization methods and a subtree of it
        is selected to be donated to the parent.

        Parameters
        ----------
        random_state : RandomState instance
            The random number generator.

        Returns
        -------
        program : list
            The flattened tree representation of the program.

        """
        # Build a new naive program
        chicken = self.build_program(random_state)
        # Do subtree mutation via the headless chicken method!
        return self.crossover(chicken, random_state)

    def hoist_mutation(self, random_state):
        """Perform the hoist mutation operation on the program.

        Hoist mutation selects a random subtree from the embedded program to
        be replaced. A random subtree of that subtree is then selected and this
        is 'hoisted' into the original subtrees location to form an offspring.
        This method helps to control bloat.

        Parameters
        ----------
        random_state : RandomState instance
            The random number generator.

        Returns
        -------
        program : list
            The flattened tree representation of the program.

        """
        # Get a subtree to replace
        start, end = self.get_subtree(random_state)
        subtree = self.program[start:end]
        # Get a subtree of the subtree to hoist
        sub_start, sub_end = self.get_subtree(random_state, subtree)
        hoist = subtree[sub_start:sub_end]
        # Determine which nodes were removed for plotting
        removed = list(set(range(start, end)) -
                       set(range(start + sub_start, start + sub_end)))
        return self.program[:start] + hoist + self.program[end:], removed

    def point_mutation(self, random_state):
        """Perform the point mutation operation on the program.

        Point mutation selects random nodes from the embedded program to be
        replaced. Terminals are replaced by other terminals and functions are
        replaced by other functions that require the same number of arguments
        as the original node. The resulting tree forms an offspring.

        Parameters
        ----------
        random_state : RandomState instance
            The random number generator.

        Returns
        -------
        program : list
            The flattened tree representation of the program.

        """
        program = copy(self.program)

        # Get the nodes to modify
        mutate = np.where(random_state.uniform(size=len(program)) <
                          self.p_point_replace)[0]

        for node in mutate:
            if isinstance(program[node], _Function):
                arity = program[node].arity
                # Find a valid replacement with same arity
                replacement = len(self.arities[arity])
                replacement = random_state.randint(replacement)
                replacement = self.arities[arity][replacement]
                replacement.baseConst = program[node].baseConst
                program[node] = replacement
            else:
                # We've got a terminal, add a const or variable
                if self.const_range is not None:
                    terminal = random_state.randint(self.n_features + 1)
                else:
                    terminal = random_state.randint(self.n_features)
                if terminal == self.n_features:
                    terminal = random_state.uniform(*self.const_range)
                    if self.const_range is None:
                        # We should never get here
                        raise ValueError('A constant was produced with '
                                         'const_range=None.')
                program[node] = terminal

        return program, list(mutate)

    depth_ = property(_depth)
    length_ = property(_length)
    indices_ = property(_indices)
