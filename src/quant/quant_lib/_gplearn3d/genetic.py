"""Genetic Programming in Python, with a scikit-learn inspired API

The :mod:`gplearn.genetic` module implements Genetic Programming. These
are supervised learning methods based on applying evolutionary operations on
computer programs.
"""

import itertools
from abc import ABCMeta, abstractmethod
from collections import Counter
from time import time
from warnings import warn
import warnings
import re
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import rankdata
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin, ClassifierMixin
from sklearn.exceptions import NotFittedError
from sklearn.utils import compute_sample_weight
from sklearn.utils.validation import check_array, _check_sample_weight
from sklearn.utils.multiclass import check_classification_targets
from .utils import _syntax_adapter, _partition_estimators, check_random_state, check_floats
from ._program import _Program
from .functions import _Function
from .fitness import _fitness_map, _Fitness, _extra_map, compute_ic
from .functions import _function_map, _Function, sig1 as sigmoid
from .more_functions import _extra_function_map


_all_func_dictionary = dict(_function_map, **_extra_function_map)

__all__ = ['SymbolicRegressor', 'SymbolicClassifier', 'SymbolicTransformer']

MAX_INT = np.iinfo(np.int32).max


def _build_time_series_cv_splits(train_idx, n_folds=4):
    train_idx = np.asarray(train_idx, dtype=int)
    if train_idx.size == 0 or n_folds <= 1:
        return []
    # 等分为若干连续时间段
    segments = [seg for seg in np.array_split(train_idx, n_folds) if seg.size > 0]
    if len(segments) <= 1:
        return []
    splits = []
    # 从第二段开始做验证，保证每折都有“过去→未来”的方向
    for k in range(1, len(segments)):
        val_idx = segments[k]
        if val_idx.size == 0:
            continue
        train_parts = segments[:k]
        train_concat = np.concatenate(train_parts) if len(train_parts) == 1 else np.concatenate(train_parts, axis=0)
        if train_concat.size == 0:
            continue
        splits.append((np.asarray(train_concat, dtype=int), np.asarray(val_idx, dtype=int)))
    return splits


def _parallel_evolve_3D(n_programs, parents, X, y, sample_weight, seeds, params):
    """Private function used to build a batch of programs within a job."""
    # 这里是并行调用Parallel 进行改进的地方，通过修改这个部分，能够把数据结构进行修正
    n_dates, n_features, n_stocks = X.shape

    # Unpack parameters
    tournament_size = params['tournament_size']
    function_set = params['function_set']
    arities = params['arities']
    init_depth = params['init_depth']
    init_method = params['init_method']
    const_range = params['const_range']
    metric = params['_metric']
    transformer = params['_transformer']
    parsimony_coefficient = params['parsimony_coefficient']
    method_probs = params['method_probs']
    p_point_replace = params['p_point_replace']
    max_samples = params['max_samples']
    feature_names = params['feature_names']
    
    max_samples = int(max_samples * n_dates)


    def _tournament():
        """Find the fittest individual from a sub-population."""
        contenders = random_state.randint(0, len(parents), tournament_size)
        # Robust selection: handle None parents or None/NaN/Inf fitness gracefully
        scored = []
        for p_idx in contenders:
            par = parents[p_idx]
            fit_val = None if par is None else getattr(par, 'fitness_', None)
            if fit_val is None or not np.isfinite(fit_val):
                fit = -np.inf if metric.greater_is_better else np.inf
            else:
                fit = float(fit_val)
            scored.append((fit, p_idx))
        if metric.greater_is_better:
            parent_index = max(scored, key=lambda x: x[0])[1]
        else:
            parent_index = min(scored, key=lambda x: x[0])[1]
        return parents[parent_index], parent_index

    # Build programs
    programs = []

    for i in range(n_programs):

        random_state = check_random_state(seeds[i])
        parent_age = 0

        if parents is None:
            program = None
            genome = None
        else:
            method = random_state.uniform()
            parent, parent_index = _tournament()
            parent_age = int(getattr(parent, 'age_', 0))

            if method < method_probs[0]:
                # crossover
                donor, donor_index = _tournament()
                program, removed, remains = parent.crossover(donor.program,
                                                             random_state)
                genome = {'method': 'Crossover',
                          'parent_idx': parent_index,
                          'parent_nodes': removed,
                          'donor_idx': donor_index,
                          'donor_nodes': remains}
            elif method < method_probs[1]:
                # subtree_mutation
                program, removed, _ = parent.subtree_mutation(random_state)
                genome = {'method': 'Subtree Mutation',
                          'parent_idx': parent_index,
                          'parent_nodes': removed}
            elif method < method_probs[2]:
                # hoist_mutation
                program, removed = parent.hoist_mutation(random_state)
                genome = {'method': 'Hoist Mutation',
                          'parent_idx': parent_index,
                          'parent_nodes': removed}
            elif method < method_probs[3]:
                # point_mutation
                program, mutated = parent.point_mutation(random_state)
                genome = {'method': 'Point Mutation',
                          'parent_idx': parent_index,
                          'parent_nodes': mutated}
            else:
                # reproduction
                program = parent.reproduce()
                genome = {'method': 'Reproduction',
                          'parent_idx': parent_index,
                          'parent_nodes': []}

        program = _Program(function_set=function_set,
                           arities=arities,
                           init_depth=init_depth,
                           init_method=init_method,
                           n_features=n_features,
                           metric=metric,
                           transformer=transformer,
                           const_range=const_range,
                           p_point_replace=p_point_replace,
                           parsimony_coefficient=parsimony_coefficient,
                           feature_names=feature_names,
                           random_state=random_state,
                           program=program,
                           op_penalty_map=params.get('op_penalty_map'),
                           feature_min_unique=params.get('feature_min_unique', 0),
                           feature_penalty=params.get('feature_penalty', 0.0),
                           depth_threshold=params.get('depth_threshold'),
                           depth_penalty=params.get('depth_penalty', 0.0))

        program.parents = genome
        # 新个体年龄：从父代继承并+1；初始化代或无父代则为0
        try:
            program.age_ = 0 if parents is None else max(0, parent_age + 1)
        except Exception:
            program.age_ = 0

        # Draw samples, using sample weights, and then fit
        if sample_weight is None:
            curr_sample_weight = np.ones(n_dates)
        else:
            curr_sample_weight = sample_weight.copy()
        # 构建训练/OOB样本权重：优先使用外部时间序列切分
        if sample_weight is not None and np.any(curr_sample_weight == 0) and np.any(curr_sample_weight > 0):
            # 使用外部提供的时间切分（训练=1，OOB=0）
            oob_sample_weight = (curr_sample_weight == 0).astype(float)
        else:
            # 使用内部随机采样逻辑
            oob_sample_weight = curr_sample_weight.copy()
            indices, not_indices = program.get_all_indices(n_dates,
                                                           max_samples,
                                                           random_state)
            curr_sample_weight[not_indices] = 0
            oob_sample_weight[indices] = 0

        program.raw_fitness_ = program.raw_fitness_3D(X, y, curr_sample_weight)
        # 只要存在 OOB 样本就计算 OOB fitness（支持外部切分或内部采样）
        if np.any(oob_sample_weight > 0):
            program.oob_fitness_ = program.raw_fitness_3D(X, y, oob_sample_weight)
        programs.append(program)
    
    return programs

def _parallel_evolve(n_programs, parents, X, y, sample_weight, seeds, params):
    """Private function used to build a batch of programs within a job."""
    n_samples, n_features = X.shape
    # Unpack parameters
    tournament_size = params['tournament_size']
    function_set = params['function_set']
    arities = params['arities']
    init_depth = params['init_depth']
    init_method = params['init_method']
    const_range = params['const_range']
    metric = params['_metric']
    transformer = params['_transformer']
    parsimony_coefficient = params['parsimony_coefficient']
    method_probs = params['method_probs']
    p_point_replace = params['p_point_replace']
    max_samples = params['max_samples']
    feature_names = params['feature_names']
    
    max_samples = int(max_samples * n_samples)

    def _tournament():
        """Find the fittest individual from a sub-population."""
        contenders = random_state.randint(0, len(parents), tournament_size)
        # Robust selection: guard against None parents or None/NaN/Inf fitness
        scored = []
        for p_idx in contenders:
            par = parents[p_idx]
            fit_val = None if par is None else getattr(par, 'fitness_', None)
            if fit_val is None or not np.isfinite(fit_val):
                fit = -np.inf if metric.greater_is_better else np.inf
            else:
                fit = float(fit_val)
            scored.append((fit, p_idx))
        if metric.greater_is_better:
            parent_index = max(scored, key=lambda x: x[0])[1]
        else:
            parent_index = min(scored, key=lambda x: x[0])[1]
        return parents[parent_index], parent_index

    # Build programs
    programs = []

    for i in range(n_programs):

        random_state = check_random_state(seeds[i])
        parent_age = 0

        if parents is None:
            program = None
            genome = None
        else:
            method = random_state.uniform()
            parent, parent_index = _tournament()
            parent_age = int(getattr(parent, 'age_', 0))

            if method < method_probs[0]:
                # crossover
                donor, donor_index = _tournament()
                program, removed, remains = parent.crossover(donor.program,
                                                             random_state)
                genome = {'method': 'Crossover',
                          'parent_idx': parent_index,
                          'parent_nodes': removed,
                          'donor_idx': donor_index,
                          'donor_nodes': remains}
            elif method < method_probs[1]:
                # subtree_mutation
                program, removed, _ = parent.subtree_mutation(random_state)
                genome = {'method': 'Subtree Mutation',
                          'parent_idx': parent_index,
                          'parent_nodes': removed}
            elif method < method_probs[2]:
                # hoist_mutation
                program, removed = parent.hoist_mutation(random_state)
                genome = {'method': 'Hoist Mutation',
                          'parent_idx': parent_index,
                          'parent_nodes': removed}
            elif method < method_probs[3]:
                # point_mutation
                program, mutated = parent.point_mutation(random_state)
                genome = {'method': 'Point Mutation',
                          'parent_idx': parent_index,
                          'parent_nodes': mutated}
            else:
                # reproduction
                program = parent.reproduce()
                genome = {'method': 'Reproduction',
                          'parent_idx': parent_index,
                          'parent_nodes': []}

        program = _Program(function_set=function_set,
                           arities=arities,
                           init_depth=init_depth,
                           init_method=init_method,
                           n_features=n_features,
                           metric=metric,
                           transformer=transformer,
                           const_range=const_range,
                           p_point_replace=p_point_replace,
                           parsimony_coefficient=parsimony_coefficient,
                           feature_names=feature_names,
                           random_state=random_state,
                           program=program,
                           op_penalty_map=params.get('op_penalty_map'),
                           feature_min_unique=params.get('feature_min_unique', 0),
                           feature_penalty=params.get('feature_penalty', 0.0),
                           depth_threshold=params.get('depth_threshold'),
                           depth_penalty=params.get('depth_penalty', 0.0))

        program.parents = genome
        try:
            program.age_ = 0 if parents is None else max(0, parent_age + 1)
        except Exception:
            program.age_ = 0

        # Draw samples, using sample weights, and then fit
        if sample_weight is None:
            curr_sample_weight = np.ones((n_samples,))
        else:
            curr_sample_weight = sample_weight.copy()
        oob_sample_weight = curr_sample_weight.copy()

        indices, not_indices = program.get_all_indices(n_samples,
                                                       max_samples,
                                                       random_state)

        curr_sample_weight[not_indices] = 0
        oob_sample_weight[indices] = 0

        program.raw_fitness_ = program.raw_fitness(X, y, curr_sample_weight)
        if max_samples < n_samples:
            # Calculate OOB fitness
            program.oob_fitness_ = program.raw_fitness(X, y, oob_sample_weight)

        programs.append(program)

    return programs


class BaseSymbolic(BaseEstimator, metaclass=ABCMeta):
    """Base class for symbolic regression / classification estimators.

    Warning: This class should not be used directly.
    Use derived classes instead.

    """

    @abstractmethod
    def __init__(self,
                 *,
                 population_size=1000,
                 hall_of_fame=None,
                 n_components=None,
                 generations=20,
                 tournament_size=20,
                 stopping_criteria=0.0,
                 const_range=(-1., 1.),
                 init_depth=(2, 6),
                 init_method='half and half',
                 function_set=('add', 'sub', 'mul', 'div'),
                 transformer=None,
                 metric='mean absolute error',
                 parsimony_coefficient=0.001,
                 p_crossover=0.9,
                 p_subtree_mutation=0.01,
                 p_hoist_mutation=0.01,
                 p_point_mutation=0.01,
                 p_point_replace=0.05,
                 max_samples=1.0,
                 oob_weight=0.0,
                 class_weight=None,
                 feature_names=None,
                 warm_start=False,
                 low_memory=False,
                 n_jobs=1,
                 verbose=0,
                 random_state=None,
                 # Diversity/structure penalty configs
                 op_penalty_map=None,
                 feature_min_unique=0,
                 feature_penalty=0.0,
                 depth_threshold=None,
                 depth_penalty=0.0,
                 # Beam Search config
                 beam_width=None,
                 # Composite fitness (IS/OOB + penalties)
                 use_composite_in_engine=False,
                 w_is=1.0,
                 w_oob=0.0,
                 w_icir_comp=0.0,
                 lambda_complexity=0.0,
                 mu_instability=0.0,
                 complexity_a1=1.0,
                 complexity_a2=1.0,
                 instability_K=4,
                 fitness_epsilon=1e-9,
                 oob_ir_min_clip=-1.0,
                 # Beam mixing: elites + beam exploitation + global exploration
                 beam_ratio=None,
                 # Age-layered protection for immigrants / newcomers
                 age_protection=False,
                 age_threshold=2,
                 young_beam_ratio=0.3):

        self.population_size = population_size
        self.hall_of_fame = hall_of_fame
        self.n_components = n_components
        self.generations = generations
        self.tournament_size = tournament_size
        self.stopping_criteria = stopping_criteria
        self.const_range = const_range
        self.init_depth = init_depth
        self.init_method = init_method
        self.function_set = function_set
        self.transformer = transformer
        self.metric = metric
        self.parsimony_coefficient = parsimony_coefficient
        self.p_crossover = p_crossover
        self.p_subtree_mutation = p_subtree_mutation
        self.p_hoist_mutation = p_hoist_mutation
        self.p_point_mutation = p_point_mutation
        self.p_point_replace = p_point_replace
        self.max_samples = max_samples
        # Store OOB blending weight for downstream fitness blending
        self.oob_weight = oob_weight
        self.class_weight = class_weight
        self.feature_names = feature_names
        self.warm_start = warm_start
        self.low_memory = low_memory
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.random_state = random_state
        # Penalty knobs
        self.op_penalty_map = op_penalty_map
        self.feature_min_unique = feature_min_unique
        self.feature_penalty = feature_penalty
        self.depth_threshold = depth_threshold
        self.depth_penalty = depth_penalty
        # Beam Search: 保留 top-k 个体进行下一代进化
        self.beam_width = beam_width

        # Composite fitness controls
        self.use_composite_in_engine = use_composite_in_engine
        self.w_is = w_is
        self.w_oob = w_oob
        self.w_icir_comp = w_icir_comp
        self.lambda_complexity = lambda_complexity
        self.mu_instability = mu_instability
        self.complexity_a1 = complexity_a1
        self.complexity_a2 = complexity_a2
        self.instability_K = instability_K
        self.fitness_epsilon = fitness_epsilon
        self.oob_ir_min_clip = oob_ir_min_clip
        # Beam mixing ratio (None = keep legacy behavior)
        self.beam_ratio = beam_ratio
        # Age-layered protection controls
        self.age_protection = bool(age_protection)
        self.age_threshold = int(age_threshold)
        self.young_beam_ratio = float(young_beam_ratio)

        
    def _verbose_reporter(self, run_details=None):
        """A report of the progress of the evolution process.

        Parameters
        ----------
        run_details : dict
            Information about the evolution.

        """
        if run_details is None:
            print('    |{:^25}|{:^42}|'.format('Population Average',
                                               'Best Individual'))
            print('-' * 4 + ' ' + '-' * 25 + ' ' + '-' * 42 + ' ' + '-' * 10)
            line_format = '{:>4} {:>8} {:>16} {:>8} {:>16} {:>16} {:>10}'
            print(line_format.format('Gen', 'Length', 'Fitness', 'Length',
                                     'Fitness', 'OOB Fitness', 'Time Left'))

        else:
            # Estimate remaining time for run
            gen = run_details['generation'][-1]
            generation_time = run_details['generation_time'][-1]
            remaining_time = (self.generations - gen - 1) * generation_time
            if remaining_time > 60:
                remaining_time = '{0:.2f}m'.format(remaining_time / 60.0)
            else:
                remaining_time = '{0:.2f}s'.format(remaining_time)

            oob_fitness = 'N/A'
            line_format = '{:4d} {:8.2f} {:16g} {:8d} {:16g} {:>16} {:>10}'
            # 显示已经计算到的 OOB fitness（支持外部/内部切分）
            if len(run_details['best_oob_fitness']) > 0 and not np.isnan(run_details['best_oob_fitness'][-1]):
                oob_fitness = run_details['best_oob_fitness'][-1]
                line_format = '{:4d} {:8.2f} {:16g} {:8d} {:16g} {:16g} {:>10}'

            print(line_format.format(run_details['generation'][-1],
                                     run_details['average_length'][-1],
                                     run_details['average_fitness'][-1],
                                     run_details['best_length'][-1],
                                     run_details['best_fitness'][-1],
                                     oob_fitness,
                                     remaining_time))

    def fit_3D(self, X, y, baseline=None, sample_weight=None, need_parallel=True):
        """Fit the Genetic Program according to X, y.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.

        y : array-like, shape = [n_samples]
            Target values.

        sample_weight : array-like, shape = [n_samples], optional
            Weights applied to individual samples.

        Returns
        -------
        self : object
            Returns self.

        """
        random_state = check_random_state(self.random_state)


        hall_of_fame = self.hall_of_fame
        if hall_of_fame is None:
            hall_of_fame = self.population_size
        if hall_of_fame > self.population_size or hall_of_fame < 1:
            raise ValueError('hall_of_fame (%d) must be less than or equal to '
                             'population_size (%d).' % (self.hall_of_fame,
                                                        self.population_size))
        n_components = self.n_components
        if n_components is None:
            n_components = hall_of_fame
        if n_components > hall_of_fame or n_components < 1:
            raise ValueError('n_components (%d) must be less than or equal to '
                             'hall_of_fame (%d).' % (self.n_components,
                                                     self.hall_of_fame))

        self._function_set = []
        for function in self.function_set:
            if isinstance(function, str):
                if function not in _all_func_dictionary:
                    raise ValueError('invalid function name %s found in '
                                     '`function_set`.' % function)
                self._function_set.append(_all_func_dictionary[function])
            elif isinstance(function, _Function):
                self._function_set.append(function)
            else:
                raise ValueError('invalid type %s found in `function_set`.'
                                 % type(function))
        if not self._function_set:
            raise ValueError('No valid functions found in `function_set`.')

        # For point-mutation to find a compatible replacement node
        self._arities = {}
        for function in self._function_set:
            arity = function.arity
            self._arities[arity] = self._arities.get(arity, [])
            self._arities[arity].append(function)

        if isinstance(self.metric, _Fitness):
            self._metric = self.metric
        elif isinstance(self, RegressorMixin):
            base_method = ('mean absolute error', 'mse', 'rmse',
                           'pearson', 'spearman')
            extra_method = tuple(_extra_map.keys())
            total_method = base_method + extra_method
            if self.metric not in total_method:
                raise ValueError('Unsupported metric: %s' % self.metric)
            self._metric = _fitness_map[self.metric]
        elif isinstance(self, ClassifierMixin):
            if self.metric != 'log loss':
                raise ValueError('Unsupported metric: %s' % self.metric)
            self._metric = _fitness_map[self.metric]
        elif isinstance(self, TransformerMixin):
            base_method = ('pearson', 'spearman')
            extra_method = tuple(_extra_map.keys())
            total_method = base_method + extra_method
            if self.metric not in total_method:
                raise ValueError('Unsupported metric: %s' % self.metric)
            self._metric = _fitness_map[self.metric]

        self._method_probs = np.array([self.p_crossover,
                                       self.p_subtree_mutation,
                                       self.p_hoist_mutation,
                                       self.p_point_mutation])
        self._method_probs = np.cumsum(self._method_probs)

        if self._method_probs[-1] > 1:
            raise ValueError('The sum of p_crossover, p_subtree_mutation, '
                             'p_hoist_mutation and p_point_mutation should '
                             'total to 1.0 or less.')

        if self.init_method not in ('half and half', 'grow', 'full'):
            raise ValueError('Valid program initializations methods include '
                             '"grow", "full" and "half and half". Given %s.'
                             % self.init_method)

        if not ((isinstance(self.const_range, tuple) and
                 len(self.const_range) == 2) or self.const_range is None):
            raise ValueError('const_range should be a tuple with length two, '
                             'or None.')

        if (not isinstance(self.init_depth, tuple) or
                len(self.init_depth) != 2):
            raise ValueError('init_depth should be a tuple with length two.')
        if self.init_depth[0] > self.init_depth[1]:
            raise ValueError('init_depth should be in increasing numerical '
                             'order: (min_depth, max_depth).')

        if self.transformer is not None:
            if isinstance(self.transformer, _Function):
                self._transformer = self.transformer
            elif self.transformer == 'sigmoid':
                self._transformer = sigmoid
            else:
                raise ValueError('Invalid `transformer`. Expected either '
                                 '"sigmoid" or _Function object, got %s' %
                                 type(self.transformer))
            if self._transformer.arity != 1:
                raise ValueError('Invalid arity for `transformer`. Expected 1, '
                                 'got %d.' % (self._transformer.arity))

        params = self.get_params()
        params['_metric'] = self._metric
        if hasattr(self, '_transformer'):
            params['_transformer'] = self._transformer
        else:
            params['_transformer'] = None
        params['function_set'] = self._function_set
        params['arities'] = self._arities
        params['method_probs'] = self._method_probs

        if not self.warm_start or not hasattr(self, '_programs'):
            # Free allocated memory, if any
            self._programs = []
            self.run_details_ = {'generation': [],
                                 'average_length': [],
                                 'average_fitness': [],
                                 'best_length': [],
                                 'best_fitness': [],
                                 'best_oob_fitness': [],
                                 'generation_time': []}

        prior_generations = len(self._programs)
        n_more_generations = self.generations - prior_generations

        if n_more_generations < 0:
            raise ValueError('generations=%d must be larger or equal to '
                             'len(_programs)=%d when warm_start==True'
                             % (self.generations, len(self._programs)))
        elif n_more_generations == 0:
            fitness = [program.raw_fitness_ for program in self._programs[-1]]
            warn('Warm-start fitting without increasing n_estimators does not '
                 'fit new programs.')

        if self.warm_start:
            # Generate and discard seeds that would have been produced on the
            # initial fit call.
            for i in range(len(self._programs)):
                _ = random_state.randint(MAX_INT, size=self.population_size)

        if self.verbose:
            # Print header fields
            self._verbose_reporter()
            import sys
            sys.stdout.flush()  # 强制刷新输出
            
        #---------------genetic program starts from here---------------
        if baseline is not None:
            self._total_program = []
        for gen in range(prior_generations, self.generations):
            # 添加调试输出，确认进入第几代循环
            if self.verbose and gen == prior_generations:
                print(f"\n[DEBUG] 进入遗传规划循环: gen={gen}, generations={self.generations}, population_size={self.population_size}", flush=True)

            start_time = time()

            if gen == 0:
                parents = None
            else:
                parents = self._programs[gen - 1]

            # Parallel loop
            n_jobs, n_programs, starts = _partition_estimators(
                self.population_size, self.n_jobs)
            seeds = random_state.randint(MAX_INT, size=self.population_size)
            if need_parallel:
                population = Parallel(n_jobs=n_jobs,
                                      verbose=int(self.verbose > 1))(
                    delayed(_parallel_evolve_3D)(n_programs[i],
                                              parents,
                                              X,
                                              y,
                                              sample_weight,
                                              seeds[starts[i]:starts[i + 1]],
                                              params)
                    for i in range(n_jobs))
            else:
                population = []
                for i in range(n_jobs):
                    population.append(
                        _parallel_evolve_3D(n_programs[i], parents, X, y, sample_weight, seeds[starts[i]:starts[i + 1]],
                                            params))

            # Reduce, maintaining order across different n_jobs
            population = list(itertools.chain.from_iterable(population))
            
            if baseline is not None:
                for program in population:
                    if program.raw_fitness_ > baseline:
                        self._total_program.append(program)

            fitness = [program.raw_fitness_ for program in population]
            length = [program.length_ for program in population]

            if self.use_composite_in_engine:
                # 复合适应度：基于当前种群的复杂度/不稳定度归一化（始终启用）
                # 复杂度原始值：a1 * nodes + a2 * depth + a3 * op_complexity
                nodes_arr = np.array([p.length_ for p in population], dtype=float)
                depth_arr = np.array([getattr(p, 'depth_', 0) for p in population], dtype=float)

                # 算子级复杂度：按每个节点的 complexity_weight 求和
                op_complexity_list = []
                for prog in population:
                    score = 0.0
                    try:
                        for node in getattr(prog, "program", []):
                            if isinstance(node, _Function):
                                score += getattr(node, "complexity_weight", 1.0)
                    except Exception:
                        pass
                    op_complexity_list.append(score)
                op_complexity_arr = np.asarray(op_complexity_list, dtype=float)

                # complexity_a2 继续对应 depth；增加 complexity_a3 作为算子复杂度系数
                complexity_a1 = float(getattr(self, "complexity_a1", 1.0))
                complexity_a2 = float(getattr(self, "complexity_a2", 1.0))
                complexity_a3 = float(getattr(self, "complexity_a3", 1.0))

                c_raw = (complexity_a1 * nodes_arr +
                         complexity_a2 * depth_arr +
                         complexity_a3 * op_complexity_arr)
                c_min, c_max = float(np.nanmin(c_raw)), float(np.nanmax(c_raw))
                c_denom = (c_max - c_min) + float(self.fitness_epsilon)
                c_norm = (c_raw - c_min) / c_denom

                # ===== 风格奖励：PRICE + RET 混合结构（例如 price + delta(price)） =====
                style_bonus_list = []
                price_feature_names = {'adj_close', 'adj_open', 'adj_high', 'adj_low', 'vwap'}
                style_gamma = float(getattr(self, "style_gamma", 0.0))
                for prog in population:
                    has_price = False
                    has_ret = False
                    feat_names = getattr(prog, "feature_names", None)
                    try:
                        for node in getattr(prog, "program", []):
                            if isinstance(node, int) and feat_names is not None:
                                if 0 <= node < len(feat_names):
                                    if feat_names[node] in price_feature_names:
                                        has_price = True
                            elif isinstance(node, _Function):
                                # 只要用到了 delta，就认为存在“变化/收益”结构
                                if getattr(node, "name", "") == "delta":
                                    has_ret = True
                            if has_price and has_ret:
                                break
                    except Exception:
                        has_price = has_price
                        has_ret = has_ret
                    style_bonus_list.append(1.0 if (has_price and has_ret) else 0.0)
                style_bonus_arr = np.asarray(style_bonus_list, dtype=float)
                if style_bonus_arr.max() > 0:
                    style_bonus_arr /= style_bonus_arr.max()

                # IS 时间掩码（sample_weight>0 表示训练期）
                n_dates = X.shape[0]
                if sample_weight is None:
                    train_mask = np.ones((n_dates,), dtype=bool)
                else:
                    train_mask = (np.asarray(sample_weight) > 0)

                train_idx = np.where(train_mask)[0]
                # 防御：若训练期太短，避免 split 抛错
                K = int(max(1, self.instability_K))
                splits = []
                if train_idx.size > 0:
                    # 连续时间等分（供不稳定度和CV使用）
                    for seg in np.array_split(train_idx, K):
                        if seg.size > 0:
                            splits.append(seg)
                # 构建时间序列CV切分（仅在需要时使用）
                cv_folds = int(getattr(self, 'cv_folds', 0) or 0)
                self._cv_splits = _build_time_series_cv_splits(train_idx, n_folds=cv_folds) if cv_folds > 1 else []
                # 计算不稳定度原始值：K 个子样本 IR 的标准差
                i_raw_list = []
                for p in population:
                    if len(splits) <= 1:
                        i_raw_list.append(0.0)
                        continue
                    # 预先执行一次，避免重复计算
                    try:
                        y_pred = p.execute_3D(X)
                        if p.transformer:
                            y_pred = p.transformer(y_pred)
                    except Exception:
                        y_pred = None
                    ir_sub = []
                    for seg in splits:
                        sw = np.zeros((n_dates,), dtype=float)
                        sw[seg] = 1.0
                        try:
                            if y_pred is None:
                                raise RuntimeError('no y_pred')
                            # 直接用 metric 计算（与 raw_fitness_3D 口径一致）
                            ir_k = p.metric(y, y_pred, sw)
                        except Exception:
                            # 最稳健的回退：按接口重新执行一次
                            try:
                                ir_k = p.raw_fitness_3D(X, y, sw)
                            except Exception:
                                ir_k = 0.0
                        if np.isfinite(ir_k):
                            ir_sub.append(float(ir_k))
                    if len(ir_sub) <= 1:
                        i_raw_list.append(0.0)
                    else:
                        i_raw_list.append(float(np.std(ir_sub)))

                i_raw = np.asarray(i_raw_list, dtype=float)
                i_min, i_max = float(np.nanmin(i_raw)), float(np.nanmax(i_raw))
                i_denom = (i_max - i_min) + float(self.fitness_epsilon)
                i_norm = (i_raw - i_min) / i_denom

                # 稳健归一化函数（使用5%-95%分位数，抗异常值）
                def robust_normalize(arr, eps=1e-9):
                    """使用5%-95%分位数归一化，避免极端值失真"""
                    finite_vals = arr[np.isfinite(arr)]
                    if len(finite_vals) < 2:
                        return np.zeros_like(arr)
                    p5, p95 = np.nanpercentile(finite_vals, [5, 95])
                    if p95 - p5 < eps:
                        return np.zeros_like(arr)
                    clipped = np.clip(arr, p5, p95)
                    return (clipped - p5) / (p95 - p5 + eps)

                # 是否启用CV版适应度（仅依赖验证段IR）
                use_cv_fitness = bool(getattr(self, 'use_cv_fitness', False)) and len(getattr(self, '_cv_splits', [])) > 0

                if use_cv_fitness:
                    # === 基于时间序列CV的适应度：在各折验证段上计算IR，并聚合为cv_raw ===
                    cv_splits = getattr(self, '_cv_splits', [])
                    # 精细化CV评分权重：IR_q10 与正收益折数占比
                    alpha_q10 = float(getattr(self, 'cv_alpha_q10', 0.0))
                    beta_pos = float(getattr(self, 'cv_beta_pos_frac', 0.0))
                    cv_raw_vals = []
                    for p in population:
                        if not cv_splits:
                            cv_raw_vals.append(0.0)
                            continue
                        try:
                            y_pred_full = p.execute_3D(X)
                            if p.transformer:
                                y_pred_full = p.transformer(y_pred_full)
                        except Exception:
                            cv_raw_vals.append(0.0)
                            continue
                        ir_vals = []
                        for _, val_idx in cv_splits:
                            if val_idx.size <= 5:
                                continue
                            sw = np.zeros((n_dates,), dtype=float)
                            sw[val_idx] = 1.0
                            try:
                                ir_k = self._metric(y, y_pred_full, sw)
                            except Exception:
                                try:
                                    ir_k = p.raw_fitness_3D(X, y, sw)
                                except Exception:
                                    ir_k = 0.0
                            if np.isfinite(ir_k):
                                ir_vals.append(float(ir_k))
                        if len(ir_vals) == 0:
                            cv_raw_vals.append(0.0)
                        else:
                            ir_arr = np.asarray(ir_vals, dtype=float)
                            ir_mean = float(np.mean(ir_arr))
                            # 10%分位数（坏情况尾部）
                            if ir_arr.size >= 2:
                                try:
                                    ir_q10 = float(np.quantile(ir_arr, 0.10))
                                except Exception:
                                    ir_q10 = ir_mean
                            else:
                                ir_q10 = ir_mean
                            # 正IR折数占比
                            pos_frac = float(np.mean(ir_arr > 0.0))
                            cv_raw = ir_mean + alpha_q10 * ir_q10 + beta_pos * pos_frac
                            cv_raw_vals.append(cv_raw)

                    cv_raw_arr = np.asarray(cv_raw_vals, dtype=float)
                    cv_norm = robust_normalize(cv_raw_arr, eps=float(self.fitness_epsilon))

                    # 使用原有的不稳定度归一化 i_norm 作为折间波动惩罚
                    comp_vals = []
                    for idx, p in enumerate(population):
                        # 主体部分：CV 适应度 - 复杂度惩罚 - 不稳定度惩罚
                        comp = (self.w_is * cv_norm[idx]) \
                               - (self.lambda_complexity * c_norm[idx]) \
                               - (self.mu_instability * i_norm[idx])
                        # 轻微风格奖励：若表达式中同时使用了 PRICE 特征和 delta 结构，则加一点 bonus
                        if style_gamma > 0.0:
                            comp += style_gamma * style_bonus_arr[idx]
                        p.fitness_ = float(comp)
                        p.composite_fitness_ = float(comp)
                        p.complexity_nodes_ = float(nodes_arr[idx])
                        p.complexity_depth_ = float(depth_arr[idx])
                        p.complexity_raw_ = float(c_raw[idx])
                        p.complexity_norm_ = float(c_norm[idx])
                        p.instability_raw_ = float(i_raw[idx])
                        p.instability_norm_ = float(i_norm[idx])
                        comp_vals.append(float(comp))

                    fitness = comp_vals
                    length = [p.length_ for p in population]
                else:
                    # === 旧逻辑：基于 IS/OOB IR + IS/OOB ICIR 的复合适应度 ===
                    # 准备 IR 归一化（稳健版本，防止极端值压缩）
                    fit_arr = np.asarray([p.raw_fitness_ if (p.raw_fitness_ is not None and np.isfinite(p.raw_fitness_)) else 0.0
                                          for p in population], dtype=float)
                    fit_norm = robust_normalize(fit_arr, eps=float(self.fitness_epsilon))
                    
                    oob_arr = np.asarray([float(getattr(p, 'oob_fitness_', np.nan)) for p in population], dtype=float)
                    oob_arr[~np.isfinite(oob_arr)] = 0.0
                    if self.oob_ir_min_clip is not None:
                        try:
                            oob_arr = np.maximum(oob_arr, float(self.oob_ir_min_clip))
                        except Exception:
                            pass
                    oob_norm = robust_normalize(oob_arr, eps=float(self.fitness_epsilon))

                    # 训练期 ICIR 和 OOB ICIR 计算（与 IR 完全对齐：多视野、裁剪、方向对齐）
                    icir_is_vals = []
                    icir_oob_vals = []
                    # 获取与 IR 一致的配置参数（从 metric 闭包中提取）
                    try:
                        abs_clip = float(getattr(self._metric.function, '__defaults__', [None])[0] or 0.0)
                        horizons = getattr(self._metric.function, '__defaults__', [None, None, None])[2]
                        quantile = float(getattr(self._metric.function, '__defaults__', [0.2])[0] or 0.2)
                    except Exception:
                        abs_clip = 0.10  # 默认裁剪（匹配A股涨跌停±10%）
                        horizons = [5]   # 默认视野
                        quantile = 0.2
                    
                    # 辅助函数：计算ICIR
                    def calc_icir(factor_df, return_df):
                        ic_series = compute_ic(return_df.values, factor_df.values, 
                                               np.ones(return_df.shape[0], dtype=float), rank_ic=True)
                        ic_mean = ic_series.mean()
                        ic_std = ic_series.std()
                        return float(ic_mean / ic_std) if np.isfinite(ic_std) and ic_std > 1e-9 else 0.0
                    
                    comp_vals = []
                    w_icir_is = float(getattr(self, 'w_icir_is', 0.3))
                    w_icir_oob = float(getattr(self, 'w_icir_oob', 0.5))
                    
                    for idx, p in enumerate(population):
                        # === 训练集 ICIR (IS) ===
                        try:
                            y_pred_full = p.execute_3D(X)
                            if p.transformer:
                                y_pred_full = p.transformer(y_pred_full)
                            # 训练期数据
                            y_pred_is = y_pred_full[train_mask.astype(bool)]
                            y_train_is = y[train_mask.astype(bool)]
                            
                            # 转为 DataFrame 以使用 pandas 滚动和 IC 计算
                            r_df_is = pd.DataFrame(y_train_is)
                            f_df_is = pd.DataFrame(y_pred_is)
                            
                            # 多视野处理（与 IR 一致）
                            if horizons and isinstance(horizons, list) and len(horizons) == 1:
                                r_h_is = r_df_is.rolling(window=int(horizons[0]), min_periods=1).sum()
                            else:
                                r_h_is = r_df_is
                            
                            # 收益裁剪（与 IR 一致）
                            if abs_clip and abs_clip > 0:
                                r_h_is = r_h_is.clip(lower=-float(abs_clip), upper=float(abs_clip))
                            
                            # 方向对齐：计算正向和反向的 ICIR，取最大值
                            icir_pos_is = calc_icir(f_df_is, r_h_is)
                            icir_neg_is = calc_icir(-f_df_is, r_h_is)
                            icir_is_val = max(icir_pos_is, icir_neg_is)
                        except Exception:
                            icir_is_val = 0.0
                        icir_is_vals.append(icir_is_val)
                        
                        # === OOB ICIR (验证集) ===
                        try:
                            # 检查是否有OOB数据
                            oob_mask = ~train_mask.astype(bool)
                            if np.sum(oob_mask) > 10:  # 至少10个样本
                                y_pred_oob = y_pred_full[oob_mask]
                                y_oob = y[oob_mask]
                                
                                r_df_oob = pd.DataFrame(y_oob)
                                f_df_oob = pd.DataFrame(y_pred_oob)
                                
                                # 多视野处理
                                if horizons and isinstance(horizons, list) and len(horizons) == 1:
                                    r_h_oob = r_df_oob.rolling(window=int(horizons[0]), min_periods=1).sum()
                                else:
                                    r_h_oob = r_df_oob
                                
                                # 收益裁剪
                                if abs_clip and abs_clip > 0:
                                    r_h_oob = r_h_oob.clip(lower=-float(abs_clip), upper=float(abs_clip))
                                
                                # 方向对齐
                                icir_pos_oob = calc_icir(f_df_oob, r_h_oob)
                                icir_neg_oob = calc_icir(-f_df_oob, r_h_oob)
                                icir_oob_val = max(icir_pos_oob, icir_neg_oob)
                            else:
                                icir_oob_val = 0.0
                        except Exception:
                            icir_oob_val = 0.0
                        icir_oob_vals.append(icir_oob_val)
                    
                    icir_is_arr = np.asarray(icir_is_vals, dtype=float)
                    icir_is_norm = robust_normalize(icir_is_arr, eps=float(self.fitness_epsilon))
                    
                    icir_oob_arr = np.asarray(icir_oob_vals, dtype=float)
                    icir_oob_norm = robust_normalize(icir_oob_arr, eps=float(self.fitness_epsilon))
                    
                    # 合成复合适应度（使用归一化后的 IS/OOB IR 和 IS/OOB ICIR）
                    for idx, p in enumerate(population):
                        # 确保 raw_fitness_ 有效
                        raw_fit = p.raw_fitness_
                        if raw_fit is None or not np.isfinite(raw_fit):
                            ir_is = 0.0
                        else:
                            ir_is = float(raw_fit)
                        
                        ir_oob = float(getattr(p, 'oob_fitness_', np.nan))
                        if not np.isfinite(ir_oob):
                            ir_oob = 0.0
                        # 可选下限裁剪
                        if self.oob_ir_min_clip is not None:
                            try:
                                ir_oob = max(ir_oob, float(self.oob_ir_min_clip))
                            except Exception:
                                pass
                        comp = (w_icir_is * icir_is_norm[idx]) + (w_icir_oob * icir_oob_norm[idx]) \
                               + (self.w_is * fit_norm[idx]) + (self.w_oob * oob_norm[idx]) \
                               - (self.lambda_complexity * c_norm[idx]) \
                               - (self.mu_instability * i_norm[idx])
                        p.fitness_ = float(comp)
                        p.composite_fitness_ = float(comp)
                        p.complexity_nodes_ = float(nodes_arr[idx])
                        p.complexity_depth_ = float(depth_arr[idx])
                        p.complexity_raw_ = float(c_raw[idx])
                        p.complexity_norm_ = float(c_norm[idx])
                        p.instability_raw_ = float(i_raw[idx])
                        p.instability_norm_ = float(i_norm[idx])
                        comp_vals.append(float(comp))
                    
                    fitness = comp_vals
                    length = [p.length_ for p in population]
            # 束搜索 (Beam Search): 从第4代开始启动，前期充分探索
            if self.beam_width is not None and self.beam_width < len(population) and gen >= 4:
                # ✅ 多样性增强：可配置fitness/complexity比例
                beam_diversity_ratio = float(getattr(self, 'beam_diversity_ratio', 0.30))
                diversity_quota = int(self.beam_width * beam_diversity_ratio)
                fitness_quota = self.beam_width - diversity_quota
                
                # 按适应度排序并选取 top fitness_quota（支持年龄保护）
                def _safe_fitness(val):
                    if val is None:
                        return None
                    try:
                        val = float(val)
                    except Exception:
                        return None
                    return val if np.isfinite(val) else None

                raw_fitness_vals = [_safe_fitness(getattr(p, 'fitness_', None)) for p in population]
                default_bad = -np.inf if self._metric.greater_is_better else np.inf
                fitness_for_sort = [val if val is not None else default_bad for val in raw_fitness_vals]

                if self._metric.greater_is_better:
                    sorted_indices = np.argsort(fitness_for_sort)[::-1]
                else:
                    sorted_indices = np.argsort(fitness_for_sort)
                
                # 年龄分层：区分新人(age<=threshold)与老个体
                age_protection = bool(getattr(self, 'age_protection', False))
                age_threshold = int(getattr(self, 'age_threshold', 2))
                young_beam_ratio = float(getattr(self, 'young_beam_ratio', 0.3))
                program_ages = [int(getattr(p, 'age_', 0)) for p in population]

                if age_protection and fitness_quota > 0:
                    young_idx = {i for i, a in enumerate(program_ages) if a <= age_threshold}
                    old_idx = set(range(len(population))) - young_idx
                    fitness_quota_young = int(round(fitness_quota * young_beam_ratio))
                    fitness_quota_young = min(fitness_quota_young, len(young_idx))
                    fitness_quota_old = fitness_quota - fitness_quota_young
                    fitness_quota_old = min(fitness_quota_old, len(old_idx))
                    # 如果老个体不足，将名额让给新人，反之亦然
                    if fitness_quota_old < 0:
                        fitness_quota_old = 0
                    total_alloc = fitness_quota_old + fitness_quota_young
                    if total_alloc < fitness_quota:
                        # 优先补给新人
                        extra = min(fitness_quota - total_alloc, len(young_idx) - fitness_quota_young)
                        fitness_quota_young += max(0, extra)
                        total_alloc = fitness_quota_old + fitness_quota_young
                    fitness_indices = []
                    # 先选老个体份额
                    if fitness_quota_old > 0 and old_idx:
                        for idx in sorted_indices:
                            if idx in old_idx:
                                fitness_indices.append(idx)
                                if len(fitness_indices) >= fitness_quota_old:
                                    break
                    # 再选新人份额
                    if fitness_quota_young > 0 and young_idx:
                        selected_set = set(fitness_indices)
                        for idx in sorted_indices:
                            if idx in young_idx and idx not in selected_set:
                                fitness_indices.append(idx)
                                if len(fitness_indices) >= fitness_quota_old + fitness_quota_young:
                                    break
                    beam_by_fitness = [population[i] for i in fitness_indices]
                    selected_idx = set(fitness_indices)
                else:
                    # 无年龄保护时，按全局fitness排序
                    beam_by_fitness = [population[i] for i in sorted_indices[:fitness_quota]]
                    selected_idx = set(sorted_indices[:fitness_quota])

                # 提前计算每个个体的基础特征/风格集合
                feature_names = getattr(self, 'feature_names', None)
                if feature_names is None or not isinstance(feature_names, (list, tuple)):
                    feature_names = [f'X{i}' for i in range(X.shape[1])]
                style_map = getattr(self, 'beam_feature_style_map', None)

                def _to_style(idx: int):
                    if idx is None or idx < 0 or idx >= len(feature_names):
                        return None
                    base_name = feature_names[idx]
                    if isinstance(style_map, dict):
                        return style_map.get(base_name, style_map.get(idx, base_name))
                    return base_name

                program_feature_sets = []
                for prog in population:
                    feat_set = set()
                    try:
                        for node in prog.program:
                            if isinstance(node, int):
                                styled = _to_style(node)
                                if styled is not None:
                                    feat_set.add(styled)
                    except Exception:
                        pass
                    program_feature_sets.append(feat_set)

                # 2. 多样性配额：优先选择“风格/特征”覆盖度高、能补足 beam 的个体
                remaining_idx = [i for i in range(len(population)) if i not in selected_idx]
                beam_by_diversity = []
                if diversity_quota > 0 and len(remaining_idx) > 0:
                    feature_counter = Counter()
                    for idx in selected_idx:
                        feature_counter.update(program_feature_sets[idx])

                    diversity_candidates = []
                    for idx in remaining_idx:
                        used_feats = program_feature_sets[idx]
                        if not used_feats:
                            score = 0.0
                        else:
                            score = 0.0
                            for feat in used_feats:
                                score += 1.0 / (1.0 + feature_counter.get(feat, 0))
                            # 额外鼓励多特征组合
                            score += 0.01 * len(used_feats)
                        diversity_candidates.append((score, idx))

                    diversity_candidates.sort(key=lambda x: x[0], reverse=True)
                    for score, idx in diversity_candidates:
                        if len(beam_by_diversity) >= diversity_quota:
                            break
                        beam_by_diversity.append(population[idx])
                        selected_idx.add(idx)
                        feature_counter.update(program_feature_sets[idx])

                    # 兜底：若特征得分不足以填满多样性名额，退回到旧的复杂度排序
                    # 不再退回长度复杂度；若得分不足，保持实际挑选数量
                
                beam_population = beam_by_fitness + beam_by_diversity
                
                # 【止崩三件套】混合生成下一代：少量Elite + Exploitation + Exploration + Random Immigrants
                
                # 辅助函数：创建新程序
                def _create_program(program_tree=None):
                    return _Program(function_set=params['function_set'],
                                   arities=params['arities'],
                                   init_depth=params['init_depth'],
                                   init_method=params['init_method'],
                                   n_features=X.shape[1],
                                   metric=params['_metric'],
                                   transformer=params['_transformer'],
                                   const_range=params['const_range'],
                                   p_point_replace=params['p_point_replace'],
                                   parsimony_coefficient=params['parsimony_coefficient'],
                                   feature_names=params['feature_names'],
                                   random_state=check_random_state(self.random_state),
                                   program=program_tree)
                
                # 1️⃣ 精英率和突变率（可配置）
                elite_ratio = float(getattr(self, 'elite_ratio', 0.05))
                elite_mutation_rate = float(getattr(self, 'elite_mutation_rate', 0.30))
                n_elites = max(10, int(self.population_size * elite_ratio))
                n_elites = min(n_elites, len(beam_population))
                elites = []
                for elite in beam_population[:n_elites]:
                    try:
                        # ✅ 可配置概率点突变避免纯克隆
                        if check_random_state(self.random_state).uniform() < elite_mutation_rate:
                            mutated, _ = elite.point_mutation(check_random_state(self.random_state))
                            elites.append(_create_program(mutated))
                        else:
                            # ✅ 深拷贝：创建新Program对象，避免引用污染
                            elites.append(_create_program(elite.program))
                    except Exception:
                        # ✅ 异常情况也深拷贝
                        elites.append(_create_program(elite.program))
                
                remaining = self.population_size - len(elites)
                
                # 2️⃣ 配额分配：可配置immigrants比例
                immigrants_ratio = float(getattr(self, 'immigrants_ratio', 0.30))
                n_immigrants = int(remaining * immigrants_ratio)
                n_remaining = remaining - n_immigrants
                n_explore = int(n_remaining * 0.50)  # 剩余部分50%探索
                n_exploit = n_remaining - n_explore   # 剩余部分50%利用
                
                # 3️⃣ 非beam父代配额：Exploration强制从非beam抽取≥50%父代
                nonbeam_population = [p for p in population if p not in beam_population]
                if len(nonbeam_population) < 10:  # 如果非beam太少，用全体
                    nonbeam_population = population
                
                def _spawn(n_offspring: int, parent_pool, use_low_pressure=False):
                    """生成后代，可选低选择压力"""
                    if n_offspring <= 0:
                        return []
                    local_seeds = check_random_state(self.random_state).randint(MAX_INT, size=n_offspring)
                    # 如果需要低选择压力，临时修改tournament_size
                    orig_t_size = params['tournament_size']
                    if use_low_pressure:
                        params['tournament_size'] = 1  # 近似随机选择
                    try:
                        offspring = _parallel_evolve_3D(n_offspring, parent_pool, X, y, sample_weight, local_seeds, params)
                    finally:
                        params['tournament_size'] = orig_t_size  # 恢复原值
                    return offspring

                # 从非beam探索（低选择压力，tournament_size=1）
                off_explore = _spawn(n_explore, nonbeam_population, use_low_pressure=True)
                
                # 从beam利用（正常选择压力）
                off_exploit = _spawn(n_exploit, beam_population, use_low_pressure=False)
                
                # 4️⃣ 随机移民：完全随机生成（20%）
                off_immigrants = []
                for _ in range(n_immigrants):
                    try:
                        off_immigrants.append(_create_program(None))
                    except Exception:
                        pass

                new_population = elites + off_exploit + off_explore + off_immigrants
                
                # 5️⃣ 去重机制：保证唯一率≥70%
                def _program_hash(prog):
                    try:
                        return hash(str(prog.program))
                    except Exception:
                        return hash(id(prog))
                
                seen_hashes = set()
                unique_population = []
                for prog in new_population:
                    h = _program_hash(prog)
                    if h not in seen_hashes:
                        seen_hashes.add(h)
                        unique_population.append(prog)
                
                # 如果唯一率<70%，用随机移民填补
                target_unique = max(int(self.population_size * 0.70), len(unique_population))
                while len(unique_population) < target_unique:
                    try:
                        immigrant = _create_program(None)
                        h = _program_hash(immigrant)
                        if h not in seen_hashes:
                            seen_hashes.add(h)
                            unique_population.append(immigrant)
                    except Exception:
                        break
                
                # 数量修正：不足则从beam补充
                if len(unique_population) < self.population_size:
                    for prog in beam_population:
                        if len(unique_population) >= self.population_size:
                            break
                        h = _program_hash(prog)
                        if h not in seen_hashes:
                            seen_hashes.add(h)
                            unique_population.append(prog)
                elif len(unique_population) > self.population_size:
                    unique_population = unique_population[:self.population_size]
                
                population = unique_population
                # Recompute composite fitness for updated population (including new offsprings)
                nodes_arr = np.array([p.length_ for p in population], dtype=float)
                depth_arr = np.array([getattr(p, 'depth_', 0) for p in population], dtype=float)

                # 算子级复杂度：与主循环一致，按 complexity_weight 统计
                op_complexity_list = []
                for prog in population:
                    score = 0.0
                    try:
                        for node in getattr(prog, "program", []):
                            if isinstance(node, _Function):
                                score += getattr(node, "complexity_weight", 1.0)
                    except Exception:
                        pass
                    op_complexity_list.append(score)
                op_complexity_arr = np.asarray(op_complexity_list, dtype=float)

                complexity_a1 = float(getattr(self, "complexity_a1", 1.0))
                complexity_a2 = float(getattr(self, "complexity_a2", 1.0))
                complexity_a3 = float(getattr(self, "complexity_a3", 1.0))

                c_raw = (complexity_a1 * nodes_arr +
                         complexity_a2 * depth_arr +
                         complexity_a3 * op_complexity_arr)
                c_min, c_max = float(np.nanmin(c_raw)), float(np.nanmax(c_raw))
                c_denom = (c_max - c_min) + float(self.fitness_epsilon)
                c_norm = (c_raw - c_min) / c_denom

                # 与主循环一致的风格奖励计算
                style_bonus_list = []
                price_feature_names = {'adj_close', 'adj_open', 'adj_high', 'adj_low', 'vwap'}
                style_gamma = float(getattr(self, "style_gamma", 0.0))
                for prog in population:
                    has_price = False
                    has_ret = False
                    feat_names = getattr(prog, "feature_names", None)
                    try:
                        for node in getattr(prog, "program", []):
                            if isinstance(node, int) and feat_names is not None:
                                if 0 <= node < len(feat_names):
                                    if feat_names[node] in price_feature_names:
                                        has_price = True
                            elif isinstance(node, _Function):
                                if getattr(node, "name", "") == "delta":
                                    has_ret = True
                            if has_price and has_ret:
                                break
                    except Exception:
                        has_price = has_price
                        has_ret = has_ret
                    style_bonus_list.append(1.0 if (has_price and has_ret) else 0.0)
                style_bonus_arr = np.asarray(style_bonus_list, dtype=float)
                if style_bonus_arr.max() > 0:
                    style_bonus_arr /= style_bonus_arr.max()

                n_dates = X.shape[0]
                if sample_weight is None:
                    train_mask = np.ones((n_dates,), dtype=bool)
                else:
                    train_mask = (np.asarray(sample_weight) > 0)

                train_idx = np.where(train_mask)[0]
                K = int(max(1, self.instability_K))
                splits = []
                if train_idx.size > 0:
                    for seg in np.array_split(train_idx, K):
                        if seg.size > 0:
                            splits.append(seg)
                i_raw_list = []
                for p in population:
                    if len(splits) <= 1:
                        i_raw_list.append(0.0)
                        continue
                    try:
                        y_pred = p.execute_3D(X)
                        if p.transformer:
                            y_pred = p.transformer(y_pred)
                    except Exception:
                        y_pred = None
                    ir_sub = []
                    for seg in splits:
                        sw = np.zeros((n_dates,), dtype=float)
                        sw[seg] = 1.0
                        try:
                            ir_val = self._metric(y, y_pred, sw)
                        except Exception:
                            ir_val = np.nan
                        if not np.isfinite(ir_val):
                            ir_val = 0.0
                        ir_sub.append(float(ir_val))
                    i_raw_list.append(float(np.std(ir_sub)) if len(ir_sub) > 0 else 0.0)

                i_raw = np.array(i_raw_list, dtype=float)
                i_norm = robust_normalize(i_raw, eps=float(self.fitness_epsilon))

                # Beam分支：就当前 population 重新计算当代内 IS/OOB IR 归一化（稳健版本）
                fit_arr = np.asarray([p.raw_fitness_ if (p.raw_fitness_ is not None and np.isfinite(p.raw_fitness_)) else 0.0
                                      for p in population], dtype=float)
                fit_norm = robust_normalize(fit_arr, eps=float(self.fitness_epsilon))
                oob_arr = np.asarray([float(getattr(p, 'oob_fitness_', np.nan)) for p in population], dtype=float)
                oob_arr[~np.isfinite(oob_arr)] = 0.0
                if self.oob_ir_min_clip is not None:
                    try:
                        oob_arr = np.maximum(oob_arr, float(self.oob_ir_min_clip))
                    except Exception:
                        pass
                oob_norm = robust_normalize(oob_arr, eps=float(self.fitness_epsilon))
                
                # Beam分支也需要重新计算IS/OOB ICIR（与主分支一致）
                icir_is_vals_beam = []
                icir_oob_vals_beam = []
                try:
                    abs_clip = float(getattr(self._metric.function, '__defaults__', [None])[0] or 0.0)
                    horizons = getattr(self._metric.function, '__defaults__', [None, None, None])[2]
                except Exception:
                    abs_clip = 0.10  # 匹配A股涨跌停±10%
                    horizons = [5]
                
                def calc_icir_beam(factor_df, return_df):
                    ic_series = compute_ic(return_df.values, factor_df.values, 
                                           np.ones(return_df.shape[0], dtype=float), rank_ic=True)
                    ic_mean = ic_series.mean()
                    ic_std = ic_series.std()
                    return float(ic_mean / ic_std) if np.isfinite(ic_std) and ic_std > 1e-9 else 0.0
                
                for p in population:
                    # IS ICIR
                    try:
                        y_pred_full = p.execute_3D(X)
                        if p.transformer:
                            y_pred_full = p.transformer(y_pred_full)
                        y_pred_is = y_pred_full[train_mask.astype(bool)]
                        y_train_is = y[train_mask.astype(bool)]
                        r_df_is = pd.DataFrame(y_train_is)
                        f_df_is = pd.DataFrame(y_pred_is)
                        if horizons and isinstance(horizons, list) and len(horizons) == 1:
                            r_h_is = r_df_is.rolling(window=int(horizons[0]), min_periods=1).sum()
                        else:
                            r_h_is = r_df_is
                        if abs_clip and abs_clip > 0:
                            r_h_is = r_h_is.clip(lower=-float(abs_clip), upper=float(abs_clip))
                        icir_pos_is = calc_icir_beam(f_df_is, r_h_is)
                        icir_neg_is = calc_icir_beam(-f_df_is, r_h_is)
                        icir_is_val = max(icir_pos_is, icir_neg_is)
                    except Exception:
                        icir_is_val = 0.0
                    icir_is_vals_beam.append(icir_is_val)
                    
                    # OOB ICIR
                    try:
                        oob_mask = ~train_mask.astype(bool)
                        if np.sum(oob_mask) > 10:
                            y_pred_oob = y_pred_full[oob_mask]
                            y_oob = y[oob_mask]
                            r_df_oob = pd.DataFrame(y_oob)
                            f_df_oob = pd.DataFrame(y_pred_oob)
                            if horizons and isinstance(horizons, list) and len(horizons) == 1:
                                r_h_oob = r_df_oob.rolling(window=int(horizons[0]), min_periods=1).sum()
                            else:
                                r_h_oob = r_df_oob
                            if abs_clip and abs_clip > 0:
                                r_h_oob = r_h_oob.clip(lower=-float(abs_clip), upper=float(abs_clip))
                            icir_pos_oob = calc_icir_beam(f_df_oob, r_h_oob)
                            icir_neg_oob = calc_icir_beam(-f_df_oob, r_h_oob)
                            icir_oob_val = max(icir_pos_oob, icir_neg_oob)
                        else:
                            icir_oob_val = 0.0
                    except Exception:
                        icir_oob_val = 0.0
                    icir_oob_vals_beam.append(icir_oob_val)
                
                icir_is_arr_beam = np.asarray(icir_is_vals_beam, dtype=float)
                icir_is_norm_beam = robust_normalize(icir_is_arr_beam, eps=float(self.fitness_epsilon))
                icir_oob_arr_beam = np.asarray(icir_oob_vals_beam, dtype=float)
                icir_oob_norm_beam = robust_normalize(icir_oob_arr_beam, eps=float(self.fitness_epsilon))

                # 获取权重
                w_icir_is = float(getattr(self, 'w_icir_is', 0.3))
                w_icir_oob = float(getattr(self, 'w_icir_oob', 0.5))
                
                comp_vals_beam = []
                for idx, p in enumerate(population):
                    # 使用归一化后的 IS/OOB ICIR + IS/OOB IR 进行加权合成
                    comp = (w_icir_is * icir_is_norm_beam[idx]) + (w_icir_oob * icir_oob_norm_beam[idx]) \
                           + (self.w_is * fit_norm[idx]) + (self.w_oob * oob_norm[idx]) \
                           - (self.lambda_complexity * c_norm[idx]) \
                           - (self.mu_instability * i_norm[idx])
                    p.fitness_ = float(comp)
                    p.composite_fitness_ = float(comp)
                    comp_vals_beam.append(float(comp))
                # 使用 beam 后的适应度与长度作为当前代口径
                fitness = comp_vals_beam
                length = [p.length_ for p in population]

            self._programs.append(population)

            # Remove old programs that didn't make it into the new population.
            if not self.low_memory:
                for old_gen in np.arange(gen, 0, -1):
                    indices = []
                    for program in self._programs[old_gen]:
                        if program is not None:
                            for idx in program.parents:
                                if 'idx' in idx:
                                    indices.append(program.parents[idx])
                    indices = set(indices)
                    for idx in range(self.population_size):
                        if idx not in indices:
                            self._programs[old_gen - 1][idx] = None
            elif gen > 0:
                # Remove old generations
                self._programs[gen - 1] = None

            # Record run details
            if self._metric.greater_is_better:
                best_program = population[np.argmax(fitness)]
            else:
                best_program = population[np.argmin(fitness)]

            self.run_details_['generation'].append(gen)
            self.run_details_['average_length'].append(np.mean(length))
            self.run_details_['average_fitness'].append(np.mean(fitness))
            self.run_details_['best_length'].append(best_program.length_)
            # ✅ 显示composite fitness（如果可用），否则显示raw_fitness_
            if hasattr(best_program, 'composite_fitness_'):
                self.run_details_['best_fitness'].append(best_program.composite_fitness_)
            else:
                self.run_details_['best_fitness'].append(best_program.raw_fitness_)
            oob_fitness = np.nan
            # 兼容外部/内部切分：只要最佳个体具备 oob_fitness_ 就展示
            if hasattr(best_program, 'oob_fitness_'):
                oob_fitness = best_program.oob_fitness_
            self.run_details_['best_oob_fitness'].append(oob_fitness)
            generation_time = time() - start_time
            self.run_details_['generation_time'].append(generation_time)

            if self.verbose:
                self._verbose_reporter(self.run_details_)
                import sys
                sys.stdout.flush()  # 强制刷新输出，确保每代结果立即显示

            # Check for early stopping
            if self._metric.greater_is_better:
                best_fitness = fitness[np.argmax(fitness)]
                if best_fitness >= self.stopping_criteria:
                    break
            else:
                best_fitness = fitness[np.argmin(fitness)]
                if best_fitness <= self.stopping_criteria:
                    break

        if isinstance(self, TransformerMixin):
            # Find the best individuals in the final generation
            fitness = np.array(fitness)
            if self._metric.greater_is_better:
                hall_of_fame = fitness.argsort()[::-1][:self.hall_of_fame]
            else:
                hall_of_fame = fitness.argsort()[:self.hall_of_fame]
            evaluation = np.array([gp.execute_3D(X).flatten() for gp in
                                   [self._programs[-1][i] for
                                    i in hall_of_fame]])

            with np.errstate(divide='ignore', invalid='ignore'):
                correlations = evaluation - np.nanmean(evaluation, axis=1).reshape((evaluation.shape[0], 1))
                correlations = np.abs(np.corrcoef(np.nan_to_num(correlations, nan=0.)))
            np.fill_diagonal(correlations, 0.)
            components = list(range(self.hall_of_fame))
            indices = list(range(self.hall_of_fame))
            # Iteratively remove least fit individual of most correlated pair
            while len(components) > self.n_components:
                most_correlated = np.unravel_index(np.argmax(correlations),
                                                   correlations.shape)
                # The correlation matrix is sorted by fitness, so identifying
                # the least fit of the pair is simply getting the higher index
                worst = max(most_correlated)
                components.pop(worst)
                indices.remove(worst)
                correlations = correlations[:, indices][indices, :]
                indices = list(range(len(components)))
            self._best_programs = [self._programs[-1][i] for i in
                                   hall_of_fame[components]]
        else:
            # Find the best individual in the final generation
            if self._metric.greater_is_better:
                self._program = self._programs[-1][np.argmax(fitness)]
            else:
                self._program = self._programs[-1][np.argmin(fitness)]

        return self

    def fit(self, X, y, sample_weight=None):
        """Fit the Genetic Program according to X, y.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.

        y : array-like, shape = [n_samples]
            Target values.

        sample_weight : array-like, shape = [n_samples], optional
            Weights applied to individual samples.

        Returns
        -------
        self : object
            Returns self.

        """
        random_state = check_random_state(self.random_state)

        # Check arrays
        if sample_weight is not None:
            sample_weight = _check_sample_weight(sample_weight, X)

        if isinstance(self, ClassifierMixin):
            X, y = self._validate_data(X, y, y_numeric=False)
            check_classification_targets(y)

            if self.class_weight:
                if sample_weight is None:
                    sample_weight = 1.
                # modify the sample weights with the corresponding class weight
                sample_weight = (sample_weight *
                                 compute_sample_weight(self.class_weight, y))

            self.classes_, y = np.unique(y, return_inverse=True)
            n_trim_classes = np.count_nonzero(np.bincount(y, sample_weight))
            if n_trim_classes != 2:
                raise ValueError("y contains %d class after sample_weight "
                                 "trimmed classes with zero weights, while 2 "
                                 "classes are required."
                                 % n_trim_classes)
            self.n_classes_ = len(self.classes_)

        else:
            X, y = self._validate_data(X, y, y_numeric=True)

        hall_of_fame = self.hall_of_fame
        if hall_of_fame is None:
            hall_of_fame = self.population_size
        if hall_of_fame > self.population_size or hall_of_fame < 1:
            raise ValueError('hall_of_fame (%d) must be less than or equal to '
                             'population_size (%d).' % (self.hall_of_fame,
                                                        self.population_size))
        n_components = self.n_components
        if n_components is None:
            n_components = hall_of_fame
        if n_components > hall_of_fame or n_components < 1:
            raise ValueError('n_components (%d) must be less than or equal to '
                             'hall_of_fame (%d).' % (self.n_components,
                                                     self.hall_of_fame))

        self._function_set = []
        for function in self.function_set:
            if isinstance(function, str):
                if function not in _function_map:
                    raise ValueError('invalid function name %s found in '
                                     '`function_set`.' % function)
                self._function_set.append(_function_map[function])
            elif isinstance(function, _Function):
                self._function_set.append(function)
            else:
                raise ValueError('invalid type %s found in `function_set`.'
                                 % type(function))
        if not self._function_set:
            raise ValueError('No valid functions found in `function_set`.')

        # For point-mutation to find a compatible replacement node
        self._arities = {}
        for function in self._function_set:
            arity = function.arity
            self._arities[arity] = self._arities.get(arity, [])
            self._arities[arity].append(function)

        if isinstance(self.metric, _Fitness):
            self._metric = self.metric
        elif isinstance(self, RegressorMixin):
            if self.metric not in ('mean absolute error', 'mse', 'rmse',
                                   'pearson', 'spearman'):
                raise ValueError('Unsupported metric: %s' % self.metric)
            self._metric = _fitness_map[self.metric]
        elif isinstance(self, ClassifierMixin):
            if self.metric != 'log loss':
                raise ValueError('Unsupported metric: %s' % self.metric)
            self._metric = _fitness_map[self.metric]
        elif isinstance(self, TransformerMixin):
            if self.metric not in ('pearson', 'spearman'):
                raise ValueError('Unsupported metric: %s' % self.metric)
            self._metric = _fitness_map[self.metric]

        self._method_probs = np.array([self.p_crossover,
                                       self.p_subtree_mutation,
                                       self.p_hoist_mutation,
                                       self.p_point_mutation])
        self._method_probs = np.cumsum(self._method_probs)

        if self._method_probs[-1] > 1:
            raise ValueError('The sum of p_crossover, p_subtree_mutation, '
                             'p_hoist_mutation and p_point_mutation should '
                             'total to 1.0 or less.')

        if self.init_method not in ('half and half', 'grow', 'full'):
            raise ValueError('Valid program initializations methods include '
                             '"grow", "full" and "half and half". Given %s.'
                             % self.init_method)

        if not((isinstance(self.const_range, tuple) and
                len(self.const_range) == 2) or self.const_range is None):
            raise ValueError('const_range should be a tuple with length two, '
                             'or None.')

        if (not isinstance(self.init_depth, tuple) or
                len(self.init_depth) != 2):
            raise ValueError('init_depth should be a tuple with length two.')
        if self.init_depth[0] > self.init_depth[1]:
            raise ValueError('init_depth should be in increasing numerical '
                             'order: (min_depth, max_depth).')

        if self.feature_names is not None:
            if self.n_features_in_ != len(self.feature_names):
                raise ValueError('The supplied `feature_names` has different '
                                 'length to n_features. Expected %d, got %d.'
                                 % (self.n_features_in_,
                                    len(self.feature_names)))
            for feature_name in self.feature_names:
                if not isinstance(feature_name, str):
                    raise ValueError('invalid type %s found in '
                                     '`feature_names`.' % type(feature_name))

        if self.transformer is not None:
            if isinstance(self.transformer, _Function):
                self._transformer = self.transformer
            elif self.transformer == 'sigmoid':
                self._transformer = sigmoid
            else:
                raise ValueError('Invalid `transformer`. Expected either '
                                 '"sigmoid" or _Function object, got %s' %
                                 type(self.transformer))
            if self._transformer.arity != 1:
                raise ValueError('Invalid arity for `transformer`. Expected 1, '
                                 'got %d.' % (self._transformer.arity))

        params = self.get_params()
        params['_metric'] = self._metric
        if hasattr(self, '_transformer'):
            params['_transformer'] = self._transformer
        else:
            params['_transformer'] = None
        params['function_set'] = self._function_set
        params['arities'] = self._arities
        params['method_probs'] = self._method_probs

        if not self.warm_start or not hasattr(self, '_programs'):
            # Free allocated memory, if any
            self._programs = []
            self.run_details_ = {'generation': [],
                                 'average_length': [],
                                 'average_fitness': [],
                                 'best_length': [],
                                 'best_fitness': [],
                                 'best_oob_fitness': [],
                                 'generation_time': []}

        prior_generations = len(self._programs)
        n_more_generations = self.generations - prior_generations

        if n_more_generations < 0:
            raise ValueError('generations=%d must be larger or equal to '
                             'len(_programs)=%d when warm_start==True'
                             % (self.generations, len(self._programs)))
        elif n_more_generations == 0:
            fitness = [program.raw_fitness_ for program in self._programs[-1]]
            warn('Warm-start fitting without increasing n_estimators does not '
                 'fit new programs.')

        if self.warm_start:
            # Generate and discard seeds that would have been produced on the
            # initial fit call.
            for i in range(len(self._programs)):
                _ = random_state.randint(MAX_INT, size=self.population_size)

        if self.verbose:
            # Print header fields
            self._verbose_reporter()

        for gen in range(prior_generations, self.generations):

            start_time = time()

            if gen == 0:
                parents = None
            else:
                parents = self._programs[gen - 1]

            # Parallel loop
            n_jobs, n_programs, starts = _partition_estimators(
                self.population_size, self.n_jobs)
            seeds = random_state.randint(MAX_INT, size=self.population_size)

            population = Parallel(n_jobs=n_jobs,
                                  verbose=int(self.verbose > 1))(
                delayed(_parallel_evolve)(n_programs[i],
                                          parents,
                                          X,
                                          y,
                                          sample_weight,
                                          seeds[starts[i]:starts[i + 1]],
                                          params)
                for i in range(n_jobs))

            # Reduce, maintaining order across different n_jobs
            population = list(itertools.chain.from_iterable(population))

            fitness = [program.raw_fitness_ for program in population]
            length = [program.length_ for program in population]

            parsimony_coefficient = None
            if self.parsimony_coefficient == 'auto':
                parsimony_coefficient = (np.cov(length, fitness)[1, 0] /
                                         np.var(length))
            for program in population:
                # Blend OOB fitness into selection if available
                program.fitness_ = program.fitness(parsimony_coefficient, oob_weight=self.oob_weight)

            self._programs.append(population)

            # Remove old programs that didn't make it into the new population.
            if not self.low_memory:
                for old_gen in np.arange(gen, 0, -1):
                    indices = []
                    for program in self._programs[old_gen]:
                        if program is not None:
                            for idx in program.parents:
                                if 'idx' in idx:
                                    indices.append(program.parents[idx])
                    indices = set(indices)
                    for idx in range(self.population_size):
                        if idx not in indices:
                            self._programs[old_gen - 1][idx] = None
            elif gen > 0:
                # Remove old generations
                self._programs[gen - 1] = None

            # Record run details
            if self._metric.greater_is_better:
                best_program = population[np.argmax(fitness)]
            else:
                best_program = population[np.argmin(fitness)]

            self.run_details_['generation'].append(gen)
            self.run_details_['average_length'].append(np.mean(length))
            self.run_details_['average_fitness'].append(np.mean(fitness))
            self.run_details_['best_length'].append(best_program.length_)
            # ✅ 显示composite fitness（如果可用），否则显示raw_fitness_
            if hasattr(best_program, 'composite_fitness_'):
                self.run_details_['best_fitness'].append(best_program.composite_fitness_)
            else:
                self.run_details_['best_fitness'].append(best_program.raw_fitness_)
            oob_fitness = np.nan
            # 兼容外部/内部切分：只要最佳个体具备 oob_fitness_ 就展示
            if hasattr(best_program, 'oob_fitness_'):
                oob_fitness = best_program.oob_fitness_
            self.run_details_['best_oob_fitness'].append(oob_fitness)
            generation_time = time() - start_time
            self.run_details_['generation_time'].append(generation_time)

            if self.verbose:
                self._verbose_reporter(self.run_details_)

            # Check for early stopping
            if self._metric.greater_is_better:
                best_fitness = fitness[np.argmax(fitness)]
                if best_fitness >= self.stopping_criteria:
                    break
            else:
                best_fitness = fitness[np.argmin(fitness)]
                if best_fitness <= self.stopping_criteria:
                    break

        if isinstance(self, TransformerMixin):
            # Find the best individuals in the final generation
            fitness = np.array(fitness)
            if self._metric.greater_is_better:
                hall_of_fame = fitness.argsort()[::-1][:self.hall_of_fame]
            else:
                hall_of_fame = fitness.argsort()[:self.hall_of_fame]
            evaluation = np.array([gp.execute(X) for gp in
                                   [self._programs[-1][i] for
                                    i in hall_of_fame]])
            if self.metric == 'spearman':
                evaluation = np.apply_along_axis(rankdata, 1, evaluation)

            with np.errstate(divide='ignore', invalid='ignore'):
                correlations = np.abs(np.corrcoef(evaluation))
            np.fill_diagonal(correlations, 0.)
            components = list(range(self.hall_of_fame))
            indices = list(range(self.hall_of_fame))
            # Iteratively remove least fit individual of most correlated pair
            while len(components) > self.n_components:
                most_correlated = np.unravel_index(np.argmax(correlations),
                                                   correlations.shape)
                # The correlation matrix is sorted by fitness, so identifying
                # the least fit of the pair is simply getting the higher index
                worst = max(most_correlated)
                components.pop(worst)
                indices.remove(worst)
                correlations = correlations[:, indices][indices, :]
                indices = list(range(len(components)))
            self._best_programs = [self._programs[-1][i] for i in
                                   hall_of_fame[components]]

        else:
            # Find the best individual in the final generation
            if self._metric.greater_is_better:
                self._program = self._programs[-1][np.argmax(fitness)]
            else:
                self._program = self._programs[-1][np.argmin(fitness)]

        return self
    

class SymbolicRegressor(BaseSymbolic, RegressorMixin):
    """A Genetic Programming symbolic regressor.

    A symbolic regressor is an estimator that begins by building a population
    of naive random formulas to represent a relationship. The formulas are
    represented as tree-like structures with mathematical functions being
    recursively applied to variables and constants. Each successive generation
    of programs is then evolved from the one that came before it by selecting
    the fittest individuals from the population to undergo genetic operations
    such as crossover, mutation or reproduction.

    Parameters
    ----------
    population_size : integer, optional (default=1000)
        The number of programs in each generation.

    generations : integer, optional (default=20)
        The number of generations to evolve.

    tournament_size : integer, optional (default=20)
        The number of programs that will compete to become part of the next
        generation.

    stopping_criteria : float, optional (default=0.0)
        The required metric value required in order to stop evolution early.

    const_range : tuple of two floats, or None, optional (default=(-1., 1.))
        The range of constants to include in the formulas. If None then no
        constants will be included in the candidate programs.

    init_depth : tuple of two ints, optional (default=(2, 6))
        The range of tree depths for the initial population of naive formulas.
        Individual trees will randomly choose a maximum depth from this range.
        When combined with `init_method='half and half'` this yields the well-
        known 'ramped half and half' initialization method.

    init_method : str, optional (default='half and half')
        - 'grow' : Nodes are chosen at random from both functions and
          terminals, allowing for smaller trees than `init_depth` allows. Tends
          to grow asymmetrical trees.
        - 'full' : Functions are chosen until the `init_depth` is reached, and
          then terminals are selected. Tends to grow 'bushy' trees.
        - 'half and half' : Trees are grown through a 50/50 mix of 'full' and
          'grow', making for a mix of tree shapes in the initial population.

    function_set : iterable, optional (default=('add', 'sub', 'mul', 'div'))
        The functions to use when building and evolving programs. This iterable
        can include strings to indicate either individual functions as outlined
        below, or you can also include your own functions as built using the
        ``make_function`` factory from the ``functions`` module.

        Available individual functions are:

        - 'add' : addition, arity=2.
        - 'sub' : subtraction, arity=2.
        - 'mul' : multiplication, arity=2.
        - 'div' : protected division where a denominator near-zero returns 1.,
          arity=2.
        - 'sqrt' : protected square root where the absolute value of the
          argument is used, arity=1.
        - 'log' : protected log where the absolute value of the argument is
          used and a near-zero argument returns 0., arity=1.
        - 'abs' : absolute value, arity=1.
        - 'neg' : negative, arity=1.
        - 'inv' : protected inverse where a near-zero argument returns 0.,
          arity=1.
        - 'max' : maximum, arity=2.
        - 'min' : minimum, arity=2.
        - 'sin' : sine (radians), arity=1.
        - 'cos' : cosine (radians), arity=1.
        - 'tan' : tangent (radians), arity=1.

    metric : str, optional (default='mean absolute error')
        The name of the raw fitness metric. Available options include:

        - 'mean absolute error'.
        - 'mse' for mean squared error.
        - 'rmse' for root mean squared error.
        - 'pearson', for Pearson's product-moment correlation coefficient.
        - 'spearman' for Spearman's rank-order correlation coefficient.

        Note that 'pearson' and 'spearman' will not directly predict the target
        but could be useful as value-added features in a second-step estimator.
        This would allow the user to generate one engineered feature at a time,
        using the SymbolicTransformer would allow creation of multiple features
        at once.

    parsimony_coefficient : float or "auto", optional (default=0.001)
        This constant penalizes large programs by adjusting their fitness to
        be less favorable for selection. Larger values penalize the program
        more which can control the phenomenon known as 'bloat'. Bloat is when
        evolution is increasing the size of programs without a significant
        increase in fitness, which is costly for computation time and makes for
        a less understandable final result. This parameter may need to be tuned
        over successive runs.

        If "auto" the parsimony coefficient is recalculated for each generation
        using c = Cov(l,f)/Var( l), where Cov(l,f) is the covariance between
        program size l and program fitness f in the population, and Var(l) is
        the variance of program sizes.

    p_crossover : float, optional (default=0.9)
        The probability of performing crossover on a tournament winner.
        Crossover takes the winner of a tournament and selects a random subtree
        from it to be replaced. A second tournament is performed to find a
        donor. The donor also has a subtree selected at random and this is
        inserted into the original parent to form an offspring in the next
        generation.

    p_subtree_mutation : float, optional (default=0.01)
        The probability of performing subtree mutation on a tournament winner.
        Subtree mutation takes the winner of a tournament and selects a random
        subtree from it to be replaced. A donor subtree is generated at random
        and this is inserted into the original parent to form an offspring in
        the next generation.

    p_hoist_mutation : float, optional (default=0.01)
        The probability of performing hoist mutation on a tournament winner.
        Hoist mutation takes the winner of a tournament and selects a random
        subtree from it. A random subtree of that subtree is then selected
        and this is 'hoisted' into the original subtrees location to form an
        offspring in the next generation. This method helps to control bloat.

    p_point_mutation : float, optional (default=0.01)
        The probability of performing point mutation on a tournament winner.
        Point mutation takes the winner of a tournament and selects random
        nodes from it to be replaced. Terminals are replaced by other terminals
        and functions are replaced by other functions that require the same
        number of arguments as the original node. The resulting tree forms an
        offspring in the next generation.

        Note : The above genetic operation probabilities must sum to less than
        one. The balance of probability is assigned to 'reproduction', where a
        tournament winner is cloned and enters the next generation unmodified.

    p_point_replace : float, optional (default=0.05)
        For point mutation only, the probability that any given node will be
        mutated.

    max_samples : float, optional (default=1.0)
        The fraction of samples to draw from X to evaluate each program on.

    feature_names : list, optional (default=None)
        Optional list of feature names, used purely for representations in
        the `print` operation or `export_graphviz`. If None, then X0, X1, etc
        will be used for representations.

    warm_start : bool, optional (default=False)
        When set to ``True``, reuse the solution of the previous call to fit
        and add more generations to the evolution, otherwise, just fit a new
        evolution.

    low_memory : bool, optional (default=False)
        When set to ``True``, only the current generation is retained. Parent
        information is discarded. For very large populations or runs with many
        generations, this can result in substantial memory use reduction.

    n_jobs : integer, optional (default=1)
        The number of jobs to run in parallel for `fit`. If -1, then the number
        of jobs is set to the number of cores.

    verbose : int, optional (default=0)
        Controls the verbosity of the evolution building process.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    Attributes
    ----------
    run_details_ : dict
        Details of the evolution process. Includes the following elements:

        - 'generation' : The generation index.
        - 'average_length' : The average program length of the generation.
        - 'average_fitness' : The average program fitness of the generation.
        - 'best_length' : The length of the best program in the generation.
        - 'best_fitness' : The fitness of the best program in the generation.
        - 'best_oob_fitness' : The out of bag fitness of the best program in
          the generation (requires `max_samples` < 1.0).
        - 'generation_time' : The time it took for the generation to evolve.

    See Also
    --------
    SymbolicTransformer

    References
    ----------
    .. [1] J. Koza, "Genetic Programming", 1992.

    .. [2] R. Poli, et al. "A Field Guide to Genetic Programming", 2008.

    """

    def __init__(self,
                 *,
                 population_size=1000,
                 generations=20,
                 tournament_size=20,
                 stopping_criteria=0.0,
                 const_range=(-1., 1.),
                 init_depth=(2, 6),
                 init_method='half and half',
                 function_set=('add', 'sub', 'mul', 'div'),
                 metric='mean absolute error',
                 parsimony_coefficient=0.001,
                 p_crossover=0.9,
                 p_subtree_mutation=0.01,
                 p_hoist_mutation=0.01,
                 p_point_mutation=0.01,
                 p_point_replace=0.05,
                 max_samples=1.0,
                 feature_names=None,
                 warm_start=False,
                 low_memory=False,
                 n_jobs=1,
                 verbose=0,
                 random_state=None,
                 op_penalty_map=None,
                 feature_min_unique=0,
                 feature_penalty=0.0,
                 depth_threshold=None,
                 depth_penalty=0.0,
                 beam_width=None):
        super(SymbolicRegressor, self).__init__(
            population_size=population_size,
            generations=generations,
            tournament_size=tournament_size,
            stopping_criteria=stopping_criteria,
            const_range=const_range,
            init_depth=init_depth,
            init_method=init_method,
            function_set=function_set,
            metric=metric,
            parsimony_coefficient=parsimony_coefficient,
            p_crossover=p_crossover,
            p_subtree_mutation=p_subtree_mutation,
            p_hoist_mutation=p_hoist_mutation,
            p_point_mutation=p_point_mutation,
            p_point_replace=p_point_replace,
            max_samples=max_samples,
            feature_names=feature_names,
            warm_start=warm_start,
            low_memory=low_memory,
            n_jobs=n_jobs,
            verbose=verbose,
            random_state=random_state,
            op_penalty_map=op_penalty_map,
            feature_min_unique=feature_min_unique,
            feature_penalty=feature_penalty,
            depth_threshold=depth_threshold,
            depth_penalty=depth_penalty,
            beam_width=beam_width)

    def __str__(self):
        """Overloads `print` output of the object to resemble a LISP tree."""
        if not hasattr(self, '_program'):
            return self.__repr__()
        return self._program.__str__()

    def predict(self, X):
        """Perform regression on test vectors X.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Input vectors, where n_samples is the number of samples
            and n_features is the number of features.

        Returns
        -------
        y : array, shape = [n_samples]
            Predicted values for X.

        """
        if not hasattr(self, '_program'):
            raise NotFittedError('SymbolicRegressor not fitted.')

        X = check_array(X)
        _, n_features = X.shape
        if self.n_features_in_ != n_features:
            raise ValueError('Number of features of the model must match the '
                             'input. Model n_features is %s and input '
                             'n_features is %s.'
                             % (self.n_features_in_, n_features))

        y = self._program.execute(X)

        return y


class SymbolicClassifier(BaseSymbolic, ClassifierMixin):
    """A Genetic Programming symbolic classifier.

    A symbolic classifier is an estimator that begins by building a population
    of naive random formulas to represent a relationship. The formulas are
    represented as tree-like structures with mathematical functions being
    recursively applied to variables and constants. Each successive generation
    of programs is then evolved from the one that came before it by selecting
    the fittest individuals from the population to undergo genetic operations
    such as crossover, mutation or reproduction.

    Parameters
    ----------
    population_size : integer, optional (default=500)
        The number of programs in each generation.

    generations : integer, optional (default=10)
        The number of generations to evolve.

    tournament_size : integer, optional (default=20)
        The number of programs that will compete to become part of the next
        generation.

    stopping_criteria : float, optional (default=0.0)
        The required metric value required in order to stop evolution early.

    const_range : tuple of two floats, or None, optional (default=(-1., 1.))
        The range of constants to include in the formulas. If None then no
        constants will be included in the candidate programs.

    init_depth : tuple of two ints, optional (default=(2, 6))
        The range of tree depths for the initial population of naive formulas.
        Individual trees will randomly choose a maximum depth from this range.
        When combined with `init_method='half and half'` this yields the well-
        known 'ramped half and half' initialization method.

    init_method : str, optional (default='half and half')
        - 'grow' : Nodes are chosen at random from both functions and
          terminals, allowing for smaller trees than `init_depth` allows. Tends
          to grow asymmetrical trees.
        - 'full' : Functions are chosen until the `init_depth` is reached, and
          then terminals are selected. Tends to grow 'bushy' trees.
        - 'half and half' : Trees are grown through a 50/50 mix of 'full' and
          'grow', making for a mix of tree shapes in the initial population.

    function_set : iterable, optional (default=('add', 'sub', 'mul', 'div'))
        The functions to use when building and evolving programs. This iterable
        can include strings to indicate either individual functions as outlined
        below, or you can also include your own functions as built using the
        ``make_function`` factory from the ``functions`` module.

        Available individual functions are:

        - 'add' : addition, arity=2.
        - 'sub' : subtraction, arity=2.
        - 'mul' : multiplication, arity=2.
        - 'div' : protected division where a denominator near-zero returns 1.,
          arity=2.
        - 'sqrt' : protected square root where the absolute value of the
          argument is used, arity=1.
        - 'log' : protected log where the absolute value of the argument is
          used and a near-zero argument returns 0., arity=1.
        - 'abs' : absolute value, arity=1.
        - 'neg' : negative, arity=1.
        - 'inv' : protected inverse where a near-zero argument returns 0.,
          arity=1.
        - 'max' : maximum, arity=2.
        - 'min' : minimum, arity=2.
        - 'sin' : sine (radians), arity=1.
        - 'cos' : cosine (radians), arity=1.
        - 'tan' : tangent (radians), arity=1.

    transformer : str, optional (default='sigmoid')
        The name of the function through which the raw decision function is
        passed. This function will transform the raw decision function into
        probabilities of each class.

        This can also be replaced by your own functions as built using the
        ``make_function`` factory from the ``functions`` module.

    metric : str, optional (default='log loss')
        The name of the raw fitness metric. Available options include:

        - 'log loss' aka binary cross-entropy loss.

    parsimony_coefficient : float or "auto", optional (default=0.001)
        This constant penalizes large programs by adjusting their fitness to
        be less favorable for selection. Larger values penalize the program
        more which can control the phenomenon known as 'bloat'. Bloat is when
        evolution is increasing the size of programs without a significant
        increase in fitness, which is costly for computation time and makes for
        a less understandable final result. This parameter may need to be tuned
        over successive runs.

        If "auto" the parsimony coefficient is recalculated for each generation
        using c = Cov(l,f)/Var( l), where Cov(l,f) is the covariance between
        program size l and program fitness f in the population, and Var(l) is
        the variance of program sizes.

    p_crossover : float, optional (default=0.9)
        The probability of performing crossover on a tournament winner.
        Crossover takes the winner of a tournament and selects a random subtree
        from it to be replaced. A second tournament is performed to find a
        donor. The donor also has a subtree selected at random and this is
        inserted into the original parent to form an offspring in the next
        generation.

    p_subtree_mutation : float, optional (default=0.01)
        The probability of performing subtree mutation on a tournament winner.
        Subtree mutation takes the winner of a tournament and selects a random
        subtree from it to be replaced. A donor subtree is generated at random
        and this is inserted into the original parent to form an offspring in
        the next generation.

    p_hoist_mutation : float, optional (default=0.01)
        The probability of performing hoist mutation on a tournament winner.
        Hoist mutation takes the winner of a tournament and selects a random
        subtree from it. A random subtree of that subtree is then selected
        and this is 'hoisted' into the original subtrees location to form an
        offspring in the next generation. This method helps to control bloat.

    p_point_mutation : float, optional (default=0.01)
        The probability of performing point mutation on a tournament winner.
        Point mutation takes the winner of a tournament and selects random
        nodes from it to be replaced. Terminals are replaced by other terminals
        and functions are replaced by other functions that require the same
        number of arguments as the original node. The resulting tree forms an
        offspring in the next generation.

        Note : The above genetic operation probabilities must sum to less than
        one. The balance of probability is assigned to 'reproduction', where a
        tournament winner is cloned and enters the next generation unmodified.

    p_point_replace : float, optional (default=0.05)
        For point mutation only, the probability that any given node will be
        mutated.

    max_samples : float, optional (default=1.0)
        The fraction of samples to draw from X to evaluate each program on.

    class_weight : dict, 'balanced' or None, optional (default=None)
        Weights associated with classes in the form ``{class_label: weight}``.
        If not given, all classes are supposed to have weight one.

        The "balanced" mode uses the values of y to automatically adjust
        weights inversely proportional to class frequencies in the input data
        as ``n_samples / (n_classes * np.bincount(y))``

    feature_names : list, optional (default=None)
        Optional list of feature names, used purely for representations in
        the `print` operation or `export_graphviz`. If None, then X0, X1, etc
        will be used for representations.

    warm_start : bool, optional (default=False)
        When set to ``True``, reuse the solution of the previous call to fit
        and add more generations to the evolution, otherwise, just fit a new
        evolution.

    low_memory : bool, optional (default=False)
        When set to ``True``, only the current generation is retained. Parent
        information is discarded. For very large populations or runs with many
        generations, this can result in substantial memory use reduction.

    n_jobs : integer, optional (default=1)
        The number of jobs to run in parallel for `fit`. If -1, then the number
        of jobs is set to the number of cores.

    verbose : int, optional (default=0)
        Controls the verbosity of the evolution building process.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    Attributes
    ----------
    run_details_ : dict
        Details of the evolution process. Includes the following elements:

        - 'generation' : The generation index.
        - 'average_length' : The average program length of the generation.
        - 'average_fitness' : The average program fitness of the generation.
        - 'best_length' : The length of the best program in the generation.
        - 'best_fitness' : The fitness of the best program in the generation.
        - 'best_oob_fitness' : The out of bag fitness of the best program in
          the generation (requires `max_samples` < 1.0).
        - 'generation_time' : The time it took for the generation to evolve.

    See Also
    --------
    SymbolicTransformer

    References
    ----------
    .. [1] J. Koza, "Genetic Programming", 1992.

    .. [2] R. Poli, et al. "A Field Guide to Genetic Programming", 2008.

    """

    def __init__(self,
                 *,
                 population_size=1000,
                 generations=20,
                 tournament_size=20,
                 stopping_criteria=0.0,
                 const_range=(-1., 1.),
                 init_depth=(2, 6),
                 init_method='half and half',
                 function_set=('add', 'sub', 'mul', 'div'),
                 transformer='sigmoid',
                 metric='log loss',
                 parsimony_coefficient=0.001,
                 p_crossover=0.9,
                 p_subtree_mutation=0.01,
                 p_hoist_mutation=0.01,
                 p_point_mutation=0.01,
                 p_point_replace=0.05,
                 max_samples=1.0,
                 class_weight=None,
                 feature_names=None,
                 warm_start=False,
                 low_memory=False,
                 n_jobs=1,
                 verbose=0,
                 random_state=None,
                 op_penalty_map=None,
                 feature_min_unique=0,
                 feature_penalty=0.0,
                 depth_threshold=None,
                 depth_penalty=0.0,
                 beam_width=None):
        super(SymbolicClassifier, self).__init__(
            population_size=population_size,
            generations=generations,
            tournament_size=tournament_size,
            stopping_criteria=stopping_criteria,
            const_range=const_range,
            init_depth=init_depth,
            init_method=init_method,
            function_set=function_set,
            transformer=transformer,
            metric=metric,
            parsimony_coefficient=parsimony_coefficient,
            p_crossover=p_crossover,
            p_subtree_mutation=p_subtree_mutation,
            p_hoist_mutation=p_hoist_mutation,
            p_point_mutation=p_point_mutation,
            p_point_replace=p_point_replace,
            max_samples=max_samples,
            class_weight=class_weight,
            feature_names=feature_names,
            warm_start=warm_start,
            low_memory=low_memory,
            n_jobs=n_jobs,
            verbose=verbose,
            random_state=random_state,
            op_penalty_map=op_penalty_map,
            feature_min_unique=feature_min_unique,
            feature_penalty=feature_penalty,
            depth_threshold=depth_threshold,
            depth_penalty=depth_penalty,
            beam_width=beam_width)

    def __str__(self):
        """Overloads `print` output of the object to resemble a LISP tree."""
        if not hasattr(self, '_program'):
            return self.__repr__()
        return self._program.__str__()

    def _more_tags(self):
        return {'binary_only': True}

    def predict_proba(self, X):
        """Predict probabilities on test vectors X.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Input vectors, where n_samples is the number of samples
            and n_features is the number of features.

        Returns
        -------
        proba : array, shape = [n_samples, n_classes]
            The class probabilities of the input samples. The order of the
            classes corresponds to that in the attribute `classes_`.

        """
        if not hasattr(self, '_program'):
            raise NotFittedError('SymbolicClassifier not fitted.')

        X = check_array(X)
        _, n_features = X.shape
        if self.n_features_in_ != n_features:
            raise ValueError('Number of features of the model must match the '
                             'input. Model n_features is %s and input '
                             'n_features is %s.'
                             % (self.n_features_in_, n_features))

        scores = self._program.execute(X)
        proba = self._transformer(scores)
        proba = np.vstack([1 - proba, proba]).T
        return proba

    def predict(self, X):
        """Predict classes on test vectors X.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Input vectors, where n_samples is the number of samples
            and n_features is the number of features.

        Returns
        -------
        y : array, shape = [n_samples,]
            The predicted classes of the input samples.

        """
        proba = self.predict_proba(X)
        return self.classes_.take(np.argmax(proba, axis=1), axis=0)


class SymbolicTransformer(BaseSymbolic, TransformerMixin):
    """A Genetic Programming symbolic transformer.

    A symbolic transformer is a supervised transformer that begins by building
    a population of naive random formulas to represent a relationship. The
    formulas are represented as tree-like structures with mathematical
    functions being recursively applied to variables and constants. Each
    successive generation of programs is then evolved from the one that came
    before it by selecting the fittest individuals from the population to
    undergo genetic operations such as crossover, mutation or reproduction.
    The final population is searched for the fittest individuals with the least
    correlation to one another.

    Parameters
    ----------
    population_size : integer, optional (default=1000)
        The number of programs in each generation.

    hall_of_fame : integer, or None, optional (default=100)
        The number of fittest programs to compare from when finding the
        least-correlated individuals for the n_components. If `None`, the
        entire final generation will be used.

    n_components : integer, or None, optional (default=10)
        The number of best programs to return after searching the hall_of_fame
        for the least-correlated individuals. If `None`, the entire
        hall_of_fame will be used.

    generations : integer, optional (default=20)
        The number of generations to evolve.

    tournament_size : integer, optional (default=20)
        The number of programs that will compete to become part of the next
        generation.

    stopping_criteria : float, optional (default=1.0)
        The required metric value required in order to stop evolution early.

    const_range : tuple of two floats, or None, optional (default=(-1., 1.))
        The range of constants to include in the formulas. If None then no
        constants will be included in the candidate programs.

    init_depth : tuple of two ints, optional (default=(2, 6))
        The range of tree depths for the initial population of naive formulas.
        Individual trees will randomly choose a maximum depth from this range.
        When combined with `init_method='half and half'` this yields the well-
        known 'ramped half and half' initialization method.

    init_method : str, optional (default='half and half')
        - 'grow' : Nodes are chosen at random from both functions and
          terminals, allowing for smaller trees than `init_depth` allows. Tends
          to grow asymmetrical trees.
        - 'full' : Functions are chosen until the `init_depth` is reached, and
          then terminals are selected. Tends to grow 'bushy' trees.
        - 'half and half' : Trees are grown through a 50/50 mix of 'full' and
          'grow', making for a mix of tree shapes in the initial population.

    function_set : iterable, optional (default=('add', 'sub', 'mul', 'div'))
        The functions to use when building and evolving programs. This iterable
        can include strings to indicate either individual functions as outlined
        below, or you can also include your own functions as built using the
        ``make_function`` factory from the ``functions`` module.

        Available individual functions are:

        - 'add' : addition, arity=2.
        - 'sub' : subtraction, arity=2.
        - 'mul' : multiplication, arity=2.
        - 'div' : protected division where a denominator near-zero returns 1.,
          arity=2.
        - 'sqrt' : protected square root where the absolute value of the
          argument is used, arity=1.
        - 'log' : protected log where the absolute value of the argument is
          used and a near-zero argument returns 0., arity=1.
        - 'abs' : absolute value, arity=1.
        - 'neg' : negative, arity=1.
        - 'inv' : protected inverse where a near-zero argument returns 0.,
          arity=1.
        - 'max' : maximum, arity=2.
        - 'min' : minimum, arity=2.
        - 'sin' : sine (radians), arity=1.
        - 'cos' : cosine (radians), arity=1.
        - 'tan' : tangent (radians), arity=1.

    metric : str, optional (default='pearson')
        The name of the raw fitness metric. Available options include:

        - 'pearson', for Pearson's product-moment correlation coefficient.
        - 'spearman' for Spearman's rank-order correlation coefficient.

    parsimony_coefficient : float or "auto", optional (default=0.001)
        This constant penalizes large programs by adjusting their fitness to
        be less favorable for selection. Larger values penalize the program
        more which can control the phenomenon known as 'bloat'. Bloat is when
        evolution is increasing the size of programs without a significant
        increase in fitness, which is costly for computation time and makes for
        a less understandable final result. This parameter may need to be tuned
        over successive runs.

        If "auto" the parsimony coefficient is recalculated for each generation
        using c = Cov(l,f)/Var( l), where Cov(l,f) is the covariance between
        program size l and program fitness f in the population, and Var(l) is
        the variance of program sizes.

    p_crossover : float, optional (default=0.9)
        The probability of performing crossover on a tournament winner.
        Crossover takes the winner of a tournament and selects a random subtree
        from it to be replaced. A second tournament is performed to find a
        donor. The donor also has a subtree selected at random and this is
        inserted into the original parent to form an offspring in the next
        generation.

    p_subtree_mutation : float, optional (default=0.01)
        The probability of performing subtree mutation on a tournament winner.
        Subtree mutation takes the winner of a tournament and selects a random
        subtree from it to be replaced. A donor subtree is generated at random
        and this is inserted into the original parent to form an offspring in
        the next generation.

    p_hoist_mutation : float, optional (default=0.01)
        The probability of performing hoist mutation on a tournament winner.
        Hoist mutation takes the winner of a tournament and selects a random
        subtree from it. A random subtree of that subtree is then selected
        and this is 'hoisted' into the original subtrees location to form an
        offspring in the next generation. This method helps to control bloat.

    p_point_mutation : float, optional (default=0.01)
        The probability of performing point mutation on a tournament winner.
        Point mutation takes the winner of a tournament and selects random
        nodes from it to be replaced. Terminals are replaced by other terminals
        and functions are replaced by other functions that require the same
        number of arguments as the original node. The resulting tree forms an
        offspring in the next generation.

        Note : The above genetic operation probabilities must sum to less than
        one. The balance of probability is assigned to 'reproduction', where a
        tournament winner is cloned and enters the next generation unmodified.

    p_point_replace : float, optional (default=0.05)
        For point mutation only, the probability that any given node will be
        mutated.

    max_samples : float, optional (default=1.0)
        The fraction of samples to draw from X to evaluate each program on.

    feature_names : list, optional (default=None)
        Optional list of feature names, used purely for representations in
        the `print` operation or `export_graphviz`. If None, then X0, X1, etc
        will be used for representations.

    warm_start : bool, optional (default=False)
        When set to ``True``, reuse the solution of the previous call to fit
        and add more generations to the evolution, otherwise, just fit a new
        evolution.

    low_memory : bool, optional (default=False)
        When set to ``True``, only the current generation is retained. Parent
        information is discarded. For very large populations or runs with many
        generations, this can result in substantial memory use reduction.

    n_jobs : integer, optional (default=1)
        The number of jobs to run in parallel for `fit`. If -1, then the number
        of jobs is set to the number of cores.

    verbose : int, optional (default=0)
        Controls the verbosity of the evolution building process.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    Attributes
    ----------
    run_details_ : dict
        Details of the evolution process. Includes the following elements:

        - 'generation' : The generation index.
        - 'average_length' : The average program length of the generation.
        - 'average_fitness' : The average program fitness of the generation.
        - 'best_length' : The length of the best program in the generation.
        - 'best_fitness' : The fitness of the best program in the generation.
        - 'best_oob_fitness' : The out of bag fitness of the best program in
          the generation (requires `max_samples` < 1.0).
        - 'generation_time' : The time it took for the generation to evolve.

    See Also
    --------
    SymbolicRegressor

    References
    ----------
    .. [1] J. Koza, "Genetic Programming", 1992.

    .. [2] R. Poli, et al. "A Field Guide to Genetic Programming", 2008.

    """

    def __init__(self,
                 *,
                 population_size=1000,
                 hall_of_fame=100,
                 n_components=10,
                 generations=20,
                 tournament_size=20,
                 stopping_criteria=1.0,
                 const_range=(-1., 1.),
                 init_depth=(2, 6),
                 init_method='half and half',
                 function_set=('add', 'sub', 'mul', 'div'),
                 metric='pearson',
                 parsimony_coefficient=0.001,
                 p_crossover=0.9,
                 p_subtree_mutation=0.01,
                 p_hoist_mutation=0.01,
                 p_point_mutation=0.01,
                 p_point_replace=0.05,
                 max_samples=1.0,
                 oob_weight=0.0,
                 feature_names=None,
                 warm_start=False,
                 low_memory=False,
                 n_jobs=1,
                 verbose=0,
                 random_state=None,
                 beam_width=None,
                 # Composite fitness passthrough
                 use_composite_in_engine=False,
                 w_is=1.0,
                 w_oob=0.0,
                 lambda_complexity=0.0,
                 mu_instability=0.0,
                 complexity_a1=1.0,
                 complexity_a2=1.0,
                 instability_K=4,
                 fitness_epsilon=1e-9,
                 oob_ir_min_clip=-1.0,
                 # Beam mixing
                 beam_ratio=None,
                 # Age-layered protection
                 age_protection=False,
                 age_threshold=2,
                 young_beam_ratio=0.3):
        super(SymbolicTransformer, self).__init__(
            population_size=population_size,
            hall_of_fame=hall_of_fame,
            n_components=n_components,
            generations=generations,
            tournament_size=tournament_size,
            stopping_criteria=stopping_criteria,
            const_range=const_range,
            init_depth=init_depth,
            init_method=init_method,
            function_set=function_set,
            metric=metric,
            parsimony_coefficient=parsimony_coefficient,
            p_crossover=p_crossover,
            p_subtree_mutation=p_subtree_mutation,
            p_hoist_mutation=p_hoist_mutation,
            p_point_mutation=p_point_mutation,
            p_point_replace=p_point_replace,
            max_samples=max_samples,
            oob_weight=oob_weight,
            feature_names=feature_names,
            warm_start=warm_start,
            low_memory=low_memory,
            n_jobs=n_jobs,
            verbose=verbose,
            random_state=random_state,
            beam_width=beam_width,
            # Composite fitness passthrough to BaseSymbolic
            use_composite_in_engine=use_composite_in_engine,
            w_is=w_is,
            w_oob=w_oob,
            lambda_complexity=lambda_complexity,
            mu_instability=mu_instability,
            complexity_a1=complexity_a1,
            complexity_a2=complexity_a2,
            instability_K=instability_K,
            fitness_epsilon=fitness_epsilon,
            oob_ir_min_clip=oob_ir_min_clip,
            # Beam mixing
            beam_ratio=beam_ratio,
            # Age-layered protection
            age_protection=age_protection,
            age_threshold=age_threshold,
            young_beam_ratio=young_beam_ratio)

    def __len__(self):
        """Overloads `len` output to be the number of fitted components."""
        if not hasattr(self, '_best_programs'):
            return 0
        return self.n_components

    def __getitem__(self, item):
        """Return the ith item of the fitted components."""
        if item >= len(self):
            raise IndexError
        return self._best_programs[item]

    def __str__(self):
        """Overloads `print` output of the object to resemble LISP trees."""
        if not hasattr(self, '_best_programs'):
            return self.__repr__()
        output = str([gp.__str__() for gp in self])
        return output.replace("',", ",\n").replace("'", "")

    def show_program(self, X, Y, sample_weight=None, baseline=False):
        '''Show program results in a dictionary format.'''
        if sample_weight is None:
            sample_weight = np.array([1] * len(X))

        oob_sample_weight = np.where(sample_weight > 0, 0, 1)

        result = []
        for program in self._total_program if baseline else self._best_programs: 
            Y_pred = program.execute_3D(X)

            rank_ic = "_fitness_map['rank_ic'](Y, Y_pred, sample_weight)"
            rank_icir = "_fitness_map['rank_icir'](Y, Y_pred, sample_weight)"
            quantile_max = "_fitness_map['quantile_max'](Y, Y_pred, sample_weight)"
            quantile_mono = "_fitness_map['quantile_mono'](Y, Y_pred, sample_weight)"

            OOB_rank_ic = rank_ic.replace('sample_weight', 'oob_sample_weight')
            OOB_rank_icir = rank_icir.replace('sample_weight', 'oob_sample_weight')
            OOB_quantile_max = quantile_max.replace('sample_weight', 'oob_sample_weight')
            OOB_quantile_mono = quantile_mono.replace('sample_weight', 'oob_sample_weight')

            rank_ic = eval(rank_ic)
            rank_icir = eval(rank_icir)
            quantile_max = eval(quantile_max)
            quantile_mono = eval(quantile_mono)

            #  OOB Calculation
            OOB_rank_ic = eval(OOB_rank_ic)
            OOB_rank_icir = eval(OOB_rank_icir)
            OOB_quantile_max = eval(OOB_quantile_max)
            OOB_quantile_mono = eval(OOB_quantile_mono)

            OOB_fitness = np.nan
            if self.max_samples < 1.0:
                OOB_fitness = program.oob_fitness_


            result.append({"Expression": program.__str__(),
                           "IS fitness": program.raw_fitness_,
                           "OOS fitness": OOB_fitness,
                           "IS RankIC": rank_ic,
                           "OOS RankIC": OOB_rank_ic,
                           "IS RankICIR": rank_icir,
                           "OOS RankICIR": OOB_rank_icir,
                        #    "IS TopQuantileRet": quantile_max,
                        #    "OOS TopQuantileRet": OOB_quantile_max,
                        #    "IS QuantileMono": quantile_mono,
                        #    "OOS QuantileMono": OOB_quantile_mono,
                           })
        return result


    def show_program_simple(self, baseline=False):
        result = []
        
        if baseline:
            for program in self._total_program:
                result.append({"Expression": program.__str__(),
                               "Fitness": program.raw_fitness_,
                               "OOB Fitness": program.oob_fitness_,
                              })
        else:            
            for program in self._best_programs:
                result.append({"Expression": program.__str__(),
                               "Fitness": program.raw_fitness_,
                               "OOB Fitness": program.oob_fitness_,
                              })
        return result

    def _more_tags(self):
        return {
            "_xfail_checks": {
                "check_sample_weights_invariance": (
                    "zero sample_weight is not equivalent to removing samples"
                ),
            }
        }

    def transform(self, X):
        """Transform X according to the fitted transformer.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Input vectors, where n_samples is the number of samples
            and n_features is the number of features.

        Returns
        -------
        X_new : array-like, shape = [n_samples, n_components]
            Transformed array.

        """
        if not hasattr(self, '_best_programs'):
            raise NotFittedError('SymbolicTransformer not fitted.')

        X = check_array(X)
        _, n_features = X.shape
        if self.n_features_in_ != n_features:
            raise ValueError('Number of features of the model must match the '
                             'input. Model n_features is %s and input '
                             'n_features is %s.'
                             % (self.n_features_in_, n_features))

        X_new = np.array([gp.execute(X) for gp in self._best_programs]).T

        return X_new

    def fit_transform(self, X, y, sample_weight=None):
        """Fit to data, then transform it.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.

        y : array-like, shape = [n_samples]
            Target values.

        sample_weight : array-like, shape = [n_samples], optional
            Weights applied to individual samples.

        Returns
        -------
        X_new : array-like, shape = [n_samples, n_components]
            Transformed array.

        """
        return self.fit(X, y, sample_weight).transform(X)
