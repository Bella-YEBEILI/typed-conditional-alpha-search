import numpy as np
import pandas as pd
import operator
from joblib import Parallel, delayed


def datatype_adapter(func):
    def wrapper(*args):
        if len(args) == 1 and isinstance(args[0], np.ndarray):
            new_args = (pd.DataFrame(args[0]),)
            result = func(*new_args)
            return result
        if len(args) == 1 and isinstance(args[0], (float, int)):
            new_args = (pd.DataFrame([args[0]]),)
            result = func(*new_args)
            return float(result.iloc[0])
        if (len(args) == 2 and isinstance(args[0], np.ndarray) and not isinstance(args[1], np.ndarray)):
            new_args = (pd.DataFrame(args[0]), args[1])
            result = func(*new_args)
        elif (len(args) == 2 and isinstance(args[1], np.ndarray) and not isinstance(args[0], np.ndarray)):
            new_args = (args[0], pd.DataFrame(args[1]))
            result = func(*new_args)
        else:
            result = func(*args)
        return result

    return wrapper

@datatype_adapter
def DELTA(df:pd.DataFrame, p:int=1):
    return df.groupby('instrument').transform(lambda x: x.diff(periods=p))

@datatype_adapter
def RANK(df:pd.DataFrame):
    """Cross-sectional rank."""
    return df.groupby('datetime').rank(pct=True)

@datatype_adapter
def MEAN(df:pd.DataFrame):
    """Cross-sectional mean."""
    return df.groupby('datetime').mean()

@datatype_adapter
def STD(df:pd.DataFrame):
    """Cross-sectional std."""
    return df.groupby('datetime').std()

@datatype_adapter
def SKEW(df:pd.DataFrame):
    """Cross-sectional skewness."""
    from scipy.stats import skew as scipy_skew
    return df.groupby('datetime').transform(lambda x: scipy_skew(x.dropna(), nan_policy='omit') if len(x.dropna()) >= 3 else np.nan)

@datatype_adapter
def KURT(df:pd.DataFrame):
    """Cross-sectional kurtosis."""
    from scipy.stats import kurtosis
    def calc_kurt(group):
        k = kurtosis(group.dropna(), fisher=True, nan_policy='omit')
        return pd.Series(k, index=group.index)
    return df.groupby('datetime').transform(lambda x: kurtosis(x.dropna(), fisher=True, nan_policy='omit') if len(x.dropna()) >= 4 else np.nan)

@datatype_adapter
def MAX(df:pd.DataFrame):
    """Cross-sectional max."""
    return df.groupby('datetime').max()

@datatype_adapter
def MIN(df:pd.DataFrame):
    """Cross-sectional min."""
    return df.groupby('datetime').min()

@datatype_adapter
def MEDIAN(df:pd.DataFrame):
    """Cross-sectional median."""
    return df.groupby('datetime').median()


@datatype_adapter
def TS_KURT(df:pd.DataFrame, p:int=5):
    """Rolling kurtosis."""
    from scipy.stats import kurtosis
    def rolling_kurt(x):
        return x.rolling(p, min_periods=min(4, p)).apply(
            lambda arr: kurtosis(arr, fisher=True, nan_policy='omit') if len(arr.dropna()) >= 4 else np.nan,
            raw=False
        )
    return df.groupby('instrument').transform(rolling_kurt)

@datatype_adapter
def TS_SKEW(df:pd.DataFrame, p:int=5):
    """Rolling skewness."""
    from scipy.stats import skew as scipy_skew
    def rolling_skew(x):
        return x.rolling(p, min_periods=min(3, p)).apply(
            lambda arr: scipy_skew(arr, nan_policy='omit') if len(arr.dropna()) >= 3 else np.nan,
            raw=False
        )
    return df.groupby('instrument').transform(rolling_skew)

@datatype_adapter
def TS_RANK(df:pd.DataFrame, p:int=5):
    """Time-series percentile rank."""
    return df.groupby('instrument').transform(lambda x: x.rolling(p, min_periods=1).rank(pct=True))

@datatype_adapter
def TS_MAX(df:pd.DataFrame, p:int=5):
    """Time-series max."""
    return df.groupby('instrument').transform(lambda x: x.rolling(p, min_periods=1).max())

@datatype_adapter
def TS_MIN(df:pd.DataFrame, p:int=5):
    """Time-series min."""
    return df.groupby('instrument').transform(lambda x: x.rolling(p, min_periods=1).min())

@datatype_adapter
def TS_MEAN(df:pd.DataFrame, p:int=5):
    """Time-series mean."""
    return df.groupby('instrument').transform(lambda x: x.rolling(p, min_periods=1).mean())

@datatype_adapter
def TS_MEDIAN(df:pd.DataFrame, p:int=5):
    """Time-series median."""
    return df.groupby('instrument').transform(lambda x: x.rolling(p, min_periods=1).median())

@datatype_adapter
def PERCENTILE(df: pd.DataFrame, q: float, p: int = None):
    """
    Quantile of given data. q in [0,1]; if p given, rolling quantile.
    """
    assert 0 <= q <= 1, "Quantile q must be in [0, 1]"
    
    if p is not None:
        return df.groupby('instrument').transform(lambda x: x.rolling(p, min_periods=1).quantile(q))
    else:
        return df.groupby('instrument').transform(lambda x: x.quantile(q))



@datatype_adapter
def TS_SUM(df:pd.DataFrame, p:int=5):
    """Time-series rolling sum."""
    return df.groupby('instrument').transform(lambda x: x.rolling(p, min_periods=1).sum())


@datatype_adapter
def TS_ARGMAX(df: pd.DataFrame, p: int = 5):
    """Days since max in past p days."""
    def rolling_argmax(window):
        return len(window) - window.argmax() - 1
    return df.groupby('instrument').transform(lambda x: x.rolling(p, min_periods=1).apply(rolling_argmax, raw=True))

@datatype_adapter
def TS_ARGMIN(df: pd.DataFrame, p: int = 5):
    """Days since min in past p days."""
    def rolling_argmin(window):
        return len(window) - window.argmin() - 1
    return df.groupby('instrument').transform(lambda x: x.rolling(p, min_periods=1).apply(rolling_argmin, raw=True))



def MAX(x:pd.DataFrame, y:pd.DataFrame, z:pd.DataFrame=None):
    """Element-wise max of DataFrames."""
    if z is None:
        return np.maximum(x, y)
    else:
        return np.maximum(np.maximum(x, y), z)




def MIN(x:pd.DataFrame, y:pd.DataFrame, z:pd.DataFrame=None):
    """Element-wise min of DataFrames.""" 
    if z is None:
        return np.minimum(x, y)
    else:
        return np.minimum(np.minimum(x, y), z)
    


@datatype_adapter
def ABS(df:pd.DataFrame):
    """Element-wise absolute value."""   
    return df.groupby('instrument').transform(lambda x: x.abs())    

@datatype_adapter
def DELAY(df:pd.DataFrame, p:int=1):
    """Delay data by p periods."""
    assert p >= 0, ValueError("DELAY period must be >= 0 (look-ahead bias)")
    return df.groupby('instrument').transform(lambda x: x.shift(p))


def TS_CORR(df1:pd.Series, df2: np.ndarray | pd.Series, p:int=5):
    """Rolling correlation of two series."""
    if isinstance(df2, np.ndarray):
        if p != len(df2):
            p = len(df2)
        def corr(window):
            x = window
            y = df2[:len(window)]
            mean_x = np.mean(x)
            mean_y = np.mean(y)
            
            cov = np.sum((x - mean_x) * (y - mean_y))
            std_x = np.sqrt(np.sum((x - mean_x) ** 2))
            std_y = np.sqrt(np.sum((y - mean_y) ** 2))
            
            if std_x == 0 or std_y == 0:
                return 0.0
            return cov / (std_x * std_y)
        
        return df1.groupby('instrument').transform(lambda x: x.rolling(p, min_periods=2).apply(corr, raw=True))
    elif isinstance(df2, (pd.Series, pd.DataFrame)):
        def rolling_corr(group, df2, p):
            instrument = group.name
            if isinstance(df2, pd.DataFrame) and 'instrument' in df2.index.names:
                df2_group = df2.xs(instrument, level='instrument')
            elif isinstance(df2, pd.Series) and 'instrument' in df2.index.names:
                df2_group = df2.xs(instrument, level='instrument')
            else:
                df2_group = df2
            return group.rolling(p, min_periods=2).corr(df2_group)

        result = df1.groupby('instrument').apply(lambda x: rolling_corr(x, df2, p))
        result = result.reset_index(level=0, drop=True).sort_index()
        return result
    else:
        raise TypeError(f"TS_CORR does not support df2 type: {type(df2)}")


def TS_COVARIANCE(df1:pd.DataFrame, df2:pd.DataFrame, p:int=5):  
    """Rolling covariance of two series."""
    if isinstance(df2, np.ndarray):
        if p != len(df2):
            p = len(df2)
        def cov(window):
            return np.cov(window, df2[:len(window)])[0, 1] if len(window) > 1 else 0.0
        return df1.groupby('instrument').transform(lambda x: x.rolling(p, min_periods=2).apply(cov, raw=True))
    elif isinstance(df2, (pd.Series, pd.DataFrame)):
        def rolling_cov(group, df2, p):
            instrument = group.name
            if isinstance(df2, pd.DataFrame) and 'instrument' in df2.index.names:
                df2_group = df2.xs(instrument, level='instrument')
            elif isinstance(df2, pd.Series) and 'instrument' in df2.index.names:
                df2_group = df2.xs(instrument, level='instrument')
            else:
                df2_group = df2
            return group.rolling(p, min_periods=2).cov(df2_group)

        result = df1.groupby('instrument').apply(lambda x: rolling_cov(x, df2, p))
        result = result.reset_index(level=0, drop=True).sort_index()
        return result
    else:
        raise TypeError(f"TS_COVARIANCE does not support df2 type: {type(df2)}")

@datatype_adapter
def TS_STD(df:pd.DataFrame, p:int=20):
    """Rolling standard deviation."""
    return df.groupby('instrument').transform(lambda x: x.rolling(p, min_periods=1).std())





@datatype_adapter
def TS_VAR(df: pd.DataFrame, p: int = 5, ddof: int = 1):
    """Rolling variance."""
    return df.groupby('instrument').transform(
        lambda x: x.rolling(p, min_periods=1).var(ddof=ddof)
    )

@datatype_adapter
def SIGN(df: pd.DataFrame):
    """Element-wise sign."""
    return np.sign(df)

@datatype_adapter
def SMA(df:pd.DataFrame, m:float=None, n:float=None):
    """Simple moving average. Y_{i+1} = m/n*X_i + (1 - m/n)*Y_i if n given."""
        
    if isinstance(m, int) and m >= 1 and n is None:
        return df.groupby('instrument').transform(lambda x: x.rolling(m, min_periods=1).mean())
    else:
        return df.groupby('instrument').transform(lambda x: x.ewm(alpha=n/m).mean())

@datatype_adapter
def EMA(df:pd.DataFrame, p):
    """Exponential moving average with period p."""
    return df.groupby('instrument').transform(lambda x: x.ewm(span=int(p), min_periods=1).mean())
    
@datatype_adapter
def WMA(df:pd.DataFrame, p:int=20):
    """
    Weighted moving average over p periods (recent has higher weight).
    """
    weights = [0.9**i for i in range(p)][::-1]
    def calculate_wma(window):
        return (window * weights[:len(window)]).sum() / sum(weights[:len(window)])

    return df.groupby('instrument').transform(lambda x: x.rolling(window=p, min_periods=1).apply(calculate_wma, raw=True))

@datatype_adapter
def COUNT(cond:pd.DataFrame, p:int=20):
    """
    Conditional count over rolling window p.
    """
    return cond.groupby('instrument').transform(lambda x: x.rolling(p, min_periods=1).sum())

@datatype_adapter
def SUMIF(df:pd.DataFrame, p:int, cond:pd.DataFrame):
    """
    Rolling sum of df where cond is true over window p.
    """
    return (df * cond).groupby('instrument').transform(lambda x: x.rolling(p, min_periods=1).sum())

@datatype_adapter
def FILTER(df:pd.DataFrame, cond:pd.DataFrame):
    """
    Filter series by condition; where cond is false, set to 0.
    """
    return df.mul(cond)
    

@datatype_adapter
def PROD(df:pd.DataFrame, p:int=5):
    """
    Rolling product over window p.
    """

    if isinstance(p, int):
        return df.groupby('instrument').transform(lambda x: x.rolling(p, min_periods=1).apply(lambda x: x.prod(), raw=True))
    else:
        return df.mul(p)    

@datatype_adapter
def DECAYLINEAR(df:pd.DataFrame, p:int=5):
    """
    Linearly decay weighted average over p periods.
    """
    assert isinstance(p, int), ValueError(f"DECAYLINEAR expects positive int, got {type(p).__name__}")
    decay_weights = np.arange(1, p+1, 1)
    decay_weights = decay_weights / decay_weights.sum()
    
    def calculate_deycaylinear(window):
        return (window * decay_weights[:len(window)]).sum()
    
    return df.groupby('instrument').transform(lambda x: x.rolling(p, min_periods=1).apply(calculate_deycaylinear, raw=True))

@datatype_adapter
def HIGHDAY(df:pd.DataFrame, p:int=5):
    """
    Days since max in window p.
    """
    assert isinstance(p, int), ValueError(f"HIGHDAY expects positive int, got {type(p).__name__}")
    def highday(window):
        return len(window) - window.argmax(axis=0)
    return df.groupby('instrument').transform(lambda x: x.rolling(p, min_periods=1).apply(highday, raw=True))

@datatype_adapter
def LOWDAY(df:pd.DataFrame, p:int=5):
    """
    Days since min in window p.
    """
    assert isinstance(p, int), ValueError(f"LOWDAY expects positive int, got {type(p).__name__}")
    def lowday(window):
        return len(window) - window.argmin(axis=0)
    return df.groupby('instrument').transform(lambda x: x.rolling(p, min_periods=1).apply(lowday, raw=True))
    

def SEQUENCE(n):
    """
    Sequence 1 to n.
    """
    assert isinstance(n, int), ValueError(f"SEQUENCE(n) expects positive int, got {type(n).__name__}")
    return np.linspace(1, n, n, dtype=np.float32)

@datatype_adapter
def SUMAC(df:pd.DataFrame, p:int=10):
    """
    Rolling cumulative sum over window p.
    """
    assert isinstance(p, int), ValueError(f"SUMAC expects positive int, got {type(p).__name__}")
    return df.groupby('instrument').transform(lambda x: x.rolling(p, min_periods=1).sum())



def calculate_beta(y, x):
    """Regression coefficient (beta)."""
    X = np.vstack([x, np.ones(len(x))]).T
    beta, _ = np.linalg.lstsq(X, y, rcond=None)[0]
    return beta

def rolling_beta(df1_group, df2_group, p):
    """Rolling beta of df1 on df2."""
    result = np.empty(len(df1_group))
    result[:] = np.nan

    for i in range(p - 1, len(df1_group)):
        window_y = df1_group.iloc[i - p + 1 : i + 1].values
        window_x = df2_group.iloc[:p].values if df1_group.shape != df2_group.shape else df2_group.iloc[i - p + 1 : i + 1].values
        result[i] = calculate_beta(window_y, window_x)

    return pd.Series(result, index=df1_group.index)


def REGBETA(df1: pd.DataFrame, df2: pd.DataFrame, p: int = 5, n_jobs: int = -1):
    """
    Rolling regression coefficient (beta) of df1 on df2.
    """
    assert not (isinstance(df2, np.ndarray) and isinstance(df1, np.ndarray)), "df1 and df2 cannot both be np.ndarray; at least one must be DataFrame (e.g. $close)."
    if isinstance(df2, np.ndarray) or isinstance(df1, np.ndarray):
        if isinstance(df1, np.ndarray):
            df3 = df1
            df1 = df2
            df2 = df3
            p = min(len(df2), p)
            df2 = pd.Series(df2)
        df1 = df1.fillna(0)
        
        df1_groups = list(df1.groupby('instrument'))
        df2 = pd.Series(df2[:p])
        
        results = Parallel(n_jobs=n_jobs)(
            delayed(rolling_beta)(df1_group, df2, p)
            for _, df1_group in df1_groups
        )
        
        result = pd.concat(results)
        result = result.sort_index()
        return result
    
    else:
        assert df1.index.equals(df2.index), "df1 and df2 indices must align"
        
        df1 = df1.fillna(0)
        df2 = df2.fillna(0)
        
        df1_groups = list(df1.groupby('instrument'))
        df2_groups = list(df2.groupby('instrument'))
        
        if len(df1_groups) != len(df2_groups):
            raise ValueError("df1 and df2 group counts must match.")
        
        results = Parallel(n_jobs=n_jobs)(
            delayed(rolling_beta)(df1_group, df2_group, p)
            for (_, df1_group), (_, df2_group) in zip(df1_groups, df2_groups)
        )
        
        result = pd.concat(results)
        result = result.sort_index()
        return result



def calculate_residuals(y, x):
    """Residual (actual - predicted)."""
    X = np.vstack([x, np.ones(len(x))]).T
    beta, intercept = np.linalg.lstsq(X, y, rcond=None)[0]
    y_pred = beta * x + intercept
    residuals = y - y_pred
    return residuals[-1]

def rolling_residuals(df1_group, df2_group, p):
    """Rolling residual of df1 on df2."""
    result = np.empty(len(df1_group))
    result[:] = np.nan

    for i in range(p - 1, len(df1_group)):
        window_y = df1_group.iloc[i - p + 1 : i + 1].values
        window_x = df2_group.iloc[:p].values if df1_group.shape != df2_group.shape else df2_group.iloc[i - p + 1 : i + 1].values
        result[i] = calculate_residuals(window_y, window_x)

    return pd.Series(result, index=df1_group.index)


def REGRESI(df1: pd.DataFrame, df2: pd.DataFrame, p: int = 5, n_jobs: int = -1):
    """
    Rolling residual of df1 on df2.
    """
    
    assert not (isinstance(df2, np.ndarray) and isinstance(df1, np.ndarray)), "df1 and df2 cannot both be np.ndarray; at least one must be DataFrame (e.g. $close)."
    if isinstance(df2, np.ndarray) or isinstance(df1, np.ndarray):
        if isinstance(df1, np.ndarray):
            df3 = df1
            df1 = df2
            df2 = df3
            p = min(len(df2), p)
        df1 = df1.fillna(0)
        df2 = pd.Series(df2[:p])
        
        df1_groups = list(df1.groupby('instrument'))
        
        results = Parallel(n_jobs=n_jobs)(
            delayed(rolling_residuals)(df1_group, df2, p)
            for _, df1_group in df1_groups
        )
        
        result = pd.concat(results)
        result = result.sort_index()
        return result
    
    else:
        if isinstance(df1.index, pd.MultiIndex) and not isinstance(df2.index, pd.MultiIndex):
            datetime_level = df1.index.get_level_values('datetime')
            df2_aligned = df2.reindex(datetime_level, method='ffill')
            df2_aligned.index = df1.index
            df2 = df2_aligned
        elif not df1.index.equals(df2.index):
            if isinstance(df1.index, pd.MultiIndex) and isinstance(df2.index, pd.MultiIndex):
                try:
                    df2 = df2.reindex(df1.index)
                except Exception:
                    assert df1.index.equals(df2.index), "df1 and df2 indices must align"
            else:
                assert df1.index.equals(df2.index), "df1 and df2 indices must align"
        
        df1 = df1.fillna(0)
        df2 = df2.fillna(0)
        
        df1_groups = list(df1.groupby('instrument'))
        
        if isinstance(df2.index, pd.MultiIndex) and 'instrument' in df2.index.names:
            df2_groups = list(df2.groupby('instrument'))
            if len(df1_groups) != len(df2_groups):
                raise ValueError("df1 and df2 group counts must match.")
            results = Parallel(n_jobs=n_jobs)(
                delayed(rolling_residuals)(df1_group, df2_group, p)
                for (_, df1_group), (_, df2_group) in zip(df1_groups, df2_groups)
            )
        else:
            results = Parallel(n_jobs=n_jobs)(
                delayed(rolling_residuals)(df1_group, df2, p)
                for _, df1_group in df1_groups
            )
        
        result = pd.concat(results)
        result = result.sort_index()
        return result

        
# Math
@datatype_adapter
def EXP(df:pd.DataFrame):
    """
    Element-wise exp.
    """
    return df.apply(np.exp)

@datatype_adapter
def SQRT(df: pd.DataFrame):
    """Element-wise sqrt."""
    if isinstance(df, int):
        return np.sqrt(df)
    return df.apply(np.sqrt)

@datatype_adapter
def LOG(df:pd.DataFrame):
    """Natural logarithm."""
    if isinstance(df, int):
        return np.log(df)
    return (df+1).apply(np.log)

@datatype_adapter
def INV(df: pd.DataFrame):
    """Reciprocal (1/x)."""
    return 1 / df

@datatype_adapter
def POW(df:pd.DataFrame, n:int):
    """Element-wise power."""
    return np.power(df, n)

def FLOOR(df:pd.DataFrame):
    """Floor (round down)."""
    return df.apply(np.floor)

@datatype_adapter
def TS_ZSCORE(df: pd.DataFrame, p:int=5):
    assert isinstance(p, int), ValueError(f"TS_ZSCORE expects positive int, got {type(p).__name__}")
    # assert isinstance(df, pd.DataFrame), ValueError(f"TS_ZSCORE expects pd.DataFrame, got {type(df).__name__}")
    return (df - df.groupby('instrument').transform(lambda x: x.rolling(p, min_periods=1).mean())) / df.groupby('instrument').transform(lambda x: x.rolling(p, min_periods=1).std())

@datatype_adapter
def ZSCORE(df):
    mean = df.groupby('datetime').mean()
    std = df.groupby('datetime').std()
    zscore = (df - mean) / std
    return zscore

@datatype_adapter
def SCALE(df: pd.DataFrame, target_sum: float = 1.0):
    """Scale series so absolute sum equals target_sum."""
    abs_sum = ABS(df).groupby('datetime').sum()
    return df.multiply(target_sum).div(abs_sum, axis=0)


@datatype_adapter
def TS_MAD(df: pd.DataFrame, p: int = 5):
    """Rolling median absolute deviation (MAD = median(|X_i - median(X)|))."""
    def rolling_mad(window):
        median_val = np.median(window)
        abs_dev = np.abs(window - median_val)
        return np.median(abs_dev)
    
    return df.groupby('instrument').transform(
        lambda x: x.rolling(p, min_periods=1).apply(rolling_mad, raw=True)
    )


@datatype_adapter
def TS_QUANTILE(df: pd.DataFrame, p: int = 5, q: float = 0.5):
    """Rolling quantile. Auto-detects parameter order if swapped (q, p -> p, q)."""
    if isinstance(p, float) and 0 < p < 1 and isinstance(q, (int, float)) and q > 1:
        p, q = int(q), p
    p = int(p)
    q = float(q)
    assert 0 <= q <= 1, f"Quantile q must be in [0, 1], got {q}"
    assert p >= 1, f"Window p must >= 1, got {p}"
    return df.groupby('instrument').transform(lambda x: x.rolling(p, min_periods=1).quantile(q))

@datatype_adapter
def TS_PCTCHANGE(df: pd.DataFrame, p: int = 1):
    """Percentage change over p periods (default 1)."""
    return df.groupby('instrument').transform(lambda x: x.pct_change(periods=p, fill_method=None).fillna(0))


def ADD(df1, df2):
    """Add with index alignment."""
    return _arithmetic_with_alignment(df1, df2, np.add)

def SUBTRACT(df1, df2):
    """Subtract with index alignment."""
    return _arithmetic_with_alignment(df1, df2, np.subtract)

def MULTIPLY(df1, df2):
    """Multiply with index alignment."""
    return _arithmetic_with_alignment(df1, df2, np.multiply)

def DIVIDE(df1, df2):
    """Divide with index alignment."""
    return _arithmetic_with_alignment(df1, df2, np.divide)

def _arithmetic_with_alignment(df1, df2, op_func):
    """Arithmetic op with index alignment."""
    if not isinstance(df1, (pd.DataFrame, pd.Series)) and not isinstance(df2, (pd.DataFrame, pd.Series)):
        return op_func(df1, df2)
    
    if not isinstance(df1, (pd.DataFrame, pd.Series)):
        return op_func(df1, df2)
    if not isinstance(df2, (pd.DataFrame, pd.Series)):
        return op_func(df1, df2)
    
    if isinstance(df1.index, pd.MultiIndex) and not isinstance(df2.index, pd.MultiIndex):
        datetime_level = df1.index.get_level_values('datetime')
        if isinstance(df2, pd.DataFrame):
            df2_aligned = df2.reindex(datetime_level, method='ffill')
        else:
            df2_aligned = df2.reindex(datetime_level, method='ffill')
        df2_aligned.index = df1.index
        df2 = df2_aligned
    elif not isinstance(df1.index, pd.MultiIndex) and isinstance(df2.index, pd.MultiIndex):
        datetime_level = df2.index.get_level_values('datetime')
        if isinstance(df1, pd.DataFrame):
            df1_aligned = df1.reindex(datetime_level, method='ffill')
        else:
            df1_aligned = df1.reindex(datetime_level, method='ffill')
        df1_aligned.index = df2.index
        df1 = df1_aligned
    elif not df1.index.equals(df2.index):
        try:
            if isinstance(df1.index, pd.MultiIndex) and isinstance(df2.index, pd.MultiIndex):
                df2 = df2.reindex(df1.index)
            else:
                df2 = df2.reindex(df1.index)
        except Exception:
            pass
    
    try:
        result = op_func(df1, df2)
    except (ValueError, TypeError) as e:
        if 'identically-labeled' in str(e) or 'Can only compare' in str(e) or 'index' in str(e).lower():
            if isinstance(df1.index, pd.MultiIndex) and isinstance(df2.index, pd.MultiIndex):
                df2 = df2.reindex(df1.index, fill_value=0)
            elif isinstance(df1.index, pd.MultiIndex):
                datetime_level = df1.index.get_level_values('datetime')
                df2 = df2.reindex(datetime_level, method='ffill')
                df2.index = df1.index
            result = op_func(df1, df2)
        else:
            raise
    
    return result
    
def AND(df1, df2):
    """Logical AND with index alignment."""
    df1_aligned, df2_aligned = _align_for_operation(df1, df2)
    return np.bitwise_and(df1_aligned.astype(np.bool_), df2_aligned.astype(np.bool_))

def OR(df1, df2):
    """Logical OR with index alignment."""
    df1_aligned, df2_aligned = _align_for_operation(df1, df2)
    return np.bitwise_or(df1_aligned.astype(np.bool_), df2_aligned.astype(np.bool_))

def WHERE(condition, true_value, false_value):
    """Conditional expression (WHERE) with index alignment."""
    
    if isinstance(condition, (pd.DataFrame, pd.Series)):
        target_index = condition.index
    elif isinstance(true_value, (pd.DataFrame, pd.Series)):
        target_index = true_value.index
    elif isinstance(false_value, (pd.DataFrame, pd.Series)):
        target_index = false_value.index
    else:
        return np.where(condition, true_value, false_value)
    
    if isinstance(true_value, (pd.DataFrame, pd.Series)) and not true_value.index.equals(target_index):
        if isinstance(target_index, pd.MultiIndex) and not isinstance(true_value.index, pd.MultiIndex):
            datetime_level = target_index.get_level_values('datetime')
            true_value = true_value.reindex(datetime_level, method='ffill')
            true_value.index = target_index
        else:
            true_value = true_value.reindex(target_index, fill_value=0)
    
    if isinstance(false_value, (pd.DataFrame, pd.Series)) and not false_value.index.equals(target_index):
        if isinstance(target_index, pd.MultiIndex) and not isinstance(false_value.index, pd.MultiIndex):
            datetime_level = target_index.get_level_values('datetime')
            false_value = false_value.reindex(datetime_level, method='ffill')
            false_value.index = target_index
        else:
            false_value = false_value.reindex(target_index, fill_value=0)
    
    if isinstance(condition, (pd.DataFrame, pd.Series)) and not condition.index.equals(target_index):
        condition = condition.reindex(target_index, fill_value=False)
    
    result = np.where(condition, true_value, false_value)
    
    if isinstance(result, np.ndarray) and isinstance(target_index, pd.MultiIndex):
        result = pd.Series(result, index=target_index)
    elif isinstance(result, np.ndarray) and isinstance(target_index, pd.Index):
        result = pd.Series(result, index=target_index)
    
    return result

def where(condition, true_value, false_value):
    return WHERE(condition, true_value, false_value)

def IF_ELSE(condition, true_value, false_value):
    return WHERE(condition, true_value, false_value)

def if_else(condition, true_value, false_value):
    return WHERE(condition, true_value, false_value)

def ifelse(condition, true_value, false_value):
    return WHERE(condition, true_value, false_value)

def _align_for_operation(df1, df2):
    """Align two DataFrame/Series indices for binary ops."""
    if not isinstance(df1, (pd.DataFrame, pd.Series)) and not isinstance(df2, (pd.DataFrame, pd.Series)):
        return df1, df2
    
    if not isinstance(df1, (pd.DataFrame, pd.Series)):
        return df1, df2
    if not isinstance(df2, (pd.DataFrame, pd.Series)):
        return df1, df2
    
    if isinstance(df1.index, pd.MultiIndex) and not isinstance(df2.index, pd.MultiIndex):
        datetime_level = df1.index.get_level_values('datetime')
        if isinstance(df2, pd.DataFrame):
            df2_aligned = df2.reindex(datetime_level, method='ffill')
        else:
            df2_aligned = df2.reindex(datetime_level, method='ffill')
        df2_aligned.index = df1.index
        return df1, df2_aligned
    elif not isinstance(df1.index, pd.MultiIndex) and isinstance(df2.index, pd.MultiIndex):
        datetime_level = df2.index.get_level_values('datetime')
        if isinstance(df1, pd.DataFrame):
            df1_aligned = df1.reindex(datetime_level, method='ffill')
        else:
            df1_aligned = df1.reindex(datetime_level, method='ffill')
        df1_aligned.index = df2.index
        return df1_aligned, df2
    elif not df1.index.equals(df2.index):
        try:
            if isinstance(df1.index, pd.MultiIndex) and isinstance(df2.index, pd.MultiIndex):
                df2_aligned = df2.reindex(df1.index)
                return df1, df2_aligned
            else:
                df2_aligned = df2.reindex(df1.index)
                return df1, df2_aligned
        except Exception:
            return df1, df2
    
    return df1, df2

def GT(df1, df2):
    """Greater than with index alignment."""
    return _compare_with_alignment(df1, df2, operator.gt)

def LT(df1, df2):
    """Less than with index alignment."""
    return _compare_with_alignment(df1, df2, operator.lt)

def GE(df1, df2):
    """Greater or equal with index alignment."""
    return _compare_with_alignment(df1, df2, operator.ge)

def LE(df1, df2):
    """Less or equal with index alignment."""
    return _compare_with_alignment(df1, df2, operator.le)

def EQ(df1, df2):
    """Equal with index alignment."""
    return _compare_with_alignment(df1, df2, operator.eq)

def NE(df1, df2):
    """Not equal with index alignment."""
    return _compare_with_alignment(df1, df2, operator.ne)

def _compare_with_alignment(df1, df2, op_func):
    """Compare two DataFrame/Series with index alignment."""
    
    if not isinstance(df1, (pd.DataFrame, pd.Series)) and not isinstance(df2, (pd.DataFrame, pd.Series)):
        return op_func(df1, df2)
    
    if not isinstance(df1, (pd.DataFrame, pd.Series)):
        return op_func(df2, df1) if op_func in [operator.lt, operator.le] else op_func(df1, df2)
    if not isinstance(df2, (pd.DataFrame, pd.Series)):
        return op_func(df1, df2)
    
    if isinstance(df1.index, pd.MultiIndex) and not isinstance(df2.index, pd.MultiIndex):
        datetime_level = df1.index.get_level_values('datetime')
        if isinstance(df2, pd.DataFrame):
            df2_aligned = df2.reindex(datetime_level, method='ffill')
        else:
            df2_aligned = df2.reindex(datetime_level, method='ffill')
        df2_aligned.index = df1.index
        df2 = df2_aligned
    elif not isinstance(df1.index, pd.MultiIndex) and isinstance(df2.index, pd.MultiIndex):
        datetime_level = df2.index.get_level_values('datetime')
        if isinstance(df1, pd.DataFrame):
            df1_aligned = df1.reindex(datetime_level, method='ffill')
        else:
            df1_aligned = df1.reindex(datetime_level, method='ffill')
        df1_aligned.index = df2.index
        df1 = df1_aligned
    elif not df1.index.equals(df2.index):
        try:
            if isinstance(df1.index, pd.MultiIndex) and isinstance(df2.index, pd.MultiIndex):
                df2 = df2.reindex(df1.index)
            else:
                df2 = df2.reindex(df1.index)
        except Exception:
            pass
    
    try:
        result = op_func(df1, df2)
    except (ValueError, TypeError) as e:
        if 'identically-labeled' in str(e) or 'Can only compare' in str(e):
            if isinstance(df1.index, pd.MultiIndex) and isinstance(df2.index, pd.MultiIndex):
                df2 = df2.reindex(df1.index, fill_value=0)
            elif isinstance(df1.index, pd.MultiIndex):
                datetime_level = df1.index.get_level_values('datetime')
                df2 = df2.reindex(datetime_level, method='ffill')
                df2.index = df1.index
            result = op_func(df1, df2)
        else:
            raise
    
    return result



def MACD(price_df, short_window=12, long_window=26):
    """MACD indicator (short EMA - long EMA)."""
    short_ema = EMA(price_df, short_window)
    long_ema = EMA(price_df, long_window)
    macd = short_ema - long_ema
    return macd


def RSI(price_df, window=14):
    """RSI (Relative Strength Index)."""
    price_change = DELTA(price_df, 1)
    up = (price_change > 0) * price_change
    down = (price_change < 0) * ABS(price_change)
    avg_up = EMA(up, window)
    avg_down = EMA(down, window)
    rsi = 100 - (100 / (1 + (avg_up / avg_down)))
    return rsi




def _calculate_rolling_mean(group_data):
    """Dynamic rolling mean for one group."""
    price_group, window_group, group_name = group_data
    result = pd.Series(index=price_group.index, dtype=float)
    
    for i in range(len(price_group)):
        curr_window = int(window_group.iloc[i].values)
        if curr_window < 1:
            curr_window = 1
        if i < curr_window:
            result.iloc[i] = price_group.iloc[:i+1].mean()
        else:
            result.iloc[i] = price_group.iloc[i-curr_window+1:i+1].mean()
    
    return group_name, result

def _calculate_rolling_std(group_data):
    """Dynamic rolling std for one group."""
    price_group, window_group, group_name = group_data
    result = pd.Series(index=price_group.index, dtype=float)
    
    for i in range(len(price_group)):
        curr_window = int(window_group.iloc[i].values)
        if curr_window < 1:
            curr_window = 1
        if i < curr_window:
            result.iloc[i] = price_group.iloc[:i+1].std()
        else:
            result.iloc[i] = price_group.iloc[i-curr_window+1:i+1].std()
    
    return group_name, result



@datatype_adapter
def BB_MIDDLE(price_df, window, n_jobs=-1):
    """Bollinger Band middle (supports dynamic window, parallel)."""
    if isinstance(window, (int, float)):
        return price_df.groupby('instrument').transform(lambda x: x.rolling(int(window), min_periods=1).mean())
    else:
        window.index = price_df.index
        groups_data = [
            (price_group, 
             window.xs(group_name, level='instrument'), 
             group_name)
            for group_name, price_group in price_df.groupby('instrument')
        ]
        
        results = Parallel(n_jobs=n_jobs)(
            delayed(_calculate_rolling_mean)(group_data)
            for group_data in groups_data
        )
        
        final_result = pd.concat([result for _, result in sorted(results, key=lambda x: x[0])])
        return final_result

@datatype_adapter
def BB_UPPER(price_df, window, n_jobs=-1):
    """Bollinger Band upper (supports dynamic window, parallel)."""
    
    if isinstance(window, (int, float)):
        middle_band = BB_MIDDLE(price_df, window, n_jobs)
        std = price_df.groupby('instrument').transform(lambda x: x.rolling(int(window), min_periods=1).std())
    else:
        window.index = price_df.index
        middle_band = BB_MIDDLE(price_df, window, n_jobs)
        groups_data = [
            (price_group, 
             window.xs(group_name, level='instrument'), 
             group_name)
            for group_name, price_group in price_df.groupby('instrument')
        ]
        
        results = Parallel(n_jobs=n_jobs)(
            delayed(_calculate_rolling_std)(group_data)
            for group_data in groups_data
        )
        
        std = pd.concat([result for _, result in sorted(results, key=lambda x: x[0])])
    
    return middle_band + std

@datatype_adapter
def BB_LOWER(price_df, window, n_jobs=-1):
    """Bollinger Band lower (supports dynamic window, parallel)."""
    
    if isinstance(window, (int, float)):
        middle_band = BB_MIDDLE(price_df, window, n_jobs)
        std = price_df.groupby('instrument').transform(lambda x: x.rolling(int(window), min_periods=1).std())
    else:
        window.index = price_df.index
        middle_band = BB_MIDDLE(price_df, window, n_jobs)
        groups_data = [
            (price_group, 
             window.xs(group_name, level='instrument'), 
             group_name)
            for group_name, price_group in price_df.groupby('instrument')
        ]
        
        results = Parallel(n_jobs=n_jobs)(
            delayed(_calculate_rolling_std)(group_data)
            for group_data in groups_data
        )
        
        std = pd.concat([result for _, result in sorted(results, key=lambda x: x[0])])
    
    return middle_band - std


# ------------------------------
# Lowercase operators aligned to the current operator library behavior
# ------------------------------
def _coerce_integral_window(p, name: str = "window") -> int:
    if isinstance(p, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer-compatible value, got {p!r}")
    if isinstance(p, (int, np.integer)):
        return int(p)
    if isinstance(p, (float, np.floating)):
        if not np.isfinite(p) or not float(p).is_integer():
            raise ValueError(f"{name} must be an integer-compatible value, got {p!r}")
        return int(p)
    out = int(p)
    if out != p:
        raise ValueError(f"{name} must be an integer-compatible value, got {p!r}")
    return out


def _window(p, name: str = "window") -> int:
    out = _coerce_integral_window(p, name)
    if out < 1:
        raise ValueError(f"{name} must be >= 1, got {out}")
    return out


def _default_min_periods(p: int) -> int:
    p = _window(p)
    return max(1, int(p / 2))


def _ensure_series(x):
    if isinstance(x, pd.DataFrame):
        if x.shape[1] == 1:
            return x.iloc[:, 0]
    return x


def _to_wide_panel(x):
    s = _ensure_series(x)
    if isinstance(s, pd.Series) and isinstance(s.index, pd.MultiIndex) and 'instrument' in s.index.names:
        return s.unstack('instrument')
    if isinstance(s, pd.DataFrame):
        return s
    return pd.DataFrame(s)


def _from_wide_panel(wide: pd.DataFrame, template):
    s = _ensure_series(template)
    if isinstance(s, pd.Series) and isinstance(s.index, pd.MultiIndex) and 'instrument' in s.index.names:
        try:
            out = wide.stack(future_stack=True)
        except TypeError:
            out = wide.stack(dropna=False)
        out.index = out.index.set_names(['datetime', 'instrument'])
        return out.reindex(s.index)
    return wide


def _group_roll(series, p, op):
    p = _window(p)
    mp = _default_min_periods(p)
    return series.groupby(level='instrument').transform(lambda s: op(s.rolling(p, min_periods=mp)))


def neg(df):
    return -df


def pos(df):
    return np.maximum(df, 0.0)


def neg_part(df):
    return np.minimum(df, 0.0)


def add(df1, df2):
    return df1 + df2


def sub(df1, df2):
    return df1 - df2


def mul(df1, df2):
    return df1 * df2


def div(df1, df2):
    with np.errstate(divide='ignore', invalid='ignore'):
        out = df1 / df2
    return out.replace([np.inf, -np.inf], np.nan)


def pct(df1, df2):
    return div(df1, df2) - 1


def last(value):
    return value


def safe_div(df1, df2, eps=5e-2):
    v1 = np.asarray(df1, dtype=float)
    v2 = np.asarray(df2, dtype=float)
    v2 = np.where(np.abs(v2) < eps, np.where(v2 >= 0, eps, -eps), v2)
    out = v1 / v2
    out = np.where(np.isinf(out), np.nan, out)
    if isinstance(df1, (pd.Series, pd.DataFrame)):
        return pd.DataFrame(out, index=df1.index, columns=getattr(df1, 'columns', None)).squeeze()
    return out


def diff(df):
    return ts_delta(df, 1)


def diff_q(df):
    return ts_pct(df, 1)


def tanh(df):
    return np.tanh(df)


def cs_rank(df, pct=True, ascending=True):
    s = _ensure_series(df)
    if isinstance(s, pd.Series):
        grp = s.groupby(level='datetime')
        r = grp.rank(method='average', ascending=ascending)
        if not pct:
            return r
        cnt = grp.transform('count')
        out = (r - 1.0) / (cnt - 1.0)
        out = out.where(cnt > 1, 0.5)
        return out.where(s.notna())
    r = s.rank(axis=1, method='average', ascending=ascending)
    if not pct:
        return r
    cnt = s.notna().sum(axis=1)
    out = r.sub(1.0).div((cnt - 1.0).replace(0, np.nan), axis=0)
    single_mask = cnt == 1
    if single_mask.any():
        out.loc[single_mask] = out.loc[single_mask].where(s.loc[single_mask].isna(), 0.5)
    return out.where(s.notna())


def cs_neutralize(df):
    df = _ensure_series(df)
    if isinstance(df, pd.Series):
        return df - df.groupby(level='datetime').transform('mean')
    return df.sub(df.mean(axis=1), axis=0)


def cs_zscore(df):
    df = _ensure_series(df)
    if isinstance(df, pd.Series):
        mean = df.groupby(level='datetime').transform('mean')
        std = df.groupby(level='datetime').transform('std')
        std = std.where(std > 0, np.nan)
        return (df - mean) / std
    std = df.std(axis=1).replace(0, np.nan)
    return df.sub(df.mean(axis=1), axis=0).div(std, axis=0)


def cs_weighted_zscore(df, weights):
    df = _ensure_series(df)
    w = _ensure_series(weights)
    if isinstance(df, pd.Series):
        data = pd.DataFrame({'v': df, 'w': w})
        def _z(g):
            ww = g['w'].fillna(0.0)
            vv = g['v']
            wsum = ww.sum()
            if wsum <= 0:
                return pd.Series(np.nan, index=g.index)
            mean = (vv * ww).sum() / wsum
            var = ((vv - mean) ** 2 * ww).sum() / wsum
            std = np.sqrt(var)
            if std <= 0:
                return pd.Series(0.0, index=g.index)
            return (vv - mean) / std
        return data.groupby(level='datetime', group_keys=False).apply(_z)
    return cs_zscore(df)


def cs_scale(df):
    df = _ensure_series(df)
    if isinstance(df, pd.Series):
        g = df.groupby(level='datetime')
        mn = g.transform('min')
        mx = g.transform('max')
        den = (mx - mn).replace(0, np.nan)
        return (df - mn) / den
    mn = df.min(axis=1)
    mx = df.max(axis=1)
    return df.sub(mn, axis=0).div((mx - mn).replace(0, np.nan), axis=0)


def cs_mean(df):
    return _ensure_series(df).groupby(level='datetime').transform('mean')


def cs_std(df):
    return _ensure_series(df).groupby(level='datetime').transform('std')


def cs_sum(df):
    return _ensure_series(df).groupby(level='datetime').transform('sum')


def cs_median(df):
    return _ensure_series(df).groupby(level='datetime').transform('median')


def mean(df):
    return cs_mean(df)


def std(df):
    return cs_std(df)


def skew(df):
    return SKEW(df)


def kurt(df):
    return KURT(df)


def median(df):
    return cs_median(df)


def rank(df):
    return cs_rank(df)


def ts_delay(df, n):
    return delay(df, n)


def delay(df, p=1):
    p = _coerce_integral_window(p, "p")
    s = _ensure_series(df)
    return s.groupby(level='instrument').transform(lambda x: x.shift(p))


def ts_delta(df, p=1):
    p = _coerce_integral_window(p, "p")
    s = _ensure_series(df)
    return s.groupby(level='instrument').transform(lambda x: x - x.shift(p))


def delta(df, p=1):
    return ts_delta(df, p)


def ts_pct(df, p=1):
    with np.errstate(divide='ignore', invalid='ignore'):
        out = _ensure_series(df) / delay(df, p) - 1
    return out.replace([np.inf, -np.inf], np.nan)


def ts_rank(df, p=5):
    p = _window(p, "p")
    s = _ensure_series(df)
    mp = _default_min_periods(p)
    return s.groupby(level='instrument').transform(lambda x: x.rolling(p, min_periods=mp).rank(pct=True))


def ts_decay_linear(df, p=5):
    p = _window(p, "p")
    wide = _to_wide_panel(df)
    weights = np.arange(1, p + 1)
    df_filled = wide.fillna(0)
    df_mask = wide.notna().astype(float)
    numerator = 0
    denominator = 0
    for i in range(p):
        w = weights[i]
        shifted_df = df_filled.shift(p - 1 - i).fillna(0)
        shifted_mask = df_mask.shift(p - 1 - i).fillna(0)
        numerator += shifted_df * shifted_mask * w
        denominator += shifted_mask * w
    result = numerator / denominator
    result = result.where(wide.notna())
    return _from_wide_panel(result, df)


def ts_sum(df, p=5):
    return _group_roll(_ensure_series(df), p, lambda r: r.sum())


def ts_prod(df, p=5):
    p = _window(p, "p")
    mp = _default_min_periods(p)
    s = _ensure_series(df)
    return s.groupby(level='instrument').transform(lambda x: x.rolling(p, min_periods=mp).apply(np.prod, raw=True))


def ts_mean(df, p=5):
    return _group_roll(_ensure_series(df), p, lambda r: r.mean())


def ts_var(df, p=5):
    return _group_roll(_ensure_series(df), p, lambda r: r.var())


def ts_std(df, p=20):
    return _group_roll(_ensure_series(df), p, lambda r: r.std())


def ts_ir(df, p=20):
    m = ts_mean(df, p)
    s = ts_std(df, p)
    return m / s.replace(0, np.nan)


def ts_skew(df, p=5):
    p = _window(p, "p")
    s = _ensure_series(df)
    mp = _default_min_periods(p)
    return s.groupby(level='instrument').transform(lambda x: x.rolling(p, min_periods=mp).skew())


def ts_kur(df, p=5):
    p = _window(p, "p")
    s = _ensure_series(df)
    mp = _default_min_periods(p)
    return s.groupby(level='instrument').transform(lambda x: x.rolling(p, min_periods=mp).kurt())


def ts_zscore(df, p=5):
    s = _ensure_series(df)
    m = ts_mean(s, p)
    sd = ts_std(s, p).replace(0, np.nan)
    return (s - m) / sd


def ts_median(df, p=5):
    return _group_roll(_ensure_series(df), p, lambda r: r.median())


def ts_argmax(df, p=5):
    p = _window(p, "p")
    mp = _default_min_periods(p)
    s = _ensure_series(df)
    return s.groupby(level='instrument').transform(
        lambda x: x.rolling(p, min_periods=mp).apply(lambda w: len(w) - int(np.argmax(w)) - 1, raw=True)
    )


def ts_argmin(df, p=5):
    p = _window(p, "p")
    mp = _default_min_periods(p)
    s = _ensure_series(df)
    return s.groupby(level='instrument').transform(
        lambda x: x.rolling(p, min_periods=mp).apply(lambda w: len(w) - int(np.argmin(w)) - 1, raw=True)
    )


def ts_max(df, p=5):
    return _group_roll(_ensure_series(df), p, lambda r: r.max())


def ts_min(df, p=5):
    return _group_roll(_ensure_series(df), p, lambda r: r.min())


def ts_range(df, p=5):
    return ts_max(df, p) - ts_min(df, p)


def ts_corr(df1, df2, p=5):
    p = _window(p, "p")
    mp = _default_min_periods(p)
    s1 = _ensure_series(df1)
    s2 = _ensure_series(df2)
    return s1.groupby(level='instrument').apply(lambda x: x.rolling(p, min_periods=mp).corr(s2.xs(x.name, level='instrument'))).reset_index(level=0, drop=True)


def ts_cov(df1, df2, p=5):
    p = _window(p, "p")
    mp = _default_min_periods(p)
    s1 = _ensure_series(df1)
    s2 = _ensure_series(df2)
    return s1.groupby(level='instrument').apply(lambda x: x.rolling(p, min_periods=mp).cov(s2.xs(x.name, level='instrument'))).reset_index(level=0, drop=True)


def ts_corr1(df1, df2, p=5):
    return ts_corr(df1, df2, p)


def ts_corr2(df1, df2, p=5):
    return ts_corr(df1, df2, p)


def ts_return(df, p=1):
    return ts_pct(df, p)


def ts_mad(df, p=5):
    return TS_MAD(_ensure_series(df), p)


def ts_quantile(df, p=5, q=0.5):
    return TS_QUANTILE(_ensure_series(df), p, q)


def percentile(df, q, p=None):
    return PERCENTILE(_ensure_series(df), q, p)


def highday(df, p=5):
    return HIGHDAY(_ensure_series(df), p)


def lowday(df, p=5):
    return LOWDAY(_ensure_series(df), p)


def sumac(df, p=10):
    return SUMAC(_ensure_series(df), p)


def sma(df, n=None, m=None):
    return SMA(_ensure_series(df), n, m)


def wma(df, p=20):
    return WMA(_ensure_series(df), p)


def ema1(df, p):
    p = _window(p, "p")
    mp = _default_min_periods(p)
    s = _ensure_series(df)
    return s.groupby(level='instrument').transform(lambda x: x.ewm(span=int(p), min_periods=mp).mean())


def ema2(df, p):
    p = _window(p, "p")
    mp = _default_min_periods(p)
    s = _ensure_series(df)
    return s.groupby(level='instrument').transform(lambda x: x.ewm(com=int(p), min_periods=mp).mean())


def ema3(df, alpha, min_periods=2):
    min_periods = _coerce_integral_window(min_periods, "min_periods")
    s = _ensure_series(df)
    return s.groupby(level='instrument').transform(lambda x: x.ewm(alpha=alpha, min_periods=min_periods).mean())


def log(df):
    s = _ensure_series(df)
    with np.errstate(divide='ignore', invalid='ignore'):
        out = np.log(s)
    if isinstance(out, (pd.Series, pd.DataFrame)):
        return out.replace([np.inf, -np.inf], np.nan)
    return out


def sqrt(df):
    return SQRT(_ensure_series(df))


def power(df, n=2):
    return POW(_ensure_series(df), n)


def sign(df):
    return SIGN(_ensure_series(df))


def signal(df):
    return sign(df)


def exp(df):
    return EXP(_ensure_series(df))


def abs(df):
    return ABS(_ensure_series(df))


def max(df1, df2=None, df3=None):
    if df2 is None:
        return _ensure_series(df1).groupby(level='datetime').transform('max')
    return MAX(df1, df2, df3)


def min(df1, df2=None, df3=None):
    if df2 is None:
        return _ensure_series(df1).groupby(level='datetime').transform('min')
    return MIN(df1, df2, df3)


def inv(df):
    return INV(_ensure_series(df))


def floor(df):
    return FLOOR(_ensure_series(df))


def count(cond, p=20):
    return COUNT(_ensure_series(cond), p)


def sumif(df, p, cond):
    return SUMIF(_ensure_series(df), p, _ensure_series(cond))


def filter(df, cond):
    return FILTER(_ensure_series(df), _ensure_series(cond))


def sequence(n):
    return SEQUENCE(n)


def regbeta(df1, df2, p=5):
    return REGBETA(_ensure_series(df1), _ensure_series(df2), p)


def regresi(df1, df2, p=5):
    return REGRESI(_ensure_series(df1), _ensure_series(df2), p)


def ts_reg(df1, df2, p=5, rettype=0):
    if rettype == 1:
        return regbeta(df1, df2, p)
    return regresi(df1, df2, p)


def ts_regression(df1, df2, p=5):
    return ts_reg(df1, df2, p, rettype=1)


def ts_regression2(df1, df2, p=5, rettype=0):
    return ts_reg(df1, df2, p, rettype=rettype)


def ts_beta(df1, df2, p=5):
    return regbeta(df1, df2, p)


def rsi(df, p=14):
    return RSI(_ensure_series(df), p)


def macd(df, short_window=12, long_window=26):
    return MACD(_ensure_series(df), short_window, long_window)


def bb_middle(df, p, n_jobs=-1):
    return BB_MIDDLE(_ensure_series(df), p, n_jobs)


def bb_upper(df, p, n_jobs=-1):
    return BB_UPPER(_ensure_series(df), p, n_jobs)


def bb_lower(df, p, n_jobs=-1):
    return BB_LOWER(_ensure_series(df), p, n_jobs)


def bucket(df, n):
    wide = _to_wide_panel(df)
    ranking = wide.rank(axis=1, method='dense')
    rank_min = ranking.min(axis=1)
    rank_max = ranking.max(axis=1)
    ranking = ranking.sub(rank_min, axis=0).div(rank_max - rank_min, axis=0)
    groups = np.floor(ranking * (n - 1e-9))
    return _from_wide_panel(groups, df)


def _group_transform(df, group):
    s = _ensure_series(df)
    g = _ensure_series(group)
    frame = pd.DataFrame({'v': s, 'g': g})
    frame = frame[frame['g'].notna()]
    return frame


def group_rank(df, group=None):
    if isinstance(group, str) and group == 'market':
        return cs_rank(df)
    wide_df = _to_wide_panel(df)
    wide_group = _to_wide_panel(group)
    df_long = wide_df.stack(future_stack=True)
    group_long = wide_group.stack(future_stack=True)
    df_long = df_long.where(group_long.notna())
    grouper = [df_long.index.get_level_values(0), group_long]
    ranking = df_long.groupby(grouper).rank(method='average', pct=False)
    group_count = df_long.groupby(grouper).transform('count')
    out = (ranking - 1.0) / (group_count - 1.0)
    out = out.where(group_count > 1, 0.5)
    return _from_wide_panel(out.unstack(), df)


def group_neutralize(df, group=None):
    if isinstance(group, str) and group == 'market':
        return cs_neutralize(df)
    wide_df = _to_wide_panel(df)
    wide_group = _to_wide_panel(group)
    df_long = wide_df.stack(future_stack=True)
    group_long = wide_group.stack(future_stack=True)
    df_long = df_long.where(group_long.notna())
    grouper = [df_long.index.get_level_values(0), group_long]
    group_mean = df_long.groupby(grouper).transform('mean')
    return _from_wide_panel((df_long - group_mean).unstack(), df)


def group_zscore(df, group=None):
    if isinstance(group, str) and group == 'market':
        return cs_zscore(df)
    wide_df = _to_wide_panel(df)
    wide_group = _to_wide_panel(group)
    df_long = wide_df.stack(future_stack=True)
    group_long = wide_group.stack(future_stack=True)
    df_long = df_long.where(group_long.notna())
    grouper = [df_long.index.get_level_values(0), group_long]
    group_mean = df_long.groupby(grouper).transform('mean')
    group_std = df_long.groupby(grouper).transform('std').replace(0, np.nan)
    return _from_wide_panel(((df_long - group_mean) / group_std).unstack(), df)


def group_scale(df, group=None):
    if isinstance(group, str) and group == 'market':
        return cs_scale(df)
    wide_df = _to_wide_panel(df)
    wide_group = _to_wide_panel(group)
    df_long = wide_df.stack(future_stack=True)
    group_long = wide_group.stack(future_stack=True)
    df_long = df_long.where(group_long.notna())
    grouper = [df_long.index.get_level_values(0), group_long]
    group_min = df_long.groupby(grouper).transform('min')
    group_max = df_long.groupby(grouper).transform('max')
    return _from_wide_panel(((df_long - group_min) / (group_max - group_min)).unstack(), df)


def group_mean(df, group):
    wide_df = _to_wide_panel(df)
    wide_group = _to_wide_panel(group)
    df_long = wide_df.stack(future_stack=True)
    group_long = wide_group.stack(future_stack=True)
    df_long = df_long.where(group_long.notna())
    grouper = [df_long.index.get_level_values(0), group_long]
    out = df_long.groupby(grouper).transform('mean').unstack().where(wide_df.notna())
    return _from_wide_panel(out, df)


def group_min(df, group):
    wide_df = _to_wide_panel(df)
    wide_group = _to_wide_panel(group)
    df_long = wide_df.stack(future_stack=True)
    group_long = wide_group.stack(future_stack=True)
    df_long = df_long.where(group_long.notna())
    grouper = [df_long.index.get_level_values(0), group_long]
    out = df_long.groupby(grouper).transform('min').unstack().where(wide_df.notna())
    return _from_wide_panel(out, df)


def group_max(df, group):
    wide_df = _to_wide_panel(df)
    wide_group = _to_wide_panel(group)
    df_long = wide_df.stack(future_stack=True)
    group_long = wide_group.stack(future_stack=True)
    df_long = df_long.where(group_long.notna())
    grouper = [df_long.index.get_level_values(0), group_long]
    out = df_long.groupby(grouper).transform('max').unstack().where(wide_df.notna())
    return _from_wide_panel(out, df)


def trade_when(df, con1, con2=-1):
    s = _ensure_series(df)
    c1 = _ensure_series(con1).fillna(False).astype(bool)
    if isinstance(con2, (int, float)):
        c2 = pd.Series(False, index=s.index)
    else:
        c2 = _ensure_series(con2).fillna(False).astype(bool)
    filtered = s.where(c1).where(~c2)
    regime = c1 | c2
    return filtered.groupby(level='instrument').apply(lambda x: x.groupby(regime.xs(x.name, level='instrument').cumsum()).ffill()).reset_index(level=0, drop=True)


def select_time_single(df, con, d=0.2):
    return trade_when(df, con, -1)


def fd(df1, df2, n, weights, method='decay'):
    z1 = cs_zscore(df1)
    z2 = cs_zscore(df2)
    score = z1.abs() * z2.abs()
    masks = [(z1 > 0) & (z2 > 0), (z1 < 0) & (z2 < 0), (z1 > 0) & (z2 < 0), (z1 < 0) & (z2 > 0)]
    total = pd.Series(0.0, index=_ensure_series(df1).index)
    for m, w in zip(masks, weights):
        total = total.add(score.where(m, other=0) * w, fill_value=0)
    if method == 'decay':
        return ts_decay_linear(total, n)
    return ts_mean(total, n)


def winsorize1(df, n=4):
    s = _ensure_series(df)
    g = s.groupby(level='datetime')
    mean = g.transform('mean')
    std = g.transform('std')
    return s.clip(lower=mean - n * std, upper=mean + n * std)


def winsorize2(df, lower_bound=0.01, upper_bound=0.99):
    s = _ensure_series(df)
    g = s.groupby(level='datetime')
    lower = g.transform(lambda x: x.quantile(lower_bound))
    upper = g.transform(lambda x: x.quantile(upper_bound))
    return s.clip(lower=lower, upper=upper)


def gaussian_rank(df):
    from scipy.stats import norm
    r = cs_rank(df, pct=False)
    count = _ensure_series(df).groupby(level='datetime').transform(lambda x: x.notna().sum())
    return pd.Series(norm.ppf(np.clip(r / count.replace(0, np.nan), 1e-6, 1 - 1e-6)), index=_ensure_series(df).index)


def midmap(df):
    return -abs(cs_zscore(df) - 1)


def leftmap(df):
    r = cs_rank(df)
    return -r * log(r)


def rightmap(df):
    r = cs_rank(df)
    return -(1 - r) * log(1 - r)


def extreme_rightmap(df):
    r = cs_rank(df)
    return np.exp(-(r - 0.85) ** 2)


def extreme_leftmap(df):
    r = cs_rank(df)
    return np.exp(-(r - 0.15) ** 2)


def divide(df1, df2):
    return div(df1, df2)


def cs_corr(df1, df2):
    s1 = _ensure_series(df1)
    s2 = _ensure_series(df2)
    frame = pd.DataFrame({'x': s1, 'y': s2})
    out = frame.groupby(level='datetime').apply(lambda g: g['x'].corr(g['y']))
    return pd.Series(out, index=out.index)


def cs_multireg(X_list, y_df, min_cs=30):
    y = _ensure_series(y_df)
    x_list = [_ensure_series(x) for x in X_list]

    def _fit_one(group):
        dt = group.index.get_level_values('datetime')[0]
        yv = y.xs(dt, level='datetime').astype(float)
        xm = [x.xs(dt, level='datetime').astype(float) for x in x_list]
        X = np.column_stack([xv.values for xv in xm])
        mask = np.isfinite(yv.values) & np.isfinite(X).all(axis=1)
        if mask.sum() < min_cs:
            return pd.Series(np.nan, index=group.index)
        beta, *_ = np.linalg.lstsq(X[mask], yv.values[mask], rcond=None)
        pred = X @ beta
        resid = yv.values - pred
        return pd.Series(resid, index=group.index)

    return y.groupby(level='datetime', group_keys=False).apply(_fit_one)


def cs_pc1(df_list, min_cs=30):
    s_list = [_ensure_series(x) for x in df_list]
    base = s_list[0]

    def _pc1(group):
        dt = group.index.get_level_values('datetime')[0]
        mat = np.column_stack([s.xs(dt, level='datetime').values for s in s_list])
        mask = np.isfinite(mat).all(axis=1)
        if mask.sum() < min_cs:
            return pd.Series(np.nan, index=group.index)
        m = mat[mask]
        m = m - m.mean(axis=0)
        _, _, vt = np.linalg.svd(m, full_matrices=False)
        pc1 = m @ vt[0]
        out = np.full(mat.shape[0], np.nan)
        out[np.where(mask)[0]] = pc1
        return pd.Series(out, index=group.index)

    return base.groupby(level='datetime', group_keys=False).apply(_pc1)


def eqw(df_list, method='zscore'):
    s_list = [_ensure_series(x) for x in df_list]
    result = pd.Series(0.0, index=s_list[0].index)
    cnt = pd.Series(0.0, index=s_list[0].index)
    for s in s_list:
        if method == 'rank':
            v = cs_rank(s).fillna(0.5)
        else:
            v = cs_zscore(s).fillna(0.0)
        result = result.add(v, fill_value=0.0)
        cnt = cnt.add(s.notna().astype(float), fill_value=0.0)
    return div(result, cnt.replace(0, np.nan))


def ts_multireg(X_list, y_df, n, min_periods=None, rettype=0):
    if rettype == 3:
        # Approximate R^2 by corr^2 using equal-weight composite regressor.
        x_proxy = eqw(X_list, method='zscore')
        corr = ts_corr(x_proxy, y_df, n)
        return corr ** 2
    x_proxy = eqw(X_list, method='zscore')
    return ts_reg(x_proxy, y_df, n, rettype=0)


def regime_switch(df):
    # Keep behavior explicit; TQ currently keeps this as placeholder as well.
    return _ensure_series(df)


def fast_corr(df):
    d = df if isinstance(df, pd.DataFrame) else pd.DataFrame(df)
    return d.corr()


def precise_corr(df):
    d = df if isinstance(df, pd.DataFrame) else pd.DataFrame(df)
    return d.corr()
