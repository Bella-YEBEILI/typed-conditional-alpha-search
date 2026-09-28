import h5py
import numpy as np
try:
    from optree import NONE_IS_LEAF
except ModuleNotFoundError:
    NONE_IS_LEAF = object()
import pandas as pd
from typing import Union,Callable,Optional
import os,sys,inspect,json,hashlib,time,inspect

from quantaalpha.runtime import env_path, minute_h5_path

# --------------------------
# 基础辅助函数
# --------------------------

def _load_axes(f:h5py.File)->tuple[pd.DatetimeIndex,pd.Index,pd.Index]:
    # 读取axis组下的元数据
    dates = f["axis/dates"][:].astype(str)
    minutes = f["axis/minutes"][:].astype(str)
    stocks = f["axis/stocks"][:].astype(str)
    return pd.DatetimeIndex(dates),pd.Index(minutes),pd.Index(stocks)

def _get_date_range(dates:pd.DatetimeIndex,startdate:str|None,enddate:str|None)->tuple[int,int]:
    if startdate is not None:
        start_i = dates.searchsorted(str(startdate),side="left")
    else:
        start_i = 0
    if enddate is not None:
        end_i = dates.searchsorted(str(enddate),side="right")
    else:
        end_i = len(dates)
    return start_i,end_i

def _get_minute_range(minutes:pd.Index,startminute:str|None,endminute:str|None)->tuple[int,int]:
    if startminute is not None:
        start_i = int(minutes.searchsorted(str(startminute), side="left"))
    else:
        start_i=0
    if endminute is not None:
        end_i = int(minutes.searchsorted(str(endminute), side="right"))
    else:
        end_i=len(minutes)
    start_i = max(0, min(start_i, len(minutes)))
    end_i = max(start_i, min(end_i, len(minutes)))
    return int(start_i),int(end_i)

# --------------------------
# 切片读取 
# --------------------------

# fld: opens,highs,lows,closes,volumes,amounts,vwaps,turnovers,ratios(量比)
# minute: "0931" -> "1500"

def load_single_date(f:h5py.File,fld:str,date:str,startminute:str|None=None,endminute:str|None=None)->pd.DataFrame:
    # 读取某天所有股票的分钟线数据
    dates,minutes,stocks = _load_axes(f)
    try:
        di=np.where(dates==pd.to_datetime(date))[0][0]
    except IndexError:
        raise ValueError(f"Date {date} not found")
    si,ei = _get_minute_range(minutes,startminute,endminute)
    data = f[f"data/{fld}"][di,si:ei,:]
    return pd.DataFrame(data,index=minutes[si:ei],columns=stocks)

def load_single_minute(f:h5py.File,fld:str,minute:str,startdate:str|None=None,enddate:str|None=None)->pd.DataFrame:
    dates,minutes,stocks = _load_axes(f)
    mi = np.where(minutes==minute)[0][0]
    si,ei = _get_date_range(dates,startdate,enddate)
    n_days = ei-si
    n_stocks = len(stocks)
    ds = f[f"data/{fld}"]
    out = np.empty((n_days,n_stocks),dtype=np.float32)
    batch_size = 1
    for i in range(0,n_days,batch_size):
        curr_s = si+i
        curr_e = min(si+n_days,curr_s+batch_size)
        out_s = i
        out_e = out_s+(curr_e-curr_s)
        out[out_s:out_e,:]=ds[curr_s:curr_e,mi,:]
    return pd.DataFrame(out,index=dates[si:ei],columns=stocks)

# --------------------------
# 缓存功能辅助函数 
# --------------------------

def _stable_json(obj):
    # 将字典对象序列化为json字符串
    return json.dumps(obj,sort_keys=True,ensure_ascii=False,separators=(",",":"))

def _sha256(s):
    # 将字符串hash成固定长度
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def _func_code_hash(f):
    # 算子版本控制
    try:
        g = f.py_func
    except Exception:
        g = f
    try:
        b = g.__code__.co_code # python字节码
        c = repr(g.__code__.co_consts).encode("utf-8") # 常量池
        return hashlib.sha256(b+c).hexdigest()
    except Exception:
        return _sha256(repr(f))

def _df_save_npz(path,df):
    arr = np.asarray(df.values,dtype=np.float32,order="C")
    idx = np.asarray(df.index.astype(str),dtype="U")
    cols = np.asarray(df.columns.astype(str),dtype="U")
    tmp = path+f".tmp.{os.getpid()}.npz"
    np.savez(tmp,arr=arr,index=idx,columns=cols)
    os.replace(tmp,path)

def _df_load_npz(path):
    d = np.load(path,allow_pickle=False)
    idx = pd.DatetimeIndex(d["index"].astype(str))
    cols = pd.Index(d["columns"].astype(str))
    return pd.DataFrame(d["arr"],index=idx,columns=cols)

def _caller_script_dir():
    # 自动定位缓存目录
    m = sys.modules.get("__main__",None)
    p = getattr(m,"__file__",None) if m is not None else None
    if p:
        return os.path.dirname(os.path.abspath(p))
    # fallback:从调用栈找第一个“不是本文件”的frame
    this = os.path.abspath(__file__)
    for fr in inspect.stack():
        fp = os.path.abspath(fr.filename)
        if fp!=this and os.path.exists(fp):
            return os.path.dirname(fp)
    return os.getcwd()

def _get_last_all_date():
    # 用get_data("all_dates")最后一天检查数据更新
    try:
        h5_path = minute_h5_path()
        if not os.path.exists(h5_path):
            return "NA"
        with h5py.File(h5_path, "r") as f:
            dates = f["axis/dates"][:].astype(str)
            if len(dates) > 0:
                return str(dates[-1])
    except Exception:
        pass
    return "NA"
    
# --------------------------
# 因子研究接口 
# --------------------------
class MinuteFactorEngine:
    """
    核心功能: 
    循环处理每一天, 把minutes x stocks的二维block处理成stocks的一维vector, 最后拼成dates x stocks的二维block
    """
    _OPS:dict[str,Callable]={} 
    _OPS_PARAM_ORDER:dict[str,tuple[str,...]]={}

    def __init__(self,h5_path:str|None=None):
        self.h5_path = str(h5_path) if h5_path is not None else str(
            minute_h5_path()
        )

    @classmethod
    def register(cls,name:str,param_order:tuple[str,...]=()):
        """通用注册装饰器"""
        def decorator(func):
            cls._OPS[name] = func
            cls._OPS_PARAM_ORDER[name] = tuple(param_order) if param_order else ()
            return func
        return decorator
    
    @staticmethod
    def _parse_op_spec(spec):
        if spec is None:
            return None,()
        if isinstance(spec,str):
            return spec,()
        if isinstance(spec,tuple):
            if len(spec)==0:
                raise ValueError("Empty operator spec tuple")
            name = spec[0]
            if len(spec)==1:
                return name,()
            if len(spec)==2 and isinstance(spec[1],dict):
                return name,spec[1]
            return name,spec[1:]
        raise ValueError(F"Invalid operator spec:{spec}")
    
    def _resolve_op(self,spec,role:str):
        name,params = self._parse_op_spec(spec)
        if name is None:
            return None,()
        func = self._OPS.get(name)
        if not func:
            raise ValueError(f"{role} '{name} not found. Available:{list(self._OPS.keys())}")
        if isinstance(params,dict):
            order = self._OPS_PARAM_ORDER.get(name,())
            if not order:
                raise ValueError(f"{role} '{name}' does not define param_order,kwargs-style params not allowed")
            extra_args = []
            for k in order:
                if k not in params:
                    raise ValueError(f"{role} '{name}' missing param:{k}")
                extra_args.append(params[k])
            return func,tuple(extra_args)
        if isinstance(params,tuple):
            return func,params
        raise ValueError(f"{role} '{name}' invalid params:{params}")

    def run(self,
            inputs:Union[str,list[str]],
            operator,
            aggregator=None,
            preprocess=None,
            startminute:str="0931",
            endminute:str="1457",
            startdate:str=None,
            enddate:str=None)->pd.DataFrame:
        # --- 参数标准化 ---
        if isinstance(inputs,str):inputs=[inputs]
        if preprocess:
            if not isinstance(preprocess,list):
                preprocess = [preprocess] 
            if len(preprocess) != len(inputs):
                raise ValueError("Preprocess ops length must match inputs count")
        # --- 获取算子和参数 ---
        main_func,main_extra = self._resolve_op(operator,"Operator")
        agg_func,agg_extra = self._resolve_op(aggregator,"Aggregator") if aggregator else (None,())
        pre_funcs = []
        pre_extras = []
        if preprocess:
            for p in preprocess:
                f,ex = self._resolve_op(p,"Preprocess")
                pre_funcs.append(f)
                pre_extras.append(ex)
        # --- 逐日循环 ---
        with h5py.File(self.h5_path,"r") as f:
            dates,minutes,stocks = _load_axes(f)
            dsi,dei = _get_date_range(dates,startdate,enddate)
            msi,mei = _get_minute_range(minutes,startminute,endminute)
            n_days = dei-dsi
            n_stocks = len(stocks)
            input_dsets = [f[f"data/{fld}"] for fld in inputs]
            out = np.full((n_days,n_stocks),np.nan,dtype=np.float32)
            for i in range(n_days):
                day_idx = dsi+i
                raw_data_list = [ds[day_idx,msi:mei,:] for ds in input_dsets]
                if preprocess:
                    proc_data = []
                    for k,data in enumerate(raw_data_list):
                        pf = pre_funcs[k]
                        if pf:
                            proc_data.append(pf(data,*pre_extras[k]))
                        else:
                            proc_data.append(data)
                else:
                    proc_data = raw_data_list
                res = main_func(*proc_data,*main_extra)
                if agg_func:
                    out[i,:] = agg_func(res,*agg_extra)
                else:
                    out[i,:] = res
        return pd.DataFrame(out,index=dates[dsi:dei],columns=stocks)
    
    def cached_run(self,
                   inputs:Union[str,list[str]],
                   operator,
                   aggregator=None,
                   preprocess=None,
                   startminute:str="0931",
                   endminute:str="1500",
                   startdate:str=None,
                   enddate:str=None,
                   force:bool=False)->pd.DataFrame:
        # 自动定位“当前调用脚本”的目录,在旁边建缓存目录
        base_dir = _caller_script_dir()
        cache_dir = os.path.join(base_dir,".mfe_cache")
        os.makedirs(cache_dir,exist_ok=True)
        # 参数标准化(和run保持一致口径)
        if isinstance(inputs,str):
            inputs = [inputs]
        pp = None
        if preprocess:
            if not isinstance(preprocess,list):
                pp_specs = [preprocess]
            else:
                pp_specs = list(preprocess)
            if len(pp_specs)!=len(inputs):
                raise ValueError("Preprocess ops length must match inputs count")
        else:
            pp_specs = []
        # 用到的函数代码及其参数标识
        def _spec_to_meta(role,spec):
            name,params = self._parse_op_spec(spec)
            if name is None:
                return None
            fn = self._OPS.get(name,None)
            return {"role":role,
                    "name":name,
                    "params":params if params is not None else (),
                    "code_hash":_func_code_hash(fn) if fn is not None else None}
        ops_meta = []
        m = _spec_to_meta("operator",operator)
        if m is None:
            raise ValueError("Operator is None")
        ops_meta.append(m)
        if aggregator:
            m = _spec_to_meta("aggregator",aggregator)
            if m is None:
                raise ValueError("Aggregator spec invalid")
            ops_meta.append(m)
        for p in pp_specs:
            name,params = self._parse_op_spec(p)
            if name is None:
                ops_meta.append({"role":"preprocess","name":None,"params":(),"code_hash":None})
                continue
            fn = self._OPS.get(name,None)
            ops_meta.append({
                "role":"preprocess",
                "name":name,
                "params":params if params is not None else (),
                "code_hash":_func_code_hash(fn) if fn is not None else None
            })
        payload = {
            "inputs":list(inputs),
            "operator":operator,
            "aggregator":aggregator,
            "preprocess":list(pp_specs),
            "startminute":startminute,
            "endminute":endminute,
            "startdate":startdate,
            "enddate":enddate,
            "ops":ops_meta,
            "all_dates_last":_get_last_all_date(),
        }
        key = _sha256(_stable_json(payload))
        npz_path = os.path.join(cache_dir,key+".npz")
        meta_path = os.path.join(cache_dir,key+".json")
        if (not force) and os.path.exists(npz_path) and os.path.exists(meta_path):
            print(f"[MFE] cache hit: {npz_path}")
            return _df_load_npz(npz_path)
        df = self.run(inputs=inputs,
                      operator=operator,
                      aggregator=aggregator,
                      preprocess=pp_specs if pp_specs else None,
                      startminute=startminute,
                      endminute=endminute,
                      startdate=startdate,
                      enddate=enddate)
        _df_save_npz(npz_path,df)
        print(f"[MFE] cache saved: {npz_path}")
        meta = {"key":key,"payload":payload,"created_at":time.strftime("%Y-%m-%d %H:%M:%S")}
        tmpm = meta_path+f".tmp.{os.getpid()}"
        with open(tmpm,"w",encoding="utf-8") as f:
            f.write(_stable_json(meta))
        os.replace(tmpm,meta_path)
        return df
