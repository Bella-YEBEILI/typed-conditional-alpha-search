import os
import sys
from pathlib import Path

import pandas as pd
from flask import Flask,render_template,request

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__),"..",".."))
if project_root not in sys.path:
    sys.path.insert(0,project_root)

from quant.data_system.data_hub.main import DataManager

app = Flask(__name__)
_dashboard_cache:dict[str,dict] = {}
STRATEGY_BASE_DIR = Path("/home/workspace/common/quant/quant_strategy/strategy_results")

BENCHMARK_OPTIONS = [
    ("","不选"),
    ("hs300s_returns","沪深300"),
    ("sz50s_returns","上证50"),
    ("zz500s_returns","中证500"),
    ("zz800s_returns","中证800"),
    ("zz1000s_returns","中证1000"),
    ("mscias_returns","MSCI A"),
    ("zzhls_returns","中证红利低波"),
    ("zzhls_h_returns","中证红利低波高股息"),
]
BENCHMARK_LABELS = dict(BENCHMARK_OPTIONS)


class StrategyDashboard:
    def __init__(self,pkl_path:str):
        self.pkl_path = str(Path(pkl_path).expanduser())
        bundle = pd.read_pickle(self.pkl_path)
        self.gross_pnl = self._to_series(bundle["gross_pnl"])
        self.pnl = self._to_series(bundle["pnl"])
        self.turnovers = self._to_series(bundle["turnovers"])
        self.effs = self._to_series(bundle["effs"])
        self.index_data = DataManager().get_data("index_data")

    def _to_series(self,value)->pd.Series:
        if isinstance(value,pd.DataFrame):
            value = value.iloc[:,0]
        series = pd.Series(value,dtype=float).copy()
        series.index = pd.to_datetime(series.index)
        return series.sort_index().fillna(0.0)

    def _format_series(self,series:pd.Series)->list[list[str|float]]:
        return [[idx.strftime("%Y-%m-%d"),float(val)] for idx,val in series.items()]

    def _summary_value(self,series:pd.Series)->float:
        return float(series.iloc[-1])

    def _nav_to_returns(self,series:pd.Series)->pd.Series:
        returns = series.pct_change()
        if len(returns)>0:
            returns.iloc[0] = 0.0
        return returns.fillna(0.0)

    def _win_rate(self,series:pd.Series)->float|None:
        valid = series.iloc[1:]
        if len(valid)==0:
            return None
        return float((valid>0).mean())

    def build_net_payload(self,name:str,color:str,kind:str)->dict:
        raw_nav = self.pnl.sort_index()
        nav_ret = self._nav_to_returns(raw_nav)
        nav = (1+nav_ret).cumprod()
        nav_drawdown = nav/nav.cummax()-1
        return {
            "series":{
                "kind":kind,
                "name":name,
                "color":color,
                "data":self._format_series(nav_ret),
            },
            "summary":{
                "latest_date":nav.index[-1].strftime("%Y-%m-%d"),
                "nav":self._summary_value(nav),
                "nav_drawdown":float(nav_drawdown.min()),
                "daily_win_rate":self._win_rate(nav_ret),
            },
        }

    def build_payload(self,benchmark:str)->dict:
        raw_nav = self.pnl.sort_index()
        raw_gross_nav = self.gross_pnl.reindex(raw_nav.index).ffill()

        gross_ret = self._nav_to_returns(raw_gross_nav)
        nav_ret = self._nav_to_returns(raw_nav)

        gross_nav = (1+gross_ret).cumprod()
        nav = (1+nav_ret).cumprod()
        nav_drawdown = nav/nav.cummax()-1

        return_series = [
            {"kind":"gross","name":"毛净值","color":"#5b6577","data":self._format_series(gross_ret)},
            {"kind":"net","name":"净值","color":"#0b6e4f","data":self._format_series(nav_ret)},
        ]

        benchmark_nav = None
        excess_nav = None
        excess_drawdown = None
        has_benchmark = benchmark!=""

        if has_benchmark:
            benchmark_ret = self.index_data[benchmark].reindex(nav.index).fillna(0.0)
            benchmark_nav = (1+benchmark_ret).cumprod()
            excess_ret = nav_ret-benchmark_ret
            excess_nav = (1+excess_ret).cumprod()
            excess_drawdown = excess_nav/excess_nav.cummax()-1
            return_series.extend([
                {"kind":"benchmark","name":"基准净值","color":"#e59f00","data":self._format_series(benchmark_ret)},
                {"kind":"excess","name":"超额净值","color":"#1d4ed8","data":self._format_series(excess_ret)},
            ])

        return {
            "chart_payload":{
                "compare_mode":False,
                "drawdown_kinds":["net","excess"],
                "return_series":return_series,
                "has_benchmark":has_benchmark,
            },
            "summary":{
                "latest_date":nav.index[-1].strftime("%Y-%m-%d"),
                "gross_nav":self._summary_value(gross_nav),
                "nav":self._summary_value(nav),
                "nav_drawdown":float(nav_drawdown.min()),
                "daily_win_rate":self._win_rate(nav_ret),
                "benchmark_nav":None if benchmark_nav is None else self._summary_value(benchmark_nav),
                "excess_nav":None if excess_nav is None else self._summary_value(excess_nav),
                "excess_drawdown":None if excess_drawdown is None else float(excess_drawdown.min()),
            },
        }


def build_compare_payload(primary_dashboard:StrategyDashboard,
                          primary_name:str,
                          compare_dashboard:StrategyDashboard,
                          compare_name:str)->dict:
    primary_payload = primary_dashboard.build_net_payload(primary_name,"#0b6e4f","primary")
    compare_payload = compare_dashboard.build_net_payload(compare_name,"#1d4ed8","secondary")
    latest_date = max(primary_payload["summary"]["latest_date"],compare_payload["summary"]["latest_date"])
    return {
        "chart_payload":{
            "compare_mode":True,
            "drawdown_kinds":["primary","secondary"],
            "return_series":[primary_payload["series"],compare_payload["series"]],
            "has_benchmark":False,
        },
        "summary":{
            "latest_date":latest_date,
            "primary_nav":primary_payload["summary"]["nav"],
            "primary_drawdown":primary_payload["summary"]["nav_drawdown"],
            "secondary_nav":compare_payload["summary"]["nav"],
            "secondary_drawdown":compare_payload["summary"]["nav_drawdown"],
        },
    }


def prompt_pkl_path()->str:
    while True:
        pkl_path = input("Input strategy pkl path: ").strip()
        if pkl_path:
            return str(Path(pkl_path).expanduser())


def get_default_pkl_path()->str:
    pkl_path = os.environ.get("STRATEGY_DASHBOARD_PKL_PATH","").strip()
    if not pkl_path:
        pkl_path = prompt_pkl_path()
        os.environ["STRATEGY_DASHBOARD_PKL_PATH"] = pkl_path
    return str(Path(pkl_path).expanduser())


def list_strategy_options()->list[tuple[str,str]]:
    options = []
    for path in sorted(STRATEGY_BASE_DIR.rglob("*.pkl")):
        rel_path = path.relative_to(STRATEGY_BASE_DIR)
        options.append((rel_path.as_posix(),rel_path.with_suffix("").as_posix()))
    return options


def resolve_strategy(strategy:str)->tuple[str,str,list[tuple[str,str]]]:
    strategy_options = list_strategy_options()
    if strategy_options:
        option_map = dict(strategy_options)
        if strategy not in option_map:
            strategy = strategy_options[0][0]
        pkl_path = str((STRATEGY_BASE_DIR/strategy).resolve())
        return pkl_path,strategy,strategy_options

    pkl_path = get_default_pkl_path()
    fallback_key = Path(pkl_path).name
    fallback_label = Path(pkl_path).stem
    return pkl_path,fallback_key,[(fallback_key,fallback_label)]


def get_dashboard(pkl_path:str)->StrategyDashboard:
    cache_key = str(Path(pkl_path).expanduser())
    mtime = os.path.getmtime(cache_key)
    cached = _dashboard_cache.get(cache_key)
    if cached is None or cached["mtime"]!=mtime:
        cached = {
            "mtime":mtime,
            "dashboard":StrategyDashboard(cache_key),
        }
        _dashboard_cache[cache_key] = cached
    return cached["dashboard"]


def fmt_nav(v):
    if v is None:
        return "-"
    return f"{v:.4f}"


def fmt_pct(v,decimals=2):
    if v is None:
        return "-"
    return f"{v*100:.{decimals}f}%"


app.jinja_env.globals.update(fmt_nav=fmt_nav,fmt_pct=fmt_pct)


@app.route("/")
def index():
    strategy = request.args.get("strategy","").strip()
    pkl_path,strategy,strategy_options = resolve_strategy(strategy)
    option_map = dict(strategy_options)
    dashboard = get_dashboard(pkl_path)
    strategy_name = option_map.get(strategy,Path(pkl_path).stem)

    compare_strategy = request.args.get("compare_strategy","").strip()
    if compare_strategy not in option_map or compare_strategy==strategy:
        compare_strategy = ""
    compare_strategy_name = option_map.get(compare_strategy,"")
    compare_dashboard = None if compare_strategy=="" else get_dashboard(str((STRATEGY_BASE_DIR/compare_strategy).resolve()))

    benchmark = request.args.get("benchmark","").strip()
    if benchmark not in BENCHMARK_LABELS:
        benchmark = ""

    compare_mode = compare_dashboard is not None
    if compare_mode:
        payload = build_compare_payload(dashboard,strategy_name,compare_dashboard,compare_strategy_name)
    else:
        payload = dashboard.build_payload(benchmark)

    return render_template(
        "strategy_overview.html",
        pkl_path=dashboard.pkl_path,
        pkl_name=Path(dashboard.pkl_path).name,
        strategy=strategy,
        strategy_name=strategy_name,
        compare_strategy=compare_strategy,
        compare_strategy_name=compare_strategy_name,
        strategy_options=strategy_options,
        benchmark=benchmark,
        benchmark_options=BENCHMARK_OPTIONS,
        benchmark_name=BENCHMARK_LABELS.get(benchmark,""),
        compare_mode=compare_mode,
        chart_payload=payload["chart_payload"],
        summary=payload["summary"],
    )


if __name__=="__main__":
    get_dashboard(resolve_strategy("")[0])
    app.run(host="0.0.0.0",port=5002,debug=False,use_reloader=False)
