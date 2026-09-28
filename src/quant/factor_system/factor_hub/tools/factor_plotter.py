"""
FactorPlotter用于把单个因子的回测结果画成研究图。

当前职责：
1. 从factor_result中提取收益、换手、RankIC、分组收益等序列。
2. 按统一布局绘制净值、RankIC累计、分组收益、换手和覆盖率图。
3. 根据传入的params截取回测区间，并在图中体现交易成本影响。

当前规则：
- 输入应当是已经计算完成的factor_result字典。
- 缺失字段会退回为空Series或空DataFrame，避免绘图阶段直接报错。
- 图形主要面向研究场景展示，不负责结果持久化。

这个组件只负责结果可视化，不负责计算因子值、回测结果或绩效指标。
"""

import matplotlib.pyplot as plt
import pandas as pd
from typing import Any

from ..core.factor_performance_engine import DEFAULT_PARAMS

class FactorPlotter:
    def __init__(self):
        pass

    def plot_result(self,
                    id:str,
                    factor_result:dict[str,Any],
                    params:dict=DEFAULT_PARAMS,
                    plot_size:tuple[float,float]=(6,4.5)):
        start = params["start"]
        end = params["end"]
        cost = params["cost"]

        long_rets = factor_result.get("long_rets",pd.Series(dtype=float)).loc[start:end]
        long_turnovers = factor_result.get("long_turnovers",pd.Series(dtype=float)).loc[start:end]
        short_rets = factor_result.get("short_rets",pd.Series(dtype=float)).loc[start:end]
        short_turnovers = factor_result.get("short_turnovers",pd.Series(dtype=float)).loc[start:end]
        rankics = factor_result.get("rankics",pd.Series(dtype=float)).loc[start:end]
        long_nums = factor_result.get("long_nums",pd.Series(dtype=float)).loc[start:end]
        coverages = factor_result.get("coverages",pd.Series(dtype=float)).loc[start:end]

        long_netrets = long_rets-(long_turnovers/2*cost)
        short_netrets = short_rets-(short_turnovers/2*cost)
        ls_rets = (long_rets+short_rets)/2
        ls_netrets = (long_netrets+short_netrets)/2
        ls_turnovers = (long_turnovers+short_turnovers)/2

        group_rets = factor_result.get("group_rets",pd.DataFrame()).loc[start:end]
        demeaned_group_rets = group_rets.sub(group_rets.mean(axis=1),axis=0)

        fig,axes = plt.subplots(3,2,figsize=plot_size,constrained_layout=True)
        fig.suptitle(id,fontsize=10)

        nav_df = pd.DataFrame({
            "long_rets":(1+long_rets.fillna(0)).cumprod(),
            "long_netrets":(1+long_netrets.fillna(0)).cumprod(),
            "ls_rets":(1+ls_rets.fillna(0)).cumprod(),
            "ls_netrets":(1+ls_netrets.fillna(0)).cumprod(),
        })
        nav_df.plot(ax=axes[0,0],title="Return NAV",legend=True,linewidth=1)
        axes[0,0].legend(fontsize=6,loc="upper left",framealpha=0.5)

        rankic_curve_df = pd.DataFrame({
            "rankics":rankics.fillna(0).cumsum(),
        })
        rankic_curve_df.plot(ax=axes[0,1],title="RankIC CumSum",legend=True,linewidth=1)
        axes[0,1].legend(fontsize=6,loc="upper left",framealpha=0.5)

        group_cumret = group_rets.fillna(0).cumsum()
        group_cumret.plot(ax=axes[1,0],title="Group CumRet",legend=False,linewidth=0.8)
        demeaned_group_rets.fillna(0).cumsum().plot(ax=axes[1,1],title="Demeaned Group CumRet",legend=False,linewidth=0.8)

        group_handles,group_labels = axes[1,0].get_legend_handles_labels()
        if group_handles:
            axes[1,0].legend(group_handles,[x.replace("group_","") for x in group_labels],fontsize=5,loc="upper left",ncol=2,framealpha=0.5)
        demeaned_handles,demeaned_labels = axes[1,1].get_legend_handles_labels()
        if demeaned_handles:
            axes[1,1].legend(demeaned_handles,[x.replace("group_","") for x in demeaned_labels],fontsize=5,loc="upper left",ncol=2,framealpha=0.5)

        turnover_df = pd.DataFrame({
            "long_turnovers":long_turnovers,
            "ls_turnovers":ls_turnovers,
        })
        turnover_df.plot(ax=axes[2,0],title="Turnover",legend=True,linewidth=1)
        turnover_handles,turnover_labels = axes[2,0].get_legend_handles_labels()
        axes[2,0].legend(turnover_handles,[x.replace("_turnovers","") for x in turnover_labels],fontsize=6,loc="upper left",framealpha=0.5)

        ax_l = axes[2,1]
        ax_r = ax_l.twinx()
        long_nums.plot(ax=ax_l,color="tab:blue",label="long_nums",title="Long Nums / Coverage",linewidth=1)
        coverages.plot(ax=ax_r,color="tab:orange",label="coverages",linewidth=1)
        h1,l1 = ax_l.get_legend_handles_labels()
        h2,l2 = ax_r.get_legend_handles_labels()
        ax_l.legend(h1+h2,l1+l2,loc="upper left",fontsize=8)
        old_legend = ax_r.get_legend()
        if old_legend is not None:
            old_legend.remove()
        ax_l.set_ylabel("long_nums")
        ax_r.set_ylabel("coverages")

        ax_r.tick_params(axis="y",labelsize=8)
        for ax in axes.flat:
            ax.grid(True,alpha=0.3)
            ax.tick_params(axis="x",labelrotation=30,labelsize=7)
            ax.tick_params(axis="y",labelsize=8)
            ax.title.set_fontsize(10)
        plt.show()
