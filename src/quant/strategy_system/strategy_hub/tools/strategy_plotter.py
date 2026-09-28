import matplotlib.pyplot as plt
import pandas as pd

from ..core.config import STRATEGY_START_DATE,STRATEGY_END_DATE

class StrategyPlotter:
    def __init__(self,dm):
        self.dm = dm

    def plot_strategy(self,
                      strategy_result:dict[str,pd.Series],
                      start:str|None=None,
                      end:str|None=None,
                      benchmark:str="zz1000s",
                      plot_size:tuple=(6,4.5)):
        if start is None:
            start = STRATEGY_START_DATE
        if end is None:
            end = STRATEGY_END_DATE

        dates = strategy_result["pnl"].sort_index().loc[start:end].index
        gross_pnl = strategy_result["gross_pnl"].sort_index().reindex(index=dates)
        pnl = strategy_result["pnl"].sort_index().reindex(index=dates)
        turnovers = strategy_result["turnovers"].sort_index().reindex(index=dates).fillna(0.0)
        effs = strategy_result["effs"].sort_index().reindex(index=dates).fillna(0.0)
        benchmark_returns = self.dm.get_data("index_data")[benchmark+"_returns"].reindex(index=dates).fillna(0.0)
        benchmark_pnl = (1+benchmark_returns).cumprod()

        def _calc_rets(pnl:pd.Series)->pd.Series:
            rets = pnl/pnl.shift(1)-1
            rets.iloc[0] = pnl.iloc[0]-1
            return rets

        gross_rets = _calc_rets(gross_pnl)
        rets = _calc_rets(pnl)
        gross_excess_pnl = (1+(gross_rets-benchmark_returns)).cumprod()
        excess_pnl = (1+(rets-benchmark_returns)).cumprod()

        fig,axes = plt.subplots(2,2,figsize=plot_size,constrained_layout=True)

        pd.DataFrame({
            "gross_pnl":gross_pnl,
            f"{benchmark}_pnl":benchmark_pnl,
            "gross_excess_pnl":gross_excess_pnl,
        }).plot(ax=axes[0,0],title="Gross PNL",linewidth=1)

        pd.DataFrame({
            "pnl":pnl,
            f"{benchmark}_pnl":benchmark_pnl,
            "excess_pnl":excess_pnl,
        }).plot(ax=axes[0,1],title="Net PNL",linewidth=1)

        turnovers.plot(ax=axes[1,0],title="Turnover",linewidth=1,label="turnovers")
        effs.plot(ax=axes[1,1],title="Effs",linewidth=1,label="effs")

        for ax in axes.flat:
            ax.grid(True,alpha=0.3)
            ax.tick_params(axis="x",labelrotation=30,labelsize=7)
            ax.tick_params(axis="y",labelsize=8)
            ax.title.set_fontsize(10)
            ax.legend(fontsize=6,loc="upper left",framealpha=0.5)

        plt.show()
        return fig,axes
