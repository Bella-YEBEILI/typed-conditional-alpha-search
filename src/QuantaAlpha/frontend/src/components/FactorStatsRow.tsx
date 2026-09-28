import React from 'react';
import { TrendingDown, Activity, BarChart3, Layers } from 'lucide-react';
import { RealtimeMetrics } from '@/types';
import { formatPercent, formatNumber } from '@/utils';

interface FactorStatsRowProps {
  metrics: RealtimeMetrics | null;
  onBacktest?: () => void;
}

export const FactorStatsRow: React.FC<FactorStatsRowProps> = ({ metrics, onBacktest }) => {
  const hasValue = (v?: number | null) => v !== undefined && v !== null && v !== 0;

  return (
    <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-5 gap-4 animate-fade-in-up w-full">
      {/* RankIC */}
      <div className="glass rounded-xl p-4 card-hover h-[120px] flex flex-col justify-between">
        <div className="flex items-center gap-2 mb-2">
          <div className="p-1.5 rounded-lg bg-primary/10">
            <Activity className="h-4 w-4 text-primary" />
          </div>
          <span className="text-xs text-muted-foreground font-medium">RankIC</span>
        </div>
        <div className={`text-2xl font-bold ${hasValue(metrics?.rankIc) ? 'text-primary' : 'text-muted-foreground/40'}`}>
          {hasValue(metrics?.rankIc) ? formatNumber(metrics!.rankIc, 4) : 'N/A'}
        </div>
      </div>

      {/* RankICIR */}
      <div className="glass rounded-xl p-4 card-hover h-[120px] flex flex-col justify-between">
        <div className="flex items-center gap-2 mb-2">
          <div className="p-1.5 rounded-lg bg-purple-500/10">
            <BarChart3 className="h-4 w-4 text-purple-500" />
          </div>
          <span className="text-xs text-muted-foreground font-medium">RankICIR</span>
        </div>
        <div className={`text-2xl font-bold ${hasValue(metrics?.rankIcir) ? 'text-purple-500' : 'text-muted-foreground/40'}`}>
          {hasValue(metrics?.rankIcir) ? formatNumber(metrics!.rankIcir, 4) : 'N/A'}
        </div>
      </div>

      {/* Max Drawdown */}
      <div className="glass rounded-xl p-4 card-hover h-[120px] flex flex-col justify-between">
        <div className="flex items-center gap-2 mb-2">
          <div className="p-1.5 rounded-lg bg-destructive/10">
            <TrendingDown className="h-4 w-4 text-destructive" />
          </div>
          <span className="text-xs text-muted-foreground font-medium">多头扣费最大回撤</span>
        </div>
        <div className={`text-2xl font-bold ${hasValue(metrics?.longNetMaxDrawdown) ? 'text-destructive' : 'text-muted-foreground/40'}`}>
          {hasValue(metrics?.longNetMaxDrawdown) ? formatPercent(metrics!.longNetMaxDrawdown!) : 'N/A'}
        </div>
      </div>

      {/* Quality Distribution */}
      <div className="glass rounded-xl p-4 card-hover h-[120px] flex flex-col justify-between">
        <div className="flex items-center gap-2 mb-2">
          <div className="p-1.5 rounded-lg bg-amber-500/10">
            <Layers className="h-4 w-4 text-amber-500" />
          </div>
          <span className="text-xs text-muted-foreground font-medium">因子质量分布</span>
        </div>
        <div className="flex items-center gap-3">
          <div className="text-center">
            <div className="text-lg font-bold text-success">{metrics?.highQualityFactors ?? 0}</div>
            <div className="text-[10px] text-muted-foreground">高质量</div>
          </div>
          <div className="text-muted-foreground/30">/</div>
          <div className="text-center">
            <div className="text-lg font-bold text-amber-500">{metrics?.mediumQualityFactors ?? 0}</div>
            <div className="text-[10px] text-muted-foreground">中质量</div>
          </div>
          <div className="text-muted-foreground/30">/</div>
          <div className="text-center">
            <div className="text-lg font-bold text-muted-foreground">{metrics?.lowQualityFactors ?? 0}</div>
            <div className="text-[10px] text-muted-foreground">低质量</div>
          </div>
        </div>
      </div>

      {/* Backtest Button */}
      <div
        className="h-[120px] flex flex-col justify-center items-center cursor-pointer hover:scale-[1.05] transition-all group"
        onClick={onBacktest}
      >
        <div className="p-4 rounded-full bg-primary text-primary-foreground shadow-lg mb-2 group-hover:shadow-xl group-hover:bg-primary/90 transition-all duration-300">
          <BarChart3 className="h-6 w-6" />
        </div>
        <div className="text-sm font-bold text-foreground/80 group-hover:text-primary transition-colors">
          一键回测
        </div>
      </div>
    </div>
  );
};
