import React, { useRef, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/Card';
import { Badge } from '@/components/ui/Badge';
import { TimeSeriesData, RealtimeMetrics, LogEntry } from '@/types';
import { formatNumber, formatPercent, formatDateTime } from '@/utils';
import { TrendingUp, TrendingDown, BarChart3, Activity } from 'lucide-react';

interface LiveChartsProps {
  equityCurve: TimeSeriesData[];
  drawdownCurve: TimeSeriesData[];
  metrics: RealtimeMetrics | null;
  isRunning: boolean;
  logs: LogEntry[];
}

export const LiveCharts: React.FC<LiveChartsProps> = ({
  equityCurve,
  drawdownCurve,
  metrics,
  isRunning,
  logs,
}) => {
  const logContainerRef = useRef<HTMLDivElement>(null);
  const logEndRef = useRef<HTMLDivElement>(null);
  const isAutoScrollRef = useRef(true);

  const handleScroll = () => {
    if (logContainerRef.current) {
      const { scrollHeight, clientHeight, scrollTop } = logContainerRef.current;
      const distanceToBottom = Math.abs(scrollHeight - clientHeight - scrollTop);
      isAutoScrollRef.current = distanceToBottom < 100;
    }
  };

  useEffect(() => {
    if (isAutoScrollRef.current) {
      requestAnimationFrame(() => {
        if (logContainerRef.current) {
          const { scrollHeight, clientHeight } = logContainerRef.current;
          logContainerRef.current.scrollTo({
            top: scrollHeight - clientHeight,
            behavior: 'smooth'
          });
        }
      });
    }
  }, [logs]);

  const handleMouseLeave = () => {
    if (logContainerRef.current) {
      const { scrollHeight, clientHeight, scrollTop } = logContainerRef.current;
      const distanceToBottom = Math.abs(scrollHeight - clientHeight - scrollTop);
      if (distanceToBottom < 100) {
        isAutoScrollRef.current = true;
      }
    }
  };

  const getLogIcon = (level: LogEntry['level']) => {
    switch (level) {
      case 'success': return '✅';
      case 'error': return '❌';
      case 'warning': return '⚠️';
      default: return '•';
    }
  };

  const getLogColor = (level: LogEntry['level']) => {
    switch (level) {
      case 'success': return 'text-success';
      case 'error': return 'text-destructive';
      case 'warning': return 'text-warning';
      default: return 'text-muted-foreground';
    }
  };

  const StatCard = ({ icon: Icon, label, value, format, color, bgColor }: {
    icon: any; label: string; value?: number | null; format: 'percent' | 'number';
    color: string; bgColor: string;
  }) => {
    const hasValue = value !== undefined && value !== null && value !== 0;
    const displayValue = hasValue
      ? format === 'percent' ? formatPercent(value!) : formatNumber(value!, 4)
      : 'N/A';
    return (
      <div className="glass rounded-xl p-4 card-hover h-[120px] flex flex-col justify-between">
        <div className="flex items-center gap-2 mb-2">
          <div className={`p-1.5 rounded-lg ${bgColor}`}>
            <Icon className={`h-4 w-4 ${color}`} />
          </div>
          <span className="text-xs text-muted-foreground font-medium">{label}</span>
        </div>
        <div className={`text-2xl font-bold ${hasValue ? color : 'text-muted-foreground/40'}`}>
          {displayValue}
        </div>
      </div>
    );
  };

  return (
    <div className="space-y-4">
      {metrics && (
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 animate-fade-in-up">
          <StatCard
            icon={TrendingUp}
            label="多头年化收益"
            value={metrics.annualReturn}
            format="percent"
            color="text-success"
            bgColor="bg-success/10"
          />
          <StatCard
            icon={TrendingUp}
            label="多头扣费收益"
            value={metrics.longNetReturn}
            format="percent"
            color="text-emerald-500"
            bgColor="bg-emerald-500/10"
          />
          <StatCard
            icon={BarChart3}
            label="多空年化收益"
            value={metrics.lsReturn}
            format="percent"
            color="text-blue-500"
            bgColor="bg-blue-500/10"
          />
          <StatCard
            icon={BarChart3}
            label="多空扣费收益"
            value={metrics.lsNetReturn}
            format="percent"
            color="text-indigo-500"
            bgColor="bg-indigo-500/10"
          />
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-4 gap-4">
        <Card className="glass card-hover animate-fade-in-left lg:col-span-4 h-[400px] flex flex-col">
          <CardHeader className="pb-3">
            <CardTitle className="text-base flex items-center gap-2">
              实时日志
            </CardTitle>
          </CardHeader>
          <CardContent className="flex-1 min-h-0">
            <div
              ref={logContainerRef}
              onScroll={handleScroll}
              onMouseLeave={handleMouseLeave}
              className="h-full overflow-y-auto rounded-lg bg-yellow-50 p-3 font-mono text-xs space-y-1 border border-yellow-100 scroll-smooth"
            >
              {logs.length === 0 ? (
                <div className="flex h-full items-center justify-center text-muted-foreground">
                  等待日志输出...
                </div>
              ) : (
                <>
                  {logs.map((log) => (
                    <div key={log.id} className="flex gap-2 items-start animate-fade-in-up">
                      <span className="text-muted-foreground shrink-0">
                        {formatDateTime(log.timestamp).split(' ')[1]}
                      </span>
                      <span className="shrink-0">{getLogIcon(log.level)}</span>
                      <span className={getLogColor(log.level)}>{log.message}</span>
                    </div>
                  ))}
                  <div ref={logEndRef} className="h-px w-full" />
                </>
              )}
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
};
