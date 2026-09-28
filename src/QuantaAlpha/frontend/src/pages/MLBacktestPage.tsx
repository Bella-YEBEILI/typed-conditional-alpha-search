import React, { useState, useEffect, useCallback, useRef } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/Card';
import { Button } from '@/components/ui/Button';
import { Badge } from '@/components/ui/Badge';
import {
  TrendingUp, Play, Square, Loader2, AlertCircle,
  FileText, Settings2, ChevronDown, RefreshCw, Shield
} from 'lucide-react';
import { listFactorLibraries } from '@/services/api';
import { formatNumber, formatPercent } from '@/utils';

// ========================== Types ==========================

interface MLBacktestConfig {
  model: 'lgbm' | 'xgb';
  optuna_trials: number;
  label_type: 'return_regression' | 'direction_classification' | 'top5_classification';
  split_mode: 'walk_forward';
  train_lookback: number;
  validation_size: number;
  gap_size: number;
  test_size: number;
  walk_test_start: string;
  walk_test_end: string;
  top_n: number;
  universe: 'hs300' | 'zz500' | 'zz800' | 'zz1000' | 'sz50';
  factor_library: string;
  quality_filter: 'high' | 'medium' | 'all';
}

interface MLBacktestMetrics {
  IC?: number;
  ICIR?: number;
  RankIC?: number;
  RankICIR?: number;
  annualized_return?: number;
  max_drawdown?: number;
  information_ratio?: number;
  sharpe_ratio?: number;
  calmar_ratio?: number;
  win_rate?: number;
  cumulative_curve?: { date: string; value: number }[];
  daily_selected_stocks?: { date: string; stocks: string[] }[];
  [key: string]: any;
}

interface LogEntry {
  id: string;
  timestamp: number;
  level: 'info' | 'warning' | 'error' | 'success';
  message: string;
}

type TaskStatus = 'idle' | 'running' | 'completed' | 'failed' | 'cancelled';

// ========================== Constants ==========================

const LABEL_OPTIONS = [
  { value: 'return_regression', label: '收益率回归', desc: '预测个股原始收益率' },
  { value: 'direction_classification', label: '涨跌分类', desc: '预测涨跌方向(二分类)' },
  { value: 'top5_classification', label: 'Top5%分类', desc: '预测排名前5%的股票(二分类)' },
] as const;

const MODEL_OPTIONS = [
  { value: 'lgbm', label: 'LightGBM', desc: '梯度提升树，训练速度快' },
  { value: 'xgb', label: 'XGBoost', desc: '经典梯度提升，泛化能力强' },
] as const;

const QUALITY_OPTIONS = [
  { value: 'high', label: '仅高质量', desc: 'Train与Test均通过质量检测' },
  { value: 'medium', label: '中等及以上', desc: 'Train或Test至少一项通过' },
  { value: 'all', label: '全部因子', desc: '不做质量过滤' },
] as const;

// ========================== Sub-Components ==========================

/** Visual timeline diagram showing the rolling window structure */
const RollingWindowDiagram: React.FC<{
  rollingWindow: number;
  validationSize: number;
  gapSize: number;
  testSize: number;
}> = ({ rollingWindow, validationSize, gapSize, testSize }) => {
  const total = rollingWindow + validationSize + gapSize + testSize;

  const trainPct = (rollingWindow / total) * 100;
  const valPct = (validationSize / total) * 100;
  const gapPct = Math.max((gapSize / total) * 100, 1.5); // min width for visibility
  const testPct = (testSize / total) * 100;

  return (
    <div className="space-y-3">
      <div className="flex items-center gap-2 text-sm font-medium">
        <Settings2 className="h-4 w-4 text-primary" />
        滚动窗口结构示意
      </div>

      {/* Bar diagram */}
      <div className="relative h-12 rounded-lg overflow-hidden flex border border-border/50">
        {/* Train */}
        <div
          className="h-full bg-blue-500/70 flex items-center justify-center relative group transition-all"
          style={{ width: `${trainPct}%` }}
        >
          <span className="text-xs font-medium text-white drop-shadow-sm truncate px-1">
            Train ({rollingWindow}天)
          </span>
        </div>

        {/* Validation */}
        <div
          className="h-full bg-yellow-500/70 flex items-center justify-center relative group transition-all"
          style={{ width: `${valPct}%` }}
        >
          <span className="text-xs font-medium text-white drop-shadow-sm truncate px-1">
            Val ({validationSize}天)
          </span>
        </div>

        {/* Gap */}
        <div
          className="h-full bg-red-500/60 flex items-center justify-center relative group transition-all border-l border-r border-red-700/30"
          style={{ width: `${gapPct}%`, minWidth: '20px' }}
        >
          <span className="text-xs font-bold text-white drop-shadow-sm">
            {gapSize}
          </span>
        </div>

        {/* Test */}
        <div
          className="h-full bg-green-500/70 flex items-center justify-center relative group transition-all"
          style={{ width: `${testPct}%` }}
        >
          <span className="text-xs font-medium text-white drop-shadow-sm truncate px-1">
            Test (~{testSize}天)
          </span>
        </div>
      </div>

      {/* Legend */}
      <div className="flex flex-wrap gap-4 text-xs text-muted-foreground">
        <span className="flex items-center gap-1.5">
          <span className="w-3 h-3 rounded-sm bg-blue-500/70" />
          训练集: 模型学习特征
        </span>
        <span className="flex items-center gap-1.5">
          <span className="w-3 h-3 rounded-sm bg-yellow-500/70" />
          验证集: Optuna调参
        </span>
        <span className="flex items-center gap-1.5">
          <span className="w-3 h-3 rounded-sm bg-red-500/60" />
          Gap: 防信息泄漏
        </span>
        <span className="flex items-center gap-1.5">
          <span className="w-3 h-3 rounded-sm bg-green-500/70" />
          测试集: 样本外评估
        </span>
      </div>
    </div>
  );
};

const getCalendarGapDays = (valEnd: string, testStart: string) => {
  if (!valEnd || !testStart) return 0;
  const diffMs = new Date(testStart).getTime() - new Date(valEnd).getTime();
  return Math.max(0, Math.floor(diffMs / 86400000) - 1);
};

const validateWalkForwardConfig = (
  trainLookback: number,
  validationSize: number,
  gapSize: number,
  testSize: number,
  walkTestStart: string,
  walkTestEnd: string,
  topN: number
) => {
  if (!walkTestStart || !walkTestEnd) return '请填写样本外测试区间';
  if (new Date(walkTestStart).getTime() > new Date(walkTestEnd).getTime()) return '测试开始日期不能晚于结束日期';
  if (trainLookback < 20) return '训练回看天数至少 20 个交易日';
  if (validationSize < 5) return 'Valid 天数至少 5 个交易日';
  if (gapSize < 1) return 'Gap 至少 1 个交易日';
  if (testSize < 1) return '每次预测/持有天数至少 1 个交易日';
  if (topN < 1) return 'Top N 选股数量至少为 1';
  return null;
};

const DateRangeDiagram: React.FC<{
  trainStart: string;
  trainEnd: string;
  valStart: string;
  valEnd: string;
  testStart: string;
  testEnd: string;
}> = ({ trainStart, trainEnd, valStart, valEnd, testStart, testEnd }) => {
  const gapDays = getCalendarGapDays(valEnd, testStart);

  return (
    <div className="space-y-3">
      <div className="flex items-center gap-2 text-sm font-medium">
        <Settings2 className="h-4 w-4 text-primary" />
        固定样本外切分示意
      </div>
      <div className="relative h-12 rounded-lg overflow-hidden flex border border-border/50">
        <div className="h-full bg-blue-500/70 flex items-center justify-center transition-all" style={{ width: '42%' }}>
          <span className="text-xs font-medium text-white drop-shadow-sm truncate px-1">Train</span>
        </div>
        <div className="h-full bg-yellow-500/70 flex items-center justify-center transition-all" style={{ width: '24%' }}>
          <span className="text-xs font-medium text-white drop-shadow-sm truncate px-1">Valid</span>
        </div>
        <div className="h-full bg-red-500/60 flex items-center justify-center transition-all border-l border-r border-red-700/30" style={{ width: '10%', minWidth: '32px' }}>
          <span className="text-xs font-bold text-white drop-shadow-sm">Gap</span>
        </div>
        <div className="h-full bg-green-500/70 flex items-center justify-center transition-all" style={{ width: '24%' }}>
          <span className="text-xs font-medium text-white drop-shadow-sm truncate px-1">Test</span>
        </div>
      </div>
      <div className="grid grid-cols-1 md:grid-cols-4 gap-2 text-xs text-muted-foreground">
        <span className="flex items-center gap-1.5">
          <span className="w-3 h-3 rounded-sm bg-blue-500/70" />
          Train: {trainStart} ~ {trainEnd}
        </span>
        <span className="flex items-center gap-1.5">
          <span className="w-3 h-3 rounded-sm bg-yellow-500/70" />
          Valid: {valStart} ~ {valEnd}
        </span>
        <span className="flex items-center gap-1.5">
          <span className="w-3 h-3 rounded-sm bg-red-500/60" />
          Gap: 约 {gapDays} 个自然日
        </span>
        <span className="flex items-center gap-1.5">
          <span className="w-3 h-3 rounded-sm bg-green-500/70" />
          Test: {testStart} ~ {testEnd}
        </span>
      </div>
    </div>
  );
};

/** Metric display card */
const MetricCard: React.FC<{ label: string; value?: number; unit?: string; highlight?: boolean }> = ({
  label, value, unit = '', highlight = false,
}) => (
  <div className={`rounded-xl p-4 text-center ${highlight ? 'glass border-primary/30' : 'bg-secondary/30 rounded-lg'}`}>
    <div className="text-xs text-muted-foreground mb-1">{label}</div>
    <div className={`text-lg font-bold font-mono ${highlight ? 'text-primary' : 'text-foreground'}`}>
      {typeof value === 'number'
        ? `${formatNumber(value, 4)}${unit}`
        : '--'}
    </div>
  </div>
);

// ========================== Main Component ==========================

export const MLBacktestPage: React.FC = () => {
  // -- Configuration State --
  const [model, setModel] = useState<'lgbm' | 'xgb'>('lgbm');
  const [optunaTrialsStr, setOptunaTrialsStr] = useState('50');
  const [labelType, setLabelType] = useState<MLBacktestConfig['label_type']>('return_regression');
  const [trainLookback, setTrainLookback] = useState(504);
  const [validationSize, setValidationSize] = useState(63);
  const [gapSize, setGapSize] = useState(1);
  const [testSize, setTestSize] = useState(21);
  const [walkTestStart, setWalkTestStart] = useState('2024-01-01');
  const [walkTestEnd, setWalkTestEnd] = useState('2024-12-31');
  const [topN, setTopN] = useState(50);
  const [universe, setUniverse] = useState<'hs300' | 'zz500' | 'zz800' | 'zz1000' | 'sz50'>('hs300');
  const [factorLibrary, setFactorLibrary] = useState('');
  const [qualityFilter, setQualityFilter] = useState<'high' | 'medium' | 'all'>('high');

  // -- Libraries --
  const [libraries, setLibraries] = useState<string[]>([]);
  const [libsLoading, setLibsLoading] = useState(false);

  // -- Task State --
  const [taskId, setTaskId] = useState<string | null>(null);
  const [status, setStatus] = useState<TaskStatus>('idle');
  const [progress, setProgress] = useState<{ message: string; percent: number }>({ message: '', percent: 0 });
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const [metrics, setMetrics] = useState<MLBacktestMetrics>({});
  const [isStarting, setIsStarting] = useState(false);

  // -- Refs --
  const wsRef = useRef<WebSocket | null>(null);
  const pollingRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const logsEndRef = useRef<HTMLDivElement>(null);
  const logIdCounter = useRef(0);

  // ========================== Effects ==========================

  // Load factor libraries on mount
  const loadLibraries = useCallback(async () => {
    setLibsLoading(true);
    try {
      const resp = await listFactorLibraries();
      if (resp.success && resp.data) {
        const libs = resp.data.libraries || [];
        setLibraries(libs);
        if (libs.length > 0 && (!factorLibrary || !libs.includes(factorLibrary))) {
          setFactorLibrary(libs[0]);
        }
      }
    } catch {
      // backend might not be up
    }
    setLibsLoading(false);
  }, [factorLibrary]);

  const initDone = useRef(false);
  useEffect(() => {
    if (initDone.current) return;
    initDone.current = true;
    loadLibraries();
  }, [loadLibraries]);

  // Auto-scroll logs
  useEffect(() => {
    logsEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [logs]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (wsRef.current) {
        wsRef.current.close();
        wsRef.current = null;
      }
      if (pollingRef.current) {
        clearInterval(pollingRef.current);
        pollingRef.current = null;
      }
    };
  }, []);

  // ========================== WebSocket & Polling ==========================

  const addLog = useCallback((level: LogEntry['level'], message: string) => {
    logIdCounter.current += 1;
    setLogs(prev => [...prev, {
      id: `log-${logIdCounter.current}`,
      timestamp: Date.now(),
      level,
      message,
    }]);
  }, []);

  const connectWebSocket = useCallback((tid: string) => {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws/mining/${tid}`;
    const ws = new WebSocket(wsUrl);

    ws.onopen = () => {
      addLog('info', '[WS] 已连接到回测任务');
    };

    ws.onmessage = (event) => {
      try {
        const msg = JSON.parse(event.data);

        if (msg.type === 'log' || msg.type === 'info') {
          const logData = msg.data && typeof msg.data === 'object' ? msg.data : null;
          const logMsg = msg.message || (logData ? logData.message : msg.data) || '';
          const logLevel = msg.level || (logData ? logData.level : 'info') || 'info';
          addLog(logLevel, String(logMsg));
        } else if (msg.type === 'progress') {
          setProgress({
            message: msg.message || '',
            percent: msg.progress || msg.percent || 0,
          });
        } else if (msg.type === 'metrics' || msg.type === 'result') {
          const resultData = msg.data || {};
          const nextMetrics = resultData.metrics || msg.metrics || resultData;
          setMetrics(nextMetrics || {});
          if (resultData.status === 'completed') {
            setStatus('completed');
            setProgress({ message: '已完成', percent: 100 });
            addLog('success', '回测已完成');
          } else if (resultData.status === 'failed') {
            setStatus('failed');
            addLog('error', resultData.error || '回测失败');
          }
        } else if (msg.type === 'status') {
          if (msg.status === 'completed') {
            setStatus('completed');
            if (msg.metrics) setMetrics(msg.metrics);
            addLog('success', '回测已完成');
          } else if (msg.status === 'failed') {
            setStatus('failed');
            addLog('error', `回测失败: ${msg.error || '未知错误'}`);
          }
        } else if (msg.type === 'error') {
          addLog('error', msg.message || '未知错误');
        }
      } catch {
        // non-JSON message, treat as log
        addLog('info', event.data);
      }
    };

    ws.onclose = () => {
      addLog('info', '[WS] 连接已关闭');
      // Start polling as fallback
      startPolling(tid);
    };

    ws.onerror = () => {
      addLog('warning', '[WS] 连接出错，切换到轮询模式');
    };

    wsRef.current = ws;
  }, [addLog]);

  const startPolling = useCallback((tid: string) => {
    if (pollingRef.current) return; // already polling

    pollingRef.current = setInterval(async () => {
      try {
        const res = await fetch(`/api/v1/ml-backtest/${tid}`);
        if (!res.ok) return;
        const data = await res.json();
        const task = data.data?.task;

        if (task?.status === 'completed') {
          setStatus('completed');
          if (task.metrics) setMetrics(task.metrics);
          setProgress({ message: '已完成', percent: 100 });
          if (pollingRef.current) {
            clearInterval(pollingRef.current);
            pollingRef.current = null;
          }
        } else if (task?.status === 'failed') {
          setStatus('failed');
          addLog('error', task.progress?.message || '回测失败');
          if (pollingRef.current) {
            clearInterval(pollingRef.current);
            pollingRef.current = null;
          }
        } else if (task?.progress) {
          setProgress({
            message: task.progress.message || '',
            percent: task.progress.percent || task.progress.progress || 0,
          });
        }
      } catch {
        // ignore polling errors
      }
    }, 3000);
  }, [addLog]);

  // ========================== Actions ==========================

  const handleStart = async () => {
    if (!factorLibrary) return;
    const validationError = validateWalkForwardConfig(
      trainLookback,
      validationSize,
      gapSize,
      testSize,
      walkTestStart,
      walkTestEnd,
      topN
    );
    if (validationError) {
      addLog('error', validationError);
      return;
    }
    setIsStarting(true);
    setLogs([]);
    setMetrics({});
    setProgress({ message: '初始化中...', percent: 0 });

    const config: MLBacktestConfig = {
      model,
      optuna_trials: Math.max(1, parseInt(optunaTrialsStr, 10) || 1),
      label_type: labelType,
      split_mode: 'walk_forward',
      train_lookback: trainLookback,
      validation_size: validationSize,
      gap_size: gapSize,
      test_size: testSize,
      walk_test_start: walkTestStart,
      walk_test_end: walkTestEnd,
      top_n: topN,
      universe,
      factor_library: factorLibrary,
      quality_filter: qualityFilter,
    };

    try {
      const res = await fetch('/api/v1/ml-backtest/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(config),
      });

      if (!res.ok) {
        const text = await res.text();
        throw new Error(`HTTP ${res.status}: ${text}`);
      }

      const data = await res.json();
      const tid = data.data?.taskId || data.taskId;

      if (!tid) throw new Error('未返回任务ID');

      setTaskId(tid);
      setStatus('running');
      addLog('info', `任务已启动: ${tid}`);
      addLog('info', `模型: ${model.toUpperCase()} | Optuna试验: ${Math.max(1, parseInt(optunaTrialsStr, 10) || 1)} | 标签: ${labelType}`);
      addLog('info', `Walk-forward: train lookback ${trainLookback} trading days | valid ${validationSize} | gap ${gapSize} | test step ${testSize} | Top ${topN} | 股票池: ${universe}`);

      // Connect WebSocket
      connectWebSocket(tid);
    } catch (err: any) {
      addLog('error', `启动失败: ${err.message}`);
      setStatus('failed');
    } finally {
      setIsStarting(false);
    }
  };

  const handleStop = async () => {
    if (!taskId) return;

    try {
      await fetch(`/api/v1/ml-backtest/${taskId}/cancel`, { method: 'POST' });
      setStatus('cancelled');
      addLog('warning', '用户已取消回测任务');
    } catch {
      addLog('error', '取消任务失败');
    }

    // Cleanup connections
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
    if (pollingRef.current) {
      clearInterval(pollingRef.current);
      pollingRef.current = null;
    }
  };

  // ========================== Derived State ==========================

  const isRunning = status === 'running';
  const isFinished = status === 'completed' || status === 'failed' || status === 'cancelled';
  const showGapWarning = gapSize < 1;

  // ========================== Render ==========================

  return (
    <div className="space-y-6 animate-fade-in-up">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold flex items-center gap-3">
          <TrendingUp className="h-8 w-8 text-primary" />
          模型回测
        </h1>
        <p className="text-muted-foreground mt-1">
          使用机器学习模型对因子库进行滚动窗口回测
        </p>
      </div>

      {/* Info Banner: Anti-Leakage Explanation */}
      <Card className="glass border-primary/30">
        <CardContent className="p-4">
          <div className="flex gap-3">
            <Shield className="h-5 w-5 text-primary flex-shrink-0 mt-0.5" />
            <div className="text-sm space-y-2">
              <p className="text-muted-foreground">
                <strong className="text-foreground">滚动窗口回测</strong>：将历史数据按时间切分为
                <strong className="text-blue-400"> 训练集</strong>、
                <strong className="text-yellow-400"> 验证集</strong>、
                <strong className="text-red-400"> Gap间隔</strong>、
                <strong className="text-green-400"> 测试集</strong>，
                窗口向前滚动以模拟真实交易场景。
              </p>
              <p className="text-muted-foreground">
                <strong className="text-foreground">防信息泄漏机制</strong>：验证集与测试集之间设置 Gap 间隔（默认5个交易日），
                确保模型在调参阶段无法"看到"测试期的未来数据。Optuna 超参搜索仅在验证集上评估，
                最终模型表现以测试集为准。
              </p>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Lookahead Bias Warning */}
      {showGapWarning && (
        <Card className="glass border-destructive/50">
          <CardContent className="p-4 flex items-center gap-3">
            <AlertCircle className="h-5 w-5 text-destructive flex-shrink-0" />
            <div className="text-sm">
              <p className="font-medium text-destructive">信息泄漏风险警告</p>
              <p className="text-muted-foreground mt-0.5">
                Gap 设置为 0 天，验证集与测试集之间没有缓冲区。这可能导致<strong className="text-destructive">前视偏差(lookahead bias)</strong>，
                因为某些因子（如移动平均线、波动率等）天然包含未来信息。建议将 Gap 设为至少 5 个交易日。
              </p>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Configuration Panel */}
      <Card className="glass card-hover">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Settings2 className="h-5 w-5" />
            回测配置
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-6">
          {/* Row 1: Model + Optuna Trials */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {/* Model Selection */}
            <div>
              <label className="block text-sm font-medium mb-2">模型选择</label>
              <div className="flex gap-3">
                {MODEL_OPTIONS.map(opt => (
                  <button
                    key={opt.value}
                    onClick={() => setModel(opt.value as 'lgbm' | 'xgb')}
                    disabled={isRunning}
                    className={`flex-1 px-4 py-2.5 rounded-lg border text-sm font-medium transition-all ${
                      model === opt.value
                        ? 'border-primary bg-primary/10 text-primary'
                        : 'border-input bg-background text-muted-foreground hover:border-primary/50'
                    }`}
                  >
                    {opt.label}
                    <span className="block text-xs font-normal mt-0.5">{opt.desc}</span>
                  </button>
                ))}
              </div>
            </div>

            {/* Optuna Trials */}
            <div>
              <label className="block text-sm font-medium mb-2">Optuna 试验次数</label>
              <input
                type="number"
                value={optunaTrialsStr}
                onChange={e => setOptunaTrialsStr(e.target.value)}
                disabled={isRunning}
                min={1}
                max={500}
                className="w-full rounded-lg border border-input bg-background px-4 py-2.5 text-sm focus:border-primary focus:outline-none focus:ring-2 focus:ring-primary transition-all font-mono"
              />
              <p className="text-xs text-muted-foreground mt-1">
                超参搜索次数，越大越精确但耗时越长（建议 30-100）
              </p>
            </div>
          </div>

          {/* Row 2: Label Type */}
          <div>
            <label className="block text-sm font-medium mb-2">预测标签 (Y)</label>
            <div className="relative">
              <select
                value={labelType}
                onChange={e => setLabelType(e.target.value as MLBacktestConfig['label_type'])}
                disabled={isRunning}
                className="w-full rounded-lg border border-input bg-background px-4 py-2.5 text-sm focus:border-primary focus:outline-none focus:ring-2 focus:ring-primary transition-all appearance-none pr-10"
              >
                {LABEL_OPTIONS.map(opt => (
                  <option key={opt.value} value={opt.value}>
                    {opt.label} — {opt.desc}
                  </option>
                ))}
              </select>
              <ChevronDown className="absolute right-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground pointer-events-none" />
            </div>
          </div>

          {/* Row 3: Rolling Window Parameters */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {/* Rolling Window Size */}
            <div>
              <label className="block text-sm font-medium mb-2">滚动窗口大小</label>
              <input
                type="number"
                value={trainLookback}
                onChange={e => setTrainLookback(Math.max(20, parseInt(e.target.value) || 504))}
                disabled={isRunning}
                min={20}
                max={2000}
                className="w-full rounded-lg border border-input bg-background px-4 py-2.5 text-sm focus:border-primary focus:outline-none focus:ring-2 focus:ring-primary transition-all font-mono"
              />
              <p className="text-xs text-muted-foreground mt-1">
                训练集交易日数（默认504天 ≈ 2年）
              </p>
            </div>

            {/* Validation Size */}
            <div>
              <label className="block text-sm font-medium mb-2">验证集大小</label>
              <input
                type="number"
                value={validationSize}
                onChange={e => setValidationSize(Math.max(10, parseInt(e.target.value) || 63))}
                disabled={isRunning}
                min={10}
                max={504}
                className="w-full rounded-lg border border-input bg-background px-4 py-2.5 text-sm focus:border-primary focus:outline-none focus:ring-2 focus:ring-primary transition-all font-mono"
              />
              <p className="text-xs text-muted-foreground mt-1">
                Optuna调参用（默认63天 ≈ 3个月）
              </p>
            </div>

            {/* Gap Size */}
            <div>
              <label className="block text-sm font-medium mb-2">
                Gap 间隔
                {showGapWarning && (
                  <AlertCircle className="inline h-3.5 w-3.5 text-destructive ml-1" />
                )}
              </label>
              <input
                type="number"
                value={gapSize}
                onChange={e => setGapSize(Math.max(1, parseInt(e.target.value) || 1))}
                disabled={isRunning}
                min={1}
                max={60}
                className={`w-full rounded-lg border bg-background px-4 py-2.5 text-sm focus:outline-none focus:ring-2 transition-all font-mono ${
                  showGapWarning
                    ? 'border-destructive focus:border-destructive focus:ring-destructive'
                    : 'border-input focus:border-primary focus:ring-primary'
                }`}
              />
              <p className={`text-xs mt-1 ${showGapWarning ? 'text-destructive font-medium' : 'text-muted-foreground'}`}>
                {showGapWarning
                  ? '警告: Gap=0 有信息泄漏风险!'
                  : `验证集与测试集间隔（默认5天）`}
              </p>
            </div>
          </div>

          {/* Row 4: Walk-forward Test Period + Portfolio Construction */}
          <div className="grid grid-cols-1 md:grid-cols-5 gap-4">
            <div>
              <label className="block text-sm font-medium mb-2">样本外开始日期</label>
              <input
                type="date"
                value={walkTestStart}
                onChange={e => setWalkTestStart(e.target.value)}
                disabled={isRunning}
                className="w-full rounded-lg border border-input bg-background px-4 py-2.5 text-sm focus:border-primary focus:outline-none focus:ring-2 focus:ring-primary transition-all font-mono"
              />
            </div>
            <div>
              <label className="block text-sm font-medium mb-2">样本外结束日期</label>
              <input
                type="date"
                value={walkTestEnd}
                onChange={e => setWalkTestEnd(e.target.value)}
                disabled={isRunning}
                className="w-full rounded-lg border border-input bg-background px-4 py-2.5 text-sm focus:border-primary focus:outline-none focus:ring-2 focus:ring-primary transition-all font-mono"
              />
            </div>
            <div>
              <label className="block text-sm font-medium mb-2">每次预测/持有</label>
              <input
                type="number"
                value={testSize}
                onChange={e => setTestSize(Math.max(1, parseInt(e.target.value) || 1))}
                disabled={isRunning}
                min={1}
                max={252}
                className="w-full rounded-lg border border-input bg-background px-4 py-2.5 text-sm focus:border-primary focus:outline-none focus:ring-2 focus:ring-primary transition-all font-mono"
              />
              <p className="text-xs text-muted-foreground mt-1">交易日步长，例如 1 为每日滚动，21 约月度</p>
            </div>
            <div>
              <label className="block text-sm font-medium mb-2">Top N 选股数</label>
              <input
                type="number"
                value={topN}
                onChange={e => setTopN(Math.max(1, parseInt(e.target.value) || 50))}
                disabled={isRunning}
                min={1}
                max={500}
                className="w-full rounded-lg border border-input bg-background px-4 py-2.5 text-sm focus:border-primary focus:outline-none focus:ring-2 focus:ring-primary transition-all font-mono"
              />
              <p className="text-xs text-muted-foreground mt-1">每天按预测分数选择前 N 只股票</p>
            </div>
            <div>
              <label className="block text-sm font-medium mb-2">股票池</label>
              <select
                value={universe}
                onChange={e => setUniverse(e.target.value as typeof universe)}
                disabled={isRunning}
                className="w-full rounded-lg border border-input bg-background px-4 py-2.5 text-sm focus:border-primary focus:outline-none focus:ring-2 focus:ring-primary transition-all"
              >
                <option value="hs300">沪深300</option>
                <option value="zz500">中证500</option>
                <option value="zz800">中证800</option>
                <option value="zz1000">中证1000</option>
                <option value="sz50">上证50</option>
              </select>
              <p className="text-xs text-muted-foreground mt-1">选股和回测使用的成分股范围</p>
            </div>
          </div>

          {/* Row 4: Factor Library + Quality Filter */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {/* Factor Library */}
            <div>
              <label className="block text-sm font-medium mb-2">因子库</label>
              <div className="flex gap-2">
                <div className="relative flex-1">
                  <select
                    value={factorLibrary}
                    onChange={e => setFactorLibrary(e.target.value)}
                    disabled={isRunning}
                    className="w-full rounded-lg border border-input bg-background px-4 py-2.5 text-sm focus:border-primary focus:outline-none focus:ring-2 focus:ring-primary transition-all appearance-none pr-10"
                  >
                    {libraries.length === 0 && (
                      <option value="">暂无因子库文件</option>
                    )}
                    {libraries.map(lib => (
                      <option key={lib} value={lib}>{lib}</option>
                    ))}
                  </select>
                  <ChevronDown className="absolute right-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground pointer-events-none" />
                </div>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={loadLibraries}
                  disabled={libsLoading}
                  title="刷新因子库列表"
                  className="px-2.5 self-center"
                >
                  <RefreshCw className={`h-4 w-4 ${libsLoading ? 'animate-spin' : ''}`} />
                </Button>
              </div>
            </div>

            {/* Quality Filter */}
            <div>
              <label className="block text-sm font-medium mb-2">因子质量过滤</label>
              <div className="flex gap-2">
                {QUALITY_OPTIONS.map(opt => (
                  <button
                    key={opt.value}
                    onClick={() => setQualityFilter(opt.value as 'high' | 'medium' | 'all')}
                    disabled={isRunning}
                    className={`flex-1 px-3 py-2 rounded-lg border text-xs font-medium transition-all ${
                      qualityFilter === opt.value
                        ? 'border-primary bg-primary/10 text-primary'
                        : 'border-input bg-background text-muted-foreground hover:border-primary/50'
                    }`}
                  >
                    {opt.label}
                    <span className="block text-[10px] font-normal mt-0.5 opacity-80">{opt.desc}</span>
                  </button>
                ))}
              </div>
            </div>
          </div>

          {/* Visual Diagram */}
          <div className="pt-4 border-t border-border/50">
            <RollingWindowDiagram
              rollingWindow={trainLookback}
              validationSize={validationSize}
              gapSize={gapSize}
              testSize={testSize}
            />
          </div>

          {/* Start / Stop Button */}
          <div className="flex justify-end gap-3 pt-4 border-t border-border/50">
            {isRunning ? (
              <Button variant="destructive" onClick={handleStop}>
                <Square className="h-4 w-4 mr-2" />
                停止回测
              </Button>
            ) : (
              <Button
                variant="primary"
                onClick={handleStart}
                disabled={!factorLibrary || isStarting}
              >
                {isStarting ? (
                  <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                ) : (
                  <Play className="h-4 w-4 mr-2" />
                )}
                开始ML回测
              </Button>
            )}
          </div>
        </CardContent>
      </Card>

      {/* Progress Section */}
      {status !== 'idle' && (
        <Card className={`glass card-hover ${isRunning ? 'border-primary/50' : ''}`}>
          <CardHeader>
            <CardTitle className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                {isRunning ? (
                  <div className="relative h-5 w-5">
                    <div className="absolute inset-0 rounded-full bg-primary/30 animate-ping" />
                    <TrendingUp className="relative h-5 w-5 text-primary" />
                  </div>
                ) : status === 'completed' ? (
                  <TrendingUp className="h-5 w-5 text-green-500" />
                ) : status === 'failed' ? (
                  <AlertCircle className="h-5 w-5 text-red-500" />
                ) : (
                  <Square className="h-5 w-5 text-muted-foreground" />
                )}
                回测进度
              </div>
              <Badge
                variant={
                  status === 'completed' ? 'default' :
                  status === 'running' ? 'default' :
                  status === 'failed' ? 'destructive' : 'outline'
                }
              >
                {status === 'running' ? '运行中' :
                 status === 'completed' ? '已完成' :
                 status === 'failed' ? '失败' :
                 status === 'cancelled' ? '已取消' : status}
              </Badge>
            </CardTitle>
          </CardHeader>
          <CardContent>
            {/* Running animation */}
            {isRunning && (
              <div className="mb-4 rounded-lg bg-primary/5 border border-primary/20 p-3 flex items-center gap-3">
                <Loader2 className="h-5 w-5 text-primary animate-spin flex-shrink-0" />
                <div className="flex-1 min-w-0">
                  <p className="text-sm font-medium text-primary">ML回测正在执行中</p>
                  <p className="text-xs text-muted-foreground truncate">
                    {progress.message || '正在加载因子数据并训练模型...'}
                  </p>
                </div>
              </div>
            )}

            {/* Progress bar */}
            <div className="mb-4">
              <div className="flex justify-between text-sm mb-1">
                <span className="text-muted-foreground">{progress.message || '等待中...'}</span>
                <span className="text-muted-foreground">
                  {status === 'completed' ? '100%' :
                   status === 'failed' ? '失败' :
                   progress.percent > 0 ? `${Math.round(progress.percent)}%` :
                   isRunning ? '运行中...' : ''}
                </span>
              </div>
              <div className="h-2 bg-secondary rounded-full overflow-hidden">
                {isRunning && progress.percent <= 0 ? (
                  <div
                    className="h-full w-1/3 rounded-full bg-gradient-to-r from-transparent via-primary to-transparent"
                    style={{ animation: 'shimmer 1.5s ease-in-out infinite' }}
                  />
                ) : (
                  <div
                    className={`h-full rounded-full transition-all duration-500 ${
                      status === 'completed' ? 'bg-green-500' :
                      status === 'failed' ? 'bg-red-500' :
                      'bg-primary'
                    }`}
                    style={{
                      width: status === 'completed' ? '100%' :
                             status === 'failed' ? '100%' :
                             progress.percent > 0 ? `${progress.percent}%` : '0%',
                    }}
                  />
                )}
              </div>
            </div>

            {/* Task info */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-xs text-muted-foreground">
              <div>任务 ID: <span className="font-mono text-foreground">{taskId || '--'}</span></div>
              <div>模型: <span className="font-mono text-foreground">{model.toUpperCase()}</span></div>
              <div>标签: <span className="font-mono text-foreground">
                {LABEL_OPTIONS.find(o => o.value === labelType)?.label || labelType}
              </span></div>
              <div>试验数: <span className="font-mono text-foreground">{optunaTrialsStr || '--'}</span></div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Results Section */}
      {isFinished && status === 'completed' && Object.keys(metrics).length > 0 && (
        <Card className="glass card-hover animate-fade-in-up">
          <CardHeader>
            <CardTitle className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <TrendingUp className="h-5 w-5 text-green-500" />
                ML回测结果
              </div>
              {metrics.__elapsed_seconds != null && (
                <span className="text-xs text-muted-foreground font-normal">
                  耗时 {Math.round(metrics.__elapsed_seconds)}s
                </span>
              )}
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-6">
            {/* Metrics Table */}
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-border/50 text-muted-foreground">
                    <th className="text-left py-2 px-3 font-medium">IC</th>
                    <th className="text-left py-2 px-3 font-medium">ICIR</th>
                    <th className="text-left py-2 px-3 font-medium">Rank IC</th>
                    <th className="text-left py-2 px-3 font-medium">Rank ICIR</th>
                    <th className="text-left py-2 px-3 font-medium">{metrics.test_days != null && metrics.test_days < 20 ? '累计收益' : '年化收益'}</th>
                    <th className="text-left py-2 px-3 font-medium">最大回撤</th>
                    <th className="text-left py-2 px-3 font-medium">Sharpe</th>
                    <th className="text-left py-2 px-3 font-medium">Calmar</th>
                    <th className="text-left py-2 px-3 font-medium">信息比率</th>
                    <th className="text-left py-2 px-3 font-medium">胜率</th>
                  </tr>
                </thead>
                <tbody>
                  <tr className="font-mono text-foreground">
                    <td className="py-2.5 px-3 font-bold text-primary">{formatNumber(metrics.IC ?? 0, 4)}</td>
                    <td className="py-2.5 px-3 font-bold text-primary">{formatNumber(metrics.ICIR ?? 0, 4)}</td>
                    <td className="py-2.5 px-3">{formatNumber(metrics.RankIC ?? 0, 4)}</td>
                    <td className="py-2.5 px-3">{formatNumber(metrics.RankICIR ?? 0, 4)}</td>
                    <td className={`py-2.5 px-3 ${(metrics.annualized_return ?? 0) >= 0 ? 'text-green-600' : 'text-red-500'}`}>
                      {metrics.annualized_return != null ? `${formatNumber(metrics.annualized_return * 100, 2)}%` : '--'}
                    </td>
                    <td className="py-2.5 px-3 text-red-500">
                      {metrics.max_drawdown != null ? `${formatNumber(metrics.max_drawdown * 100, 2)}%` : '--'}
                    </td>
                    <td className="py-2.5 px-3">{formatNumber(metrics.sharpe_ratio ?? 0, 4)}</td>
                    <td className="py-2.5 px-3">{formatNumber(metrics.calmar_ratio ?? 0, 4)}</td>
                    <td className="py-2.5 px-3">{formatNumber(metrics.information_ratio ?? 0, 4)}</td>
                    <td className="py-2.5 px-3">
                      {metrics.win_rate != null ? `${formatNumber(metrics.win_rate * 100, 2)}%` : '--'}
                    </td>
                  </tr>
                </tbody>
              </table>
            </div>
            {metrics.test_days != null && metrics.test_days <= 5 && (
              <p className="text-xs text-yellow-600 bg-yellow-50 dark:bg-yellow-900/20 rounded-lg px-3 py-2">
                测试期仅 {metrics.test_days} 天，ICIR/Sharpe/Calmar 等需要多日数据才有统计意义，当前值仅供参考。
              </p>
            )}

            {/* Cumulative Curve Placeholder */}
            {metrics.cumulative_curve && metrics.cumulative_curve.length > 0 && (
              <div className="pt-4 border-t border-border/50">
                <h4 className="text-sm font-medium mb-4 flex items-center gap-2">
                  <TrendingUp className="h-4 w-4 text-primary" />
                  超额累计收益曲线
                </h4>
                <div className="h-[300px] w-full bg-secondary/20 rounded-lg flex items-center justify-center">
                  {/* Simple SVG line chart for the equity curve */}
                  <EquityCurveSVG data={metrics.cumulative_curve} />
                </div>
              </div>
            )}

            {metrics.daily_selected_stocks && metrics.daily_selected_stocks.length > 0 && (
              <div className="pt-4 border-t border-border/50">
                <h4 className="text-sm font-medium mb-4 flex items-center gap-2">
                  <TrendingUp className="h-4 w-4 text-primary" />
                  样本外选股结果
                </h4>
                <div className="rounded-lg border border-border/50 overflow-hidden">
                  <table className="w-full text-sm">
                    <thead className="bg-secondary/40 text-muted-foreground">
                      <tr>
                        <th className="text-left px-3 py-2">日期</th>
                        <th className="text-left px-3 py-2">Top {metrics.top_n || topN} 股票</th>
                      </tr>
                    </thead>
                    <tbody>
                      {metrics.daily_selected_stocks.slice(-20).map(row => (
                        <tr key={row.date} className="border-t border-border/40">
                          <td className="px-3 py-2 font-mono text-xs whitespace-nowrap">{row.date}</td>
                          <td className="px-3 py-2 text-xs text-muted-foreground break-all">
                            {row.stocks.join(', ')}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}
          </CardContent>
        </Card>
      )}

      {/* Streaming Log Panel */}
      {(logs.length > 0 || isRunning) && (
        <Card className="glass card-hover">
          <CardHeader>
            <CardTitle className="flex items-center justify-between">
              <span className="flex items-center gap-2">
                <FileText className="h-5 w-5" />
                运行日志
              </span>
              <span className="text-xs text-muted-foreground font-normal">
                {logs.length} 条日志
              </span>
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="bg-black/50 rounded-lg p-4 max-h-[400px] overflow-y-auto font-mono text-xs">
              {logs.length === 0 && isRunning && (
                <div className="text-muted-foreground flex items-center gap-2">
                  <Loader2 className="h-3 w-3 animate-spin" />
                  等待日志输出...
                </div>
              )}
              {logs.map((log) => (
                <div key={log.id} className="py-0.5 flex gap-2">
                  <span className="text-muted-foreground whitespace-nowrap">
                    {new Date(log.timestamp).toLocaleTimeString()}
                  </span>
                  <span
                    className={
                      log.level === 'error' ? 'text-red-400' :
                      log.level === 'warning' ? 'text-yellow-400' :
                      log.level === 'success' ? 'text-green-400' :
                      'text-gray-300'
                    }
                  >
                    {log.message}
                  </span>
                </div>
              ))}
              <div ref={logsEndRef} />
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
};

// ========================== Equity Curve SVG ==========================

/** Lightweight SVG equity curve (no recharts dependency needed for this page) */
const EquityCurveSVG: React.FC<{ data: { date: string; value: number }[] }> = ({ data }) => {
  if (!data || data.length < 2) {
    return <span className="text-muted-foreground text-sm">数据不足，无法绘制曲线</span>;
  }

  const width = 800;
  const height = 260;
  const padding = { top: 20, right: 20, bottom: 30, left: 50 };

  const values = data.map(d => d.value);
  const minVal = Math.min(...values);
  const maxVal = Math.max(...values);
  const valRange = maxVal - minVal || 1;

  const xScale = (i: number) =>
    padding.left + (i / (data.length - 1)) * (width - padding.left - padding.right);
  const yScale = (v: number) =>
    height - padding.bottom - ((v - minVal) / valRange) * (height - padding.top - padding.bottom);

  // Build SVG path
  const pathD = data
    .map((d, i) => `${i === 0 ? 'M' : 'L'} ${xScale(i).toFixed(1)} ${yScale(d.value).toFixed(1)}`)
    .join(' ');

  // Fill area path
  const areaD = `${pathD} L ${xScale(data.length - 1).toFixed(1)} ${(height - padding.bottom).toFixed(1)} L ${padding.left.toFixed(1)} ${(height - padding.bottom).toFixed(1)} Z`;

  // Y-axis labels
  const yTicks = 5;
  const yLabels = Array.from({ length: yTicks + 1 }, (_, i) => {
    const val = minVal + (valRange * i) / yTicks;
    return { y: yScale(val), label: formatPercent(val) };
  });

  // X-axis: show a few date labels
  const xTickCount = Math.min(6, data.length);
  const xLabels = Array.from({ length: xTickCount }, (_, i) => {
    const idx = Math.round((i / (xTickCount - 1)) * (data.length - 1));
    return { x: xScale(idx), label: data[idx].date.slice(0, 7) };
  });

  const finalValue = values[values.length - 1];
  const isPositive = finalValue >= 0;

  return (
    <svg viewBox={`0 0 ${width} ${height}`} className="w-full h-full" preserveAspectRatio="xMidYMid meet">
      {/* Grid lines */}
      {yLabels.map((tick, i) => (
        <g key={i}>
          <line
            x1={padding.left}
            y1={tick.y}
            x2={width - padding.right}
            y2={tick.y}
            stroke="currentColor"
            strokeOpacity={0.1}
            strokeDasharray="4 4"
          />
          <text
            x={padding.left - 8}
            y={tick.y + 4}
            textAnchor="end"
            className="fill-muted-foreground"
            fontSize="10"
          >
            {tick.label}
          </text>
        </g>
      ))}

      {/* X-axis labels */}
      {xLabels.map((tick, i) => (
        <text
          key={i}
          x={tick.x}
          y={height - 8}
          textAnchor="middle"
          className="fill-muted-foreground"
          fontSize="10"
        >
          {tick.label}
        </text>
      ))}

      {/* Zero line */}
      {minVal < 0 && maxVal > 0 && (
        <line
          x1={padding.left}
          y1={yScale(0)}
          x2={width - padding.right}
          y2={yScale(0)}
          stroke="currentColor"
          strokeOpacity={0.3}
          strokeWidth={1}
        />
      )}

      {/* Area fill */}
      <path
        d={areaD}
        fill={isPositive ? '#10B981' : '#EF4444'}
        fillOpacity={0.1}
      />

      {/* Line */}
      <path
        d={pathD}
        fill="none"
        stroke={isPositive ? '#10B981' : '#EF4444'}
        strokeWidth={2}
        strokeLinecap="round"
        strokeLinejoin="round"
      />

      {/* End point dot */}
      <circle
        cx={xScale(data.length - 1)}
        cy={yScale(finalValue)}
        r={4}
        fill={isPositive ? '#10B981' : '#EF4444'}
      />
    </svg>
  );
};
