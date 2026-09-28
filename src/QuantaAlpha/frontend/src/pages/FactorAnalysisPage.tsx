import React, { useState, useEffect, useCallback, useRef } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/Card';
import { Button } from '@/components/ui/Button';
import { Badge } from '@/components/ui/Badge';
import {
  FlaskConical,
  Image,
  FolderPlus,
  Check,
  Loader2,
  RefreshCw,
  Play,
  Square,
  FileText,
  AlertCircle,
  ChevronDown,
  CheckSquare,
  Square as SquareIcon,
  X,
  Plus,
  Pin,
  PinOff,
  Eye,
  Code,
} from 'lucide-react';
import { getFactors, listFactorLibraries, connectMiningWs } from '@/services/api';
import { formatNumber } from '@/utils';
import { getDirectionLabel } from '@/utils/miningDirections';
import type { Factor, LogEntry, WsMessage } from '@/types';

// ========================== Types ==========================

interface AnalysisTask {
  taskId: string;
  status: 'idle' | 'running' | 'completed' | 'failed';
  factorNames: string[];
  completedImages: string[];
}

// ========================== Component ==========================

export const FactorAnalysisPage: React.FC = () => {
  // -- Library & Factor State --
  const [libraries, setLibraries] = useState<string[]>([]);
  const [selectedLibrary, setSelectedLibrary] = useState<string>(
    localStorage.getItem('quantaalpha_active_library') || ''
  );
  const [filteredFactors, setFilteredFactors] = useState<Factor[]>([]);
  const [qualityFilter, setQualityFilter] = useState<'high' | 'medium'>('high');
  const [selectedFactorIds, setSelectedFactorIds] = useState<Set<string>>(new Set());
  const [libsLoading, setLibsLoading] = useState(false);
  const [factorsLoading, setFactorsLoading] = useState(false);

  // -- Analysis State --
  const [analysisTask, setAnalysisTask] = useState<AnalysisTask | null>(null);
  const [isStarting, setIsStarting] = useState(false);
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const [resultImages, setResultImages] = useState<string[]>([]);
  const logsEndRef = useRef<HTMLDivElement>(null);
  const wsRef = useRef<WebSocket | null>(null);

  // -- Detail Panel State --
  const [viewingFactor, setViewingFactor] = useState<Factor | null>(null);
  const [detailPinned, setDetailPinned] = useState(true);

  // -- Add to Library State --
  const [showAddDialog, setShowAddDialog] = useState(false);
  const [targetLibrary, setTargetLibrary] = useState<string>('');
  const [newLibraryName, setNewLibraryName] = useState('');
  const [createNewLib, setCreateNewLib] = useState(false);
  const [addingToLibrary, setAddingToLibrary] = useState(false);
  const [addSuccess, setAddSuccess] = useState(false);

  // ========================== Load Libraries ==========================

  const loadLibraries = useCallback(async () => {
    setLibsLoading(true);
    try {
      const resp = await listFactorLibraries();
      if (resp.success && resp.data) {
        const libs = resp.data.libraries || [];
        setLibraries(libs);
        if (libs.length > 0 && (!selectedLibrary || !libs.includes(selectedLibrary))) {
          setSelectedLibrary(libs[0]);
        }
      }
    } catch {
      // backend might not be up
    }
    setLibsLoading(false);
  }, [selectedLibrary]);

  // Init on mount
  const initDone = useRef(false);
  useEffect(() => {
    if (initDone.current) return;
    initDone.current = true;
    loadLibraries();
  }, [loadLibraries]);

  // ========================== Load Factors ==========================

  const loadFactors = useCallback(async () => {
    if (!selectedLibrary) return;
    setFactorsLoading(true);
    try {
      const resp = await getFactors({
        library: selectedLibrary,
        quality: qualityFilter,
        limit: 500,
      });
      if (resp.success && resp.data) {
        const apiFactors: Factor[] = resp.data.factors.map((f: any) => ({
          factorId: f.factorId || '',
          factorName: f.factorName || 'Unknown',
          factorExpression: f.factorExpression || '',
          factorDescription: f.factorDescription || '',
          quality: f.quality || 'low',
          ic: f.ic || 0,
          icir: f.icir || 0,
          rankIc: f.rankIc || 0,
          rankIcir: f.rankIcir || 0,
          round: f.round || 0,
          direction: String(f.direction ?? ''),
          createdAt: f.createdAt || new Date().toISOString(),
        }));
        setFilteredFactors(apiFactors);
      }
    } catch {
      setFilteredFactors([]);
    }
    setFactorsLoading(false);
  }, [selectedLibrary, qualityFilter]);

  useEffect(() => {
    loadFactors();
    setSelectedFactorIds(new Set());
  }, [loadFactors]);

  // ========================== Factor Selection ==========================

  const toggleFactorSelection = (factorId: string) => {
    setSelectedFactorIds((prev) => {
      const next = new Set(prev);
      if (next.has(factorId)) {
        next.delete(factorId);
      } else {
        next.add(factorId);
      }
      return next;
    });
  };

  const selectAll = () => {
    setSelectedFactorIds(new Set(filteredFactors.map((f) => f.factorId)));
  };

  const deselectAll = () => {
    setSelectedFactorIds(new Set());
  };

  const handleFactorClick = (factor: Factor) => {
    if (isRunning) return;
    toggleFactorSelection(factor.factorId);
    setViewingFactor(factor);
  };

  // ========================== Analysis ==========================

  const handleStartAnalysis = async () => {
    if (selectedFactorIds.size === 0) return;

    setIsStarting(true);
    setLogs([]);
    setResultImages([]);

    const selectedFactors = filteredFactors.filter((f) => selectedFactorIds.has(f.factorId));
    const factorNames = selectedFactors.map((f) => f.factorName);

    try {
      const res = await fetch('/api/v1/analysis/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          factorIds: Array.from(selectedFactorIds),
          library: selectedLibrary,
        }),
      });
      const data = await res.json();

      if (data.success && data.data?.taskId) {
        const taskId = data.data.taskId;
        setAnalysisTask({
          taskId,
          status: 'running',
          factorNames,
          completedImages: [],
        });

        // Connect WebSocket for streaming logs
        connectAnalysisWs(taskId, factorNames);
      } else {
        addLog('error', data.error || '启动分析失败');
        setAnalysisTask(null);
      }
    } catch (err: any) {
      addLog('error', `请求失败: ${err.message}`);
      setAnalysisTask(null);
    }

    setIsStarting(false);
  };

  const handleStopAnalysis = () => {
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
    if (analysisTask) {
      setAnalysisTask({ ...analysisTask, status: 'failed' });
      addLog('warning', '分析已手动停止');
    }
  };

  const connectAnalysisWs = (taskId: string, factorNames: string[]) => {
    // Cleanup previous ws
    if (wsRef.current) {
      wsRef.current.close();
    }

    const ws = connectMiningWs(
      taskId,
      (msg: WsMessage) => {
        switch (msg.type) {
          case 'log':
            addLog(
              msg.data?.level || 'info',
              msg.data?.message || JSON.stringify(msg.data)
            );
            break;
          case 'progress':
            if (msg.data?.message) {
              addLog('info', msg.data.message);
            }
            // Check if a factor image is ready
            if (msg.data?.completedFactor) {
              setResultImages((prev) => [...prev, msg.data.completedFactor]);
              setAnalysisTask((prev) =>
                prev
                  ? {
                      ...prev,
                      completedImages: [...prev.completedImages, msg.data.completedFactor],
                    }
                  : prev
              );
            }
            break;
          case 'result':
            // Analysis complete
            setAnalysisTask((prev) =>
              prev ? { ...prev, status: 'completed' } : prev
            );
            addLog('success', '所有因子分析完成');
            // Set all factor images as available
            setResultImages(factorNames);
            break;
          case 'error':
            addLog('error', msg.data?.message || '分析出错');
            setAnalysisTask((prev) =>
              prev ? { ...prev, status: 'failed' } : prev
            );
            break;
          default:
            break;
        }
      },
      () => {
        // onClose - mark completed if still running
        setAnalysisTask((prev) => {
          if (prev && prev.status === 'running') {
            // If ws closes and we have images, treat as completed
            if (resultImages.length > 0 || prev.completedImages.length > 0) {
              return { ...prev, status: 'completed' };
            }
            return { ...prev, status: 'completed' };
          }
          return prev;
        });
      },
      () => {
        addLog('error', 'WebSocket 连接失败');
      }
    );

    wsRef.current = ws;
  };

  // Cleanup WebSocket on unmount
  useEffect(() => {
    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, []);

  // ========================== Add to Library ==========================

  const handleAddToLibrary = async () => {
    const target = createNewLib ? newLibraryName.trim() : targetLibrary;
    if (!target) return;

    setAddingToLibrary(true);
    setAddSuccess(false);

    try {
      const res = await fetch('/api/v1/factors/add-to-library', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          factorIds: Array.from(selectedFactorIds),
          targetLibrary: target,
          createNew: createNewLib,
        }),
      });
      const data = await res.json();
      if (data.success) {
        setAddSuccess(true);
        addLog('success', `已将 ${selectedFactorIds.size} 个因子添加到 ${target}`);
        setTimeout(() => {
          setShowAddDialog(false);
          setAddSuccess(false);
        }, 1500);
        // Refresh library list
        loadLibraries();
      } else {
        addLog('error', data.error || '添加失败');
      }
    } catch (err: any) {
      addLog('error', `添加失败: ${err.message}`);
    }

    setAddingToLibrary(false);
  };

  // ========================== Helpers ==========================

  const addLog = (level: LogEntry['level'], message: string) => {
    setLogs((prev) => [
      ...prev,
      {
        id: `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
        timestamp: new Date().toISOString(),
        level,
        message,
      },
    ]);
  };

  // Auto-scroll logs
  useEffect(() => {
    logsEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [logs]);

  const isRunning = analysisTask?.status === 'running';
  const isCompleted = analysisTask?.status === 'completed';

  // ========================== Render ==========================

  return (
    <div className="space-y-6 animate-fade-in-up">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold flex items-center gap-3">
          <FlaskConical className="h-8 w-8 text-primary" />
          因子分析
        </h1>
        <p className="text-muted-foreground mt-1">
          对中/高质量因子执行 TQ 回测分析，查看因子诊断图
        </p>
      </div>

      {/* Configuration Section */}
      <Card className="glass card-hover">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <FileText className="h-5 w-5" />
            分析配置
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {/* Library Selector */}
            <div>
              <label className="block text-sm font-medium mb-2">因子库</label>
              <div className="flex gap-2">
                <div className="relative flex-1">
                  <select
                    value={selectedLibrary}
                    onChange={(e) => setSelectedLibrary(e.target.value)}
                    disabled={isRunning}
                    className="w-full rounded-lg border border-input bg-background px-4 py-2.5 text-sm focus:border-primary focus:outline-none focus:ring-2 focus:ring-primary transition-all appearance-none pr-10"
                  >
                    {libraries.length === 0 && (
                      <option value="">暂无因子库文件</option>
                    )}
                    {libraries.map((lib) => (
                      <option key={lib} value={lib}>
                        {lib}
                      </option>
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
              <label className="block text-sm font-medium mb-2">质量筛选</label>
              <div className="flex gap-3">
                <button
                  onClick={() => setQualityFilter('high')}
                  disabled={isRunning}
                  className={`flex-1 px-4 py-2.5 rounded-lg border text-sm font-medium transition-all ${
                    qualityFilter === 'high'
                      ? 'border-green-500 bg-green-500/10 text-green-500'
                      : 'border-input bg-background text-muted-foreground hover:border-green-500/50'
                  }`}
                >
                  高质量
                  <span className="block text-xs font-normal mt-0.5">
                    两项检验均通过
                  </span>
                </button>
                <button
                  onClick={() => setQualityFilter('medium')}
                  disabled={isRunning}
                  className={`flex-1 px-4 py-2.5 rounded-lg border text-sm font-medium transition-all ${
                    qualityFilter === 'medium'
                      ? 'border-yellow-500 bg-yellow-500/10 text-yellow-500'
                      : 'border-input bg-background text-muted-foreground hover:border-yellow-500/50'
                  }`}
                >
                  中质量
                  <span className="block text-xs font-normal mt-0.5">
                    一项检验通过
                  </span>
                </button>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Factor List */}
      <Card className="glass card-hover">
        <CardHeader>
          <CardTitle className="flex items-center justify-between">
            <span className="flex items-center gap-2">
              <FileText className="h-5 w-5" />
              因子列表
              {filteredFactors.length > 0 && (
                <Badge variant="outline" className="ml-2">
                  {filteredFactors.length} 个因子
                </Badge>
              )}
            </span>
            <div className="flex items-center gap-2">
              <Button
                variant="ghost"
                size="sm"
                onClick={selectAll}
                disabled={isRunning || filteredFactors.length === 0}
                className="text-xs"
              >
                全选
              </Button>
              <Button
                variant="ghost"
                size="sm"
                onClick={deselectAll}
                disabled={isRunning || selectedFactorIds.size === 0}
                className="text-xs"
              >
                取消全选
              </Button>
              <Badge variant="default">
                已选 {selectedFactorIds.size}
              </Badge>
            </div>
          </CardTitle>
        </CardHeader>
        <CardContent>
          {factorsLoading ? (
            <div className="flex items-center justify-center py-12 text-muted-foreground">
              <Loader2 className="h-5 w-5 animate-spin mr-2" />
              加载因子中...
            </div>
          ) : filteredFactors.length === 0 ? (
            <div className="flex flex-col items-center justify-center py-12 text-muted-foreground">
              <AlertCircle className="h-8 w-8 mb-3 opacity-50" />
              <p>当前筛选条件下没有因子</p>
              <p className="text-xs mt-1">尝试切换质量筛选或选择其他因子库</p>
            </div>
          ) : (
            <div className="flex gap-4">
              {/* Factor List (left side, shrinks when detail panel is open) */}
              <div className={`max-h-[500px] overflow-y-auto space-y-1.5 pr-1 ${viewingFactor && detailPinned ? 'w-1/2 flex-shrink-0' : 'flex-1'}`}>
                {filteredFactors.map((factor) => {
                  const isSelected = selectedFactorIds.has(factor.factorId);
                  const isViewing = viewingFactor?.factorId === factor.factorId;
                  return (
                    <div
                      key={factor.factorId}
                      onClick={() => handleFactorClick(factor)}
                      className={`flex items-center gap-3 p-3 rounded-lg border cursor-pointer transition-all ${
                        isViewing
                          ? 'border-blue-500/60 bg-blue-500/10 ring-1 ring-blue-500/30'
                          : isSelected
                          ? 'border-primary/50 bg-primary/5'
                          : 'border-border/50 hover:border-primary/30 hover:bg-secondary/30'
                      } ${isRunning ? 'opacity-60 cursor-not-allowed' : ''}`}
                    >
                      {/* Checkbox */}
                      <div className="flex-shrink-0" onClick={(e) => { e.stopPropagation(); if (!isRunning) toggleFactorSelection(factor.factorId); }}>
                        {isSelected ? (
                          <CheckSquare className="h-5 w-5 text-primary" />
                        ) : (
                          <SquareIcon className="h-5 w-5 text-muted-foreground/50" />
                        )}
                      </div>

                      {/* Factor Info */}
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-2">
                          <span className="font-medium text-sm truncate">
                            {factor.factorName}
                          </span>
                          <Badge
                            variant={factor.quality === 'high' ? 'default' : 'outline'}
                            className={
                              factor.quality === 'high'
                                ? 'bg-green-500/20 text-green-500 border-green-500/30'
                                : 'bg-yellow-500/20 text-yellow-500 border-yellow-500/30'
                            }
                          >
                            {factor.quality === 'high' ? '高' : '中'}
                          </Badge>
                          {factor.direction && (
                            <Badge variant="outline" className="bg-blue-500/10 text-blue-400 border-blue-500/30 text-[10px]">
                              {getDirectionLabel(factor.direction)}
                            </Badge>
                          )}
                        </div>
                        <p className="text-xs text-muted-foreground truncate mt-0.5">
                          {factor.factorExpression || factor.factorDescription || '--'}
                        </p>
                      </div>

                      {/* Metrics */}
                      <div className="flex-shrink-0 grid grid-cols-2 gap-x-4 gap-y-0.5 text-xs">
                        <span className="text-muted-foreground">
                          IC: <span className="text-foreground font-mono">{formatNumber(factor.ic, 4)}</span>
                        </span>
                        <span className="text-muted-foreground">
                          ICIR: <span className="text-foreground font-mono">{formatNumber(factor.icir, 4)}</span>
                        </span>
                        <span className="text-muted-foreground">
                          RankIC: <span className="text-foreground font-mono">{formatNumber(factor.rankIc, 4)}</span>
                        </span>
                        <span className="text-muted-foreground">
                          RankICIR: <span className="text-foreground font-mono">{formatNumber(factor.rankIcir, 4)}</span>
                        </span>
                      </div>
                    </div>
                  );
                })}
              </div>

              {/* Pinned Detail Panel (right side) */}
              {viewingFactor && detailPinned && (
                <div className="w-1/2 flex-shrink-0 max-h-[500px] overflow-y-auto rounded-lg border border-blue-500/30 bg-secondary/20 p-4 space-y-4 animate-fade-in-up">
                  {/* Header */}
                  <div className="flex items-start justify-between">
                    <div className="flex-1 min-w-0">
                      <h3 className="font-semibold text-base truncate">{viewingFactor.factorName}</h3>
                      <div className="flex items-center gap-2 mt-1.5">
                        <Badge
                          className={
                            viewingFactor.quality === 'high'
                              ? 'bg-green-500/20 text-green-500 border-green-500/30'
                              : 'bg-yellow-500/20 text-yellow-500 border-yellow-500/30'
                          }
                        >
                          {viewingFactor.quality === 'high' ? '高质量' : '中质量'}
                        </Badge>
                        {viewingFactor.direction && (
                          <Badge variant="outline" className="bg-blue-500/10 text-blue-400 border-blue-500/30">
                            {getDirectionLabel(viewingFactor.direction)}
                          </Badge>
                        )}
                      </div>
                    </div>
                    <div className="flex items-center gap-1">
                      <button
                        onClick={() => setDetailPinned(false)}
                        className="p-1.5 rounded-md text-muted-foreground hover:text-foreground hover:bg-secondary/50 transition-colors"
                        title="取消固定"
                      >
                        <PinOff className="h-4 w-4" />
                      </button>
                      <button
                        onClick={() => setViewingFactor(null)}
                        className="p-1.5 rounded-md text-muted-foreground hover:text-foreground hover:bg-secondary/50 transition-colors"
                        title="关闭"
                      >
                        <X className="h-4 w-4" />
                      </button>
                    </div>
                  </div>

                  {/* Expression */}
                  <div className="rounded-lg bg-white p-3 border border-border/50">
                    <div className="flex items-center gap-2 mb-2">
                      <Code className="h-3.5 w-3.5 text-gray-500" />
                      <span className="text-xs text-gray-500 font-medium">表达式</span>
                    </div>
                    <code className="text-xs font-mono text-gray-900 break-all leading-relaxed">
                      {viewingFactor.factorExpression || '--'}
                    </code>
                  </div>

                  {/* Description */}
                  {viewingFactor.factorDescription && (
                    <div>
                      <span className="text-xs text-muted-foreground font-medium">描述</span>
                      <p className="text-sm mt-1">{viewingFactor.factorDescription}</p>
                    </div>
                  )}

                  {/* Metrics Grid */}
                  <div>
                    <span className="text-xs text-muted-foreground font-medium">回测指标</span>
                    <div className="grid grid-cols-2 gap-3 mt-2">
                      {[
                        { label: 'IC', value: viewingFactor.ic },
                        { label: 'ICIR', value: viewingFactor.icir },
                        { label: 'RankIC', value: viewingFactor.rankIc },
                        { label: 'RankICIR', value: viewingFactor.rankIcir },
                      ].map((m) => (
                        <div key={m.label} className="rounded-lg bg-secondary/40 p-2.5 text-center">
                          <div className="text-[10px] text-muted-foreground uppercase tracking-wider">{m.label}</div>
                          <div className="text-sm font-mono font-semibold mt-0.5">{formatNumber(m.value, 4)}</div>
                        </div>
                      ))}
                    </div>
                  </div>

                  {/* Metadata */}
                  <div className="text-xs text-muted-foreground space-y-1 pt-2 border-t border-border/30">
                    <div>ID: <span className="font-mono text-foreground">{viewingFactor.factorId}</span></div>
                    {viewingFactor.round > 0 && <div>轮次: <span className="text-foreground">{viewingFactor.round}</span></div>}
                    {viewingFactor.createdAt && <div>创建: <span className="text-foreground">{new Date(viewingFactor.createdAt).toLocaleString()}</span></div>}
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Unpinned detail hint */}
          {viewingFactor && !detailPinned && (
            <div className="flex items-center gap-2 mt-3 p-2.5 rounded-lg border border-blue-500/20 bg-blue-500/5 text-sm">
              <Eye className="h-4 w-4 text-blue-400 flex-shrink-0" />
              <span className="text-muted-foreground truncate">
                查看: <span className="text-foreground font-medium">{viewingFactor.factorName}</span>
              </span>
              <button
                onClick={() => setDetailPinned(true)}
                className="ml-auto flex items-center gap-1 px-2 py-1 rounded-md text-xs text-blue-400 hover:bg-blue-500/10 transition-colors"
              >
                <Pin className="h-3.5 w-3.5" />
                固定详情
              </button>
              <button
                onClick={() => setViewingFactor(null)}
                className="p-1 rounded-md text-muted-foreground hover:text-foreground transition-colors"
              >
                <X className="h-3.5 w-3.5" />
              </button>
            </div>
          )}

          {/* Action Buttons */}
          <div className="flex justify-end gap-3 pt-4 border-t border-border/50 mt-4">
            {isRunning ? (
              <Button variant="outline" onClick={handleStopAnalysis}>
                <Square className="h-4 w-4 mr-2" />
                停止分析
              </Button>
            ) : (
              <>
                {isCompleted && resultImages.length > 0 && (
                  <Button
                    variant="outline"
                    onClick={() => setShowAddDialog(true)}
                  >
                    <FolderPlus className="h-4 w-4 mr-2" />
                    添加到因子库
                  </Button>
                )}
                <Button
                  variant="primary"
                  onClick={handleStartAnalysis}
                  disabled={selectedFactorIds.size === 0 || isStarting}
                >
                  {isStarting ? (
                    <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                  ) : (
                    <Play className="h-4 w-4 mr-2" />
                  )}
                  开始分析 ({selectedFactorIds.size})
                </Button>
              </>
            )}
          </div>
        </CardContent>
      </Card>

      {/* Analysis Progress */}
      {analysisTask && (
        <Card className={`glass card-hover ${isRunning ? 'border-primary/50' : ''}`}>
          <CardHeader>
            <CardTitle className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                {isRunning ? (
                  <div className="relative h-5 w-5">
                    <div className="absolute inset-0 rounded-full bg-primary/30 animate-ping" />
                    <FlaskConical className="relative h-5 w-5 text-primary" />
                  </div>
                ) : isCompleted ? (
                  <Check className="h-5 w-5 text-green-500" />
                ) : (
                  <AlertCircle className="h-5 w-5 text-red-500" />
                )}
                分析进度
              </div>
              <Badge
                variant={
                  isCompleted ? 'default' :
                  isRunning ? 'default' :
                  'destructive'
                }
              >
                {isRunning ? '运行中' :
                 isCompleted ? '已完成' :
                 '失败'}
              </Badge>
            </CardTitle>
          </CardHeader>
          <CardContent>
            {isRunning && (
              <div className="mb-4 rounded-lg bg-primary/5 border border-primary/20 p-3 flex items-center gap-3">
                <Loader2 className="h-5 w-5 text-primary animate-spin flex-shrink-0" />
                <div className="flex-1 min-w-0">
                  <p className="text-sm font-medium text-primary">
                    TQ 因子分析正在执行
                  </p>
                  <p className="text-xs text-muted-foreground">
                    正在处理 {analysisTask.factorNames.length} 个因子的诊断分析...
                  </p>
                </div>
              </div>
            )}

            {/* Progress info */}
            <div className="text-xs text-muted-foreground mb-2">
              任务 ID: <span className="font-mono text-foreground">{analysisTask.taskId}</span>
              {' | '}
              因子数量: {analysisTask.factorNames.length}
              {analysisTask.completedImages.length > 0 && (
                <>
                  {' | '}
                  已完成: {analysisTask.completedImages.length}/{analysisTask.factorNames.length}
                </>
              )}
            </div>

            {/* Progress bar */}
            <div className="h-2 bg-secondary rounded-full overflow-hidden">
              {isRunning && analysisTask.completedImages.length === 0 ? (
                <div
                  className="h-full w-1/3 rounded-full bg-gradient-to-r from-transparent via-primary to-transparent"
                  style={{ animation: 'shimmer 1.5s ease-in-out infinite' }}
                />
              ) : (
                <div
                  className={`h-full rounded-full transition-all duration-500 ${
                    isCompleted ? 'bg-green-500' :
                    analysisTask.status === 'failed' ? 'bg-red-500' :
                    'bg-primary'
                  }`}
                  style={{
                    width: isCompleted
                      ? '100%'
                      : analysisTask.factorNames.length > 0
                      ? `${(analysisTask.completedImages.length / analysisTask.factorNames.length) * 100}%`
                      : '0%',
                  }}
                />
              )}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Results Section - PNG Images */}
      {resultImages.length > 0 && (
        <Card className="glass card-hover animate-fade-in-up">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Image className="h-5 w-5 text-green-500" />
              诊断图结果
              <Badge variant="outline" className="ml-2">
                {resultImages.length} 张
              </Badge>
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {resultImages.map((factorName) => (
                <div
                  key={factorName}
                  className="rounded-lg border border-border/50 overflow-hidden bg-secondary/20"
                >
                  {/* Factor name header */}
                  <div className="px-4 py-2 bg-secondary/40 border-b border-border/50">
                    <h4 className="text-sm font-medium truncate" title={factorName}>
                      {factorName}
                    </h4>
                  </div>
                  {/* Image */}
                  <div className="p-2">
                    <img
                      src={`/api/v1/analysis/images/${encodeURIComponent(factorName)}`}
                      alt={`${factorName} 诊断图`}
                      className="w-full h-auto rounded border border-border/30"
                      loading="lazy"
                      onError={(e) => {
                        const target = e.target as HTMLImageElement;
                        target.style.display = 'none';
                        const parent = target.parentElement;
                        if (parent && !parent.querySelector('.error-placeholder')) {
                          const placeholder = document.createElement('div');
                          placeholder.className = 'error-placeholder flex items-center justify-center h-48 text-muted-foreground text-sm';
                          placeholder.textContent = '图片加载失败';
                          parent.appendChild(placeholder);
                        }
                      }}
                    />
                  </div>
                </div>
              ))}
            </div>
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
                分析日志
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
                      log.level === 'error'
                        ? 'text-red-400'
                        : log.level === 'warning'
                        ? 'text-yellow-400'
                        : log.level === 'success'
                        ? 'text-green-400'
                        : 'text-gray-300'
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

      {/* Add to Library Dialog (Overlay) */}
      {showAddDialog && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm">
          <Card className="glass w-full max-w-md mx-4 shadow-2xl">
            <CardHeader>
              <CardTitle className="flex items-center justify-between">
                <span className="flex items-center gap-2">
                  <FolderPlus className="h-5 w-5 text-primary" />
                  添加到因子库
                </span>
                <button
                  onClick={() => setShowAddDialog(false)}
                  className="text-muted-foreground hover:text-foreground transition-colors"
                >
                  <X className="h-5 w-5" />
                </button>
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <p className="text-sm text-muted-foreground">
                将已选的 {selectedFactorIds.size} 个因子添加到目标因子库
              </p>

              {/* Toggle: existing or new */}
              <div className="flex gap-2">
                <button
                  onClick={() => setCreateNewLib(false)}
                  className={`flex-1 px-3 py-2 rounded-lg border text-sm transition-all ${
                    !createNewLib
                      ? 'border-primary bg-primary/10 text-primary'
                      : 'border-input text-muted-foreground hover:border-primary/50'
                  }`}
                >
                  选择已有库
                </button>
                <button
                  onClick={() => setCreateNewLib(true)}
                  className={`flex-1 px-3 py-2 rounded-lg border text-sm transition-all ${
                    createNewLib
                      ? 'border-primary bg-primary/10 text-primary'
                      : 'border-input text-muted-foreground hover:border-primary/50'
                  }`}
                >
                  <Plus className="h-3.5 w-3.5 inline mr-1" />
                  创建新库
                </button>
              </div>

              {/* Library selection or name input */}
              {createNewLib ? (
                <div>
                  <label className="block text-sm font-medium mb-1.5">新因子库名称</label>
                  <input
                    type="text"
                    value={newLibraryName}
                    onChange={(e) => setNewLibraryName(e.target.value)}
                    placeholder="例如: my_alpha_factors.json"
                    className="w-full rounded-lg border border-input bg-background px-4 py-2.5 text-sm focus:border-primary focus:outline-none focus:ring-2 focus:ring-primary transition-all"
                  />
                </div>
              ) : (
                <div>
                  <label className="block text-sm font-medium mb-1.5">目标因子库</label>
                  <div className="relative">
                    <select
                      value={targetLibrary}
                      onChange={(e) => setTargetLibrary(e.target.value)}
                      className="w-full rounded-lg border border-input bg-background px-4 py-2.5 text-sm focus:border-primary focus:outline-none focus:ring-2 focus:ring-primary transition-all appearance-none pr-10"
                    >
                      <option value="">-- 请选择 --</option>
                      {libraries.map((lib) => (
                        <option key={lib} value={lib}>
                          {lib}
                        </option>
                      ))}
                    </select>
                    <ChevronDown className="absolute right-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground pointer-events-none" />
                  </div>
                </div>
              )}

              {/* Actions */}
              <div className="flex justify-end gap-3 pt-2">
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => setShowAddDialog(false)}
                >
                  取消
                </Button>
                <Button
                  variant="primary"
                  size="sm"
                  onClick={handleAddToLibrary}
                  disabled={
                    addingToLibrary ||
                    (createNewLib ? !newLibraryName.trim() : !targetLibrary)
                  }
                >
                  {addingToLibrary ? (
                    <Loader2 className="h-4 w-4 mr-1.5 animate-spin" />
                  ) : addSuccess ? (
                    <Check className="h-4 w-4 mr-1.5 text-green-500" />
                  ) : (
                    <FolderPlus className="h-4 w-4 mr-1.5" />
                  )}
                  {addSuccess ? '已添加' : '确认添加'}
                </Button>
              </div>
            </CardContent>
          </Card>
        </div>
      )}
    </div>
  );
};
