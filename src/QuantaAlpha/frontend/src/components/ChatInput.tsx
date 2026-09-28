import React, { useState, useEffect } from 'react';
import { Send, Square, Settings2, PenLine, ChevronDown, ChevronUp, Check } from 'lucide-react';
import { TaskConfig } from '@/types';
import { MINING_DIRECTIONS, getDefaultMiningDirection } from '@/utils/miningDirections';
import { getInitialDirections, DirectionPortfolio } from '@/services/api';

interface ChatInputProps {
  onSubmit: (config: TaskConfig) => void;
  onStop?: () => void;
  isRunning?: boolean;
}

export const ChatInput: React.FC<ChatInputProps> = ({ onSubmit, onStop, isRunning = false }) => {
  const [direction, setDirection] = useState(getDefaultMiningDirection());
  const [isCustom, setIsCustom] = useState(false);
  const [customDirection, setCustomDirection] = useState('');
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [numDirectionsStr, setNumDirectionsStr] = useState('');
  const [maxRoundsStr, setMaxRoundsStr] = useState('');
  const [maxLoopsStr, setMaxLoopsStr] = useState('');
  const [factorsPerHypothesisStr, setFactorsPerHypothesisStr] = useState('');
  const [librarySuffix, setLibrarySuffix] = useState('');

  // Initial direction portfolios
  const [portfolios, setPortfolios] = useState<DirectionPortfolio[]>([]);
  const [selectedPortfolioIds, setSelectedPortfolioIds] = useState<Set<number>>(new Set());
  const [expandedPortfolioId, setExpandedPortfolioId] = useState<number | null>(null);
  const [portfoliosLoading, setPortfoliosLoading] = useState(false);

  const getDefaults = () => {
    try {
      const raw = localStorage.getItem('quantaalpha_config');
      if (raw) {
        const c = JSON.parse(raw);
        return {
          numDirections: c.defaultNumDirections || 2,
          maxRounds: c.defaultMaxRounds || 3,
          maxLoops: c.defaultMaxLoops || 2,
          factorsPerHypothesis: c.defaultFactorsPerHypothesis || 3,
        };
      }
    } catch {}
    return { numDirections: 2, maxRounds: 3, maxLoops: 2, factorsPerHypothesis: 3 };
  };
  const defaults = getDefaults();

  // Load portfolios when domain changes
  useEffect(() => {
    if (isCustom) return;
    setPortfoliosLoading(true);
    getInitialDirections(direction)
      .then((resp) => {
        if (resp.success && resp.data) {
          setPortfolios(resp.data.portfolios || []);
        } else {
          setPortfolios([]);
        }
      })
      .catch(() => setPortfolios([]))
      .finally(() => setPortfoliosLoading(false));
    setSelectedPortfolioIds(new Set());
    setExpandedPortfolioId(null);
  }, [direction, isCustom]);

  const togglePortfolio = (id: number) => {
    setSelectedPortfolioIds((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  };

  const toggleExpandPortfolio = (id: number) => {
    setExpandedPortfolioId((prev) => (prev === id ? null : id));
  };

  const handleSubmit = () => {
    if (isRunning) return;
    if (isCustom && !customDirection.trim()) return;
    if (!isCustom && portfolios.length > 0 && selectedPortfolioIds.size === 0) return;

    let userInput: string;
    if (isCustom) {
      userInput = customDirection.trim();
    } else if (selectedPortfolioIds.size > 0) {
      const selected = portfolios.filter((p) => selectedPortfolioIds.has(p.portfolio_id));
      userInput = selected.map((p) => p.description).join('\n---\n');
    } else {
      userInput = direction;
    }

    const numDir = numDirectionsStr ? parseInt(numDirectionsStr) : undefined;
    const rounds = maxRoundsStr ? parseInt(maxRoundsStr) : undefined;
    const loops = maxLoopsStr ? parseInt(maxLoopsStr) : undefined;
    const fph = factorsPerHypothesisStr ? parseInt(factorsPerHypothesisStr) : undefined;
    onSubmit({
      userInput,
      useCustomMiningDirection: true,
      numDirections: numDir && numDir > 0 ? numDir : undefined,
      maxRounds: rounds && rounds > 0 ? rounds : undefined,
      maxLoops: loops && loops > 0 ? loops : undefined,
      factorsPerHypothesis: fph && fph > 0 ? fph : undefined,
      librarySuffix: librarySuffix.trim() || undefined,
    } as TaskConfig);
  };

  const handleSelectPreset = (id: string) => {
    setDirection(id);
    setIsCustom(false);
  };

  const selectAll = () => {
    setSelectedPortfolioIds(new Set(portfolios.map((p) => p.portfolio_id)));
  };
  const deselectAll = () => {
    setSelectedPortfolioIds(new Set());
  };

  // Collapsed running state: minimal bar with status + stop button
  if (isRunning) {
    return (
      <div className="fixed bottom-0 left-0 right-0 z-50 pb-4">
        <div className="container mx-auto px-6">
          <div className="glass-strong rounded-xl px-4 py-3 flex items-center justify-between">
            <p className="text-sm text-muted-foreground">
              实验运行中...可切换页面，任务不会中断
            </p>
            {onStop && (
              <button
                onClick={onStop}
                className="px-4 py-2 rounded-lg bg-red-500 text-white hover:bg-red-600 transition-all hover:scale-105 active:scale-95 flex items-center gap-2"
                title="中断实验"
              >
                <Square className="h-4 w-4" />
                停止
              </button>
            )}
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="fixed bottom-0 left-0 right-0 z-50 pb-6">
      <div className="container mx-auto px-6">
        <div className="gradient-border">
          <div className="gradient-border-content">
            <div className="glass-strong rounded-xl p-4">
              {/* Direction Selector */}
              <div className="flex items-center gap-3 mb-3 flex-wrap">
                <span className="text-sm text-muted-foreground font-medium whitespace-nowrap">挖掘方向:</span>
                <div className="flex gap-2 flex-wrap">
                  {MINING_DIRECTIONS.map((d) => (
                    <button
                      key={d.id}
                      onClick={() => handleSelectPreset(d.id)}
                      className={`px-4 py-2 rounded-lg text-sm font-medium transition-all ${
                        !isCustom && direction === d.id
                          ? 'bg-primary text-primary-foreground shadow-lg shadow-primary/25'
                          : 'bg-secondary/50 text-muted-foreground hover:text-foreground hover:bg-secondary'
                      }`}
                      title={d.description}
                    >
                      {d.label}
                    </button>
                  ))}
                  <button
                    onClick={() => setIsCustom(true)}
                    className={`px-4 py-2 rounded-lg text-sm font-medium transition-all flex items-center gap-1.5 ${
                      isCustom
                        ? 'bg-primary text-primary-foreground shadow-lg shadow-primary/25'
                        : 'bg-secondary/50 text-muted-foreground hover:text-foreground hover:bg-secondary'
                    }`}
                    title="手动输入自定义方向"
                  >
                    <PenLine className="h-3.5 w-3.5" />
                    自定义
                  </button>
                </div>
                <button
                  onClick={() => setShowAdvanced(!showAdvanced)}
                  className={`ml-auto p-2 rounded-lg transition-all ${
                    showAdvanced ? 'bg-primary/15 text-primary' : 'text-muted-foreground hover:bg-secondary/50'
                  }`}
                  title="高级设置"
                >
                  <Settings2 className="h-4 w-4" />
                </button>
              </div>

              {/* Custom Direction Input */}
              {isCustom && (
                <div className="mb-3">
                  <input
                    type="text"
                    value={customDirection}
                    onChange={(e) => setCustomDirection(e.target.value)}
                    onKeyDown={(e) => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); handleSubmit(); } }}
                    placeholder="输入自定义挖掘方向，例如：基于波动率和换手率的动量因子"
                    className="w-full px-4 py-2.5 rounded-lg border border-primary/30 bg-background text-sm focus:border-primary focus:outline-none focus:ring-2 focus:ring-primary/30 transition-all placeholder:text-muted-foreground/50"
                    autoFocus
                  />
                </div>
              )}

              {/* Advanced Settings (includes portfolios, parallel settings, etc.) */}
              {showAdvanced && (
                <div className="mb-3 space-y-3">
                  {/* Initial Direction Portfolios (when not custom) */}
                  {!isCustom && portfolios.length > 0 && (
                    <div className="max-h-48 overflow-y-auto rounded-lg border border-border/50 bg-secondary/20 p-2 space-y-1">
                      <div className="flex items-center justify-between px-1 pb-1 border-b border-border/30">
                        <span className="text-xs text-muted-foreground font-medium">
                          初始方向组合 ({selectedPortfolioIds.size}/{portfolios.length} 已选)
                        </span>
                        <div className="flex gap-2">
                          <button onClick={selectAll} className="text-xs text-primary hover:underline">
                            全选
                          </button>
                          <button onClick={deselectAll} className="text-xs text-muted-foreground hover:underline">
                            清除
                          </button>
                        </div>
                      </div>
                      {portfolios.map((p) => {
                        const isSelected = selectedPortfolioIds.has(p.portfolio_id);
                        const isExpanded = expandedPortfolioId === p.portfolio_id;
                        const shortDesc = p.description.split(',').slice(0, 1).join('').trim();
                        return (
                          <div key={p.portfolio_id} className={`rounded-md transition-all ${isSelected ? 'bg-primary/10 border border-primary/30' : 'border border-transparent hover:bg-secondary/40'}`}>
                            <div className="flex items-center gap-2 px-2 py-1.5">
                              <button
                                onClick={() => togglePortfolio(p.portfolio_id)}
                                className={`flex-shrink-0 w-4 h-4 rounded border flex items-center justify-center transition-all ${
                                  isSelected ? 'bg-primary border-primary' : 'border-muted-foreground/40'
                                }`}
                              >
                                {isSelected && <Check className="h-3 w-3 text-primary-foreground" />}
                              </button>
                              <span className="text-xs font-medium flex-1 truncate">
                                组合 {p.portfolio_id}
                              </span>
                              <span className="text-[10px] text-muted-foreground truncate max-w-[300px]">
                                {shortDesc}
                              </span>
                              <button
                                onClick={() => toggleExpandPortfolio(p.portfolio_id)}
                                className="flex-shrink-0 p-0.5 rounded text-muted-foreground hover:text-foreground"
                              >
                                {isExpanded ? <ChevronUp className="h-3 w-3" /> : <ChevronDown className="h-3 w-3" />}
                              </button>
                            </div>
                            {isExpanded && (
                              <div className="px-8 pb-2">
                                <p className="text-[11px] text-muted-foreground break-all leading-relaxed">{p.description}</p>
                              </div>
                            )}
                          </div>
                        );
                      })}
                    </div>
                  )}

                  {!isCustom && portfoliosLoading && (
                    <div className="text-xs text-muted-foreground text-center py-2">加载初始方向...</div>
                  )}

                  {/* Parameter Settings */}
                  <div className="grid grid-cols-2 md:grid-cols-5 gap-3 p-3 rounded-lg bg-secondary/30">
                    <div>
                      <label className="text-xs text-muted-foreground">并行方向数</label>
                      <input
                        type="number"
                        min={1}
                        value={numDirectionsStr}
                        onChange={(e) => setNumDirectionsStr(e.target.value)}
                        placeholder={`默认 ${defaults.numDirections}`}
                        className="w-full mt-1 px-3 py-1.5 rounded-md border border-input bg-background text-sm"
                      />
                    </div>
                    <div>
                      <label className="text-xs text-muted-foreground">每方向因子数</label>
                      <input
                        type="number"
                        min={1}
                        max={10}
                        value={factorsPerHypothesisStr}
                        onChange={(e) => setFactorsPerHypothesisStr(e.target.value)}
                        placeholder={`默认 ${defaults.factorsPerHypothesis}`}
                        className="w-full mt-1 px-3 py-1.5 rounded-md border border-input bg-background text-sm"
                      />
                    </div>
                    <div>
                      <label className="text-xs text-muted-foreground">进化轮次</label>
                      <input
                        type="number"
                        min={1}
                        value={maxRoundsStr}
                        onChange={(e) => setMaxRoundsStr(e.target.value)}
                        placeholder={`默认 ${defaults.maxRounds}`}
                        className="w-full mt-1 px-3 py-1.5 rounded-md border border-input bg-background text-sm"
                      />
                    </div>
                    <div>
                      <label className="text-xs text-muted-foreground">每方向循环数</label>
                      <input
                        type="number"
                        min={1}
                        value={maxLoopsStr}
                        onChange={(e) => setMaxLoopsStr(e.target.value)}
                        placeholder={`默认 ${defaults.maxLoops}`}
                        className="w-full mt-1 px-3 py-1.5 rounded-md border border-input bg-background text-sm"
                      />
                    </div>
                    <div>
                      <label className="text-xs text-muted-foreground">因子库后缀 (可选)</label>
                      <input
                        type="text"
                        value={librarySuffix}
                        onChange={(e) => setLibrarySuffix(e.target.value)}
                        placeholder="留空自动生成"
                        className="w-full mt-1 px-3 py-1.5 rounded-md border border-input bg-background text-sm"
                      />
                    </div>
                  </div>
                </div>
              )}

              {/* Submit Row */}
              <div className="flex items-center justify-between">
                <p className="text-xs text-muted-foreground">
                  {isCustom
                    ? (customDirection.trim() ? `将使用自定义方向「${customDirection.trim()}」` : '请输入自定义挖掘方向')
                    : selectedPortfolioIds.size > 0
                      ? `已选 ${selectedPortfolioIds.size} 个初始方向组合`
                      : `将使用「${MINING_DIRECTIONS.find(d => d.id === direction)?.description}」模式`}
                </p>
                <button
                  onClick={handleSubmit}
                  className="px-4 py-2 rounded-lg bg-primary text-primary-foreground hover:bg-primary/90 disabled:opacity-50 disabled:cursor-not-allowed transition-all hover:scale-105 active:scale-95 flex items-center gap-2"
                  title="开始挖掘 (Enter)"
                >
                  <Send className="h-4 w-4" />
                  开始挖掘
                </button>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
