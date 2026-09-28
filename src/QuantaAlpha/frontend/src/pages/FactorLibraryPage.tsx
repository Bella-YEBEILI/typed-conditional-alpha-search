import React, { useState, useEffect, useCallback } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/Card';
import { Button } from '@/components/ui/Button';
import { Badge } from '@/components/ui/Badge';
import { Factor, FactorQuality } from '@/types';
import { formatNumber, formatPercent, getQualityBadgeClass } from '@/utils';
import { getFactors, getFactorDetail } from '@/services/api';
import {
  Database, Search, Download, RefreshCw, TrendingUp, Code,
  BarChart3, AlertCircle, Edit3, Save, X, FolderPlus, Trash2,
} from 'lucide-react';

export const FactorLibraryPage: React.FC = () => {
  const [factors, setFactors] = useState<Factor[]>([]);
  const [filteredFactors, setFilteredFactors] = useState<Factor[]>([]);
  const [searchQuery, setSearchQuery] = useState('');
  const [qualityFilter, setQualityFilter] = useState<FactorQuality | 'all'>('high');
  const [selectedFactor, setSelectedFactor] = useState<any | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [libraries, setLibraries] = useState<string[]>([]);
  const [selectedLibrary, setSelectedLibrary] = useState<string>('');
  const [metadata, setMetadata] = useState<any>(null);

  // Edit state
  const [isEditing, setIsEditing] = useState(false);
  const [editExpression, setEditExpression] = useState('');
  const [editDescription, setEditDescription] = useState('');
  const [editFormulation, setEditFormulation] = useState('');
  const [isSaving, setIsSaving] = useState(false);

  // Delete state
  const [isDeleting, setIsDeleting] = useState(false);

  // Move to library state
  const [showMoveDialog, setShowMoveDialog] = useState(false);
  const [moveTarget, setMoveTarget] = useState('');
  const [newLibName, setNewLibName] = useState('');
  const [isMoving, setIsMoving] = useState(false);

  useEffect(() => { loadFactors(); }, [selectedLibrary]);
  useEffect(() => { filterFactors(); }, [factors, searchQuery, qualityFilter]);

  const loadFactors = useCallback(async () => {
    setIsLoading(true);
    setError(null);
    try {
      const resp = await getFactors({ library: selectedLibrary || undefined, limit: 500 });
      if (resp.success && resp.data) {
        // Helper: safely convert to number, replacing NaN/null/undefined with 0
        const safeNum = (v: any, fallback = 0): number => {
          if (v == null) return fallback;
          const n = Number(v);
          return isNaN(n) || !isFinite(n) ? fallback : n;
        };
        const apiFactors: Factor[] = resp.data.factors.map((f: any) => {
          return {
            factorId: f.factorId || '',
            factorName: f.factorName || 'Unknown',
            factorExpression: f.factorExpression || '',
            factorDescription: f.factorDescription || '',
            quality: (f.quality || 'low') as FactorQuality,
            ic: safeNum(f.ic),
            icir: safeNum(f.icir),
            rankIc: safeNum(f.rankIc),
            rankIcir: safeNum(f.rankIcir),
            annualReturn: safeNum(f.annualReturn),
            maxDrawdown: safeNum(f.maxDrawdown),
            sharpeRatio: safeNum(f.sharpeRatio),
            round: f.round || 0,
            direction: String(f.direction ?? ''),
            createdAt: f.createdAt || new Date().toISOString(),
            backtestResults: f.backtestResults,
            factorFormulation: f.factorFormulation,
          };
        });
        setFactors(apiFactors);
        setLibraries(resp.data.libraries || []);
        setMetadata(resp.data.metadata || null);
      }
    } catch (err: any) {
      setError('无法连接后端服务');
    } finally {
      setIsLoading(false);
    }
  }, [selectedLibrary]);

  const filterFactors = () => {
    let filtered = factors;
    if (qualityFilter !== 'all') {
      filtered = filtered.filter((f) => f.quality === qualityFilter);
    }
    if (searchQuery) {
      const query = searchQuery.toLowerCase();
      filtered = filtered.filter((f) =>
        f.factorName.toLowerCase().includes(query) ||
        f.factorExpression.toLowerCase().includes(query) ||
        f.factorDescription.toLowerCase().includes(query)
      );
    }
    const qualityOrder: Record<string, number> = { high: 0, medium: 1, low: 2 };
    filtered = [...filtered].sort((a, b) => {
      const qa = qualityOrder[a.quality] ?? 3;
      const qb = qualityOrder[b.quality] ?? 3;
      if (qa !== qb) return qa - qb;
      return Math.abs(b.rankIc) - Math.abs(a.rankIc);
    });
    setFilteredFactors(filtered);
  };

  const handleExport = () => {
    const dataStr = JSON.stringify(factors, null, 2);
    const blob = new Blob([dataStr], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `factors_${new Date().toISOString().split('T')[0]}.json`;
    link.click();
    URL.revokeObjectURL(url);
  };

  const handleSelectFactor = async (factor: Factor) => {
    try {
      const resp = await getFactorDetail(factor.factorId);
      if (resp.success && resp.data?.factor) {
        setSelectedFactor({ ...factor, ...resp.data.factor });
        setEditExpression(resp.data.factor.factor_expression || factor.factorExpression);
        setEditDescription(resp.data.factor.factor_description || factor.factorDescription);
        setEditFormulation(resp.data.factor.factor_formulation || '');
        setIsEditing(false);
        return;
      }
    } catch { /* fallback */ }
    setSelectedFactor(factor);
    setEditExpression(factor.factorExpression);
    setEditDescription(factor.factorDescription);
    setEditFormulation('');
    setIsEditing(false);
  };

  const [saveMessage, setSaveMessage] = useState('');

  const handleSaveFactor = async () => {
    if (!selectedFactor) return;
    const expressionChanged = editExpression !== (selectedFactor.factorExpression || selectedFactor.factor_expression || '');
    setIsSaving(true);
    setSaveMessage(expressionChanged ? '正在重新回测...' : '保存中...');
    try {
      const fid = selectedFactor.factorId || selectedFactor.factor_id;
      const qs = selectedLibrary ? `?library=${encodeURIComponent(selectedLibrary)}` : '';
      const res = await fetch(`/api/v1/factors/${fid}${qs}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          factorExpression: editExpression,
          factorDescription: editDescription,
          factorFormulation: editFormulation,
        }),
      });
      if (res.ok) {
        const data = await res.json();
        setSaveMessage(data.message || '已保存');
        setIsEditing(false);
        await loadFactors();
        // Refresh the selected factor detail
        const resp = await getFactorDetail(fid);
        if (resp.success && resp.data?.factor) {
          const detail = resp.data.factor;
          setSelectedFactor((prev: any) => ({ ...prev, ...detail, quality: data.data?.quality || prev?.quality }));
        }
        setTimeout(() => setSaveMessage(''), 3000);
      } else {
        setSaveMessage('保存失败');
        setTimeout(() => setSaveMessage(''), 3000);
      }
    } catch (err) {
      console.error('Failed to save factor:', err);
      setSaveMessage('保存失败');
      setTimeout(() => setSaveMessage(''), 3000);
    }
    setIsSaving(false);
  };

  const handleMoveToLibrary = async () => {
    if (!selectedFactor) return;
    const target = newLibName.trim() ? `all_factors_library_${newLibName.trim()}.json` : moveTarget;
    if (!target) return;
    setIsMoving(true);
    try {
      const fid = selectedFactor.factorId || selectedFactor.factor_id;
      const res = await fetch('/api/v1/factors/add-to-library', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          factorIds: [fid],
          targetLibrary: target,
          createNew: !!newLibName.trim(),
        }),
      });
      if (res.ok) {
        setShowMoveDialog(false);
        setNewLibName('');
        loadFactors();
      }
    } catch (err) {
      console.error('Failed to move factor:', err);
    }
    setIsMoving(false);
  };

  const handleDeleteFactor = async () => {
    if (!selectedFactor) return;
    const fid = selectedFactor.factorId || selectedFactor.factor_id;
    if (!confirm(`确定要删除因子「${selectedFactor.factorName || selectedFactor.factor_name}」吗？此操作不可撤销。`)) return;
    setIsDeleting(true);
    try {
      const qs = selectedLibrary ? `?library=${encodeURIComponent(selectedLibrary)}` : '';
      const res = await fetch(`/api/v1/factors/${fid}${qs}`, { method: 'DELETE' });
      if (res.ok) {
        setSelectedFactor(null);
        loadFactors();
      } else {
        const data = await res.json().catch(() => ({}));
        alert(data.detail || '删除失败');
      }
    } catch (err) {
      console.error('Failed to delete factor:', err);
      alert('删除请求失败');
    }
    setIsDeleting(false);
  };

  const handleDeleteLibrary = async () => {
    if (!selectedLibrary) {
      alert('请先选择一个因子库');
      return;
    }
    if (!confirm(`确定要删除整个因子库「${selectedLibrary}」吗？此操作不可撤销！`)) return;
    try {
      const res = await fetch(`/api/v1/factors/libraries/${encodeURIComponent(selectedLibrary)}`, { method: 'DELETE' });
      const data = await res.json().catch(() => ({}));
      if (res.ok) {
        setSelectedLibrary('');
        setSelectedFactor(null);
        loadFactors();
      } else {
        alert(data.detail || '删除因子库失败');
      }
    } catch (err) {
      console.error('Failed to delete library:', err);
      alert('删除因子库请求失败');
    }
  };

  const stats = {
    total: factors.length,
    high: factors.filter((f) => f.quality === 'high').length,
    medium: factors.filter((f) => f.quality === 'medium').length,
    low: factors.filter((f) => f.quality === 'low').length,
  };

  return (
    <div className="space-y-6 animate-fade-in-up">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold flex items-center gap-3">
            <Database className="h-8 w-8 text-primary" />
            因子库
          </h1>
          <p className="text-muted-foreground mt-1">
            浏览、管理和编辑挖掘的因子
            {metadata?.total_factors != null && (
              <span className="ml-2 text-xs">
                (更新于 {metadata.last_updated ? new Date(metadata.last_updated).toLocaleString('zh-CN') : '未知'})
              </span>
            )}
          </p>
        </div>
        <div className="flex gap-3">
          {libraries.length > 1 && (
            <div className="flex items-center gap-2">
              <select
                value={selectedLibrary}
                onChange={(e) => setSelectedLibrary(e.target.value)}
                className="rounded-lg border border-input bg-background px-3 py-2 text-sm"
              >
                <option value="">最新因子库</option>
                {libraries.map((lib) => (
                  <option key={lib} value={lib}>{lib}</option>
                ))}
              </select>
              {selectedLibrary && (
                <Button variant="outline" size="sm" onClick={handleDeleteLibrary}
                  className="text-destructive hover:bg-destructive/10">
                  <Trash2 className="h-4 w-4" />
                </Button>
              )}
            </div>
          )}
          <Button variant="outline" onClick={loadFactors} disabled={isLoading}>
            <RefreshCw className={`h-4 w-4 mr-2 ${isLoading ? 'animate-spin' : ''}`} />
            刷新
          </Button>
          <Button variant="primary" onClick={handleExport}>
            <Download className="h-4 w-4 mr-2" />
            导出
          </Button>
        </div>
      </div>

      {error && (
        <div className="glass rounded-lg p-4 flex items-center gap-3 bg-warning/10 border-warning/50">
          <AlertCircle className="h-5 w-5 text-warning flex-shrink-0" />
          <span className="text-sm text-warning">{error}</span>
        </div>
      )}

      {/* Stats */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        <Card className="glass card-hover">
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <div className="text-sm text-muted-foreground">总因子数</div>
                <div className="text-2xl font-bold mt-1">{stats.total}</div>
              </div>
              <div className="p-3 rounded-lg bg-primary/20">
                <BarChart3 className="h-6 w-6 text-primary" />
              </div>
            </div>
          </CardContent>
        </Card>
        <Card className="glass card-hover">
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <div className="text-sm text-muted-foreground">高质量 (双通过)</div>
                <div className="text-2xl font-bold mt-1 text-success">{stats.high}</div>
              </div>
              <div className="p-3 rounded-lg bg-success/20">
                <TrendingUp className="h-6 w-6 text-success" />
              </div>
            </div>
          </CardContent>
        </Card>
        <Card className="glass card-hover">
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <div className="text-sm text-muted-foreground">中等质量 (单通过)</div>
                <div className="text-2xl font-bold mt-1 text-warning">{stats.medium}</div>
              </div>
              <div className="p-3 rounded-lg bg-warning/20">
                <BarChart3 className="h-6 w-6 text-warning" />
              </div>
            </div>
          </CardContent>
        </Card>
        <Card className="glass card-hover">
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <div className="text-sm text-muted-foreground">低质量</div>
                <div className="text-2xl font-bold mt-1 text-destructive">{stats.low}</div>
              </div>
              <div className="p-3 rounded-lg bg-destructive/20">
                <BarChart3 className="h-6 w-6 text-destructive" />
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Filters */}
      <Card className="glass">
        <CardContent className="p-4">
          <div className="flex flex-col md:flex-row gap-4">
            <div className="flex-1">
              <div className="relative">
                <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                <input
                  type="text"
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  placeholder="搜索因子名称、表达式或描述..."
                  className="w-full pl-10 pr-4 py-2 rounded-lg border border-input bg-background text-sm focus:border-primary focus:outline-none focus:ring-2 focus:ring-primary transition-all"
                />
              </div>
            </div>
            <div className="flex gap-2">
              {(['all', 'high', 'medium', 'low'] as const).map((q) => (
                <Button
                  key={q}
                  variant={qualityFilter === q ? 'primary' : 'outline'}
                  size="sm"
                  onClick={() => setQualityFilter(q)}
                >
                  {q === 'all' ? `全部 (${stats.total})` :
                   q === 'high' ? `高 (${stats.high})` :
                   q === 'medium' ? `中 (${stats.medium})` :
                   `低 (${stats.low})`}
                </Button>
              ))}
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Factor List - 紧凑列表 */}
      <Card className="glass">
        <CardContent className="p-0">
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-border/50 bg-secondary/20">
                  <th className="py-2.5 px-3 text-left font-medium text-muted-foreground text-xs">质量</th>
                  <th className="py-2.5 px-3 text-left font-medium text-muted-foreground text-xs">因子名</th>
                  <th className="py-2.5 px-3 text-left font-medium text-muted-foreground text-xs">表达式</th>
                  <th className="py-2.5 px-3 text-right font-medium text-muted-foreground text-xs">RankIC</th>
                  <th className="py-2.5 px-3 text-right font-medium text-muted-foreground text-xs">RankICIR</th>
                  <th className="py-2.5 px-3 text-right font-medium text-muted-foreground text-xs">IC</th>
                  <th className="py-2.5 px-3 text-right font-medium text-muted-foreground text-xs">ICIR</th>
                  <th className="py-2.5 px-3 text-right font-medium text-muted-foreground text-xs">多头收益</th>
                  <th className="py-2.5 px-3 text-right font-medium text-muted-foreground text-xs">Sharpe</th>
                  <th className="py-2.5 px-3 text-right font-medium text-muted-foreground text-xs">最大回撤</th>
                </tr>
              </thead>
              <tbody>
                {filteredFactors.map((factor) => (
                  <tr
                    key={factor.factorId}
                    className="border-b border-border/30 hover:bg-muted/50 cursor-pointer transition-colors"
                    onClick={() => handleSelectFactor(factor)}
                  >
                    <td className="py-2 px-3">
                      <Badge className={`${getQualityBadgeClass(factor.quality)} text-[10px] px-1.5 py-0`}>
                        {factor.quality === 'high' ? '高' : factor.quality === 'medium' ? '中' : '低'}
                      </Badge>
                    </td>
                    <td className="py-2 px-3 font-medium text-xs max-w-[160px] truncate" title={factor.factorName}>
                      {factor.factorName}
                    </td>
                    <td className="py-2 px-3 font-mono text-[11px] text-muted-foreground max-w-[250px] truncate" title={factor.factorExpression}>
                      {factor.factorExpression}
                    </td>
                    <td className="py-2 px-3 text-right font-mono text-xs font-bold text-primary">{formatNumber(factor.rankIc, 4)}</td>
                    <td className="py-2 px-3 text-right font-mono text-xs">{formatNumber(factor.rankIcir, 3)}</td>
                    <td className="py-2 px-3 text-right font-mono text-xs">{formatNumber(factor.ic, 4)}</td>
                    <td className="py-2 px-3 text-right font-mono text-xs">{formatNumber(factor.icir, 3)}</td>
                    <td className={`py-2 px-3 text-right font-mono text-xs font-bold ${(factor.annualReturn ?? 0) > 0 ? 'text-success' : (factor.annualReturn ?? 0) < 0 ? 'text-destructive' : ''}`}>
                      {factor.annualReturn != null && factor.annualReturn !== 0 ? formatPercent(factor.annualReturn) : '-'}
                    </td>
                    <td className={`py-2 px-3 text-right font-mono text-xs ${(factor.sharpeRatio ?? 0) > 0 ? 'text-success' : (factor.sharpeRatio ?? 0) < 0 ? 'text-destructive' : ''}`}>
                      {factor.sharpeRatio != null && factor.sharpeRatio !== 0 ? formatNumber(factor.sharpeRatio, 2) : '-'}
                    </td>
                    <td className="py-2 px-3 text-right font-mono text-xs text-destructive">
                      {factor.maxDrawdown != null && factor.maxDrawdown !== 0 ? formatPercent(factor.maxDrawdown) : '-'}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </CardContent>
      </Card>

      {filteredFactors.length === 0 && !isLoading && (
        <Card className="glass">
          <CardContent className="p-12 text-center">
            <Database className="h-16 w-16 mx-auto text-muted-foreground mb-4" />
            <h3 className="text-lg font-medium mb-2">暂无因子</h3>
            <p className="text-sm text-muted-foreground">
              {searchQuery || qualityFilter !== 'all' ? '没有符合筛选条件的因子' : '开始挖掘因子后，结果将显示在这里'}
            </p>
          </CardContent>
        </Card>
      )}

      {/* Factor Detail Modal with Edit */}
      {selectedFactor && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm p-6" onClick={() => setSelectedFactor(null)}>
          <Card className="glass-strong max-w-3xl w-full max-h-[85vh] overflow-y-auto animate-scale-in" onClick={(e: React.MouseEvent) => e.stopPropagation()}>
            <CardHeader>
              <div className="flex items-start justify-between">
                <div className="flex-1">
                  <CardTitle className="text-xl">{selectedFactor.factorName || selectedFactor.factor_name}</CardTitle>
                  <div className="flex items-center gap-2 mt-2">
                    <Badge className={getQualityBadgeClass(selectedFactor.quality || 'medium')}>
                      {selectedFactor.quality === 'high' ? '高质量 (双通过)' : selectedFactor.quality === 'medium' ? '中等质量 (单通过)' : '低质量'}
                    </Badge>
                  </div>
                </div>
                <div className="flex gap-2">
                  {!isEditing ? (
                    <>
                      <Button variant="outline" size="sm" onClick={() => setIsEditing(true)}>
                        <Edit3 className="h-4 w-4 mr-1" /> 编辑
                      </Button>
                      <Button variant="outline" size="sm" onClick={() => setShowMoveDialog(true)}>
                        <FolderPlus className="h-4 w-4 mr-1" /> 移入因子库
                      </Button>
                      <Button variant="outline" size="sm" onClick={handleDeleteFactor} disabled={isDeleting}
                        className="text-destructive border-destructive/50 hover:bg-destructive/10">
                        <Trash2 className="h-4 w-4 mr-1" /> {isDeleting ? '删除中...' : '删除'}
                      </Button>
                    </>
                  ) : (
                    <>
                      <Button variant="primary" size="sm" onClick={handleSaveFactor} disabled={isSaving}>
                        <Save className="h-4 w-4 mr-1" /> {isSaving ? (saveMessage || '保存中...') : '保存'}
                      </Button>
                      <Button variant="outline" size="sm" onClick={() => setIsEditing(false)}>
                        <X className="h-4 w-4" />
                      </Button>
                    </>
                  )}
                  <Button variant="ghost" onClick={() => setSelectedFactor(null)}>✕</Button>
                </div>
              </div>
            </CardHeader>
            <CardContent className="space-y-4">
              {/* Description */}
              <div>
                <h4 className="text-sm font-medium mb-2">因子描述</h4>
                {isEditing ? (
                  <textarea
                    value={editDescription}
                    onChange={(e) => setEditDescription(e.target.value)}
                    className="w-full rounded-lg border border-input bg-background p-3 text-sm min-h-[60px]"
                  />
                ) : (
                  <p className="text-sm text-muted-foreground">{selectedFactor.factorDescription || selectedFactor.factor_description || '无描述'}</p>
                )}
              </div>

              {/* Expression */}
              <div>
                <h4 className="text-sm font-medium mb-2">因子表达式</h4>
                {isEditing ? (
                  <textarea
                    value={editExpression}
                    onChange={(e) => setEditExpression(e.target.value)}
                    className="w-full rounded-lg border border-input bg-background p-3 text-sm font-mono min-h-[80px]"
                  />
                ) : (
                  <div className="rounded-lg bg-secondary/30 p-4">
                    <code className="text-sm font-mono break-all">{selectedFactor.factorExpression || selectedFactor.factor_expression || ''}</code>
                  </div>
                )}
              </div>

              {/* Formulation */}
              <div>
                <h4 className="text-sm font-medium mb-2">数学公式</h4>
                {isEditing ? (
                  <textarea
                    value={editFormulation}
                    onChange={(e) => setEditFormulation(e.target.value)}
                    className="w-full rounded-lg border border-input bg-background p-3 text-sm font-mono min-h-[60px]"
                    placeholder="可选：输入因子的数学公式表示"
                  />
                ) : (
                  (selectedFactor.factorFormulation || selectedFactor.factor_formulation) && (
                    <div className="rounded-lg bg-secondary/30 p-4">
                      <code className="text-sm font-mono break-all">{selectedFactor.factorFormulation || selectedFactor.factor_formulation}</code>
                    </div>
                  )
                )}
              </div>

              {/* Backtest Results */}
              {(selectedFactor.backtestResults || selectedFactor.backtest_results) && (
                <div>
                  <h4 className="text-sm font-medium mb-2">回测指标</h4>
                  <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
                    {Object.entries(selectedFactor.backtestResults || selectedFactor.backtest_results || {}).map(([key, val]) => (
                      <div key={key} className="rounded-lg bg-secondary/30 p-3">
                        <div className="text-xs text-muted-foreground truncate" title={key}>{key}</div>
                        <div className="text-sm font-bold font-mono mt-1">{typeof val === 'number' ? formatNumber(val, 4) : String(val)}</div>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Meta */}
              <div>
                <h4 className="text-sm font-medium mb-2">元信息</h4>
                <div className="space-y-2 text-sm">
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">因子ID:</span>
                    <span className="font-mono">{selectedFactor.factorId || selectedFactor.factor_id || ''}</span>
                  </div>
                  {(selectedFactor.createdAt || selectedFactor.added_at) && (
                    <div className="flex justify-between">
                      <span className="text-muted-foreground">创建时间:</span>
                      <span>{new Date(selectedFactor.createdAt || selectedFactor.added_at).toLocaleString('zh-CN')}</span>
                    </div>
                  )}
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
      )}

      {/* Move to Library Dialog */}
      {showMoveDialog && (
        <div className="fixed inset-0 z-[60] flex items-center justify-center bg-black/50 backdrop-blur-sm p-6" onClick={() => setShowMoveDialog(false)}>
          <Card className="glass-strong max-w-md w-full animate-scale-in" onClick={(e: React.MouseEvent) => e.stopPropagation()}>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <FolderPlus className="h-5 w-5" />
                添加到因子库
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div>
                <label className="text-sm font-medium mb-2 block">选择已有因子库</label>
                <select
                  value={moveTarget}
                  onChange={(e) => { setMoveTarget(e.target.value); setNewLibName(''); }}
                  className="w-full rounded-lg border border-input bg-background px-3 py-2 text-sm"
                >
                  <option value="">-- 选择因子库 --</option>
                  {libraries.map((lib) => (
                    <option key={lib} value={lib}>{lib}</option>
                  ))}
                </select>
              </div>
              <div className="text-center text-xs text-muted-foreground">或</div>
              <div>
                <label className="text-sm font-medium mb-2 block">新建因子库</label>
                <div className="flex items-center gap-2">
                  <span className="text-xs text-muted-foreground whitespace-nowrap">all_factors_library_</span>
                  <input
                    type="text"
                    value={newLibName}
                    onChange={(e) => { setNewLibName(e.target.value); setMoveTarget(''); }}
                    placeholder="custom_name"
                    className="flex-1 rounded-lg border border-input bg-background px-3 py-2 text-sm"
                  />
                  <span className="text-xs text-muted-foreground">.json</span>
                </div>
              </div>
              <div className="flex justify-end gap-2 pt-2">
                <Button variant="outline" onClick={() => setShowMoveDialog(false)}>取消</Button>
                <Button variant="primary" onClick={handleMoveToLibrary} disabled={isMoving || (!moveTarget && !newLibName.trim())}>
                  {isMoving ? '添加中...' : '确认添加'}
                </Button>
              </div>
            </CardContent>
          </Card>
        </div>
      )}
    </div>
  );
};
