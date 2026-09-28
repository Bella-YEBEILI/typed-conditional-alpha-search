import React from 'react';
import { ChatInput } from '@/components/ChatInput';
import { Layout } from '@/components/layout/Layout';
import type { PageId } from '@/components/layout/Layout';
import { useTaskContext } from '@/context/TaskContext';

interface HomePageProps {
  onNavigate?: (page: PageId) => void;
}

export const HomePage: React.FC<HomePageProps> = ({ onNavigate }) => {
  const { backendAvailable, miningTask: task, startMining, stopMining } = useTaskContext();

  return (
    <Layout currentPage="home" onNavigate={onNavigate || (() => {})} showNavigation={!!onNavigate}>
      <div className="flex flex-col items-center justify-center min-h-[60vh] pb-8 animate-fade-in-up">
        <div className="text-center mb-10">
          <h2 className="text-4xl font-bold mb-4 bg-gradient-to-r from-primary via-purple-500 to-pink-500 bg-clip-text text-transparent">
            欢迎使用 QuantaAlpha
          </h2>
          <p className="text-lg text-muted-foreground">
            选择挖掘方向，AI 自动挖掘高质量量化因子
          </p>
          {backendAvailable === false && (
            <p className="text-sm text-warning mt-2">后端未连接，将使用模拟数据演示</p>
          )}
          {backendAvailable === true && (
            <p className="text-sm text-success mt-2">已连接后端服务</p>
          )}
        </div>

        {/* Feature Cards */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 max-w-4xl w-full mb-10">
          <div className="glass rounded-2xl p-6 card-hover text-center cursor-pointer" onClick={() => onNavigate?.('home')}>
            <div className="text-4xl mb-3">🤖</div>
            <h3 className="font-semibold mb-2">AI 因子挖掘</h3>
            <p className="text-sm text-muted-foreground">支持 PV / Minutes / Joint 三种数据域</p>
          </div>
          <div className="glass rounded-2xl p-6 card-hover text-center cursor-pointer" onClick={() => onNavigate?.('analysis')}>
            <div className="text-4xl mb-3">🔬</div>
            <h3 className="font-semibold mb-2">因子分析</h3>
            <p className="text-sm text-muted-foreground">TQ 诊断图分析，筛选入库因子</p>
          </div>
          <div className="glass rounded-2xl p-6 card-hover text-center cursor-pointer" onClick={() => onNavigate?.('backtest')}>
            <div className="text-4xl mb-3">🚀</div>
            <h3 className="font-semibold mb-2">模型回测</h3>
            <p className="text-sm text-muted-foreground">LightGBM/XGBoost 滚动窗口回测</p>
          </div>
        </div>

        {/* Info */}
        <div className="w-full max-w-4xl glass rounded-2xl p-6 text-sm space-y-3">
          <h4 className="font-semibold text-foreground mb-3 flex items-center gap-2">
            <span className="text-lg">💡</span> 使用须知
          </h4>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-x-8 gap-y-2 text-muted-foreground">
            <div className="flex items-start gap-2">
              <span className="text-primary mt-0.5">&#9679;</span>
              <span><strong className="text-foreground">PV 方向：</strong>使用日频价量数据（open/close/high/low/volume）</span>
            </div>
            <div className="flex items-start gap-2">
              <span className="text-primary mt-0.5">&#9679;</span>
              <span><strong className="text-foreground">Minutes 方向：</strong>使用分钟级别高频数据</span>
            </div>
            <div className="flex items-start gap-2">
              <span className="text-primary mt-0.5">&#9679;</span>
              <span><strong className="text-foreground">Joint 方向：</strong>联合 PV + Minutes 数据域</span>
            </div>
            <div className="flex items-start gap-2">
              <span className="text-primary mt-0.5">&#9679;</span>
              <span><strong className="text-foreground">因子质量：</strong>高质量=双检通过，中质量=单检通过</span>
            </div>
            <div className="flex items-start gap-2">
              <span className="text-primary mt-0.5">&#9679;</span>
              <span><strong className="text-foreground">防信息泄露：</strong>训练/验证/测试集严格分离，有 Gap 隔离</span>
            </div>
            <div className="flex items-start gap-2">
              <span className="text-primary mt-0.5">&#9679;</span>
              <span><strong className="text-foreground">资源消耗：</strong>与（进化轮次 x 并行方向数）成正比</span>
            </div>
          </div>
        </div>
      </div>

      <ChatInput onSubmit={startMining} onStop={stopMining} isRunning={task?.status === 'running'} />
    </Layout>
  );
};
