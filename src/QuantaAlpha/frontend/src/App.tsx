import React, { useState, useEffect } from 'react';
import { HomePage } from '@/pages/HomePage';
import { MiningDashboardPage } from '@/pages/MiningDashboardPage';
import { FactorLibraryPage } from '@/pages/FactorLibraryPage';
import { FactorAnalysisPage } from '@/pages/FactorAnalysisPage';
import { MLBacktestPage } from '@/pages/MLBacktestPage';
import { SettingsPage } from '@/pages/SettingsPage';
import { Layout } from '@/components/layout/Layout';
import type { PageId } from '@/components/layout/Layout';
import { ParticleBackground } from '@/components/ParticleBackground';
import { TaskProvider, useTaskContext } from '@/context/TaskContext';

class GlobalErrorBoundary extends React.Component<
  { children: React.ReactNode },
  { hasError: boolean; error: string }
> {
  constructor(props: any) {
    super(props);
    this.state = { hasError: false, error: '' };
  }
  static getDerivedStateFromError(error: Error) {
    return { hasError: true, error: error.message };
  }
  render() {
    if (this.state.hasError) {
      return (
        <div style={{
          minHeight: '100vh', display: 'flex', alignItems: 'center', justifyContent: 'center',
          background: '#0a0a0f', color: '#e2e8f0', fontFamily: 'system-ui, sans-serif',
        }}>
          <div style={{ textAlign: 'center', maxWidth: 480, padding: 32 }}>
            <h2 style={{ fontSize: 24, fontWeight: 700, marginBottom: 12 }}>页面加载出错</h2>
            <p style={{ color: '#94a3b8', fontSize: 14, marginBottom: 20 }}>{this.state.error}</p>
            <button
              onClick={() => { this.setState({ hasError: false, error: '' }); window.location.reload(); }}
              style={{
                padding: '10px 24px', borderRadius: 8, border: 'none', cursor: 'pointer',
                background: 'linear-gradient(135deg, #6366f1, #8b5cf6)', color: '#fff', fontWeight: 600,
              }}
            >
              刷新页面
            </button>
          </div>
        </div>
      );
    }
    return this.props.children;
  }
}

class PageErrorBoundary extends React.Component<
  { children: React.ReactNode; pageName: string },
  { hasError: boolean; error: string }
> {
  constructor(props: any) {
    super(props);
    this.state = { hasError: false, error: '' };
  }
  static getDerivedStateFromError(error: Error) {
    return { hasError: true, error: error.message };
  }
  render() {
    if (this.state.hasError) {
      return (
        <div className="space-y-6 animate-fade-in-up">
          <div className="glass rounded-xl p-8 text-center space-y-4">
            <div className="text-4xl">⚠️</div>
            <h3 className="text-lg font-semibold">{this.props.pageName} 加载出错</h3>
            <p className="text-sm text-muted-foreground">{this.state.error}</p>
            <button
              onClick={() => this.setState({ hasError: false, error: '' })}
              className="px-4 py-2 rounded-lg bg-primary text-primary-foreground hover:bg-primary/90 transition-all"
            >
              重试
            </button>
          </div>
        </div>
      );
    }
    return this.props.children;
  }
}

const PATH_TO_PAGE: Record<string, PageId> = {
  '/': 'home',
  '/library': 'library',
  '/analysis': 'analysis',
  '/backtest': 'backtest',
  '/settings': 'settings',
  '/mining': 'mining_dashboard',
};
const PAGE_TO_PATH: Record<string, string> = Object.fromEntries(
  Object.entries(PATH_TO_PAGE).map(([k, v]) => [v, k])
);

function getInitialPage(): PageId {
  const path = window.location.pathname.replace(/\/+$/, '') || '/';
  return PATH_TO_PAGE[path] || 'home';
}

const AppContent: React.FC = () => {
  const [currentPage, setCurrentPage] = useState<PageId>(getInitialPage);
  const { miningTask } = useTaskContext();

  useEffect(() => {
    const targetPath = PAGE_TO_PATH[currentPage] || '/';
    if (window.location.pathname !== targetPath) {
      window.history.pushState(null, '', targetPath);
    }
  }, [currentPage]);

  useEffect(() => {
    const onPopState = () => setCurrentPage(getInitialPage());
    window.addEventListener('popstate', onPopState);
    return () => window.removeEventListener('popstate', onPopState);
  }, []);

  useEffect(() => {
    if (miningTask && miningTask.status === 'running' && currentPage === 'home') {
      setCurrentPage('mining_dashboard');
    }
  }, [miningTask?.taskId]);

  return (
    <>
      <ParticleBackground />
      <div style={{ display: currentPage === 'home' ? 'block' : 'none' }}>
        <HomePage onNavigate={setCurrentPage} />
      </div>
      <div style={{ display: currentPage === 'mining_dashboard' ? 'block' : 'none' }}>
        <MiningDashboardPage onNavigate={setCurrentPage} />
      </div>
      <div style={{ display: currentPage === 'library' ? 'block' : 'none' }}>
        <Layout currentPage={currentPage} onNavigate={setCurrentPage}>
          <PageErrorBoundary pageName="因子库">
            <FactorLibraryPage />
          </PageErrorBoundary>
        </Layout>
      </div>
      <div style={{ display: currentPage === 'analysis' ? 'block' : 'none' }}>
        <Layout currentPage={currentPage} onNavigate={setCurrentPage}>
          <PageErrorBoundary pageName="因子分析">
            <FactorAnalysisPage />
          </PageErrorBoundary>
        </Layout>
      </div>
      <div style={{ display: currentPage === 'backtest' ? 'block' : 'none' }}>
        <Layout currentPage={currentPage} onNavigate={setCurrentPage}>
          <PageErrorBoundary pageName="模型回测">
            <MLBacktestPage />
          </PageErrorBoundary>
        </Layout>
      </div>
      <div style={{ display: currentPage === 'settings' ? 'block' : 'none' }}>
        <Layout currentPage={currentPage} onNavigate={setCurrentPage}>
          <PageErrorBoundary pageName="设置">
            <SettingsPage />
          </PageErrorBoundary>
        </Layout>
      </div>
    </>
  );
};

export const App: React.FC = () => {
  return (
    <GlobalErrorBoundary>
      <TaskProvider>
        <AppContent />
      </TaskProvider>
    </GlobalErrorBoundary>
  );
};
