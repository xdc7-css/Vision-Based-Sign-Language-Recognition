import React, { useEffect, useState, useCallback } from 'react';
import { Navbar } from './components/Navbar';
import { SettingsModal } from './components/SettingsModal';
import { RecognizePage } from './pages/RecognizePage';
import { TeachSignsPage } from './pages/TeachSignsPage';
import { TrainModelPage } from './pages/TrainModelPage';
import { fetchActiveModel } from './services/api';
import type { TabType, ActiveModelInfo } from './types';

export const App: React.FC = () => {
  const [activeTab, setActiveTab] = useState<TabType>('recognize');
  const [activeModel, setActiveModel] = useState<ActiveModelInfo | null>(null);
  const [isSettingsOpen, setIsSettingsOpen] = useState<boolean>(false);

  // Synchronize URL hash with active tab and settings modal
  useEffect(() => {
    const handleHash = () => {
      const h = window.location.hash.replace('#', '').toLowerCase();
      if (h === 'teach') {
        setActiveTab('teach');
        setIsSettingsOpen(false);
      } else if (h === 'train') {
        setActiveTab('train');
        setIsSettingsOpen(false);
      } else if (h === 'recognize') {
        setActiveTab('recognize');
        setIsSettingsOpen(false);
      } else if (h === 'settings') {
        setIsSettingsOpen(true);
      }
    };
    handleHash();
    window.addEventListener('hashchange', handleHash);
    return () => window.removeEventListener('hashchange', handleHash);
  }, []);

  const changeTab = (tab: TabType) => {
    setActiveTab(tab);
    window.location.hash = tab;
  };

  // Global Settings State
  const [confidenceThreshold, setConfidenceThreshold] = useState<number>(0.65);
  const [autoSpeak, setAutoSpeak] = useState<boolean>(false);
  const [speechRate, setSpeechRate] = useState<number>(1.0);

  // Load Active Model Metadata
  const refreshActiveModel = useCallback(async () => {
    try {
      const data = await fetchActiveModel();
      setActiveModel(data);
    } catch (err) {
      console.error('Failed to load active model metadata:', err);
    }
  }, []);

  useEffect(() => {
    refreshActiveModel();
  }, [refreshActiveModel]);

  // Global shortcut: Ctrl+, opens Settings
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && e.key === ',') {
        e.preventDefault();
        setIsSettingsOpen(true);
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, []);

  return (
    <div className="min-h-screen bg-[#101419] text-[#F3F4F6] flex flex-col font-sans selection:bg-[#2563EB] selection:text-white">
      {/* Navbar with 3 Tabs & Model Status */}
      <Navbar
        activeTab={activeTab}
        onTabChange={changeTab}
        onOpenSettings={() => {
          setIsSettingsOpen(true);
          window.location.hash = 'settings';
        }}
        activeModel={activeModel}
      />

      {/* Main Content Area */}
      <main className="flex-1 max-w-7xl w-full mx-auto p-4 sm:p-6 lg:p-8">
        {activeTab === 'recognize' && (
          <RecognizePage
            activeModel={activeModel}
            onNavigateToTeach={() => setActiveTab('teach')}
            confidenceThreshold={confidenceThreshold}
            autoSpeak={autoSpeak}
            speechRate={speechRate}
          />
        )}

        {activeTab === 'teach' && (
          <TeachSignsPage onNavigateToTrain={() => setActiveTab('train')} />
        )}

        {activeTab === 'train' && (
          <TrainModelPage
            onNavigateToRecognize={() => {
              refreshActiveModel();
              setActiveTab('recognize');
            }}
            onModelTrained={refreshActiveModel}
          />
        )}
      </main>

      {/* Settings Modal */}
      <SettingsModal
        isOpen={isSettingsOpen}
        onClose={() => setIsSettingsOpen(false)}
        confidenceThreshold={confidenceThreshold}
        onConfidenceChange={setConfidenceThreshold}
        autoSpeak={autoSpeak}
        onAutoSpeakChange={setAutoSpeak}
        speechRate={speechRate}
        onSpeechRateChange={setSpeechRate}
      />

      {/* Footer */}
      <footer className="border-t border-[#2B3442] bg-[#151B24] py-4 px-6 text-center text-xs text-[#94A3B8]">
        <div className="max-w-7xl mx-auto flex flex-wrap items-center justify-between gap-2">
          <span>ArSL Studio — Computer Engineering Graduation Project</span>
          <span className="font-mono text-[11px] text-[#64748B]">Local Engine: Python 3.12 • MediaPipe • Scikit-learn</span>
        </div>
      </footer>
    </div>
  );
};

export default App;
