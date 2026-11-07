import React from 'react';
import { Camera, BookOpen, BrainCircuit, Settings } from 'lucide-react';
import type { TabType, ActiveModelInfo } from '../types';

interface NavbarProps {
  activeTab: TabType;
  onTabChange: (tab: TabType) => void;
  onOpenSettings: () => void;
  activeModel: ActiveModelInfo | null;
}

export const Navbar: React.FC<NavbarProps> = ({
  activeTab,
  onTabChange,
  onOpenSettings,
  activeModel,
}) => {
  const tabs = [
    { id: 'recognize' as TabType, label: 'Recognize', icon: Camera },
    { id: 'teach' as TabType, label: 'Teach Signs', icon: BookOpen },
    { id: 'train' as TabType, label: 'Train Model', icon: BrainCircuit },
  ];

  return (
    <header className="bg-[#151B24] border-b border-[#2B3442] sticky top-0 z-40 px-4 lg:px-8">
      <div className="max-w-7xl mx-auto flex items-center justify-between h-16">
        {/* Brand */}
        <div className="flex items-center space-x-3 cursor-pointer" onClick={() => onTabChange('recognize')}>
          <div className="w-10 h-10 rounded-lg bg-gradient-to-tr from-[#2563EB] to-[#14B8A6] flex items-center justify-center shadow-md">
            <span className="text-white font-bold text-lg">Ar</span>
          </div>
          <div>
            <div className="flex items-center space-x-2">
              <h1 className="text-lg font-bold text-[#F3F4F6] tracking-tight">ArSL Studio</h1>
              <span className="text-xs px-2 py-0.5 rounded bg-[#1B2330] border border-[#2B3442] text-[#14B8A6] font-semibold">
                v2.0 Web
              </span>
            </div>
            <p className="text-xs text-[#94A3B8]">Arabic Sign Language Recognition</p>
          </div>
        </div>

        {/* 3 Primary Navigation Tabs */}
        <nav className="flex items-center space-x-1 sm:space-x-2">
          {tabs.map((tab) => {
            const Icon = tab.icon;
            const isActive = activeTab === tab.id;
            return (
              <button
                key={tab.id}
                onClick={() => onTabChange(tab.id)}
                className={`flex items-center space-x-2 px-3 sm:px-5 py-2 rounded-lg text-sm font-semibold transition-all ${
                  isActive
                    ? 'bg-[#1B2330] text-[#14B8A6] border border-[#2B3442] shadow-sm'
                    : 'text-[#94A3B8] hover:text-[#F3F4F6] hover:bg-[#1B2330]/60'
                }`}
              >
                <Icon className={`w-4 h-4 ${isActive ? 'text-[#14B8A6]' : 'text-[#94A3B8]'}`} />
                <span>{tab.label}</span>
              </button>
            );
          })}
        </nav>

        {/* Right Corner: Active Model Badge & Settings */}
        <div className="flex items-center space-x-3">
          <div className="hidden md:flex items-center space-x-2 bg-[#1B2330] border border-[#2B3442] px-3 py-1.5 rounded-full text-xs">
            <span
              className={`w-2.5 h-2.5 rounded-full ${
                activeModel?.is_trained ? 'bg-[#10B981] animate-pulse' : 'bg-[#F59E0B]'
              }`}
            />
            <span className="text-[#94A3B8]">
              {activeModel?.is_trained
                ? `Model Active: ${activeModel.num_classes} signs`
                : 'Model: Untrained'}
            </span>
          </div>

          <button
            onClick={onOpenSettings}
            title="Preferences (Ctrl+,)"
            className="p-2 rounded-lg bg-[#1B2330] text-[#94A3B8] hover:text-[#F3F4F6] hover:bg-[#222C3D] border border-[#2B3442] transition-colors"
          >
            <Settings className="w-5 h-5" />
          </button>
        </div>
      </div>
    </header>
  );
};

