import React, { useEffect, useState } from 'react';
import { X, Sliders, Volume2, HardDrive, Cpu } from 'lucide-react';
import { fetchSystemStatus } from '../services/api';

interface SettingsModalProps {
  isOpen: boolean;
  onClose: () => void;
  confidenceThreshold: number;
  onConfidenceChange: (val: number) => void;
  autoSpeak: boolean;
  onAutoSpeakChange: (val: boolean) => void;
  speechRate: number;
  onSpeechRateChange: (val: number) => void;
}

export const SettingsModal: React.FC<SettingsModalProps> = ({
  isOpen,
  onClose,
  confidenceThreshold,
  onConfidenceChange,
  autoSpeak,
  onAutoSpeakChange,
  speechRate,
  onSpeechRateChange,
}) => {
  const [telemetry, setTelemetry] = useState<{
    version: string;
    cpu_percent: number;
    memory_percent: number;
    active_model: string;
    total_gestures: number;
  } | null>(null);

  useEffect(() => {
    if (isOpen) {
      fetchSystemStatus()
        .then(setTelemetry)
        .catch((err) => console.error('Error fetching system status:', err));
    }
  }, [isOpen]);

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/70 backdrop-blur-sm animate-fade-in">
      <div className="bg-[#151B24] border border-[#2B3442] rounded-2xl w-full max-w-xl overflow-hidden shadow-2xl">
        {/* Header */}
        <div className="p-5 border-b border-[#2B3442] flex items-center justify-between">
          <div className="flex items-center space-x-2.5">
            <div className="w-8 h-8 rounded-lg bg-[#2563EB]/20 text-[#2563EB] flex items-center justify-center">
              <Sliders className="w-4 h-4" />
            </div>
            <div>
              <h2 className="text-base font-bold text-[#F3F4F6]">ArSL Studio Preferences</h2>
              <p className="text-xs text-[#94A3B8]">Configure detection sensitivity, speech, and telemetry</p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-1.5 rounded-lg text-[#94A3B8] hover:text-[#F3F4F6] hover:bg-[#1B2330] transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Content Body */}
        <div className="p-6 space-y-6 max-h-[80vh] overflow-y-auto">
          {/* Recognition Sensitivity */}
          <div className="space-y-3">
            <h3 className="text-xs font-semibold text-[#14B8A6] uppercase tracking-wider flex items-center space-x-1.5">
              <Sliders className="w-3.5 h-3.5" />
              <span>Recognition Sensitivity</span>
            </h3>
            <div className="bg-[#1B2330] border border-[#2B3442] rounded-xl p-4 space-y-3">
              <div className="flex justify-between text-xs">
                <span className="text-[#F3F4F6] font-medium">Confidence Threshold</span>
                <span className="text-[#14B8A6] font-mono font-bold">{(confidenceThreshold * 100).toFixed(0)}%</span>
              </div>
              <input
                type="range"
                min="0.5"
                max="0.95"
                step="0.05"
                value={confidenceThreshold}
                onChange={(e) => onConfidenceChange(parseFloat(e.target.value))}
                className="w-full accent-[#2563EB] cursor-pointer"
              />
              <p className="text-[11px] text-[#94A3B8]">
                Higher thresholds minimize false positives, while lower thresholds increase detection responsiveness.
              </p>
            </div>
          </div>

          {/* Audio & Speech Synthesizer */}
          <div className="space-y-3">
            <h3 className="text-xs font-semibold text-[#14B8A6] uppercase tracking-wider flex items-center space-x-1.5">
              <Volume2 className="w-3.5 h-3.5" />
              <span>Text-to-Speech (TTS)</span>
            </h3>
            <div className="bg-[#1B2330] border border-[#2B3442] rounded-xl p-4 space-y-4 text-xs">
              <div className="flex items-center justify-between">
                <div>
                  <span className="text-[#F3F4F6] font-medium block">Auto-speak sentences</span>
                  <span className="text-[#94A3B8] text-[11px]">Automatically vocalize when a word completes</span>
                </div>
                <input
                  type="checkbox"
                  checked={autoSpeak}
                  onChange={(e) => onAutoSpeakChange(e.target.checked)}
                  className="w-4 h-4 accent-[#2563EB] cursor-pointer rounded"
                />
              </div>

              <div className="space-y-2 pt-2 border-t border-[#2B3442]">
                <div className="flex justify-between">
                  <span className="text-[#F3F4F6] font-medium">Speech Rate</span>
                  <span className="text-[#14B8A6] font-mono font-bold">{speechRate.toFixed(1)}x</span>
                </div>
                <input
                  type="range"
                  min="0.7"
                  max="1.4"
                  step="0.1"
                  value={speechRate}
                  onChange={(e) => onSpeechRateChange(parseFloat(e.target.value))}
                  className="w-full accent-[#2563EB] cursor-pointer"
                />
              </div>
            </div>
          </div>

          {/* System & Telemetry */}
          {telemetry && (
            <div className="space-y-3">
              <h3 className="text-xs font-semibold text-[#14B8A6] uppercase tracking-wider flex items-center space-x-1.5">
                <HardDrive className="w-3.5 h-3.5" />
                <span>Backend Telemetry</span>
              </h3>
              <div className="bg-[#1B2330] border border-[#2B3442] rounded-xl p-4 grid grid-cols-2 gap-3 text-xs">
                <div className="flex items-center space-x-2">
                  <Cpu className="w-4 h-4 text-[#94A3B8]" />
                  <div>
                    <span className="text-[#94A3B8] block text-[10px]">CPU Usage</span>
                    <span className="text-[#F3F4F6] font-semibold">{telemetry.cpu_percent}%</span>
                  </div>
                </div>
                <div className="flex items-center space-x-2">
                  <HardDrive className="w-4 h-4 text-[#94A3B8]" />
                  <div>
                    <span className="text-[#94A3B8] block text-[10px]">RAM Usage</span>
                    <span className="text-[#F3F4F6] font-semibold">{telemetry.memory_percent}%</span>
                  </div>
                </div>
                <div className="col-span-2 pt-2 border-t border-[#2B3442] flex justify-between">
                  <span className="text-[#94A3B8]">Active AI Model:</span>
                  <span className="text-[#10B981] font-semibold">{telemetry.active_model}</span>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="p-4 bg-[#101419] border-t border-[#2B3442] flex justify-end">
          <button
            onClick={onClose}
            className="px-5 py-2 bg-[#2563EB] hover:bg-[#3B82F6] text-white rounded-lg text-xs font-semibold transition-colors"
          >
            Save & Close
          </button>
        </div>
      </div>
    </div>
  );
};

