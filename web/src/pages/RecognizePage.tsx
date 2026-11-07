import React, { useEffect, useRef, useState, useCallback } from 'react';
import { Volume2, Delete, Trash2, ArrowRight, Sparkles, CheckCircle2 } from 'lucide-react';
import { CameraViewport } from '../components/CameraViewport';
import { deleteSentenceWord, clearSentence } from '../services/api';
import type { ActiveModelInfo, RecognitionStreamResult } from '../types';

interface RecognizePageProps {
  activeModel: ActiveModelInfo | null;
  onNavigateToTeach: () => void;
  confidenceThreshold: number;
  autoSpeak: boolean;
  speechRate: number;
}

export const RecognizePage: React.FC<RecognizePageProps> = ({
  activeModel,
  onNavigateToTeach,
  confidenceThreshold,
  autoSpeak,
  speechRate,
}) => {
  const wsRef = useRef<WebSocket | null>(null);
  const [streamResult, setStreamResult] = useState<RecognitionStreamResult | null>(null);
  const [isWsConnected, setIsWsConnected] = useState<boolean>(false);
  const [sentenceText, setSentenceText] = useState<string>('');
  const lastSpokenWordRef = useRef<string>('');

  // Initialize Recognition WebSocket
  const connectWebSocket = useCallback(() => {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const host = window.location.host;
    const wsUrl = `${protocol}//${host}/ws/recognition`;

    const ws = new WebSocket(wsUrl);
    wsRef.current = ws;

    ws.onopen = () => {
      setIsWsConnected(true);
    };

    ws.onmessage = (event) => {
      try {
        const data: RecognitionStreamResult = JSON.parse(event.data);
        setStreamResult(data);
        if (data.sentence !== undefined) {
          setSentenceText(data.sentence);
        }

        // Auto-speak new stable recognized word if enabled
        if (autoSpeak && data.is_valid && data.recognized_sign && data.recognized_sign !== lastSpokenWordRef.current) {
          lastSpokenWordRef.current = data.recognized_sign;
          speakText(data.recognized_sign);
        }
      } catch (err) {
        console.error('Error parsing recognition stream message:', err);
      }
    };

    ws.onclose = () => {
      setIsWsConnected(false);
      // Auto-reconnect after delay
      setTimeout(() => {
        if (wsRef.current === ws) {
          connectWebSocket();
        }
      }, 2000);
    };

    ws.onerror = (err) => {
      console.error('Recognition WebSocket error:', err);
      ws.close();
    };
  }, [autoSpeak]);

  useEffect(() => {
    connectWebSocket();
    return () => {
      if (wsRef.current) {
        wsRef.current.close();
        wsRef.current = null;
      }
    };
  }, [connectWebSocket]);

  // Stream captured frame binary JPEG to backend WebSocket
  const handleFrameCapture = useCallback((blob: Blob) => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      blob.arrayBuffer().then((buf) => {
        if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
          wsRef.current.send(buf);
        }
      });
    }
  }, []);

  // Text-to-Speech execution
  const speakText = (text: string) => {
    if (!text || !window.speechSynthesis) return;
    window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.rate = speechRate;

    // Look for Arabic voice
    const voices = window.speechSynthesis.getVoices();
    const arVoice = voices.find((v) => v.lang.startsWith('ar'));
    if (arVoice) {
      utterance.voice = arVoice;
    }
    window.speechSynthesis.speak(utterance);
  };

  const handleDeleteWord = async () => {
    try {
      const res = await deleteSentenceWord();
      setSentenceText(res.sentence);
    } catch (err) {
      console.error('Error deleting word:', err);
    }
  };

  const handleClearSentence = async () => {
    try {
      const res = await clearSentence();
      setSentenceText(res.sentence);
      lastSpokenWordRef.current = '';
    } catch (err) {
      console.error('Error clearing sentence:', err);
    }
  };

  const recognizedWord = streamResult?.recognized_sign || '';
  const confidence = streamResult?.confidence || 0.0;
  const isHighConfidence = confidence >= confidenceThreshold && streamResult?.is_valid;

  return (
    <div className="space-y-6">
      {/* Top Banner */}
      <div className="bg-[#151B24] border border-[#2B3442] rounded-xl p-4 flex flex-wrap items-center justify-between gap-3 shadow-md">
        <div className="flex items-center space-x-3">
          <div className="w-10 h-10 rounded-lg bg-[#14B8A6]/20 text-[#14B8A6] flex items-center justify-center">
            <Sparkles className="w-5 h-5" />
          </div>
          <div>
            <div className="flex items-center space-x-2">
              <span className="text-sm font-bold text-[#F3F4F6]">
                {activeModel?.name || 'Iraqi Multimodal Classifier'}
              </span>
              <span className="text-xs px-2 py-0.5 rounded-full bg-[#10B981]/20 text-[#10B981] font-semibold border border-[#10B981]/30">
                {activeModel?.num_classes || 0} Signs Ready
              </span>
            </div>
            <p className="text-xs text-[#94A3B8]">
              Perform Arabic sign language gestures in front of the camera for live translation
            </p>
          </div>
        </div>

        <button
          onClick={onNavigateToTeach}
          className="flex items-center space-x-2 px-4 py-2 bg-[#1B2330] hover:bg-[#222C3D] text-[#14B8A6] border border-[#14B8A6]/30 hover:border-[#14B8A6] rounded-lg text-xs font-semibold transition-all shadow-sm"
        >
          <span>Teach Signs</span>
          <ArrowRight className="w-4 h-4" />
        </button>
      </div>

      {/* Main Two-Column Layout */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
        {/* Left Column: Camera Viewport (7 cols) */}
        <div className="lg:col-span-7">
          <CameraViewport
            onFrameCapture={handleFrameCapture}
            landmarks={streamResult?.landmarks}
            fps={streamResult?.fps}
            latencyMs={streamResult?.latency_ms}
            isStreaming={isWsConnected}
          />
        </div>

        {/* Right Column: Recognition Card & Sentence Builder (5 cols) */}
        <div className="lg:col-span-5 space-y-5">
          {/* Real-time Recognition Sign Card */}
          <div className="bg-[#1B2330] border border-[#2B3442] rounded-xl p-6 shadow-lg relative overflow-hidden">
            <div className="flex items-center justify-between mb-4">
              <span className="text-xs font-semibold text-[#94A3B8] uppercase tracking-wider">
                Recognized Sign
              </span>
              {isHighConfidence && (
                <div className="flex items-center space-x-1.5 text-xs text-[#10B981] font-semibold">
                  <CheckCircle2 className="w-4 h-4" />
                  <span>Detected</span>
                </div>
              )}
            </div>

            {/* Arabic Word Display */}
            <div className="min-h-[110px] flex items-center justify-center bg-[#101419] border border-[#2B3442] rounded-xl p-4 mb-4">
              {recognizedWord ? (
                <span
                  className={`text-4xl sm:text-5xl font-bold font-arabic transition-transform duration-200 ${
                    isHighConfidence ? 'text-[#10B981] scale-105' : 'text-[#94A3B8]'
                  }`}
                >
                  {recognizedWord}
                </span>
              ) : (
                <span className="text-sm text-[#64748B] italic">
                  Waiting for sign...
                </span>
              )}
            </div>

            {/* Confidence Bar */}
            <div className="space-y-1.5">
              <div className="flex justify-between text-xs">
                <span className="text-[#94A3B8]">Confidence</span>
                <span className="font-mono font-bold text-[#F3F4F6]">
                  {(confidence * 100).toFixed(0)}%
                </span>
              </div>
              <div className="w-full bg-[#101419] rounded-full h-2.5 overflow-hidden border border-[#2B3442]">
                <div
                  className={`h-full transition-all duration-200 rounded-full ${
                    isHighConfidence ? 'bg-[#10B981]' : 'bg-[#2563EB]'
                  }`}
                  style={{ width: `${Math.min(100, confidence * 100)}%` }}
                />
              </div>
            </div>
          </div>

          {/* Arabic Sentence Builder Card */}
          <div className="bg-[#1B2330] border border-[#2B3442] rounded-xl p-6 shadow-lg space-y-4">
            <div className="flex items-center justify-between">
              <span className="text-xs font-semibold text-[#94A3B8] uppercase tracking-wider">
                Sentence Builder
              </span>
              <span className="text-xs text-[#64748B]">RTL Arabic</span>
            </div>

            {/* Sentence Text Box */}
            <div className="min-h-[80px] max-h-[140px] overflow-y-auto bg-[#101419] border border-[#2B3442] rounded-xl p-4 font-arabic text-xl sm:text-2xl text-[#F3F4F6] text-right leading-relaxed">
              {sentenceText || (
                <span className="text-xs text-[#64748B] font-sans italic block text-left">
                  Accumulated sentence will appear here as you sign...
                </span>
              )}
            </div>

            {/* Action Buttons */}
            <div className="grid grid-cols-3 gap-2">
              <button
                onClick={() => speakText(sentenceText)}
                disabled={!sentenceText}
                className="flex items-center justify-center space-x-1.5 py-2.5 px-3 bg-[#2563EB] hover:bg-[#3B82F6] disabled:bg-[#151B24] disabled:text-[#64748B] text-white rounded-lg text-xs font-semibold transition-colors shadow-sm"
              >
                <Volume2 className="w-4 h-4" />
                <span>Speak</span>
              </button>

              <button
                onClick={handleDeleteWord}
                disabled={!sentenceText}
                className="flex items-center justify-center space-x-1.5 py-2.5 px-3 bg-[#151B24] hover:bg-[#222C3D] disabled:opacity-50 text-[#F3F4F6] border border-[#2B3442] rounded-lg text-xs font-semibold transition-colors"
              >
                <Delete className="w-4 h-4" />
                <span>Delete</span>
              </button>

              <button
                onClick={handleClearSentence}
                disabled={!sentenceText}
                className="flex items-center justify-center space-x-1.5 py-2.5 px-3 bg-[#151B24] hover:bg-[#EF4444]/20 hover:text-[#EF4444] disabled:opacity-50 text-[#94A3B8] border border-[#2B3442] rounded-lg text-xs font-semibold transition-colors"
              >
                <Trash2 className="w-4 h-4" />
                <span>Clear</span>
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

