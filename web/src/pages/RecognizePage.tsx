import React, { useEffect, useRef, useState, useCallback } from 'react';
import { Volume2, Delete, Trash2, ArrowRight, Sparkles, CheckCircle2, Activity, ChevronDown, ChevronUp } from 'lucide-react';
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

// Pack 16-byte header [uint64 frameId, float64 timestamp] + JPEG ArrayBuffer
async function packFramePayload(blob: Blob, frameId: number, timestamp: number): Promise<ArrayBuffer> {
  const buf = await blob.arrayBuffer();
  const payload = new Uint8Array(16 + buf.byteLength);
  const view = new DataView(payload.buffer);
  view.setBigUint64(0, BigInt(frameId), false); // big-endian
  view.setFloat64(8, timestamp, false);
  payload.set(new Uint8Array(buf), 16);
  return payload.buffer;
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
  const [isProcessing, setIsProcessing] = useState<boolean>(false);
  const [sentenceText, setSentenceText] = useState<string>('');
  const [showDiagnostics, setShowDiagnostics] = useState<boolean>(false);

  // Latency & Telemetry stats
  const latencySamplesRef = useRef<number[]>([]);
  const [latencyStats, setLatencyStats] = useState<{ avg: number; p50: number; p95: number }>({
    avg: 0,
    p50: 0,
    p95: 0,
  });

  const inFlightRef = useRef<boolean>(false);
  const inFlightTimerRef = useRef<number | null>(null);
  const lastProcessedFrameIdRef = useRef<number>(0);
  const lastEmittedWordRef = useRef<string>('');

  // Record End-to-End latency samples and compute rolling percentiles
  const recordLatencySample = useCallback((sampleMs: number) => {
    if (sampleMs <= 0 || sampleMs > 3000) return;
    const samples = latencySamplesRef.current;
    samples.push(sampleMs);
    if (samples.length > 30) {
      samples.shift();
    }
    const sorted = [...samples].sort((a, b) => a - b);
    const avg = samples.reduce((acc, v) => acc + v, 0) / samples.length;
    const p50 = sorted[Math.floor(sorted.length * 0.5)] || avg;
    const p95 = sorted[Math.floor(sorted.length * 0.95)] || avg;
    setLatencyStats({ avg: Math.round(avg), p50: Math.round(p50), p95: Math.round(p95) });
  }, []);

  // Text-to-Speech execution
  const speakText = useCallback((text: string) => {
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
  }, [speechRate]);

  // Initialize Recognition WebSocket
  const connectWebSocket = useCallback(() => {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const host = window.location.host;
    const wsUrl = `${protocol}//${host}/ws/recognition`;

    const ws = new WebSocket(wsUrl);
    wsRef.current = ws;

    ws.onopen = () => {
      setIsWsConnected(true);
      inFlightRef.current = false;
      setIsProcessing(false);
    };

    ws.onmessage = (event) => {
      try {
        const data: RecognitionStreamResult = JSON.parse(event.data);

        // Release in-flight backpressure lock immediately
        inFlightRef.current = false;
        setIsProcessing(false);
        if (inFlightTimerRef.current !== null) {
          clearTimeout(inFlightTimerRef.current);
          inFlightTimerRef.current = null;
        }

        // Drop out-of-order responses
        if (data.frame_id && data.frame_id < lastProcessedFrameIdRef.current) {
          return;
        }
        if (data.frame_id) {
          lastProcessedFrameIdRef.current = data.frame_id;
        }

        // Measure true browser-to-browser End-to-End Latency
        if (data.client_time_ms && data.client_time_ms > 0) {
          const trueE2E = performance.now() - data.client_time_ms;
          recordLatencySample(trueE2E);
        }

        setStreamResult(data);

        // Update sentence text from authoritative backend sentence builder
        if (data.sentence !== undefined) {
          setSentenceText(data.sentence);
        }

        // Auto-speak ONLY when state machine emits a newly confirmed sign!
        if (autoSpeak && data.emitted_word && data.emitted_word !== lastEmittedWordRef.current) {
          lastEmittedWordRef.current = data.emitted_word;
          speakText(data.emitted_word);
        }
      } catch (err) {
        console.error('Error parsing recognition stream message:', err);
      }
    };

    ws.onclose = () => {
      setIsWsConnected(false);
      inFlightRef.current = false;
      setIsProcessing(false);
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
  }, [autoSpeak, recordLatencySample, speakText]);

  useEffect(() => {
    connectWebSocket();
    return () => {
      if (wsRef.current) {
        wsRef.current.close();
        wsRef.current = null;
      }
      if (inFlightTimerRef.current !== null) {
        clearTimeout(inFlightTimerRef.current);
      }
    };
  }, [connectWebSocket]);

  // Stream captured frame to backend with strict backpressure & watchdog
  const handleFrameCapture = useCallback(async (blob: Blob, frameId: number, timestamp: number) => {
    if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) return;
    if (wsRef.current.bufferedAmount > 32768) return; // Prevent socket congestion

    inFlightRef.current = true;
    setIsProcessing(true);

    // Watchdog timer: release lock after 250ms if packet was dropped
    if (inFlightTimerRef.current !== null) {
      clearTimeout(inFlightTimerRef.current);
    }
    inFlightTimerRef.current = window.setTimeout(() => {
      inFlightRef.current = false;
      setIsProcessing(false);
      inFlightTimerRef.current = null;
    }, 250);

    try {
      const payload = await packFramePayload(blob, frameId, timestamp);
      if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
        wsRef.current.send(payload);
      }
    } catch (err) {
      inFlightRef.current = false;
      setIsProcessing(false);
    }
  }, []);

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
      lastEmittedWordRef.current = '';
    } catch (err) {
      console.error('Error clearing sentence:', err);
    }
  };

  const recognizedWord = streamResult?.recognized_sign || '';
  const confidence = streamResult?.confidence || 0.0;
  const isHighConfidence = confidence >= confidenceThreshold && streamResult?.is_valid;
  const state = streamResult?.state || 'idle';
  const stateProgress = streamResult?.state_progress || 0.0;
  const isHolding = streamResult?.is_holding || false;
  const statusLabel = streamResult?.status_label;

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
              Perform Arabic sign language gestures in front of the camera for real-time stabilized recognition
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
        <div className="lg:col-span-7 space-y-4">
          <CameraViewport
            onFrameCapture={handleFrameCapture}
            isProcessing={isProcessing}
            landmarks={streamResult?.landmarks}
            fps={streamResult?.fps}
            latencyMs={latencyStats.p50 > 0 ? latencyStats.p50 : streamResult?.latency_ms}
            isStreaming={isWsConnected}
            statusLabel={statusLabel}
            stateProgress={stateProgress}
          />

          {/* Collapsible Performance Diagnostics Drawer */}
          <div className="bg-[#151B24] border border-[#2B3442] rounded-xl overflow-hidden shadow-sm">
            <button
              onClick={() => setShowDiagnostics((prev) => !prev)}
              className="w-full px-4 py-2.5 flex items-center justify-between text-xs font-semibold text-[#94A3B8] hover:text-[#F3F4F6] transition-colors"
            >
              <div className="flex items-center space-x-2">
                <Activity className="w-4 h-4 text-[#38BDF8]" />
                <span>Real-Time Latency & Pipeline Profiling</span>
                <span className="font-mono text-[#38BDF8]">
                  P50: {latencyStats.p50}ms • P95: {latencyStats.p95}ms
                </span>
              </div>
              {showDiagnostics ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
            </button>

            {showDiagnostics && (
              <div className="p-4 border-t border-[#2B3442] bg-[#101419]/50 grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs font-mono">
                <div className="bg-[#1B2330] p-2.5 rounded-lg border border-[#2B3442]/60">
                  <div className="text-[#64748B] text-[11px]">E2E Latency (Avg)</div>
                  <div className="text-sm font-bold text-[#38BDF8]">{latencyStats.avg} ms</div>
                </div>
                <div className="bg-[#1B2330] p-2.5 rounded-lg border border-[#2B3442]/60">
                  <div className="text-[#64748B] text-[11px]">P95 Latency</div>
                  <div className="text-sm font-bold text-[#F59E0B]">{latencyStats.p95} ms</div>
                </div>
                <div className="bg-[#1B2330] p-2.5 rounded-lg border border-[#2B3442]/60">
                  <div className="text-[#64748B] text-[11px]">Hands Detection</div>
                  <div className="text-sm font-bold text-[#F3F4F6]">
                    {streamResult?.perf?.hands_ms ? `${streamResult.perf.hands_ms.toFixed(1)} ms` : '--'}
                  </div>
                </div>
                <div className="bg-[#1B2330] p-2.5 rounded-lg border border-[#2B3442]/60">
                  <div className="text-[#64748B] text-[11px]">Face Detection</div>
                  <div className="text-sm font-bold text-[#F3F4F6]">
                    {streamResult?.perf?.face_ms ? `${streamResult.perf.face_ms.toFixed(1)} ms` : '--'}
                  </div>
                </div>
                <div className="bg-[#1B2330] p-2.5 rounded-lg border border-[#2B3442]/60">
                  <div className="text-[#64748B] text-[11px]">ML Inference</div>
                  <div className="text-sm font-bold text-[#10B981]">
                    {streamResult?.perf?.infer_ms ? `${streamResult.perf.infer_ms.toFixed(1)} ms` : '--'}
                  </div>
                </div>
                <div className="bg-[#1B2330] p-2.5 rounded-lg border border-[#2B3442]/60">
                  <div className="text-[#64748B] text-[11px]">Backend Total</div>
                  <div className="text-sm font-bold text-[#F3F4F6]">
                    {streamResult?.perf?.backend_total_ms ? `${streamResult.perf.backend_total_ms.toFixed(1)} ms` : '--'}
                  </div>
                </div>
                <div className="bg-[#1B2330] p-2.5 rounded-lg border border-[#2B3442]/60">
                  <div className="text-[#64748B] text-[11px]">Stabilizer State</div>
                  <div className="text-sm font-bold text-[#38BDF8] capitalize">{state}</div>
                </div>
                <div className="bg-[#1B2330] p-2.5 rounded-lg border border-[#2B3442]/60">
                  <div className="text-[#64748B] text-[11px]">Dropped Frames</div>
                  <div className="text-sm font-bold text-[#94A3B8]">{streamResult?.dropped_frames || 0}</div>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Right Column: Recognition Card & Sentence Builder (5 cols) */}
        <div className="lg:col-span-5 space-y-5">
          {/* Real-time Recognition Sign Card */}
          <div className="bg-[#1B2330] border border-[#2B3442] rounded-xl p-6 shadow-lg relative overflow-hidden">
            <div className="flex items-center justify-between mb-4">
              <span className="text-xs font-semibold text-[#94A3B8] uppercase tracking-wider">
                Recognized Sign
              </span>
              <div className="flex items-center space-x-2">
                {isHolding ? (
                  <span className="px-2 py-0.5 rounded-full bg-[#10B981]/20 text-[#10B981] font-semibold text-xs border border-[#10B981]/30">
                    Holding Sign
                  </span>
                ) : state === 'candidate' ? (
                  <span className="px-2 py-0.5 rounded-full bg-[#38BDF8]/20 text-[#38BDF8] font-semibold text-xs border border-[#38BDF8]/30">
                    Stabilizing ({Math.round(stateProgress * 100)}%)
                  </span>
                ) : isHighConfidence ? (
                  <div className="flex items-center space-x-1 text-xs text-[#10B981] font-semibold">
                    <CheckCircle2 className="w-4 h-4" />
                    <span>Candidate</span>
                  </div>
                ) : null}
              </div>
            </div>

            {/* Arabic Word Display */}
            <div className="min-h-[110px] flex items-center justify-center bg-[#101419] border border-[#2B3442] rounded-xl p-4 mb-4">
              {recognizedWord ? (
                <span
                  className={`text-4xl sm:text-5xl font-bold font-arabic transition-transform duration-200 ${
                    isHolding
                      ? 'text-[#10B981] scale-105'
                      : isHighConfidence
                      ? 'text-[#38BDF8]'
                      : 'text-[#94A3B8]'
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

            {/* Confidence Bar & Stability Progress */}
            <div className="space-y-2">
              <div className="space-y-1">
                <div className="flex justify-between text-xs">
                  <span className="text-[#94A3B8]">Confidence</span>
                  <span className="font-mono font-bold text-[#F3F4F6]">
                    {(confidence * 100).toFixed(0)}%
                  </span>
                </div>
                <div className="w-full bg-[#101419] rounded-full h-2 overflow-hidden border border-[#2B3442]">
                  <div
                    className={`h-full transition-all duration-150 rounded-full ${
                      isHighConfidence ? 'bg-[#10B981]' : 'bg-[#2563EB]'
                    }`}
                    style={{ width: `${Math.min(100, confidence * 100)}%` }}
                  />
                </div>
              </div>

              {state === 'candidate' && (
                <div className="space-y-1 pt-1">
                  <div className="flex justify-between text-[11px] text-[#38BDF8]">
                    <span>Confirmation Progress</span>
                    <span className="font-mono">{Math.round(stateProgress * 100)}%</span>
                  </div>
                  <div className="w-full bg-[#101419] rounded-full h-1.5 overflow-hidden border border-[#2B3442]/60">
                    <div
                      className="h-full bg-[#38BDF8] transition-all duration-100 rounded-full"
                      style={{ width: `${Math.min(100, stateProgress * 100)}%` }}
                    />
                  </div>
                </div>
              )}
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
                  Confirmed signs will be appended here once held stably...
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
