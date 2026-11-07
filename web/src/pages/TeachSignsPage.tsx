import React, { useEffect, useRef, useState, useCallback } from 'react';
import {
  Plus,
  Edit2,
  Trash2,
  Search,
  CheckCircle2,
  AlertCircle,
  ArrowRight,
  Zap,
  Radio,
  X,
} from 'lucide-react';
import { CameraViewport } from '../components/CameraViewport';
import {
  fetchGestures,
  createGesture,
  updateGesture,
  deleteGesture,
  fetchGestureSamples,
  deleteGestureSample,
} from '../services/api';
import type { GestureClass, SampleInfo } from '../types';

interface TeachSignsPageProps {
  onNavigateToTrain: () => void;
}

export const TeachSignsPage: React.FC<TeachSignsPageProps> = ({ onNavigateToTrain }) => {
  const [gestures, setGestures] = useState<GestureClass[]>([]);
  const [selectedGesture, setSelectedGesture] = useState<GestureClass | null>(null);
  const [searchQuery, setSearchQuery] = useState<string>('');
  const [samplesOnly, setSamplesOnly] = useState<boolean>(false);
  const [isLoading, setIsLoading] = useState<boolean>(true);

  // Recording State
  const [isRecording, setIsRecording] = useState<boolean>(false);
  const [targetSampleCount, setTargetSampleCount] = useState<number>(10);
  const [currentRecordedCount, setCurrentRecordedCount] = useState<number>(0);
  const [sessionCount, setSessionCount] = useState<number>(0);
  const [feedbackMsg, setFeedbackMsg] = useState<{ text: string; type: 'success' | 'warn' } | null>(null);

  // Modal Dialogs State
  const [showAddModal, setShowAddModal] = useState<boolean>(false);
  const [newSignName, setNewSignName] = useState<string>('');
  const [newSignCategory, setNewSignCategory] = useState<string>('static');

  const [showRenameModal, setShowRenameModal] = useState<boolean>(false);
  const [renameValue, setRenameValue] = useState<string>('');

  const [showDeleteModal, setShowDeleteModal] = useState<boolean>(false);
  const [purgeSamples, setPurgeSamples] = useState<boolean>(false);

  const [inspectSamples, setInspectSamples] = useState<SampleInfo[] | null>(null);

  const captureWsRef = useRef<WebSocket | null>(null);

  // Load Gestures from backend
  const loadGestures = useCallback(async () => {
    try {
      setIsLoading(true);
      const data = await fetchGestures(false);
      setGestures(data);
      if (data.length > 0 && !selectedGesture) {
        setSelectedGesture(data[0]);
      } else if (selectedGesture) {
        const updated = data.find((g) => g.id === selectedGesture.id);
        if (updated) setSelectedGesture(updated);
      }
    } catch (err) {
      console.error('Failed to load gestures:', err);
    } finally {
      setIsLoading(false);
    }
  }, [selectedGesture]);

  useEffect(() => {
    loadGestures();
  }, []);

  // Connect Capture WebSocket
  useEffect(() => {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const host = window.location.host;
    const ws = new WebSocket(`${protocol}//${host}/ws/capture`);
    captureWsRef.current = ws;

    ws.onmessage = (event) => {
      try {
        const msg = JSON.parse(event.data);
        if (msg.status === 'sample_saved') {
          setCurrentRecordedCount(msg.current_count);
          setSessionCount((prev) => prev + 1);
          setFeedbackMsg({
            text: `Saved sample ${msg.current_count}/${msg.target_count} for "${msg.class_name}"`,
            type: 'success',
          });
        } else if (msg.status === 'complete') {
          setCurrentRecordedCount(msg.target_count);
          setIsRecording(false);
          setFeedbackMsg({
            text: `Successfully completed recording ${msg.target_count} samples for "${msg.class_name}"!`,
            type: 'success',
          });
          loadGestures();
        } else if (msg.status === 'invalid_frame') {
          setFeedbackMsg({
            text: 'Keep hands clearly visible in front of camera',
            type: 'warn',
          });
        }
      } catch (err) {
        console.error('Capture message parse error:', err);
      }
    };

    return () => {
      ws.close();
      captureWsRef.current = null;
    };
  }, [loadGestures]);

  // Handle Recording Trigger
  const startRecordingSession = (count: number = targetSampleCount) => {
    if (!selectedGesture || !captureWsRef.current) return;
    setIsRecording(true);
    setCurrentRecordedCount(0);
    setFeedbackMsg({ text: `Recording ${count} samples for "${selectedGesture.name}"...`, type: 'success' });

    // Send capture setup
    captureWsRef.current.send(
      JSON.stringify({
        class_name: selectedGesture.name,
        target_count: count,
        category: selectedGesture.category,
        command: 'reset',
      })
    );
  };

  // Keyboard shortcut: Spacebar records
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.code === 'Space' && !['INPUT', 'TEXTAREA'].includes((e.target as HTMLElement)?.tagName)) {
        e.preventDefault();
        if (!isRecording && selectedGesture) {
          startRecordingSession();
        }
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [isRecording, selectedGesture]);

  // Stream captured frame to capture WebSocket
  const handleFrameCapture = useCallback(
    (blob: Blob) => {
      if (isRecording && captureWsRef.current && captureWsRef.current.readyState === WebSocket.OPEN) {
        blob.arrayBuffer().then((buf) => {
          if (captureWsRef.current && captureWsRef.current.readyState === WebSocket.OPEN) {
            captureWsRef.current.send(buf);
          }
        });
      }
    },
    [isRecording]
  );

  // Gesture Management Handlers
  const handleCreateSign = async () => {
    if (!newSignName.trim()) return;
    try {
      const created = await createGesture({
        name: newSignName.trim(),
        category: newSignCategory,
      });
      setShowAddModal(false);
      setNewSignName('');
      await loadGestures();
      setSelectedGesture(created);
    } catch (err: any) {
      alert(err.message || 'Failed to add sign');
    }
  };

  const handleRenameSign = async () => {
    if (!selectedGesture || !renameValue.trim()) return;
    try {
      await updateGesture(selectedGesture.id, { name: renameValue.trim() });
      setShowRenameModal(false);
      setRenameValue('');
      await loadGestures();
    } catch (err: any) {
      alert(err.message || 'Failed to rename sign');
    }
  };

  const handleDeleteSign = async () => {
    if (!selectedGesture) return;
    try {
      await deleteGesture(selectedGesture.id, purgeSamples);
      setShowDeleteModal(false);
      setSelectedGesture(null);
      await loadGestures();
    } catch (err: any) {
      alert(err.message || 'Failed to delete sign');
    }
  };

  const handleOpenInspect = async (g: GestureClass) => {
    try {
      const list = await fetchGestureSamples(g.id);
      setInspectSamples(list);
    } catch (err) {
      console.error('Failed to load samples:', err);
    }
  };

  const handleDeleteSample = async (sampleId: string) => {
    if (!selectedGesture) return;
    try {
      await deleteGestureSample(selectedGesture.id, sampleId);
      setInspectSamples((prev) => prev?.filter((s) => s.sample_id !== sampleId) || null);
      await loadGestures();
    } catch (err) {
      console.error('Failed to delete sample:', err);
    }
  };

  // Filter Gestures
  const filteredGestures = gestures.filter((g) => {
    const matchesSearch = g.name.toLowerCase().includes(searchQuery.toLowerCase());
    const matchesSamples = samplesOnly ? g.sample_count > 0 : true;
    return matchesSearch && matchesSamples;
  });

  return (
    <div className="space-y-6">
      {/* 5-Step Workflow Banner */}
      <div className="bg-[#151B24] border border-[#2B3442] rounded-xl p-4 shadow-md">
        <div className="flex items-center justify-between text-xs text-[#94A3B8] font-semibold overflow-x-auto gap-4">
          <div className="flex items-center space-x-2 text-[#14B8A6]">
            <span className="w-5 h-5 rounded-full bg-[#14B8A6] text-[#101419] flex items-center justify-center font-bold">1</span>
            <span>Select Sign</span>
          </div>
          <span>→</span>
          <div className="flex items-center space-x-2 text-[#F3F4F6]">
            <span className="w-5 h-5 rounded-full bg-[#2563EB] text-white flex items-center justify-center font-bold">2</span>
            <span>Camera Check</span>
          </div>
          <span>→</span>
          <div className="flex items-center space-x-2 text-[#F3F4F6]">
            <span className="w-5 h-5 rounded-full bg-[#2563EB] text-white flex items-center justify-center font-bold">3</span>
            <span>Record (Space)</span>
          </div>
          <span>→</span>
          <div className="flex items-center space-x-2 text-[#F3F4F6]">
            <span className="w-5 h-5 rounded-full bg-[#2563EB] text-white flex items-center justify-center font-bold">4</span>
            <span>Review Samples</span>
          </div>
          <span>→</span>
          <div className="flex items-center space-x-2 text-[#F3F4F6]">
            <span className="w-5 h-5 rounded-full bg-[#10B981] text-white flex items-center justify-center font-bold">5</span>
            <span>Train Model</span>
          </div>
        </div>
      </div>

      {/* Main Two-Column Layout */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
        {/* Left Column: Camera + Recording Controls (7 cols) */}
        <div className="lg:col-span-7 space-y-4">
          <CameraViewport
            onFrameCapture={handleFrameCapture}
            isStreaming={isRecording}
          />

          {/* Multimodal Tracking Chips */}
          <div className="grid grid-cols-4 gap-2 text-xs">
            <div className="bg-[#1B2330] border border-[#2B3442] p-2.5 rounded-lg text-center">
              <span className="text-[#94A3B8] block text-[10px]">Left Hand</span>
              <span className="text-[#14B8A6] font-semibold">Active</span>
            </div>
            <div className="bg-[#1B2330] border border-[#2B3442] p-2.5 rounded-lg text-center">
              <span className="text-[#94A3B8] block text-[10px]">Right Hand</span>
              <span className="text-[#14B8A6] font-semibold">Active</span>
            </div>
            <div className="bg-[#1B2330] border border-[#2B3442] p-2.5 rounded-lg text-center">
              <span className="text-[#94A3B8] block text-[10px]">Face Mesh</span>
              <span className="text-[#14B8A6] font-semibold">24 pts</span>
            </div>
            <div className="bg-[#1B2330] border border-[#2B3442] p-2.5 rounded-lg text-center">
              <span className="text-[#94A3B8] block text-[10px]">Body Pose</span>
              <span className="text-[#94A3B8] font-semibold">Optional</span>
            </div>
          </div>

          {/* Recording Action Card */}
          <div className="bg-[#1B2330] border border-[#2B3442] rounded-xl p-5 shadow-lg space-y-4">
            <div className="flex items-center justify-between">
              <div>
                <span className="text-xs text-[#94A3B8] block">Current Sign Selected:</span>
                <span className="text-2xl font-bold font-arabic text-[#14B8A6]">
                  {selectedGesture ? selectedGesture.name : 'No sign selected'}
                </span>
              </div>
              <div className="text-right">
                <span className="text-xs text-[#94A3B8] block">Saved Samples:</span>
                <span className="text-lg font-mono font-bold text-[#F3F4F6]">
                  {selectedGesture ? selectedGesture.sample_count : 0}
                </span>
              </div>
            </div>

            {/* Recording Controls */}
            <div className="flex flex-wrap items-center gap-3">
              <button
                onClick={() => startRecordingSession(targetSampleCount)}
                disabled={!selectedGesture || isRecording}
                className="flex-1 py-3 px-4 bg-[#2563EB] hover:bg-[#3B82F6] disabled:bg-[#151B24] disabled:text-[#64748B] text-white rounded-xl text-sm font-bold flex items-center justify-center space-x-2 transition-all shadow-md"
              >
                <Radio className={`w-4 h-4 ${isRecording ? 'animate-pulse text-[#EF4444]' : ''}`} />
                <span>
                  {isRecording
                    ? `Recording (${currentRecordedCount}/${targetSampleCount})...`
                    : '● Record Sample (Space)'}
                </span>
              </button>

              <button
                onClick={() => startRecordingSession(5)}
                disabled={!selectedGesture || isRecording}
                className="py-3 px-4 bg-[#151B24] hover:bg-[#222C3D] border border-[#2B3442] text-[#14B8A6] rounded-xl text-xs font-semibold flex items-center space-x-1.5 transition-colors"
              >
                <Zap className="w-4 h-4" />
                <span>Burst (5)</span>
              </button>

              <select
                value={targetSampleCount}
                onChange={(e) => setTargetSampleCount(parseInt(e.target.value))}
                disabled={isRecording}
                className="bg-[#151B24] border border-[#2B3442] text-[#F3F4F6] text-xs rounded-xl px-3 py-3"
              >
                <option value={5}>5 Samples</option>
                <option value={10}>10 Samples</option>
                <option value={15}>15 Samples</option>
                <option value={25}>25 Samples</option>
              </select>
            </div>

            {/* Recording Feedback Alert */}
            {feedbackMsg && (
              <div
                className={`p-3 rounded-lg text-xs flex items-center space-x-2 ${
                  feedbackMsg.type === 'success'
                    ? 'bg-[#10B981]/15 text-[#10B981] border border-[#10B981]/30'
                    : 'bg-[#F59E0B]/15 text-[#F59E0B] border border-[#F59E0B]/30'
                }`}
              >
                {feedbackMsg.type === 'success' ? (
                  <CheckCircle2 className="w-4 h-4 shrink-0" />
                ) : (
                  <AlertCircle className="w-4 h-4 shrink-0" />
                )}
                <span>{feedbackMsg.text}</span>
              </div>
            )}

            <div className="flex justify-between items-center text-[11px] text-[#94A3B8] pt-1">
              <span>Session captures: {sessionCount}</span>
              <span>Feature schema: v2_multimodal (206 dims)</span>
            </div>
          </div>
        </div>

        {/* Right Column: Gesture Classes Vocabulary & Actions (5 cols) */}
        <div className="lg:col-span-5 space-y-4">
          <div className="bg-[#1B2330] border border-[#2B3442] rounded-xl p-5 shadow-lg space-y-4">
            <div className="flex items-center justify-between">
              <div>
                <h2 className="text-sm font-bold text-[#F3F4F6]">Gesture Vocabulary</h2>
                <p className="text-xs text-[#94A3B8]">Select sign to record or manage</p>
              </div>
              <button
                onClick={() => setShowAddModal(true)}
                className="px-3 py-1.5 bg-[#2563EB] hover:bg-[#3B82F6] text-white rounded-lg text-xs font-semibold flex items-center space-x-1.5 transition-colors"
              >
                <Plus className="w-3.5 h-3.5" />
                <span>Add Sign</span>
              </button>
            </div>

            {/* Search and Filter */}
            <div className="space-y-2">
              <div className="relative">
                <Search className="w-4 h-4 text-[#94A3B8] absolute left-3 top-2.5" />
                <input
                  type="text"
                  placeholder="Search Arabic signs..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  className="w-full bg-[#151B24] border border-[#2B3442] rounded-lg pl-9 pr-3 py-2 text-xs text-[#F3F4F6] placeholder-[#64748B] focus:outline-none focus:border-[#2563EB]"
                />
              </div>

              <label className="flex items-center space-x-2 text-xs text-[#94A3B8] cursor-pointer">
                <input
                  type="checkbox"
                  checked={samplesOnly}
                  onChange={(e) => setSamplesOnly(e.target.checked)}
                  className="rounded accent-[#2563EB]"
                />
                <span>Signs with samples first</span>
              </label>
            </div>

            {/* Gesture List */}
            <div className="max-h-[340px] overflow-y-auto space-y-1.5 pr-1">
              {filteredGestures.map((g) => {
                const isSelected = selectedGesture?.id === g.id;
                const isReady = g.sample_count >= 15;
                return (
                  <div
                    key={g.id}
                    onClick={() => setSelectedGesture(g)}
                    className={`p-3 rounded-lg border cursor-pointer transition-all flex items-center justify-between ${
                      isSelected
                        ? 'bg-[#151B24] border-[#14B8A6] shadow-sm'
                        : 'bg-[#151B24]/50 border-[#2B3442] hover:bg-[#151B24]'
                    }`}
                  >
                    <div className="flex items-center space-x-3">
                      <span className="text-base font-bold font-arabic text-[#F3F4F6]">
                        {g.name}
                      </span>
                      <span className="text-[10px] px-2 py-0.5 rounded bg-[#101419] text-[#94A3B8]">
                        {g.category}
                      </span>
                    </div>

                    <div className="flex items-center space-x-2">
                      <span
                        className={`text-xs px-2 py-0.5 rounded-full font-mono font-bold ${
                          isReady
                            ? 'bg-[#10B981]/20 text-[#10B981]'
                            : g.sample_count > 0
                            ? 'bg-[#2563EB]/20 text-[#3B82F6]'
                            : 'bg-[#2B3442] text-[#64748B]'
                        }`}
                      >
                        {g.sample_count} {g.sample_count === 1 ? 'sample' : 'samples'}
                      </span>
                    </div>
                  </div>
                );
              })}

              {filteredGestures.length === 0 && !isLoading && (
                <div className="p-6 text-center text-xs text-[#64748B]">
                  No matching gesture classes found.
                </div>
              )}
            </div>

            {/* Selected Sign Actions */}
            {selectedGesture && (
              <div className="pt-3 border-t border-[#2B3442] flex items-center justify-between text-xs">
                <div className="flex items-center space-x-2">
                  <button
                    onClick={() => {
                      setRenameValue(selectedGesture.name);
                      setShowRenameModal(true);
                    }}
                    className="p-1.5 text-[#94A3B8] hover:text-[#F3F4F6] hover:bg-[#151B24] rounded border border-[#2B3442]"
                    title="Rename Sign"
                  >
                    <Edit2 className="w-3.5 h-3.5" />
                  </button>

                  <button
                    onClick={() => setShowDeleteModal(true)}
                    className="p-1.5 text-[#94A3B8] hover:text-[#EF4444] hover:bg-[#EF4444]/10 rounded border border-[#2B3442]"
                    title="Delete Sign"
                  >
                    <Trash2 className="w-3.5 h-3.5" />
                  </button>
                </div>

                <button
                  onClick={() => handleOpenInspect(selectedGesture)}
                  className="px-3 py-1.5 bg-[#151B24] hover:bg-[#222C3D] text-[#14B8A6] rounded border border-[#2B3442] font-semibold"
                >
                  Inspect Samples
                </button>
              </div>
            )}
          </div>

          {/* Step 5 CTA Card */}
          <div className="bg-gradient-to-r from-[#1B2330] to-[#151B24] border border-[#14B8A6]/30 rounded-xl p-4 flex items-center justify-between shadow-lg">
            <div>
              <span className="text-xs font-bold text-[#F3F4F6] block">Ready to train?</span>
              <span className="text-[11px] text-[#94A3B8]">
                Train your model once you have recorded samples
              </span>
            </div>
            <button
              onClick={onNavigateToTrain}
              className="px-4 py-2 bg-[#14B8A6] hover:bg-[#0D9488] text-[#101419] font-bold rounded-lg text-xs flex items-center space-x-1.5 transition-colors shadow-sm"
            >
              <span>Train Model Now</span>
              <ArrowRight className="w-4 h-4" />
            </button>
          </div>
        </div>
      </div>

      {/* Add Sign Modal */}
      {showAddModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/70 backdrop-blur-sm">
          <div className="bg-[#151B24] border border-[#2B3442] rounded-xl p-6 w-full max-w-md space-y-4">
            <h3 className="text-base font-bold text-[#F3F4F6]">Add New Arabic Sign</h3>
            <div className="space-y-3">
              <div>
                <label className="text-xs text-[#94A3B8] block mb-1">Arabic Name</label>
                <input
                  type="text"
                  placeholder="مثال: شكرا، مع السلامة"
                  value={newSignName}
                  onChange={(e) => setNewSignName(e.target.value)}
                  className="w-full bg-[#1B2330] border border-[#2B3442] rounded-lg p-2.5 text-sm font-arabic text-[#F3F4F6] text-right focus:outline-none focus:border-[#2563EB]"
                />
              </div>

              <div>
                <label className="text-xs text-[#94A3B8] block mb-1">Category</label>
                <select
                  value={newSignCategory}
                  onChange={(e) => setNewSignCategory(e.target.value)}
                  className="w-full bg-[#1B2330] border border-[#2B3442] rounded-lg p-2.5 text-xs text-[#F3F4F6]"
                >
                  <option value="static">Static Gesture (Hands + Face)</option>
                  <option value="dynamic">Dynamic Sequence (Motion)</option>
                  <option value="alphabet">Arabic Alphabet Letter</option>
                </select>
              </div>
            </div>

            <div className="flex justify-end space-x-2 pt-2">
              <button
                onClick={() => setShowAddModal(false)}
                className="px-4 py-2 text-xs text-[#94A3B8] hover:text-[#F3F4F6]"
              >
                Cancel
              </button>
              <button
                onClick={handleCreateSign}
                disabled={!newSignName.trim()}
                className="px-4 py-2 bg-[#2563EB] hover:bg-[#3B82F6] disabled:opacity-50 text-white rounded-lg text-xs font-semibold"
              >
                Create Sign
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Rename Sign Modal */}
      {showRenameModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/70 backdrop-blur-sm">
          <div className="bg-[#151B24] border border-[#2B3442] rounded-xl p-6 w-full max-w-md space-y-4">
            <h3 className="text-base font-bold text-[#F3F4F6]">Rename Sign</h3>
            <div>
              <label className="text-xs text-[#94A3B8] block mb-1">New Arabic Name</label>
              <input
                type="text"
                value={renameValue}
                onChange={(e) => setRenameValue(e.target.value)}
                className="w-full bg-[#1B2330] border border-[#2B3442] rounded-lg p-2.5 text-sm font-arabic text-[#F3F4F6] text-right focus:outline-none focus:border-[#2563EB]"
              />
            </div>
            <div className="flex justify-end space-x-2 pt-2">
              <button
                onClick={() => setShowRenameModal(false)}
                className="px-4 py-2 text-xs text-[#94A3B8] hover:text-[#F3F4F6]"
              >
                Cancel
              </button>
              <button
                onClick={handleRenameSign}
                disabled={!renameValue.trim()}
                className="px-4 py-2 bg-[#2563EB] hover:bg-[#3B82F6] disabled:opacity-50 text-white rounded-lg text-xs font-semibold"
              >
                Save
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Delete Sign Modal */}
      {showDeleteModal && selectedGesture && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/70 backdrop-blur-sm">
          <div className="bg-[#151B24] border border-[#2B3442] rounded-xl p-6 w-full max-w-md space-y-4">
            <h3 className="text-base font-bold text-[#EF4444]">Delete Sign</h3>
            <p className="text-xs text-[#94A3B8]">
              Are you sure you want to delete <span className="font-bold text-[#F3F4F6]">"{selectedGesture.name}"</span>?
            </p>
            <label className="flex items-center space-x-2 text-xs text-[#EF4444] cursor-pointer">
              <input
                type="checkbox"
                checked={purgeSamples}
                onChange={(e) => setPurgeSamples(e.target.checked)}
                className="rounded accent-[#EF4444]"
              />
              <span>Also permanently delete all {selectedGesture.sample_count} recorded samples on disk</span>
            </label>
            <div className="flex justify-end space-x-2 pt-2">
              <button
                onClick={() => setShowDeleteModal(false)}
                className="px-4 py-2 text-xs text-[#94A3B8] hover:text-[#F3F4F6]"
              >
                Cancel
              </button>
              <button
                onClick={handleDeleteSign}
                className="px-4 py-2 bg-[#EF4444] hover:bg-[#DC2626] text-white rounded-lg text-xs font-semibold"
              >
                Confirm Delete
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Inspect Samples Modal */}
      {inspectSamples && selectedGesture && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/70 backdrop-blur-sm">
          <div className="bg-[#151B24] border border-[#2B3442] rounded-xl p-6 w-full max-w-xl space-y-4">
            <div className="flex items-center justify-between border-b border-[#2B3442] pb-3">
              <div>
                <h3 className="text-base font-bold text-[#F3F4F6]">
                  Inspect Samples: <span className="font-arabic text-[#14B8A6]">{selectedGesture.name}</span>
                </h3>
                <p className="text-xs text-[#94A3B8]">
                  Total: {inspectSamples.length} recorded multimodal samples
                </p>
              </div>
              <button
                onClick={() => setInspectSamples(null)}
                className="p-1 text-[#94A3B8] hover:text-[#F3F4F6]"
              >
                <X className="w-5 h-5" />
              </button>
            </div>

            <div className="max-h-[300px] overflow-y-auto space-y-2 pr-1">
              {inspectSamples.map((s) => (
                <div
                  key={s.sample_id}
                  className="bg-[#1B2330] border border-[#2B3442] p-3 rounded-lg flex items-center justify-between text-xs"
                >
                  <div>
                    <span className="font-mono font-bold text-[#F3F4F6] block">
                      ID: {s.sample_id}
                    </span>
                    <span className="text-[10px] text-[#94A3B8]">
                      {new Date(s.timestamp * 1000).toLocaleString()} • {s.schema_version}
                    </span>
                  </div>

                  <div className="flex items-center space-x-3">
                    <span className="text-[10px] px-2 py-0.5 rounded bg-[#101419] text-[#14B8A6]">
                      {s.has_face ? 'Hands + Face' : 'Hands Only'}
                    </span>
                    <button
                      onClick={() => handleDeleteSample(s.sample_id)}
                      className="p-1.5 text-[#EF4444] hover:bg-[#EF4444]/15 rounded transition-colors"
                      title="Delete this sample"
                    >
                      <Trash2 className="w-4 h-4" />
                    </button>
                  </div>
                </div>
              ))}

              {inspectSamples.length === 0 && (
                <div className="p-6 text-center text-xs text-[#64748B]">
                  No samples recorded for this sign yet.
                </div>
              )}
            </div>

            <div className="flex justify-end pt-2 border-t border-[#2B3442]">
              <button
                onClick={() => setInspectSamples(null)}
                className="px-4 py-2 bg-[#2563EB] text-white rounded-lg text-xs font-semibold"
              >
                Close
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
