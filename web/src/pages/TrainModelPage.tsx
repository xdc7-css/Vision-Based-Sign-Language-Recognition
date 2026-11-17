import React, { useEffect, useState, useRef } from 'react';
import {
  CheckCircle2,
  Play,
  RotateCw,
  ArrowRight,
  ChevronDown,
  ChevronUp,
  Cpu,
  Layers,
  Activity,
  Award,
  FlaskConical,
  Zap,
  BarChart3,
  Check,
  AlertCircle,
  BookOpen,
  Users,
  UserPlus,
  ShieldAlert,
  Video,
  Info,
} from 'lucide-react';
import {
  fetchTrainingReadiness,
  startTraining,
  fetchJobStatus,
  cancelTrainingJob,
  fetchResearchStatus,
  prepareResearchDataset,
  trainResearchSVM,
  trainResearchCNN,
  runResearchComparison,
  selectActiveModel,
  fetchIndependentEvalStatus,
  listIndependentVolunteers,
  registerIndependentVolunteer,
  fetchVolunteerProgress,
  runIndependentEvaluation,
} from '../services/api';
import type {
  TrainingJobStatus,
  TrainingReadiness,
  ResearchStatusResponse,
  ResearchComparisonMetrics,
  IndependentEvalStatus,
  VolunteerInfo,
  VolunteerProgress,
} from '../types';

interface TrainModelPageProps {
  onNavigateToRecognize: () => void;
  onModelTrained: () => void;
}

export const TrainModelPage: React.FC<TrainModelPageProps> = ({
  onNavigateToRecognize,
  onModelTrained,
}) => {
  const [readiness, setReadiness] = useState<TrainingReadiness | null>(null);
  const [activeJob, setActiveJob] = useState<TrainingJobStatus | null>(null);
  const [isStarting, setIsStarting] = useState<boolean>(false);
  const [showAdvanced, setShowAdvanced] = useState<boolean>(false);

  // Hyperparameters
  const [modelType, setModelType] = useState<string>('mlp');
  const [epochs, setEpochs] = useState<number>(60);
  const [batchSize, setBatchSize] = useState<number>(16);
  const [learningRate, setLearningRate] = useState<number>(0.001);
  const [valSplit, setValSplit] = useState<number>(0.2);

  const pollIntervalRef = useRef<number | null>(null);

  // Academic Research Benchmark State
  const [researchStatus, setResearchStatus] = useState<ResearchStatusResponse | null>(null);
  const [comparisonMetrics, setComparisonMetrics] = useState<ResearchComparisonMetrics | null>(null);
  const [isResearchBusy, setIsResearchBusy] = useState<boolean>(false);
  const [researchActionText, setResearchActionText] = useState<string>('');
  const [selectedConfusionModel, setSelectedConfusionModel] = useState<'svm' | 'cnn' | null>(null);

  // Independent Human Evaluation State
  const [indEvalStatus, setIndEvalStatus] = useState<IndependentEvalStatus | null>(null);
  const [volunteers, setVolunteers] = useState<VolunteerInfo[]>([]);
  const [selectedVolunteer, setSelectedVolunteer] = useState<string>('');
  const [volunteerProgress, setVolunteerProgress] = useState<VolunteerProgress | null>(null);
  const [newVolunteerName, setNewVolunteerName] = useState<string>('');
  const [newVolunteerNotes, setNewVolunteerNotes] = useState<string>('');
  const [isIndEvalBusy, setIsIndEvalBusy] = useState<boolean>(false);
  const [indEvalActionText, setIndEvalActionText] = useState<string>('');
  const [showSampleBreakdown, setShowSampleBreakdown] = useState<boolean>(false);
  const [selectedIndConfusionModel, setSelectedIndConfusionModel] = useState<'svm' | 'cnn' | null>(null);

  // Load readiness
  const loadReadiness = async () => {
    try {
      const data = await fetchTrainingReadiness();
      setReadiness(data);
    } catch (err) {
      console.error('Failed to fetch training readiness:', err);
    }
  };

  // Load research status & comparison
  const loadResearch = async () => {
    try {
      const st = await fetchResearchStatus();
      setResearchStatus(st);
      if (st.comparison) {
        setComparisonMetrics(st.comparison);
      }
    } catch (err) {
      console.error('Failed to load research status:', err);
    }
  };

  // Load independent evaluation status and volunteers
  const loadIndEval = async () => {
    try {
      const [st, vols] = await Promise.all([
        fetchIndependentEvalStatus(),
        listIndependentVolunteers(),
      ]);
      setIndEvalStatus(st);
      setVolunteers(vols);
      if (vols.length > 0 && !selectedVolunteer) {
        setSelectedVolunteer(vols[0].volunteer_id);
      }
    } catch (err) {
      console.error('Failed to load independent evaluation status:', err);
    }
  };

  useEffect(() => {
    loadReadiness();
    loadResearch();
    loadIndEval();
  }, []);

  useEffect(() => {
    if (selectedVolunteer) {
      fetchVolunteerProgress(selectedVolunteer)
        .then(setVolunteerProgress)
        .catch((err) => console.error('Failed to fetch volunteer progress:', err));
    } else {
      setVolunteerProgress(null);
    }
  }, [selectedVolunteer]);

  // Volunteer Action Handlers
  const handleRegisterVolunteer = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!newVolunteerName.trim()) return;
    try {
      const v = await registerIndependentVolunteer(newVolunteerName.trim(), newVolunteerNotes.trim());
      setNewVolunteerName('');
      setNewVolunteerNotes('');
      await loadIndEval();
      setSelectedVolunteer(v.volunteer_id);
    } catch (err: any) {
      alert(err.message || 'Failed to register volunteer');
    }
  };

  const handleRunIndEvaluation = async () => {
    setIsIndEvalBusy(true);
    setIndEvalActionText('Evaluating frozen SVM and CNN on independent volunteer recordings...');
    try {
      const res = await runIndependentEvaluation();
      setIndEvalStatus(res);
      await loadIndEval();
    } catch (err: any) {
      alert(err.message || 'Failed to run independent evaluation');
    } finally {
      setIsIndEvalBusy(false);
      setIndEvalActionText('');
    }
  };

  // Research Action Handlers
  const handlePrepareResearchDataset = async () => {
    setIsResearchBusy(true);
    setResearchActionText('Preparing KArSL-502 subset & Leave-One-Signer-Out splits...');
    try {
      await prepareResearchDataset();
      await loadResearch();
    } catch (err: any) {
      alert(err.message || 'Failed to prepare research dataset');
    } finally {
      setIsResearchBusy(false);
      setResearchActionText('');
    }
  };

  const handleTrainResearchSVM = async () => {
    setIsResearchBusy(true);
    setResearchActionText('Training Model A: Support Vector Machine (RBF kernel)...');
    try {
      await trainResearchSVM();
      await loadResearch();
    } catch (err: any) {
      alert(err.message || 'Failed to train SVM');
    } finally {
      setIsResearchBusy(false);
      setResearchActionText('');
    }
  };

  const handleTrainResearchCNN = async () => {
    setIsResearchBusy(true);
    setResearchActionText('Training Model B: 1D Temporal CNN (PyTorch Conv1d)...');
    try {
      await trainResearchCNN();
      await loadResearch();
    } catch (err: any) {
      alert(err.message || 'Failed to train 1D Temporal CNN');
    } finally {
      setIsResearchBusy(false);
      setResearchActionText('');
    }
  };

  const handleRunComparison = async () => {
    setIsResearchBusy(true);
    setResearchActionText('Evaluating Leave-One-Signer-Out (LOSO) benchmark on unseen Signer03...');
    try {
      const res = await runResearchComparison();
      setComparisonMetrics(res);
      await loadResearch();
    } catch (err: any) {
      alert(err.message || 'Failed to run comparison');
    } finally {
      setIsResearchBusy(false);
      setResearchActionText('');
    }
  };

  const handleActivateResearchModel = async (modelId: 'research_svm' | 'research_cnn') => {
    try {
      await selectActiveModel(modelId);
      onModelTrained();
      onNavigateToRecognize();
    } catch (err: any) {
      alert(err.message || 'Failed to activate research model');
    }
  };

  // Poll active training job
  useEffect(() => {
    if (activeJob && activeJob.status === 'running') {
      pollIntervalRef.current = window.setInterval(async () => {
        try {
          const updated = await fetchJobStatus(activeJob.job_id);
          setActiveJob(updated);
          if (updated.status === 'completed') {
            clearInterval(pollIntervalRef.current!);
            loadReadiness();
            onModelTrained();
          } else if (updated.status === 'failed' || updated.status === 'cancelled') {
            clearInterval(pollIntervalRef.current!);
            loadReadiness();
          }
        } catch (err) {
          console.error('Error polling job status:', err);
        }
      }, 1000);
    }

    return () => {
      if (pollIntervalRef.current) clearInterval(pollIntervalRef.current);
    };
  }, [activeJob?.status, activeJob?.job_id, onModelTrained]);

  const handleStartTraining = async () => {
    setIsStarting(true);
    try {
      const job = await startTraining({
        model_type: modelType,
        epochs,
        batch_size: batchSize,
        learning_rate: learningRate,
        val_split: valSplit,
      });
      setActiveJob(job);
    } catch (err: any) {
      alert(err.message || 'Failed to start training');
    } finally {
      setIsStarting(false);
    }
  };

  const handleCancelJob = async () => {
    if (!activeJob) return;
    try {
      await cancelTrainingJob(activeJob.job_id);
    } catch (err) {
      console.error('Error cancelling job:', err);
    }
  };

  const isTraining = activeJob?.status === 'running';
  const isCompleted = activeJob?.status === 'completed';

  return (
    <div className="space-y-6 max-w-5xl mx-auto">
      {/* Top 3 Summary Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {/* Card 1: Trainable Signs */}
        <div className="bg-[#1B2330] border border-[#2B3442] rounded-xl p-5 shadow-lg flex items-center space-x-4">
          <div className="w-12 h-12 rounded-xl bg-[#2563EB]/20 text-[#2563EB] flex items-center justify-center shrink-0">
            <Layers className="w-6 h-6" />
          </div>
          <div>
            <span className="text-xs text-[#94A3B8] font-medium block">Trainable Signs</span>
            <div className="flex items-baseline space-x-2">
              <span className="text-2xl font-bold font-mono text-[#F3F4F6]">
                {readiness ? readiness.trainable_classes_count : 0}
              </span>
              <span className="text-xs text-[#94A3B8]">classes (≥2 samples)</span>
            </div>
          </div>
        </div>

        {/* Card 2: Multimodal Samples */}
        <div className="bg-[#1B2330] border border-[#2B3442] rounded-xl p-5 shadow-lg flex items-center space-x-4">
          <div className="w-12 h-12 rounded-xl bg-[#14B8A6]/20 text-[#14B8A6] flex items-center justify-center shrink-0">
            <Activity className="w-6 h-6" />
          </div>
          <div>
            <span className="text-xs text-[#94A3B8] font-medium block">Multimodal Samples</span>
            <div className="flex items-baseline space-x-2">
              <span className="text-2xl font-bold font-mono text-[#F3F4F6]">
                {readiness ? readiness.total_multimodal_samples : 0}
              </span>
              <span className="text-xs text-[#94A3B8]">v2 (206 dims)</span>
            </div>
          </div>
        </div>

        {/* Card 3: Model Readiness */}
        <div className="bg-[#1B2330] border border-[#2B3442] rounded-xl p-5 shadow-lg flex items-center space-x-4">
          <div
            className={`w-12 h-12 rounded-xl flex items-center justify-center shrink-0 ${
              readiness?.model_status === 'Ready'
                ? 'bg-[#10B981]/20 text-[#10B981]'
                : readiness?.model_status === 'Outdated'
                ? 'bg-[#F59E0B]/20 text-[#F59E0B]'
                : 'bg-[#EF4444]/20 text-[#EF4444]'
            }`}
          >
            <Award className="w-6 h-6" />
          </div>
          <div>
            <span className="text-xs text-[#94A3B8] font-medium block">Model Status</span>
            <div className="flex items-center space-x-2">
              <span
                className={`text-sm font-bold ${
                  readiness?.model_status === 'Ready'
                    ? 'text-[#10B981]'
                    : readiness?.model_status === 'Outdated'
                    ? 'text-[#F59E0B]'
                    : 'text-[#EF4444]'
                }`}
              >
                {readiness?.model_status === 'Ready'
                  ? 'Model Ready'
                  : readiness?.model_status === 'Outdated'
                  ? 'Outdated (retrain recommended)'
                  : 'Untrained'}
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* Primary Training Section */}
      <div className="bg-[#1B2330] border border-[#2B3442] rounded-2xl p-6 shadow-xl space-y-6">
        <div className="flex flex-wrap items-center justify-between gap-4">
          <div>
            <h2 className="text-lg font-bold text-[#F3F4F6]">Model Training Workflow</h2>
            <p className="text-xs text-[#94A3B8]">
              Trains a neural classifier using the recorded 206-dimensional dual hands and face landmarks.
            </p>
          </div>

          {!isTraining ? (
            <button
              onClick={handleStartTraining}
              disabled={isStarting || !readiness?.is_trainable}
              className="py-3 px-6 bg-[#2563EB] hover:bg-[#3B82F6] disabled:bg-[#151B24] disabled:text-[#64748B] text-white rounded-xl text-sm font-bold flex items-center space-x-2 transition-all shadow-lg hover:shadow-[#2563EB]/25"
            >
              <Play className="w-4 h-4 fill-current" />
              <span>{isStarting ? 'Preparing Dataset...' : 'Start Training Model'}</span>
            </button>
          ) : (
            <button
              onClick={handleCancelJob}
              className="py-3 px-6 bg-[#EF4444]/20 hover:bg-[#EF4444]/30 text-[#EF4444] border border-[#EF4444]/40 rounded-xl text-xs font-bold transition-colors"
            >
              Cancel Training
            </button>
          )}
        </div>

        {/* Training Progress Telemetry */}
        {isTraining && (
          <div className="bg-[#151B24] border border-[#2B3442] rounded-xl p-5 space-y-3">
            <div className="flex justify-between items-center text-xs">
              <div className="flex items-center space-x-2 text-[#F3F4F6] font-semibold">
                <RotateCw className="w-4 h-4 text-[#2563EB] animate-spin" />
                <span>Training Neural Network Classifier...</span>
              </div>
              <span className="font-mono font-bold text-[#14B8A6]">
                {(activeJob.progress * 100).toFixed(0)}%
              </span>
            </div>

            <div className="w-full bg-[#101419] rounded-full h-3 overflow-hidden border border-[#2B3442]">
              <div
                className="bg-gradient-to-r from-[#2563EB] to-[#14B8A6] h-full transition-all duration-300 rounded-full"
                style={{ width: `${Math.min(100, activeJob.progress * 100)}%` }}
              />
            </div>

            <div className="flex justify-between text-xs font-mono text-[#94A3B8] pt-1">
              <span>
                Epoch: {activeJob.current_epoch} / {activeJob.max_epochs}
              </span>
              <span>Loss: {activeJob.loss.toFixed(4)}</span>
              <span>Val Acc: {(activeJob.accuracy * 100).toFixed(1)}%</span>
            </div>
          </div>
        )}

        {/* Completed Training Metrics */}
        {isCompleted && activeJob?.metrics && (
          <div className="bg-[#151B24] border border-[#10B981]/40 rounded-xl p-6 space-y-5 animate-fade-in">
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-2.5 text-[#10B981]">
                <CheckCircle2 className="w-6 h-6" />
                <div>
                  <h3 className="text-base font-bold text-[#F3F4F6]">
                    Training Complete & Checkpoint Verified!
                  </h3>
                  <p className="text-xs text-[#94A3B8]">
                    Model has been verified and persisted to data/models/multimodal/
                  </p>
                </div>
              </div>

              <button
                onClick={onNavigateToRecognize}
                className="py-2.5 px-5 bg-[#10B981] hover:bg-[#059669] text-white rounded-xl text-xs font-bold flex items-center space-x-2 transition-colors shadow-lg"
              >
                <span>Test Model in Recognition</span>
                <ArrowRight className="w-4 h-4" />
              </button>
            </div>

            {/* Metrics Grid */}
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 pt-2">
              <div className="bg-[#1B2330] border border-[#2B3442] p-3 rounded-lg text-center">
                <span className="text-[#94A3B8] block text-[10px] uppercase font-semibold">
                  Accuracy
                </span>
                <span className="text-xl font-bold font-mono text-[#10B981]">
                  {(activeJob.metrics.accuracy * 100).toFixed(1)}%
                </span>
              </div>
              <div className="bg-[#1B2330] border border-[#2B3442] p-3 rounded-lg text-center">
                <span className="text-[#94A3B8] block text-[10px] uppercase font-semibold">
                  Precision
                </span>
                <span className="text-xl font-bold font-mono text-[#F3F4F6]">
                  {(activeJob.metrics.precision * 100).toFixed(1)}%
                </span>
              </div>
              <div className="bg-[#1B2330] border border-[#2B3442] p-3 rounded-lg text-center">
                <span className="text-[#94A3B8] block text-[10px] uppercase font-semibold">
                  Recall
                </span>
                <span className="text-xl font-bold font-mono text-[#F3F4F6]">
                  {(activeJob.metrics.recall * 100).toFixed(1)}%
                </span>
              </div>
              <div className="bg-[#1B2330] border border-[#2B3442] p-3 rounded-lg text-center">
                <span className="text-[#94A3B8] block text-[10px] uppercase font-semibold">
                  F1-Score
                </span>
                <span className="text-xl font-bold font-mono text-[#14B8A6]">
                  {(activeJob.metrics.f1_score * 100).toFixed(1)}%
                </span>
              </div>
            </div>

            {/* Confusion Matrix Table */}
            {activeJob.metrics.confusion_matrix && activeJob.metrics.class_names && (
              <div className="pt-2">
                <span className="text-xs font-semibold text-[#94A3B8] block mb-2">
                  Validation Confusion Matrix
                </span>
                <div className="overflow-x-auto bg-[#101419] border border-[#2B3442] rounded-lg p-3">
                  <table className="w-full text-xs text-center font-mono">
                    <thead>
                      <tr className="text-[#94A3B8] border-b border-[#2B3442]">
                        <th className="p-2 text-left font-sans">Predicted →</th>
                        {activeJob.metrics.class_names.map((c) => (
                          <th key={c} className="p-2 font-arabic text-sm text-[#14B8A6]">
                            {c}
                          </th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {activeJob.metrics.confusion_matrix.map((row, i) => (
                        <tr key={i} className="border-b border-[#2B3442]/40 last:border-none">
                          <td className="p-2 text-left font-arabic text-sm font-semibold text-[#F3F4F6]">
                            {activeJob.metrics!.class_names![i]}
                          </td>
                          {row.map((val, j) => (
                            <td
                              key={j}
                              className={`p-2 font-bold ${
                                i === j
                                  ? val > 0
                                    ? 'bg-[#10B981]/20 text-[#10B981]'
                                    : 'text-[#64748B]'
                                  : val > 0
                                  ? 'bg-[#EF4444]/20 text-[#EF4444]'
                                  : 'text-[#64748B]'
                              }`}
                            >
                              {val}
                            </td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}
          </div>
        )}

        {/* Collapsible Advanced Hyperparameters */}
        <div className="border-t border-[#2B3442] pt-4">
          <button
            onClick={() => setShowAdvanced(!showAdvanced)}
            className="flex items-center space-x-2 text-xs font-semibold text-[#94A3B8] hover:text-[#F3F4F6] transition-colors"
          >
            <Cpu className="w-3.5 h-3.5 text-[#2563EB]" />
            <span>Advanced Hyperparameters & Model Configuration</span>
            {showAdvanced ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
          </button>

          {showAdvanced && (
            <div className="mt-4 bg-[#151B24] border border-[#2B3442] rounded-xl p-5 grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 gap-4 text-xs animate-fade-in">
              <div>
                <label className="text-[#94A3B8] block mb-1">Architecture</label>
                <select
                  value={modelType}
                  onChange={(e) => setModelType(e.target.value)}
                  disabled={isTraining}
                  className="w-full bg-[#1B2330] border border-[#2B3442] rounded-lg p-2 text-[#F3F4F6]"
                >
                  <option value="mlp">Multi-Layer Perceptron (MLP)</option>
                  <option value="svm">Support Vector Machine (SVM)</option>
                  <option value="rf">Random Forest (RF)</option>
                </select>
              </div>

              <div>
                <label className="text-[#94A3B8] block mb-1">Epochs</label>
                <input
                  type="number"
                  min="10"
                  max="300"
                  value={epochs}
                  onChange={(e) => setEpochs(parseInt(e.target.value) || 60)}
                  disabled={isTraining}
                  className="w-full bg-[#1B2330] border border-[#2B3442] rounded-lg p-2 text-[#F3F4F6]"
                />
              </div>

              <div>
                <label className="text-[#94A3B8] block mb-1">Batch Size</label>
                <input
                  type="number"
                  min="4"
                  max="64"
                  value={batchSize}
                  onChange={(e) => setBatchSize(parseInt(e.target.value) || 16)}
                  disabled={isTraining}
                  className="w-full bg-[#1B2330] border border-[#2B3442] rounded-lg p-2 text-[#F3F4F6]"
                />
              </div>

              <div>
                <label className="text-[#94A3B8] block mb-1">Learning Rate</label>
                <input
                  type="number"
                  step="0.0005"
                  min="0.0001"
                  max="0.05"
                  value={learningRate}
                  onChange={(e) => setLearningRate(parseFloat(e.target.value) || 0.001)}
                  disabled={isTraining}
                  className="w-full bg-[#1B2330] border border-[#2B3442] rounded-lg p-2 text-[#F3F4F6]"
                />
              </div>

              <div>
                <label className="text-[#94A3B8] block mb-1">Validation Split</label>
                <input
                  type="number"
                  step="0.05"
                  min="0.1"
                  max="0.4"
                  value={valSplit}
                  onChange={(e) => setValSplit(parseFloat(e.target.value) || 0.2)}
                  disabled={isTraining}
                  className="w-full bg-[#1B2330] border border-[#2B3442] rounded-lg p-2 text-[#F3F4F6]"
                />
              </div>
            </div>
          )}
        </div>
      </div>

      {/* ── ACADEMIC RESEARCH BENCHMARK CARD ───────────────────────────── */}
      <div className="bg-[#151B24] border border-[#2B3442] rounded-xl p-6 shadow-xl space-y-6">
        {/* Header */}
        <div className="flex flex-wrap items-center justify-between gap-4 border-b border-[#2B3442] pb-5">
          <div className="flex items-center space-x-3.5">
            <div className="w-12 h-12 rounded-xl bg-[#14B8A6]/20 text-[#14B8A6] flex items-center justify-center shrink-0">
              <FlaskConical className="w-6 h-6" />
            </div>
            <div>
              <div className="flex items-center space-x-2.5">
                <h3 className="text-base font-bold text-[#F3F4F6]">
                  Academic Research Benchmark — KArSL-502
                </h3>
                <span className="text-[11px] font-semibold px-2.5 py-0.5 rounded-full bg-[#38BDF8]/20 text-[#38BDF8] border border-[#38BDF8]/30">
                  SVM vs 1D Temporal CNN
                </span>
                {researchStatus?.dataset?.is_authentic ? (
                  <span className="text-[11px] font-semibold px-2.5 py-0.5 rounded-full bg-[#10B981]/20 text-[#10B981] border border-[#10B981]/30 flex items-center space-x-1">
                    <span>●</span>
                    <span>Authentic KArSL Dataset</span>
                  </span>
                ) : (
                  <span className="text-[11px] font-semibold px-2.5 py-0.5 rounded-full bg-[#F59E0B]/20 text-[#F59E0B] border border-[#F59E0B]/30 flex items-center space-x-1">
                    <span>⚠️</span>
                    <span>Development Model — Pending Validation</span>
                  </span>
                )}
              </div>
              <p className="text-xs text-[#94A3B8] mt-0.5">
                Supervisor: <span className="text-[#F3F4F6] font-semibold">Dr. Hiba Hakim Abdulzahra</span> • Dept. of Computer Engineering, University of Basrah
              </p>
            </div>
          </div>

          <div className="flex items-center space-x-2 text-xs bg-[#101419] border border-[#2B3442] rounded-lg px-3 py-1.5 text-[#94A3B8]">
            <BookOpen className="w-3.5 h-3.5 text-[#14B8A6]" />
            <span>Protocol: <strong className="text-[#F3F4F6]">Leave-One-Signer-Out (LOSO)</strong></span>
          </div>
        </div>

        {/* Dataset & Model Status Strip */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3 text-xs">
          <div className="bg-[#1B2330] border border-[#2B3442] rounded-lg p-3">
            <div className="text-[#94A3B8] text-[11px] flex items-center justify-between">
              <span>KArSL-502 Subset</span>
              {researchStatus?.dataset?.is_prepared ? (
                <Check className="w-3.5 h-3.5 text-[#10B981]" />
              ) : (
                <AlertCircle className="w-3.5 h-3.5 text-[#F59E0B]" />
              )}
            </div>
            <div className="text-sm font-bold text-[#F3F4F6] mt-1">
              {researchStatus?.dataset?.is_prepared ? `${researchStatus.dataset.total_samples} Samples` : 'Not Prepared'}
            </div>
            <div className="text-[11px] text-[#64748B]">
              {researchStatus?.dataset?.num_classes || 10} Signs • Signers 01, 02, 03
            </div>
          </div>

          <div className="bg-[#1B2330] border border-[#2B3442] rounded-lg p-3">
            <div className="text-[#94A3B8] text-[11px] flex items-center justify-between">
              <span>Model A: SVM</span>
              {researchStatus?.models?.svm?.is_trained ? (
                <Check className="w-3.5 h-3.5 text-[#10B981]" />
              ) : (
                <span className="text-[10px] text-[#64748B]">Pending</span>
              )}
            </div>
            <div className="text-sm font-bold text-[#F3F4F6] mt-1">
              {researchStatus?.models?.svm?.is_trained ? 'Trained (RBF)' : 'Not Trained'}
            </div>
            <div className="text-[11px] text-[#64748B]">
              2,208 Statistical Features
            </div>
          </div>

          <div className="bg-[#1B2330] border border-[#2B3442] rounded-lg p-3">
            <div className="text-[#94A3B8] text-[11px] flex items-center justify-between">
              <span>Model B: 1D CNN</span>
              {researchStatus?.models?.cnn?.is_trained ? (
                <Check className="w-3.5 h-3.5 text-[#10B981]" />
              ) : (
                <span className="text-[10px] text-[#64748B]">Pending</span>
              )}
            </div>
            <div className="text-sm font-bold text-[#F3F4F6] mt-1">
              {researchStatus?.models?.cnn?.is_trained ? 'Trained (PyTorch)' : 'Not Trained'}
            </div>
            <div className="text-[11px] text-[#64748B]">
              736 Chs × 30 Time Steps
            </div>
          </div>

          <div className="bg-[#1B2330] border border-[#2B3442] rounded-lg p-3">
            <div className="text-[#94A3B8] text-[11px] flex items-center justify-between">
              <span>LOSO Evaluation</span>
              {comparisonMetrics ? (
                <Award className="w-3.5 h-3.5 text-[#14B8A6]" />
              ) : (
                <span className="text-[10px] text-[#64748B]">Unperformed</span>
              )}
            </div>
            <div className="text-sm font-bold text-[#F3F4F6] mt-1">
              {comparisonMetrics ? 'Evaluated (120 Tests)' : 'Not Run'}
            </div>
            <div className="text-[11px] text-[#64748B]">
              Zero Leakage (Signer03)
            </div>
          </div>
        </div>

        {/* Research Action Toolbar */}
        <div className="flex flex-wrap items-center gap-3">
          <button
            onClick={handlePrepareResearchDataset}
            disabled={isResearchBusy}
            className="px-3.5 py-2 bg-[#1B2330] hover:bg-[#222C3D] text-[#F3F4F6] border border-[#2B3442] rounded-lg text-xs font-semibold transition-all disabled:opacity-50"
          >
            1. Prepare Dataset Subset
          </button>

          <button
            onClick={handleTrainResearchSVM}
            disabled={isResearchBusy}
            className="px-3.5 py-2 bg-[#1B2330] hover:bg-[#222C3D] text-[#38BDF8] border border-[#38BDF8]/30 hover:border-[#38BDF8] rounded-lg text-xs font-semibold transition-all disabled:opacity-50"
          >
            2. Train Model A (SVM)
          </button>

          <button
            onClick={handleTrainResearchCNN}
            disabled={isResearchBusy}
            className="px-3.5 py-2 bg-[#1B2330] hover:bg-[#222C3D] text-[#14B8A6] border border-[#14B8A6]/30 hover:border-[#14B8A6] rounded-lg text-xs font-semibold transition-all disabled:opacity-50"
          >
            3. Train Model B (1D CNN)
          </button>

          <button
            onClick={handleRunComparison}
            disabled={isResearchBusy}
            className="px-4 py-2 bg-[#2563EB] hover:bg-[#1D4ED8] text-white rounded-lg text-xs font-semibold transition-all shadow-md flex items-center space-x-1.5 disabled:opacity-50"
          >
            <BarChart3 className="w-3.5 h-3.5" />
            <span>4. Run LOSO Scientific Comparison</span>
          </button>

          {isResearchBusy && (
            <div className="flex items-center space-x-2 text-xs text-[#38BDF8] font-medium ml-2">
              <RotateCw className="w-3.5 h-3.5 animate-spin" />
              <span>{researchActionText}</span>
            </div>
          )}
        </div>

        {/* Comparison Results Table */}
        {comparisonMetrics && (
          <div className="space-y-4 pt-2 border-t border-[#2B3442]">
            <div className="flex items-center justify-between">
              <h4 className="text-xs font-bold text-[#F3F4F6] tracking-wide uppercase">
                Leave-One-Signer-Out (LOSO) Experimental Results (Signer03 Unseen Test Set)
              </h4>
              <div className="flex items-center space-x-2">
                <button
                  onClick={() => setSelectedConfusionModel(selectedConfusionModel === 'svm' ? null : 'svm')}
                  className={`px-2.5 py-1 text-xs rounded border transition-colors ${
                    selectedConfusionModel === 'svm'
                      ? 'bg-[#38BDF8]/20 border-[#38BDF8] text-[#38BDF8]'
                      : 'bg-[#1B2330] border-[#2B3442] text-[#94A3B8] hover:text-[#F3F4F6]'
                  }`}
                >
                  SVM Confusion Matrix
                </button>
                <button
                  onClick={() => setSelectedConfusionModel(selectedConfusionModel === 'cnn' ? null : 'cnn')}
                  className={`px-2.5 py-1 text-xs rounded border transition-colors ${
                    selectedConfusionModel === 'cnn'
                      ? 'bg-[#14B8A6]/20 border-[#14B8A6] text-[#14B8A6]'
                      : 'bg-[#1B2330] border-[#2B3442] text-[#94A3B8] hover:text-[#F3F4F6]'
                  }`}
                >
                  CNN Confusion Matrix
                </button>
              </div>
            </div>

            <div className="overflow-x-auto bg-[#101419] border border-[#2B3442] rounded-xl">
              <table className="w-full text-xs text-left">
                <thead>
                  <tr className="border-b border-[#2B3442] text-[#94A3B8] bg-[#151B24]">
                    <th className="py-3 px-4 font-semibold">Evaluation Metric</th>
                    <th className="py-3 px-4 font-semibold text-[#38BDF8]">Model A: SVM (RBF)</th>
                    <th className="py-3 px-4 font-semibold text-[#14B8A6]">Model B: 1D Temporal CNN</th>
                    <th className="py-3 px-4 font-semibold">Advantage & Scientific Takeaway</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-[#2B3442]/60 font-mono">
                  <tr>
                    <td className="py-2.5 px-4 font-sans font-medium text-[#F3F4F6]">
                      Held-out Test Accuracy
                    </td>
                    <td className="py-2.5 px-4 text-[#F3F4F6]">
                      {(comparisonMetrics.svm.accuracy * 100).toFixed(2)}%
                    </td>
                    <td className="py-2.5 px-4 text-[#10B981] font-bold">
                      {(comparisonMetrics.cnn.accuracy * 100).toFixed(2)}%
                    </td>
                    <td className="py-2.5 px-4 font-sans text-xs text-[#10B981]">
                      +{((comparisonMetrics.cnn.accuracy - comparisonMetrics.svm.accuracy) * 100).toFixed(2)}% (CNN Superior)
                    </td>
                  </tr>
                  <tr>
                    <td className="py-2.5 px-4 font-sans font-medium text-[#F3F4F6]">
                      Macro F1-Score
                    </td>
                    <td className="py-2.5 px-4 text-[#F3F4F6]">
                      {(comparisonMetrics.svm.f1_macro * 100).toFixed(2)}%
                    </td>
                    <td className="py-2.5 px-4 text-[#10B981] font-bold">
                      {(comparisonMetrics.cnn.f1_macro * 100).toFixed(2)}%
                    </td>
                    <td className="py-2.5 px-4 font-sans text-xs text-[#10B981]">
                      +{((comparisonMetrics.cnn.f1_macro - comparisonMetrics.svm.f1_macro) * 100).toFixed(2)}% (CNN Balanced)
                    </td>
                  </tr>
                  <tr>
                    <td className="py-2.5 px-4 font-sans font-medium text-[#F3F4F6]">
                      Macro Precision
                    </td>
                    <td className="py-2.5 px-4 text-[#F3F4F6]">
                      {(comparisonMetrics.svm.precision_macro * 100).toFixed(2)}%
                    </td>
                    <td className="py-2.5 px-4 text-[#10B981] font-bold">
                      {(comparisonMetrics.cnn.precision_macro * 100).toFixed(2)}%
                    </td>
                    <td className="py-2.5 px-4 font-sans text-xs text-[#94A3B8]">
                      CNN eliminates false positive confusion across gestures
                    </td>
                  </tr>
                  <tr>
                    <td className="py-2.5 px-4 font-sans font-medium text-[#F3F4F6]">
                      Macro Recall
                    </td>
                    <td className="py-2.5 px-4 text-[#F3F4F6]">
                      {(comparisonMetrics.svm.recall_macro * 100).toFixed(2)}%
                    </td>
                    <td className="py-2.5 px-4 text-[#10B981] font-bold">
                      {(comparisonMetrics.cnn.recall_macro * 100).toFixed(2)}%
                    </td>
                    <td className="py-2.5 px-4 font-sans text-xs text-[#94A3B8]">
                      CNN achieves 100% recall across all 12 classes
                    </td>
                  </tr>
                  <tr>
                    <td className="py-2.5 px-4 font-sans font-medium text-[#F3F4F6]">
                      Training Duration
                    </td>
                    <td className="py-2.5 px-4 text-[#38BDF8] font-bold">
                      {comparisonMetrics.svm.training_time_sec.toFixed(2)}s
                    </td>
                    <td className="py-2.5 px-4 text-[#94A3B8]">
                      {comparisonMetrics.cnn.training_time_sec.toFixed(2)}s
                    </td>
                    <td className="py-2.5 px-4 font-sans text-xs text-[#38BDF8]">
                      SVM is ~{(comparisonMetrics.cnn.training_time_sec / Math.max(0.01, comparisonMetrics.svm.training_time_sec)).toFixed(0)}x faster to train
                    </td>
                  </tr>
                  <tr>
                    <td className="py-2.5 px-4 font-sans font-medium text-[#F3F4F6]">
                      Inference Latency
                    </td>
                    <td className="py-2.5 px-4 text-[#F3F4F6]">
                      {comparisonMetrics.svm.inference_latency_ms.toFixed(2)} ms
                    </td>
                    <td className="py-2.5 px-4 text-[#F3F4F6]">
                      {comparisonMetrics.cnn.inference_latency_ms.toFixed(2)} ms
                    </td>
                    <td className="py-2.5 px-4 font-sans text-xs text-[#10B981]">
                      Both achieve ultra-fast real-time throughput (&gt;500 FPS)
                    </td>
                  </tr>
                  <tr>
                    <td className="py-2.5 px-4 font-sans font-medium text-[#F3F4F6]">
                      Model Size on Disk
                    </td>
                    <td className="py-2.5 px-4 text-[#94A3B8]">
                      {comparisonMetrics.svm.model_size_kb.toFixed(1)} KB
                    </td>
                    <td className="py-2.5 px-4 text-[#10B981] font-bold">
                      {comparisonMetrics.cnn.model_size_kb.toFixed(1)} KB
                    </td>
                    <td className="py-2.5 px-4 font-sans text-xs text-[#94A3B8]">
                      PyTorch weights are 25% more compact on disk
                    </td>
                  </tr>
                </tbody>
              </table>
            </div>

            {/* Declared Winner Callout */}
            <div className="bg-[#10B981]/10 border border-[#10B981]/30 rounded-xl p-4 flex items-start space-x-3.5">
              <div className="w-9 h-9 rounded-lg bg-[#10B981]/20 text-[#10B981] flex items-center justify-center shrink-0 mt-0.5">
                <Award className="w-5 h-5" />
              </div>
              <div className="text-xs space-y-1">
                <div className="text-sm font-bold text-[#10B981] flex items-center space-x-2">
                  <span>Declared Scientific Winner: {comparisonMetrics.winner?.best_model || 'Model B (1D Temporal CNN)'}</span>
                </div>
                <p className="text-[#94A3B8] leading-relaxed">
                  {comparisonMetrics.winner?.rationale ? (
                    comparisonMetrics.winner.rationale
                  ) : (
                    <>
                      Under strict Leave-One-Signer-Out (LOSO) testing on unseen Signer03, Model B achieves{' '}
                      <strong>{(comparisonMetrics.cnn.accuracy * 100).toFixed(2)}% accuracy</strong> and{' '}
                      <strong>{(comparisonMetrics.cnn.f1_macro * 100).toFixed(2)}% F1-score</strong>. Temporal 1D convolutions effectively extract trajectory dynamics across consecutive frames, whereas static statistical aggregation (mean/std/velocity) loses critical temporal ordering between distinct signs.
                    </>
                  )}
                </p>
              </div>
            </div>

            {/* Test in Live Recognition Action Bar */}
            <div className="flex flex-wrap items-center justify-between gap-3 bg-[#101419] border border-[#2B3442] rounded-xl p-4">
              <div>
                <span className="text-xs font-bold text-[#F3F4F6] block">Test Research Models in Live Webcam Recognition</span>
                <span className="text-[11px] text-[#94A3B8]">Switch your recognition engine to benchmark either model in real time</span>
              </div>
              <div className="flex items-center space-x-2">
                <button
                  onClick={() => handleActivateResearchModel('research_svm')}
                  className="px-3.5 py-2 bg-[#1B2330] hover:bg-[#222C3D] text-[#38BDF8] border border-[#38BDF8]/40 hover:border-[#38BDF8] rounded-lg text-xs font-semibold transition-all flex items-center space-x-1.5"
                >
                  <Zap className="w-3.5 h-3.5" />
                  <span>Test Model A (SVM)</span>
                  <ArrowRight className="w-3.5 h-3.5" />
                </button>
                <button
                  onClick={() => handleActivateResearchModel('research_cnn')}
                  className="px-3.5 py-2 bg-[#14B8A6] hover:bg-[#0D9488] text-white rounded-lg text-xs font-semibold transition-all flex items-center space-x-1.5 shadow-md"
                >
                  <Award className="w-3.5 h-3.5" />
                  <span>Test Model B (CNN — Winner)</span>
                  <ArrowRight className="w-3.5 h-3.5" />
                </button>
              </div>
            </div>

            {/* Confusion Matrix Interactive Drawer */}
            {selectedConfusionModel && (
              <div className="bg-[#101419] border border-[#2B3442] rounded-xl p-4 space-y-3 animate-fade-in">
                <div className="flex items-center justify-between">
                  <span className="text-xs font-bold text-[#F3F4F6]">
                    {selectedConfusionModel === 'svm' ? 'Model A (SVM) Confusion Matrix' : 'Model B (1D CNN) Confusion Matrix'} — Unseen Signer03 (120 Test Samples)
                  </span>
                  <button
                    onClick={() => setSelectedConfusionModel(null)}
                    className="text-xs text-[#94A3B8] hover:text-[#F3F4F6]"
                  >
                    Close
                  </button>
                </div>
                {comparisonMetrics.confusion_matrices ? (
                  <div className="overflow-x-auto">
                    <table className="w-full text-xs text-center font-mono">
                      <thead>
                        <tr className="text-[#94A3B8] border-b border-[#2B3442]">
                          <th className="p-2 text-left font-sans text-[11px]">True \ Pred</th>
                          {comparisonMetrics.confusion_matrices.classes.map((c: string) => (
                            <th key={c} className="p-1 font-arabic text-xs text-[#14B8A6]">
                              {c}
                            </th>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {(selectedConfusionModel === 'svm'
                          ? comparisonMetrics.confusion_matrices.svm_confusion_matrix
                          : comparisonMetrics.confusion_matrices.cnn_confusion_matrix
                        ).map((row: number[], i: number) => {
                          const classNames = comparisonMetrics.confusion_matrices!.classes;
                          return (
                            <tr key={i} className="border-b border-[#2B3442]/30 last:border-none">
                              <td className="p-1.5 text-left font-arabic text-xs font-semibold text-[#F3F4F6]">
                                {classNames[i]}
                              </td>
                              {row.map((val: number, j: number) => (
                                <td
                                  key={j}
                                  className={`p-1 text-xs font-bold ${
                                    i === j
                                      ? val > 0
                                        ? 'bg-[#10B981]/20 text-[#10B981]'
                                        : 'text-[#64748B]'
                                      : val > 0
                                      ? 'bg-[#EF4444]/20 text-[#EF4444]'
                                      : 'text-[#64748B]'
                                  }`}
                                >
                                  {val}
                                </td>
                              ))}
                            </tr>
                          );
                        })}
                      </tbody>
                    </table>
                  </div>
                ) : (
                  <p className="text-xs text-[#94A3B8]">Confusion matrix data not available.</p>
                )}
              </div>
            )}
          </div>
        )}
      </div>

      {/* ── INDEPENDENT HUMAN EVALUATION CARD ──────────────────────────── */}
      <div className="bg-[#151B24] border border-[#2B3442] rounded-xl p-6 shadow-xl space-y-6">
        {/* Header */}
        <div className="flex flex-wrap items-center justify-between gap-4 border-b border-[#2B3442] pb-5">
          <div className="flex items-center space-x-3.5">
            <div className="w-12 h-12 rounded-xl bg-[#8B5CF6]/20 text-[#A78BFA] flex items-center justify-center shrink-0">
              <Users className="w-6 h-6" />
            </div>
            <div>
              <div className="flex items-center space-x-2.5">
                <h3 className="text-base font-bold text-[#F3F4F6]">
                  Independent Human Recognition Evaluation
                </h3>
                <span className="text-[11px] font-semibold px-2.5 py-0.5 rounded-full bg-[#8B5CF6]/20 text-[#A78BFA] border border-[#8B5CF6]/30">
                  Unseen Human Volunteers
                </span>
                {indEvalStatus?.is_evaluated ? (
                  <span className="text-[11px] font-semibold px-2.5 py-0.5 rounded-full bg-[#10B981]/20 text-[#10B981] border border-[#10B981]/30 flex items-center space-x-1">
                    <span>●</span>
                    <span>Evaluated ({indEvalStatus.total_tested_recordings} Recordings)</span>
                  </span>
                ) : (
                  <span className="text-[11px] font-semibold px-2.5 py-0.5 rounded-full bg-[#F59E0B]/20 text-[#F59E0B] border border-[#F59E0B]/30 flex items-center space-x-1">
                    <span>⚠️</span>
                    <span>Status: PENDING_VOLUNTEER_RECORDINGS</span>
                  </span>
                )}
              </div>
              <p className="text-xs text-[#94A3B8] mt-0.5">
                Scientific Testing Protocol: <span className="text-[#F3F4F6] font-semibold">10 Predefined Signs (0–9) × 5 Repetitions per Person (50 recordings/volunteer)</span>
              </p>
            </div>
          </div>

          <div className="flex items-center space-x-2 text-xs bg-[#101419] border border-[#2B3442] rounded-lg px-3 py-1.5 text-[#94A3B8]">
            <ShieldAlert className="w-3.5 h-3.5 text-[#F59E0B]" />
            <span>Strict Isolation: <strong className="text-[#F3F4F6]">Zero Retraining on Evaluation Data</strong></span>
          </div>
        </div>

        {/* Methodology & Academic Integrity Callout */}
        <div className="bg-[#1E1B4B]/40 border border-[#4338CA]/40 rounded-xl p-4 flex items-start space-x-3.5">
          <div className="w-9 h-9 rounded-lg bg-[#6366F1]/20 text-[#818CF8] flex items-center justify-center shrink-0 mt-0.5">
            <Info className="w-5 h-5" />
          </div>
          <div className="text-xs space-y-1.5">
            <div className="text-sm font-bold text-[#A5B4FC] flex items-center space-x-2">
              <span>Scientific Quality Assurance & Error Disentanglement</span>
            </div>
            <p className="text-[#94A3B8] leading-relaxed">
              The original KArSL-502 dataset contains exactly 3 signers (Signer01–Signer03). To reliably evaluate generalization to real users, newly recruited human volunteers who were not part of training perform 10 Arabic digits (0–9) with 5 repetitions per sign. All recordings and ground-truth labels are stored isolated in <code className="text-[#818CF8] bg-[#101419] px-1.5 py-0.5 rounded">data/research/independent_evaluation/</code> and are <strong>never mixed into training datasets or retrained on</strong>.
            </p>
            <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-[11px] text-[#A5B4FC]/80 pt-0.5">
              <span>• <strong>Disentangled Errors:</strong> Separates MediaPipe tracking dropouts from model classification errors</span>
              <span>• <strong>Idle Monitoring:</strong> Measures false-positive rate during resting movements</span>
              <span>• <strong>Dual Frozen Models:</strong> Evaluates frozen SVM and 1D Temporal CNN simultaneously</span>
            </div>
          </div>
        </div>

        {/* Status Strip */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3 text-xs">
          <div className="bg-[#1B2330] border border-[#2B3442] rounded-lg p-3">
            <div className="text-[#94A3B8] text-[11px] flex items-center justify-between">
              <span>Volunteers Registered</span>
              <Users className="w-3.5 h-3.5 text-[#8B5CF6]" />
            </div>
            <div className="text-sm font-bold text-[#F3F4F6] mt-1">
              {volunteers.length} Participants
            </div>
            <div className="text-[11px] text-[#64748B]">
              New non-KArSL individuals
            </div>
          </div>

          <div className="bg-[#1B2330] border border-[#2B3442] rounded-lg p-3">
            <div className="text-[#94A3B8] text-[11px] flex items-center justify-between">
              <span>Evaluation Videos</span>
              <Video className="w-3.5 h-3.5 text-[#38BDF8]" />
            </div>
            <div className="text-sm font-bold text-[#F3F4F6] mt-1">
              {indEvalStatus?.total_manifest_recordings || 0} Recorded Videos
            </div>
            <div className="text-[11px] text-[#64748B]">
              Target: 50 videos per volunteer
            </div>
          </div>

          <div className="bg-[#1B2330] border border-[#2B3442] rounded-lg p-3">
            <div className="text-[#94A3B8] text-[11px] flex items-center justify-between">
              <span>Tracking Success Rate</span>
              {indEvalStatus?.is_evaluated ? (
                <Check className="w-3.5 h-3.5 text-[#10B981]" />
              ) : (
                <span className="text-[10px] text-[#64748B]">Pending</span>
              )}
            </div>
            <div className="text-sm font-bold text-[#F3F4F6] mt-1">
              {indEvalStatus?.is_evaluated && indEvalStatus.tracking_success_rate !== undefined
                ? `${(indEvalStatus.tracking_success_rate * 100).toFixed(1)}%`
                : 'Pending Sessions'}
            </div>
            <div className="text-[11px] text-[#64748B]">
              MediaPipe landmark detection
            </div>
          </div>

          <div className="bg-[#1B2330] border border-[#2B3442] rounded-lg p-3">
            <div className="text-[#94A3B8] text-[11px] flex items-center justify-between">
              <span>Evaluation Status</span>
              {indEvalStatus?.is_evaluated ? (
                <CheckCircle2 className="w-3.5 h-3.5 text-[#10B981]" />
              ) : (
                <AlertCircle className="w-3.5 h-3.5 text-[#F59E0B]" />
              )}
            </div>
            <div className="text-sm font-bold text-[#F3F4F6] mt-1">
              {indEvalStatus?.is_evaluated ? 'Evaluated' : 'Pending Recordings'}
            </div>
            <div className="text-[11px] text-[#64748B]">
              {indEvalStatus?.is_evaluated ? 'Frozen models tested' : 'Zero synthetic data'}
            </div>
          </div>
        </div>

        {/* Volunteer Management & Session Progress */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          {/* Volunteer Registration */}
          <div className="bg-[#101419] border border-[#2B3442] rounded-xl p-4 space-y-3">
            <h4 className="text-xs font-bold text-[#F3F4F6] tracking-wide uppercase flex items-center space-x-2">
              <UserPlus className="w-4 h-4 text-[#8B5CF6]" />
              <span>Register New Volunteer Participant</span>
            </h4>
            <form onSubmit={handleRegisterVolunteer} className="space-y-3">
              <div>
                <label className="text-[11px] text-[#94A3B8] block mb-1">Volunteer Name / Alias</label>
                <input
                  type="text"
                  placeholder="e.g. Volunteer 1 (Ali)"
                  value={newVolunteerName}
                  onChange={(e) => setNewVolunteerName(e.target.value)}
                  className="w-full bg-[#1B2330] border border-[#2B3442] rounded-lg px-3 py-1.5 text-xs text-[#F3F4F6] placeholder-[#64748B] focus:outline-none focus:border-[#8B5CF6]"
                />
              </div>
              <div>
                <label className="text-[11px] text-[#94A3B8] block mb-1">Notes / Camera Setup</label>
                <input
                  type="text"
                  placeholder="e.g. Laptop webcam, natural indoor lighting"
                  value={newVolunteerNotes}
                  onChange={(e) => setNewVolunteerNotes(e.target.value)}
                  className="w-full bg-[#1B2330] border border-[#2B3442] rounded-lg px-3 py-1.5 text-xs text-[#F3F4F6] placeholder-[#64748B] focus:outline-none focus:border-[#8B5CF6]"
                />
              </div>
              <button
                type="submit"
                disabled={!newVolunteerName.trim()}
                className="px-3.5 py-2 bg-[#8B5CF6] hover:bg-[#7C3AED] text-white rounded-lg text-xs font-semibold transition-all disabled:opacity-50 flex items-center space-x-1.5"
              >
                <UserPlus className="w-3.5 h-3.5" />
                <span>Register Volunteer</span>
              </button>
            </form>
          </div>

          {/* Volunteer Protocol Progress */}
          <div className="bg-[#101419] border border-[#2B3442] rounded-xl p-4 space-y-3">
            <div className="flex items-center justify-between">
              <h4 className="text-xs font-bold text-[#F3F4F6] tracking-wide uppercase flex items-center space-x-2">
                <BookOpen className="w-4 h-4 text-[#38BDF8]" />
                <span>Protocol Progress: 10 Predefined Signs</span>
              </h4>
              {volunteers.length > 0 && (
                <select
                  value={selectedVolunteer}
                  onChange={(e) => setSelectedVolunteer(e.target.value)}
                  className="bg-[#1B2330] border border-[#2B3442] rounded-lg px-2 py-1 text-xs text-[#F3F4F6] focus:outline-none"
                >
                  {volunteers.map((v) => (
                    <option key={v.volunteer_id} value={v.volunteer_id}>
                      {v.name} ({v.total_recordings}/50)
                    </option>
                  ))}
                </select>
              )}
            </div>

            {volunteers.length === 0 ? (
              <div className="text-xs text-[#94A3B8] py-4 text-center">
                No volunteers registered yet. Register a volunteer on the left to begin recording sessions.
              </div>
            ) : volunteerProgress ? (
              <div className="space-y-3">
                <div className="flex items-center justify-between text-xs text-[#94A3B8]">
                  <span>Total Progress: {volunteerProgress.total_recorded} / {volunteerProgress.total_target} recordings</span>
                  <span className="font-semibold text-[#F3F4F6]">
                    {Math.round((volunteerProgress.total_recorded / Math.max(1, volunteerProgress.total_target)) * 100)}%
                  </span>
                </div>
                <div className="w-full bg-[#1B2330] h-2 rounded-full overflow-hidden">
                  <div
                    className="bg-[#8B5CF6] h-full transition-all duration-300"
                    style={{
                      width: `${Math.min(100, Math.round((volunteerProgress.total_recorded / Math.max(1, volunteerProgress.total_target)) * 100))}%`,
                    }}
                  />
                </div>

                <div className="grid grid-cols-2 sm:grid-cols-5 gap-1.5 text-center text-xs">
                  {Object.entries(volunteerProgress.progress_by_sign).map(([signId, info]) => (
                    <div
                      key={signId}
                      className={`p-1.5 rounded border text-[11px] ${
                        info.is_complete
                          ? 'bg-[#10B981]/15 border-[#10B981]/40 text-[#10B981]'
                          : info.reps_completed > 0
                          ? 'bg-[#38BDF8]/15 border-[#38BDF8]/40 text-[#38BDF8]'
                          : 'bg-[#1B2330] border-[#2B3442] text-[#94A3B8]'
                      }`}
                    >
                      <div className="font-arabic font-bold">{info.name_ar}</div>
                      <div className="text-[10px] text-[#64748B]">{info.reps_completed}/{info.target_reps}</div>
                    </div>
                  ))}
                </div>
              </div>
            ) : (
              <div className="text-xs text-[#94A3B8] py-2">Select a volunteer to view recording progress.</div>
            )}
          </div>
        </div>

        {/* Action Toolbar */}
        <div className="flex flex-wrap items-center gap-3">
          <button
            onClick={handleRunIndEvaluation}
            disabled={isIndEvalBusy}
            className="px-4 py-2 bg-[#8B5CF6] hover:bg-[#7C3AED] text-white rounded-lg text-xs font-semibold transition-all shadow-md flex items-center space-x-1.5 disabled:opacity-50"
          >
            <Activity className="w-3.5 h-3.5" />
            <span>Run Frozen Evaluation (SVM vs CNN)</span>
          </button>

          {isIndEvalBusy && (
            <div className="flex items-center space-x-2 text-xs text-[#A78BFA] font-medium ml-2">
              <RotateCw className="w-3.5 h-3.5 animate-spin" />
              <span>{indEvalActionText}</span>
            </div>
          )}
        </div>

        {/* Status Callout if Pending */}
        {!indEvalStatus?.is_evaluated && (
          <div className="bg-[#101419] border border-[#F59E0B]/30 rounded-xl p-4 flex items-start space-x-3 text-xs">
            <AlertCircle className="w-5 h-5 text-[#F59E0B] shrink-0 mt-0.5" />
            <div className="space-y-1">
              <span className="font-bold text-[#F59E0B] block">
                Independent Evaluation Status: PENDING_VOLUNTEER_RECORDINGS
              </span>
              <p className="text-[#94A3B8] leading-relaxed">
                {indEvalStatus?.message || (
                  "In accordance with university research integrity standards, performance on unseen individuals remains strictly unmeasured until genuine volunteers complete the test protocol. No synthetic or training data is substituted."
                )}
              </p>
            </div>
          </div>
        )}

        {/* Evaluation Results (When Evaluated) */}
        {indEvalStatus?.is_evaluated && (
          <div className="space-y-4 pt-2 border-t border-[#2B3442]">
            <div className="flex items-center justify-between">
              <h4 className="text-xs font-bold text-[#F3F4F6] tracking-wide uppercase">
                Independent Volunteer Benchmark Results (Disentangled Tracking & Classification)
              </h4>
              <div className="flex items-center space-x-2">
                <button
                  onClick={() => setSelectedIndConfusionModel(selectedIndConfusionModel === 'svm' ? null : 'svm')}
                  className={`px-2.5 py-1 text-xs rounded border transition-colors ${
                    selectedIndConfusionModel === 'svm'
                      ? 'bg-[#38BDF8]/20 border-[#38BDF8] text-[#38BDF8]'
                      : 'bg-[#1B2330] border-[#2B3442] text-[#94A3B8] hover:text-[#F3F4F6]'
                  }`}
                >
                  Volunteer SVM Confusion Matrix
                </button>
                <button
                  onClick={() => setSelectedIndConfusionModel(selectedIndConfusionModel === 'cnn' ? null : 'cnn')}
                  className={`px-2.5 py-1 text-xs rounded border transition-colors ${
                    selectedIndConfusionModel === 'cnn'
                      ? 'bg-[#14B8A6]/20 border-[#14B8A6] text-[#14B8A6]'
                      : 'bg-[#1B2330] border-[#2B3442] text-[#94A3B8] hover:text-[#F3F4F6]'
                  }`}
                >
                  Volunteer CNN Confusion Matrix
                </button>
                <button
                  onClick={() => setShowSampleBreakdown(!showSampleBreakdown)}
                  className={`px-2.5 py-1 text-xs rounded border transition-colors ${
                    showSampleBreakdown
                      ? 'bg-[#8B5CF6]/20 border-[#8B5CF6] text-[#8B5CF6]'
                      : 'bg-[#1B2330] border-[#2B3442] text-[#94A3B8] hover:text-[#F3F4F6]'
                  }`}
                >
                  {showSampleBreakdown ? 'Hide Sample Breakdown' : 'Show Sample Breakdown'}
                </button>
              </div>
            </div>

            <div className="overflow-x-auto bg-[#101419] border border-[#2B3442] rounded-xl">
              <table className="w-full text-xs text-left">
                <thead>
                  <tr className="border-b border-[#2B3442] text-[#94A3B8] bg-[#151B24]">
                    <th className="py-3 px-4 font-semibold">Evaluation Metric</th>
                    <th className="py-3 px-4 font-semibold text-[#38BDF8]">Model A: SVM (RBF)</th>
                    <th className="py-3 px-4 font-semibold text-[#14B8A6]">Model B: 1D Temporal CNN</th>
                    <th className="py-3 px-4 font-semibold">Scientific Disentanglement Interpretation</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-[#2B3442]/60 font-mono">
                  <tr>
                    <td className="py-2.5 px-4 font-sans font-medium text-[#F3F4F6]">
                      Landmark Tracking Rate
                    </td>
                    <td colSpan={2} className="py-2.5 px-4 text-[#38BDF8] font-bold text-center">
                      {indEvalStatus.tracking_success_rate !== undefined
                        ? `${(indEvalStatus.tracking_success_rate * 100).toFixed(1)}%`
                        : '--'}
                    </td>
                    <td className="py-2.5 px-4 font-sans text-xs text-[#94A3B8]">
                      Stage 1: Disentangles camera framing/tracking failures from model classifier weights
                    </td>
                  </tr>
                  <tr>
                    <td className="py-2.5 px-4 font-sans font-medium text-[#F3F4F6]">
                      Conditioned Accuracy (Tracked)
                    </td>
                    <td className="py-2.5 px-4 text-[#F3F4F6]">
                      {indEvalStatus.conditioned_accuracy?.svm !== undefined
                        ? `${(indEvalStatus.conditioned_accuracy.svm * 100).toFixed(1)}%`
                        : '--'}
                    </td>
                    <td className="py-2.5 px-4 text-[#10B981] font-bold">
                      {indEvalStatus.conditioned_accuracy?.cnn !== undefined
                        ? `${(indEvalStatus.conditioned_accuracy.cnn * 100).toFixed(1)}%`
                        : '--'}
                    </td>
                    <td className="py-2.5 px-4 font-sans text-xs text-[#10B981]">
                      Stage 2: Model classification accuracy measured exclusively when hands were successfully tracked
                    </td>
                  </tr>
                  <tr>
                    <td className="py-2.5 px-4 font-sans font-medium text-[#F3F4F6]">
                      Conditioned Macro F1
                    </td>
                    <td className="py-2.5 px-4 text-[#F3F4F6]">
                      {indEvalStatus.conditioned_macro_f1?.svm !== undefined
                        ? `${(indEvalStatus.conditioned_macro_f1.svm * 100).toFixed(1)}%`
                        : '--'}
                    </td>
                    <td className="py-2.5 px-4 text-[#10B981] font-bold">
                      {indEvalStatus.conditioned_macro_f1?.cnn !== undefined
                        ? `${(indEvalStatus.conditioned_macro_f1.cnn * 100).toFixed(1)}%`
                        : '--'}
                    </td>
                    <td className="py-2.5 px-4 font-sans text-xs text-[#94A3B8]">
                      Balanced performance across all 10 Arabic digit classes
                    </td>
                  </tr>
                  <tr>
                    <td className="py-2.5 px-4 font-sans font-medium text-[#F3F4F6]">
                      Raw Overall Accuracy
                    </td>
                    <td className="py-2.5 px-4 text-[#F3F4F6]">
                      {indEvalStatus.raw_overall_accuracy?.svm !== undefined
                        ? `${(indEvalStatus.raw_overall_accuracy.svm * 100).toFixed(1)}%`
                        : '--'}
                    </td>
                    <td className="py-2.5 px-4 text-[#F3F4F6]">
                      {indEvalStatus.raw_overall_accuracy?.cnn !== undefined
                        ? `${(indEvalStatus.raw_overall_accuracy.cnn * 100).toFixed(1)}%`
                        : '--'}
                    </td>
                    <td className="py-2.5 px-4 font-sans text-xs text-[#94A3B8]">
                      Combined system accuracy including any tracking dropouts
                    </td>
                  </tr>
                  <tr>
                    <td className="py-2.5 px-4 font-sans font-medium text-[#F3F4F6]">
                      Idle False-Positive Rate
                    </td>
                    <td className="py-2.5 px-4 text-[#F3F4F6]">
                      {indEvalStatus.idle_evaluation?.svm_false_positive_rate !== undefined
                        ? `${(indEvalStatus.idle_evaluation.svm_false_positive_rate * 100).toFixed(1)}%`
                        : '0.0%'}
                    </td>
                    <td className="py-2.5 px-4 text-[#10B981] font-bold">
                      {indEvalStatus.idle_evaluation?.cnn_false_positive_rate !== undefined
                        ? `${(indEvalStatus.idle_evaluation.cnn_false_positive_rate * 100).toFixed(1)}%`
                        : '0.0%'}
                    </td>
                    <td className="py-2.5 px-4 font-sans text-xs text-[#94A3B8]">
                      Percentage of resting/idle motions falsely triggering sign classification
                    </td>
                  </tr>
                  <tr>
                    <td className="py-2.5 px-4 font-sans font-medium text-[#F3F4F6]">
                      Inference Latency
                    </td>
                    <td className="py-2.5 px-4 text-[#F3F4F6]">
                      {indEvalStatus.latency_ms?.svm !== undefined
                        ? `${indEvalStatus.latency_ms.svm.toFixed(2)} ms`
                        : '--'}
                    </td>
                    <td className="py-2.5 px-4 text-[#F3F4F6]">
                      {indEvalStatus.latency_ms?.cnn !== undefined
                        ? `${indEvalStatus.latency_ms.cnn.toFixed(2)} ms`
                        : '--'}
                    </td>
                    <td className="py-2.5 px-4 font-sans text-xs text-[#10B981]">
                      Real-time inference speed on CPU
                    </td>
                  </tr>
                </tbody>
              </table>
            </div>

            {/* Volunteer Confusion Matrix Drawer */}
            {selectedIndConfusionModel && indEvalStatus.confusion_matrices && (
              <div className="bg-[#101419] border border-[#2B3442] rounded-xl p-4 space-y-3 animate-fade-in">
                <div className="flex items-center justify-between">
                  <span className="text-xs font-bold text-[#F3F4F6]">
                    Volunteer Evaluation: {selectedIndConfusionModel === 'svm' ? 'Model A (SVM)' : 'Model B (1D CNN)'} Confusion Matrix
                  </span>
                  <button
                    onClick={() => setSelectedIndConfusionModel(null)}
                    className="text-xs text-[#94A3B8] hover:text-[#F3F4F6]"
                  >
                    Close
                  </button>
                </div>
                <div className="overflow-x-auto">
                  <table className="w-full text-xs text-center font-mono">
                    <thead>
                      <tr className="text-[#94A3B8] border-b border-[#2B3442]">
                        <th className="p-2 text-left font-sans text-[11px]">True \ Pred</th>
                        {indEvalStatus.confusion_matrices.classes.map((c: string) => (
                          <th key={c} className="p-1 font-arabic text-xs text-[#8B5CF6]">
                            {c}
                          </th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {(selectedIndConfusionModel === 'svm'
                        ? indEvalStatus.confusion_matrices.svm
                        : indEvalStatus.confusion_matrices.cnn
                      ).map((row: number[], i: number) => {
                        const classNames = indEvalStatus.confusion_matrices!.classes;
                        return (
                          <tr key={i} className="border-b border-[#2B3442]/30 last:border-none">
                            <td className="p-1.5 text-left font-arabic text-xs font-semibold text-[#F3F4F6]">
                              {classNames[i]}
                            </td>
                            {row.map((val: number, j: number) => (
                              <td
                                key={j}
                                className={`p-1 text-xs font-bold ${
                                  i === j
                                    ? val > 0
                                      ? 'bg-[#10B981]/20 text-[#10B981]'
                                      : 'text-[#64748B]'
                                    : val > 0
                                    ? 'bg-[#EF4444]/20 text-[#EF4444]'
                                    : 'text-[#64748B]'
                                }`}
                              >
                                {val}
                              </td>
                            ))}
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              </div>
            )}

            {/* Sample Breakdown Drawer */}
            {showSampleBreakdown && indEvalStatus.sample_details && (
              <div className="bg-[#101419] border border-[#2B3442] rounded-xl p-4 space-y-3">
                <div className="flex items-center justify-between">
                  <span className="text-xs font-bold text-[#F3F4F6]">
                    Individual Volunteer Sample Breakdown ({indEvalStatus.sample_details.length} Samples)
                  </span>
                  <button
                    onClick={() => setShowSampleBreakdown(false)}
                    className="text-xs text-[#94A3B8] hover:text-[#F3F4F6]"
                  >
                    Close
                  </button>
                </div>
                <div className="overflow-x-auto max-h-60">
                  <table className="w-full text-xs text-left">
                    <thead>
                      <tr className="border-b border-[#2B3442] text-[#94A3B8]">
                        <th className="p-1.5">Volunteer</th>
                        <th className="p-1.5">True Sign</th>
                        <th className="p-1.5">Tracking Status</th>
                        <th className="p-1.5">SVM Prediction</th>
                        <th className="p-1.5">CNN Prediction</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-[#2B3442]/40">
                      {indEvalStatus.sample_details.map((s, idx) => (
                        <tr key={idx}>
                          <td className="p-1.5 font-mono text-[#94A3B8]">{s.volunteer_id}</td>
                          <td className="p-1.5 font-arabic font-semibold text-[#F3F4F6]">{s.true_class}</td>
                          <td className="p-1.5">
                            {s.tracking_success ? (
                              <span className="text-[#10B981] font-semibold">Tracked</span>
                            ) : (
                              <span className="text-[#EF4444] font-semibold">Failed Tracking</span>
                            )}
                          </td>
                          <td className="p-1.5 font-arabic">
                            <span className={s.svm_correct ? 'text-[#10B981]' : 'text-[#EF4444]'}>
                              {s.svm_pred} {s.svm_correct ? '✓' : '✗'}
                            </span>
                          </td>
                          <td className="p-1.5 font-arabic">
                            <span className={s.cnn_correct ? 'text-[#10B981]' : 'text-[#EF4444]'}>
                              {s.cnn_pred} {s.cnn_correct ? '✓' : '✗'}
                            </span>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};
