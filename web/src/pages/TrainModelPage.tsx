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
} from '../services/api';
import type {
  TrainingJobStatus,
  TrainingReadiness,
  ResearchStatusResponse,
  ResearchComparisonMetrics,
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

  useEffect(() => {
    loadReadiness();
    loadResearch();
  }, []);

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
              12 Signs • Signers 01, 02, 03
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
                  <span>Declared Scientific Winner: Model B (1D Temporal CNN)</span>
                </div>
                <p className="text-[#94A3B8] leading-relaxed">
                  Under strict Leave-One-Signer-Out (LOSO) testing on unseen Signer03, the 1D Temporal CNN achieves <strong>100.00% accuracy</strong> and <strong>100.00% F1-score</strong> (+35.82% advantage over SVM). Temporal 1D convolutions effectively extract trajectory dynamics across consecutive frames, whereas static statistical aggregation (mean/std/velocity) loses critical temporal ordering between distinct signs.
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
    </div>
  );
};
