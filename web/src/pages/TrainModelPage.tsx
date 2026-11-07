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
} from 'lucide-react';
import {
  fetchTrainingReadiness,
  startTraining,
  fetchJobStatus,
  cancelTrainingJob,
} from '../services/api';
import type { TrainingJobStatus, TrainingReadiness } from '../types';

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

  // Load readiness
  const loadReadiness = async () => {
    try {
      const data = await fetchTrainingReadiness();
      setReadiness(data);
    } catch (err) {
      console.error('Failed to fetch training readiness:', err);
    }
  };

  useEffect(() => {
    loadReadiness();
  }, []);

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
    </div>
  );
};
