import type {
  ActiveModelInfo,
  AvailableModel,
  DatasetSummary,
  GestureClass,
  IndependentEvalStatus,
  ResearchComparisonMetrics,
  ResearchStatusResponse,
  SampleInfo,
  TrainingJobStatus,
  TrainingReadiness,
  VolunteerInfo,
  VolunteerProgress,
} from '../types';

const API_BASE = '';

export async function fetchHealth(): Promise<{ status: string }> {
  const res = await fetch(`${API_BASE}/api/health`);
  return res.json();
}

export async function fetchSystemStatus(): Promise<{
  status: string;
  version: string;
  cpu_percent: number;
  memory_percent: number;
  active_model: string;
  total_gestures: number;
}> {
  const res = await fetch(`${API_BASE}/api/system/status`);
  return res.json();
}

// ── Gestures ─────────────────────────────────────────────────────────────

export async function fetchGestures(includeArchived = false): Promise<GestureClass[]> {
  const res = await fetch(`${API_BASE}/api/gestures?include_archived=${includeArchived}`);
  if (!res.ok) throw new Error('Failed to fetch gestures');
  return res.json();
}

export async function createGesture(payload: {
  name: string;
  category: string;
  description?: string;
}): Promise<GestureClass> {
  const res = await fetch(`${API_BASE}/api/gestures`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Failed to create gesture' }));
    throw new Error(err.detail || 'Failed to create gesture');
  }
  return res.json();
}

export async function updateGesture(
  id: string,
  payload: { name?: string; category?: string; description?: string }
): Promise<GestureClass> {
  const res = await fetch(`${API_BASE}/api/gestures/${id}`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Failed to update gesture' }));
    throw new Error(err.detail || 'Failed to update gesture');
  }
  return res.json();
}

export async function deleteGesture(id: string, purgeSamples = false): Promise<boolean> {
  const res = await fetch(`${API_BASE}/api/gestures/${id}?purge_samples=${purgeSamples}`, {
    method: 'DELETE',
  });
  return res.ok;
}

export async function archiveGesture(id: string): Promise<boolean> {
  const res = await fetch(`${API_BASE}/api/gestures/${id}/archive`, { method: 'POST' });
  return res.ok;
}

export async function restoreGesture(id: string): Promise<boolean> {
  const res = await fetch(`${API_BASE}/api/gestures/${id}/restore`, { method: 'POST' });
  return res.ok;
}

// ── Dataset ──────────────────────────────────────────────────────────────

export async function fetchDatasetSummary(): Promise<DatasetSummary> {
  const res = await fetch(`${API_BASE}/api/dataset/summary`);
  if (!res.ok) throw new Error('Failed to fetch dataset summary');
  return res.json();
}

export async function fetchGestureSamples(gestureId: string): Promise<SampleInfo[]> {
  const res = await fetch(`${API_BASE}/api/dataset/gestures/${gestureId}/samples`);
  if (!res.ok) throw new Error('Failed to fetch gesture samples');
  return res.json();
}

export async function deleteGestureSample(gestureId: string, sampleId: string): Promise<boolean> {
  const res = await fetch(`${API_BASE}/api/dataset/gestures/${gestureId}/samples/${sampleId}`, {
    method: 'DELETE',
  });
  return res.ok;
}

// ── Models ───────────────────────────────────────────────────────────────

export async function fetchActiveModel(): Promise<ActiveModelInfo> {
  const res = await fetch(`${API_BASE}/api/models/active`);
  if (!res.ok) throw new Error('Failed to fetch active model info');
  return res.json();
}

export async function reloadModel(): Promise<{ success: boolean }> {
  const res = await fetch(`${API_BASE}/api/models/reload`, { method: 'POST' });
  return res.json();
}

// ── Training ─────────────────────────────────────────────────────────────

export async function fetchTrainingReadiness(): Promise<TrainingReadiness> {
  const res = await fetch(`${API_BASE}/api/training/readiness`);
  if (!res.ok) throw new Error('Failed to fetch training readiness');
  return res.json();
}

export async function startTraining(payload: {
  model_type: string;
  epochs: number;
  batch_size: number;
  learning_rate: number;
  val_split: number;
  allowed_classes?: string[];
}): Promise<TrainingJobStatus> {
  const res = await fetch(`${API_BASE}/api/training/start`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Failed to start training' }));
    throw new Error(err.detail || 'Failed to start training');
  }
  return res.json();
}

export async function fetchJobStatus(jobId: string): Promise<TrainingJobStatus> {
  const res = await fetch(`${API_BASE}/api/training/jobs/${jobId}`);
  if (!res.ok) throw new Error('Failed to fetch job status');
  return res.json();
}

export async function cancelTrainingJob(jobId: string): Promise<boolean> {
  const res = await fetch(`${API_BASE}/api/training/jobs/${jobId}/cancel`, { method: 'POST' });
  return res.ok;
}

// ── Sentence Actions ─────────────────────────────────────────────────────

export async function addSentenceWord(word: string): Promise<{ sentence: string }> {
  const res = await fetch(`${API_BASE}/api/sentence/word`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ word }),
  });
  return res.json();
}

export async function deleteSentenceWord(): Promise<{ sentence: string }> {
  const res = await fetch(`${API_BASE}/api/sentence/delete`, { method: 'POST' });
  return res.json();
}

export async function clearSentence(): Promise<{ sentence: string }> {
  const res = await fetch(`${API_BASE}/api/sentence/clear`, { method: 'POST' });
  return res.json();
}

export async function fetchSentence(): Promise<{ sentence: string }> {
  const res = await fetch(`${API_BASE}/api/sentence`);
  return res.json();
}

// ── Models & Available Models ───────────────────────────────────────────

export async function fetchAvailableModels(): Promise<AvailableModel[]> {
  const res = await fetch(`${API_BASE}/api/models/available`);
  return res.json();
}

export async function selectActiveModel(modelId: string): Promise<{ success: boolean; active_model: ActiveModelInfo }> {
  const res = await fetch(`${API_BASE}/api/models/select`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ model_id: modelId }),
  });
  return res.json();
}

// ── Research Endpoints (KArSL-502 SVM vs CNN) ───────────────────────────

export async function fetchResearchStatus(): Promise<ResearchStatusResponse> {
  const res = await fetch(`${API_BASE}/api/research/status`);
  return res.json();
}

export async function prepareResearchDataset(samplesPerSignPerSigner = 15, force = false): Promise<any> {
  const res = await fetch(`${API_BASE}/api/research/prepare-dataset`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ samples_per_sign_per_signer: samplesPerSignPerSigner, force }),
  });
  if (!res.ok) {
    const err = await res.json();
    throw new Error(err.detail || 'Failed to prepare research dataset');
  }
  return res.json();
}

export async function trainResearchSVM(kernel = 'rbf', c = 2.0): Promise<any> {
  const res = await fetch(`${API_BASE}/api/research/train-svm`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ kernel, C: c }),
  });
  if (!res.ok) {
    const err = await res.json();
    throw new Error(err.detail || 'Failed to train SVM');
  }
  return res.json();
}

export async function trainResearchCNN(epochs = 35, batchSize = 16): Promise<any> {
  const res = await fetch(`${API_BASE}/api/research/train-cnn`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ epochs, batch_size: batchSize }),
  });
  if (!res.ok) {
    const err = await res.json();
    throw new Error(err.detail || 'Failed to train CNN');
  }
  return res.json();
}

export async function runResearchComparison(fold = 'test_Signer03'): Promise<ResearchComparisonMetrics> {
  const res = await fetch(`${API_BASE}/api/research/compare`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ fold }),
  });
  if (!res.ok) {
    const err = await res.json();
    throw new Error(err.detail || 'Failed to run comparison');
  }
  return res.json();
}

export async function fetchResearchComparison(): Promise<ResearchComparisonMetrics> {
  const res = await fetch(`${API_BASE}/api/research/comparison`);
  if (!res.ok) {
    const err = await res.json();
    throw new Error(err.detail || 'Failed to fetch comparison');
  }
  return res.json();
}

// ── Independent Human Evaluation Endpoints ─────────────────────────────────

export async function fetchIndependentEvalStatus(): Promise<IndependentEvalStatus> {
  const res = await fetch(`${API_BASE}/api/research/independent-eval/status`);
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || 'Failed to fetch independent evaluation status');
  }
  return res.json();
}

export async function listIndependentVolunteers(): Promise<VolunteerInfo[]> {
  const res = await fetch(`${API_BASE}/api/research/independent-eval/volunteers`);
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || 'Failed to list volunteers');
  }
  return res.json();
}

export async function registerIndependentVolunteer(
  name: string,
  notes = ''
): Promise<VolunteerInfo> {
  const res = await fetch(`${API_BASE}/api/research/independent-eval/volunteers`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ name, notes }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || 'Failed to register volunteer');
  }
  return res.json();
}

export async function fetchVolunteerProgress(
  volunteerId: string
): Promise<VolunteerProgress> {
  const res = await fetch(
    `${API_BASE}/api/research/independent-eval/volunteers/${volunteerId}/progress`
  );
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || 'Failed to fetch volunteer progress');
  }
  return res.json();
}

export async function runIndependentEvaluation(): Promise<IndependentEvalStatus> {
  const res = await fetch(`${API_BASE}/api/research/independent-eval/evaluate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || 'Failed to execute independent evaluation');
  }
  return res.json();
}


