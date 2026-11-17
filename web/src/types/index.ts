export type TabType = 'recognize' | 'teach' | 'train';

export interface GestureClass {
  id: string;
  name: string;
  category: 'static' | 'dynamic' | 'both' | 'alphabet';
  description?: string;
  is_active: boolean;
  is_default: boolean;
  sample_count: number;
  is_ready: boolean;
  created_at: number;
  updated_at: number;
}

export interface DatasetClassDetail {
  id: string;
  name: string;
  category: string;
  multimodal_samples: number;
  legacy_samples: number;
  total_samples: number;
  is_trainable: boolean;
}

export interface DatasetSummary {
  total_classes: number;
  total_samples: number;
  trainable_classes: number;
  multimodal_samples: number;
  legacy_samples: number;
  classes: DatasetClassDetail[];
}

export interface SampleInfo {
  sample_id: string;
  timestamp: number;
  schema_version: string;
  num_frames: number;
  has_face: boolean;
  has_left: boolean;
  has_right: boolean;
  handedness: string;
}

export interface ActiveModelInfo {
  id?: string;
  name: string;
  model_type: string;
  classes: string[];
  num_classes: number;
  accuracy?: number;
  is_trained: boolean;
  updated_at?: number;
}

export interface TrainingReadiness {
  is_trainable: boolean;
  trainable_classes_count: number;
  total_multimodal_samples: number;
  accepted_classes: Record<string, number>;
  insufficient_classes: Record<string, number>;
  model_status: 'Ready' | 'Outdated' | 'Untrained';
  reason?: string;
  last_trained_timestamp?: number;
}

export interface TrainingJobStatus {
  job_id: string;
  status: 'idle' | 'running' | 'completed' | 'failed' | 'cancelled';
  progress: number;
  current_epoch: number;
  max_epochs: number;
  loss: number;
  accuracy: number;
  metrics?: {
    accuracy: number;
    precision: number;
    recall: number;
    f1_score: number;
    confusion_matrix?: number[][];
    class_names?: string[];
  };
  error?: string;
}

export interface NormalizedPoint {
  x: number;
  y: number;
  z: number;
}

export interface RecognitionPerformanceMetrics {
  decode_ms: number;
  hands_ms: number;
  face_ms: number;
  pose_ms: number;
  infer_ms: number;
  backend_total_ms: number;
}

export interface RecognitionStreamResult {
  frame_id?: number;
  client_time_ms?: number;
  recognized_sign: string;
  confidence: number;
  is_valid: boolean;
  state?: 'idle' | 'candidate' | 'confirmed' | 'emitted' | 'wait_for_release' | 'ready';
  state_progress?: number;
  status_label?: string;
  is_holding?: boolean;
  is_neutral?: boolean;
  emitted_word?: string | null;
  active_model: string;
  num_classes: number;
  latency_ms: number;
  fps: number;
  sentence: string;
  dropped_frames?: number;
  landmarks: {
    left_hand: NormalizedPoint[] | null;
    right_hand: NormalizedPoint[] | null;
    face: NormalizedPoint[] | null;
    pose: NormalizedPoint[] | null;
  };
  indicators: {
    has_left: boolean;
    has_right: boolean;
    has_face: boolean;
    has_pose: boolean;
    mouth_openness: number;
    eyebrow_elevation: number;
  };
  perf?: RecognitionPerformanceMetrics;
}

export interface ResearchDatasetStatus {
  is_prepared: boolean;
  total_samples: number;
  num_classes: number;
  classes: string[];
  signers: string[];
  samples_per_signer: Record<string, number>;
  class_counts: Record<string, number>;
  has_raw_videos: boolean;
  raw_video_count: number;
  has_splits: boolean;
  feature_dim: number;
  seq_len: number;
}

export interface ResearchModelStatus {
  is_trained: boolean;
  metrics?: Record<string, any> | null;
}

export interface ResearchComparisonMetrics {
  timestamp: number;
  protocol: string;
  fold: string;
  train_signers: string[];
  test_signer: string;
  train_samples: number;
  test_samples: number;
  num_classes: number;
  classes: string[];
  svm: {
    model_name: string;
    algorithm: string;
    input_type: string;
    accuracy: number;
    precision_macro: number;
    recall_macro: number;
    f1_macro: number;
    training_time_sec: number;
    inference_latency_ms: number;
    model_size_kb: number;
    test_signer: string;
  };
  cnn: {
    model_name: string;
    algorithm: string;
    input_type: string;
    accuracy: number;
    precision_macro: number;
    recall_macro: number;
    f1_macro: number;
    training_time_sec: number;
    inference_latency_ms: number;
    model_size_kb: number;
    test_signer: string;
  };
  winner: {
    best_model: string;
    f1_difference: number;
    accuracy_difference: number;
    rationale: string;
    speed_advantage: string;
    size_advantage: string;
  };
  total_benchmark_time_sec: number;
  confusion_matrices?: {
    classes: string[];
    svm_confusion_matrix: number[][];
    cnn_confusion_matrix: number[][];
  };
}

export interface ResearchStatusResponse {
  dataset: ResearchDatasetStatus;
  models: {
    svm: ResearchModelStatus;
    cnn: ResearchModelStatus;
  };
  comparison: ResearchComparisonMetrics | null;
  active_job?: {
    job: string;
    status: string;
    error?: string;
  } | null;
}

export interface AvailableModel {
  id: string;
  name: string;
  type: string;
  num_classes: number;
  is_trained: boolean;
  is_active: boolean;
}


