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

export interface RecognitionStreamResult {
  recognized_sign: string;
  confidence: number;
  is_valid: boolean;
  active_model: string;
  num_classes: number;
  latency_ms: number;
  fps: number;
  sentence: string;
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
}

