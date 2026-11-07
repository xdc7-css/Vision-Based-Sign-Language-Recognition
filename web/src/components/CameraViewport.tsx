import React, { useEffect, useRef, useState, useCallback } from 'react';
import { Camera, CameraOff, RefreshCw, Eye, EyeOff, FlipHorizontal } from 'lucide-react';
import type { NormalizedPoint } from '../types';

interface CameraViewportProps {
  onFrameCapture?: (blob: Blob, frameId: number, timestamp: number) => void;
  isProcessing?: boolean; // Signal from parent to apply backpressure
  landmarks?: {
    left_hand: NormalizedPoint[] | null;
    right_hand: NormalizedPoint[] | null;
    face: NormalizedPoint[] | null;
    pose: NormalizedPoint[] | null;
  };
  fps?: number;
  latencyMs?: number;
  isStreaming?: boolean;
  statusLabel?: string;
  stateProgress?: number;
}

// MediaPipe 21 Hand Landmarks Skeleton Connections
const HAND_CONNECTIONS: [number, number][] = [
  // Thumb
  [0, 1], [1, 2], [2, 3], [3, 4],
  // Index
  [0, 5], [5, 6], [6, 7], [7, 8],
  // Middle
  [0, 9], [9, 10], [10, 11], [11, 12],
  // Ring
  [0, 13], [13, 14], [14, 15], [15, 16],
  // Pinky
  [0, 17], [17, 18], [18, 19], [19, 20],
  // Palm Base
  [5, 9], [9, 13], [13, 17],
];

export const CameraViewport: React.FC<CameraViewportProps> = ({
  onFrameCapture,
  isProcessing = false,
  landmarks,
  fps = 0,
  latencyMs = 0,
  isStreaming = false,
  statusLabel,
  stateProgress = 0,
}) => {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const captureCanvasRef = useRef<HTMLCanvasElement | null>(null);

  const [devices, setDevices] = useState<MediaDeviceInfo[]>([]);
  const [selectedDeviceId, setSelectedDeviceId] = useState<string>('');
  const [isCameraActive, setIsCameraActive] = useState<boolean>(false);
  const [showLandmarks, setShowLandmarks] = useState<boolean>(true);
  const [mirror, setMirror] = useState<boolean>(true);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  const frameIdCounterRef = useRef<number>(0);
  const lastCaptureTimeRef = useRef<number>(0);
  const isProcessingRef = useRef<boolean>(isProcessing);

  useEffect(() => {
    isProcessingRef.current = isProcessing;
  }, [isProcessing]);

  // Discover connected camera devices
  const enumerateCameras = useCallback(async () => {
    try {
      const allDevices = await navigator.mediaDevices.enumerateDevices();
      const videoDevs = allDevices.filter((d) => d.kind === 'videoinput');
      setDevices(videoDevs);
      if (videoDevs.length > 0 && !selectedDeviceId) {
        setSelectedDeviceId(videoDevs[0].deviceId);
      }
    } catch (err) {
      console.error('Error discovering camera devices:', err);
    }
  }, [selectedDeviceId]);

  useEffect(() => {
    enumerateCameras();
  }, [enumerateCameras]);

  // Start Camera Stream
  const startCamera = async () => {
    setErrorMsg(null);
    try {
      if (videoRef.current?.srcObject) {
        const tracks = (videoRef.current.srcObject as MediaStream).getTracks();
        tracks.forEach((t) => t.stop());
      }

      const constraints: MediaStreamConstraints = {
        video: {
          deviceId: selectedDeviceId ? { exact: selectedDeviceId } : undefined,
          width: { ideal: 640 },
          height: { ideal: 480 },
          frameRate: { ideal: 30, max: 30 },
        },
        audio: false,
      };

      const stream = await navigator.mediaDevices.getUserMedia(constraints);
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
        setIsCameraActive(true);
      }
      enumerateCameras();
    } catch (err: any) {
      console.error('Camera access error:', err);
      setErrorMsg(
        err.name === 'NotAllowedError'
          ? 'Camera permission denied. Please allow camera access in your browser settings.'
          : `Could not start camera: ${err.message || 'Device in use or not found'}`
      );
      setIsCameraActive(false);
    }
  };

  // Stop Camera Stream
  const stopCamera = () => {
    if (videoRef.current?.srcObject) {
      const tracks = (videoRef.current.srcObject as MediaStream).getTracks();
      tracks.forEach((t) => t.stop());
      videoRef.current.srcObject = null;
    }
    setIsCameraActive(false);

    // Clear overlay canvas
    if (canvasRef.current) {
      const ctx = canvasRef.current.getContext('2d');
      ctx?.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
    }
  };

  // Auto-start camera on mount
  useEffect(() => {
    startCamera();
    return () => {
      stopCamera();
    };
  }, [selectedDeviceId]);

  // High-performance, bounded frame capture loop with requestVideoFrameCallback + Backpressure
  useEffect(() => {
    if (!isCameraActive || !onFrameCapture) return;

    if (!captureCanvasRef.current) {
      captureCanvasRef.current = document.createElement('canvas');
      captureCanvasRef.current.width = 640;
      captureCanvasRef.current.height = 480;
    }
    const capCanvas = captureCanvasRef.current;
    const capCtx = capCanvas.getContext('2d', { willReadFrequently: true });

    let isCapturing = true;
    let animFrameId: number | null = null;
    let rvfcId: number | null = null;

    const MIN_INTERVAL_MS = 48; // Maximum ~20 FPS capture rate to match laptop inference capability

    const captureCurrentFrame = (timestamp: number) => {
      if (!isCapturing || !videoRef.current || videoRef.current.readyState < 2) return;

      // Backpressure Check: if previous frame is still in-flight on backend, DROP this capture!
      // This enforces strict latest-frame-wins policy and eliminates stale queue lag.
      if (isProcessingRef.current) {
        return;
      }

      // Rate limit check
      if (timestamp - lastCaptureTimeRef.current < MIN_INTERVAL_MS) {
        return;
      }

      if (capCtx && videoRef.current) {
        lastCaptureTimeRef.current = timestamp;
        frameIdCounterRef.current += 1;
        const currentFrameId = frameIdCounterRef.current;
        const clientTimestamp = performance.now();

        // Draw video frame to offscreen canvas
        capCtx.drawImage(videoRef.current, 0, 0, capCanvas.width, capCanvas.height);

        capCanvas.toBlob(
          (blob) => {
            if (blob && isCapturing) {
              onFrameCapture(blob, currentFrameId, clientTimestamp);
            }
          },
          'image/jpeg',
          0.68 // Optimized compression quality: 60% smaller payload, identical landmark precision
        );
      }
    };

    // Use modern requestVideoFrameCallback when supported by browser
    const videoElem = videoRef.current;
    const hasRVFC = videoElem && typeof (videoElem as any).requestVideoFrameCallback === 'function';

    if (hasRVFC && videoElem) {
      const onVideoFrame = (now: DOMHighResTimeStamp) => {
        if (!isCapturing) return;
        captureCurrentFrame(now);
        rvfcId = (videoElem as any).requestVideoFrameCallback(onVideoFrame);
      };
      rvfcId = (videoElem as any).requestVideoFrameCallback(onVideoFrame);
    } else {
      // Standard requestAnimationFrame fallback
      const loop = (now: DOMHighResTimeStamp) => {
        if (!isCapturing) return;
        captureCurrentFrame(now);
        animFrameId = requestAnimationFrame(loop);
      };
      animFrameId = requestAnimationFrame(loop);
    }

    return () => {
      isCapturing = false;
      if (animFrameId !== null) cancelAnimationFrame(animFrameId);
      if (rvfcId !== null && videoElem && typeof (videoElem as any).cancelVideoFrameCallback === 'function') {
        (videoElem as any).cancelVideoFrameCallback(rvfcId);
      }
    };
  }, [isCameraActive, onFrameCapture]);

  // Render Landmarks on Canvas with Batched Drawing
  useEffect(() => {
    const canvas = canvasRef.current;
    const video = videoRef.current;
    if (!canvas || !video || !isCameraActive) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Match canvas dimensions to video display
    const width = video.videoWidth || 640;
    const height = video.videoHeight || 480;
    if (canvas.width !== width || canvas.height !== height) {
      canvas.width = width;
      canvas.height = height;
    }

    ctx.clearRect(0, 0, width, height);
    if (!showLandmarks || !landmarks) return;

    ctx.save();
    if (mirror) {
      ctx.translate(width, 0);
      ctx.scale(-1, 1);
    }

    const drawHand = (points: NormalizedPoint[] | null, isLeft: boolean) => {
      if (!points || points.length !== 21) return;

      // Draw connections in a single batch
      ctx.strokeStyle = isLeft ? '#2DD4BF' : '#14B8A6';
      ctx.lineWidth = 3;
      ctx.lineCap = 'round';
      ctx.beginPath();

      HAND_CONNECTIONS.forEach(([startIdx, endIdx]) => {
        const p1 = points[startIdx];
        const p2 = points[endIdx];
        if (p1 && p2) {
          ctx.moveTo(p1.x * width, p1.y * height);
          ctx.lineTo(p2.x * width, p2.y * height);
        }
      });
      ctx.stroke();

      // Draw standard knuckles (white) in batch
      ctx.fillStyle = '#FFFFFF';
      ctx.beginPath();
      points.forEach((pt, idx) => {
        if (idx !== 4 && idx !== 8 && idx !== 12 && idx !== 16 && idx !== 20) {
          ctx.moveTo(pt.x * width + 3.5, pt.y * height);
          ctx.arc(pt.x * width, pt.y * height, 3.5, 0, 2 * Math.PI);
        }
      });
      ctx.fill();

      // Draw fingertips (coral red) in batch
      ctx.fillStyle = '#EF4444';
      ctx.beginPath();
      [4, 8, 12, 16, 20].forEach((idx) => {
        const pt = points[idx];
        if (pt) {
          ctx.moveTo(pt.x * width + 4.5, pt.y * height);
          ctx.arc(pt.x * width, pt.y * height, 4.5, 0, 2 * Math.PI);
        }
      });
      ctx.fill();
    };

    // Draw Face Landmarks in a single batched path
    if (landmarks.face && landmarks.face.length > 0) {
      ctx.fillStyle = 'rgba(45, 212, 191, 0.75)';
      ctx.beginPath();
      landmarks.face.forEach((pt) => {
        ctx.moveTo(pt.x * width + 2, pt.y * height);
        ctx.arc(pt.x * width, pt.y * height, 2, 0, 2 * Math.PI);
      });
      ctx.fill();
    }

    // Draw Upper-Body Pose Keypoints in a single batched path
    if (landmarks.pose && landmarks.pose.length > 0) {
      ctx.fillStyle = '#3B82F6';
      ctx.beginPath();
      landmarks.pose.forEach((pt) => {
        ctx.moveTo(pt.x * width + 4, pt.y * height);
        ctx.arc(pt.x * width, pt.y * height, 4, 0, 2 * Math.PI);
      });
      ctx.fill();
    }

    // Draw Left & Right Hands
    drawHand(landmarks.left_hand, true);
    drawHand(landmarks.right_hand, false);

    ctx.restore();
  }, [landmarks, showLandmarks, mirror, isCameraActive]);

  return (
    <div className="flex flex-col bg-[#1B2330] border border-[#2B3442] rounded-xl overflow-hidden shadow-lg">
      {/* Video Viewport Container */}
      <div className="relative aspect-video w-full bg-[#101419] flex items-center justify-center overflow-hidden">
        <video
          ref={videoRef}
          autoPlay
          playsInline
          muted
          className={`w-full h-full object-cover ${mirror ? 'scale-x-[-1]' : ''}`}
        />
        <canvas
          ref={canvasRef}
          className="absolute inset-0 w-full h-full object-cover pointer-events-none"
        />

        {/* Telemetry Chips Top Left */}
        <div className="absolute top-3 left-3 flex flex-wrap items-center gap-2">
          <div
            className={`flex items-center space-x-1.5 px-2.5 py-1 rounded-full text-xs font-semibold backdrop-blur-md ${
              isCameraActive && isStreaming
                ? 'bg-[#10B981]/20 text-[#10B981] border border-[#10B981]/40'
                : isCameraActive
                ? 'bg-[#2563EB]/20 text-[#2563EB] border border-[#2563EB]/40'
                : 'bg-[#EF4444]/20 text-[#EF4444] border border-[#EF4444]/40'
            }`}
          >
            <span
              className={`w-2 h-2 rounded-full ${
                isCameraActive ? 'bg-current animate-ping' : 'bg-current'
              }`}
            />
            <span>
              {isCameraActive && isStreaming ? 'Live Streaming' : isCameraActive ? 'Camera Active' : 'Standby'}
            </span>
          </div>

          {isCameraActive && (
            <div className="bg-[#151B24]/80 text-[#94A3B8] border border-[#2B3442] px-2.5 py-1 rounded-full text-xs font-mono backdrop-blur-md">
              {fps.toFixed(1)} FPS • {latencyMs.toFixed(0)} ms E2E
            </div>
          )}

          {statusLabel && (
            <div className="bg-[#151B24]/90 text-[#38BDF8] border border-[#38BDF8]/30 px-2.5 py-1 rounded-full text-xs font-medium backdrop-blur-md flex items-center space-x-1.5">
              {stateProgress > 0 && stateProgress < 1 && (
                <span className="w-1.5 h-1.5 rounded-full bg-[#38BDF8] animate-pulse" />
              )}
              <span>{statusLabel}</span>
            </div>
          )}
        </div>

        {/* Error Notification Overlay */}
        {errorMsg && (
          <div className="absolute inset-0 bg-[#101419]/90 flex flex-col items-center justify-center p-6 text-center z-20">
            <CameraOff className="w-12 h-12 text-[#EF4444] mb-3" />
            <h3 className="text-base font-bold text-[#F3F4F6] mb-1">Camera Initialization Error</h3>
            <p className="text-xs text-[#94A3B8] max-w-md mb-4">{errorMsg}</p>
            <button
              onClick={startCamera}
              className="px-4 py-2 bg-[#2563EB] hover:bg-[#3B82F6] text-white rounded-lg text-xs font-semibold flex items-center space-x-2 transition-colors"
            >
              <RefreshCw className="w-3.5 h-3.5" />
              <span>Retry Camera</span>
            </button>
          </div>
        )}
      </div>

      {/* Camera Controls Bar */}
      <div className="p-3 bg-[#151B24] border-t border-[#2B3442] flex flex-wrap items-center justify-between gap-3 text-xs">
        {/* Device Selector */}
        <div className="flex items-center space-x-2">
          <label className="text-[#94A3B8] font-medium hidden sm:inline">Camera:</label>
          <select
            value={selectedDeviceId}
            onChange={(e) => setSelectedDeviceId(e.target.value)}
            className="bg-[#1B2330] border border-[#2B3442] text-[#F3F4F6] rounded-lg px-2.5 py-1.5 focus:outline-none focus:border-[#2563EB] max-w-[200px] truncate"
          >
            {devices.map((d, idx) => (
              <option key={d.deviceId || idx} value={d.deviceId}>
                {d.label || `Camera ${idx + 1}`}
              </option>
            ))}
          </select>
        </div>

        {/* Viewport Toggles */}
        <div className="flex items-center space-x-2">
          <button
            onClick={() => setShowLandmarks((prev) => !prev)}
            className={`px-3 py-1.5 rounded-lg border font-semibold flex items-center space-x-1.5 transition-colors ${
              showLandmarks
                ? 'bg-[#14B8A6]/20 border-[#14B8A6]/40 text-[#14B8A6]'
                : 'bg-[#1B2330] border-[#2B3442] text-[#94A3B8] hover:text-[#F3F4F6]'
            }`}
            title="Toggle landmark skeletons"
          >
            {showLandmarks ? <Eye className="w-3.5 h-3.5" /> : <EyeOff className="w-3.5 h-3.5" />}
            <span>Landmarks</span>
          </button>

          <button
            onClick={() => setMirror((prev) => !prev)}
            className={`px-3 py-1.5 rounded-lg border font-semibold flex items-center space-x-1.5 transition-colors ${
              mirror
                ? 'bg-[#2563EB]/20 border-[#2563EB]/40 text-[#2563EB]'
                : 'bg-[#1B2330] border-[#2B3442] text-[#94A3B8] hover:text-[#F3F4F6]'
            }`}
            title="Mirror preview"
          >
            <FlipHorizontal className="w-3.5 h-3.5" />
            <span>Mirror</span>
          </button>

          <button
            onClick={isCameraActive ? stopCamera : startCamera}
            className={`px-3 py-1.5 rounded-lg font-semibold flex items-center space-x-1.5 transition-colors ${
              isCameraActive
                ? 'bg-[#EF4444]/20 border border-[#EF4444]/40 text-[#EF4444] hover:bg-[#EF4444]/30'
                : 'bg-[#10B981]/20 border border-[#10B981]/40 text-[#10B981] hover:bg-[#10B981]/30'
            }`}
          >
            <Camera className="w-3.5 h-3.5" />
            <span>{isCameraActive ? 'Stop Camera' : 'Start Camera'}</span>
          </button>
        </div>
      </div>
    </div>
  );
};
