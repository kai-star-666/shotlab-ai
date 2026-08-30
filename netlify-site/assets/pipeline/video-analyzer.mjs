import { FilesetResolver, PoseLandmarker } from "/assets/mediapipe/vision_bundle.mjs?v=20260830-1";
import { preprocessLandmarkFrames, LANDMARK_FILTER_VERSION } from "./landmark-series.mjs";

export const PIPELINE_MANIFEST = Object.freeze({
  pipelineVersion: "3.0.0",
  sampleIntervalMs: 100,
  maxDurationMs: 30000,
  maxCanvasEdge: 720,
  delegate: "CPU",
  modelSha256: "59929e1d1ee95287735ddd833b19cf4ac46d29bc7afddbbf6753c459690d574a",
  visionBundleSha256: "d885630c297c0b20b1fe86096cb06291c4c8080876f27852e724f24ac603713f",
  landmarkFilterVersion: LANDMARK_FILTER_VERSION,
});

let visionPromise;
const vision = () => visionPromise ||= FilesetResolver.forVisionTasks("/assets/mediapipe/wasm");

function waitFor(target, event) {
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error("视频读取超时")), 10000);
    target.addEventListener(event, () => { clearTimeout(timer); resolve(); }, { once: true });
  });
}

async function seek(video, timeSeconds) {
  const safe = Math.min(timeSeconds, Math.max(0, video.duration - 0.001));
  if (Math.abs(video.currentTime - safe) < 0.002) return;
  const ready = waitFor(video, "seeked");
  video.currentTime = safe;
  await ready;
}

function canvasSize(width, height) {
  const scale = Math.min(1, PIPELINE_MANIFEST.maxCanvasEdge / Math.max(width, height));
  const even = (value) => Math.max(2, Math.floor(value * scale / 2) * 2);
  return { width: even(width), height: even(height) };
}

function canonical(value) {
  if (Array.isArray(value)) return value.map(canonical);
  if (value && typeof value === "object") return Object.fromEntries(Object.keys(value).sort().map((key) => [key, canonical(value[key])]));
  return Number.isFinite(value) && typeof value === "number" ? Number(value.toFixed(8)) : value;
}

export async function sha256Json(value) {
  const bytes = new TextEncoder().encode(JSON.stringify(canonical(value)));
  const digest = await crypto.subtle.digest("SHA-256", bytes);
  return [...new Uint8Array(digest)].map((byte) => byte.toString(16).padStart(2, "0")).join("");
}

export async function analyzeVideoDeterministic(video, { shootingHand = "right", sampleIntervalMs = 100, maxDurationMs = 30000, onProgress = () => {} } = {}) {
  if (!Number.isFinite(video.duration) || !video.videoWidth || !video.videoHeight) throw new Error("视频元数据不可用");
  if (sampleIntervalMs !== 100) throw new Error("当前管线只允许100ms固定采样");
  const durationMs = Math.min(maxDurationMs, Math.floor(video.duration * 1000));
  const sampleCount = Math.floor(durationMs / sampleIntervalMs) + 1;
  const size = canvasSize(video.videoWidth, video.videoHeight);
  const canvas = document.createElement("canvas");
  canvas.width = size.width; canvas.height = size.height;
  const context = canvas.getContext("2d", { alpha: false, willReadFrequently: false });
  const signatureCanvas = document.createElement("canvas"); signatureCanvas.width = 16; signatureCanvas.height = 16;
  const signatureContext = signatureCanvas.getContext("2d", { alpha: false, willReadFrequently: true });
  const landmarker = await PoseLandmarker.createFromOptions(await vision(), {
    baseOptions: { modelAssetPath: "/assets/models/pose_landmarker_lite.task" },
    runningMode: "VIDEO", numPoses: 1,
    minPoseDetectionConfidence: 0.5, minPosePresenceConfidence: 0.5, minTrackingConfidence: 0.5,
  });
  const decoded = [];
  const decodedSignatures = [];
  try {
    for (let sampleIndex = 0; sampleIndex < sampleCount; sampleIndex += 1) {
      const timeMs = sampleIndex * sampleIntervalMs;
      await seek(video, timeMs / 1000);
      context.drawImage(video, 0, 0, size.width, size.height);
      signatureContext.drawImage(canvas, 0, 0, 16, 16);
      decodedSignatures.push({ sampleIndex, timeMs, pixels: [...signatureContext.getImageData(0, 0, 16, 16).data] });
      const result = landmarker.detectForVideo(canvas, timeMs);
      decoded.push({ sampleIndex, timeMs, time: timeMs / 1000, landmarks: result.landmarks?.[0] || [] });
      onProgress(sampleIndex + 1, sampleCount);
      if (sampleIndex % 3 === 0) await new Promise(requestAnimationFrame);
    }
  } finally {
    landmarker.close();
  }
  const processed = preprocessLandmarkFrames(decoded);
  const hashes = {
    decodedFrames: await sha256Json({ size, frames: decodedSignatures }),
    rawLandmarks: await sha256Json(processed.raw),
    filteredLandmarks: await sha256Json(processed.filtered),
  };
  return { shootingHand, frames: processed.filtered, rawFrames: processed.raw, sampleCount, durationMs, canvasSize: size, hashes, manifest: PIPELINE_MANIFEST };
}
