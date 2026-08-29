import { FilesetResolver, PoseLandmarker } from "/assets/mediapipe/vision_bundle.mjs";
import { formatMetric, summarizePoseFrames } from "/assets/analyzer-core.mjs";

const $ = (selector) => document.querySelector(selector);
const video = $("#video-preview");
const input = $("#video-file");
const button = $("#analyze-button");
const message = $("#analysis-message");
const progressWrap = $("#analysis-progress");
const progressBar = $("#progress-bar");
const progressLabel = $("#progress-label");
const progressValue = $("#progress-value");
let objectUrl = null;
let poseLandmarker = null;
let inferenceTimestamp = 0;

if (/MicroMessenger/i.test(navigator.userAgent)) document.documentElement.classList.add("wechat-browser");

function setProgress(value, label) {
  const percent = Math.round(value);
  progressWrap.hidden = false;
  progressBar.value = percent;
  progressValue.textContent = `${percent}%`;
  progressLabel.textContent = label;
}

function waitFor(target, event) {
  return new Promise((resolve, reject) => {
    const timeout = setTimeout(() => reject(new Error("视频读取超时")), 10000);
    target.addEventListener(event, () => { clearTimeout(timeout); resolve(); }, { once: true });
  });
}

async function seek(time) {
  if (Math.abs(video.currentTime - time) < 0.005) return;
  const ready = waitFor(video, "seeked");
  video.currentTime = Math.min(time, Math.max(0, video.duration - 0.001));
  await ready;
}

async function loadModel() {
  if (poseLandmarker) return poseLandmarker;
  setProgress(4, "正在加载本地 AI 模型");
  const vision = await FilesetResolver.forVisionTasks("/assets/mediapipe/wasm");
  const options = {
    baseOptions: { modelAssetPath: "/assets/models/pose_landmarker_lite.task", delegate: "GPU" },
    runningMode: "VIDEO", numPoses: 1, minPoseDetectionConfidence: 0.45, minTrackingConfidence: 0.45,
  };
  try {
    poseLandmarker = await PoseLandmarker.createFromOptions(vision, options);
  } catch (gpuError) {
    console.warn("GPU delegate unavailable; falling back to CPU.", gpuError);
    options.baseOptions.delegate = "CPU";
    poseLandmarker = await PoseLandmarker.createFromOptions(vision, options);
  }
  return poseLandmarker;
}

input.addEventListener("change", async () => {
  const file = input.files?.[0];
  if (!file) return;
  if (file.size > 200 * 1024 * 1024) {
    message.textContent = "视频超过 200MB，请先裁剪到 30 秒左右再分析。";
    input.value = "";
    return;
  }
  if (objectUrl) URL.revokeObjectURL(objectUrl);
  objectUrl = URL.createObjectURL(file);
  video.src = objectUrl;
  video.hidden = false;
  message.textContent = "正在读取视频信息…";
  try {
    if (video.readyState < 1) await waitFor(video, "loadedmetadata");
    if (!Number.isFinite(video.duration) || video.duration <= 0) throw new Error("浏览器无法解码这个视频");
    $("#file-meta").textContent = `${file.name} · ${(file.size / 1024 / 1024).toFixed(1)}MB · ${video.duration.toFixed(1)}秒 · ${video.videoWidth}×${video.videoHeight}`;
    message.textContent = video.duration > 30 ? "视频较长，将抽样分析前 30 秒。" : "视频已就绪，点击开始分析。";
    button.disabled = false;
  } catch (error) {
    message.textContent = `${error.message}。建议使用 MP4（H.264）格式。`;
    button.disabled = true;
  }
});

button.addEventListener("click", async () => {
  button.disabled = true;
  $("#analysis-result").hidden = true;
  $("#empty-state").hidden = false;
  $("#result-panel").classList.add("empty");
  try {
    await loadModel();
    const duration = Math.min(video.duration, 30);
    const sampleCount = Math.min(180, Math.max(24, Math.ceil(duration * 10)));
    const frames = [];
    for (let index = 0; index < sampleCount; index += 1) {
      const time = sampleCount === 1 ? 0 : duration * index / (sampleCount - 1);
      await seek(time);
      inferenceTimestamp = Math.max(performance.now(), inferenceTimestamp + 1);
      const result = poseLandmarker.detectForVideo(video, inferenceTimestamp);
      if (result.landmarks?.[0]) frames.push({ time, landmarks: result.landmarks[0] });
      setProgress(12 + 78 * (index + 1) / sampleCount, `正在识别动作 ${index + 1}/${sampleCount}`);
      if (index % 3 === 0) await new Promise(requestAnimationFrame);
    }
    const hand = document.querySelector('input[name="hand"]:checked').value;
    const summary = summarizePoseFrames(frames, hand);
    setProgress(94, "正在生成训练建议");
    await renderResult(summary, hand);
    setProgress(100, "分析完成");
    message.textContent = "分析完成。你可以更换视频继续测试。";
  } catch (error) {
    console.error(error);
    message.textContent = `分析未完成：${error.message || "浏览器不支持当前模型"}。请优先使用最新版 Chrome、Edge 或 Safari。`;
    progressLabel.textContent = "分析失败";
  } finally {
    button.disabled = false;
  }
});

async function renderResult(summary, hand) {
  const percent = Math.round(summary.validRatio * 100);
  $("#quality-score").textContent = `${percent}%`;
  $("#quality-label").textContent = percent >= 80 ? "识别稳定" : percent >= 55 ? "基本可用" : "建议重拍";
  $("#metric-elbow").textContent = formatMetric(summary.metrics.elbowRelease);
  $("#metric-knee").textContent = formatMetric(summary.metrics.kneeLowest);
  $("#metric-range").textContent = formatMetric(summary.metrics.kneeRange);
  $("#metric-trunk").textContent = formatMetric(summary.metrics.trunkRelease);
  $("#advice-list").replaceChildren(...summary.suggestions.map((item, index) => {
    const article = document.createElement("article");
    article.innerHTML = `<b>${String(index + 1).padStart(2, "0")}</b><div><h4></h4><p></p></div>`;
    article.querySelector("h4").textContent = item.title;
    article.querySelector("p").textContent = item.detail;
    return article;
  }));
  if (summary.release) {
    await seek(summary.release.time);
    drawKeyframe(summary.release.landmarks, hand);
  }
  $("#empty-state").hidden = true;
  $("#analysis-result").hidden = false;
  $("#result-panel").classList.remove("empty");
}

function drawKeyframe(landmarks, hand) {
  const canvas = $("#keyframe-canvas");
  const maxWidth = 820;
  const scale = Math.min(1, maxWidth / video.videoWidth);
  canvas.width = Math.round(video.videoWidth * scale);
  canvas.height = Math.round(video.videoHeight * scale);
  const context = canvas.getContext("2d");
  context.drawImage(video, 0, 0, canvas.width, canvas.height);
  const connections = [[11,12],[11,13],[13,15],[12,14],[14,16],[11,23],[12,24],[23,24],[23,25],[25,27],[24,26],[26,28]];
  context.lineCap = "round";
  context.lineWidth = Math.max(2, canvas.width / 260);
  for (const [a, b] of connections) {
    const active = hand === "right" ? [12,14,16,24,26,28].includes(a) && [12,14,16,24,26,28].includes(b) : [11,13,15,23,25,27].includes(a) && [11,13,15,23,25,27].includes(b);
    context.strokeStyle = active ? "#ff7a31" : "rgba(86,229,208,.9)";
    context.beginPath();
    context.moveTo(landmarks[a].x * canvas.width, landmarks[a].y * canvas.height);
    context.lineTo(landmarks[b].x * canvas.width, landmarks[b].y * canvas.height);
    context.stroke();
  }
}
