import { FilesetResolver, PoseLandmarker } from "/assets/mediapipe/vision_bundle.mjs";
import { formatMetric, summarizePoseFrames } from "/assets/analyzer-core.mjs";

const $ = (selector) => document.querySelector(selector);
const clamp = (value, min, max) => Math.min(max, Math.max(min, value));
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
let replayAnimation = null;
let replayState = null;

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
      frames.push({ time, landmarks: result.landmarks?.[0] || [] });
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
  stopReplay();
  const percent = summary.capture.score;
  $("#quality-score").textContent = `${percent}/100`;
  $("#quality-label").textContent = `${summary.capture.confidence}可信度`;
  $("#metric-elbow").textContent = formatMetric(summary.metrics.elbowRelease);
  $("#metric-knee").textContent = formatMetric(summary.metrics.kneeLowest);
  $("#metric-range").textContent = formatMetric(summary.metrics.kneeRange);
  $("#metric-trunk").textContent = formatMetric(summary.metrics.trunkRelease);
  $("#analysis-summary-text").textContent = summary.capture.confidence === "低"
    ? `本次综合识别质量 ${percent}/100，有效帧 ${Math.round(summary.validRatio * 100)}%。当前证据不足，系统已停止输出精确动作纠正。`
    : `本次综合识别质量 ${percent}/100，共分析 ${summary.totalFrames} 个时间点，其中 ${summary.validFrames} 帧可用于动作判断。平均关键点可见度 ${Math.round(summary.capture.averageVisibility * 100)}%，人物高度约占画面 ${Math.round(summary.capture.bodyScale * 100)}%。以下结论优先基于相对幅度、变化趋势和动作连续性。`;

  $("#strength-list").replaceChildren(...summary.strengths.map((text) => makeCard("✓", "继续保持", text, "strength-card")));
  $("#advice-list").replaceChildren(...summary.priorities.map((issue, index) => {
    const article = document.createElement("article");
    article.className = "priority-card";
    const top = document.createElement("div");
    top.className = "priority-top";
    top.append(makeText("b", "priority-number", String(index + 1).padStart(2, "0")), makeText("span", `severity severity-${issue.severity}`, issue.severity), makeText("span", "confidence-tag", `${issue.confidence}可信度`));
    const title = makeText("h4", "", issue.title);
    const evidence = labeledLine("数据依据", issue.evidence);
    const impact = labeledLine("可能影响", issue.impact);
    const next = labeledLine("下一球", issue.nextBall);
    article.append(top, title, evidence, impact, next);
    return article;
  }));
  if (!summary.priorities.length) $("#advice-list").append(makeCard("✓", "本次没有突出问题", "先保持当前动作，下一球重点验证能否重复。", "strength-card"));
  $("#next-shot-list").replaceChildren(...summary.nextShot.slice(0, 2).map((text) => makeText("li", "", text)));
  renderDataTable(summary.dataRows);
  renderJointSections(summary.sections.filter((section) => section.id !== "rhythm"));
  renderRhythm(summary);
  renderPrescriptions(summary.prescriptions);
  renderAngleChart(summary.curves, summary.events);
  prepareReplay(summary.analysisFrames, hand);
  if (summary.release) {
    await seek(summary.release.time);
    drawKeyframe(summary.release.landmarks, hand);
  } else {
    clearCanvas($("#keyframe-canvas"), "关键帧证据不足，请按拍摄建议重新拍摄");
  }
  $("#empty-state").hidden = true;
  $("#analysis-result").hidden = false;
  $("#result-panel").classList.remove("empty");
}

function makeText(tag, className, text) {
  const element = document.createElement(tag);
  if (className) element.className = className;
  element.textContent = text;
  return element;
}

function labeledLine(label, text) {
  const paragraph = document.createElement("p");
  paragraph.append(makeText("b", "", `${label}：`), document.createTextNode(text));
  return paragraph;
}

function makeCard(icon, title, text, className) {
  const article = document.createElement("article");
  article.className = className;
  article.append(makeText("b", "", icon));
  const body = document.createElement("div");
  body.append(makeText("h4", "", title), makeText("p", "", text));
  article.append(body);
  return article;
}

function renderDataTable(rows) {
  const body = $("#data-table-body");
  if (!rows.length) {
    const tr = document.createElement("tr");
    const td = makeText("td", "", "当前识别可信度低，请复拍后再生成数据对照。");
    td.colSpan = 7;
    tr.append(td);
    body.replaceChildren(tr);
    return;
  }
  body.replaceChildren(...rows.map((row) => {
    const tr = document.createElement("tr");
    for (const key of ["metric", "current", "reference", "deviation", "evaluation", "nextStep", "confidence"]) tr.append(makeText("td", key === "current" ? "measured" : "", row[key]));
    return tr;
  }));
}

function renderJointSections(sections) {
  $("#joint-sections").replaceChildren(...sections.map((item) => {
    const details = document.createElement("details");
    details.className = "joint-detail";
    details.open = item.id === "elbow" || item.id === "knee";
    const summary = document.createElement("summary");
    const title = document.createElement("div");
    title.append(makeText("h4", "", item.title), makeText("span", "", item.evaluation));
    summary.append(title, makeText("b", `confidence confidence-${item.confidence[0]}`, `${item.confidence[0]}可信度`));
    const chips = document.createElement("div");
    chips.className = "data-chips";
    chips.append(...item.currentData.map((text) => makeText("span", "", text)));
    const grid = document.createElement("div");
    grid.className = "detail-grid";
    for (const [label, value, kind] of [["建议参考", item.reference, "reference"], ["做得好", item.good, "good"], ["当前问题", item.problem, "problem"], ["可能影响", item.impact, "impact"], ["下一球", item.nextShot, "next"], ["长期训练", item.longTerm, "long"]]) {
      const box = document.createElement("div");
      box.className = kind;
      box.append(makeText("b", "", label), makeText("p", "", value));
      grid.append(box);
    }
    details.append(summary, chips, grid, labeledLine("当前结论可信度", item.confidence));
    return details;
  }));
}

function renderRhythm(summary) {
  const events = Object.values(summary.events);
  $("#rhythm-timeline").replaceChildren(...events.map((event, index) => {
    const item = document.createElement("div");
    item.append(makeText("b", "", event.time === null ? "—" : `${event.time.toFixed(2)}s`), makeText("span", "", event.label));
    if (index < events.length - 1) item.append(makeText("i", "", "→"));
    return item;
  }));
  const rhythm = summary.sections.find((section) => section.id === "rhythm");
  $("#rhythm-copy").replaceChildren(labeledLine("当前评价", rhythm.evaluation), labeledLine("趋势判断", rhythm.problem), labeledLine("下一球", rhythm.nextShot), labeledLine("可信度", rhythm.confidence));
}

function renderPrescriptions(drills) {
  $("#prescription-list").replaceChildren(...drills.map((drill, index) => {
    const article = document.createElement("article");
    article.append(makeText("span", "", `处方 ${String(index + 1).padStart(2, "0")}`), makeText("h4", "", drill.name), labeledLine("对应问题", drill.forIssue), labeledLine("训练目的", drill.purpose), labeledLine("具体做法", drill.steps));
    const dose = document.createElement("div");
    dose.className = "drill-dose";
    dose.append(makeText("b", "", drill.sets), makeText("b", "", drill.reps));
    article.append(dose, labeledLine("训练关注点", drill.focus));
    return article;
  }));
}

function clearCanvas(canvas, text) {
  canvas.width = canvas.width || 720;
  canvas.height = canvas.height || 360;
  const context = canvas.getContext("2d");
  context.fillStyle = "#071017";
  context.fillRect(0, 0, canvas.width, canvas.height);
  context.fillStyle = "#8da3b0";
  context.font = "16px system-ui";
  context.textAlign = "center";
  context.fillText(text, canvas.width / 2, canvas.height / 2);
}

function renderAngleChart(curves, events) {
  const canvas = $("#angle-chart");
  const context = canvas.getContext("2d");
  context.clearRect(0, 0, canvas.width, canvas.height);
  context.fillStyle = "#071017";
  context.fillRect(0, 0, canvas.width, canvas.height);
  if (curves.length < 2) return clearCanvas(canvas, "有效帧不足，无法生成角度曲线");
  const pad = { left: 48, right: 18, top: 24, bottom: 35 };
  const width = canvas.width - pad.left - pad.right;
  const height = canvas.height - pad.top - pad.bottom;
  const maxTime = Math.max(...curves.map((point) => point.time)) || 1;
  context.strokeStyle = "rgba(157,178,191,.16)";
  context.fillStyle = "#78909d";
  context.font = "11px system-ui";
  context.textAlign = "right";
  for (let angle = 0; angle <= 180; angle += 30) {
    const y = pad.top + height * (1 - angle / 180);
    context.beginPath(); context.moveTo(pad.left, y); context.lineTo(canvas.width - pad.right, y); context.stroke();
    context.fillText(`${angle}°`, pad.left - 7, y + 4);
  }
  const series = [["elbow", "#ff7a31"], ["knee", "#37d4bd"], ["hip", "#8fa8ff"], ["trunk", "#f5c45e"]];
  for (const [key, color] of series) {
    context.strokeStyle = color; context.lineWidth = key === "trunk" ? 2 : 3; context.beginPath();
    curves.forEach((point, index) => {
      const x = pad.left + point.time / maxTime * width;
      const normalized = key === "trunk" ? clamp(point[key] + 90, 0, 180) : clamp(point[key], 0, 180);
      const y = pad.top + height * (1 - normalized / 180);
      index ? context.lineTo(x, y) : context.moveTo(x, y);
    });
    context.stroke();
  }
  for (const event of Object.values(events)) {
    if (event.time === null) continue;
    const x = pad.left + event.time / maxTime * width;
    context.strokeStyle = "rgba(255,255,255,.18)"; context.setLineDash([4, 5]); context.beginPath(); context.moveTo(x, pad.top); context.lineTo(x, pad.top + height); context.stroke(); context.setLineDash([]);
  }
}

const CONNECTIONS = [[11,12],[11,13],[13,15],[12,14],[14,16],[11,23],[12,24],[23,24],[23,25],[25,27],[24,26],[26,28]];

function drawPose(context, landmarks, width, height, hand, background = true) {
  if (background) { context.fillStyle = "#071017"; context.fillRect(0, 0, width, height); }
  const activePoints = hand === "right" ? [12,14,16,24,26,28] : [11,13,15,23,25,27];
  context.lineCap = "round";
  context.lineWidth = Math.max(3, width / 180);
  for (const [a, b] of CONNECTIONS) {
    if (!landmarks[a] || !landmarks[b]) continue;
    const active = activePoints.includes(a) && activePoints.includes(b);
    context.strokeStyle = active ? "#ff7a31" : "rgba(86,229,208,.9)";
    context.beginPath(); context.moveTo(landmarks[a].x * width, landmarks[a].y * height); context.lineTo(landmarks[b].x * width, landmarks[b].y * height); context.stroke();
  }
  for (const index of [11,12,13,14,15,16,23,24,25,26,27,28]) {
    if (!landmarks[index]) continue;
    context.fillStyle = activePoints.includes(index) ? "#ff9a61" : "#84eee0";
    context.beginPath(); context.arc(landmarks[index].x * width, landmarks[index].y * height, Math.max(3, width / 130), 0, Math.PI * 2); context.fill();
  }
}

function prepareReplay(frames, hand) {
  replayState = { frames, hand };
  $("#replay-range").value = 0;
  if (!frames.length) return clearCanvas($("#skeleton-canvas"), "有效帧不足，无法生成骨架回放");
  drawReplayFrame(0);
}

function drawReplayFrame(index) {
  const frames = replayState?.frames || [];
  if (!frames.length) return;
  const safeIndex = clamp(index, 0, frames.length - 1);
  const frame = frames[safeIndex];
  const canvas = $("#skeleton-canvas");
  drawPose(canvas.getContext("2d"), frame.landmarks, canvas.width, canvas.height, replayState.hand);
  $("#replay-range").value = frames.length === 1 ? 0 : safeIndex / (frames.length - 1) * 100;
  $("#replay-time").textContent = `${frame.time.toFixed(2)}s`;
}

function stopReplay() {
  if (replayAnimation) cancelAnimationFrame(replayAnimation);
  replayAnimation = null;
  const replayButton = $("#replay-button");
  if (replayButton) replayButton.textContent = "▶ 播放骨架";
}

$("#replay-button").addEventListener("click", () => {
  if (replayAnimation) return stopReplay();
  const frames = replayState?.frames || [];
  if (!frames.length) return;
  const startIndex = Math.round(Number($("#replay-range").value) / 100 * (frames.length - 1));
  const firstTime = frames[startIndex].time;
  const started = performance.now();
  $("#replay-button").textContent = "Ⅱ 暂停";
  const tick = (now) => {
    const targetTime = firstTime + (now - started) / 1000;
    let index = frames.findIndex((frame) => frame.time >= targetTime);
    if (index < 0) { drawReplayFrame(frames.length - 1); stopReplay(); return; }
    drawReplayFrame(index);
    replayAnimation = requestAnimationFrame(tick);
  };
  replayAnimation = requestAnimationFrame(tick);
});

$("#replay-range").addEventListener("input", (event) => {
  stopReplay();
  const frames = replayState?.frames || [];
  if (frames.length) drawReplayFrame(Math.round(Number(event.target.value) / 100 * (frames.length - 1)));
});

function drawKeyframe(landmarks, hand) {
  const canvas = $("#keyframe-canvas");
  const maxWidth = 820;
  const scale = Math.min(1, maxWidth / video.videoWidth);
  canvas.width = Math.round(video.videoWidth * scale);
  canvas.height = Math.round(video.videoHeight * scale);
  const context = canvas.getContext("2d");
  context.drawImage(video, 0, 0, canvas.width, canvas.height);
  drawPose(context, landmarks, canvas.width, canvas.height, hand, false);
}
