import { formatMetric, summarizePoseFrames } from "/assets/analyzer-core.mjs?v=20260830-2";
import { analyzeVideoDeterministic, sha256Json } from "/assets/pipeline/video-analyzer.mjs?v=20260830-3";
import { evaluateIssues, loadIssueKnowledge } from "/assets/coaching/issue-engine.mjs?v=20260830-2";
import { appendShot, buildSessionAnalytics, buildSessionSummary, createSession } from "/assets/session/session-coach.mjs?v=20260830-3";
import { openSessionStore } from "/assets/session/session-store.mjs?v=20260830-2";

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
let replayAnimation = null;
let replayState = null;
let selectedFile = null;
let currentSummary = null;
let currentShotId = null;
let activeSession = null;
let sessionStore = null;
let issueKnowledge = null;

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

const issueCopy = {
  CAPTURE_QUALITY_LOW: ["拍摄质量不足", "画面证据不足会放大角度和阶段误差", "capture"],
  TRUNK_FORWARD_LEAN: ["躯干偏移需要控制", "可能改变出手空间和落地点", "trunk"],
  TRUNK_UNSTABLE: ["躯干稳定性需要观察", "轴线波动可能降低出手重复性", "trunk"],
  KNEE_DIP_SHALLOW: ["下沉幅度偏浅", "下肢储能可能不足，上肢承担更多发力", "knee"],
  KNEE_DIP_DEEP: ["下沉幅度可能偏深", "动作时间可能拉长，急停时不易重复", "knee"],
  KNEE_DIP_INCONSISTENT: ["多球下沉深度不稳定", "每球发力条件变化会降低重复性", "knee"],
  ELBOW_EXTENSION_LIMITED: ["出手伸展可能不足", "随挥重复性可能下降", "elbow"],
  ELBOW_PATH_UNSTABLE: ["肘部伸展趋势不稳定", "举球与出手可能出现分段", "elbow"],
  WRIST_PATH_UNSTABLE: ["手腕二维轨迹不稳定", "左右方向更依赖临场修正", "wrist"],
  RHYTHM_DISCONNECTED: ["动力链可能存在停顿", "下肢动量可能在出手前中断", "rhythm"],
};

function applyStableIssues(summary) {
  const confidence = summary.capture.confidence;
  const confidences = Object.fromEntries(Object.keys(summary.metrics).map((key) => [key, confidence]));
  const groups = {
    elbow: ["elbowRelease", "elbowMin", "elbowRange", "elbowExtensionTrend", "elbowInstability"],
    knee: ["kneeLowest", "kneeRange", "kneeRiseDuration"],
    hip: ["hipLowest", "hipRange", "hipKneeCoordination"],
    trunk: ["trunkMax", "trunkRelease", "trunkDrift"],
    wrist: ["wristRise", "wristLateralDrift", "wristPathStability"],
    rhythm: ["pauseDuration", "rhythmDuration"],
  };
  for (const [joint, keys] of Object.entries(groups)) {
    for (const key of keys) if (key in confidences) confidences[key] = summary.jointConfidence?.[joint] || confidence;
  }
  if (summary.capture.bodyScale < 0.18) {
    for (const key of ["trunkMax", "trunkRelease", "trunkDrift", "wristPathStability", "wristLateralDrift", "wristRise"]) confidences[key] = "低";
  }
  const evaluated = evaluateIssues({ metrics: summary.metrics, confidences, capture: summary.capture }, issueKnowledge);
  summary.priorities = evaluated.priorities.map((issue) => {
    const [title, impact, joint] = issueCopy[issue.issueCode] || [issue.issueCode, "建议继续观察同机位重复性", "capture"];
    return { ...issue, title, impact, joint, evidence: `${issue.metricKey} 当前实测 ${issue.value}`, nextBall: issue.cue };
  });
  summary.nextRep = evaluated.nextRep;
  summary.next_rep = evaluated.nextRep;
  summary.nextShot = evaluated.nextRep;
  summary.why = summary.priorities.length ? summary.priorities.map((issue) => `${issue.title}：${issue.impact}`) : ["当前没有高优先级动作问题，下一球先验证重复性。"];
  summary.issueCodes = evaluated.issues.map((issue) => issue.issueCode);
  summary.summary.oneLine = summary.priorities.length
    ? `本次优先处理：${summary.priorities.map((issue) => issue.title).join("；")}。`
    : confidence === "低" ? "本次拍摄证据不足，先提高机位与入镜质量再判断动作。" : "当前没有高优先级动作问题，下一球先验证动作重复性。";
  return summary;
}

async function thumbnailDataUrl() {
  if (!video.videoWidth) return null;
  const canvas = document.createElement("canvas");
  const scale = Math.min(1, 320 / Math.max(video.videoWidth, video.videoHeight));
  canvas.width = Math.max(2, Math.round(video.videoWidth * scale));
  canvas.height = Math.max(2, Math.round(video.videoHeight * scale));
  canvas.getContext("2d").drawImage(video, 0, 0, canvas.width, canvas.height);
  return canvas.toDataURL("image/jpeg", 0.72);
}

input.addEventListener("change", async () => {
  const file = input.files?.[0];
  if (!file) return;
  selectedFile = file;
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
    message.textContent = `${error.message}。建议手机使用 MP4（H.264），桌面也可尝试 WebM。`;
    button.disabled = true;
  }
});

button.addEventListener("click", async () => {
  button.disabled = true;
  $("#analysis-result").hidden = true;
  $("#empty-state").hidden = false;
  $("#result-panel").classList.add("empty");
  try {
    const hand = document.querySelector('input[name="hand"]:checked').value;
    if (!activeSession) activeSession = createSession({ shootingHand: hand, distanceCategory: $("#distance-category").value });
    if (activeSession.shootingHand !== hand) throw new Error("同一训练已锁定投篮手，请结束训练后重新选择");
    setProgress(4, "正在创建独立 CPU 姿态模型");
    const pipeline = await analyzeVideoDeterministic(video, {
      shootingHand: hand,
      onProgress: (index, count) => setProgress(10 + 80 * index / count, `固定采样 ${index}/${count}`),
    });
    let summary = summarizePoseFrames(pipeline.frames, hand, {
      duration: pipeline.durationMs / 1000, sourceFrames: null, fileName: selectedFile?.name || "当前视频",
      rawFrames: pipeline.rawFrames, preprocessed: true, orientationPass: pipeline.orientationPass, orientationScores: pipeline.orientationScores,
    });
    summary.pipeline = { ...pipeline.manifest, canvasSize: pipeline.canvasSize, hashes: pipeline.hashes };
    summary.phasesHash = await sha256Json(summary.events);
    summary = applyStableIssues(summary);
    summary.deterministicNextRep = [...summary.nextRep];
    summary.finalHash = await sha256Json({ metrics: summary.metrics, events: summary.events, strengths: summary.strengths, issueCodes: summary.priorities.map((issue) => issue.issueCode), nextRep: summary.nextRep });
    setProgress(94, "正在生成训练建议");
    activeSession = appendShot(activeSession, {
      analysis: summary,
      fileMeta: { name: selectedFile?.name || "当前视频", size: selectedFile?.size || 0, duration: video.duration },
      thumbnail: await thumbnailDataUrl(),
    });
    currentShotId = activeSession.shots.at(-1).shotId;
    activeSession.sessionSummary = buildSessionSummary(activeSession);
    await sessionStore.save(activeSession);
    await renderResult(activeSession.shots.at(-1).analysis, hand);
    renderSession();
    setProgress(100, "分析完成");
    message.textContent = `第 ${activeSession.shots.length} 球分析完成。更换视频即可继续本次训练。`;
  } catch (error) {
    console.error(error);
    message.textContent = `分析未完成：${error.message || "浏览器不支持当前模型"}。请优先使用最新版 Chrome、Edge 或 Safari。`;
    progressLabel.textContent = "分析失败";
  } finally {
    button.disabled = false;
  }
});

function renderSession() {
  const session = activeSession;
  $("#session-state").textContent = session ? `${session.status === "completed" ? "已结束" : "进行中"} · ${session.shots.length} 球 · ${$("#distance-category").selectedOptions[0].textContent}` : "尚未开始训练";
  $("#end-session").disabled = !session || session.shots.length < 2;
  $("#distance-category").disabled = Boolean(session);
  document.querySelectorAll('input[name="hand"]').forEach((item) => { item.disabled = Boolean(session); });
  const tabs = $("#shot-tabs");
  tabs.replaceChildren(...(session?.shots || []).map((shot) => {
    const tab = makeText("button", shot.shotId === currentShotId ? "active" : "", `Shot ${shot.shotNumber}`);
    tab.type = "button";
    tab.addEventListener("click", async () => { currentShotId = shot.shotId; await renderResult(shot.analysis, session.shootingHand, { archivedVideo: shot !== session.shots.at(-1) }); renderSession(); });
    return tab;
  }));
  if (session?.shots.length) {
    const selectedShot = session.shots.find((shot) => shot.shotId === currentShotId) || session.shots.at(-1);
    renderComparison(selectedShot);
    renderSessionAnalytics(buildSessionAnalytics(session, selectedShot.shotNumber), session);
  } else renderSessionAnalytics(null, null);
  $("#session-summary").textContent = session?.sessionSummary?.status === "formal"
    ? `正式总结：${session.baseline?.sampleCount || 0} 球基线已建立；改善最多 ${session.sessionSummary.mostImprovedMetric || "暂未确认"}；有效 cue ${session.sessionSummary.mostEffectiveCue || "继续收集"}；仍需观察 ${session.sessionSummary.stillUnstableMetric || "暂无突出项"}；下次重点 ${session.sessionSummary.nextFocus || "保持稳定动作"}。`
    : session?.shots.length >= 2 ? `暂定总结：已完成 ${session.shots.length} 球，可继续到 5 球建立个人基线。` : "完成至少 2 球后生成训练总结。";
}

function renderComparison(shot) {
  const target = $("#shot-comparison");
  if (!shot?.comparison) { target.textContent = "完成下一球后，这里会显示与上一球的变化。"; return; }
  if (!shot.comparison.comparable) { target.textContent = `本球不做强对比：${shot.comparison.reason}`; return; }
  const labels = { improved: "改善", stable: "稳定", worsened: "退步", overcorrected: "纠正过头", not_comparable: "不可比" };
  const rows = Object.entries(shot.comparison.metrics).filter(([, item]) => Number.isFinite(item.delta) && item.status !== "stable").slice(0, 4);
  target.replaceChildren(...rows.map(([key, item]) => makeText("span", `comparison-${item.status}`, `与第 ${shot.shotNumber - 1} 球比 · ${key} ${item.delta > 0 ? "+" : ""}${item.delta.toFixed(1)} · ${labels[item.status]}`)));
  if (!rows.length) target.textContent = `与第 ${shot.shotNumber - 1} 球相比，核心指标保持稳定。`;
}

function renderSessionAnalytics(analytics, session) {
  const comparisonTarget = $("#history-comparison-grid");
  const improvementTarget = $("#improvement-summary");
  const problemTarget = $("#new-problem-list");
  const maintainTarget = $("#session-maintain-list");
  const bestTarget = $("#best-shot-card");
  const historyHead = $("#history-table-head");
  const historyBody = $("#history-table-body");
  const trendTarget = $("#history-trend-list");
  const adjustmentBody = $("#adjustment-table-body");
  const trainingBody = $("#training-table-body");
  const finalTarget = $("#session-final-summary");
  if (!analytics || !session) {
    comparisonTarget.replaceChildren(makeText("p", "empty-note", "上传第一球后开始建立历史参照。"));
    improvementTarget.replaceChildren(makeText("p", "empty-note", "完成下一球后判断改善。"));
    problemTarget.replaceChildren(makeText("p", "empty-note", "当前没有历史球可用于识别新问题。"));
    maintainTarget.replaceChildren(makeText("p", "empty-note", "完成分析后显示保持项。"));
    bestTarget.textContent = "完成至少一球后评估本轮最佳参考球。";
    historyHead.replaceChildren(); historyBody.replaceChildren(); trendTarget.replaceChildren(); adjustmentBody.replaceChildren(); trainingBody.replaceChildren();
    finalTarget.hidden = true; finalTarget.replaceChildren();
    return;
  }

  const layerNames = { previous: "和上一球相比", first: "和第 1 球相比", sessionAverage: "和此前 Session 平均相比", best: "和当前最佳球相比" };
  const statusNames = { improved: "改善", stable: "稳定", worsened: "退步", overcorrected: "纠正过头", not_comparable: "不可比" };
  const metricLabels = Object.fromEntries(analytics.historyRows.map((row) => [row.key, row.label]));
  comparisonTarget.replaceChildren(...Object.entries(analytics.comparisons).map(([key, layer]) => {
    const card = document.createElement("article"); card.className = "history-comparison-card";
    const reference = layer.referenceShotNumber ? ` · Shot ${layer.referenceShotNumber}` : "";
    card.append(makeText("small", "", `${layerNames[key]}${reference}`));
    if (!layer.comparable) {
      card.append(makeText("strong", "", "暂不可比"), makeText("p", "", layer.reason)); return card;
    }
    const changed = Object.entries(layer.metrics).filter(([, item]) => Number.isFinite(item.delta)).toSorted((a, b) => (a[1].status === "stable") - (b[1].status === "stable")).slice(0, 2);
    card.append(makeText("strong", "", changed.some(([, item]) => item.status === "improved") ? "检测到改善" : "以稳定为主"));
    card.append(makeText("p", "", changed.map(([metric, item]) => `${metricLabels[metric] || metric} ${item.delta > 0 ? "+" : ""}${item.delta.toFixed(item.tolerance < 1 ? 2 : 1)} · ${statusNames[item.status]}`).join("；") || "暂无可靠数值"));
    return card;
  }));

  const fillList = (target, items, fallback) => target.replaceChildren(...(items.length ? items.map((item) => makeText("p", "", typeof item === "string" ? item : item.text)) : [makeText("p", "empty-note", fallback)]));
  fillList(improvementTarget, analytics.improvements, analytics.currentShotNumber === 1 ? "这是本次训练基准球。" : "当前没有达到容差以上的明确改善。" );
  fillList(problemTarget, analytics.newProblems, "这一球没有检测到新的高优先级问题。" );
  fillList(maintainTarget, analytics.maintain, "当前没有足够证据生成保持项。" );

  bestTarget.replaceChildren(makeText("strong", "", `Shot ${analytics.bestShot.shotNumber} · 当前最佳参考球`), document.createTextNode(`　${analytics.bestShot.explanation}`));

  const shotNumbers = analytics.historyRows[0]?.values.map((item) => item.shotNumber) || [];
  const headerRow = document.createElement("tr");
  headerRow.append(makeText("th", "", "指标"), ...shotNumbers.map((number) => makeText("th", "", `Shot ${number}`)), makeText("th", "", "趋势"));
  historyHead.replaceChildren(headerRow);
  const trendNames = { IMPROVING: "持续改善", STABLE: "基本稳定", WORSENING: "有所退步", INCONSISTENT: "波动较大", UNCERTAIN: "数据不足" };
  historyBody.replaceChildren(...analytics.historyRows.map((row) => {
    const tr = document.createElement("tr"); tr.append(makeText("td", "", row.label));
    tr.append(...row.values.map((item) => makeText("td", item.confidence === "低" ? "low-confidence-value" : "", Number.isFinite(item.value) ? `${item.value.toFixed(row.unit === "s" ? 2 : 1)}${row.unit}` : "—")));
    tr.append(makeText("td", `trend-${row.trend}`, trendNames[row.trend])); return tr;
  }));
  trendTarget.replaceChildren(...analytics.historyRows.slice(0, 6).map((row) => {
    const card = document.createElement("article"); card.className = "trend-card";
    card.append(makeText("b", "", row.label), makeText("span", `trend-${row.trend}`, trendNames[row.trend]));
    const spark = document.createElement("div"); spark.className = "trend-spark";
    const numeric = row.values.map((item) => item.value).filter(Number.isFinite); const min = Math.min(...numeric); const max = Math.max(...numeric); const span = Math.max(max - min, row.tolerance || 1);
    spark.append(...row.values.map((item) => { const bar = document.createElement("i"); bar.style.height = Number.isFinite(item.value) ? `${20 + (item.value - min) / span * 80}%` : "4%"; bar.title = `Shot ${item.shotNumber}: ${Number.isFinite(item.value) ? item.value : "—"}`; return bar; }));
    card.append(spark); return card;
  }));

  adjustmentBody.replaceChildren(...analytics.adjustmentRows.map((row) => {
    const tr = document.createElement("tr");
    for (const value of [row.priority, row.part, row.performance, row.current, row.target, row.direction, row.cue, row.hint]) tr.append(makeText("td", "", value));
    return tr;
  }));
  trainingBody.replaceChildren(...(analytics.trainingRows.length ? analytics.trainingRows : [{ name: "动作复现", issue: "当前没有突出问题", method: "同机位再投一球，复现当前动作。", volume: "3组 × 5球", focus: "保持距离、机位和节奏一致" }]).map((row) => {
    const tr = document.createElement("tr");
    for (const value of [row.name, row.issue, row.method, row.volume, row.focus]) tr.append(makeText("td", "", value || "—"));
    return tr;
  }));

  const summary = session.sessionSummary;
  finalTarget.hidden = session.status !== "completed" || !summary || summary.status === "insufficient";
  if (!finalTarget.hidden) {
    const changes = (summary.fromFirstToLast || []).filter((item) => ["improved", "stable", "worsened", "overcorrected"].includes(item.status)).slice(0, 5);
    const list = document.createElement("ul");
    list.append(...changes.map((item) => makeText("li", "", `${item.label}：Shot 1 ${item.first.toFixed(item.unit === "s" ? 2 : 1)}${item.unit} → Shot ${summary.shotCount} ${item.last.toFixed(item.unit === "s" ? 2 : 1)}${item.unit}（${statusNames[item.status]}）`)));
    finalTarget.replaceChildren(makeText("h4", "", `本次训练总结 · ${summary.shotCount} 球`), makeText("p", "", `当前最佳参考球：Shot ${summary.bestShot?.shotNumber || "—"}。下一次只关注：${(summary.nextFocuses || []).join("、") || "保持当前稳定动作"}。`), list);
  }
}

$("#new-session").addEventListener("click", () => {
  activeSession = null; currentShotId = null; renderSession();
  message.textContent = "已准备新训练，请选择投篮手、距离和第一球视频。";
});

$("#end-session").addEventListener("click", async () => {
  if (!activeSession || activeSession.shots.length < 2) return;
  activeSession = { ...activeSession, status: "completed", sessionSummary: buildSessionSummary(activeSession), updatedAt: new Date().toISOString() };
  await sessionStore.save(activeSession); renderSession();
  message.textContent = activeSession.sessionSummary.status === "formal" ? "训练已结束，正式总结已生成。" : "训练已结束，已生成暂定总结。";
});

$("#clear-sessions").addEventListener("click", async () => {
  await sessionStore.clear(); activeSession = null; currentShotId = null; renderSession();
  message.textContent = "本机训练记录已清空。";
});

$("#delete-session").addEventListener("click", async () => {
  if (!activeSession) return;
  await sessionStore.delete(activeSession.sessionId); activeSession = null; currentShotId = null; renderSession();
  message.textContent = "当前训练已从本机删除。";
});

async function initializeSession() {
  [issueKnowledge, sessionStore] = await Promise.all([loadIssueKnowledge(), openSessionStore()]);
  activeSession = await sessionStore.latestActive();
  if (activeSession) {
    currentShotId = activeSession.shots.at(-1)?.shotId || null;
    const hand = document.querySelector(`input[name="hand"][value="${activeSession.shootingHand}"]`); if (hand) hand.checked = true;
    $("#distance-category").value = activeSession.distanceCategory;
    message.textContent = `已恢复本机训练（${activeSession.shots.length} 球）。旧视频未长期保存，可继续上传下一球。`;
  }
  renderSession();
  if (activeSession?.shots.length) {
    await renderResult(activeSession.shots.at(-1).analysis, activeSession.shootingHand, { archivedVideo: true });
    renderSession();
  }
}

initializeSession().catch((error) => { console.error(error); message.textContent = `训练数据初始化失败：${error.message}`; });

async function renderResult(summary, hand, { archivedVideo = false } = {}) {
  stopReplay();
  currentSummary = summary;
  const percent = summary.capture.score;
  $("#quality-score").textContent = `${percent}/100`;
  $("#quality-label").textContent = `${summary.capture.confidence}可信度`;
  $("#one-line-summary").textContent = summary.summary.oneLine;
  $("#metric-elbow").textContent = formatMetric(summary.metrics.elbowRelease);
  $("#metric-knee").textContent = formatMetric(summary.metrics.kneeLowest);
  $("#metric-range").textContent = formatMetric(summary.metrics.kneeRange);
  $("#metric-trunk").textContent = formatMetric(summary.metrics.trunkRelease);
  $("#analysis-summary-text").textContent = summary.capture.confidence === "低"
    ? `本次综合识别质量 ${percent}/100，有效帧 ${Math.round(summary.validRatio * 100)}%。当前证据不足，系统已停止输出精确动作纠正。`
    : `本次综合识别质量 ${percent}/100，共分析 ${summary.totalFrames} 个时间点，其中 ${summary.validFrames} 帧可用于动作判断。系统判断为${summary.capture.viewpoint?.label || "未知视角"}。${summary.capture.shootingArmOccluded ? "投篮手位于画面远侧，阶段与节奏可用，肘腕精细指标已降级。" : "以下结论优先基于相对幅度、变化趋势和动作连续性。"}`;
  renderSummaryFacts(summary.summary, summary.capture);

  const strengths = summary.strengths.length ? summary.strengths : ["当前没有足够证据判断动作优点，请先提高拍摄质量。"];
  $("#strength-list").replaceChildren(...strengths.map((text) => makeCard("✓", "继续保持", text, "strength-card")));
  $("#keep-list").replaceChildren(makeText("h4", "mini-heading", "这次先不要改"), ...summary.keep.map((text) => makeText("p", "keep-item", text)));
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
  $("#why-list").replaceChildren(makeText("h4", "mini-heading", "为什么优先改这些"), ...summary.why.map((text) => makeText("p", "why-item", text)));
  $("#next-shot-list").replaceChildren(...summary.nextRep.slice(0, 2).map((text) => makeText("li", "", text)));
  renderDataTable(summary.dataRows);
  renderJointSections(summary.sections.filter((section) => section.id !== "rhythm"));
  renderRhythm(summary);
  renderPrescriptions(summary.prescriptions);
  renderAngleChart(summary.curves || [], summary.events);
  prepareReplay(summary.analysisFrames || [], hand, summary.events);
  if (archivedVideo) {
    $("#processed-video").hidden = true; $("#processed-overlay").hidden = true; $("#processed-video-fallback").hidden = false;
    $("#processed-video-fallback").textContent = "原视频未长期保存；本机仅保留分析数据和缩略图。";
  } else prepareOverlayReplay(summary, hand);
  $("#limitation-list").replaceChildren(...summary.technicalLimitations.map((text) => makeText("li", "", text)));
  renderDebugData(summary);
  if (summary.release && !archivedVideo) {
    await seek(summary.release.time);
    drawKeyframe(summary.release, hand, summary.events);
  } else {
    clearCanvas($("#keyframe-canvas"), "关键帧证据不足，请按拍摄建议重新拍摄");
  }
  $("#empty-state").hidden = true;
  $("#analysis-result").hidden = false;
  $("#result-panel").classList.remove("empty");
}

function renderSummaryFacts(summary, capture = {}) {
  const facts = [
    ["文件", summary.fileName], ["时长", `${summary.duration.toFixed(2)}s`],
    ["源总帧", summary.sourceFrames ?? "—（浏览器未提供）"], ["采样点", summary.sampledFrames],
    ["有效姿态帧", summary.validPoseFrames], ["检测率", `${summary.poseDetectionRate}%`],
    ["投篮手", summary.shootingHand], ["自动视角", capture.viewpoint?.label || "未稳定判断"],
    ["分析侧", capture.shootingArmOccluded ? "可见侧代理阶段（投篮臂精细指标降级）" : "投篮侧"],
    ["方向校正", capture.orientationPass === "mirrored-normalized" ? "已自动镜像校正" : "原方向通过"],
    ["关键阶段", summary.keyPhasesFound.length ? summary.keyPhasesFound.join(" / ") : "未稳定识别"],
  ];
  $("#summary-facts").replaceChildren(...facts.map(([label, value]) => {
    const item = document.createElement("div");
    item.append(makeText("small", "", label), makeText("b", "", String(value)));
    return item;
  }));
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
  const cards = $("#data-card-list");
  if (!rows.length) {
    const tr = document.createElement("tr");
    const td = makeText("td", "", "当前识别可信度低，请复拍后再生成数据对照。");
    td.colSpan = 7;
    tr.append(td);
    body.replaceChildren(tr);
    cards.replaceChildren(makeCard("!", "暂无可靠数据", "当前识别可信度低，请复拍后再生成数据对照。", "data-card"));
    return;
  }
  body.replaceChildren(...rows.map((row) => {
    const tr = document.createElement("tr");
    for (const key of ["metric", "current", "reference", "deviation", "evaluation", "nextStep", "confidence"]) tr.append(makeText("td", key === "current" ? "measured" : "", row[key]));
    return tr;
  }));
  cards.replaceChildren(...rows.map((row) => {
    const article = document.createElement("article");
    article.className = "data-card";
    article.append(makeText("h4", "", row.metric), makeText("strong", "", row.current));
    for (const [label, key] of [["训练参考", "reference"], ["偏差", "deviation"], ["评价", "evaluation"], ["下一步", "nextStep"], ["可信度", "confidence"]]) article.append(labeledLine(label, row[key]));
    return article;
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
  const safeDrills = drills.length ? drills : [{ name: "暂不安排专项纠正", forIssue: "数据不足", purpose: "先获得可信画面", steps: "按拍摄要求重新拍摄一球。", sets: "1 组", reps: "3 球", focus: "全身入镜、机位固定。", rightFeeling: "画面中关键点连续稳定。", commonMistakes: "人物过小或手持跟拍。" }];
  $("#prescription-list").replaceChildren(...safeDrills.map((drill, index) => {
    const article = document.createElement("article");
    article.append(makeText("span", "", `处方 ${String(index + 1).padStart(2, "0")}`), makeText("h4", "", drill.name), labeledLine("对应问题", drill.forIssue), labeledLine("训练目的", drill.purpose), labeledLine("具体做法", drill.steps));
    const dose = document.createElement("div");
    dose.className = "drill-dose";
    dose.append(makeText("b", "", drill.sets), makeText("b", "", drill.reps));
    article.append(dose, labeledLine("训练关注点", drill.focus), labeledLine("做对的感觉", drill.rightFeeling), labeledLine("常见错误", drill.commonMistakes));
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

function prepareReplay(frames, hand, events) {
  replayState = { frames, hand, events };
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
  drawFrameLabels(canvas.getContext("2d"), frame, replayState.events, canvas.width);
  $("#replay-range").value = frames.length === 1 ? 0 : safeIndex / (frames.length - 1) * 100;
  $("#replay-time").textContent = `${frame.time.toFixed(2)}s`;
}

function nearestPhase(time, events) {
  const list = Object.values(events || {}).filter((event) => Number.isFinite(event.time));
  if (!list.length) return "阶段未识别";
  return list.reduce((best, item) => Math.abs(item.time - time) < Math.abs(best.time - time) ? item : best).label;
}

function drawFrameLabels(context, frame, events, width) {
  context.fillStyle = "rgba(3,10,15,.78)";
  context.fillRect(12, 12, Math.min(390, width - 24), 76);
  context.fillStyle = "#f4f7f8";
  context.font = `${Math.max(12, width / 55)}px system-ui`;
  context.textAlign = "left";
  context.fillText(`${nearestPhase(frame.time, events)} · ${frame.time.toFixed(2)}s`, 24, 38);
  context.fillStyle = "#9fb3bd";
  context.font = `${Math.max(10, width / 72)}px system-ui`;
  const values = [`肘 ${formatMetric(frame.elbowAngle)}`, `膝 ${formatMetric(frame.kneeAngle)}`, `髋 ${formatMetric(frame.hipAngle)}`, `躯干 ${formatMetric(frame.trunkLean)}`];
  context.fillText(values.join("  "), 24, 67);
}

function prepareOverlayReplay(summary, hand) {
  const player = $("#processed-video");
  const overlay = $("#processed-overlay");
  const fallback = $("#processed-video-fallback");
  if (!summary.processedVideo.available || !objectUrl) {
    player.hidden = true;
    overlay.hidden = true;
    fallback.hidden = false;
    return;
  }
  player.hidden = false;
  overlay.hidden = false;
  fallback.hidden = true;
  if (player.src !== objectUrl) player.src = objectUrl;
  const update = () => {
    if (!player.videoWidth || !summary.analysisFrames.length) return;
    overlay.width = player.videoWidth;
    overlay.height = player.videoHeight;
    const frame = summary.analysisFrames.reduce((best, item) => Math.abs(item.time - player.currentTime) < Math.abs(best.time - player.currentTime) ? item : best);
    const context = overlay.getContext("2d");
    context.clearRect(0, 0, overlay.width, overlay.height);
    drawPose(context, frame.landmarks, overlay.width, overlay.height, hand, false);
    drawFrameLabels(context, frame, summary.events, overlay.width);
  };
  player.ontimeupdate = update;
  player.onseeked = update;
  player.onplay = update;
  player.currentTime = summary.processedVideo.startTime || 0;
  update();
}

function renderDebugData(summary) {
  const debug = $("#debug-analysis-data");
  const local = ["localhost", "127.0.0.1"].includes(location.hostname) && new URLSearchParams(location.search).get("debug") === "1";
  debug.hidden = !local;
  if (local) $("#debug-analysis-json").textContent = JSON.stringify(summary, (key, value) => key === "landmarks" ? `[${value?.length || 0} landmarks]` : value, 2);
}

function stopReplay() {
  if (replayAnimation) cancelAnimationFrame(replayAnimation);
  replayAnimation = null;
  const replayButton = $("#replay-button");
  if (replayButton) replayButton.textContent = "▶ 播放火柴人";
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

function drawKeyframe(frame, hand, events) {
  const canvas = $("#keyframe-canvas");
  const maxWidth = 820;
  const scale = Math.min(1, maxWidth / video.videoWidth);
  canvas.width = Math.round(video.videoWidth * scale);
  canvas.height = Math.round(video.videoHeight * scale);
  const context = canvas.getContext("2d");
  context.drawImage(video, 0, 0, canvas.width, canvas.height);
  drawPose(context, frame.landmarks, canvas.width, canvas.height, hand, false);
  drawFrameLabels(context, frame, events, canvas.width);
}
