import { selectDeterministicPhases } from "./pipeline/phase-detector.mjs";

const SIDE = {
  left: { shoulder: 11, elbow: 13, wrist: 15, hip: 23, knee: 25, ankle: 27 },
  right: { shoulder: 12, elbow: 14, wrist: 16, hip: 24, knee: 26, ankle: 28 },
};

const SEVERITY_WEIGHT = { 明显: 3, 中等: 2, 轻微: 1 };
const clamp = (value, min, max) => Math.min(max, Math.max(min, value));
const round = (value, digits = 1) => Number.isFinite(value) ? Number(value.toFixed(digits)) : null;
const mean = (values) => values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : Number.NaN;
const extent = (values) => values.length ? Math.max(...values) - Math.min(...values) : Number.NaN;
const standardDeviation = (values) => {
  if (!values.length) return Number.NaN;
  const average = mean(values);
  return Math.sqrt(mean(values.map((value) => (value - average) ** 2)));
};
const median = (values) => {
  const sorted = values.filter(Number.isFinite).toSorted((a, b) => a - b);
  if (!sorted.length) return Number.NaN;
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
};
const percentile = (values, fraction) => {
  const sorted = values.filter(Number.isFinite).toSorted((a, b) => a - b);
  if (!sorted.length) return Number.NaN;
  const position = (sorted.length - 1) * fraction;
  const lower = Math.floor(position);
  const upper = Math.ceil(position);
  return sorted[lower] + (sorted[upper] - sorted[lower]) * (position - lower);
};

export function calculateAngle(a, b, c) {
  if (!a || !b || !c) return Number.NaN;
  const ab = { x: a.x - b.x, y: a.y - b.y };
  const cb = { x: c.x - b.x, y: c.y - b.y };
  const denominator = Math.hypot(ab.x, ab.y) * Math.hypot(cb.x, cb.y);
  if (!denominator) return Number.NaN;
  return Math.acos(clamp((ab.x * cb.x + ab.y * cb.y) / denominator, -1, 1)) * 180 / Math.PI;
}

function regressionSlope(frames, key) {
  const points = frames.filter((frame) => Number.isFinite(frame[key]));
  if (points.length < 2) return 0;
  const timeMean = mean(points.map((frame) => frame.time));
  const valueMean = mean(points.map((frame) => frame[key]));
  const denominator = points.reduce((sum, frame) => sum + (frame.time - timeMean) ** 2, 0);
  if (!denominator) return 0;
  return points.reduce((sum, frame) => sum + (frame.time - timeMean) * (frame[key] - valueMean), 0) / denominator;
}

function correlation(a, b) {
  if (a.length !== b.length || a.length < 3) return Number.NaN;
  const ma = mean(a);
  const mb = mean(b);
  const numerator = a.reduce((sum, value, index) => sum + (value - ma) * (b[index] - mb), 0);
  const denominator = Math.sqrt(a.reduce((sum, value) => sum + (value - ma) ** 2, 0) * b.reduce((sum, value) => sum + (value - mb) ** 2, 0));
  return denominator ? numerator / denominator : 0;
}

function pointDistance(a, b) {
  return a && b ? Math.hypot(a.x - b.x, a.y - b.y) : Number.NaN;
}

const OTHER_SIDE = { left: "right", right: "left" };
const ARM_KEYS = ["shoulder", "elbow", "wrist"];
const BODY_KEYS = ["hip", "knee", "ankle"];
const PHASE_BODY_KEYS = ["hip", "knee"];
const visiblePoint = (point, threshold = 0.45) => Boolean(point && (point.visibility ?? 1) >= threshold);

function chainCoverage(frames, side, keys) {
  if (!frames.length) return 0;
  const indexes = SIDE[side];
  return frames.filter((frame) => keys.every((key) => visiblePoint(frame.landmarks?.[indexes[key]]))).length / frames.length;
}

function chooseVisibleSide(frames, preferred, keys) {
  const alternate = OTHER_SIDE[preferred];
  const preferredCoverage = chainCoverage(frames, preferred, keys);
  const alternateCoverage = chainCoverage(frames, alternate, keys);
  const side = preferredCoverage >= 0.5 || preferredCoverage >= alternateCoverage - 0.12 ? preferred : alternate;
  return { side, coverage: side === preferred ? preferredCoverage : alternateCoverage, preferredCoverage, alternateCoverage };
}

function selectAnalysisSides(frames, shootingHand) {
  const preferred = SIDE[shootingHand] ? shootingHand : "right";
  const arm = chooseVisibleSide(frames, preferred, ARM_KEYS);
  const body = chooseVisibleSide(frames, preferred, PHASE_BODY_KEYS);
  const fullBodyCoverage = chainCoverage(frames, body.side, BODY_KEYS);
  return {
    arm: arm.side,
    body: body.side,
    shooting: preferred,
    shootingArmOccluded: arm.side !== preferred,
    coverage: { arm: round(arm.coverage, 2), body: round(body.coverage, 2), fullBody: round(fullBodyCoverage, 2), shootingArm: round(arm.preferredCoverage, 2) },
  };
}

function midpointOrSingle(a, b, fallback) {
  const visible = [a, b].filter((point) => visiblePoint(point));
  if (visible.length === 2) return { x: (visible[0].x + visible[1].x) / 2, y: (visible[0].y + visible[1].y) / 2 };
  return visible[0] || fallback || null;
}

function estimateViewpoint(frames) {
  const ratios = [];
  for (const frame of frames) {
    const p = frame.landmarks || [];
    if (![p[11], p[12], p[23], p[24]].every((point) => visiblePoint(point))) continue;
    const shoulderCenter = midpointOrSingle(p[11], p[12]);
    const hipCenter = midpointOrSingle(p[23], p[24]);
    const torsoLength = pointDistance(shoulderCenter, hipCenter);
    if (torsoLength > 0.04) ratios.push(pointDistance(p[11], p[12]) / torsoLength);
  }
  const ratio = median(ratios);
  if (!Number.isFinite(ratio)) return { category: "unknown", label: "视角未能稳定判断", confidence: "低", shoulderToTorsoRatio: null };
  if (ratio >= 0.8) return { category: "front_or_rear", label: "正面或背面视角", confidence: ratios.length >= 3 ? "高" : "中", shoulderToTorsoRatio: round(ratio, 2) };
  if (ratio <= 0.35) return { category: "side", label: "侧面视角", confidence: ratios.length >= 3 ? "高" : "中", shoulderToTorsoRatio: round(ratio, 2) };
  return { category: "diagonal", label: "斜侧视角", confidence: ratios.length >= 3 ? "中" : "低", shoulderToTorsoRatio: round(ratio, 2) };
}

function frameMetrics(frame, selection) {
  const p = frame.landmarks;
  const armIndexes = SIDE[selection.arm];
  const bodyIndexes = SIDE[selection.body];
  const shootingIndexes = SIDE[selection.shooting];
  const requiredIndexes = [...ARM_KEYS.map((key) => armIndexes[key]), ...PHASE_BODY_KEYS.map((key) => bodyIndexes[key])];
  const required = requiredIndexes.map((index) => p[index]);
  const visibilityValues = required.map((point) => point?.visibility ?? 0);
  const averageVisibility = mean(visibilityValues);
  const visible = required.every((point) => visiblePoint(point));
  const visiblePoints = p.filter((point) => visiblePoint(point));
  const bodyScale = visiblePoints.length ? Math.max(...visiblePoints.map((point) => point.y)) - Math.min(...visiblePoints.map((point) => point.y)) : 0;
  if (!visible) return { ...frame, valid: false, averageVisibility, bodyScale };

  const shoulder = p[armIndexes.shoulder];
  const elbow = p[armIndexes.elbow];
  const wrist = p[armIndexes.wrist];
  const hip = p[bodyIndexes.hip];
  const knee = p[bodyIndexes.knee];
  const ankle = p[bodyIndexes.ankle];
  const shoulderCenter = midpointOrSingle(p[11], p[12], shoulder);
  const hipCenter = midpointOrSingle(p[23], p[24], hip);
  const bodyShoulder = visiblePoint(p[bodyIndexes.shoulder]) ? p[bodyIndexes.shoulder] : shoulderCenter;
  const measuredKneeAngle = visiblePoint(ankle) ? calculateAngle(hip, knee, ankle) : Number.NaN;
  const phaseKneeAngle = Number.isFinite(measuredKneeAngle) ? measuredKneeAngle : calculateAngle(bodyShoulder, hip, knee);
  const shootingShoulder = p[shootingIndexes.shoulder];
  const shootingElbow = p[shootingIndexes.elbow];
  const shootingWrist = p[shootingIndexes.wrist];
  const shootingArmVisible = [shootingShoulder, shootingElbow, shootingWrist].every((point) => visiblePoint(point));
  return {
    ...frame, valid: true, averageVisibility, bodyScale,
    elbowAngle: calculateAngle(shoulder, elbow, wrist),
    kneeAngle: phaseKneeAngle,
    measuredKneeAngle,
    hipAngle: calculateAngle(bodyShoulder, hip, knee),
    trunkLean: Math.atan2(shoulderCenter.x - hipCenter.x, hipCenter.y - shoulderCenter.y) * 180 / Math.PI,
    torsoLength: pointDistance(shoulderCenter, hipCenter), shoulderWidth: pointDistance(p[11], p[12]),
    wristX: wrist.x, wristY: wrist.y,
    shootingElbowAngle: shootingArmVisible ? calculateAngle(shootingShoulder, shootingElbow, shootingWrist) : Number.NaN,
    shootingWristX: shootingArmVisible ? shootingWrist.x : Number.NaN,
    shootingWristY: shootingArmVisible ? shootingWrist.y : Number.NaN,
  };
}

function smoothFrames(valid) {
  const keys = ["elbowAngle", "kneeAngle", "measuredKneeAngle", "hipAngle", "trunkLean", "wristX", "wristY", "shootingElbowAngle", "shootingWristX", "shootingWristY"];
  return valid.map((frame, index) => {
    if (index === 0 || index === valid.length - 1) return { ...frame };
    const nearby = valid.slice(index - 1, index + 2).filter((candidate) => Math.abs(candidate.time - frame.time) <= 0.5);
    const smoothed = { ...frame };
    for (const key of keys) {
      let samples = nearby.map((candidate) => candidate[key]);
      if (key === "trunkLean") samples = samples.filter((value) => Math.abs(value) <= 60);
      if (["elbowAngle", "shootingElbowAngle", "kneeAngle", "measuredKneeAngle", "hipAngle"].includes(key)) samples = samples.filter((value) => value >= 5 && value <= 179);
      const value = median(samples);
      if (Number.isFinite(value)) smoothed[key] = value;
    }
    return smoothed;
  });
}

function selectPrimaryShot(valid) {
  const phaseSamples = valid.map((frame, index) => ({ ...frame, sampleIndex: frame.sampleIndex ?? index, timeMs: frame.timeMs ?? Math.round(frame.time * 1000) }));
  const phases = selectDeterministicPhases(phaseSamples);
  const byIndex = (entry) => entry ? phaseSamples.find((frame) => frame.sampleIndex === entry.sampleIndex) : null;
  const loadingStart = byIndex(phases.loadingStart);
  const loading = byIndex(phases.lowestPoint);
  const release = byIndex(phases.release);
  const followThrough = byIndex(phases.followThrough);
  if (!loadingStart || !loading || !release || !followThrough) return { ready: null, loadingStart, loading, release, followThrough, analysisFrames: [], phases };
  const ready = valid.find((frame) => frame.time >= loadingStart.time) || loadingStart;
  const analysisFrames = valid.filter((frame) => frame.time >= loadingStart.time && frame.time <= followThrough.time);
  return { ready, loadingStart, loading, release, followThrough, analysisFrames, phases };
}

function captureQuality(measured, validRatio) {
  const usableFrames = measured.filter((frame) => frame.valid);
  const averageVisibility = mean(usableFrames.map((frame) => frame.averageVisibility).filter(Number.isFinite));
  const bodyScale = mean(usableFrames.map((frame) => frame.bodyScale).filter(Number.isFinite));
  const occlusionRatio = measured.length ? measured.filter((frame) => !frame.valid).length / measured.length : 1;
  const visibilityScore = clamp((averageVisibility || 0) * 100, 0, 100);
  const scaleScore = clamp((bodyScale || 0) / 0.55 * 100, 0, 100);
  const score = Math.round(validRatio * 55 + visibilityScore * 0.3 + scaleScore * 0.15);
  const confidence = validRatio >= 0.8 && averageVisibility >= 0.8 && bodyScale >= 0.3 ? "高"
    : validRatio >= 0.5 && averageVisibility >= 0.55 && bodyScale >= 0.12 ? "中" : "低";
  return { score, confidence, validRatio, averageVisibility: round(averageVisibility, 2), bodyScale: round(bodyScale, 2), occlusionRatio: round(occlusionRatio, 2) };
}

function eventAt(frame, label) {
  return frame ? { time: round(frame.time, 2), label } : { time: null, label };
}

function findLongestPause(frames) {
  let longest = 0;
  let current = 0;
  for (let index = 1; index < frames.length; index += 1) {
    const previous = frames[index - 1];
    const frame = frames[index];
    const still = Math.abs(frame.kneeAngle - previous.kneeAngle) < 1.6
      && Math.abs(frame.elbowAngle - previous.elbowAngle) < 2.2
      && Math.abs(frame.wristY - previous.wristY) < 0.008;
    current = still ? current + frame.time - previous.time : 0;
    longest = Math.max(longest, current);
  }
  return longest;
}

function confidenceNote(capture, qualifier = "") {
  const reason = capture.confidence === "高" ? "关键点可见度、有效帧比例和人物画面占比均较稳定"
    : capture.confidence === "中" ? "关键点总体可用，但拍摄角度或人物画面占比会放大误差"
      : "关键点遮挡或人物画面占比较弱，仅适合判断大方向";
  return `${capture.confidence}（${reason}${qualifier ? `；${qualifier}` : ""}）`;
}

function makeIssue(joint, title, severity, evidence, impact, nextBall, confidence) {
  const adjustedSeverity = joint !== "capture" && confidence === "中" && severity === "明显" ? "中等"
    : joint !== "capture" && confidence === "低" ? "轻微" : severity;
  return { joint, title, severity: adjustedSeverity, evidence, impact, nextBall, confidence };
}

function drillFor(joint) {
  const drills = {
    elbow: { name: "近筐单手投篮", purpose: "稳定肘腕伸展通道", steps: "距离篮筐 1–2 米，投篮手单手完成举球和出手，辅助手只轻扶球侧面；球出手后定住随挥 1 秒。", sets: "3 组", reps: "每组 10 球", focus: "肘部自然向上伸展、手腕朝篮筐方向随挥，不为追求角度强行锁肘。" },
    knee: { name: "固定下沉深度投篮", purpose: "建立可重复的下肢储能和起身节奏", steps: "先无球做 10 次下沉举手，用镜子或墙面标记头部最低高度；随后在近筐按同样深度投篮。", sets: "3 组", reps: "每组 10 球", focus: "每次下沉深度接近，脚蹬地后膝髋连续伸展，不在最低点停住。" },
    hip: { name: "髋膝同步举球", purpose: "减少只屈髋或只屈膝造成的动力链断点", steps: "持球站定，髋膝同时下沉，起身时让髋膝伸展带动球上升；先不投篮，熟练后接近筐投篮。", sets: "3 组", reps: "每组 8 次", focus: "胸口稳定、臀部不过度后坐，球与身体同时向上。" },
    trunk: { name: "垂直轴线定点投篮", purpose: "减少出手阶段躯干侧移和扑篮倾向", steps: "在脚下贴一条中线，近筐定点投篮；出手后检查胸口和头部是否仍在双脚中间上方。", sets: "3 组", reps: "每组 8 球", focus: "核心轻收紧、胸口保持稳定，落地点尽量接近起跳点。" },
    wrist: { name: "直线随挥单手投篮", purpose: "降低手腕二维轨迹的左右漂移", steps: "在篮板或墙面选择一条竖直参照线，距离 1–2 米单手投篮，让手腕沿参照线向上并朝目标随挥。", sets: "3 组", reps: "每组 10 球", focus: "观察手腕路线而不是手指细节，辅助手在出手前不主动推球。" },
    rhythm: { name: "一拍连续投篮", purpose: "减少最低点停顿，让下肢和上肢连续衔接", steps: "接球后用口令“下—上”完成一次连续动作；下沉时举球准备，起身后自然出手，中间不额外停球。", sets: "4 组", reps: "每组 6 球", focus: "速度不是越快越好，重点是连续、可重复，录像检查最低点附近是否停顿。" },
    capture: { name: "固定机位复拍", purpose: "提高后续角度和趋势判断可信度", steps: "手机固定在投篮手侧前方约 45°，镜头与腰腹同高，保证脚、头和随挥手全程入镜。", sets: "1 组", reps: "连续拍摄 3 球", focus: "人物高度占画面约 45%–70%，避免逆光和其他人遮挡。" },
  };
  const drill = drills[joint] || drills.capture;
  const commonMistakes = {
    elbow: "为了追求伸展而主动锁死肘关节，或辅助手同时把球推出去。",
    knee: "每次下沉深度不同，最低点停住后再重新起身。",
    hip: "只把臀部向后坐、胸口明显前扑，球与身体分开启动。",
    trunk: "刻意挺胸或后仰，落地位置明显越过起跳点。",
    wrist: "强行向内扭腕，或用低清晰度视频猜测手指拨球。",
    rhythm: "为了求快省略下沉，或在最低点停球后再二次发力。",
    capture: "手持跟拍、人物太小、脚或随挥手离开画面。",
  };
  const rightFeeling = {
    elbow: "球、肘和手腕连续向上，出手后手臂自然伸长但不僵硬。",
    knee: "下沉有弹性，起身像一整段动作，不需要在最低点重新用力。",
    hip: "身体和球同时上升，力量从脚下连续传到出手手臂。",
    trunk: "胸口稳定、视线平稳，起跳与落地位置接近。",
    wrist: "投篮手沿目标方向向上穿过，随挥自然落向篮筐方向。",
    rhythm: "能清楚感到“下—上”连续衔接，而不是“下—停—投”。",
    capture: "脚到随挥手全程清晰，人物占画面接近一半且没有遮挡。",
  };
  return { ...drill, rightFeeling: rightFeeling[joint] || rightFeeling.capture, commonMistakes: commonMistakes[joint] || commonMistakes.capture };
}

function reportSection(id, title, currentData, reference, evaluation, good, problem, impact, nextShot, longTerm, confidence) {
  return { id, title, currentData, reference: `训练参考：${reference}`, evaluation, good, problem, impact, nextShot, longTerm, confidence };
}

function metricText(value, suffix = "") {
  return Number.isFinite(value) ? `${value}${suffix}` : `—${suffix}`;
}

const TECHNICAL_LIMITATIONS = [
  "当前是单摄像头二维姿态趋势分析，相机角度、透视和遮挡会影响角度。",
  "系统只分析手腕二维轨迹，不从姿态点伪造手指拨球、旋转或触球细节。",
  "单球不能代表长期稳定性；多球结论需要同机位、同距离的连续样本。",
  "训练参考区间用于比较趋势，不是适用于所有人的标准动作答案。",
];

function finalizeAnalysisResult(base, shootingHand, context = {}) {
  const events = Object.values(base.events || {});
  const usable = Boolean(base.release && base.analysisFrames?.length);
  const strengths = (base.strengths || []).slice(0, 4);
  const duration = Number.isFinite(context.duration) ? round(context.duration, 2)
    : round(base.analysisFrames?.at(-1)?.time ?? base.release?.time ?? 0, 2);
  const phaseFrames = [
    [base.loading, "最低点"], [base.release, "出手候选"],
    [base.analysisFrames?.at(-1), "随挥"],
  ].filter(([frame], index, rows) => frame && rows.findIndex(([candidate]) => candidate?.time === frame.time) === index)
    .map(([frame, phase]) => ({ phase, time: round(frame.time, 2), landmarks: frame.landmarks }));
  const capture = base.capture || {};
  const priorities = base.priorities || [];
  return {
    ...base,
    schemaVersion: "3.0",
    summary: {
      oneLine: capture.confidence === "低" ? "本次拍摄证据不足，先提高机位与入镜质量再判断动作。"
        : priorities.length ? `本次最优先处理“${priorities[0].title}”，同时保持已经稳定的动作环节。`
          : "本次没有检测到高优先级问题，下一球重点验证动作能否重复。",
      duration,
      sourceFrames: context.sourceFrames !== undefined ? context.sourceFrames : (base.totalFrames ?? 0),
      sampledFrames: base.totalFrames ?? 0,
      validPoseFrames: base.validFrames ?? 0,
      poseDetectionRate: round((base.validRatio || 0) * 100),
      shootingHand: shootingHand === "left" ? "左手" : "右手",
      fileName: context.fileName || "当前视频",
      captureQuality: `${capture.score ?? 0}/100`,
      confidence: capture.confidence || "低",
      keyPhasesFound: phaseFrames.map((item) => item.phase),
    },
    strengths,
    keep: strengths.length ? strengths : ["先保持自然动作，不根据低可信度数据强行改姿势。"],
    why: priorities.length ? priorities.map((item) => `${item.title}：${item.impact}`) : ["当前没有高优先级动作问题，先观察同机位重复性。"],
    nextRep: base.nextShot || [],
    next_rep: base.nextShot || [],
    jointAnalysis: (base.sections || []).filter((section) => section.id !== "rhythm"),
    joint_analysis: (base.sections || []).filter((section) => section.id !== "rhythm"),
    rhythm: { events, analysis: (base.sections || []).find((section) => section.id === "rhythm") || null },
    keyframes: phaseFrames,
    processedVideo: { available: usable, mode: "original-video-with-live-pose-overlay", startTime: base.analysisFrames?.[0]?.time ?? null, endTime: base.analysisFrames?.at(-1)?.time ?? null },
    processed_video: { available: usable, mode: "original-video-with-live-pose-overlay" },
    skeletonVideo: { available: usable, mode: "canvas-stick-figure-replay", frameCount: base.analysisFrames?.length || 0 },
    skeleton_video: { available: usable, mode: "canvas-stick-figure-replay" },
    charts: { angleCurves: base.curves || [] },
    trainingPlan: base.prescriptions || [],
    training_plan: base.prescriptions || [],
    confidence: { level: capture.confidence || "低", score: capture.score ?? 0, validFrameRatio: capture.validRatio ?? 0, averageVisibility: capture.averageVisibility ?? 0, bodyScale: capture.bodyScale ?? 0, occlusionRatio: capture.occlusionRatio ?? 1 },
    personalBaseline: { version: 1, status: "not_enough_samples", minimumShots: 5, samples: [], comparableSetup: ["同一机位", "相近距离", "同一投篮类型"] },
    technicalLimitations: TECHNICAL_LIMITATIONS,
  };
}

function lowConfidenceReport(frames, measured, capture, shootingHand, context) {
  const reshoot = "先重新拍摄：固定手机并确保脚到随挥手全程入镜；左右侧均可，正面或背面更适合看节奏，不用于强判前后屈伸角。";
  const section = (id, title) => reportSection(id, title, ["当前有效数据不足"], "先保证关键点连续可见，再讨论动作趋势", "本项可信度低", "系统没有在证据不足时强行下动作结论", "遮挡、人物过小或有效帧不足", "具体角度和相对调整量可能被拍摄误差放大", reshoot, "保持固定机位和相似拍摄距离，才能比较多次训练趋势", confidenceNote(capture));
  const priorities = [makeIssue("capture", "先提高拍摄质量", "明显", `有效帧 ${Math.round(capture.validRatio * 100)}%`, "当前数据不足以支持精确动作纠正", reshoot, "低")];
  return finalizeAnalysisResult({
    validFrames: measured.filter((frame) => frame.valid).length, totalFrames: frames.length, validRatio: capture.validRatio, capture,
    ready: null, loading: null, release: null,
    events: { loadingStart: eventAt(null, "下沉开始"), lowest: eventAt(null, "最低点"), riseStart: eventAt(null, "起身"), release: eventAt(null, "出手"), followThrough: eventAt(null, "随挥") },
    metrics: { elbowRelease: null, elbowMin: null, elbowRange: null, elbowExtensionTrend: null, elbowInstability: null, kneeLowest: null, kneeRange: null, kneeRiseDuration: null, hipLowest: null, hipKneeCoordination: null, trunkMax: null, trunkRelease: null, trunkDrift: null, wristRise: null, wristLateralDrift: null, wristPathStability: null, pauseDuration: null },
    strengths: ["系统已识别到当前证据不足，没有输出伪精确的动作结论"], priorities, nextShot: [reshoot], dataRows: [],
    sections: [["elbow", "肘关节"], ["knee", "膝关节"], ["hip", "髋关节"], ["trunk", "躯干"], ["wrist", "手腕与出手轨迹"], ["rhythm", "投篮节奏"]].map(([id, title]) => section(id, title)),
    prescriptions: [{ ...drillFor("capture"), forIssue: "拍摄质量不足" }], curves: [], analysisFrames: [], suggestions: [{ title: "建议重新拍摄", detail: reshoot }],
  }, shootingHand, context);
}

function dataRow(metric, current, reference, deviation, evaluation, nextStep, confidence) {
  return { metric, current, reference: `训练参考：${reference}`, deviation, evaluation, nextStep, confidence };
}

export function summarizePoseFrames(frames, shootingHand = "right", context = {}) {
  const sideSelection = selectAnalysisSides(frames, shootingHand);
  const measured = frames.map((frame) => frameMetrics(frame, sideSelection));
  const rawValid = measured.filter((frame) => frame.valid);
  const valid = context.preprocessed ? rawValid : smoothFrames(rawValid);
  const validRatio = frames.length ? valid.length / frames.length : 0;
  const confidenceMeasured = context.rawFrames ? context.rawFrames.map((frame) => frameMetrics(frame, sideSelection)) : measured;
  const rawValidRatio = confidenceMeasured.length ? confidenceMeasured.filter((frame) => frame.valid).length / confidenceMeasured.length : 0;
  const viewpoint = estimateViewpoint(context.rawFrames || frames);
  const capture = {
    ...captureQuality(confidenceMeasured, rawValidRatio),
    viewpoint,
    analysisSides: { arm: sideSelection.arm, body: sideSelection.body, shooting: sideSelection.shooting },
    sideCoverage: sideSelection.coverage,
    shootingArmOccluded: sideSelection.shootingArmOccluded,
    orientationPass: context.orientationPass || "original",
    orientationScores: context.orientationScores || null,
  };
  if (!valid.length || capture.confidence === "低") return lowConfidenceReport(frames, measured, capture, shootingHand, context);

  const { ready, loadingStart, loading, release, followThrough, analysisFrames, phases } = selectPrimaryShot(valid);
  if (!release || analysisFrames.length < 3) {
    const fallback = lowConfidenceReport(frames, measured, { ...capture, confidence: "低" }, shootingHand, { ...context, phaseMissing: phases?.missing || ["release"] });
    fallback.phaseDiagnostics = { phases, samples: valid.map(({ sampleIndex, timeMs, time, kneeAngle, elbowAngle, wristY }) => ({ sampleIndex, timeMs, time, kneeAngle: round(kneeAngle), elbowAngle: round(elbowAngle), wristY: round(wristY, 3) })) };
    return fallback;
  }
  const loadingToRelease = analysisFrames.filter((frame) => frame.time >= loading.time && frame.time <= release.time);
  const values = (key) => analysisFrames.map((frame) => frame[key]).filter(Number.isFinite);
  const elbowDeltas = loadingToRelease.slice(1).map((frame, index) => (frame.elbowAngle - loadingToRelease[index].elbowAngle) / Math.max(0.001, frame.time - loadingToRelease[index].time));
  const kneeDeltas = loadingToRelease.slice(1).map((frame, index) => frame.measuredKneeAngle - loadingToRelease[index].measuredKneeAngle);
  const hipDeltas = loadingToRelease.slice(1).map((frame, index) => frame.hipAngle - loadingToRelease[index].hipAngle);
  const torsoLength = median(analysisFrames.map((frame) => frame.torsoLength).filter((value) => value > 0)) || 0.25;
  const measuredShoulderWidth = median(analysisFrames.map((frame) => frame.shoulderWidth).filter((value) => value > 0));
  const shoulderWidth = Math.max(measuredShoulderWidth || 0, torsoLength * 0.5);
  const wristPath = analysisFrames.filter((frame) => Number.isFinite(frame.shootingWristX) && Number.isFinite(frame.shootingWristY));
  const pathStart = wristPath[0];
  const pathEnd = wristPath.at(-1);
  const lineLength = pointDistance(
    pathStart && { x: pathStart.shootingWristX, y: pathStart.shootingWristY },
    pathEnd && { x: pathEnd.shootingWristX, y: pathEnd.shootingWristY },
  ) || 1;
  const pathResiduals = pathStart && pathEnd ? wristPath.map((frame) => Math.abs(
    (pathEnd.shootingWristX - pathStart.shootingWristX) * (pathStart.shootingWristY - frame.shootingWristY)
      - (pathStart.shootingWristX - frame.shootingWristX) * (pathEnd.shootingWristY - pathStart.shootingWristY),
  ) / lineLength) : [];
  const trunkValues = values("trunkLean").filter((value) => Math.abs(value) <= 60);
  const shootingArmFrames = analysisFrames.filter((frame) => Number.isFinite(frame.shootingElbowAngle));
  const shootingLoadingToRelease = loadingToRelease.filter((frame) => Number.isFinite(frame.shootingElbowAngle));
  const shootingElbowDeltas = shootingLoadingToRelease.slice(1).map((frame, index) => (frame.shootingElbowAngle - shootingLoadingToRelease[index].shootingElbowAngle) / Math.max(0.001, frame.time - shootingLoadingToRelease[index].time));
  const elbowTrendReady = shootingLoadingToRelease.length >= 4;

  const metrics = {
    elbowRelease: round(release.shootingElbowAngle), elbowMin: round(percentile(shootingArmFrames.map((frame) => frame.shootingElbowAngle), 0.1)), elbowRange: round(percentile(shootingArmFrames.map((frame) => frame.shootingElbowAngle), 0.9) - percentile(shootingArmFrames.map((frame) => frame.shootingElbowAngle), 0.1)),
    elbowExtensionTrend: elbowTrendReady ? round(regressionSlope(shootingLoadingToRelease, "shootingElbowAngle")) : null, elbowInstability: elbowTrendReady ? round(standardDeviation(shootingElbowDeltas)) : null,
    kneeLowest: round(percentile(values("measuredKneeAngle"), 0.1)), kneeRange: round(percentile(values("measuredKneeAngle"), 0.9) - percentile(values("measuredKneeAngle"), 0.1)), kneeRiseDuration: round(release.time - loading.time, 2),
    hipLowest: round(percentile(values("hipAngle"), 0.1)), hipRange: round(percentile(values("hipAngle"), 0.9) - percentile(values("hipAngle"), 0.1)), hipKneeCoordination: round(correlation(kneeDeltas, hipDeltas), 2),
    trunkMax: round(percentile(trunkValues.map(Math.abs), 0.9)), trunkRelease: round(clamp(release.trunkLean, -60, 60)), trunkDrift: round(percentile(trunkValues, 0.9) - percentile(trunkValues, 0.1)),
    wristRise: Number.isFinite(ready.shootingWristY) && Number.isFinite(release.shootingWristY) ? round((ready.shootingWristY - release.shootingWristY) / torsoLength * 100) : null,
    wristLateralDrift: wristPath.length >= 3 ? round(extent(wristPath.map((frame) => frame.shootingWristX)) / shoulderWidth * 100) : null,
    wristPathStability: pathResiduals.length >= 3 ? round(mean(pathResiduals.filter(Number.isFinite)) / shoulderWidth * 100) : null,
    pauseDuration: round(findLongestPause(loadingToRelease), 2), rhythmDuration: round(release.time - loadingStart.time, 2),
  };

  const confidence = capture.confidence;
  const frontOrRear = viewpoint.category === "front_or_rear";
  const confidenceDetail = confidenceNote(capture, `${viewpoint.label}；单摄像头只能判断画面平面内趋势`);
  const spatiallyWeak = capture.bodyScale < 0.18;
  const jointConfidence = {
    elbow: spatiallyWeak || sideSelection.shootingArmOccluded || frontOrRear || !Number.isFinite(metrics.elbowRelease) ? "低" : confidence,
    knee: spatiallyWeak || frontOrRear || !Number.isFinite(metrics.kneeRange) ? "低" : confidence,
    hip: spatiallyWeak || frontOrRear || !Number.isFinite(metrics.hipKneeCoordination) ? "低" : confidence,
    trunk: frontOrRear || spatiallyWeak || !Number.isFinite(metrics.trunkMax) || metrics.trunkMax > 35 ? "低" : confidence,
    wrist: sideSelection.shootingArmOccluded || spatiallyWeak || !Number.isFinite(metrics.wristPathStability) ? "低" : confidence,
    rhythm: confidence,
  };
  const strengths = [];
  const issues = [];
  if (jointConfidence.elbow !== "低" && metrics.elbowRelease >= 150) strengths.push(`出手候选帧肘部伸展较完整（${metrics.elbowRelease}°），这一点值得继续保持`);
  else if (jointConfidence.elbow !== "低" && metrics.elbowRelease < 145) issues.push(makeIssue("elbow", "出手伸展可能不足", metrics.elbowRelease < 135 ? "明显" : "中等", `出手肘角 ${metrics.elbowRelease}°，伸展幅度 ${metrics.elbowRange}°`, "可能让出手更多依赖手腕瞬间发力，随挥重复性下降", "下一球让肘部继续向篮筐上方伸展，并把随挥定住 1 秒；不要强行锁肘。", jointConfidence.elbow));
  else if (jointConfidence.elbow !== "低") issues.push(makeIssue("elbow", "肘部伸展仍可观察", "轻微", `出手肘角 ${metrics.elbowRelease}°`, "在疲劳或远距离时可能更容易缩短随挥", "下一球保持当前发力，不加速抢出手，只观察肘腕是否连续。", jointConfidence.elbow));

  if (jointConfidence.knee !== "低" && metrics.kneeRange >= 22 && metrics.kneeRange <= 62) strengths.push(`检测到清晰的膝关节屈伸（幅度 ${metrics.kneeRange}°），下肢参与明显`);
  else if (jointConfidence.knee !== "低" && metrics.kneeRange < 22) issues.push(makeIssue("knee", "下沉幅度偏浅", metrics.kneeRange < 12 ? "明显" : "中等", `本次屈伸幅度 ${metrics.kneeRange}°，最低点膝角 ${metrics.kneeLowest}°`, "下肢储能可能不足，上肢需要承担更多发力", `下一球比当前稍深，尝试增加约 ${metrics.kneeRange < 12 ? "10°–15°" : "5°–10°"} 的相对屈膝幅度，不追求固定角度。`, jointConfidence.knee));
  else if (jointConfidence.knee !== "低") issues.push(makeIssue("knee", "下沉幅度可能偏深", "中等", `本次屈伸幅度 ${metrics.kneeRange}°`, "动作时间可能拉长，急停投篮时不易重复", "下一球把下沉幅度比当前减少约 5°–10°，保持起身连续。", jointConfidence.knee));

  if (jointConfidence.hip !== "低" && Number.isFinite(metrics.hipKneeCoordination) && metrics.hipKneeCoordination >= 0.35) strengths.push("从当前二维趋势看，髋膝伸展方向较一致，动力链衔接较自然");
  else if (jointConfidence.hip !== "低" && Number.isFinite(metrics.hipKneeCoordination)) issues.push(makeIssue("hip", "髋膝衔接可能不同步", metrics.hipKneeCoordination < 0 ? "中等" : "轻微", `髋膝变化相关系数 ${metrics.hipKneeCoordination}`, "可能出现先起身再举球，或只用上肢补偿的分段感", "下一球只注意一件事：身体起身时让球同时开始向上，不在额前停球。", jointConfidence.hip));

  if (jointConfidence.trunk !== "低" && metrics.trunkMax <= 12 && metrics.trunkDrift <= 10) strengths.push(`躯干画面轴线较稳定（出手倾角 ${metrics.trunkRelease}°）`);
  else if (jointConfidence.trunk !== "低") issues.push(makeIssue("trunk", "躯干偏移需要控制", metrics.trunkMax > 20 ? "明显" : metrics.trunkMax > 14 ? "中等" : "轻微", `最大画面倾角 ${metrics.trunkMax}°，出手时 ${metrics.trunkRelease}°`, "可能出现身体扑向篮筐或侧移；单摄像头无法可靠区分真实前倾与相机透视", "下一球保持胸口位于双脚中间上方，出手后检查落地点是否接近起跳点。", jointConfidence.trunk));

  if (jointConfidence.wrist !== "低" && metrics.wristLateralDrift <= 45 && metrics.wristPathStability <= 18) strengths.push("手腕二维随挥路线较集中，方向重复性可以继续保持");
  else if (jointConfidence.wrist !== "低") issues.push(makeIssue("wrist", "手腕轨迹存在左右漂移", metrics.wristLateralDrift > 85 ? "明显" : "中等", `横向范围约为上身参考宽度的 ${metrics.wristLateralDrift}%`, "可能使出手方向更依赖临场修正，左右偏差增加", "下一球让手腕沿篮筐方向的竖直参照线向上随挥；不根据本视频判断手指拨球细节。", jointConfidence.wrist));

  if (metrics.pauseDuration <= 0.12 && metrics.rhythmDuration <= 1.2) strengths.push("下沉到出手的动作基本连续，没有检测到明显长停顿");
  else issues.push(makeIssue("rhythm", "动力链可能存在停顿", metrics.pauseDuration > 0.28 ? "明显" : "中等", `最长近似停顿 ${metrics.pauseDuration}s，下沉至出手 ${metrics.rhythmDuration}s`, "动作可能变成一段一段，上肢需要重新启动发力", "下一球用“下—上”两拍口令，最低点不停住，起身和举球同时发生。", jointConfidence.rhythm));

  if (!strengths.length) strengths.push("本次关键点连续可见，已经具备基于趋势进行动作复盘的条件");
  const priorities = issues.filter((issue) => issue.confidence !== "低").sort((a, b) => SEVERITY_WEIGHT[b.severity] - SEVERITY_WEIGHT[a.severity]).slice(0, 2);
  const nextShot = priorities.map((issue) => issue.nextBall);
  if (nextShot.length < 2) nextShot.push(metrics.elbowRelease >= 150 ? "继续保持当前肘部伸展，出手后把随挥定住 1 秒。" : "保持当前动作节奏，再投一球并观察相同指标能否重复。");
  if (nextShot.length < 2) nextShot.push("保持最低点不停顿，让起身和举球继续连在一起。");
  const prescriptions = (priorities.length ? priorities : [{ joint: "rhythm" }]).map((issue) => ({ ...drillFor(issue.joint), forIssue: issue.title || "动作重复性" }));
  const elbowEvaluation = !Number.isFinite(metrics.elbowRelease) ? "投篮侧手臂证据不足" : metrics.elbowRelease >= 150 ? "伸展较充分" : metrics.elbowRelease < 145 ? "伸展偏少" : "接近参考趋势，继续观察";
  const kneeEvaluation = !Number.isFinite(metrics.kneeRange) ? "膝角证据不足，仅保留动作时序" : metrics.kneeRange < 22 ? "下沉偏浅" : metrics.kneeRange > 62 ? "下沉可能偏深" : "下沉幅度较清晰";
  const trunkEvaluation = Number.isFinite(metrics.trunkMax) && metrics.trunkMax > 14 ? "画面内倾斜较明显" : "躯干轴线较稳定";
  const wristEvaluation = metrics.wristLateralDrift > 60 ? "二维轨迹漂移偏大" : "二维轨迹较集中";
  const rhythmEvaluation = metrics.pauseDuration > 0.16 ? "可能存在分段或停顿" : "动作衔接较连续";

  const sections = [
    reportSection("elbow", "肘关节", [`出手肘角 ${metrics.elbowRelease}°`, `最小肘角 ${metrics.elbowMin}°`, `伸展幅度 ${metrics.elbowRange}°`, `伸展趋势 ${metrics.elbowExtensionTrend}°/s`, `波动 ${metrics.elbowInstability}°/s`], "出手阶段通常呈持续伸展趋势；约 145°–175°只作侧面训练参考，并非标准答案", elbowEvaluation, metrics.elbowRelease >= 150 ? "伸展较完整，随挥基础较好" : "仍能观察到明确的肘部伸展动作", metrics.elbowRelease < 145 ? "出手候选帧伸展偏少" : "需继续观察疲劳和远距离时的稳定性", "伸展不足可能增加手腕瞬间发力；波动过大可能降低重复性", priorities.find((item) => item.joint === "elbow")?.nextBall || "下一球维持当前幅度，把随挥定住 1 秒。", "近筐单手投篮，3 组×10 球，比较每组出手角度和轨迹离散程度", confidenceDetail),
    reportSection("knee", "膝关节", [`最低点膝角 ${metrics.kneeLowest}°`, `屈伸幅度 ${metrics.kneeRange}°`, `最低点至出手 ${metrics.kneeRiseDuration}s`], "下沉深度取决于距离、力量和个人节奏；优先比较自身屈伸幅度与多球一致性", kneeEvaluation, metrics.kneeRange >= 22 && metrics.kneeRange <= 62 ? "下肢参与清晰，没有机械追求过度下蹲" : "动作没有出现不可控的深蹲趋势", metrics.kneeRange < 22 ? "相对屈伸幅度偏小" : metrics.kneeRange > 62 ? "相对屈伸幅度较大" : "单球无法判断多球稳定性", "偏浅可能增加上肢负担；偏深可能拖慢起身节奏", priorities.find((item) => item.joint === "knee")?.nextBall || "下一球保持当前下沉深度，重点复现同样节奏。", "固定下沉深度投篮，3 组×10 球；长期关注同机位多球标准差", confidenceDetail),
    reportSection("hip", "髋关节", [`最低点髋角 ${metrics.hipLowest}°`, `髋部变化 ${metrics.hipRange}°`, `髋膝协调趋势 ${metrics.hipKneeCoordination}`], "髋膝应呈同向、连续的屈伸趋势；相关系数只用于同一机位内比较", metrics.hipKneeCoordination >= 0.35 ? "髋膝协同趋势较好" : "髋膝协同需要观察", metrics.hipKneeCoordination >= 0.35 ? "髋膝起身方向一致" : "仍检测到髋部参与，而非完全直立手投", metrics.hipKneeCoordination < 0.35 ? "变化趋势可能不同步" : "单次视频暂不能判断多球重复性", "不同步可能让蹬地力量无法连续传到举球和出手", priorities.find((item) => item.joint === "hip")?.nextBall || "下一球保持球与身体同时向上。", "髋膝同步举球，3 组×8 次；先无球后接近筐投篮", confidenceDetail),
    reportSection("trunk", "躯干", [`最大画面倾角 ${metrics.trunkMax}°`, `出手画面倾角 ${metrics.trunkRelease}°`, `全程变化 ${metrics.trunkDrift}°`], "优先观察轴线变化和同机位一致性；单个二维角度不能直接等同真实前倾或后仰", trunkEvaluation, metrics.trunkMax <= 14 ? "胸口和髋部画面轴线较稳定" : "出手阶段仍保持了基本身体控制", metrics.trunkMax > 14 ? "可能存在扑篮、后仰或侧移，需要结合原视频方向确认" : "暂未发现明显画面内倾斜", "躯干漂移可能改变出手空间和落点，降低动作重复性", priorities.find((item) => item.joint === "trunk")?.nextBall || "下一球保持胸口稳定，出手后观察落地点。", "垂直轴线定点投篮，3 组×8 球；使用脚下中线检查起跳与落地", confidenceDetail),
    reportSection("wrist", "手腕与出手轨迹", [`相对抬升 ${metrics.wristRise}% 躯干长`, `左右范围 ${metrics.wristLateralDrift}% 肩宽`, `路线离散 ${metrics.wristPathStability}% 肩宽`], "只分析手腕二维路线、高度与随挥方向，不从低分辨率姿态点推断手指拨球", wristEvaluation, metrics.wristLateralDrift <= 60 ? "随挥二维方向较集中" : "手腕完成了明显向上运动", metrics.wristLateralDrift > 60 ? "路线存在较明显左右漂移" : "仍需用多球验证路线一致性", "横向漂移可能增加左右方向误差，但无法据此判断具体手指问题", priorities.find((item) => item.joint === "wrist")?.nextBall || "下一球让手腕沿目标方向自然随挥，不刻意扭腕。", "直线随挥单手投篮，3 组×10 球；以竖直参照线检查轨迹", confidenceDetail),
    reportSection("rhythm", "投篮节奏", [`下沉开始 ${round(loadingStart.time, 2)}s`, `最低点 ${round(loading.time, 2)}s`, `起身 ${round(loading.time, 2)}s`, `出手 ${round(release.time, 2)}s`, `随挥 ${round(followThrough.time, 2)}s`], "优先观察下沉—起身—举球—出手是否连续，以及多次动作时间差是否稳定", rhythmEvaluation, metrics.pauseDuration <= 0.16 ? "从当前二维姿态趋势看，下肢起身与上肢动作较连续" : "动作阶段仍能被完整识别", metrics.pauseDuration > 0.16 ? "最低点到出手之间可能存在停顿或分段" : "单球无法判断多球节奏稳定性", "明显停顿可能让已有下肢动量中断，出手转为上肢二次发力", priorities.find((item) => item.joint === "rhythm")?.nextBall || "下一球保持当前速度，用“下—上”口令复现节奏。", "一拍连续投篮，4 组×6 球；比较每球最低点到出手的时间", confidenceDetail),
  ];
  for (const item of sections) item.currentData = item.currentData.map((text) => text.replaceAll("null", "—").replaceAll("% 肩宽", "% 上身参考宽度"));
  for (const item of sections) item.confidence = confidenceNote({ ...capture, confidence: jointConfidence[item.id] }, "单摄像头只能判断画面平面内趋势");
  if (jointConfidence.elbow === "低") {
    const elbowSection = sections.find((item) => item.id === "elbow");
    elbowSection.evaluation = sideSelection.shootingArmOccluded ? "动作阶段已识别，投篮侧手臂精细角度暂不强判" : "当前视角不适合强判肘部屈伸角";
    elbowSection.good = "系统没有用更清楚的辅助侧手臂冒充投篮手臂数据";
    elbowSection.problem = sideSelection.shootingArmOccluded ? "投篮手臂位于身体远侧，肩肘腕连续可见帧不足" : "正面或背面透视会压缩屈伸角度";
    elbowSection.nextShot = "本球的阶段和节奏仍可使用；如需精细肘腕分析，再补拍一段投篮手侧或斜侧视频。";
  }
  if (jointConfidence.knee === "低") {
    const kneeSection = sections.find((item) => item.id === "knee");
    kneeSection.evaluation = "动作阶段已识别，前后屈膝幅度暂不强判";
    kneeSection.good = "下沉与起身时序仍可用于节奏判断";
    kneeSection.problem = "正面或背面画面无法可靠还原矢状面的真实屈膝角";
    kneeSection.nextShot = "下一球先保持自然下沉；不要仅根据本机位的二维膝角主动加深或变浅。";
  }
  if (!Number.isFinite(metrics.hipKneeCoordination)) {
    const hipSection = sections.find((item) => item.id === "hip");
    hipSection.evaluation = "主动作窗口样本不足";
    hipSection.good = "髋膝关键点已经被捕捉，系统未用少量帧强判协调性";
    hipSection.problem = "当前样本不足以稳定计算髋膝协同趋势";
  }
  if (jointConfidence.hip === "低" && frontOrRear) {
    const hipSection = sections.find((item) => item.id === "hip");
    hipSection.evaluation = "可判断髋膝启动时序，不强判前后屈髋幅度";
    hipSection.problem = "正面或背面透视不适合将二维髋角直接解释为屈髋深度";
    hipSection.nextShot = "先观察髋膝是否同时启动；需要屈髋幅度时使用侧面或斜侧补拍。";
  }
  if (jointConfidence.trunk === "低") {
    const trunkSection = sections.find((item) => item.id === "trunk");
    trunkSection.evaluation = "当前证据不足，不强判扑篮或后仰";
    trunkSection.good = "系统保留了躯干趋势，但没有把远景误差当作动作事实";
    trunkSection.problem = "人物画面占比较小或透视影响明显，躯干数值可能被放大";
    trunkSection.nextShot = "下一球先把机位移近并保持全身入镜，再判断胸口是否前扑或后仰。";
  }
  if (jointConfidence.wrist === "低") {
    const wristSection = sections.find((item) => item.id === "wrist");
    wristSection.evaluation = "当前轨迹只作方向提示";
    wristSection.good = "系统只分析手腕二维点位，没有伪造手指拨球结论";
    wristSection.problem = sideSelection.shootingArmOccluded ? "投篮手位于身体远侧，系统没有用辅助手轨迹替代投篮手轨迹" : "人物较小或有效轨迹过短，左右漂移比例不宜用于精确纠正";
    wristSection.nextShot = sideSelection.shootingArmOccluded ? "本球继续使用节奏结论；需要随挥路线时补拍投篮手侧或斜侧视频。" : "下一球先拍近一些，保持投篮手完整入镜，再比较随挥方向。";
  }

  const dataRows = [
    dataRow("出手肘角", formatMetric(metrics.elbowRelease), "约 145°–175°，结合个人出手模式", metrics.elbowRelease < 145 ? "相对偏小" : "区间内", elbowEvaluation, sections[0].nextShot, confidence),
    dataRow("肘部伸展幅度", formatMetric(metrics.elbowRange), "重点看伸展是否连续、重复", metrics.elbowRange < 25 ? "幅度偏小" : "有明显伸展", metrics.elbowExtensionTrend > 0 ? "向伸展方向" : "趋势需观察", sections[0].nextShot, confidence),
    dataRow("最低点膝角", formatMetric(metrics.kneeLowest), "不设唯一目标，结合相对下沉幅度", "仅记录", kneeEvaluation, sections[1].nextShot, confidence),
    dataRow("膝部屈伸幅度", formatMetric(metrics.kneeRange), "个人同机位约 22°–62°趋势参考", metrics.kneeRange < 22 ? "偏小" : metrics.kneeRange > 62 ? "偏大" : "范围内", kneeEvaluation, sections[1].nextShot, confidence),
    dataRow("髋膝协调", Number.isFinite(metrics.hipKneeCoordination) ? `${metrics.hipKneeCoordination}` : "—", "同向连续变化，长期看一致性", metrics.hipKneeCoordination < 0.35 ? "协同偏弱" : "协同较好", sections[2].evaluation, sections[2].nextShot, confidence),
    dataRow("出手躯干倾角", formatMetric(metrics.trunkRelease), "优先看同机位变化；绝对值仅作画面参考", Math.abs(metrics.trunkRelease) > 14 ? "偏移明显" : "相对稳定", trunkEvaluation, sections[3].nextShot, confidence),
    dataRow("手腕横向漂移", `${metrics.wristLateralDrift}% 肩宽`, "越集中越利于重复；不分析手指", metrics.wristLateralDrift > 60 ? "漂移较大" : "较集中", wristEvaluation, sections[4].nextShot, confidence),
    dataRow("最长近似停顿", `${metrics.pauseDuration}s`, "连续动作通常不出现明显静止段", metrics.pauseDuration > 0.16 ? "可能停顿" : "未见明显停顿", rhythmEvaluation, sections[5].nextShot, confidence),
  ];
  if (!Number.isFinite(metrics.hipKneeCoordination)) {
    const hipRow = dataRows.find((item) => item.metric === "髋膝协调");
    hipRow.current = "—";
    hipRow.deviation = "样本不足";
    hipRow.evaluation = "暂不强判";
  }
  const wristRow = dataRows.find((item) => item.metric === "手腕横向漂移");
  wristRow.current = wristRow.current.replace("肩宽", "上身参考宽度");
  for (const item of dataRows) {
    if (item.metric.includes("肘")) {
      item.confidence = jointConfidence.elbow;
      if (item.confidence === "低") { item.deviation = "证据不足"; item.evaluation = "暂不强判"; }
    }
    else if (item.metric.includes("膝") && !item.metric.includes("髋膝")) {
      item.confidence = jointConfidence.knee;
      if (item.confidence === "低") { item.deviation = "视角不适用"; item.evaluation = "只保留时序"; }
    }
    else if (item.metric.includes("髋膝")) {
      item.confidence = jointConfidence.hip;
      if (item.confidence === "低") { item.deviation = "视角不适用"; item.evaluation = "只观察启动时序"; }
    }
    else if (item.metric.includes("躯干")) {
      item.confidence = jointConfidence.trunk;
      if (item.confidence === "低") { item.deviation = "证据不足"; item.evaluation = "暂不强判"; }
    } else if (item.metric.includes("手腕")) {
      item.confidence = jointConfidence.wrist;
      if (item.confidence === "低") { item.deviation = "证据不足"; item.evaluation = "只作方向提示"; }
    }
    else item.confidence = jointConfidence.rhythm;
  }
  const events = { loadingStart: eventAt(loadingStart, "下沉开始"), lowest: eventAt(loading, "最低点"), riseStart: eventAt(loading, "起身"), release: eventAt(release, "出手"), followThrough: eventAt(followThrough, "随挥") };
  const curves = analysisFrames.map((frame) => ({ time: round(frame.time, 2), elbow: round(frame.shootingElbowAngle), knee: round(frame.measuredKneeAngle), hip: round(frame.hipAngle), trunk: round(frame.trunkLean), wristX: round(frame.shootingWristX, 3), wristY: round(frame.shootingWristY, 3) }));
  return finalizeAnalysisResult({
    validFrames: valid.length, totalFrames: frames.length, validRatio, capture, ready, loading, release, events, metrics,
    strengths, priorities, nextShot, dataRows, sections, prescriptions, curves, analysisFrames, jointConfidence,
    suggestions: priorities.length ? priorities.map((item) => ({ title: item.title, detail: `${item.evidence}。${item.nextBall}` })) : strengths.slice(0, 3).map((text) => ({ title: "当前保持", detail: text })),
  }, shootingHand, context);
}

export function formatMetric(value, suffix = "°") {
  return value === null || value === undefined || !Number.isFinite(value) ? "—" : `${value}${suffix}`;
}
