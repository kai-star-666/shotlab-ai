const confidenceScore = { 高: 30, 中: 18, 低: -100 };
const severityPenalty = { 明显: 12, 中等: 8, 轻微: 4 };

export const SESSION_METRICS = [
  { key: "kneeLowest", label: "最低点膝角", unit: "°", tolerance: 2, zone: null, step: 8 },
  { key: "kneeRange", label: "膝部屈伸幅度", unit: "°", tolerance: 2, zone: [22, 62], step: 5 },
  { key: "hipLowest", label: "最低点髋角", unit: "°", tolerance: 2, zone: null, step: 5 },
  { key: "elbowRelease", label: "出手肘角", unit: "°", tolerance: 2, zone: [145, 175], step: 5 },
  { key: "elbowRange", label: "肘部伸展幅度", unit: "°", tolerance: 2, zone: [25, 90], step: 5 },
  { key: "trunkRelease", label: "出手躯干倾角", unit: "°", tolerance: 2, zone: [-14, 14], step: 3 },
  { key: "wristPathStability", label: "手腕路线离散", unit: "%", tolerance: 5, zone: [0, 18], step: 5 },
  { key: "rhythmDuration", label: "下沉至出手", unit: "s", tolerance: 0.05, zone: [0, 1.2], step: 0.05 },
  { key: "pauseDuration", label: "最长近似停顿", unit: "s", tolerance: 0.05, zone: [0, 0.16], step: 0.05 },
];

const metricDefinitions = Object.fromEntries(SESSION_METRICS.map((item) => [item.key, item]));
const angleKeys = new Set(SESSION_METRICS.filter((item) => item.unit === "°").map((item) => item.key));
const targetZones = Object.fromEntries(SESSION_METRICS.filter((item) => item.zone).map((item) => [item.key, item.zone]));
const median = (values) => { const sorted = values.toSorted((a, b) => a - b); const middle = Math.floor(sorted.length / 2); return sorted.length % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2; };
const mean = (values) => values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : null;
const percentile = (values, p) => { const sorted = values.toSorted((a, b) => a - b); const at = (sorted.length - 1) * p; const low = Math.floor(at); const high = Math.ceil(at); return sorted[low] + (sorted[high] - sorted[low]) * (at - low); };
const uid = (prefix) => `${prefix}_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 8)}`;
const round = (value, digits = 1) => Number.isFinite(value) ? Number(value.toFixed(digits)) : null;
const distanceToZone = (value, zone) => !zone || !Number.isFinite(value) ? null : value < zone[0] ? zone[0] - value : value > zone[1] ? value - zone[1] : 0;
const confidenceComparable = (analysis) => ["高", "中"].includes(analysis?.capture?.confidence);

function issueMetric(issue) {
  if (issue?.metricKey) return issue.metricKey;
  if (issue?.issueCode?.startsWith("KNEE")) return "kneeRange";
  if (issue?.issueCode?.startsWith("ELBOW")) return "elbowRelease";
  if (issue?.issueCode?.startsWith("TRUNK")) return "trunkRelease";
  if (issue?.issueCode?.startsWith("WRIST")) return "wristPathStability";
  if (issue?.issueCode?.startsWith("RHYTHM")) return "pauseDuration";
  return null;
}

function directionMap(analysis) {
  return Object.fromEntries((analysis?.priorities || []).map((issue) => [issueMetric(issue), issue.direction]).filter(([key]) => key));
}

function comparisonStatus(before, after, definition, cueDirection) {
  const tolerance = definition?.tolerance ?? 0;
  const zone = definition?.zone;
  const delta = after - before;
  if (zone && ((cueDirection === "increase" && after > zone[1] + tolerance) || (cueDirection === "decrease" && after < zone[0] - tolerance))) return "overcorrected";
  if (zone && after >= zone[0] && after <= zone[1]) return before >= zone[0] && before <= zone[1] ? "stable" : "improved";
  if (Math.abs(delta) <= tolerance) return "stable";
  if (cueDirection === "increase") return delta > 0 ? "improved" : "worsened";
  if (["decrease", "decrease_variability"].includes(cueDirection)) return delta < 0 ? "improved" : "worsened";
  if (zone) {
    const beforeDistance = distanceToZone(before, zone); const afterDistance = distanceToZone(after, zone);
    if (afterDistance < beforeDistance - tolerance) return "improved";
    if (afterDistance > beforeDistance + tolerance) return "worsened";
  }
  return "stable";
}

export function createSession({ shootingHand, distanceCategory, now = new Date().toISOString() }) {
  if (!new Set(["near_basket", "free_throw", "mid_range", "three_point", "other"]).has(distanceCategory)) throw new Error("无效的投篮距离类别");
  return { schemaVersion: "TrainingSessionV1", sessionId: uid("session"), createdAt: now, updatedAt: now, status: "active", shootingHand, distanceCategory, shots: [], baseline: null, sessionAnalytics: null, sessionSummary: null };
}

export function compareShots(previous, current, { shootingHandSame = true, distanceSame = true, directions = null } = {}) {
  const scaleA = previous.capture?.bodyScale;
  const scaleB = current.capture?.bodyScale;
  const scaleChange = Number.isFinite(scaleA) && Number.isFinite(scaleB) && scaleA > 0 ? Math.abs(scaleB - scaleA) / scaleA : 0;
  const comparable = shootingHandSame && distanceSame && confidenceComparable(previous) && confidenceComparable(current) && scaleChange <= 0.25;
  const metrics = {};
  const previousDirections = { ...directionMap(previous), ...(directions || {}) };
  for (const key of new Set([...Object.keys(previous.metrics || {}), ...Object.keys(current.metrics || {})])) {
    const before = previous.metrics?.[key]; const after = current.metrics?.[key];
    const definition = metricDefinitions[key] || { tolerance: key.includes("wrist") ? 5 : 0, zone: targetZones[key] || null };
    let status = "not_comparable"; let direction = "unknown";
    if (comparable && Number.isFinite(before) && Number.isFinite(after)) {
      status = comparisonStatus(before, after, definition, previousDirections[key]);
      const delta = after - before;
      direction = delta > definition.tolerance ? "up" : delta < -definition.tolerance ? "down" : "flat";
    }
    metrics[key] = { previous: before ?? null, current: after ?? null, delta: Number.isFinite(before) && Number.isFinite(after) ? after - before : null, tolerance: definition.tolerance, direction, status };
  }
  return { comparable, reason: comparable ? null : "拍摄尺度、可信度、投篮手或距离类别不可比", metrics };
}

export function buildBaseline(shots) {
  const valid = shots.filter((shot) => confidenceComparable(shot.analysis));
  if (valid.length < 5) return null;
  const metrics = {};
  for (const key of Object.keys(valid[0].analysis.metrics || {})) {
    const values = valid.map((shot) => shot.analysis.metrics[key]).filter(Number.isFinite);
    if (values.length < 5) continue;
    const q1 = percentile(values, 0.25); const q3 = percentile(values, 0.75); const knowledgeZone = targetZones[key] || null;
    const intersection = knowledgeZone ? [Math.max(knowledgeZone[0], q1), Math.min(knowledgeZone[1], q3)] : null;
    const personalTarget = intersection && intersection[1] - intersection[0] >= (angleKeys.has(key) ? 2 : 0.05) ? intersection : knowledgeZone;
    metrics[key] = { median: median(values), q1, q3, iqr: q3 - q1, mean: mean(values), knowledgeZone, personalTarget };
  }
  return { sampleCount: valid.length, metrics };
}

function gradualTarget(current, definition, baselineTarget = null) {
  const zone = baselineTarget || definition.zone;
  if (!Number.isFinite(current) || !zone) return null;
  const tolerance = definition.tolerance; const decimals = definition.unit === "s" ? 2 : 1;
  if (current >= zone[0] && current <= zone[1]) return { current, target: [round(Math.max(zone[0], current - tolerance), decimals), round(Math.min(zone[1], current + tolerance), decimals)], action: "maintain" };
  if (current < zone[0]) return { current, target: [round(current + definition.step, decimals), round(Math.min(zone[0], current + definition.step * 2), decimals)], action: "increase" };
  return { current, target: [round(Math.max(zone[1], current - definition.step * 2), decimals), round(current - definition.step, decimals)], action: "decrease" };
}

function targetCue(key, target) {
  const range = `${target.target[0]}${metricDefinitions[key]?.unit || ""}–${target.target[1]}${metricDefinitions[key]?.unit || ""}`;
  if (target.action === "maintain") return `当前已进入本轮目标区间 ${range}，下一球保持，不要继续同方向加量。`;
  const verb = target.action === "increase" ? "增加" : "减少";
  const cueByKey = {
    kneeRange: `${verb}下沉相对幅度，下一球先尝试 ${range}；膝髋一起自然下沉，最低点不要停顿。`,
    elbowRelease: `${verb}自然伸肘幅度，下一球先尝试 ${range}；不要为了角度锁死手肘。`,
    trunkRelease: `下一球把出手躯干画面倾角向 ${range} 调整；身体主要向上，不随球明显前扑。`,
    wristPathStability: `下一球把手腕二维路线离散控制到 ${range}；随挥沿目标方向并停住1秒。`,
    pauseDuration: `下一球把最低点附近停顿控制到 ${range}；用“下—上”口令连续完成。`,
  };
  return cueByKey[key] || `下一球把${metricDefinitions[key]?.label || key}逐步调整到 ${range}。`;
}

export function derivePersonalTargets(shots, baseline = null) {
  const currentShot = shots.at(-1); const currentMetrics = currentShot?.analysis?.metrics || {};
  if (!currentShot || !confidenceComparable(currentShot.analysis)) return {};
  const targets = {};
  for (const key of ["kneeRange", "elbowRelease", "trunkRelease", "wristPathStability", "pauseDuration"]) {
    const definition = metricDefinitions[key]; const current = currentMetrics[key];
    const target = gradualTarget(current, definition, baseline?.metrics?.[key]?.personalTarget || null);
    if (!target) continue;
    targets[key] = { ...target, cue: targetCue(key, target), source: baseline?.metrics?.[key]?.personalTarget ? "personal-baseline" : "knowledge-and-session" };
  }
  if (targets.kneeRange && Number.isFinite(currentMetrics.kneeLowest)) {
    const [lowRange, highRange] = targets.kneeRange.target;
    const low = currentMetrics.kneeLowest - (highRange - currentMetrics.kneeRange); const high = currentMetrics.kneeLowest - (lowRange - currentMetrics.kneeRange);
    targets.kneeLowest = { current: currentMetrics.kneeLowest, target: [round(Math.min(low, high)), round(Math.max(low, high))], action: targets.kneeRange.action, cue: targets.kneeRange.cue, source: targets.kneeRange.source };
  }
  return targets;
}

function shotScore(shot) {
  const analysis = shot.analysis || {}; let score = confidenceScore[analysis.capture?.confidence] ?? -100;
  score += Math.min(10, (analysis.capture?.score || 0) / 10); score += Math.min(6, (analysis.strengths || []).length * 2);
  for (const issue of analysis.priorities || []) score -= severityPenalty[issue.severity] || 4;
  for (const definition of SESSION_METRICS.filter((item) => item.zone)) {
    const value = analysis.metrics?.[definition.key]; if (!Number.isFinite(value)) continue;
    const distance = distanceToZone(value, definition.zone); score += distance === 0 ? 6 : Math.max(0, 4 - distance / Math.max(definition.tolerance, 0.05));
  }
  return round(score, 2);
}

export function selectBestShot(shots) {
  if (!shots.length) return null;
  const ranked = shots.map((shot) => ({ shot, score: shotScore(shot) })).toSorted((a, b) => b.score - a.score || a.shot.shotNumber - b.shot.shotNumber);
  const winner = ranked[0]; const reasons = [];
  if (confidenceComparable(winner.shot.analysis)) reasons.push(`${winner.shot.analysis.capture.confidence}可信度`);
  if (!(winner.shot.analysis.priorities || []).length) reasons.push("没有高优先级新问题");
  if ((winner.shot.analysis.strengths || []).length) reasons.push("可保持动作较多");
  return { shotId: winner.shot.shotId, shotNumber: winner.shot.shotNumber, score: winner.score, reasons, explanation: `当前最佳球按可信度、问题数量、动作稳定性和本轮目标综合评估；${reasons.join("，") || "用于本轮相对参考"}，不代表绝对最佳投篮。` };
}

function averageAnalysis(shots, directionSource) {
  const valid = shots.filter((shot) => confidenceComparable(shot.analysis)); if (!valid.length) return null;
  const metrics = {};
  for (const key of new Set(valid.flatMap((shot) => Object.keys(shot.analysis.metrics || {})))) {
    const values = valid.map((shot) => shot.analysis.metrics[key]).filter(Number.isFinite); if (values.length) metrics[key] = mean(values);
  }
  return { metrics, capture: { confidence: "中", bodyScale: mean(valid.map((shot) => shot.analysis.capture.bodyScale).filter(Number.isFinite)) }, priorities: directionSource?.priorities || [] };
}

function classifyTrend(shots, key, target) {
  const definition = metricDefinitions[key] || { tolerance: 0, zone: null };
  const values = shots.filter((shot) => confidenceComparable(shot.analysis)).map((shot) => shot.analysis.metrics?.[key]).filter(Number.isFinite);
  if (values.length < 3) return "UNCERTAIN";
  const range = Math.max(...values) - Math.min(...values); if (range <= definition.tolerance * 2) return "STABLE";
  const zone = target?.target || definition.zone;
  if (zone) {
    const distances = values.map((value) => distanceToZone(value, zone)); const progress = distances[0] - distances.at(-1);
    const reversals = distances.slice(2).reduce((count, value, index) => count + (Math.sign(value - distances[index + 1]) !== Math.sign(distances[index + 1] - distances[index]) ? 1 : 0), 0);
    if (reversals >= 2 && range > definition.tolerance * 4) return "INCONSISTENT";
    if (progress > definition.tolerance) return "IMPROVING"; if (progress < -definition.tolerance) return "WORSENING";
  }
  return range > definition.tolerance * 4 ? "INCONSISTENT" : "STABLE";
}

function comparisonLayer(referenceAnalysis, currentAnalysis, referenceType, referenceShotNumber, directions) {
  if (!referenceAnalysis) return { referenceType, referenceShotNumber, comparable: false, reason: "暂无足够历史样本", metrics: {} };
  return { referenceType, referenceShotNumber, ...compareShots(referenceAnalysis, currentAnalysis, { directions }) };
}

function formatValue(key, value) {
  if (!Number.isFinite(value)) return "—";
  const unit = metricDefinitions[key]?.unit || ""; const digits = unit === "s" ? 2 : 1;
  return `${value.toFixed(digits)}${unit}`;
}

function adjustmentRows(current, targets) {
  const analysis = current.analysis; const priorities = analysis.priorities || [];
  const priorityLabel = (keys) => { const index = priorities.findIndex((issue) => keys.includes(issueMetric(issue))); return index === 0 ? "Priority 1" : index === 1 ? "Priority 2" : "Observe"; };
  const rows = [
    { part: "膝关节", keys: ["kneeRange", "kneeLowest"], key: "kneeLowest", performance: targets.kneeRange?.action === "maintain" ? "当前幅度可保持" : targets.kneeRange?.action === "increase" ? "下沉相对偏浅" : "下沉相对偏深", action: targets.kneeRange?.action || "observe", cue: targets.kneeLowest?.cue || "膝髋一起自然下沉，重心保持在脚掌中部附近。", hint: "膝髋一起下沉，不只向前顶膝；最低点后连续起身。" },
    { part: "髋关节", keys: ["hipLowest", "hipKneeCoordination"], key: "hipLowest", performance: "结合髋膝协同观察", action: "observe", cue: "臀部轻微向后下方移动，同时膝盖自然弯曲。", hint: "不要为了降低身体而单纯弯腰。" },
    { part: "肘关节", keys: ["elbowRelease", "elbowRange"], key: "elbowRelease", performance: targets.elbowRelease?.action === "maintain" ? "伸展区间可保持" : "继续观察自然伸展", action: targets.elbowRelease?.action || "observe", cue: targets.elbowRelease?.cue || "抬球后自然向前上方伸展，不追求180°锁肘。", hint: "自然伸展并定住随挥1秒。" },
    { part: "躯干", keys: ["trunkRelease", "trunkMax"], key: "trunkRelease", performance: targets.trunkRelease?.action === "maintain" ? "轴线趋势可保持" : "倾斜趋势需调整", action: targets.trunkRelease?.action || "observe", cue: targets.trunkRelease?.cue || "起身时身体主要向上，不随球明显前扑。", hint: "胸口保持稳定，落点尽量接近起跳点。" },
    { part: "手腕轨迹", keys: ["wristPathStability"], key: "wristPathStability", performance: targets.wristPathStability?.action === "maintain" ? "二维路线可保持" : "路线离散需观察", action: targets.wristPathStability?.action || "observe", cue: targets.wristPathStability?.cue || "沿目标方向自然随挥，不推断手指细节。", hint: "随挥方向保持一致并停住1秒。" },
    { part: "动作节奏", keys: ["pauseDuration", "rhythmDuration"], key: "pauseDuration", performance: targets.pauseDuration?.action === "maintain" ? "衔接相对连续" : "可能存在分段", action: targets.pauseDuration?.action || "observe", cue: targets.pauseDuration?.cue || "下肢开始向上时，让抬球动作自然同步发生。", hint: "从二维时间序列判断顺序，不推断具体肌肉力量。" },
  ];
  return rows.map((row) => {
    const target = targets[row.key] || targets[row.keys[0]];
    return { ...row, priority: priorityLabel(row.keys), current: formatValue(row.key, analysis.metrics?.[row.key]), target: target ? `${formatValue(row.key, target.target[0])}–${formatValue(row.key, target.target[1])}` : "继续观察同机位趋势", direction: row.action === "maintain" ? "保持" : row.action === "increase" ? "逐步增加" : row.action === "decrease" ? "逐步减少" : "观察" };
  });
}

export function buildShotRecord(analysis) {
  return {
    metrics: { lowest_knee_angle: analysis.metrics?.kneeLowest ?? null, lowest_hip_angle: analysis.metrics?.hipLowest ?? null, release_elbow_angle: analysis.metrics?.elbowRelease ?? null, release_trunk_angle: analysis.metrics?.trunkRelease ?? null, elbow_extension_range: analysis.metrics?.elbowRange ?? null, wrist_path_metric: analysis.metrics?.wristPathStability ?? null, dip_duration: analysis.metrics?.rhythmDuration ?? null, extension_duration: analysis.metrics?.kneeRiseDuration ?? null, release_timing: analysis.events?.release?.time ?? null, pose_confidence: analysis.capture?.confidence ?? "低" },
    strengths: analysis.strengths || [], primaryIssue: analysis.priorities?.[0] || null, secondaryIssue: analysis.priorities?.[1] || null,
    observations: (analysis.sections || []).map((section) => section.evaluation).filter(Boolean), nextRepCues: analysis.nextRep || [], trainingRecommendations: analysis.prescriptions || [],
  };
}

export function buildSessionAnalytics(session, currentShotNumber = session.shots.length) {
  const currentIndex = session.shots.findIndex((shot) => shot.shotNumber === currentShotNumber); if (currentIndex < 0) return null;
  const current = session.shots[currentIndex]; const prior = session.shots.slice(0, currentIndex); const visibleShots = session.shots.slice(0, currentIndex + 1);
  const directions = directionMap(session.shots[0]?.analysis); const bestShot = selectBestShot(visibleShots); const bestReference = visibleShots.find((shot) => shot.shotId === bestShot?.shotId);
  const comparisons = {
    previous: comparisonLayer(prior.at(-1)?.analysis, current.analysis, "previous", prior.at(-1)?.shotNumber ?? null, directions),
    first: comparisonLayer(prior[0]?.analysis, current.analysis, "first", prior[0]?.shotNumber ?? null, directions),
    sessionAverage: comparisonLayer(averageAnalysis(prior, session.shots[0]?.analysis), current.analysis, "sessionAverage", null, directions),
    best: comparisonLayer(bestReference?.analysis, current.analysis, "best", bestReference?.shotNumber ?? null, directions),
  };
  const targets = derivePersonalTargets(visibleShots, buildBaseline(visibleShots));
  const historyRows = SESSION_METRICS.filter((definition) => visibleShots.some((shot) => Number.isFinite(shot.analysis.metrics?.[definition.key]))).map((definition) => ({ ...definition, values: visibleShots.map((shot) => ({ shotNumber: shot.shotNumber, value: shot.analysis.metrics?.[definition.key] ?? null, confidence: shot.analysis.capture?.confidence || "低" })), trend: classifyTrend(visibleShots, definition.key, targets[definition.key]) }));
  const improvements = [];
  for (const layer of [comparisons.first, comparisons.previous]) {
    if (!layer.comparable) continue;
    for (const [key, item] of Object.entries(layer.metrics)) {
      if (item.status !== "improved" || improvements.some((existing) => existing.metricKey === key)) continue;
      improvements.push({ metricKey: key, label: metricDefinitions[key]?.label || key, referenceShotNumber: layer.referenceShotNumber, previous: item.previous, current: item.current, delta: item.delta, text: `与第 ${layer.referenceShotNumber} 球相比，${metricDefinitions[key]?.label || key}从 ${formatValue(key, item.previous)} 调整到 ${formatValue(key, item.current)}。` });
    }
  }
  const previousCodes = new Set(prior.at(-1)?.analysis?.priorities?.map((issue) => issue.issueCode) || []);
  const newProblems = (current.analysis.priorities || []).filter((issue) => !previousCodes.has(issue.issueCode)).map((issue) => ({ issueCode: issue.issueCode, title: issue.title || issue.issueCode, text: prior.length ? `第 ${current.shotNumber} 球新出现：${issue.title || issue.issueCode}。下一球保留已经改善的动作，只处理这项新问题。` : `第 ${current.shotNumber} 球当前问题：${issue.title || issue.issueCode}。` }));
  const stableMetricLabels = comparisons.previous.comparable ? Object.entries(comparisons.previous.metrics).filter(([, item]) => item.status === "stable").map(([key]) => metricDefinitions[key]?.label || key).slice(0, 2) : [];
  const maintain = [...new Set([...(current.analysis.keep || []), ...(current.analysis.strengths || []), ...stableMetricLabels.map((label) => `${label}与上一球基本稳定`)])].slice(0, 4);
  return { currentShotNumber, comparisons, historyRows, bestShot, improvements: improvements.slice(0, 3), newProblems: newProblems.slice(0, 2), maintain, targets, adjustmentRows: adjustmentRows(current, targets), trainingRows: (current.analysis.prescriptions || []).map((item) => ({ name: item.name, issue: item.forIssue, method: item.steps, volume: `${item.sets || ""} ${item.reps || ""}`.trim(), focus: item.focus })) };
}

export function appendShot(session, shot) {
  if (shot.shootingHand && shot.shootingHand !== session.shootingHand) throw new Error("同一训练不能更换投篮手");
  if (shot.distanceCategory && shot.distanceCategory !== session.distanceCategory) throw new Error("同一训练不能更换距离类别");
  const previous = session.shots.at(-1); const comparison = previous ? compareShots(previous.analysis, shot.analysis, { directions: directionMap(session.shots[0]?.analysis || previous.analysis) }) : null;
  let cues = shot.analysis.nextRep || [];
  if (comparison?.comparable && previous?.analysis?.priorities?.length) cues = previous.analysis.priorities.slice(0, 2).map((issue) => {
    const key = issueMetric(issue); const status = comparison.metrics[key]?.status;
    if (status === "improved" || status === "stable") return `保持当前${key === "kneeRange" ? "下沉" : "动作"}幅度，再验证一球，不继续同方向加量。`;
    if (status === "overcorrected") return "这次纠正越过了训练参考区间，下一球回调一半，不继续同方向加量。";
    return issue.cue || issue.nextBall;
  }).filter(Boolean);
  const coachedAnalysis = { ...shot.analysis, nextRep: cues, next_rep: cues, nextShot: cues };
  const newShot = { schemaVersion: "ShotV1", shotId: shot.shotId || uid("shot"), shotNumber: session.shots.length + 1, analyzedAt: shot.analyzedAt || new Date().toISOString(), fileMeta: shot.fileMeta || {}, thumbnail: shot.thumbnail || null, analysis: coachedAnalysis, record: buildShotRecord(coachedAnalysis), comparison, cues };
  let next = { ...session, shots: [...session.shots, newShot], updatedAt: new Date().toISOString() };
  next.baseline = buildBaseline(next.shots); next.sessionAnalytics = buildSessionAnalytics(next);
  const { historyRows, adjustmentRows: currentAdjustments, trainingRows: currentTraining, ...historySummary } = next.sessionAnalytics;
  next.shots = next.shots.map((item, index) => index === next.shots.length - 1 ? { ...item, history: historySummary, adjustmentRows: currentAdjustments, trainingRows: currentTraining } : item);
  return next;
}

export function buildSessionSummary(session) {
  if (session.shots.length < 2) return { status: "insufficient", message: "至少完成2球后才能结束训练。" };
  const issueCounts = new Map(); for (const shot of session.shots) for (const issue of shot.analysis.priorities || []) issueCounts.set(issue.issueCode, (issueCounts.get(issue.issueCode) || 0) + 1);
  const recurringIssues = [...issueCounts.entries()].toSorted((a, b) => b[1] - a[1] || a[0].localeCompare(b[0])).slice(0, 2).map(([issueCode, count]) => ({ issueCode, count }));
  const comparisons = session.shots.map((shot) => shot.comparison).filter((item) => item?.comparable); const improvementCounts = new Map(); const instability = new Map();
  for (const comparison of comparisons) for (const [key, item] of Object.entries(comparison.metrics)) { if (item.status === "improved") improvementCounts.set(key, (improvementCounts.get(key) || 0) + 1); if (["worsened", "overcorrected"].includes(item.status)) instability.set(key, (instability.get(key) || 0) + 1); }
  const top = (map) => [...map.entries()].toSorted((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]))[0]?.[0] || null;
  const effectiveShot = [...session.shots].reverse().find((shot) => shot.comparison && Object.values(shot.comparison.metrics).some((item) => item.status === "improved"));
  const baseline = buildBaseline(session.shots); const analytics = buildSessionAnalytics(session); const sessionIssues = [];
  if (baseline?.metrics?.kneeRange?.iqr > 9) sessionIssues.push({ issueCode: "KNEE_DIP_INCONSISTENT", severity: baseline.metrics.kneeRange.iqr > 17 ? "明显" : baseline.metrics.kneeRange.iqr > 12 ? "中等" : "轻微" });
  const firstToLast = compareShots(session.shots[0].analysis, session.shots.at(-1).analysis, { directions: directionMap(session.shots[0].analysis) });
  const fromFirstToLast = Object.entries(firstToLast.metrics).filter(([key, item]) => metricDefinitions[key] && Number.isFinite(item.delta)).map(([key, item]) => ({ metricKey: key, label: metricDefinitions[key].label, first: item.previous, last: item.current, delta: item.delta, status: item.status, unit: metricDefinitions[key].unit }));
  const nextFocuses = [...new Set([...(analytics?.newProblems || []).map((item) => item.issueCode), ...sessionIssues.map((item) => item.issueCode), ...recurringIssues.map((item) => item.issueCode)])].slice(0, 2);
  return { status: session.shots.length >= 5 ? "formal" : "provisional", shotCount: session.shots.length, recurringIssues, sessionIssues, baseline, bestShot: analytics?.bestShot || null, fromFirstToLast, mostImprovedMetric: top(improvementCounts), stillUnstableMetric: top(instability), mostEffectiveCue: effectiveShot?.cues?.[0] || null, mostStablePattern: baseline?.metrics || null, nextFocus: nextFocuses[0] || null, nextFocuses };
}
