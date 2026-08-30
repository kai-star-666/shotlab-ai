const angleKeys = new Set(["kneeRange", "elbowRelease", "kneeLowest", "hipLowest", "trunkRelease"]);
const rhythmKeys = new Set(["rhythmDuration", "pauseDuration"]);
const targetZones = { kneeRange: [22, 62], elbowRelease: [145, 175], wristPathStability: [0, 18], rhythmDuration: [0, 1.2] };
const median = (values) => { const sorted = values.toSorted((a, b) => a - b); const middle = Math.floor(sorted.length / 2); return sorted.length % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2; };
const percentile = (values, p) => { const sorted = values.toSorted((a, b) => a - b); const at = (sorted.length - 1) * p; const low = Math.floor(at); const high = Math.ceil(at); return sorted[low] + (sorted[high] - sorted[low]) * (at - low); };
const uid = (prefix) => `${prefix}_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 8)}`;

export function createSession({ shootingHand, distanceCategory, now = new Date().toISOString() }) {
  if (!new Set(["near_basket", "free_throw", "mid_range", "three_point", "other"]).has(distanceCategory)) throw new Error("无效的投篮距离类别");
  return { schemaVersion: "TrainingSessionV1", sessionId: uid("session"), createdAt: now, updatedAt: now, status: "active", shootingHand, distanceCategory, shots: [], baseline: null, sessionSummary: null };
}

export function compareShots(previous, current, { shootingHandSame = true, distanceSame = true } = {}) {
  const scaleA = previous.capture?.bodyScale;
  const scaleB = current.capture?.bodyScale;
  const scaleChange = Number.isFinite(scaleA) && Number.isFinite(scaleB) && scaleA > 0 ? Math.abs(scaleB - scaleA) / scaleA : 0;
  const comparable = shootingHandSame && distanceSame && previous.capture?.confidence !== "低" && current.capture?.confidence !== "低" && scaleChange <= 0.25;
  const metrics = {};
  const previousDirections = new Map((previous.priorities || []).map((issue) => [issue.metricKey || (issue.issueCode?.startsWith("KNEE") ? "kneeRange" : ""), issue.direction]));
  for (const key of new Set([...Object.keys(previous.metrics || {}), ...Object.keys(current.metrics || {})])) {
    const before = previous.metrics?.[key]; const after = current.metrics?.[key];
    const tolerance = angleKeys.has(key) ? 2 : rhythmKeys.has(key) ? 0.05 : key.includes("wrist") ? 5 : 0;
    let status = "not_comparable"; let direction = "unknown";
    if (comparable && Number.isFinite(before) && Number.isFinite(after)) {
      const delta = after - before; const zone = targetZones[key]; const cueDirection = previousDirections.get(key);
      if (zone && ((cueDirection === "increase" && after > zone[1] + tolerance) || (cueDirection === "decrease" && after < zone[0] - tolerance))) status = "overcorrected";
      else if (zone && after >= zone[0] && after <= zone[1]) status = before >= zone[0] && before <= zone[1] ? "stable" : "improved";
      else if (Math.abs(delta) <= tolerance) status = "stable";
      else if (cueDirection === "increase") status = delta > 0 ? "improved" : "worsened";
      else if (cueDirection === "decrease") status = delta < 0 ? "improved" : "worsened";
      else status = "stable";
      direction = delta > tolerance ? "up" : delta < -tolerance ? "down" : "flat";
    }
    metrics[key] = { previous: before ?? null, current: after ?? null, delta: Number.isFinite(before) && Number.isFinite(after) ? after - before : null, tolerance, direction, status };
  }
  return { comparable, reason: comparable ? null : "拍摄尺度、可信度、投篮手或距离类别不可比", metrics };
}

export function buildBaseline(shots) {
  const valid = shots.filter((shot) => ["高", "中"].includes(shot.analysis?.capture?.confidence));
  if (valid.length < 5) return null;
  const metrics = {};
  for (const key of Object.keys(valid[0].analysis.metrics || {})) {
    const values = valid.map((shot) => shot.analysis.metrics[key]).filter(Number.isFinite);
    if (values.length < 5) continue;
    const q1 = percentile(values, 0.25); const q3 = percentile(values, 0.75); const knowledgeZone = targetZones[key] || null;
    const intersection = knowledgeZone ? [Math.max(knowledgeZone[0], q1), Math.min(knowledgeZone[1], q3)] : null;
    const personalTarget = intersection && intersection[1] - intersection[0] >= (angleKeys.has(key) ? 2 : 0.05) ? intersection : knowledgeZone;
    metrics[key] = { median: median(values), q1, q3, iqr: q3 - q1, mean: values.reduce((a, b) => a + b, 0) / values.length, knowledgeZone, personalTarget };
  }
  return { sampleCount: valid.length, metrics };
}

export function appendShot(session, shot) {
  if (shot.shootingHand && shot.shootingHand !== session.shootingHand) throw new Error("同一训练不能更换投篮手");
  if (shot.distanceCategory && shot.distanceCategory !== session.distanceCategory) throw new Error("同一训练不能更换距离类别");
  const previous = session.shots.at(-1);
  const comparison = previous ? compareShots(previous.analysis, shot.analysis) : null;
  let cues = shot.analysis.nextRep || [];
  if (comparison?.comparable && previous) {
    cues = (previous.analysis.priorities || []).slice(0, 2).map((issue) => {
      const key = issue.metricKey || (issue.issueCode?.startsWith("KNEE") ? "kneeRange" : null);
      const status = comparison.metrics[key]?.status;
      if (status === "improved" || status === "stable") return `保持当前${key === "kneeRange" ? "下沉" : "动作"}幅度，再验证一球。`;
      if (status === "overcorrected") return "这次纠正越过了训练参考区间，下一球回调一半，不继续同方向加量。";
      return issue.cue || issue.nextBall;
    }).filter(Boolean);
  }
  const coachedAnalysis = { ...shot.analysis, nextRep: cues, next_rep: cues, nextShot: cues };
  const next = { ...session, shots: [...session.shots, { schemaVersion: "ShotV1", shotId: shot.shotId || uid("shot"), shotNumber: session.shots.length + 1, analyzedAt: shot.analyzedAt || new Date().toISOString(), fileMeta: shot.fileMeta || {}, thumbnail: shot.thumbnail || null, analysis: coachedAnalysis, comparison, cues }], updatedAt: new Date().toISOString() };
  next.baseline = buildBaseline(next.shots);
  return next;
}

export function buildSessionSummary(session) {
  if (session.shots.length < 2) return { status: "insufficient", message: "至少完成2球后才能结束训练。" };
  const issueCounts = new Map();
  for (const shot of session.shots) for (const issue of shot.analysis.priorities || []) issueCounts.set(issue.issueCode, (issueCounts.get(issue.issueCode) || 0) + 1);
  const recurringIssues = [...issueCounts.entries()].toSorted((a, b) => b[1] - a[1]).slice(0, 2).map(([issueCode, count]) => ({ issueCode, count }));
  const comparisons = session.shots.map((shot) => shot.comparison).filter((item) => item?.comparable);
  const improvementCounts = new Map(); const instability = new Map();
  for (const comparison of comparisons) for (const [key, item] of Object.entries(comparison.metrics)) {
    if (item.status === "improved") improvementCounts.set(key, (improvementCounts.get(key) || 0) + 1);
    if (["worsened", "overcorrected"].includes(item.status)) instability.set(key, (instability.get(key) || 0) + 1);
  }
  const top = (map) => [...map.entries()].toSorted((a, b) => b[1] - a[1])[0]?.[0] || null;
  const effectiveShot = [...session.shots].reverse().find((shot) => shot.comparison && Object.values(shot.comparison.metrics).some((item) => item.status === "improved"));
  const baseline = buildBaseline(session.shots);
  const sessionIssues = [];
  if (baseline?.metrics?.kneeRange?.iqr > 9) sessionIssues.push({ issueCode: "KNEE_DIP_INCONSISTENT", severity: baseline.metrics.kneeRange.iqr > 17 ? "明显" : baseline.metrics.kneeRange.iqr > 12 ? "中等" : "轻微" });
  return { status: session.shots.length >= 5 ? "formal" : "provisional", shotCount: session.shots.length, recurringIssues, sessionIssues, baseline, mostImprovedMetric: top(improvementCounts), stillUnstableMetric: top(instability), mostEffectiveCue: effectiveShot?.cues?.[0] || null, mostStablePattern: baseline?.metrics || null, nextFocus: sessionIssues[0]?.issueCode || recurringIssues[0]?.issueCode || null };
}
