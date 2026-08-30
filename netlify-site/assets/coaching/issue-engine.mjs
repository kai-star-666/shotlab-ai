const severityWeight = { 明显: 3, 中等: 2, 轻微: 1 };
const confidenceWeight = { 高: 1, 中: 0.65, 低: 0 };
const confidenceRank = { 低: 0, 中: 1, 高: 2 };
const get = (object, path) => path.split(".").reduce((value, key) => value?.[key], object);

function severityFor(value, bands = {}) {
  for (const [label, range] of [["明显", bands.obvious], ["中等", bands.medium], ["轻微", bands.minor]]) {
    if (range && value >= range[0] && value <= range[1]) return label;
  }
  return null;
}

export function evaluateIssues(input, knowledge) {
  const rules = knowledge.rules || knowledge;
  const issues = [];
  for (const rule of rules) {
    const value = rule.metricKey === "capture.score" ? get(input, rule.metricKey) : (input.metrics?.[rule.metricKey] ?? get(input, rule.metricKey));
    if (!Number.isFinite(value)) continue;
    const confidence = rule.issueCode === "CAPTURE_QUALITY_LOW" ? (input.capture?.confidence || "低") : (input.confidences?.[rule.metricKey] || input.capture?.confidence || "低");
    if (rule.issueCode === "CAPTURE_QUALITY_LOW" && confidence !== "低") continue;
    if (confidenceRank[confidence] < confidenceRank[rule.minimumConfidence]) continue;
    const severity = severityFor(value, rule.severityBands);
    if (!severity) continue;
    const issue = { ...rule, value, confidence, severity };
    issue.rankScore = severityWeight[severity] * confidenceWeight[confidence] * rule.coachingPriority;
    issues.push(issue);
  }
  issues.sort((a, b) => b.rankScore - a.rankScore || a.knowledgeOrder - b.knowledgeOrder || a.issueCode.localeCompare(b.issueCode));
  const priorities = issues.filter((issue) => issue.issueCode === "CAPTURE_QUALITY_LOW" || issue.confidence !== "低").slice(0, 2);
  const nextRep = priorities.map((issue) => issue.cue);
  if (!nextRep.length) nextRep.push("保持当前幅度，再投一球验证动作能否重复。");
  return { issues, priorities, nextRep: nextRep.slice(0, 2) };
}

export async function loadIssueKnowledge(url = "/knowledge/rules.v1.json") {
  const response = await fetch(url, { cache: "no-cache" });
  if (!response.ok) throw new Error(`训练规则加载失败（${response.status}）`);
  return response.json();
}
