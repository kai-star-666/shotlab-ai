import test from "node:test";
import assert from "node:assert/strict";

import {
  compareShots,
  createSession,
  appendShot,
  buildBaseline,
  buildSessionSummary,
  buildSessionAnalytics,
  derivePersonalTargets,
  selectBestShot,
} from "../netlify-site/assets/session/session-coach.mjs";
import { prepareSessionForPersistence } from "../netlify-site/assets/session/session-store.mjs";

const analysis = (kneeRange, confidence = "高", bodyScale = 0.5, overrides = {}) => ({
  metrics: { kneeRange, kneeLowest: 150 - kneeRange, hipLowest: 135, elbowRelease: 150, elbowRange: 40, trunkRelease: 10, trunkMax: 12, rhythmDuration: 0.8, pauseDuration: 0.08, wristPathStability: 10, ...overrides.metrics },
  capture: { confidence, bodyScale },
  priorities: overrides.priorities ?? (kneeRange < 22 ? [{ issueCode: "KNEE_DIP_SHALLOW", metricKey: "kneeRange", severity: "中等", confidence, direction: "increase", cue: "下沉稍深", title: "下沉偏浅" }] : []),
  nextRep: kneeRange < 22 ? ["下沉稍深"] : ["保持当前幅度"],
  strengths: overrides.strengths ?? ["躯干轴线稳定"],
  keep: overrides.keep ?? ["保持躯干控制"],
  prescriptions: overrides.prescriptions ?? [{ name: "固定下沉练习", forIssue: "下沉偏浅", steps: "无球完成下沉与起身", sets: "3组", reps: "每组10次", focus: "每次深度接近" }],
  sections: overrides.sections ?? [],
});

test("second shot comparison detects improvement and overcorrection", () => {
  const improved = compareShots(analysis(12), analysis(24), { shootingHandSame: true, distanceSame: true });
  assert.equal(improved.metrics.kneeRange.status, "improved");
  const over = compareShots(analysis(12), analysis(70), { shootingHandSame: true, distanceSame: true });
  assert.equal(over.metrics.kneeRange.status, "overcorrected");
});

test("comparison refuses different camera scale or low confidence", () => {
  const comparison = compareShots(analysis(12, "高", 0.5), analysis(24, "高", 0.7), { shootingHandSame: true, distanceSame: true });
  assert.equal(comparison.comparable, false);
  assert.equal(comparison.metrics.kneeRange.status, "not_comparable");
});

test("session locks hand and distance and builds baseline after five valid shots", () => {
  let session = createSession({ shootingHand: "right", distanceCategory: "near_basket", now: "2026-08-30T00:00:00Z" });
  for (const value of [20, 22, 24, 23, 21]) session = appendShot(session, { analysis: analysis(value), fileMeta: {} });
  assert.throws(() => appendShot(session, { analysis: analysis(24), shootingHand: "left" }));
  const baseline = buildBaseline(session.shots);
  assert.equal(baseline.sampleCount, 5);
  assert.equal(baseline.metrics.kneeRange.median, 22);
  assert.equal(buildSessionSummary(session).status, "formal");
});

test("persistence removes video-derived landmarks and motion sequences", () => {
  const session = createSession({ shootingHand: "right", distanceCategory: "near_basket" });
  session.shots.push({ analysis: { metrics: { kneeRange: 20 }, analysisFrames: [{ landmarks: [{ x: 1 }] }], curves: [{ knee: 20 }], release: { landmarks: [{ x: 1 }] } } });
  const saved = prepareSessionForPersistence(session);
  assert.equal(saved.shots[0].analysis.analysisFrames, undefined);
  assert.equal(saved.shots[0].analysis.release, undefined);
  assert.deepEqual(saved.shots[0].analysis.metrics, { kneeRange: 20 });
});

test("five-shot session retains every shot and builds previous, first, average and best comparisons", () => {
  let session = createSession({ shootingHand: "right", distanceCategory: "near_basket", now: "2026-08-30T00:00:00Z" });
  for (const value of [12, 18, 24, 25, 26]) session = appendShot(session, { analysis: analysis(value), fileMeta: { name: `shot-${value}.mp4` } });
  assert.deepEqual(session.shots.map((shot) => shot.shotNumber), [1, 2, 3, 4, 5]);
  assert.equal(session.shots[0].fileMeta.name, "shot-12.mp4");
  assert.equal(session.sessionAnalytics.currentShotNumber, 5);
  assert.deepEqual(Object.keys(session.sessionAnalytics.comparisons), ["previous", "first", "sessionAverage", "best"]);
  assert.equal(session.sessionAnalytics.comparisons.previous.referenceShotNumber, 4);
  assert.equal(session.sessionAnalytics.comparisons.first.referenceShotNumber, 1);
  assert.equal(session.sessionAnalytics.historyRows.find((row) => row.key === "kneeRange").values.length, 5);
});

test("best shot selection rewards confidence, fewer issues and balanced target-zone metrics with deterministic ties", () => {
  const shots = [
    { shotId: "one", shotNumber: 1, analysis: analysis(24, "中") },
    { shotId: "two", shotNumber: 2, analysis: analysis(24, "高", 0.5, { metrics: { elbowRelease: 160, trunkMax: 8, wristPathStability: 8 }, priorities: [], strengths: ["肘部稳定", "躯干稳定"] }) },
    { shotId: "three", shotNumber: 3, analysis: analysis(24, "高", 0.5, { metrics: { elbowRelease: 160, trunkMax: 8, wristPathStability: 8 }, priorities: [], strengths: ["肘部稳定", "躯干稳定"] }) },
  ];
  const best = selectBestShot(shots);
  assert.equal(best.shotNumber, 2);
  assert.match(best.explanation, /综合评估|稳定/);
});

test("dynamic targets give a gradual adjustment then switch to maintain inside the target zone", () => {
  const first = { shotNumber: 1, analysis: analysis(12) };
  const firstTarget = derivePersonalTargets([first], null).kneeRange;
  assert.equal(firstTarget.action, "increase");
  assert.deepEqual(firstTarget.target, [17, 22]);
  const reached = { shotNumber: 2, analysis: analysis(24) };
  const reachedTarget = derivePersonalTargets([first, reached], null).kneeRange;
  assert.equal(reachedTarget.action, "maintain");
  assert.match(reachedTarget.cue, /保持|不要继续/);
});

test("session analytics identifies improvements, new issues, stable items and explicit shot references", () => {
  let session = createSession({ shootingHand: "right", distanceCategory: "near_basket" });
  session = appendShot(session, { analysis: analysis(12) });
  session = appendShot(session, { analysis: analysis(24, "高", 0.5, { metrics: { elbowRelease: 136 }, priorities: [{ issueCode: "ELBOW_EXTENSION_LIMITED", metricKey: "elbowRelease", severity: "中等", confidence: "高", direction: "increase", cue: "自然伸肘", title: "伸肘不足" }] }) });
  const analytics = buildSessionAnalytics(session);
  assert.ok(analytics.improvements.some((item) => item.referenceShotNumber === 1 && item.metricKey === "kneeRange"));
  assert.ok(analytics.newProblems.some((item) => item.issueCode === "ELBOW_EXTENSION_LIMITED"));
  assert.ok(analytics.maintain.length >= 1);
  assert.ok(analytics.adjustmentRows.length >= 5);
  assert.match(analytics.improvements[0].text, /第 1 球/);
});

test("formal summary reports first-to-last changes, best shot and at most two next focuses", () => {
  let session = createSession({ shootingHand: "right", distanceCategory: "near_basket" });
  for (const value of [12, 18, 24, 25, 24]) session = appendShot(session, { analysis: analysis(value) });
  const summary = buildSessionSummary(session);
  assert.equal(summary.status, "formal");
  assert.equal(summary.shotCount, 5);
  assert.ok(summary.bestShot?.shotNumber);
  assert.ok(summary.fromFirstToLast.some((item) => item.metricKey === "kneeRange"));
  assert.ok(summary.nextFocuses.length <= 2);
});

test("a priority-free previous shot does not erase the current maintain cue", () => {
  let session = createSession({ shootingHand: "right", distanceCategory: "near_basket" });
  session = appendShot(session, { analysis: analysis(24) });
  session = appendShot(session, { analysis: analysis(24) });
  assert.deepEqual(session.shots[1].cues, ["保持当前幅度"]);
  assert.deepEqual(session.shots[1].analysis.nextRep, ["保持当前幅度"]);
});
