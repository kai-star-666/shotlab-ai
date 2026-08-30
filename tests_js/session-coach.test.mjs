import test from "node:test";
import assert from "node:assert/strict";

import { compareShots, createSession, appendShot, buildBaseline, buildSessionSummary } from "../netlify-site/assets/session/session-coach.mjs";
import { prepareSessionForPersistence } from "../netlify-site/assets/session/session-store.mjs";

const analysis = (kneeRange, confidence = "高", bodyScale = 0.5) => ({
  metrics: { kneeRange, elbowRelease: 150, rhythmDuration: 0.8, wristPathStability: 10 },
  capture: { confidence, bodyScale },
  priorities: kneeRange < 22 ? [{ issueCode: "KNEE_DIP_SHALLOW", direction: "increase", cue: "下沉稍深" }] : [],
  nextRep: kneeRange < 22 ? ["下沉稍深"] : ["保持当前幅度"],
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
