import test from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";

import { evaluateIssues } from "../netlify-site/assets/coaching/issue-engine.mjs";

const rules = JSON.parse(await readFile(new URL("../netlify-site/knowledge/rules.v1.json", import.meta.url)));

test("knowledge base contains every V3 stable issue code", () => {
  const codes = new Set(rules.rules.map((rule) => rule.issueCode));
  for (const code of ["KNEE_DIP_SHALLOW", "KNEE_DIP_DEEP", "KNEE_DIP_INCONSISTENT", "TRUNK_FORWARD_LEAN", "TRUNK_UNSTABLE", "ELBOW_EXTENSION_LIMITED", "ELBOW_PATH_UNSTABLE", "WRIST_PATH_UNSTABLE", "RHYTHM_DISCONNECTED", "CAPTURE_QUALITY_LOW"]) assert.ok(codes.has(code), code);
});

test("issue engine uses stable weighted ordering and codes", () => {
  const result = evaluateIssues({ metrics: { kneeRange: 10, trunkMax: 25 }, confidences: { kneeRange: "高", trunkMax: "高" }, capture: { confidence: "高", score: 90 } }, rules);
  assert.equal(result.priorities.length, 2);
  assert.equal(result.priorities[0].issueCode, "TRUNK_FORWARD_LEAN");
  assert.equal(result.priorities[1].issueCode, "KNEE_DIP_SHALLOW");
});

test("low-confidence action issues are suppressed but capture quality can lead", () => {
  const result = evaluateIssues({ metrics: { kneeRange: 8, trunkMax: 30 }, confidences: { kneeRange: "低", trunkMax: "低" }, capture: { confidence: "低", score: 34 } }, rules);
  assert.deepEqual(result.priorities.map((issue) => issue.issueCode), ["CAPTURE_QUALITY_LOW"]);
  assert.doesNotMatch(result.nextRep.join(" "), /增加|减少.*°/);
});

test("deadband prevents classification flapping at a target boundary", () => {
  const inside = evaluateIssues({ metrics: { kneeRange: 21.5 }, confidences: { kneeRange: "高" }, capture: { confidence: "高", score: 90 } }, rules);
  assert.ok(!inside.issues.some((issue) => issue.issueCode === "KNEE_DIP_SHALLOW"));
});
