import test from "node:test";
import assert from "node:assert/strict";

import { preprocessLandmarkFrames } from "../netlify-site/assets/pipeline/landmark-series.mjs";
import { selectDeterministicPhases } from "../netlify-site/assets/pipeline/phase-detector.mjs";
import { chooseOrientationPass, normalizeMirroredLandmarks } from "../netlify-site/assets/pipeline/orientation-normalizer.mjs";

const point = (x, y, visibility = 0.95) => ({ x, y, z: 0, visibility });
const frame = (sampleIndex, y, visibility = 0.95) => ({ sampleIndex, timeMs: sampleIndex * 100, landmarks: [point(0.5, y, visibility)] });

test("preprocessing drops low visibility, interpolates at most two samples and rejects isolated jumps", () => {
  const input = [frame(0, 0.5), frame(1, 0.49, 0.2), frame(2, 0.48), frame(3, 0.05), frame(4, 0.46)];
  const output = preprocessLandmarkFrames(input, { landmarkCount: 1 });
  assert.equal(output.raw[1].landmarks[0], null);
  assert.ok(output.filtered[1].landmarks[0]);
  assert.ok(Math.abs(output.filtered[3].landmarks[0].y - 0.47) < 0.04);
});

test("preprocessing does not fill gaps longer than two samples", () => {
  const input = [frame(0, 0.5), frame(1, 0.49, 0.1), frame(2, 0.48, 0.1), frame(3, 0.47, 0.1), frame(4, 0.46)];
  const output = preprocessLandmarkFrames(input, { landmarkCount: 1 });
  assert.equal(output.filtered[2].landmarks[0], null);
});

test("phase selection chooses the earliest lowest-band and near-tied release candidate", () => {
  const samples = [
    { sampleIndex: 0, timeMs: 0, kneeAngle: 150, elbowAngle: 110, wristY: 0.7 },
    { sampleIndex: 1, timeMs: 100, kneeAngle: 120, elbowAngle: 120, wristY: 0.65 },
    { sampleIndex: 2, timeMs: 200, kneeAngle: 118, elbowAngle: 135, wristY: 0.55 },
    { sampleIndex: 3, timeMs: 300, kneeAngle: 135, elbowAngle: 155, wristY: 0.35 },
    { sampleIndex: 4, timeMs: 400, kneeAngle: 150, elbowAngle: 160, wristY: 0.25 },
    { sampleIndex: 5, timeMs: 500, kneeAngle: 155, elbowAngle: 163, wristY: 0.2 },
  ];
  const phases = selectDeterministicPhases(samples);
  assert.equal(phases.lowestPoint.sampleIndex, 1);
  assert.equal(phases.release.sampleIndex, 4);
  assert.equal(phases.followThrough.sampleIndex, 5);
});

test("phase selection reports missing phases instead of inventing them", () => {
  const phases = selectDeterministicPhases([{ sampleIndex: 0, timeMs: 0, kneeAngle: null, elbowAngle: null, wristY: null }]);
  assert.equal(phases.release, null);
  assert.ok(phases.missing.includes("release"));
});

test("mirrored landmarks restore image coordinates and anatomical sides", () => {
  const points = Array.from({ length: 33 }, () => null);
  points[11] = point(0.2, 0.3);
  points[12] = point(0.7, 0.4);
  const normalized = normalizeMirroredLandmarks(points);
  assert.ok(Math.abs(normalized[11].x - 0.3) < 1e-9);
  assert.equal(normalized[11].y, 0.4);
  assert.ok(Math.abs(normalized[12].x - 0.8) < 1e-9);
  assert.equal(normalized[12].y, 0.3);
});

test("orientation pass deterministically selects the sequence with complete legs", () => {
  const weak = [{ landmarks: Array.from({ length: 33 }, (_, index) => point(0.5, 0.5, [27, 28].includes(index) ? 0.1 : 0.95)) }];
  const strong = [{ landmarks: Array.from({ length: 33 }, () => point(0.5, 0.5, 0.95)) }];
  const selected = chooseOrientationPass(weak, strong);
  assert.equal(selected.orientationPass, "mirrored-normalized");
  assert.ok(selected.scores.mirroredNormalized > selected.scores.original);
});
