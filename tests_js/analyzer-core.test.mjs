import test from "node:test";
import assert from "node:assert/strict";

import { calculateAngle, summarizePoseFrames } from "../netlify-site/assets/analyzer-core.mjs";

test("calculateAngle returns a right angle", () => {
  assert.equal(Math.round(calculateAngle({ x: 1, y: 0 }, { x: 0, y: 0 }, { x: 0, y: 1 })), 90);
});

test("summarizePoseFrames finds the loading and release candidates", () => {
  const frame = (time, kneeY, wristY, visibility = 0.99) => {
    const points = Array.from({ length: 33 }, () => ({ x: 0.5, y: 0.5, visibility }));
    points[12] = { x: 0.5, y: 0.3, visibility };
    points[14] = { x: 0.62, y: 0.38, visibility };
    points[16] = { x: 0.72, y: wristY, visibility };
    points[24] = { x: 0.5, y: 0.58, visibility };
    points[26] = { x: 0.62, y: kneeY, visibility };
    points[28] = { x: 0.54, y: 0.92, visibility };
    return { time, landmarks: points };
  };

  const result = summarizePoseFrames([
    frame(0, 0.76, 0.5),
    frame(0.2, 0.68, 0.43),
    frame(0.4, 0.76, 0.18),
    frame(0.6, 0.8, 0.28),
  ], "right");

  assert.equal(result.validFrames, 4);
  assert.equal(result.loading.time, 0.2);
  assert.equal(result.release.time, 0.4);
  assert.ok(result.metrics.elbowRelease > 0);
  assert.ok(result.suggestions.length >= 2);
});

test("low pose visibility is reported as a capture problem", () => {
  const landmarks = Array.from({ length: 33 }, () => ({ x: 0.5, y: 0.5, visibility: 0.1 }));
  const result = summarizePoseFrames([{ time: 0, landmarks }], "left");
  assert.equal(result.validFrames, 0);
  assert.ok(result.suggestions.some((item) => item.title.includes("重新拍摄")));
});
