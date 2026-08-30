import test from "node:test";
import assert from "node:assert/strict";

import { calculateAngle, summarizePoseFrames } from "../netlify-site/assets/analyzer-core.mjs";

function poseFrame(time, options = {}) {
  const visibility = options.visibility ?? 0.96;
  const points = Array.from({ length: 33 }, (_, index) => ({
    x: 0.42 + (index % 3) * 0.04,
    y: 0.18 + (index % 9) * 0.075,
    visibility,
  }));
  points[11] = { x: 0.42, y: 0.28, visibility };
  points[12] = { x: 0.54 + (options.trunkShift ?? 0), y: 0.28, visibility };
  points[14] = { x: 0.64, y: 0.38, visibility };
  points[16] = { x: options.wristX ?? 0.72, y: options.wristY ?? 0.48, visibility };
  points[23] = { x: 0.44, y: 0.58, visibility };
  points[24] = { x: 0.52, y: 0.58, visibility };
  points[26] = { x: options.kneeX ?? 0.62, y: options.kneeY ?? 0.75, visibility };
  points[28] = { x: 0.54, y: 0.92, visibility };
  return { time, landmarks: points };
}

function representativeShot(visibility = 0.96) {
  return [
    poseFrame(0, { kneeY: 0.78, wristY: 0.52, visibility }),
    poseFrame(0.1, { kneeY: 0.73, wristY: 0.50, visibility }),
    poseFrame(0.2, { kneeY: 0.68, wristY: 0.46, visibility }),
    poseFrame(0.3, { kneeY: 0.72, wristY: 0.36, visibility }),
    poseFrame(0.4, { kneeY: 0.77, wristY: 0.20, visibility }),
    poseFrame(0.5, { kneeY: 0.80, wristY: 0.16, visibility }),
    poseFrame(0.6, { kneeY: 0.80, wristY: 0.19, visibility }),
  ];
}

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
  assert.ok(result.loading.time <= result.release.time);
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

test("coach report exposes evidence, strengths, priorities and two next-shot cues", () => {
  const result = summarizePoseFrames(representativeShot(), "right");

  assert.deepEqual(result.sections.map((section) => section.id), ["elbow", "knee", "hip", "trunk", "wrist", "rhythm"]);
  for (const section of result.sections) {
    assert.ok(section.currentData.length > 0);
    assert.match(section.reference, /训练参考/);
    assert.ok(section.evaluation);
    assert.ok(section.good);
    assert.ok(section.problem);
    assert.ok(section.impact);
    assert.ok(section.nextShot);
    assert.ok(section.longTerm);
    assert.match(section.confidence, /高|中|低/);
  }
  assert.ok(result.strengths.length > 0);
  assert.ok(result.priorities.length <= 2);
  assert.ok(result.nextShot.length <= 2);
  assert.ok(result.nextShot.length > 0);
});

test("report includes measured-reference rows, rhythm events and motion curves", () => {
  const result = summarizePoseFrames(representativeShot(), "right");

  assert.ok(result.dataRows.length >= 8);
  assert.ok(result.dataRows.every((row) => row.reference.includes("训练参考")));
  assert.ok(result.dataRows.every((row) => row.current && row.evaluation && row.nextStep && row.confidence));
  assert.ok(result.events.loadingStart.time <= result.events.lowest.time);
  assert.ok(result.events.lowest.time <= result.events.release.time);
  assert.ok(result.events.followThrough.time >= result.events.release.time);
  assert.equal(result.curves.length, result.analysisFrames.length);
  assert.ok(result.curves.length >= 3);
  assert.ok(Number.isFinite(result.metrics.elbowMin));
  assert.ok(Number.isFinite(result.metrics.elbowRange));
  assert.ok("elbowExtensionTrend" in result.metrics);
  assert.ok(Number.isFinite(result.metrics.trunkMax));
  assert.ok(Number.isFinite(result.metrics.wristLateralDrift));
  assert.doesNotMatch(result.sections.flatMap((section) => section.currentData).join(" "), /null|NaN/);
});

test("each priority produces a concrete training prescription", () => {
  const result = summarizePoseFrames(representativeShot(), "right");

  assert.ok(result.prescriptions.length >= result.priorities.length);
  for (const drill of result.prescriptions) {
    assert.ok(drill.name && drill.purpose && drill.steps && drill.sets && drill.reps && drill.focus);
    assert.ok(drill.rightFeeling && drill.commonMistakes);
  }
});

test("analysis result exposes one complete stable schema for every report section", () => {
  const result = summarizePoseFrames(representativeShot(), "right", {
    duration: 3.4,
    sourceFrames: 102,
    fileName: "shot_test.mp4",
  });

  assert.equal(result.schemaVersion, "2.1");
  for (const key of ["summary", "strengths", "priorities", "nextRep", "metrics", "jointAnalysis", "rhythm", "keyframes", "processedVideo", "skeletonVideo", "charts", "trainingPlan", "confidence", "technicalLimitations", "personalBaseline"]) {
    assert.ok(key in result, `missing ${key}`);
  }
  for (const key of ["strengths", "priorities", "next_rep", "metrics", "joint_analysis", "rhythm", "processed_video", "skeleton_video", "training_plan"]) assert.ok(key in result, `missing validation field ${key}`);
  assert.equal(result.summary.duration, 3.4);
  assert.equal(result.summary.sourceFrames, 102);
  assert.equal(result.summary.shootingHand, "右手");
  assert.equal(result.summary.fileName, "shot_test.mp4");
  assert.equal(result.jointAnalysis.length, 5);
  assert.equal(result.rhythm.events.length, 5);
  assert.equal(result.processedVideo.available, true);
  assert.equal(result.skeletonVideo.available, true);
  assert.ok(result.keyframes.length >= 1);
  assert.ok(result.strengths.length <= 4);
  assert.ok(result.strengths.length >= 2);
  assert.ok(result.keep.length > 0);
  assert.ok(result.why.length > 0);
});

test("unknown source frame count stays unknown instead of copying sampled frames", () => {
  const result = summarizePoseFrames(representativeShot(), "right", { duration: 3.4, sourceFrames: null });
  assert.equal(result.summary.sourceFrames, null);
  assert.equal(result.summary.sampledFrames, result.totalFrames);
});

test("low-confidence result keeps the complete schema and explicit fallbacks", () => {
  const result = summarizePoseFrames([], "left", { duration: 2.5, sourceFrames: 75 });

  assert.equal(result.summary.shootingHand, "左手");
  assert.equal(result.processedVideo.available, false);
  assert.equal(result.skeletonVideo.available, false);
  assert.equal(result.jointAnalysis.length, 5);
  assert.equal(result.rhythm.events.length, 5);
  assert.equal(result.confidence.level, "低");
  assert.ok(result.nextRep[0].includes("重新拍摄"));
  assert.ok(result.technicalLimitations.length >= 3);
});

test("low confidence asks for a reshoot instead of precise angle changes", () => {
  const result = summarizePoseFrames(representativeShot(0.3), "right");

  assert.equal(result.capture.confidence, "低");
  assert.match(result.nextShot[0], /重新拍摄/);
  assert.doesNotMatch(result.nextShot.join(" "), /(?:增加|减少)约 \d+°/);
  assert.equal(result.priorities[0].severity, "明显");
});

test("isolated landmark jumps do not create impossible movement ranges", () => {
  const frames = representativeShot();
  const corrupted = poseFrame(0.25, { wristX: 0.99, wristY: 0.05, kneeX: 0.51, kneeY: 0.59 });
  corrupted.landmarks[12] = { x: 0.9, y: 0.82, visibility: 0.96 };
  frames.splice(3, 0, corrupted);
  const result = summarizePoseFrames(frames, "right");

  assert.ok(result.metrics.trunkMax < 60);
  assert.ok(result.metrics.kneeRange < 100);
  assert.ok(result.metrics.wristLateralDrift < 300);
});

test("rhythm analysis selects one primary shot instead of spanning repeated attempts", () => {
  const first = representativeShot();
  const second = representativeShot().map((frame) => ({ ...frame, time: frame.time + 3 }));
  const result = summarizePoseFrames([...first, ...second], "right");

  assert.ok(result.metrics.rhythmDuration < 1.5);
  assert.ok(result.analysisFrames.at(-1).time - result.analysisFrames[0].time < 1.8);
});
