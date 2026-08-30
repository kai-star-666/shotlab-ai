const LEFT_RIGHT_PAIRS = [
  [1, 4], [2, 5], [3, 6], [7, 8], [9, 10],
  [11, 12], [13, 14], [15, 16], [17, 18], [19, 20], [21, 22],
  [23, 24], [25, 26], [27, 28], [29, 30], [31, 32],
];
const COACHING_LANDMARKS = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28];
const visible = (point) => Boolean(point && (point.visibility ?? 0) >= 0.55);

export function normalizeMirroredLandmarks(landmarks = []) {
  const normalized = landmarks.map((point) => point ? { ...point, x: 1 - point.x } : null);
  // Horizontal augmentation reverses the apparent anatomical side; restore the declared body side.
  for (const [left, right] of LEFT_RIGHT_PAIRS) [normalized[left], normalized[right]] = [normalized[right], normalized[left]];
  return normalized;
}

export function scorePoseSequence(frames = []) {
  return frames.reduce((total, frame) => {
    const points = frame.landmarks || [];
    const visibleCount = COACHING_LANDMARKS.filter((index) => visible(points[index])).length;
    const completeArm = [[11, 13, 15], [12, 14, 16]].some((chain) => chain.every((index) => visible(points[index])));
    const completeLeg = [[23, 25, 27], [24, 26, 28]].some((chain) => chain.every((index) => visible(points[index])));
    return total + visibleCount + (completeArm ? 4 : 0) + (completeLeg ? 6 : 0);
  }, 0);
}

export function chooseOrientationPass(originalFrames, mirroredNormalizedFrames) {
  const originalScore = scorePoseSequence(originalFrames);
  const mirroredScore = scorePoseSequence(mirroredNormalizedFrames);
  return mirroredScore > originalScore
    ? { orientationPass: "mirrored-normalized", frames: mirroredNormalizedFrames, scores: { original: originalScore, mirroredNormalized: mirroredScore } }
    : { orientationPass: "original", frames: originalFrames, scores: { original: originalScore, mirroredNormalized: mirroredScore } };
}

export const ORIENTATION_NORMALIZER_VERSION = "dual-pass-global-v1";
