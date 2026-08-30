const SG5 = [-3 / 35, 12 / 35, 17 / 35, 12 / 35, -3 / 35];
const inRange = (point) => point && Number.isFinite(point.x) && Number.isFinite(point.y)
  && point.x >= -0.25 && point.x <= 1.25 && point.y >= -0.25 && point.y <= 1.25;
const distance = (a, b) => a && b ? Math.hypot(a.x - b.x, a.y - b.y) : Infinity;
const clonePoint = (point) => point ? { x: point.x, y: point.y, z: point.z ?? 0, visibility: point.visibility ?? 0 } : null;

function interpolateTrack(track) {
  const output = track.map(clonePoint);
  let start = 0;
  while (start < output.length) {
    if (output[start]) { start += 1; continue; }
    let end = start;
    while (end < output.length && !output[end]) end += 1;
    const gap = end - start;
    if (gap <= 2 && start > 0 && end < output.length && output[start - 1] && output[end]) {
      for (let offset = 1; offset <= gap; offset += 1) {
        const ratio = offset / (gap + 1);
        const a = output[start - 1];
        const b = output[end];
        output[start + offset - 1] = {
          x: a.x + (b.x - a.x) * ratio,
          y: a.y + (b.y - a.y) * ratio,
          z: a.z + (b.z - a.z) * ratio,
          visibility: Math.min(a.visibility, b.visibility),
          interpolated: true,
        };
      }
    }
    start = end;
  }
  return output;
}

function smoothTrack(track) {
  return track.map((point, index) => {
    if (!point) return null;
    if (index < 2 || index > track.length - 3 || track.slice(index - 2, index + 3).some((candidate) => !candidate)) return clonePoint(point);
    const window = track.slice(index - 2, index + 3);
    const smooth = (key) => window.reduce((sum, candidate, offset) => sum + candidate[key] * SG5[offset], 0);
    return { ...point, x: smooth("x"), y: smooth("y"), z: smooth("z"), filtered: true };
  });
}

export function preprocessLandmarkFrames(frames, { visibilityThreshold = 0.55, landmarkCount = 33 } = {}) {
  const raw = frames.map((frame) => ({
    ...frame,
    landmarks: Array.from({ length: landmarkCount }, (_, index) => {
      const candidate = frame.landmarks?.[index];
      return inRange(candidate) && (candidate.visibility ?? 0) >= visibilityThreshold ? clonePoint(candidate) : null;
    }),
  }));
  for (let landmark = 0; landmark < landmarkCount; landmark += 1) {
    for (let index = 1; index < raw.length - 1; index += 1) {
      const current = raw[index].landmarks[landmark];
      const previous = raw[index - 1].landmarks[landmark];
      const next = raw[index + 1].landmarks[landmark];
      if (current && previous && next && distance(current, previous) > 0.22 && distance(current, next) > 0.22 && distance(previous, next) < 0.1) raw[index].landmarks[landmark] = null;
    }
  }
  const tracks = Array.from({ length: landmarkCount }, (_, landmark) => smoothTrack(interpolateTrack(raw.map((frame) => frame.landmarks[landmark]))));
  const filtered = raw.map((frame, index) => ({ ...frame, landmarks: tracks.map((track) => track[index]) }));
  return { raw, filtered };
}

export const LANDMARK_FILTER_VERSION = "visibility55-outlier22-gap2-sg5o2";
