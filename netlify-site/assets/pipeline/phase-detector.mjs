const finite = (value) => Number.isFinite(value);
const clamp = (value, min, max) => Math.min(max, Math.max(min, value));

function phase(sample) {
  return sample ? { sampleIndex: sample.sampleIndex, timeMs: sample.timeMs } : null;
}

export function selectDeterministicPhases(samples) {
  const valid = samples.filter((sample) => finite(sample.kneeAngle) && finite(sample.elbowAngle) && finite(sample.wristY));
  if (valid.length < 3) return { loadingStart: null, lowestPoint: null, release: null, followThrough: null, missing: ["loadingStart", "lowestPoint", "release", "followThrough"] };
  const wristPeak = valid.reduce((best, sample) => sample.wristY < best.wristY ? sample : best, valid[0]);
  const loadingWindow = valid.filter((sample) => sample.timeMs <= wristPeak.timeMs && sample.timeMs >= wristPeak.timeMs - 1200);
  const minimumKnee = Math.min(...loadingWindow.map((sample) => sample.kneeAngle));
  const lowBand = minimumKnee + 4;
  const lowest = loadingWindow.find((sample) => sample.kneeAngle <= lowBand);
  const loadingCandidates = loadingWindow.filter((sample) => sample.timeMs <= lowest.timeMs);
  const initialKnee = loadingCandidates[0].kneeAngle;
  const loadingStart = loadingCandidates.find((sample) => sample.kneeAngle <= initialKnee - 3) || loadingCandidates[0];
  const releasePool = valid.filter((sample) => sample.timeMs > lowest.timeMs && sample.timeMs <= lowest.timeMs + 1200);
  const highestWrist = Math.min(...releasePool.map((sample) => sample.wristY));
  const candidates = releasePool.map((sample, index) => {
    const previous = releasePool[Math.max(0, index - 1)] || lowest;
    const dt = Math.max(0.1, (sample.timeMs - previous.timeMs) / 1000);
    const wristVelocity = clamp((previous.wristY - sample.wristY) / dt, -1, 2);
    const score = (1 - sample.wristY) * 2 + sample.elbowAngle / 180 + sample.kneeAngle / 180 + wristVelocity * 0.35;
    return { sample, score };
  }).filter(({ sample }) => sample.wristY <= highestWrist + 0.1);
  if (!candidates.length) return { loadingStart: phase(loadingStart), lowestPoint: phase(lowest), release: null, followThrough: null, missing: ["release", "followThrough"] };
  const maximum = Math.max(...candidates.map((candidate) => candidate.score));
  const release = candidates.find((candidate) => candidate.score >= maximum - 0.12)?.sample;
  const follow = valid.filter((sample) => sample.timeMs >= release.timeMs && sample.timeMs <= release.timeMs + 300).at(-1) || release;
  return { loadingStart: phase(loadingStart), lowestPoint: phase(lowest), release: phase(release), followThrough: phase(follow), missing: [] };
}
