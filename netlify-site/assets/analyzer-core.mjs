const SIDE = {
  left: { shoulder: 11, elbow: 13, wrist: 15, hip: 23, knee: 25, ankle: 27 },
  right: { shoulder: 12, elbow: 14, wrist: 16, hip: 24, knee: 26, ankle: 28 },
};

const clamp = (value, min, max) => Math.min(max, Math.max(min, value));
const round = (value, digits = 1) => Number.isFinite(value) ? Number(value.toFixed(digits)) : null;

export function calculateAngle(a, b, c) {
  if (!a || !b || !c) return Number.NaN;
  const ab = { x: a.x - b.x, y: a.y - b.y };
  const cb = { x: c.x - b.x, y: c.y - b.y };
  const denominator = Math.hypot(ab.x, ab.y) * Math.hypot(cb.x, cb.y);
  if (!denominator) return Number.NaN;
  return Math.acos(clamp((ab.x * cb.x + ab.y * cb.y) / denominator, -1, 1)) * 180 / Math.PI;
}

function frameMetrics(frame, indexes) {
  const p = frame.landmarks;
  const required = Object.values(indexes).map((index) => p[index]);
  const visible = required.every((point) => point && (point.visibility ?? 1) >= 0.45);
  if (!visible) return { ...frame, valid: false };

  const shoulder = p[indexes.shoulder];
  const elbow = p[indexes.elbow];
  const wrist = p[indexes.wrist];
  const hip = p[indexes.hip];
  const knee = p[indexes.knee];
  const ankle = p[indexes.ankle];
  return {
    ...frame,
    valid: true,
    elbowAngle: calculateAngle(shoulder, elbow, wrist),
    kneeAngle: calculateAngle(hip, knee, ankle),
    hipAngle: calculateAngle(shoulder, hip, knee),
    trunkLean: Math.atan2(shoulder.x - hip.x, hip.y - shoulder.y) * 180 / Math.PI,
    wristY: wrist.y,
  };
}

function advice(metrics, validRatio) {
  if (validRatio < 0.55) {
    return [{
      title: "建议重新拍摄",
      detail: "身体关键点被遮挡较多。固定手机，从投篮手侧前方约 45° 拍摄，并确保脚到手全程入镜。",
    }];
  }

  const items = [];
  if (metrics.elbowRelease !== null && metrics.elbowRelease < 145) {
    items.push({ title: "把随球动作做完整", detail: "出手候选帧肘部伸展偏少。练习近筐单手投篮，感受肘部向篮筐上方伸直，手腕自然下压。" });
  } else {
    items.push({ title: "保持顺畅伸展", detail: "出手阶段肘部伸展基本完整。重点保持肘、腕连续发力，不要为了追求角度刻意锁死手臂。" });
  }

  if (metrics.kneeRange !== null && metrics.kneeRange < 18) {
    items.push({ title: "增加下肢蓄力", detail: "屈伸幅度较小。先做 10 次无球下蹲举手，再做近筐投篮，让膝髋伸展带动上肢。" });
  } else {
    items.push({ title: "稳定动力链节奏", detail: "已经检测到明显屈伸。让脚蹬地、膝髋伸展和举球连续发生，避免在额前停顿后只用手臂发力。" });
  }

  if (metrics.trunkRelease !== null && Math.abs(metrics.trunkRelease) > 16) {
    items.push({ title: "减少躯干侧倾", detail: "出手时身体偏移较明显。用脚架或地面标记固定站位，做正面对框的近筐定点投篮。" });
  } else {
    items.push({ title: "保留稳定轴线", detail: "出手时躯干轴线较稳定。继续保持落地点接近起跳点，并观察疲劳后是否仍能稳定。" });
  }
  return items;
}

export function summarizePoseFrames(frames, shootingHand = "right") {
  const indexes = SIDE[shootingHand] || SIDE.right;
  const measured = frames.map((frame) => frameMetrics(frame, indexes));
  const valid = measured.filter((frame) => frame.valid);
  const validRatio = frames.length ? valid.length / frames.length : 0;

  if (!valid.length) {
    return {
      validFrames: 0,
      totalFrames: frames.length,
      validRatio,
      ready: null,
      loading: null,
      release: null,
      metrics: { elbowRelease: null, kneeLowest: null, kneeRange: null, hipLowest: null, trunkRelease: null },
      suggestions: advice({}, validRatio),
    };
  }

  const loading = valid.reduce((best, frame) => frame.kneeAngle < best.kneeAngle ? frame : best);
  const afterLoading = valid.filter((frame) => frame.time >= loading.time);
  const release = (afterLoading.length ? afterLoading : valid).reduce((best, frame) => frame.wristY < best.wristY ? frame : best);
  const beforeLoading = valid.filter((frame) => frame.time <= loading.time);
  const ready = beforeLoading[0] || valid[0];
  const kneeAngles = valid.map((frame) => frame.kneeAngle).filter(Number.isFinite);
  const metrics = {
    elbowRelease: round(release.elbowAngle),
    kneeLowest: round(loading.kneeAngle),
    kneeRange: kneeAngles.length ? round(Math.max(...kneeAngles) - Math.min(...kneeAngles)) : null,
    hipLowest: round(loading.hipAngle),
    trunkRelease: round(release.trunkLean),
  };

  return {
    validFrames: valid.length,
    totalFrames: frames.length,
    validRatio,
    ready,
    loading,
    release,
    metrics,
    suggestions: advice(metrics, validRatio),
  };
}

export function formatMetric(value, suffix = "°") {
  return value === null || value === undefined || !Number.isFinite(value) ? "—" : `${value}${suffix}`;
}
