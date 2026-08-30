const { chromium } = require("playwright");

(async () => {
  const target = process.argv[2] || "http://127.0.0.1:8088/";
  const videoPath = process.argv[3];
  if (!videoPath) throw new Error("usage: node tools/viewpoint-smoke.cjs <url> <video>");
  const browser = await chromium.launch({
    headless: true,
    executablePath: "C:/Program Files/Google/Chrome/Application/chrome.exe",
    args: ["--enable-webgl", "--use-angle=swiftshader"],
  });
  const page = await browser.newPage({ viewport: { width: 390, height: 844 } });
  await page.goto(target, { waitUntil: "domcontentloaded", timeout: 60000 });
  await page.setInputFiles("#video-file", videoPath);
  await page.waitForFunction(() => !document.querySelector("#analyze-button").disabled, null, { timeout: 15000 });
  const result = await page.evaluate(async () => {
    const { analyzeVideoDeterministic } = await import("/assets/pipeline/video-analyzer.mjs");
    const { summarizePoseFrames } = await import("/assets/analyzer-core.mjs");
    const pipeline = await analyzeVideoDeterministic(document.querySelector("#video-preview"), { shootingHand: "right" });
    const coverage = (frames) => Array.from({ length: 33 }, (_, landmark) => ({
      landmark,
      ratio: frames.length ? frames.filter((frame) => frame.landmarks?.[landmark]).length / frames.length : 0,
    })).filter((item) => [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28].includes(item.landmark));
    const armMotion = (frames, indexes) => {
      const samples = frames.map((frame) => {
        const [shoulder, elbow, wrist] = indexes.map((index) => frame.landmarks?.[index]);
        return shoulder && elbow && wrist ? { time: frame.time, wristY: wrist.y, wristX: wrist.x } : null;
      }).filter(Boolean);
      const ys = samples.map((item) => item.wristY);
      return { samples, rangeY: ys.length ? Math.max(...ys) - Math.min(...ys) : 0 };
    };
    const summary = summarizePoseFrames(pipeline.frames, "right", { rawFrames: pipeline.rawFrames, preprocessed: true, orientationPass: pipeline.orientationPass, orientationScores: pipeline.orientationScores });
    return {
      rawCoverage: coverage(pipeline.rawFrames),
      filteredCoverage: coverage(pipeline.frames),
      armMotion: { left: armMotion(pipeline.frames, [11, 13, 15]), right: armMotion(pipeline.frames, [12, 14, 16]) },
      capture: summary.capture,
      release: summary.release && { sampleIndex: summary.release.sampleIndex, time: summary.release.time },
      validFrames: summary.validFrames,
      metrics: summary.metrics,
      jointConfidence: summary.jointConfidence,
      events: summary.events,
      phaseWindow: (summary.analysisFrames || []).map((frame) => ({ time: frame.time, kneeAngle: frame.kneeAngle, hipAngle: frame.hipAngle, elbowAngle: frame.elbowAngle, wristY: frame.wristY })),
    };
  });
  console.log(JSON.stringify(result, null, 2));
  await browser.close();
  if (!result.release) process.exitCode = 1;
})().catch((error) => { console.error(error); process.exit(1); });
