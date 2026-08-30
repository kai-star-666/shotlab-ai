const { chromium } = require("playwright");
const fs = require("node:fs");
const path = require("node:path");

const target = process.argv[2] || "http://127.0.0.1:8088/?debug=1#analyzer";
const videoPath = path.resolve(process.argv[3] || "shot_test.mp4");
const outputPath = path.resolve(process.argv[4] || "outputs/repeatability_report.json");

const mean = (values) => values.reduce((sum, value) => sum + value, 0) / values.length;
const stats = (values) => {
  const usable = values.filter(Number.isFinite);
  if (!usable.length) return { mean: null, standardDeviation: null, maxDifference: null };
  const average = mean(usable);
  return { mean: average, standardDeviation: Math.sqrt(mean(usable.map((value) => (value - average) ** 2))), maxDifference: Math.max(...usable) - Math.min(...usable) };
};

function resultRow(summary, mode, run) {
  const at = (event) => Number.isFinite(event?.time) ? Math.round(event.time * 10) : null;
  return {
    mode, run,
    lowest_point_frame: at(summary.events?.lowest), release_frame: at(summary.events?.release),
    release_elbow_angle: summary.metrics?.elbowRelease ?? null,
    lowest_knee_angle: summary.metrics?.kneeLowest ?? null,
    lowest_hip_angle: summary.metrics?.hipLowest ?? null,
    release_trunk_angle: summary.metrics?.trunkRelease ?? null,
    wrist_metric: summary.metrics?.wristPathStability ?? summary.metrics?.wristLateralDrift ?? null,
    primary_issue: summary.priorities?.[0]?.issueCode ?? null,
    secondary_issue: summary.priorities?.[1]?.issueCode ?? null,
    strengths: summary.strengths || [], next_rep_cues: summary.deterministicNextRep || summary.nextRep || [],
    hashes: {
      decoded_frames: summary.pipeline?.hashes?.decodedFrames ?? null,
      raw_landmarks: summary.pipeline?.hashes?.rawLandmarks ?? null,
      filtered_landmarks: summary.pipeline?.hashes?.filteredLandmarks ?? null,
      phases: summary.phasesHash ?? null, final_json: summary.finalHash ?? null,
    },
  };
}

async function analyze(page, mode, run) {
  await page.setInputFiles("#video-file", videoPath);
  await page.waitForFunction(() => !document.querySelector("#analyze-button").disabled, null, { timeout: 20000 });
  await page.click("#analyze-button");
  await page.waitForFunction(() => !document.querySelector("#analyze-button").disabled && document.querySelector("#analysis-message")?.textContent.includes("分析完成"), null, { timeout: 180000 });
  return resultRow(JSON.parse(await page.locator("#debug-analysis-json").textContent()), mode, run);
}

(async () => {
  const browser = await chromium.launch({ headless: true, channel: "chrome", args: ["--use-angle=swiftshader"] });
  const context = await browser.newContext({ viewport: { width: 1280, height: 900 } });
  const rows = [];
  const first = await context.newPage();
  await first.goto(target, { waitUntil: "domcontentloaded", timeout: 60000 });
  for (let run = 1; run <= 5; run += 1) rows.push(await analyze(first, "same-page", run));
  await first.close();
  for (let run = 1; run <= 5; run += 1) {
    const page = await context.newPage();
    await page.goto(target, { waitUntil: "domcontentloaded", timeout: 60000 });
    rows.push(await analyze(page, "fresh-page", run));
    await page.close();
  }
  const probe = await context.newPage();
  await probe.goto(target, { waitUntil: "domcontentloaded", timeout: 60000 });
  await probe.waitForTimeout(500);
  const indexedDbRestoredShots = await probe.locator("#shot-tabs button").count();
  await browser.close();
  const metricKeys = ["lowest_point_frame", "release_frame", "release_elbow_angle", "lowest_knee_angle", "lowest_hip_angle", "release_trunk_angle", "wrist_metric"];
  const summary = Object.fromEntries(metricKeys.map((key) => [key, stats(rows.map((row) => row[key]))]));
  const classificationConsistency = (key) => new Set(rows.map((row) => row[key])).size === 1 ? 1 : 1 / new Set(rows.map((row) => row[key])).size;
  const report = {
    schemaVersion: "RepeatabilityReportV1", generatedAt: new Date().toISOString(), target, videoPath,
    runs: rows, summary,
    consistency: { primary_issue: classificationConsistency("primary_issue"), secondary_issue: classificationConsistency("secondary_issue"), cue_direction: new Set(rows.map((row) => JSON.stringify(row.next_rep_cues))).size === 1 ? 1 : 0 },
    hashesConsistent: Object.fromEntries(Object.keys(rows[0].hashes).map((key) => [key, new Set(rows.map((row) => row.hashes[key])).size === 1])),
    indexedDbRestoredShots,
  };
  fs.mkdirSync(path.dirname(outputPath), { recursive: true });
  fs.writeFileSync(outputPath, `${JSON.stringify(report, null, 2)}\n`);
  process.stdout.write(`${outputPath}\n`);
})().catch((error) => { console.error(error); process.exit(1); });
