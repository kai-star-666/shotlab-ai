const { chromium } = require("playwright");
const path = require("node:path");

const target = process.argv[2] || "http://127.0.0.1:8088/#analyzer";
const videos = process.argv.slice(3).map((item) => path.resolve(item));
if (videos.length < 5) throw new Error("需要提供至少五个不同的视频路径");

async function analyze(page, videoPath) {
  await page.setInputFiles("#video-file", videoPath);
  await page.waitForFunction(() => !document.querySelector("#analyze-button").disabled, null, { timeout: 20000 });
  await page.click("#analyze-button");
  await page.waitForFunction(() => !document.querySelector("#analyze-button").disabled && document.querySelector("#analysis-message")?.textContent.includes("分析完成"), null, { timeout: 180000 });
}

(async () => {
  const browser = await chromium.launch({ headless: true, channel: "chrome" });
  const context = await browser.newContext({ viewport: { width: 390, height: 844 } });
  const page = await context.newPage();
  await page.goto(target, { waitUntil: "domcontentloaded" });
  for (const video of videos.slice(0, 5)) await analyze(page, video);
  const beforeReload = await page.locator("#shot-tabs button").count();
  const comparison = await page.locator("#shot-comparison").textContent();
  const comparisonLayers = await page.locator("#history-comparison-grid .history-comparison-card").count();
  const historyColumns = await page.locator("#history-table-head th").count();
  const historyRows = await page.locator("#history-table-body tr").count();
  const adjustmentRows = await page.locator("#adjustment-table-body tr").count();
  const trainingRows = await page.locator("#training-table-body tr").count();
  const bestShot = await page.locator("#best-shot-card").textContent();
  const currentCues = await page.locator("#next-shot-list li").allTextContents();
  await page.locator("#shot-tabs button").first().click();
  const firstTabActive = await page.locator("#shot-tabs button").first().evaluate((node) => node.classList.contains("active"));
  await page.reload({ waitUntil: "domcontentloaded" });
  await page.waitForFunction(() => document.querySelectorAll("#shot-tabs button").length === 5, null, { timeout: 15000 });
  const afterReload = await page.locator("#shot-tabs button").count();
  const restoredReportVisible = await page.locator("#analysis-result").evaluate((node) => !node.hidden);
  await page.click("#end-session");
  const finalSummaryVisible = await page.locator("#session-final-summary").evaluate((node) => !node.hidden);
  const persisted = await page.evaluate(async () => {
    const request = indexedDB.open("shotlab-ai");
    const db = await new Promise((resolve, reject) => { request.onsuccess = () => resolve(request.result); request.onerror = () => reject(request.error); });
    const transaction = db.transaction("sessions", "readonly");
    const allRequest = transaction.objectStore("sessions").getAll();
    const sessions = await new Promise((resolve, reject) => { allRequest.onsuccess = () => resolve(allRequest.result); allRequest.onerror = () => reject(allRequest.error); });
    const session = sessions.toSorted((a, b) => b.updatedAt.localeCompare(a.updatedAt))[0];
    return {
      shots: session?.shots?.length || 0,
      hasRecord: Boolean(session?.shots?.[0]?.record),
      hasHistory: Boolean(session?.shots?.[4]?.history),
      rawFramesPersisted: Boolean(session?.shots?.some((shot) => shot.analysis?.analysisFrames)),
      comparableShots: session?.shots?.filter((shot, index) => index === 0 || shot.comparison?.comparable).length || 0,
      shotQuality: session?.shots?.map((shot) => ({ shotNumber: shot.shotNumber, confidence: shot.analysis?.capture?.confidence, score: shot.analysis?.capture?.score, bodyScale: shot.analysis?.capture?.bodyScale })) || [],
    };
  });
  const overflow = await page.evaluate(() => document.documentElement.scrollWidth > innerWidth + 2);
  await page.screenshot({ path: path.resolve("outputs/session-five-shot-mobile.png"), fullPage: true });
  await browser.close();
  const result = { beforeReload, afterReload, comparison, comparisonLayers, historyColumns, historyRows, adjustmentRows, trainingRows, bestShot, currentCues, firstTabActive, restoredReportVisible, finalSummaryVisible, persisted, overflow };
  process.stdout.write(`${JSON.stringify(result, null, 2)}\n`);
  if (beforeReload !== 5 || afterReload !== 5 || comparisonLayers !== 4 || historyColumns !== 7 || historyRows < 5 || adjustmentRows < 5 || trainingRows < 1 || !/Shot \d/.test(bestShot) || !currentCues.length || !firstTabActive || !restoredReportVisible || !finalSummaryVisible || persisted.shots !== 5 || persisted.comparableShots !== 5 || !persisted.hasRecord || !persisted.hasHistory || persisted.rawFramesPersisted || overflow) process.exit(1);
})().catch((error) => { console.error(error); process.exit(1); });
