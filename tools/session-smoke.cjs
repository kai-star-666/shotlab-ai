const { chromium } = require("playwright");
const path = require("node:path");

const target = process.argv[2] || "http://127.0.0.1:8088/#analyzer";
const videos = process.argv.slice(3).map((item) => path.resolve(item));
if (videos.length < 2) throw new Error("需要提供两个不同的视频路径");

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
  await analyze(page, videos[0]);
  await analyze(page, videos[1]);
  const beforeReload = await page.locator("#shot-tabs button").count();
  const comparison = await page.locator("#shot-comparison").textContent();
  await page.reload({ waitUntil: "domcontentloaded" });
  await page.waitForTimeout(500);
  const afterReload = await page.locator("#shot-tabs button").count();
  const overflow = await page.evaluate(() => document.documentElement.scrollWidth > innerWidth + 2);
  await browser.close();
  const result = { beforeReload, afterReload, comparison, overflow };
  process.stdout.write(`${JSON.stringify(result, null, 2)}\n`);
  if (beforeReload !== 2 || afterReload !== 2 || overflow) process.exit(1);
})().catch((error) => { console.error(error); process.exit(1); });
