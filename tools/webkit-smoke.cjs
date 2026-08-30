const { webkit } = require("playwright");
const path = require("node:path");

const target = process.argv[2] || "http://127.0.0.1:8088/#analyzer";
const videoPath = path.resolve(process.argv[3] || "shot_test.mp4");

(async () => {
  const browser = await webkit.launch({ headless: true });
  const page = await browser.newPage({ viewport: { width: 390, height: 844 } });
  await page.goto(target, { waitUntil: "domcontentloaded", timeout: 60000 });
  await page.setInputFiles("#video-file", videoPath);
  await page.waitForTimeout(3000);
  const decodeState = await page.evaluate(() => ({
    buttonEnabled: !document.querySelector("#analyze-button").disabled,
    message: document.querySelector("#analysis-message").textContent,
    mp4: document.createElement("video").canPlayType("video/mp4; codecs=avc1.4D401E"),
    webm: document.createElement("video").canPlayType("video/webm; codecs=vp8"),
    overflow: document.documentElement.scrollWidth > innerWidth + 2,
  }));
  if (!decodeState.buttonEnabled) {
    await browser.close();
    process.stdout.write(`${JSON.stringify({ analysisSupported: false, ...decodeState }, null, 2)}\n`);
    if (decodeState.overflow) process.exit(1);
    return;
  }
  await page.click("#analyze-button");
  await page.waitForFunction(() => !document.querySelector("#analyze-button").disabled && /分析完成|分析未完成/.test(document.querySelector("#analysis-message")?.textContent || ""), null, { timeout: 180000 });
  const state = await page.evaluate(() => ({
    message: document.querySelector("#analysis-message").textContent,
    resultVisible: !document.querySelector("#analysis-result").hidden,
    overflow: document.documentElement.scrollWidth > innerWidth + 2,
    shots: document.querySelectorAll("#shot-tabs button").length,
  }));
  await browser.close();
  process.stdout.write(`${JSON.stringify(state, null, 2)}\n`);
  if (!state.message.includes("分析完成") || !state.resultVisible || state.overflow || state.shots !== 1) process.exit(1);
})().catch((error) => { console.error(error); process.exit(1); });
