import test from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";

const html = await readFile(new URL("../netlify-site/index.html", import.meta.url), "utf8");
const app = await readFile(new URL("../netlify-site/assets/analyzer-app.mjs", import.meta.url), "utf8");

test("continuous coaching UI exposes every history and action section", () => {
  for (const id of [
    "history-comparison-grid",
    "improvement-summary",
    "new-problem-list",
    "session-maintain-list",
    "history-table-head",
    "history-table-body",
    "history-trend-list",
    "best-shot-card",
    "adjustment-table-body",
    "training-table-body",
    "session-final-summary",
  ]) assert.match(html, new RegExp(`id=["']${id}["']`), id);
});

test("analyzer app renders structured session analytics instead of only previous-shot chips", () => {
  assert.match(app, /function renderSessionAnalytics\(/);
  assert.match(app, /buildSessionAnalytics/);
  assert.match(app, /historyRows/);
  assert.match(app, /adjustmentRows/);
});

test("stable issue engine receives per-joint viewpoint confidence", () => {
  assert.match(app, /summary\.jointConfidence\?\.\[joint\]/);
  assert.match(app, /capture\.viewpoint\?\.label/);
  assert.match(app, /capture\.shootingArmOccluded/);
});
