// Screenshots, fit checks and a recording of the UI, for review.
//
//   node scripts/screenshots.mjs http://127.0.0.1:8000 docs/screenshots ADMIN_KEY [1920x1080,1440x900,1280x720]
//
// Drives the Google Chrome you already have (headless, through the DevTools protocol; no
// extra packages). For every size it does a Reset demo, runs the six demo-script presets
// through the UI, and writes into <out>/<W>x<H>/:
//   normal.png, present-1.png ... present-6.png, present-newrow.png, present-followup.png,
//   present-blocked.png, asks.json (every /api/ask response) and fit.json (anything cut
//   off or scrolling in presentation mode). The first size also gets normal-full.png and
//   frames/ for scripts/make_gif.py. It asks real questions: point it at a demo server.

import { spawn } from "node:child_process";
import { mkdirSync, mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

const [base = "http://127.0.0.1:8000", outRoot = "docs/screenshots", adminKey = "", sizeArg = "1920x1080,1440x900,1280x720"] =
  process.argv.slice(2);
const SIZES = sizeArg.split(",").map((s) => s.split("x").map(Number));
const CHROME = process.env.CHROME ?? "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";
const PORT = 9333;
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const DEMO = JSON.parse(readFileSync(new URL("../data/demo_script.json", import.meta.url), "utf8")).steps;

const profileDir = mkdtempSync(join(tmpdir(), "semcache-chrome-"));
const chrome = spawn(CHROME, [
  "--headless=new", `--remote-debugging-port=${PORT}`, `--user-data-dir=${profileDir}`,
  "--hide-scrollbars", "--force-device-scale-factor=1", "about:blank",
], { stdio: "ignore" });

async function json(path, method = "GET") {
  for (let i = 0; i < 50; i++) {
    try {
      return await (await fetch(`http://127.0.0.1:${PORT}${path}`, { method })).json();
    } catch {
      await sleep(200);
    }
  }
  throw new Error("Chrome did not start");
}

const target = await json(`/json/new?about:blank`, "PUT");
const ws = new WebSocket(target.webSocketDebuggerUrl);
await new Promise((r) => ws.addEventListener("open", r, { once: true }));
let nextId = 1;
const pending = new Map();
ws.addEventListener("message", (ev) => {
  const msg = JSON.parse(ev.data);
  if (msg.id && pending.has(msg.id)) {
    pending.get(msg.id)(msg);
    pending.delete(msg.id);
  }
});
const send = (method, params = {}) =>
  new Promise((resolve, reject) => {
    const id = nextId++;
    pending.set(id, (m) => (m.error ? reject(new Error(`${method}: ${m.error.message}`)) : resolve(m.result)));
    ws.send(JSON.stringify({ id, method, params }));
  });
const evaluate = async (expression) =>
  (await send("Runtime.evaluate", { expression, awaitPromise: true, returnByValue: true })).result.value;
async function waitFor(expression, ms = 60000) {
  const end = Date.now() + ms;
  while (Date.now() < end) {
    if (await evaluate(expression)) return;
    await sleep(100);
  }
  throw new Error(`timed out waiting for ${expression}`);
}

let W = 1920, H = 1080, dir = outRoot;
const size = async (w, h) => {
  [W, H] = [w, h];
  await send("Emulation.setDeviceMetricsOverride", { width: w, height: h, deviceScaleFactor: 1, mobile: false });
};
async function shot(file, full = false) {
  const params = { format: "png" };
  if (full) {
    const h = await evaluate("document.documentElement.scrollHeight");
    Object.assign(params, { captureBeyondViewport: true, clip: { x: 0, y: 0, width: W, height: h, scale: 1 } });
  }
  const { data } = await send("Page.captureScreenshot", params);
  writeFileSync(join(dir, file), Buffer.from(data, "base64"));
}
async function type(text, delay = 45) {
  await evaluate(`(() => { const el = document.getElementById("query"); el.focus();
    const set = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, "value").set;
    set.call(el, ""); el.dispatchEvent(new Event("input", { bubbles: true })); })()`);
  for (const ch of text) {
    await send("Input.insertText", { text: ch });
    await sleep(delay);
  }
}
async function key(k) {
  await evaluate("document.activeElement && document.activeElement.blur()");
  const code = /^\d$/.test(k) ? `Digit${k}` : `Key${k.toUpperCase()}`;
  await send("Input.dispatchKeyEvent", { type: "keyDown", key: k, text: k, code });
  await send("Input.dispatchKeyEvent", { type: "keyUp", key: k, code });
}
const asks = () => evaluate("window.__asks.length");
async function ask(text) {
  const before = await asks();
  await type(text);
  await sleep(400);
  await evaluate(`document.querySelector("form button[type=submit]").click()`);
  await waitFor(`window.__asks.length > ${before}`, 120000);
  return evaluate("window.__asks[window.__asks.length - 1]");
}
const stageIs = (n) => `(() => { const t = document.querySelector('[role=tab][aria-selected=true]');
  return !!t && t.textContent.trim().startsWith("${n} "); })()`;

// Is anything cut off or scrolling? (presentation mode must have neither)
const FIT = `(() => {
  const out = [];
  const vw = innerWidth, vh = innerHeight, de = document.documentElement;
  if (de.scrollHeight > vh + 1) out.push("page scrolls vertically by " + (de.scrollHeight - vh) + "px");
  if (de.scrollWidth > vw + 1) out.push("page scrolls horizontally by " + (de.scrollWidth - vw) + "px");
  for (const el of document.querySelectorAll(".math-panel, .math-panel *, aside, aside *")) {
    const cs = getComputedStyle(el), r = el.getBoundingClientRect();
    if (r.width === 0 || r.height === 0 || cs.visibility === "hidden" || cs.display === "none") continue;
    // clipped on purpose: screen-reader-only text, KaTeX's hidden MathML, the 2-line answer
    if (el.closest(".sr-only, .katex-mathml, [class*=line-clamp]")) continue;
    const name = (el.className.baseVal ?? el.className).toString().slice(0, 50);
    if (/(auto|scroll|hidden)/.test(cs.overflowY + cs.overflowX) && el.scrollHeight > el.clientHeight + 2 && el.tagName !== "SELECT")
      out.push(el.tagName + " " + name + ": content " + el.scrollHeight + " > box " + el.clientHeight);
    if (r.bottom > vh + 1 || r.right > vw + 1) out.push(el.tagName + " " + name + " ends at " + Math.round(r.right) + "x" + Math.round(r.bottom));
  }
  return [...new Set(out)].slice(0, 12);
})()`;

const fitReport = {};
async function presentShot(file) {
  await shot(file);
  fitReport[file] = await evaluate(FIT);
}

try {
  await send("Page.enable");
  await send("Runtime.enable");
  // Keep every /api/ask response, across reloads.
  await send("Page.addScriptToEvaluateOnNewDocument", { source: `
    window.__asks = [];
    const f = window.fetch;
    window.fetch = async (...a) => { const r = await f(...a);
      if (String(a[0]).includes("/api/ask")) r.clone().json().then((j) => window.__asks.push(j)).catch(() => {});
      return r; };` });

  for (const [i, [w, h]] of SIZES.entries()) {
    dir = join(outRoot, `${w}x${h}`);
    mkdirSync(dir, { recursive: true });
    for (const k of Object.keys(fitReport)) delete fitReport[k];
    await size(w, h);
    await send("Page.navigate", { url: base });
    await waitFor("document.readyState === 'complete'");
    await evaluate(`localStorage.setItem("semcache.user", "presenter"); localStorage.removeItem("semcache.thread.presenter");
      sessionStorage.setItem("semcache.adminKey", ${JSON.stringify(adminKey)}); true`);
    const reset = await evaluate(`fetch("/api/reset-demo", { method: "POST", headers: { "X-User": "presenter",
      "X-Admin-Key": ${JSON.stringify(adminKey)} } }).then((r) => r.status)`);
    await send("Page.navigate", { url: base });
    await waitFor("!!document.getElementById('query')");
    if (i === 0 && (await evaluate("document.body.innerText.includes('Model warming up')"))) await shot("warming.png");
    await waitFor("!document.body.innerText.includes('Model warming up')", 180000);
    console.log(`${w}x${h}: reset demo -> ${reset}`);

    // Normal mode: presets 1 and 2 (exact repeat, paraphrase hit), shown at the decision.
    await ask(DEMO[0].query);
    await ask(DEMO[1].query);
    await waitFor(stageIs(5));
    await sleep(1500);
    await shot("normal.png");
    if (i === 0) await shot("normal-full.png", true);

    // Presentation mode: every stage for preset 2.
    await key("p");
    await waitFor("document.documentElement.hasAttribute('data-present')");
    for (const n of [1, 2, 3, 4, 5]) {
      await key(String(n));
      await sleep(n === 3 ? 3000 : n === 4 ? 2400 : 1600);
      await presentShot(`present-${n}.png`);
    }

    // Preset 3 (a miss that grows C), recorded at the first size: typing, stages 1-5, Ask, stage 6.
    let frame = 0, recording = i === 0;
    if (recording) mkdirSync(join(dir, "frames"), { recursive: true });
    const recorder = (async () => {
      while (recording) {
        const { data } = await send("Page.captureScreenshot", { format: "jpeg", quality: 72 });
        writeFileSync(join(dir, "frames", `f${String(frame++).padStart(4, "0")}.jpg`), Buffer.from(data, "base64"));
        await sleep(80);
      }
    })();
    await type(DEMO[2].query, 70);
    if (i === 0) {
      await waitFor(stageIs(5), 30000);
      await sleep(3000);
    } else await sleep(1200);
    const before = await asks();
    await evaluate(`document.querySelector("form button[type=submit]").click()`);
    await waitFor(`window.__asks.length > ${before}`, 120000);
    await waitFor(stageIs(6), 30000);
    await sleep(3500);
    recording = false;
    await recorder;
    await presentShot("present-6.png");

    // Preset 4 hits the new row; 5 is a follow-up; 6 is blocked.
    await ask(DEMO[3].query);
    await key("4");
    await sleep(2400);
    await presentShot("present-newrow.png");
    await ask(DEMO[4].query);
    await key("5");
    await sleep(1500);
    await presentShot("present-followup.png");
    await ask(DEMO[5].query);
    await key("5");
    await sleep(1500);
    await presentShot("present-blocked.png");
    await key("p");
    await sleep(600);

    const all = await evaluate("window.__asks");
    writeFileSync(join(dir, "asks.json"), JSON.stringify(all.slice(-6), null, 1));
    writeFileSync(join(dir, "fit.json"), JSON.stringify(fitReport, null, 1));
    const problems = Object.entries(fitReport).filter(([, v]) => v.length);
    console.log(`  ${problems.length ? "NOT OK: " + JSON.stringify(Object.fromEntries(problems)) : "presentation mode: nothing cut off, no scrolling"}`);
    if (i === 0) console.log(`  ${frame} frames recorded`);
  }
} finally {
  ws.close();
  chrome.kill();
}
