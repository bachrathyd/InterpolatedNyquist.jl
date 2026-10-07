// Optional anonymous usage counts with GoatCounter (https://www.goatcounter.com: no cookies, no
// personal data). OFF unless GOATCOUNTER is set; then the official script is loaded
// asynchronously and only these are sent: one page view (the path, never the query or the
// hash -- a shared link's hash holds the equation), example/<key> when a built-in example is
// opened, custom-equation once per distinct own or edited equation (a local hash decides what is
// new; the text is never sent), the features used (3d-view, stl-export, copy-link, help-open,
// exact-off: once per page session), the GPU vendor class and no-webgpu.
//
//   GOATCOUNTER          'https://<code>.goatcounter.com/count'   ('' = off)
//   SHOW_PUBLIC_COUNTER  true: a small 'N visits' line in the footer, from the site's public
//                        counter (needs 'allow public counter' in the GoatCounter settings)

export const GOATCOUNTER = '';
export const SHOW_PUBLIC_COUNTER = false;

const queue = [];
const sent = new Set();
let ready = false;

function send(o) {
  try { window.goatcounter.count(o); } catch (e) { /* blocked or offline: ignore */ }
}

/** load the script and count the page view (no-op when GOATCOUNTER is empty) */
export function initStats() {
  if (!GOATCOUNTER) return;
  try {
    const s = document.createElement('script');
    s.async = true;
    s.src = 'https://gc.zgo.at/count.js';
    s.dataset.goatcounter = GOATCOUNTER;
    s.dataset.goatcounterSettings = '{"no_onload": true}';
    s.addEventListener('load', () => {
      ready = !!(window.goatcounter && window.goatcounter.count);
      if (!ready) return;
      send({ path: location.pathname });                       // the page view: path only
      while (queue.length) send(queue.shift());
    });
    document.head.appendChild(s);
  } catch (e) { /* ignore */ }
  if (SHOW_PUBLIC_COUNTER) publicCounter();
}

/** an event (path: a fixed label chosen by the page, never user text) */
export function countEvent(path) {
  if (!GOATCOUNTER) return;
  const o = { path, title: path, event: true };
  if (ready) send(o); else if (queue.length < 50) queue.push(o);
}

/** an event at most once per page session (per path) */
export function countOnce(path) {
  if (!GOATCOUNTER || sent.has(path)) return;
  sent.add(path);
  countEvent(path);
}

// own / edited equations: counted when the text has rested 5 s after a successful compile, once
// per distinct equation of the session (32-bit FNV-1a hash, local only)
const seenEq = new Set();
let eqTimer = null;
export function countEquation(text) {
  if (!GOATCOUNTER) return;
  clearTimeout(eqTimer);
  eqTimer = setTimeout(() => {
    let h = 0x811c9dc5;
    for (let i = 0; i < text.length; i++) { h ^= text.charCodeAt(i); h = Math.imul(h, 0x01000193) >>> 0; }
    if (seenEq.has(h)) return;
    seenEq.add(h);
    countEvent('custom-equation');
  }, 5000);
}

/** GPU vendor class from the adapter info */
export function gpuClass(info) {
  const v = `${(info && info.vendor) || ''} ${(info && info.architecture) || ''} ${(info && info.description) || ''}`.toLowerCase();
  if (/nvidia/.test(v)) return 'nvidia';
  if (/amd|ati|radeon/.test(v)) return 'amd';
  if (/intel/.test(v)) return 'intel';
  if (/apple/.test(v)) return 'apple';
  if (/qualcomm|adreno/.test(v)) return 'qualcomm';
  if (/arm|mali/.test(v)) return 'arm';
  return 'other';
}

async function publicCounter() {
  const el = document.getElementById('visits');
  if (!el) return;
  try {
    const base = GOATCOUNTER.replace(/\/count\/?$/, '');
    const r = await fetch(`${base}/counter/${encodeURIComponent(location.pathname)}.json`);
    if (!r.ok) return;
    const j = await r.json();
    if (!j || j.count === undefined) return;
    el.textContent = `${j.count} visits`;
    el.hidden = false;
  } catch (e) { /* hidden on any error */ }
}
