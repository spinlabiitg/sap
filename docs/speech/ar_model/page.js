(() => {
  const $ = id => document.getElementById(id);
  const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
  const SR = 16000;

  function setupCanvas(cv) {
    if (!cv.dataset.h) { cv.dataset.h = cv.getAttribute('height'); cv.style.height = cv.dataset.h + 'px'; }
    const dpr = window.devicePixelRatio || 1, W = cv.clientWidth, H = +cv.dataset.h;
    cv.width = W * dpr; cv.height = H * dpr; const c = cv.getContext('2d'); c.setTransform(dpr, 0, 0, dpr, 0, 0); c.clearRect(0, 0, W, H);
    return { c, W, H };
  }
  // set HTML that may contain \( \) maths, and typeset it once KaTeX's auto-render has loaded
  // (before that, the page-level auto-render pass picks it up)
  function setMath(el, html) {
    el.innerHTML = html;
    if (window.renderMathInElement) renderMathInElement(el, { delimiters: [{ left: '\\(', right: '\\)', display: false }], throwOnError: false });
  }
  let actx = null, cur = null;
  function play(y) {
    try {
      actx = actx || new (window.AudioContext || window.webkitAudioContext)();
      if (actx.state === 'suspended') actx.resume();
      if (cur) { try { cur.stop(); } catch (e) {} }
      let m = 1e-12; for (const v of y) m = Math.max(m, Math.abs(v));
      const buf = actx.createBuffer(1, y.length, SR), ch = buf.getChannelData(0);
      for (let i = 0; i < y.length; i++) ch[i] = 0.6 * y[i] / m * Math.min(1, i / 200, (y.length - i) / 300);
      const s = actx.createBufferSource(); s.buffer = buf; s.connect(actx.destination); s.start(); cur = s;
    } catch (e) { console.warn(e); }
  }
  function pulses(f0, n) { const x = new Float64Array(n); for (let t = 0; t < n; t += SR / f0) x[Math.round(t)] = 1; return x; }
  // all-pole filter: y[n] = x[n] + Σ a_k y[n-k]
  function allpole(x, a) {
    const y = new Float64Array(x.length);
    for (let n = 0; n < x.length; n++) { let s = x[n]; for (let k = 1; k <= a.length; k++) if (n - k >= 0) s += a[k - 1] * y[n - k]; y[n] = s; }
    return y;
  }
  // predictor weights from pole pairs {r, th}: expand Π (1 - 2r cos th z^-1 + r² z^-2)
  function weightsFromPairs(pairs) {
    let A = [1];
    for (const { r, th } of pairs) {
      const q = [1, -2 * r * Math.cos(th), r * r], out = new Array(A.length + 2).fill(0);
      for (let i = 0; i < A.length; i++) for (let j = 0; j < 3; j++) out[i + j] += A[i] * q[j];
      A = out;
    }
    return A.slice(1).map(v => -v);
  }
  // |1/A(e^jw)| for predictor weights a
  function respAt(a, f) {
    const w = 2 * Math.PI * f / SR; let re = 1, im = 0;
    for (let k = 1; k <= a.length; k++) { re -= a[k - 1] * Math.cos(w * k); im += a[k - 1] * Math.sin(w * k); }
    return 1 / Math.hypot(re, im);
  }
  function grid(c, W, H, pl, pr, pt, pb, fmax, floor, step) {
    const pw = W - pl - pr, ph = H - pt - pb;
    c.font = '10px Inter, sans-serif'; c.fillStyle = css('--muted'); c.strokeStyle = css('--line'); c.lineWidth = 1;
    for (let f = 0; f <= fmax; f += step) { const x = pl + f / fmax * pw; c.beginPath(); c.moveTo(x, pt); c.lineTo(x, pt + ph); c.stroke(); c.fillText((f / 1000) + 'k', x - 6, H - 4); }
    for (let d = 0; d >= floor; d -= 20) { const y = pt + (-d / -floor) * ph; c.beginPath(); c.moveTo(pl, y); c.lineTo(pl + pw, y); c.stroke(); c.fillText(d, 4, y + 3); }
    return { pw, ph, X: f => pl + f / fmax * pw, Y: d => pt + Math.min(1, Math.max(0, -d / -floor)) * ph };
  }
  function drawZ(c, W, H, poles, opts = {}) {      // poles: [{re, im}] (both members of each pair)
    const S = Math.min(W, H) - 20, cx = W / 2, cy = H / 2, R = S / 2 * 0.86;
    c.strokeStyle = css('--line'); c.lineWidth = 1;
    c.beginPath(); c.moveTo(cx - S / 2, cy); c.lineTo(cx + S / 2, cy); c.moveTo(cx, cy - S / 2); c.lineTo(cx, cy + S / 2); c.stroke();
    c.strokeStyle = css('--ink'); c.lineWidth = 1.5; c.beginPath(); c.arc(cx, cy, R, 0, 2 * Math.PI); c.stroke();
    c.fillStyle = css('--muted'); c.font = '10px Inter, sans-serif'; c.textAlign = 'center';
    c.fillText('0 Hz', cx + R + 14, cy - 4); c.fillText('8 kHz', cx - R - 16, cy - 4); c.fillText('4 kHz', cx, cy - R - 5);
    if (opts.ticks) for (const f of [1000, 2000, 3000, 5000, 6000, 7000]) { const th = 2 * Math.PI * f / SR; c.beginPath(); c.moveTo(cx + R * Math.cos(th), cy - R * Math.sin(th)); c.lineTo(cx + (R + 5) * Math.cos(th), cy - (R + 5) * Math.sin(th)); c.stroke(); }
    c.textAlign = 'left';
    const P = z => [cx + z.re * R, cy - z.im * R];
    for (const z of poles) {
      const [x, y] = P(z); const col = z.im >= 0 ? css('--accent-2') : css('--accent-4');
      c.strokeStyle = col; c.lineWidth = 2.4; c.beginPath(); c.moveTo(x - 5, y - 5); c.lineTo(x + 5, y + 5); c.moveTo(x + 5, y - 5); c.lineTo(x - 5, y + 5); c.stroke();
    }
    return { cx, cy, R, P };
  }

  // ================= 1. single resonator =================
  function rParams() { const F = +$('rF').value, r = +$('rR').value / 1000, w = 2 * Math.PI * F / SR; return { F, r, w, a: [2 * r * Math.cos(w), -r * r] }; }
  function rDraw() {
    const { F, r, w, a } = rParams(), unstable = r >= 1;
    $('rFv').textContent = F + ' Hz'; $('rRv').textContent = r.toFixed(3);
    const B = -SR / Math.PI * Math.log(r), tau = -1 / (SR * Math.log(r)) * 1000;
    setMath($('rRead'), `<div>\\(a_1 = 2\\rho\\cos\\omega\\)<b>${a[0].toFixed(3)}</b></div><div>\\(a_2 = -\\rho^2\\)<b>${a[1].toFixed(3)}</b></div>` +
      (unstable ? `<div>bandwidth<b>—</b></div><div>status<b style="color:var(--accent-2)">${r === 1 ? 'on the edge' : 'unstable: grows forever'}</b></div>`
                : `<div>bandwidth \\(B\\)<b>${B.toFixed(0)} Hz</b></div><div>ring decays (\\(\\times 1/e\\)) in<b>${tau.toFixed(1)} ms</b></div>`));
    // impulse response
    { const { c, W, H } = setupCanvas($('rImp')), n = Math.round(0.02 * SR), imp = new Float64Array(n); imp[0] = 1; const h = allpole(imp, a);
      let m = 1e-9; for (const v of h) m = Math.max(m, Math.abs(v));
      c.strokeStyle = css('--line'); c.beginPath(); c.moveTo(0, H / 2); c.lineTo(W, H / 2); c.stroke();
      c.strokeStyle = unstable ? css('--accent-2') : css('--accent'); c.lineWidth = 1.4; c.beginPath();
      for (let i = 0; i < n; i++) { const x = i / n * W, y = H / 2 - h[i] / m * (H / 2 - 8); i ? c.lineTo(x, y) : c.moveTo(x, y); }
      c.stroke(); c.fillStyle = css('--muted'); c.font = '10px Inter, sans-serif'; c.fillText('0', 2, H - 4); c.fillText('20 ms', W - 34, H - 4); }
    // spectrum + mini z-plane inset
    { const { c, W, H } = setupCanvas($('rSpec')), g = grid(c, W, H, 30, 8, 8, 18, 8000, -60, 2000);
      if (!unstable) { const v = []; let m = 0; for (let f = 0; f <= 8000; f += 20) { const d = respAt(a, f); v.push(d); m = Math.max(m, d); }
        c.strokeStyle = css('--accent'); c.lineWidth = 2; c.beginPath(); v.forEach((d, i) => { const x = g.X(i * 20), y = g.Y(20 * Math.log10(d / m)); i ? c.lineTo(x, y) : c.moveTo(x, y); }); c.stroke(); }
      else { c.fillStyle = css('--accent-2'); c.font = '12px Inter, sans-serif'; c.fillText('ρ ≥ 1: no steady frequency response', 40, H / 2); }
      const s = 74, ox = W - s - 10, oy = 10; c.fillStyle = css('--surface'); c.fillRect(ox, oy, s, s); c.strokeStyle = css('--line'); c.strokeRect(ox, oy, s, s);
      const R = s / 2 - 6, cx = ox + s / 2, cy = oy + s / 2;
      c.strokeStyle = css('--ink'); c.lineWidth = 1; c.beginPath(); c.arc(cx, cy, R, 0, 2 * Math.PI); c.stroke();
      c.strokeStyle = css('--accent-2'); c.lineWidth = 2;
      for (const sg of [1, -1]) { const x = cx + r * Math.cos(w) * R, y = cy - sg * r * Math.sin(w) * R; c.beginPath(); c.moveTo(x - 3.5, y - 3.5); c.lineTo(x + 3.5, y + 3.5); c.moveTo(x + 3.5, y - 3.5); c.lineTo(x - 3.5, y + 3.5); c.stroke(); }
      c.fillStyle = css('--muted'); c.font = '9px Inter, sans-serif'; c.fillText('poles', ox + 3, oy + s - 3); }
    $('rNote').textContent = unstable ? 'Feedback stronger than 1: every echo returns bigger. Playback disabled.' : r > 0.995 ? 'Very close to the circle: a long, pure ring and a needle-sharp peak.' : r < 0.9 ? 'Far from the circle: the ring dies almost at once and the peak is broad.' : '';
    $('rPlay').disabled = unstable;
  }
  ['rF', 'rR'].forEach(id => $(id).addEventListener('input', rDraw));
  $('rPlay').addEventListener('click', () => { const { r, a } = rParams(); if (r < 1) play(allpole(pulses(120, SR), a)); });

  // ================= 2. z-plane designer =================
  const PRE = {
    neutral: [[500, 80], [1500, 90], [2500, 120], [3500, 180]],
    a: [[730, 80], [1090, 90], [2440, 120], [3400, 180]],
    i: [[270, 60], [2290, 100], [3010, 120], [3700, 180]],
    u: [[300, 70], [870, 90], [2240, 120], [3300, 180]],
  };
  const toPair = ([F, B]) => ({ r: Math.exp(-Math.PI * B / SR), th: 2 * Math.PI * F / SR });
  let zPairs = PRE.a.map(toPair), zDrag = -1;
  function zGeom() { const cv = $('zCanvas'), W = cv.clientWidth, H = +(cv.dataset.h || cv.getAttribute('height')); const S = Math.min(W, H) - 20; return { cx: W / 2, cy: H / 2, R: S / 2 * 0.86 }; }
  function zDraw() {
    const a = weightsFromPairs(zPairs), probe = +$('zProbe').value, wp = 2 * Math.PI * probe / SR;
    $('zProbeV').textContent = probe + ' Hz';
    { const { c, W, H } = setupCanvas($('zCanvas'));
      const poles = zPairs.flatMap(({ r, th }) => [{ re: r * Math.cos(th), im: r * Math.sin(th) }, { re: r * Math.cos(th), im: -r * Math.sin(th) }]);
      const g = drawZ(c, W, H, poles, { ticks: true });
      const [px, py] = g.P({ re: Math.cos(wp), im: Math.sin(wp) });
      c.strokeStyle = css('--accent-3'); c.lineWidth = 1.2; c.setLineDash([3, 3]);
      for (const z of poles) { const [x, y] = g.P(z); c.beginPath(); c.moveTo(px, py); c.lineTo(x, y); c.stroke(); }
      c.setLineDash([]); c.fillStyle = css('--accent-3'); c.beginPath(); c.arc(px, py, 5.5, 0, 2 * Math.PI); c.fill();
      c.fillStyle = css('--ink'); c.font = '600 10px Inter, sans-serif'; c.fillText('probe', px + 7, py - 6);
      zPairs.forEach(({ r, th }, i) => { const [x, y] = g.P({ re: r * Math.cos(th), im: r * Math.sin(th) }); c.fillStyle = css('--accent-2'); c.font = '600 10px Inter, sans-serif'; c.fillText('F' + (i + 1), x + 7, y - 6); }); }
    { const { c, W, H } = setupCanvas($('zSpec')), g = grid(c, W, H, 30, 8, 8, 18, 5000, -60, 1000);
      const v = []; let m = 0; for (let f = 0; f <= 5000; f += 10) { const d = respAt(a, f); v.push(d); m = Math.max(m, d); }
      c.strokeStyle = css('--accent'); c.lineWidth = 2; c.beginPath(); v.forEach((d, i) => { const x = g.X(i * 10), y = g.Y(20 * Math.log10(d / m)); i ? c.lineTo(x, y) : c.moveTo(x, y); }); c.stroke();
      c.strokeStyle = css('--accent-3'); c.setLineDash([4, 3]); c.beginPath(); c.moveTo(g.X(probe), 8); c.lineTo(g.X(probe), H - 18); c.stroke(); c.setLineDash([]);
      const dp = 20 * Math.log10(respAt(a, probe) / m); c.fillStyle = css('--accent-3'); c.beginPath(); c.arc(g.X(probe), g.Y(dp), 4.5, 0, 2 * Math.PI); c.fill(); }
    // distances at the probe
    let prod = 1; const rows = zPairs.map(({ r, th }, i) => {
      const d1 = Math.hypot(Math.cos(wp) - r * Math.cos(th), Math.sin(wp) - r * Math.sin(th)), d2 = Math.hypot(Math.cos(wp) - r * Math.cos(th), Math.sin(wp) + r * Math.sin(th));
      prod *= d1 * d2;
      return `<tr><td>\\(F_${i + 1}\\)</td><td>${Math.round(th * SR / (2 * Math.PI))} Hz</td><td>${Math.round(-SR / Math.PI * Math.log(r))} Hz</td><td>${d1.toFixed(3)}</td></tr>`;
    }).join('');
    setMath($('zTab'), `<tr><th>pole</th><th>frequency</th><th>bandwidth</th><th>distance to probe</th></tr>${rows}`);
    const [pm, pe] = prod.toExponential(2).split('e');
    setMath($('zNote'), `At the probe, \\(|H| = 1 / (\\text{product of all 8 distances}) = 1 / (${pm} \\times 10^{${+pe}})\\).`);
  }
  function zPick(ev) {
    const cv = $('zCanvas'), rect = cv.getBoundingClientRect(), g = zGeom();
    return { re: (ev.clientX - rect.left - g.cx) / g.R, im: -(ev.clientY - rect.top - g.cy) / g.R };
  }
  $('zCanvas').addEventListener('pointerdown', ev => {
    const z = zPick(ev), g = zGeom(); let best = -1, bd = 18 / g.R;
    zPairs.forEach(({ r, th }, i) => { const d = Math.min(Math.hypot(z.re - r * Math.cos(th), z.im - r * Math.sin(th)), Math.hypot(z.re - r * Math.cos(th), z.im + r * Math.sin(th))); if (d < bd) { bd = d; best = i; } });
    if (best >= 0) { zDrag = best; $('zCanvas').setPointerCapture(ev.pointerId); document.querySelectorAll('#zPreset button').forEach(b => b.classList.remove('on')); }
  });
  $('zCanvas').addEventListener('pointermove', ev => {
    if (zDrag < 0) return;
    const z = zPick(ev), r = Math.min(0.995, Math.max(0.5, Math.hypot(z.re, z.im))), th = Math.min(Math.PI - 0.02, Math.max(0.02, Math.abs(Math.atan2(z.im, z.re))));
    zPairs[zDrag] = { r, th }; zDraw();
  });
  ['pointerup', 'pointercancel'].forEach(e => $('zCanvas').addEventListener(e, () => { zDrag = -1; }));
  document.querySelectorAll('#zPreset button').forEach(b => b.addEventListener('click', () => {
    zPairs = PRE[b.dataset.p].map(toPair); document.querySelectorAll('#zPreset button').forEach(x => x.classList.toggle('on', x === b)); zDraw();
  }));
  $('zProbe').addEventListener('input', zDraw);
  $('zPlay').addEventListener('click', () => {
    const a = weightsFromPairs(zPairs), src = pulses(120, SR);
    let lp = 0; for (let n = 0; n < src.length; n++) { lp = 0.9 * lp + src[n]; src[n] = lp; }   // one-pole low-pass: a gentle, glottal-like spectral tilt
    play(allpole(src, a));
  });

  // ================= 3. live LPC fit on the speech excerpts of linear_prediction/signals.js =================
  const SIG = window.LP_SIGNALS || {};
  const FKEYS = ['aa', 'iy', 's'].filter(k => SIG[k] && SIG[k].x && SIG[k].x.length >= 520);
  // button text comes from the excerpt's own label (e.g. 'vowel /ɑ/ (“word”)'), so it follows whatever recording signals.js holds
  const FBASE = { aa: '/ɑ/', iy: '/i/', s: '/s/' };
  const FLABEL = Object.fromEntries(FKEYS.map(k => [k, (SIG[k].label || FBASE[k]).replace(/^(vowel|fricative)\s+/i, '')]));
  let fKey = FKEYS[0] || 'aa';
  function fft(re, im) {
    const n = re.length;
    for (let i = 1, j = 0; i < n; i++) { let bit = n >> 1; for (; j & bit; bit >>= 1) j ^= bit; j ^= bit; if (i < j) { [re[i], re[j]] = [re[j], re[i]]; [im[i], im[j]] = [im[j], im[i]]; } }
    for (let len = 2; len <= n; len <<= 1) { const a = -2 * Math.PI / len, wr = Math.cos(a), wi = Math.sin(a);
      for (let i = 0; i < n; i += len) { let cr = 1, ci = 0; for (let k = 0; k < len / 2; k++) { const p = i + k, q = p + len / 2, br = re[q] * cr - im[q] * ci, bi = re[q] * ci + im[q] * cr;
        re[q] = re[p] - br; im[q] = im[p] - bi; re[p] += br; im[p] += bi; const t = cr * wr - ci * wi; ci = cr * wi + ci * wr; cr = t; } } }
  }
  function levinson(r, p) {
    const A = new Float64Array(p + 1); A[0] = 1; let e = r[0] * (1 + 1e-9);
    for (let i = 1; i <= p; i++) {
      let acc = r[i]; for (let j = 1; j < i; j++) acc += A[j] * r[i - j];
      const k = -acc / e, prev = A.slice();
      for (let j = 1; j < i; j++) A[j] = prev[j] + k * prev[i - j];
      A[i] = k; e *= 1 - k * k;
    }
    return { a: Array.from(A.slice(1), v => -v), e };
  }
  // roots of z^p - a1 z^{p-1} - ... - ap (Durand–Kerner)
  function roots(a) {
    const p = a.length, coef = [1, ...a.map(v => -v)];
    let zs = Array.from({ length: p }, (_, k) => { const ang = 2 * Math.PI * k / p + 0.4; return [0.9 * Math.cos(ang), 0.9 * Math.sin(ang)]; });
    const mul = (x, y) => [x[0] * y[0] - x[1] * y[1], x[0] * y[1] + x[1] * y[0]];
    const div = (x, y) => { const d = y[0] * y[0] + y[1] * y[1] || 1e-30; return [(x[0] * y[0] + x[1] * y[1]) / d, (x[1] * y[0] - x[0] * y[1]) / d]; };
    for (let it = 0; it < 400; it++) {
      let moved = 0;
      zs = zs.map((z, i) => {
        let v = [coef[0], 0]; for (let k = 1; k <= p; k++) { v = mul(v, z); v[0] += coef[k]; }
        let den = [1, 0]; zs.forEach((w, j) => { if (j !== i) den = mul(den, [z[0] - w[0], z[1] - w[1]]); });
        const d = div(v, den); moved = Math.max(moved, Math.hypot(d[0], d[1]));
        return [z[0] - d[0], z[1] - d[1]];
      });
      if (moved < 1e-12) break;
    }
    return zs;
  }
  function fDraw() {
    const p = +$('fP').value, pre = $('fPre').checked, raw = SIG[fKey].x, N = 512, c0 = Math.floor(raw.length / 2) - N / 2;
    $('fPv').textContent = p + (p === 18 ? ' (rule of thumb)' : '');
    const x = new Float64Array(N);
    for (let n = 0; n < N; n++) { const v = raw[c0 + n] - (pre ? 0.97 * raw[c0 + n - 1] : 0); x[n] = v * (0.54 - 0.46 * Math.cos(2 * Math.PI * n / (N - 1))); }
    const r = new Float64Array(p + 1); for (let k = 0; k <= p; k++) { let s = 0; for (let n = k; n < N; n++) s += x[n] * x[n - k]; r[k] = s; }
    const { a, e } = levinson(r, p);
    const M = 1024, re = new Float64Array(M), im = new Float64Array(M); re.set(x); fft(re, im);
    const spec = [], env = []; let mx = -1e9;
    for (let k = 0; k <= M / 2; k++) { const s = 10 * Math.log10(re[k] * re[k] + im[k] * im[k] + 1e-12); spec.push(s); mx = Math.max(mx, s); }
    for (let k = 0; k <= M / 2; k++) { const f = k * SR / M; env.push(10 * Math.log10(e * respAt(a, f) ** 2 + 1e-12)); }
    const zs = roots(a), fm = [];
    zs.forEach(([zr, zi]) => { if (zi <= 0) return; const F = Math.atan2(zi, zr) * SR / (2 * Math.PI), B = -SR / Math.PI * Math.log(Math.hypot(zr, zi)); if (F > 90 && B < 400) fm.push([F, B]); });
    fm.sort((u, v) => u[0] - v[0]);
    { const { c, W, H } = setupCanvas($('fSpec')), g = grid(c, W, H, 30, 8, 8, 18, 8000, -80, 2000);
      c.strokeStyle = css('--line-strong'); c.lineWidth = 1; c.beginPath(); spec.forEach((s, k) => { const X = g.X(k * SR / M), Y = g.Y(s - mx); k ? c.lineTo(X, Y) : c.moveTo(X, Y); }); c.stroke();
      c.strokeStyle = css('--accent-2'); c.lineWidth = 2.2; c.beginPath(); env.forEach((s, k) => { const X = g.X(k * SR / M), Y = g.Y(s - mx); k ? c.lineTo(X, Y) : c.moveTo(X, Y); }); c.stroke();
      fm.forEach(([F]) => { const k = Math.round(F / SR * M), X = g.X(F), Y = g.Y(env[k] - mx) - 6; c.fillStyle = css('--ink'); c.beginPath(); c.moveTo(X - 4, Y - 6); c.lineTo(X + 4, Y - 6); c.lineTo(X, Y); c.fill(); }); }
    { const { c, W, H } = setupCanvas($('fZ')); drawZ(c, W, H, zs.map(([re2, im2]) => ({ re: re2, im: im2 }))); }
    setMath($('fTab'), fm.length ? `<tr><th>formant (from a pole pair)</th><th>frequency</th><th>bandwidth</th></tr>` + fm.slice(0, 6).map(([F, B], i) => `<tr><td>\\(F_${i + 1}\\)</td><td>${Math.round(F)} Hz</td><td>${Math.round(B)} Hz</td></tr>`).join('')
      : '<tr><td>no pole pair sharp enough (bandwidth &lt; 400 Hz) to count as a formant</td></tr>');
    const notes = {
      aa: p < 8 ? 'Too few poles: formants merge into broad humps.' : p > 28 ? 'Too many poles: spare pairs latch onto harmonics; the envelope sprouts bumps that are not formants.' : 'Each formant has its own pole pair; read \\(F_1\\), \\(F_2\\) off the table.',
      iy: '/i/: low \\(F_1\\), and \\(F_2/F_3\\) high and close together. Watch them split apart as the order rises.',
      s: 'For /s/ the poles describe where the noise energy sits (high up), not voiced formants. Many are broad, so few pass the bandwidth test.',
    };
    setMath($('fNote'), notes[fKey] + (pre ? '' : ' (Pre-emphasis off: watch the high formants weaken or disappear.)'));
  }
  const fSeg = $('fSig');
  FKEYS.forEach(k => { const b = document.createElement('button'); b.textContent = FLABEL[k]; if (k === fKey) b.classList.add('on');
    b.addEventListener('click', () => { fKey = k; fSeg.querySelectorAll('button').forEach(x => x.classList.toggle('on', x === b)); fDraw(); }); fSeg.appendChild(b); });
  $('fP').addEventListener('input', fDraw); $('fPre').addEventListener('change', fDraw);

  window.addEventListener('resize', () => { rDraw(); zDraw(); if (FKEYS.length) fDraw(); });
  rDraw(); zDraw(); if (FKEYS.length) fDraw();
})();
