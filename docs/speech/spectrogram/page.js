// Live demos for spectrogram.html: the STFT machine, wideband vs narrowband, dB and colour, and the labelled spectrogram.
// All spectrograms are computed here in the browser (radix-2 FFT) from the embedded ksp sentence (sentence.js).
(() => {
  const $ = id => document.getElementById(id);
  const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
  const D = window.SPEC_SENT;
  if (!D) return;
  const SR = D.sr, T0 = D.T0;

  // ---------------------------------------------------------------- KaTeX in readouts
  // \( \) maths is typeset by KaTeX's auto-render; before it has loaded, the page-level pass picks it up.
  const KATEX_OPT = { delimiters: [{ left: '\\(', right: '\\)', display: false }], throwOnError: false };
  function setMath(el, html) {
    el.innerHTML = html;
    if (window.renderMathInElement) renderMathInElement(el, KATEX_OPT);
  }
  // readout rows [labelHTML, valueText, divAttrs?, bAttrs?]: labels (which may hold maths) are rebuilt and typeset
  // only when they change; on every other call just the plain-text values in the <b> elements are updated
  function setReadout(el, rows) {
    const sig = rows.map(r => r[0] + '|' + (r[2] || '') + '|' + (r[3] || '')).join('\n');
    if (el.dataset.sig !== sig) {
      el.dataset.sig = sig;
      setMath(el, rows.map(r => `<div${r[2] ? ' ' + r[2] : ''}>${r[0]}<b${r[3] ? ' ' + r[3] : ''}></b></div>`).join(''));
    }
    const bs = el.querySelectorAll(':scope > div > b');
    rows.forEach((r, i) => { if (bs[i]) bs[i].textContent = r[1]; });
  }

  // ---------------------------------------------------------------- data
  const bin = atob(D.b64), N = bin.length >> 1, x = new Float32Array(N);
  for (let i = 0; i < N; i++) { let v = bin.charCodeAt(2 * i) | (bin.charCodeAt(2 * i + 1) << 8); if (v >= 32768) v -= 65536; x[i] = v / 32768; }
  const DUR = N / SR;
  const PH = D.phones.map(p => ({ s: p[0], e: p[1], arpa: p[2], ipa: p[3], cls: p[4], word: p[5] }));
  const CM = {};
  for (const k in D.cmaps) { const h = D.cmaps[k], a = []; for (let i = 0; i < h.length; i += 6) a.push([parseInt(h.substr(i, 2), 16), parseInt(h.substr(i + 2, 2), 16), parseInt(h.substr(i + 4, 2), 16)]); CM[k] = a; }
  function colour(name, v) {
    const lut = CM[name], f = Math.max(0, Math.min(1, v)) * (lut.length - 1), i = Math.min(lut.length - 2, Math.floor(f)), r = f - i;
    return [lut[i][0] + r * (lut[i + 1][0] - lut[i][0]), lut[i][1] + r * (lut[i + 1][1] - lut[i][1]), lut[i][2] + r * (lut[i + 1][2] - lut[i][2])];
  }

  // ---------------------------------------------------------------- helpers
  function setupCanvas(cv) {
    if (!cv.dataset.h) { cv.dataset.h = cv.getAttribute('height'); cv.style.height = cv.dataset.h + 'px'; }
    const dpr = window.devicePixelRatio || 1, W = cv.clientWidth, H = +cv.dataset.h;
    cv.width = W * dpr; cv.height = H * dpr; const c = cv.getContext('2d'); c.setTransform(dpr, 0, 0, dpr, 0, 0); c.clearRect(0, 0, W, H);
    return { c, W, H };
  }
  let actx = null, cur = null;
  function play(y, onstart) {
    try {
      actx = actx || new (window.AudioContext || window.webkitAudioContext)();
      if (actx.state === 'suspended') actx.resume();
      if (cur) { try { cur.stop(); } catch (e) {} }
      const buf = actx.createBuffer(1, y.length, SR); buf.getChannelData(0).set(y);
      const s = actx.createBufferSource(); s.buffer = buf; s.connect(actx.destination); s.start(); cur = s;
      if (onstart) onstart(actx.currentTime);
      return s;
    } catch (e) { console.warn(e); return null; }
  }
  function stopAudio() { if (cur) { try { cur.stop(); } catch (e) {} cur = null; } }
  function clip(a, b, fadeN = 80) {
    const i0 = Math.max(0, Math.round(a * SR)), i1 = Math.min(N, Math.round(b * SR)), y = x.slice(i0, i1), n = Math.min(fadeN, y.length >> 1);
    for (let i = 0; i < n; i++) { const g = i / n; y[i] *= g; y[y.length - 1 - i] *= g; }
    return y;
  }
  function fft(re, im) {
    const n = re.length;
    for (let i = 1, j = 0; i < n; i++) { let bit = n >> 1; for (; j & bit; bit >>= 1) j ^= bit; j ^= bit; if (i < j) { let t = re[i]; re[i] = re[j]; re[j] = t; t = im[i]; im[i] = im[j]; im[j] = t; } }
    for (let len = 2; len <= n; len <<= 1) {
      const a = -2 * Math.PI / len, wr = Math.cos(a), wi = Math.sin(a), h = len >> 1;
      for (let i = 0; i < n; i += len) {
        let cr = 1, ci = 0;
        for (let k = 0; k < h; k++) {
          const p = i + k, q = p + h, br = re[q] * cr - im[q] * ci, bi = re[q] * ci + im[q] * cr;
          re[q] = re[p] - br; im[q] = im[p] - bi; re[p] += br; im[p] += bi;
          const t = cr * wr - ci * wi; ci = cr * wi + ci * wr; cr = t;
        }
      }
    }
  }
  const pow2 = n => { let p = 1; while (p < n) p <<= 1; return p; };
  function win(kind, L) {
    const w = new Float64Array(L);
    for (let n = 0; n < L; n++) w[n] = kind === 'rect' ? 1 : kind === 'hamming' ? 0.54 - 0.46 * Math.cos(2 * Math.PI * n / L) : 0.5 - 0.5 * Math.cos(2 * Math.PI * n / L);
    return w;
  }
  // magnitude spectrum of the frame centred at sample c (zeros outside the signal)
  function frameSpec(c, L, Nf, w, re, im) {
    re.fill(0); im.fill(0);
    const a = c - (L >> 1);
    for (let n = 0; n < L; n++) { const i = a + n; if (i >= 0 && i < N) re[n] = x[i] * w[n]; }
    fft(re, im);
  }
  // spectrogram over [t0, t1]: frames centred at m*H, magnitudes for bins 0..K-1
  const cache = new Map();
  function spectrogram(Lms, kind, Hs, t0 = 0, t1 = DUR) {
    const L = Math.max(16, Math.round(Lms * SR / 1000)), Nf = Math.max(256, pow2(L)), K = Nf / 2 + 1;
    const key = [L, kind, Hs, t0, t1].join('|');
    if (cache.has(key)) return cache.get(key);
    const w = win(kind, L), re = new Float64Array(Nf), im = new Float64Array(Nf);
    const m0 = Math.max(0, Math.floor(t0 * SR / Hs)), m1 = Math.min(Math.floor(N / Hs), Math.ceil(t1 * SR / Hs));
    const nfr = m1 - m0 + 1, mag = new Float32Array(nfr * K);
    let mx = 1e-12;
    for (let m = 0; m < nfr; m++) {
      frameSpec((m0 + m) * Hs, L, Nf, w, re, im);
      for (let k = 0; k < K; k++) { const v = Math.hypot(re[k], im[k]); mag[m * K + k] = v; if (v > mx) mx = v; }
    }
    const S = { L, Nf, K, H: Hs, m0, nfr, mag, max: mx, kind };
    if (cache.size > 40) cache.clear();
    cache.set(key, S);
    return S;
  }
  // global reference per (L, window): the loudest bin of the whole sentence, so colours mean the same thing in every view
  const refCache = new Map();
  function globalRef(Lms, kind) {
    const k = Lms + kind;
    if (!refCache.has(k)) refCache.set(k, spectrogram(Lms, kind, 64).max);
    return refCache.get(k);
  }

  // draw a spectrogram into a plot rectangle; returns the geometry for overlays
  function drawSg(c, W, H, S, o) {
    const M = { l: 40, r: o.cbar ? 62 : 10, t: 8, b: 24 + (o.extraB || 0) };
    const pw = Math.max(10, W - M.l - M.r), ph = H - M.t - M.b;
    const t0 = o.t0 ?? 0, t1 = o.t1 ?? DUR, fmax = o.fmax ?? 8000;
    const kmax = Math.min(S.K - 1, Math.round(fmax / SR * S.Nf));
    const fa = Math.max(0, Math.floor(t0 * SR / S.H) - S.m0), fb = Math.min(S.nfr - 1, Math.ceil(t1 * SR / S.H) - S.m0);
    const cols = fb - fa + 1, rows = kmax + 1;
    const off = document.createElement('canvas'); off.width = cols; off.height = rows;
    const oc = off.getContext('2d'), img = oc.createImageData(cols, rows), d = img.data;
    const ref = o.ref || S.max, rng = o.range || 80, lin = o.scale === 'lin', cm = o.cmap || 'magma';
    const upto = o.upto ?? Infinity;
    const bg = cssRgb(css('--surface'));
    for (let j = 0; j < cols; j++) {
      const m = fa + j, tc = (S.m0 + m) * S.H / SR, painted = tc <= upto;
      for (let k = 0; k <= kmax; k++) {
        const p = 4 * ((rows - 1 - k) * cols + j);
        let rgb;
        if (!painted) rgb = bg;
        else {
          const a = S.mag[m * S.K + k] / ref;
          const v = lin ? a : 1 + 20 * Math.log10(a + 1e-12) / rng;
          rgb = colour(cm, v);
        }
        d[p] = rgb[0]; d[p + 1] = rgb[1]; d[p + 2] = rgb[2]; d[p + 3] = 255;
      }
    }
    oc.putImageData(img, 0, 0);
    c.imageSmoothingEnabled = true;
    // map the image so frame centres and bin centres land on their true time and frequency
    const tA = (S.m0 + fa) * S.H / SR, tB = (S.m0 + fb) * S.H / SR, dtc = S.H / SR;
    const X = t => M.l + (t - t0) / (t1 - t0) * pw, Y = f => M.t + ph * (1 - f / fmax);
    c.save(); c.beginPath(); c.rect(M.l, M.t, pw, ph); c.clip();
    const df = SR / S.Nf;
    c.drawImage(off, X(tA - dtc / 2), Y((kmax + 0.5) * df), X(tB + dtc / 2) - X(tA - dtc / 2), Y(-0.5 * df) - Y((kmax + 0.5) * df));
    c.restore();
    c.strokeStyle = css('--line-strong'); c.lineWidth = 1; c.strokeRect(M.l + 0.5, M.t + 0.5, pw - 1, ph - 1);
    // axes
    c.fillStyle = css('--muted'); c.font = '10.5px Inter, sans-serif'; c.textAlign = 'right';
    const fstep = fmax > 4000 ? 2000 : 1000;
    for (let f = 0; f <= fmax; f += fstep) c.fillText(`${f / 1000}`, M.l - 5, Y(f) + 3.5);
    c.save(); c.translate(11, M.t + ph / 2); c.rotate(-Math.PI / 2); c.textAlign = 'center'; c.fillText('kHz', 0, 0); c.restore();
    c.textAlign = 'center';
    const span = t1 - t0, ts = span > 1.5 ? 0.5 : span > 0.6 ? 0.1 : 0.05;
    for (let t = Math.ceil(t0 / ts) * ts; t <= t1 + 1e-9; t += ts) c.fillText(t.toFixed(ts < 0.1 ? 2 : 1), X(t), M.t + ph + 14);
    if (!o.noTimeLabel) { c.textAlign = 'right'; c.fillText('time (s)', M.l + pw, M.t + ph + 24 + (o.extraB || 0) - 2); }
    if (o.cbar) drawCbar(c, M.l + pw + 10, M.t, 12, ph, o);
    return { X, Y, M, pw, ph, t0, t1 };
  }
  function cssRgb(s) {
    const m = s.match(/^#([0-9a-f]{6})$/i);
    if (m) return [parseInt(m[1].substr(0, 2), 16), parseInt(m[1].substr(2, 2), 16), parseInt(m[1].substr(4, 2), 16)];
    return [255, 255, 255];
  }
  function drawCbar(c, x0, y0, w, h, o) {
    const cm = o.cmap || 'magma';
    for (let i = 0; i < h; i++) { const rgb = colour(cm, 1 - i / (h - 1)); c.fillStyle = `rgb(${rgb[0] | 0},${rgb[1] | 0},${rgb[2] | 0})`; c.fillRect(x0, y0 + i, w, 1.2); }
    c.strokeStyle = css('--line-strong'); c.strokeRect(x0 + 0.5, y0 + 0.5, w - 1, h - 1);
    c.fillStyle = css('--muted'); c.font = '10px Inter, sans-serif'; c.textAlign = 'left';
    const lin = o.scale === 'lin', rng = o.range || 80;
    const ticks = lin ? [[1, '1.0'], [0.5, '0.5'], [0, '0']] : [0, 0.25, 0.5, 0.75, 1].map(v => [1 - v, `${Math.round(-v * rng)}`]);
    ticks.forEach(([v, lab]) => { const yy = y0 + (1 - v) * h; c.fillRect(x0 + w, yy - 0.5, 3, 1); c.fillText(lab, x0 + w + 5, yy + 3.5); });
    c.save(); c.translate(x0 + w + 40, y0 + h / 2); c.rotate(Math.PI / 2); c.textAlign = 'center';
    c.fillText(lin ? '|X| / max' : 'dB re max', 0, 0); c.restore();
  }
  function drawWave(c, W, H, t0, t1, col) {
    const a0 = Math.floor(t0 * SR), a1 = Math.ceil(t1 * SR);
    c.strokeStyle = col || css('--accent'); c.lineWidth = 1; c.beginPath();
    for (let px = 0; px < W; px++) {
      const a = a0 + Math.floor(px / W * (a1 - a0)), b = a0 + Math.floor((px + 1) / W * (a1 - a0)); let lo = 0, hi = 0;
      for (let i = a; i <= b && i < N; i++) { if (i < 0) continue; lo = Math.min(lo, x[i]); hi = Math.max(hi, x[i]); }
      c.moveTo(px + 0.5, H / 2 - hi * H / 2.1); c.lineTo(px + 0.5, H / 2 - lo * H / 2.1 + 0.5);
    }
    c.stroke();
  }
  function segButtons(sel, attr, cb) {
    const bs = document.querySelectorAll(`${sel} [${attr}]`);
    bs.forEach(b => b.addEventListener('click', () => { bs.forEach(o => o.classList.toggle('on', o === b)); cb(b.getAttribute(attr)); }));
  }
  const throttle = fn => { let q = false; return () => { if (q) return; q = true; requestAnimationFrame(() => { q = false; fn(); }); }; };

  // ---------------------------------------------------------------- 1. the STFT machine
  if ($('smWave')) {
    const st = { Lms: 25, kind: 'hann', t: 0.70, run: null };
    const HOPS = 32;                                   // 2 ms between painted columns
    function draw() {
      const L = Math.round(st.Lms * SR / 1000), w = win(st.kind, L), c0 = Math.round(st.t * SR);
      $('smTV').textContent = `${st.t.toFixed(3)} s`;
      // ① waveform with the window
      { const { c, W, H } = setupCanvas($('smWave'));
        drawWave(c, W, H, 0, DUR, css('--line-strong'));
        const X = t => t / DUR * W, a = (c0 - L / 2) / SR, b = (c0 + L / 2) / SR;
        c.fillStyle = css('--accent-2'); c.globalAlpha = 0.14; c.fillRect(X(a), 0, Math.max(1.5, X(b) - X(a)), H); c.globalAlpha = 1;
        c.strokeStyle = css('--accent-2'); c.lineWidth = 1.6; c.beginPath();
        for (let n = 0; n < L; n += Math.max(1, L >> 6)) { const px = X(a + n / SR), py = H - 3 - w[n] * (H - 8); n ? c.lineTo(px, py) : c.moveTo(px, py); }
        c.lineTo(X(b), H - 3 - w[L - 1] * (H - 8)); c.stroke();
        c.fillStyle = css('--muted'); c.font = '10.5px Inter, sans-serif'; c.textAlign = 'left';
        const p = PH.find(q => st.t >= q.s && st.t < q.e);
        c.fillText(p && p.ipa ? `/${p.ipa}/ in “${p.word}”` : 'pause', 6, 13); }
      // ② the windowed frame
      const fr = new Float64Array(L), raw = new Float64Array(L); let pk = 1e-6;
      for (let n = 0; n < L; n++) { const i = c0 - (L >> 1) + n; raw[n] = i >= 0 && i < N ? x[i] : 0; fr[n] = raw[n] * w[n]; pk = Math.max(pk, Math.abs(raw[n])); }
      { const { c, W, H } = setupCanvas($('smFrame'));
        const X = n => 6 + n / (L - 1) * (W - 12), Y = v => H / 2 - v / pk * (H / 2 - 12);
        c.strokeStyle = css('--line'); c.beginPath(); c.moveTo(0, H / 2); c.lineTo(W, H / 2); c.stroke();
        [[raw, css('--line-strong'), 1], [fr, css('--accent-2'), 1.3]].forEach(([arr, col, lw]) => {
          c.strokeStyle = col; c.lineWidth = lw; c.beginPath(); for (let n = 0; n < L; n++) n ? c.lineTo(X(n), Y(arr[n])) : c.moveTo(X(n), Y(arr[n])); c.stroke(); });
        c.fillStyle = css('--muted'); c.font = '10.5px Inter, sans-serif'; c.textAlign = 'left';
        c.fillText(`${L} samples = ${st.Lms} ms (auto-scaled)`, 6, 12); }
      // ③ its spectrum in dB, against the loudest frame of the sentence
      const S = spectrogram(st.Lms, st.kind, HOPS), ref = globalRef(st.Lms, st.kind);
      const re = new Float64Array(S.Nf), im = new Float64Array(S.Nf);
      frameSpec(c0, L, S.Nf, w, re, im);
      { const { c, W, H } = setupCanvas($('smSpec'));
        const l = 34, r = 8, t = 8, b = 22, pw = W - l - r, ph = H - t - b;
        c.strokeStyle = css('--line'); c.fillStyle = css('--muted'); c.font = '10px Inter, sans-serif'; c.lineWidth = 1;
        c.textAlign = 'right'; for (let dB = 0; dB >= -80; dB -= 20) { const y = t + ph * (-dB / 80); c.beginPath(); c.moveTo(l, y); c.lineTo(l + pw, y); c.stroke(); c.fillText(`${dB}`, l - 4, y + 3.5); }
        c.textAlign = 'center'; for (let f = 0; f <= 8; f += 2) c.fillText(`${f}`, l + f / 8 * pw, H - 8);
        c.textAlign = 'right'; c.fillText('kHz', l + pw, H - 0.5);
        c.strokeStyle = css('--accent'); c.lineWidth = 1.2; c.beginPath();
        for (let k = 0; k < S.K; k++) { const v = Math.max(-80, 20 * Math.log10(Math.hypot(re[k], im[k]) / ref + 1e-12)); const px = l + k / (S.K - 1) * pw, py = t + ph * (-v / 80); k ? c.lineTo(px, py) : c.moveTo(px, py); }
        c.stroke(); }
      // ④ the spectrogram painted up to now
      { const { c, W, H } = setupCanvas($('smSg'));
        const g = drawSg(c, W, H, S, { ref, range: 80, upto: st.t, cmap: 'magma' });
        c.strokeStyle = css('--accent-2'); c.lineWidth = 1.5; c.beginPath(); c.moveTo(g.X(st.t), g.M.t); c.lineTo(g.X(st.t), g.M.t + g.ph); c.stroke(); }
      setReadout($('smRead'), [
        ['window \\(L\\)', `${st.Lms} ms = ${L} samples`],
        ['hop \\(H\\) (columns)', `2 ms = ${HOPS} samples`],
        ['FFT size \\(N\\)', `${S.Nf}`],
        ['bin spacing \\(f_s/N\\)', `${(SR / S.Nf).toFixed(1)} Hz`],
        ['resolution \\(\\approx f_s/L\\)', `${(SR / L).toFixed(0)} Hz`]]);
    }
    const redraw = throttle(draw);
    $('smT').max = Math.floor(DUR * 1000);
    $('smT').value = Math.round(st.t * 1000);
    $('smT').addEventListener('input', () => { stopRun(); st.t = +$('smT').value / 1000; redraw(); });
    segButtons('#smL', 'data-l', v => { st.Lms = +v; redraw(); });
    segButtons('#smW', 'data-w', v => { st.kind = v; redraw(); });
    function stopRun() { if (st.run) { cancelAnimationFrame(st.run); st.run = null; stopAudio(); $('smRun').textContent = '▶ run the machine (with sound)'; } }
    $('smRun').addEventListener('click', () => {
      if (st.run) { stopRun(); return; }
      const from = st.t > DUR - 0.05 ? 0 : st.t;
      let t0a = null;
      play(clip(from, DUR, 40), now => { t0a = now; });
      $('smRun').textContent = '■ stop';
      const step = () => {
        const now = actx ? actx.currentTime : 0;
        st.t = Math.min(DUR, from + (t0a === null ? 0 : now - t0a));
        $('smT').value = Math.round(st.t * 1000); draw();
        if (st.t < DUR) st.run = requestAnimationFrame(step); else { st.run = null; $('smRun').textContent = '▶ run the machine (with sound)'; }
      };
      st.run = requestAnimationFrame(step);
    });
    $('smPlay').addEventListener('click', () => { stopRun(); play(clip(st.t - 0.15, st.t + 0.15)); });
    draw(); window.addEventListener('resize', redraw);
  }

  // ---------------------------------------------------------------- 2. wideband vs narrowband
  if ($('wnCv')) {
    const st = { Lms: 4, view: 'zoom', fmax: 4000 };
    const VIEWS = { zoom: [1.56, 1.88], whole: [0, DUR] };
    function draw() {
      $('wnLV').textContent = `${st.Lms} ms`;
      const [t0, t1] = VIEWS[st.view], H = st.view === 'zoom' ? 16 : 32;
      const S = spectrogram(st.Lms, 'hann', H, Math.max(0, t0 - 0.05), Math.min(DUR, t1 + 0.05));
      const { c, W, H: Hc } = setupCanvas($('wnCv'));
      drawSg(c, W, Hc, S, { t0, t1, fmax: st.fmax, ref: globalRef(st.Lms, 'hann'), range: 70, cmap: 'magma', cbar: false });
      const L = S.L, ppw = (L / SR) / T0;
      const kind = ppw < 1 ? 'wideband: shorter than one pitch period → vertical striations, formant bands'
        : ppw >= 2 ? 'narrowband: several periods → horizontal harmonics' : 'in between: neither striations nor harmonics are crisp';
      setReadout($('wnRead'), [
        ['window \\(L\\)', `${st.Lms} ms = ${L} samples`],
        ['\\(\\Delta f \\approx f_s/L\\)', `${(SR / L).toFixed(0)} Hz`],
        [`pitch periods per window (\\(T_0 = ${(T0 * 1000).toFixed(1)}\\) ms)`, ppw.toFixed(1)],
        ['\\(\\Delta t \\cdot \\Delta f \\approx (L/f_s)(f_s/L)\\)', '1'],
        ['what you should see', kind, 'style="flex-basis:100%"', 'style="font-family:Inter,sans-serif;font-size:0.85rem;"']]);
    }
    const redraw = throttle(draw);
    $('wnL').addEventListener('input', () => { st.Lms = +$('wnL').value; document.querySelectorAll('#wnPre [data-l]').forEach(b => b.classList.toggle('on', +b.dataset.l === st.Lms)); redraw(); });
    document.querySelectorAll('#wnPre [data-l]').forEach(b => b.addEventListener('click', () => { st.Lms = +b.dataset.l; $('wnL').value = st.Lms; document.querySelectorAll('#wnPre [data-l]').forEach(o => o.classList.toggle('on', o === b)); redraw(); }));
    segButtons('#wnView', 'data-v', v => { st.view = v; redraw(); });
    segButtons('#wnF', 'data-f', v => { st.fmax = +v; redraw(); });
    $('wnPlay').addEventListener('click', () => { const [a, b] = VIEWS[st.view]; play(clip(a, b)); });
    draw(); window.addEventListener('resize', redraw);
  }

  // ---------------------------------------------------------------- 3. dB, range and colour
  if ($('colCv')) {
    const st = { scale: 'db', range: 80, cmap: 'magma' };
    function draw() {
      $('colRV').textContent = st.scale === 'db' ? `${st.range} dB` : '(not used on a linear scale)';
      $('colR').disabled = st.scale !== 'db';
      const S = spectrogram(10, 'hann', 32);
      const { c, W, H } = setupCanvas($('colCv'));
      drawSg(c, W, H, S, { scale: st.scale, range: st.range, cmap: st.cmap, cbar: true, fmax: 8000 });
      // share of the time-frequency plane that is above the floor (dB) or above 10 % of the peak (linear)
      let above = 0; const kmax = S.K - 1;
      for (let m = 0; m < S.nfr; m++) for (let k = 0; k <= kmax; k++) { const a = S.mag[m * S.K + k] / S.max; if (st.scale === 'db' ? 20 * Math.log10(a + 1e-12) > -st.range : a > 0.1) above++; }
      setMath($('colRead'), `<div>scale<b>${st.scale === 'db' ? '\\(20\\log_{10}|X|\\) (dB)' : '\\(|X|\\) (linear)'}</b></div>` +
        `<div>${st.scale === 'db' ? 'cells above the floor' : 'cells above 10 % of the peak'}<b>${(100 * above / (S.nfr * (kmax + 1))).toFixed(1)} %</b></div>` +
        `<div>colour map<b>${st.cmap}</b></div>`);
    }
    const redraw = throttle(draw);
    segButtons('#colS', 'data-s', v => { st.scale = v; redraw(); });
    segButtons('#colM', 'data-m', v => { st.cmap = v; redraw(); });
    $('colR').addEventListener('input', () => { st.range = +$('colR').value; redraw(); });
    draw(); window.addEventListener('resize', redraw);
  }

  // ---------------------------------------------------------------- 4. reading the labelled spectrogram
  if ($('phCv')) {
    const CLS = {
      vowel: ['--accent', 'vowel: voiced, open tract. Look for dark formant bands (F1, F2, F3) and, in a wideband view, vertical glottal striations.'],
      diphthong: ['--teal', 'diphthong: a vowel that moves. The formant bands glide from one vowel position to another.'],
      fricative: ['--accent-2', 'fricative: turbulent noise through a narrow gap. Look for a block of random-looking energy, high for /s/, lower for /ʃ/, weak and flat for /f, θ/.'],
      stop: ['--accent-3', 'stop: a closure (near silence), a sharp burst, then aspiration or voicing (voice onset time).'],
      affricate: ['--accent-3', 'affricate: a stop released into a fricative.'],
      nasal: ['--nasal', 'nasal: air through the nose. Strong low-frequency murmur, weak upper formants, and abrupt edges at the oral closure and release.'],
      approximant: ['--approx', 'approximant: vowel-like, with formants that move; /ɹ/ lowers F3, /l/ shows a gap around F2, /w/ low F2, /j/ high F2.'],
    };
    const st = { hover: -1, sel: -1 };
    let geom = null;
    const SP = () => spectrogram(6, 'hann', 16);
    function draw() {
      const S = SP();
      const { c, W, H } = setupCanvas($('phCv'));
      const g = drawSg(c, W, H, S, { cmap: 'grey', range: 70, fmax: 8000, extraB: 44, noTimeLabel: true });
      geom = g;
      const yP = g.M.t + g.ph + 20, yW = yP + 26;
      c.font = '600 13px Inter, sans-serif'; c.textAlign = 'center';
      PH.forEach((p, i) => {
        if (!p.ipa) return;
        const a = g.X(p.s), b = g.X(p.e), col = css(CLS[p.cls][0]);
        c.fillStyle = col; c.globalAlpha = i === st.hover || i === st.sel ? 0.35 : 0.13; c.fillRect(a + 0.5, yP - 2, b - a - 1, 20); c.globalAlpha = 1;
        if (i === st.hover || i === st.sel) { c.globalAlpha = 0.16; c.fillRect(a, g.M.t, b - a, g.ph); c.globalAlpha = 1; c.strokeStyle = col; c.lineWidth = 1.5; c.strokeRect(a, g.M.t, b - a, g.ph); }
        c.fillStyle = col; if (b - a > 7) c.fillText(p.ipa, (a + b) / 2, yP + 13);
      });
      c.font = 'italic 12px Inter, sans-serif'; c.fillStyle = css('--ink');
      D.words.forEach(([s, e, w]) => { const a = g.X(s), b = g.X(e); c.fillText(w, (a + b) / 2, yW + 12); c.fillRect(a + 2, yW - 1, b - a - 4, 1); });
      const i = st.hover >= 0 ? st.hover : st.sel;
      if (i >= 0) {
        const p = PH[i];
        $('phRead').innerHTML = `<div>phone<b>/${p.ipa}/ &nbsp;(${p.arpa})</b></div><div>word<b>${p.word}</b></div><div>time<b>${p.s.toFixed(3)}–${p.e.toFixed(3)} s</b></div>` +
          `<div>duration<b>${Math.round((p.e - p.s) * 1000)} ms</b></div><div style="flex-basis:100%">${CLS[p.cls][1].replace(/^(\w+):/, '<b style="display:inline;font-family:Inter,sans-serif;color:' + css(CLS[p.cls][0]) + '">$1</b>:')}</div>`;
      }
    }
    const redraw = throttle(draw);
    function idxAt(ev) {
      if (!geom) return -1;
      const r = $('phCv').getBoundingClientRect(), px = ev.clientX - r.left;
      const t = geom.t0 + (px - geom.M.l) / geom.pw * (geom.t1 - geom.t0);
      return PH.findIndex(p => p.ipa && t >= p.s && t < p.e);
    }
    $('phCv').addEventListener('mousemove', ev => { const i = idxAt(ev); if (i !== st.hover) { st.hover = i; redraw(); } });
    $('phCv').addEventListener('mouseleave', () => { st.hover = -1; redraw(); });
    $('phCv').addEventListener('click', ev => {
      const i = idxAt(ev); if (i < 0) return;
      st.sel = i; redraw();
      const p = PH[i], wd = document.querySelector('#phMode .on').dataset.p === 'word';
      if (wd) { const w = D.words.find(q => p.s >= q[0] - 1e-6 && p.e <= q[1] + 1e-6); if (w) { play(clip(w[0] - 0.01, w[1] + 0.01, 60)); return; } }
      play(clip(p.s - 0.005, p.e + 0.005, 40));
    });
    segButtons('#phMode', 'data-p', () => {});
    $('phAll').addEventListener('click', () => play(clip(0, DUR, 40)));
    draw(); window.addEventListener('resize', redraw);
    if (window.matchMedia) { const mq = window.matchMedia('(prefers-color-scheme: dark)'); if (mq.addEventListener) mq.addEventListener('change', () => { cache.clear(); window.dispatchEvent(new Event('resize')); }); }
  }
  // ---------------------------------------------------------------- 5. Griffin-Lim, live
  // Keep |STFT| of the sentence, invent the phase, and iterate: X = S e^{j phi} -> y = ISTFT(X) -> Z = STFT(y) -> phi = angle(Z).
  if ($('glSpec')) {
    const L = 512, Hh = 128, K = L / 2 + 1, w = win('hann', L);     // periodic Hann, 75 % overlap
    const M = Math.floor(N / Hh) + 1;                               // frames centred at m*H (as in the other demos)
    const re = new Float64Array(L), im = new Float64Array(L);
    const den = new Float64Array(N);
    for (let m = 0; m < M; m++) { const a = m * Hh - L / 2; for (let n = 0; n < L; n++) { const i = a + n; if (i >= 0 && i < N) den[i] += w[n] * w[n]; } }
    function stftInto(sig, Zr, Zi) {
      for (let m = 0; m < M; m++) {
        re.fill(0); im.fill(0);
        const a = m * Hh - L / 2;
        for (let n = 0; n < L; n++) { const i = a + n; if (i >= 0 && i < N) re[n] = sig[i] * w[n]; }
        fft(re, im);
        for (let k = 0; k < K; k++) { Zr[m * K + k] = re[k]; Zi[m * K + k] = im[k]; }
      }
    }
    // least-squares inverse STFT (Griffin & Lim): sum_m w * IDFT(X_m), divided by sum_m w^2
    function istft(Xr, Xi, y) {
      y.fill(0);
      for (let m = 0; m < M; m++) {
        for (let k = 0; k < K; k++) { re[k] = Xr[m * K + k]; im[k] = -Xi[m * K + k]; }      // conj, so a forward FFT inverts
        for (let k = K; k < L; k++) { re[k] = Xr[m * K + L - k]; im[k] = Xi[m * K + L - k]; } // conj of the mirrored bins
        fft(re, im);
        const a = m * Hh - L / 2;
        for (let n = 0; n < L; n++) { const i = a + n; if (i >= 0 && i < N) y[i] += w[n] * re[n] / L; }
      }
      for (let i = 0; i < N; i++) y[i] = den[i] > 1e-9 ? y[i] / den[i] : 0;
    }
    // target magnitudes
    const Zr = new Float64Array(M * K), Zi = new Float64Array(M * K);
    stftInto(x, Zr, Zi);
    const S = new Float64Array(M * K); let Snorm = 0, Smax = 1e-12;
    for (let j = 0; j < S.length; j++) { S[j] = Math.hypot(Zr[j], Zi[j]); Snorm += S[j] * S[j]; if (S[j] > Smax) Smax = S[j]; }
    Snorm = Math.sqrt(Snorm);
    // state
    const Pr = new Float64Array(M * K), Pi = new Float64Array(M * K);       // current phase as unit phasors
    const Xr = new Float64Array(M * K), Xi = new Float64Array(M * K);
    const Qr = new Float64Array(M * K), Qi = new Float64Array(M * K);       // previous Z (for momentum)
    const y = new Float64Array(N), mag = new Float32Array(M * K);
    let it = 0, hist = [], y0 = null, start = 'rand', seed = 20240301, running = false, raf = null;
    const rnd = () => ((seed = (seed * 1664525 + 1013904223) >>> 0) / 4294967296);
    const ALPHA = 0.99;

    function synthAndAnalyse() {                       // steps 2-4, then the error
      for (let j = 0; j < S.length; j++) { Xr[j] = S[j] * Pr[j]; Xi[j] = S[j] * Pi[j]; }
      istft(Xr, Xi, y);
      stftInto(y, Zr, Zi);
      let e = 0;
      for (let j = 0; j < S.length; j++) { const a = Math.hypot(Zr[j], Zi[j]); mag[j] = a; const d = S[j] - a; e += d * d; }
      return Math.sqrt(e) / Snorm;
    }
    function reset() {
      for (let j = 0; j < S.length; j++) {
        if (start === 'zero') { Pr[j] = 1; Pi[j] = 0; }
        else { const t = 2 * Math.PI * rnd(); Pr[j] = Math.cos(t); Pi[j] = Math.sin(t); }
      }
      it = 0; hist = [synthAndAnalyse()]; y0 = Float32Array.from(y);
      Qr.set(Zr); Qi.set(Zi);
      draw();
    }
    function step() {                                   // step 5 (+ optional momentum), then steps 2-4
      const fast = $('glFast').checked;
      for (let j = 0; j < S.length; j++) {
        let a = Zr[j], b = Zi[j];
        if (fast) { a = Zr[j] + ALPHA * (Zr[j] - Qr[j]); b = Zi[j] + ALPHA * (Zi[j] - Qi[j]); }
        Qr[j] = Zr[j]; Qi[j] = Zi[j];
        const r = Math.hypot(a, b);
        if (r > 1e-12) { Pr[j] = a / r; Pi[j] = b / r; }
      }
      it++; hist.push(synthAndAnalyse());
    }

    // ---- drawing
    const SG = { L, Nf: L, K, H: Hh, m0: 0, nfr: M, mag, max: Smax };
    const ST = { L, Nf: L, K, H: Hh, m0: 0, nfr: M, mag: Float32Array.from(S), max: Smax };
    const opts = { ref: Smax, range: 70, cmap: 'magma', fmax: 8000 };
    const vowel = PH.find(p => p.arpa === 'ae' && /cash/i.test(p.word || '')) || PH.find(p => p.cls === 'vowel');
    const wc = vowel ? Math.round((vowel.s + vowel.e) / 2 * SR) : Math.round(N / 2), wn = Math.round(0.030 * SR);
    function drawTarget() { const { c, W, H } = setupCanvas($('glTarget')); drawSg(c, W, H, ST, opts); }
    function drawCurve() {
      const { c, W, H } = setupCanvas($('glCurve'));
      const ml = 40, mr = 10, mt = 8, mb = 24, pw = W - ml - mr, ph = H - mt - mb;
      const n = Math.max(50, Math.ceil((hist.length - 1) / 50) * 50), lo = Math.log10(0.01), hi = Math.log10(1);
      const X = i => ml + i / n * pw, Y = v => mt + ph * (1 - (Math.log10(Math.max(v, 0.01)) - lo) / (hi - lo));
      c.strokeStyle = css('--line'); c.lineWidth = 1; c.fillStyle = css('--muted'); c.font = '10.5px Inter, sans-serif'; c.textAlign = 'right';
      [1, 0.3, 0.1, 0.03, 0.01].forEach(v => { c.beginPath(); c.moveTo(ml, Y(v)); c.lineTo(ml + pw, Y(v)); c.stroke(); c.fillText(String(v), ml - 5, Y(v) + 3.5); });
      c.textAlign = 'center';
      for (let i = 0; i <= n; i += n / 5) c.fillText(String(i), X(i), mt + ph + 14);
      c.textAlign = 'right'; c.fillText('iteration', ml + pw, mt + ph + 23);
      c.strokeStyle = css('--accent-2'); c.lineWidth = 2; c.beginPath();
      hist.forEach((v, i) => { i ? c.lineTo(X(i), Y(v)) : c.moveTo(X(i), Y(v)); }); c.stroke();
      const last = hist[hist.length - 1];
      c.fillStyle = css('--accent-2'); c.beginPath(); c.arc(X(hist.length - 1), Y(last), 3.5, 0, 2 * Math.PI); c.fill();
    }
    function drawWaveCmp() {
      const { c, W, H } = setupCanvas($('glWave'));
      const a = wc - (wn >> 1); let mx = 1e-9;
      for (let n = 0; n < wn; n++) mx = Math.max(mx, Math.abs(x[a + n] || 0), Math.abs(y[a + n] || 0));
      const line = (sig, col, lw) => { c.strokeStyle = col; c.lineWidth = lw; c.beginPath(); for (let n = 0; n < wn; n++) { const px = n / (wn - 1) * W, py = H / 2 - (sig[a + n] || 0) / mx * H * 0.45; n ? c.lineTo(px, py) : c.moveTo(px, py); } c.stroke(); };
      c.strokeStyle = css('--line'); c.beginPath(); c.moveTo(0, H / 2); c.lineTo(W, H / 2); c.stroke();
      line(x, css('--line-strong'), 2.2); line(y, css('--accent'), 1.4);
    }
    function draw() {
      const { c, W, H } = setupCanvas($('glSpec')); drawSg(c, W, H, SG, opts);
      drawCurve(); drawWaveCmp();
      const sc = hist[hist.length - 1];
      $('glRead').innerHTML = `<div>iteration<b>${it}</b></div><div>spectral convergence<b>${sc.toFixed(3)}</b></div>` +
        `<div>in dB<b>${(20 * Math.log10(sc)).toFixed(1)} dB</b></div>` +
        `<div>improvement since start<b>${(20 * Math.log10(hist[0] / sc)).toFixed(1)} dB</b></div>` +
        `<div>mode<b class="txt">${$('glFast').checked ? 'fast (momentum 0.99)' : 'plain Griffin–Lim'}</b></div>`;
    }
    function loop() {
      if (!running) return;
      const t0 = performance.now();
      do { step(); } while (performance.now() - t0 < 12 && it < 400);   // a few rounds per frame on fast machines
      draw();
      if (it >= 400) { setRun(false); return; }
      raf = requestAnimationFrame(loop);
    }
    function setRun(on) {
      running = on;
      $('glRun').textContent = on ? '❚❚ Pause' : '▶ Run';
      $('glRun').classList.toggle('on', !on);
      if (on) { cancelAnimationFrame(raf); raf = requestAnimationFrame(loop); }
    }
    function playSig(sig) {
      let ex = 0, ey = 0; for (let i = 0; i < N; i++) { ex += x[i] * x[i]; ey += sig[i] * sig[i]; }
      const g = Math.sqrt(ex / Math.max(ey, 1e-12));
      const out = new Float32Array(N); let pk = 0;
      for (let i = 0; i < N; i++) { out[i] = sig[i] * g; pk = Math.max(pk, Math.abs(out[i])); }
      if (pk > 0.98) for (let i = 0; i < N; i++) out[i] *= 0.98 / pk;
      play(out);
    }
    $('glRun').addEventListener('click', () => setRun(!running));
    $('glStep').addEventListener('click', () => { setRun(false); step(); draw(); });
    $('glReset').addEventListener('click', () => { setRun(false); seed = (seed + 7919) >>> 0; reset(); });
    segButtons('#glDemo', 'data-gls', v => { start = v; setRun(false); reset(); });
    $('glFast').addEventListener('change', draw);
    $('glPlay').addEventListener('click', () => playSig(y));
    $('glPlay0').addEventListener('click', () => playSig(y0));
    $('glPlayOrig').addEventListener('click', () => playSig(x));
    reset(); drawTarget();
    window.addEventListener('resize', throttle(() => { drawTarget(); draw(); }));
  }
})();
