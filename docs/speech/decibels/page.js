// Live demos for decibels.html: dB converter, loudness levels and staircases, spectrum explorer, SNR listening, adding sources.
(() => {
  const $ = id => document.getElementById(id);
  const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
  const SR = 16000;
  const VOWEL_T = 0.69, FRIC_T = 1.36;            // frames picked acoustically by make_figures.py (numbers.json)

  function setupCanvas(cv) {
    if (!cv.dataset.h) { cv.dataset.h = cv.getAttribute('height'); cv.style.height = cv.dataset.h + 'px'; }
    const dpr = window.devicePixelRatio || 1, W = cv.clientWidth, H = +cv.dataset.h;
    cv.width = W * dpr; cv.height = H * dpr; const c = cv.getContext('2d'); c.setTransform(dpr, 0, 0, dpr, 0, 0); c.clearRect(0, 0, W, H);
    return { c, W, H };
  }
  // typeset \( \) maths written into the DOM after KaTeX's auto-render has loaded (it renders the first pass itself)
  const tex = el => { if (window.renderMathInElement) window.renderMathInElement(el, { delimiters: [{ left: '\\(', right: '\\)', display: false }], throwOnError: false }); };
  // a ratio as KaTeX source (no delimiters): 1.23, 1{,}000 or 3.98 \times 10^{6}
  const fmt = v => {
    if (v >= 1e6 || v < 1e-3) { const e = Math.floor(Math.log10(v)), m = v / Math.pow(10, e); return `${m.toFixed(2)} \\times 10^{${e}}`; }
    return v >= 100 ? Math.round(v).toLocaleString('en-US').replace(/,/g, '{,}') : v >= 1 ? v.toFixed(2) : v.toFixed(4);
  };

  // ---------------------------------------------------------------- 1. converter
  function conv() {
    const d = +$('cvDb').value, pr = Math.pow(10, d / 10), ar = Math.pow(10, d / 20);
    $('cvDbV').textContent = `${d > 0 ? '+' : ''}${d} dB`;
    $('cvRead').innerHTML = `<div>power ratio<b>\\(\\times\\, ${fmt(pr)}\\)</b></div><div>amplitude ratio<b>\\(\\times\\, ${fmt(ar)}\\)</b></div>` +
      `<div>check: \\(10\\log_{10}(\\text{power})\\)<b>${(10 * Math.log10(pr)).toFixed(1)} dB</b></div><div>check: \\(20\\log_{10}(\\text{amplitude})\\)<b>${(20 * Math.log10(ar)).toFixed(1)} dB</b></div>`;
    tex($('cvRead'));
    const { c, W, H } = setupCanvas($('cvBars'));
    const ref = Math.min(W * 0.12, 70), x0 = 10;
    c.fillStyle = css('--line-strong'); c.fillRect(x0, 14, ref, 22);
    const len = ref * ar, show = Math.min(len, W - 2 * x0);
    c.fillStyle = css('--accent'); c.fillRect(x0, 52, Math.max(show, 0.6), 22);
    if (len > W - 2 * x0) { c.fillStyle = css('--accent-2'); c.font = '12px Inter, sans-serif'; c.textAlign = 'right'; c.fillText(`→ continues for ${(len / ref).toFixed(0)} reference lengths`, W - x0, 47); }
    c.fillStyle = css('--muted'); c.font = '11px Inter, sans-serif'; c.textAlign = 'left';
    c.fillText('reference', x0 + ref + 6, 30);
  }
  if ($('cvDb')) { $('cvDb').addEventListener('input', conv); conv(); window.addEventListener('resize', conv); }

  // ---------------------------------------------------------------- speech + audio
  if (!window.CODEC_SPEECH) return;
  const bin = atob(window.CODEC_SPEECH.b64), N = bin.length >> 1, x = new Float32Array(N);
  for (let i = 0; i < N; i++) { let v = bin.charCodeAt(2 * i) | (bin.charCodeAt(2 * i + 1) << 8); if (v >= 32768) v -= 65536; x[i] = v / 32768; }
  let P = 0; for (let i = 0; i < N; i++) P += x[i] * x[i]; P /= N;

  let actx = null, cur = null;
  function play(y) {
    try {
      actx = actx || new (window.AudioContext || window.webkitAudioContext)();
      if (actx.state === 'suspended') actx.resume();
      if (cur) { try { cur.stop(); } catch (e) {} }
      const buf = actx.createBuffer(1, y.length, SR); buf.getChannelData(0).set(y.map(v => Math.max(-1, Math.min(1, v))));
      const s = actx.createBufferSource(); s.buffer = buf; s.connect(actx.destination); s.start(); cur = s;
    } catch (e) { console.warn(e); }
  }
  function loudest(len) {
    let best = -1, at = 0, e = 0;
    for (let i = 0; i < len; i++) e += x[i] * x[i];
    for (let i = 0; i + len < N; i++) { if (e > best) { best = e; at = i; } e += x[i + len] * x[i + len] - x[i] * x[i]; }
    return at;
  }
  const fade = (y, n = 160) => { for (let i = 0; i < n; i++) { const g = i / n; y[i] *= g; y[y.length - 1 - i] *= g; } return y; };

  // ---------------------------------------------------------------- 2. loudness
  const P0 = loudest(SR * 1.2), phrase = fade(x.slice(P0, P0 + SR * 1.2));
  if ($('earBtns')) {
    [0, -3, -6, -10, -20, -30, -40].forEach(d => {
      const b = document.createElement('button'); b.textContent = `▶ ${d === 0 ? '0 dB (as recorded)' : d + ' dB'}`;
      b.addEventListener('click', () => { const g = Math.pow(10, d / 20); play(phrase.map(v => v * g)); });
      $('earBtns').appendChild(b);
    });
  }
  const B0 = loudest(Math.round(SR * 0.45)), burst = fade(x.slice(B0, B0 + Math.round(SR * 0.45)));
  const STAIRS = { lin: [0.2, 0.4, 0.6, 0.8, 1.0], db: [1 / 16, 1 / 8, 1 / 4, 1 / 2, 1] };
  let active = null, timer = null;
  function drawStairs(which, k) {
    const { c, W, H } = setupCanvas($('stairCv'));
    const groups = [['lin', 'linear: +0.2 each time', css('--accent-4')], ['db', 'decibel: ×2 (+6 dB) each time', css('--accent')]];
    const gw = W / 2, bw = Math.min(34, gw / 7), base = H - 22, top = 26;
    groups.forEach(([key, label, col], gi) => {
      const g0 = gi * gw + (gw - 5 * bw * 1.5) / 2;
      STAIRS[key].forEach((a, i) => {
        const h = (base - top) * a, x0 = g0 + i * bw * 1.5;
        c.fillStyle = (which === key && i === k) ? css('--accent-2') : col;
        c.fillRect(x0, base - h, bw, h);
        if (i > 0) { c.fillStyle = css('--ink'); c.font = '10.5px JetBrains Mono, monospace'; c.textAlign = 'center';
          c.fillText(`+${(20 * Math.log10(a / STAIRS[key][i - 1])).toFixed(1)}`, x0 + bw / 2, base - h - 4); }
      });
      c.fillStyle = css('--muted'); c.font = '11.5px Inter, sans-serif'; c.textAlign = 'center';
      c.fillText(label, gi * gw + gw / 2, H - 6);
    });
  }
  function runStairs(key) {
    clearTimeout(timer);
    const gap = Math.round(SR * 0.22), L = burst.length, y = new Float32Array(5 * (L + gap));
    STAIRS[key].forEach((a, i) => { for (let n = 0; n < L; n++) y[i * (L + gap) + n] = burst[n] * a; });
    play(y);
    let k = 0; const step = () => { drawStairs(key, k); if (++k < 5) timer = setTimeout(step, (L + gap) / SR * 1000); else timer = setTimeout(() => drawStairs(null, -1), (L + gap) / SR * 1000); };
    step();
  }
  if ($('stairCv')) {
    $('stairLin').addEventListener('click', () => runStairs('lin'));
    $('stairDb').addEventListener('click', () => runStairs('db'));
    drawStairs(null, -1); window.addEventListener('resize', () => drawStairs(null, -1));
  }

  // ---------------------------------------------------------------- 3. spectrum explorer
  function fft(re, im) {
    const n = re.length;
    for (let i = 1, j = 0; i < n; i++) { let bit = n >> 1; for (; j & bit; bit >>= 1) j ^= bit; j ^= bit; if (i < j) { [re[i], re[j]] = [re[j], re[i]]; [im[i], im[j]] = [im[j], im[i]]; } }
    for (let len = 2; len <= n; len <<= 1) { const a = -2 * Math.PI / len, wr = Math.cos(a), wi = Math.sin(a);
      for (let i = 0; i < n; i += len) { let cr = 1, ci = 0; for (let k = 0; k < len / 2; k++) { const p = i + k, q = p + len / 2, br = re[q] * cr - im[q] * ci, bi = re[q] * ci + im[q] * cr;
        re[q] = re[p] - br; im[q] = im[p] - bi; re[p] += br; im[p] += bi; const t = cr * wr - ci * wi; ci = cr * wi + ci * wr; cr = t; } } }
  }
  const FL = 512, NF = 2048;
  let spT = VOWEL_T, scale = 'db';
  function drawSpec() {
    // overview
    { const { c, W, H } = setupCanvas($('spWave'));
      c.strokeStyle = css('--accent'); c.lineWidth = 1; c.beginPath();
      for (let px = 0; px < W; px++) { const a = Math.floor(px / W * N), b = Math.floor((px + 1) / W * N); let lo = 0, hi = 0;
        for (let i = a; i < b; i++) { lo = Math.min(lo, x[i]); hi = Math.max(hi, x[i]); }
        c.moveTo(px + 0.5, H / 2 - hi * H / 2); c.lineTo(px + 0.5, H / 2 - lo * H / 2 + 0.5); }
      c.stroke();
      const cx = spT / (N / SR) * W; c.fillStyle = css('--accent-2'); c.globalAlpha = 0.25; c.fillRect(cx, 0, Math.max(2, FL / N * W), H); c.globalAlpha = 1;
      c.fillRect(cx, 0, 1.5, H); }
    // spectrum
    const i0 = Math.max(0, Math.min(N - FL, Math.round(spT * SR))), re = new Float64Array(NF), im = new Float64Array(NF);
    for (let n = 0; n < FL; n++) re[n] = x[i0 + n] * (0.5 - 0.5 * Math.cos(2 * Math.PI * n / (FL - 1)));
    fft(re, im);
    const K = NF / 2 + 1, mag = new Float64Array(K); let mx = 1e-12;
    for (let k = 0; k < K; k++) { mag[k] = Math.hypot(re[k], im[k]); mx = Math.max(mx, mag[k]); }
    const rng = +$('spRange').value; $('spRangeV').textContent = `${rng} dB`;
    const val = k => scale === 'lin' ? mag[k] / mx : scale === 'pow' ? (mag[k] / mx) ** 2 : Math.max(0, 1 + 20 * Math.log10(mag[k] / mx + 1e-12) / rng);
    const { c, W, H } = setupCanvas($('spSpec'));
    const L = 44, R = 10, T = 10, Bm = 26, pw = W - L - R, ph = H - T - Bm;
    c.strokeStyle = css('--line'); c.lineWidth = 1; c.fillStyle = css('--muted'); c.font = '10.5px Inter, sans-serif';
    c.textAlign = 'center';
    for (let f = 0; f <= 8; f++) { const px = L + f / 8 * pw; c.beginPath(); c.moveTo(px, T); c.lineTo(px, T + ph); c.stroke(); c.fillText(`${f}`, px, H - 10); }
    c.fillText('kHz', L + pw - 12, H - 0.5);
    c.textAlign = 'right';
    const ticks = scale === 'db' ? Array.from({ length: Math.floor(rng / 10) + 1 }, (_, i) => [-i * 10, 1 - i * 10 / rng]) : [[0, 0], [0.5, 0.5], [1, 1]];
    ticks.forEach(([lab, v]) => { const py = T + ph * (1 - v); c.beginPath(); c.moveTo(L, py); c.lineTo(L + pw, py); c.stroke(); c.fillText(scale === 'db' ? `${lab}` : lab.toFixed(1), L - 5, py + 3.5); });
    c.save(); c.translate(11, T + ph / 2); c.rotate(-Math.PI / 2); c.textAlign = 'center';
    c.fillText(scale === 'db' ? 'dB re peak' : scale === 'pow' ? '|X|² / max' : '|X| / max', 0, 0); c.restore();
    c.strokeStyle = css('--accent'); c.lineWidth = 1.3; c.beginPath();
    for (let k = 0; k < K; k++) { const px = L + k / (K - 1) * pw, py = T + ph * (1 - val(k)); k ? c.lineTo(px, py) : c.moveTo(px, py); }
    c.stroke();
    // share of power above 1 kHz, to make the point numerically
    let tot = 0, hi = 0; for (let k = 0; k < K; k++) { const p = mag[k] ** 2; tot += p; if (k * SR / NF > 1000) hi += p; }
    c.fillStyle = css('--ink'); c.textAlign = 'right'; c.font = '11.5px Inter, sans-serif';
    c.fillText(`frame at ${spT.toFixed(2)} s · power above 1 kHz: ${(100 * hi / tot).toFixed(1)}%`, L + pw - 4, T + 14);
  }
  if ($('spSpec')) {
    document.querySelectorAll('#specDemo [data-sc]').forEach(b => b.addEventListener('click', () => {
      scale = b.dataset.sc; document.querySelectorAll('#specDemo [data-sc]').forEach(o => o.classList.toggle('on', o === b)); drawSpec(); }));
    $('spWave').addEventListener('click', e => { const r = $('spWave').getBoundingClientRect(); spT = Math.max(0, (e.clientX - r.left) / r.width * N / SR - FL / SR / 2); drawSpec(); });
    $('spV').addEventListener('click', () => { spT = VOWEL_T; drawSpec(); });
    $('spF').addEventListener('click', () => { spT = FRIC_T; drawSpec(); });
    $('spPlay').addEventListener('click', () => { const a = Math.max(0, Math.round((spT + FL / SR / 2 - 0.15) * SR)); play(fade(x.slice(a, a + Math.round(0.3 * SR)), 80)); });
    $('spRange').addEventListener('input', drawSpec);
    drawSpec(); window.addEventListener('resize', drawSpec);
  }

  // ---------------------------------------------------------------- 4. SNR listening
  let seed = 12345; const rnd = () => (seed = (seed * 1664525 + 1013904223) >>> 0) / 4294967296;
  const noise = new Float32Array(N); let pn = 0;
  for (let i = 0; i < N; i += 2) { const u = Math.max(rnd(), 1e-12), v = rnd(), r = Math.sqrt(-2 * Math.log(u));
    noise[i] = r * Math.cos(2 * Math.PI * v); if (i + 1 < N) noise[i + 1] = r * Math.sin(2 * Math.PI * v); }
  for (let i = 0; i < N; i++) pn += noise[i] * noise[i]; pn /= N;
  function snrRead(s) {
    if (s === null) { $('snrRead').innerHTML = '<div>playing<b>clean recording</b></div>'; return; }
    $('snrRead').innerHTML = `<div>SNR<b>${s} dB</b></div><div>noise power / speech power, \\(10^{-\\text{SNR}/10}\\)<b>${(100 * Math.pow(10, -s / 10)).toPrecision(3)} %</b></div>` +
      `<div>noise rms / speech rms, \\(10^{-\\text{SNR}/20}\\)<b>${(100 * Math.pow(10, -s / 20)).toPrecision(3)} %</b></div>`;
    tex($('snrRead'));
  }
  if ($('snrBtns')) {
    [null, 30, 20, 10, 5, 0].forEach(s => {
      const b = document.createElement('button'); b.textContent = s === null ? '▶ clean' : `▶ ${s} dB`;
      b.addEventListener('click', () => {
        const g = s === null ? 0 : Math.sqrt(P / pn / Math.pow(10, s / 10));
        // keep the mix's peak in range: scale speech and noise together
        const y = new Float32Array(N); let pk = 0;
        for (let i = 0; i < N; i++) { y[i] = x[i] + g * noise[i]; pk = Math.max(pk, Math.abs(y[i])); }
        const k = pk > 0.98 ? 0.98 / pk : 1; play(y.map(v => v * k)); snrRead(s);
        document.querySelectorAll('#snrBtns button').forEach(o => o.classList.toggle('on', o === b));
      });
      $('snrBtns').appendChild(b);
    });
    snrRead(20);
  }

  // ---------------------------------------------------------------- 5. adding two sources
  function add() {
    const a = +$('adA').value, b = +$('adB').value;
    $('adAV').textContent = `${a} dB`; $('adBV').textContent = `${b} dB`;
    const inc = 10 * Math.log10(Math.pow(10, a / 10) + Math.pow(10, b / 10));
    const coh = 20 * Math.log10(Math.pow(10, a / 20) + Math.pow(10, b / 20));
    const opp = Math.abs(Math.pow(10, a / 20) - Math.pow(10, b / 20));
    $('adRead').innerHTML = `<div>naive (wrong) sum<b style="text-decoration:line-through;">${a + b} dB</b></div>` +
      `<div>independent sources (powers add)<b>${inc.toFixed(1)} dB</b></div>` +
      `<div>identical, in phase (amplitudes add)<b>${coh.toFixed(1)} dB</b></div>` +
      `<div>identical, opposite phase<b>${opp < 1e-9 ? '\\(-\\infty\\) (silence)' : (20 * Math.log10(opp)).toFixed(1) + ' dB'}</b></div>` +
      `<div>louder source + this much<b>+${(inc - Math.max(a, b)).toFixed(2)} dB</b></div>`;
    tex($('adRead'));
  }
  if ($('adA')) { $('adA').addEventListener('input', add); $('adB').addEventListener('input', add); add(); }
})();
