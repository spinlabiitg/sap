(() => {
  const $ = id => document.getElementById(id);
  const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
  const SR = 16000;
  // typeset \( \) maths written into the DOM after KaTeX's auto-render has loaded (it renders the first pass itself)
  const tex = el => { if (window.renderMathInElement) window.renderMathInElement(el, { delimiters: [{ left: '\\(', right: '\\)', display: false }], throwOnError: false }); };
  if (!window.CODEC_SPEECH || !$('lvWave')) return;

  // ---- decode the embedded sentence (int16 little-endian, base64)
  const bin = atob(window.CODEC_SPEECH.b64), N = bin.length >> 1, x = new Float32Array(N);
  for (let i = 0; i < N; i++) { let v = bin.charCodeAt(2 * i) | (bin.charCodeAt(2 * i + 1) << 8); if (v >= 32768) v -= 65536; x[i] = v / 32768; }

  // start of the loudest 20 ms of the recording (for the waveform zoom)
  let LOUD0 = 0; { let best = -1; for (let i = 0; i + 320 < N; i += 160) { let e = 0; for (let k = 0; k < 320; k++) e += x[i + k] * x[i + k]; if (e > best) { best = e; LOUD0 = i; } } }

  function setupCanvas(cv) {
    if (!cv.dataset.h) { cv.dataset.h = cv.getAttribute('height'); cv.style.height = cv.dataset.h + 'px'; }
    const dpr = window.devicePixelRatio || 1, W = cv.clientWidth, H = +cv.dataset.h;
    cv.width = W * dpr; cv.height = H * dpr; const c = cv.getContext('2d'); c.setTransform(dpr, 0, 0, dpr, 0, 0); c.clearRect(0, 0, W, H);
    return { c, W, H };
  }
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

  // ---- windowed-sinc low-pass (cutoff as a fraction of 16 kHz Nyquist)
  function lowpass(sig, cutoff, taps = 129) {
    const h = new Float32Array(taps), M = (taps - 1) / 2; let s = 0;
    for (let n = 0; n < taps; n++) { const k = n - M, w = 0.54 - 0.46 * Math.cos(2 * Math.PI * n / (taps - 1)); h[n] = (k === 0 ? cutoff : Math.sin(Math.PI * cutoff * k) / (Math.PI * k)) * w; s += h[n]; }
    for (let n = 0; n < taps; n++) h[n] /= s;
    const y = new Float32Array(sig.length);
    for (let i = 0; i < sig.length; i++) { let acc = 0; for (let n = 0; n < taps; n++) { const j = i - n + M; if (j >= 0 && j < sig.length) acc += h[n] * sig[j]; } y[i] = acc; }
    return y;
  }
  const MU = 255;
  const quant = (v, bits, mode) => {
    if (bits >= 16) return v;
    const L = Math.pow(2, bits - 1), q = u => Math.max(-1, Math.min(1, (Math.floor(u * L) + 0.5) / L));
    if (mode === 'mu') { const c = Math.sign(v) * Math.log1p(MU * Math.abs(v)) / Math.log1p(MU), cq = q(c); return Math.sign(cq) * (Math.pow(1 + MU, Math.abs(cq)) - 1) / MU; }
    return q(v);
  };
  const cache = {};
  function process(M, bits, mode) {
    const key = `${M}|${bits}|${mode}`; if (cache[key]) return cache[key];
    let band = M === 1 ? x : (cache['lp' + M] || (cache['lp' + M] = lowpass(x, 0.9 / M)));
    // decimate, quantise at the low rate, then rebuild at 16 kHz (zero-stuff + the same low-pass)
    const n = Math.floor(N / M), low = new Float32Array(n);
    for (let i = 0; i < n; i++) low[i] = quant(band[i * M], bits, mode);
    let y;
    if (M === 1) y = low;
    else { const up = new Float32Array(N); for (let i = 0; i < n; i++) up[i * M] = low[i] * M; y = lowpass(up, 0.9 / M); }
    const bandRef = band;
    return (cache[key] = { y, ref: bandRef });
  }
  function segsnr(ref, y) {
    const n = 320; let emax = 0; const E = [];
    for (let i = 0; i + n < N; i += n) { let e = 0; for (let k = 0; k < n; k++) e += ref[i + k] ** 2; E.push(e); emax = Math.max(emax, e); }
    let s = 0, c = 0;
    E.forEach((e, j) => { if (e < emax * 1e-4) return; let d = 0; for (let k = 0; k < n; k++) d += (ref[j * n + k] - y[j * n + k]) ** 2; s += Math.max(-10, Math.min(40, 10 * Math.log10(e / (d + 1e-12)))); c++; });
    return s / c;
  }
  function fft(re, im) {
    const n = re.length;
    for (let i = 1, j = 0; i < n; i++) { let bit = n >> 1; for (; j & bit; bit >>= 1) j ^= bit; j ^= bit; if (i < j) { [re[i], re[j]] = [re[j], re[i]]; [im[i], im[j]] = [im[j], im[i]]; } }
    for (let len = 2; len <= n; len <<= 1) { const a = -2 * Math.PI / len, wr = Math.cos(a), wi = Math.sin(a);
      for (let i = 0; i < n; i += len) { let cr = 1, ci = 0; for (let k = 0; k < len / 2; k++) { const p = i + k, q = p + len / 2, br = re[q] * cr - im[q] * ci, bi = re[q] * ci + im[q] * cr;
        re[q] = re[p] - br; im[q] = im[p] - bi; re[p] += br; im[p] += bi; const t = cr * wr - ci * wi; ci = cr * wi + ci * wr; cr = t; } } }
  }
  function update() {
    const M = +$('lvFs').value, bits = +$('lvBits').value, mode = $('lvQ').value, fs = SR / M;
    const { y, ref } = process(M, bits, mode);
    const kbps = fs * bits / 1000;
    $('lvRate').textContent = `${kbps % 1 ? kbps.toFixed(1) : kbps} kbit/s`;
    const snr = bits >= 16 ? null : segsnr(ref, y);
    $('lvRead').innerHTML = `<div>samples/s \\(\\times\\) bits/sample<b>\\(${SR / M} \\times ${bits}\\)</b></div>` +
      `<div>segmental SNR<b>${snr === null ? 'no rounding' : snr.toFixed(1) + ' dB'}</b></div>` +
      `<div>1 hour takes<b>${(kbps * 3600 / 8 / 1000).toFixed(1)} MB</b></div>` +
      `<div>vs raw 256 kbit/s<b>\\(${(256 / kbps).toFixed(kbps < 32 ? 0 : 1)}\\times\\) smaller</b></div>`;
    tex($('lvRead'));
    // waveform, 20 ms from the vowel of "had"
    { const { c, W, H } = setupCanvas($('lvWave')), L = 320, s0 = LOUD0;
      const X = i => 4 + i / L * (W - 8), Y = v => H / 2 - v / 0.95 * (H / 2 - 6);
      c.strokeStyle = css('--line'); c.beginPath(); c.moveTo(0, H / 2); c.lineTo(W, H / 2); c.stroke();
      c.strokeStyle = css('--line-strong'); c.lineWidth = 3; c.beginPath(); for (let i = 0; i < L; i++) { const v = Y(x[s0 + i]); i ? c.lineTo(X(i), v) : c.moveTo(X(i), v); } c.stroke();
      c.strokeStyle = css('--accent-2'); c.lineWidth = 1.3; c.beginPath(); for (let i = 0; i < L; i++) { const v = Y(y[s0 + i]); i ? c.lineTo(X(i), v) : c.moveTo(X(i), v); } c.stroke(); }
    // spectrogram of the processed sentence
    { const { c, W, H } = setupCanvas($('lvSpec')), nfft = 512, hop = 256, frames = Math.floor((N - nfft) / hop);
      const img = c.createImageData(Math.max(1, Math.floor(W)), Math.max(1, Math.floor(H))), cols = img.width, rows = img.height;
      const S = []; let mx = -1e9;
      for (let f = 0; f < frames; f++) { const re = new Float64Array(nfft), im = new Float64Array(nfft);
        for (let k = 0; k < nfft; k++) re[k] = y[f * hop + k] * (0.5 - 0.5 * Math.cos(2 * Math.PI * k / (nfft - 1)));
        fft(re, im); const col = new Float32Array(nfft / 2); for (let k = 0; k < nfft / 2; k++) { col[k] = 10 * Math.log10(re[k] * re[k] + im[k] * im[k] + 1e-10); mx = Math.max(mx, col[k]); } S.push(col); }
      for (let px = 0; px < cols; px++) { const col = S[Math.min(frames - 1, Math.floor(px / cols * frames))];
        for (let py = 0; py < rows; py++) { const k = Math.min(nfft / 2 - 1, Math.floor((1 - py / rows) * nfft / 2)), v = Math.max(0, Math.min(1, (col[k] - mx + 70) / 70));
          const o = 4 * (py * cols + px); img.data[o] = 255 * Math.min(1, 1.6 * v); img.data[o + 1] = 255 * Math.max(0, 1.7 * v - 0.7); img.data[o + 2] = 255 * (v < 0.5 ? 0.6 * v : Math.max(0, 1.4 - 1.6 * v) * 0.6 + 0.1); img.data[o + 3] = 255; } }
      c.putImageData(img, 0, 0);
      const ny = (1 - (fs / 2) / 8000) * rows;
      if (M > 1) { c.strokeStyle = '#fff'; c.setLineDash([4, 3]); c.beginPath(); c.moveTo(0, ny); c.lineTo(cols, ny); c.stroke(); c.setLineDash([]); c.fillStyle = '#fff'; c.font = '10px Inter, sans-serif'; c.fillText(`Nyquist: ${fs / 2000} kHz`, 6, ny - 4); } }
    $('lvPlayP').onclick = () => play(y);
  }
  $('lvPlayO').onclick = () => play(x);
  ['lvFs', 'lvBits', 'lvQ'].forEach(id => $(id).addEventListener('change', update));
  window.addEventListener('resize', update);
  update();
  // ---------------- mu-law explorer ----------------
  if ($('muHist')) {
    let muBits = 4;
    const muFromSlider = v => (v <= 0 ? 0 : Math.pow(10, v * 3 / 100));
    const compand = (v, mu) => mu < 1e-3 ? v : Math.sign(v) * Math.log1p(mu * Math.abs(v)) / Math.log1p(mu);
    const expand = (u, mu) => mu < 1e-3 ? u : Math.sign(u) * (Math.pow(1 + mu, Math.abs(u)) - 1) / mu;
    const qmu = (v, bits, mu) => { const L = Math.pow(2, bits - 1); const u = Math.max(-1, Math.min(1, (Math.floor(compand(v, mu) * L) + 0.5) / L)); return expand(u, mu); };
    // histogram of the sentence over [-0.5, 0.5]
    const HB = 200, hist = new Float32Array(HB); let hmax = 0;
    for (let i = 0; i < N; i++) { const v = x[i]; if (Math.abs(v) < 0.5 && Math.abs(v) > 1e-4) { const k = Math.floor((v + 0.5) / 1 * HB); hist[k]++; } }
    for (const h of hist) hmax = Math.max(hmax, h);
    // frame energies for loud/quiet SNR
    const FL = 320, frames = []; let fmax = 0;
    for (let i = 0; i + FL <= N; i += FL) { let e = 0; for (let k = 0; k < FL; k++) e += x[i + k] ** 2; frames.push(e); fmax = Math.max(fmax, e); }
    function muUpdate() {
      const mu = muFromSlider(+$('muMu').value), L = Math.pow(2, muBits - 1);
      $('muMuV').textContent = mu < 1e-3 ? '0 (uniform)' : mu < 10 ? mu.toFixed(1) : Math.round(mu) + (Math.abs(mu - 255) < 30 ? '  (≈ G.711)' : '');
      const levels = []; for (let k = -L; k < L; k++) levels.push(expand((k + 0.5) / L, mu));
      { const { c, W, H } = setupCanvas($('muHist')), pl = 6, pw = W - 12, pt = 6, ph = H - 24, X = v => pl + (v + 0.5) * pw;
        c.fillStyle = css('--accent-3'); c.globalAlpha = 0.8;
        for (let k = 0; k < HB; k++) { const h = hist[k] / hmax * ph; c.fillRect(pl + k / HB * pw, pt + ph - h, pw / HB + 0.5, h); }
        c.globalAlpha = 1; c.strokeStyle = css('--accent'); c.lineWidth = 1.3;
        levels.forEach(v => { if (Math.abs(v) <= 0.5) { c.beginPath(); c.moveTo(X(v), pt); c.lineTo(X(v), pt + ph); c.stroke(); } });
        c.fillStyle = css('--muted'); c.font = '10px Inter, sans-serif';
        [-0.5, -0.25, 0, 0.25, 0.5].forEach(v => c.fillText(v, X(v) - 8, H - 6));
        c.strokeStyle = css('--accent-2'); c.setLineDash([3, 3]); [-0.1, 0.1].forEach(v => { c.beginPath(); c.moveTo(X(v), pt); c.lineTo(X(v), pt + ph); c.stroke(); }); c.setLineDash([]); }
      const y = new Float32Array(N); for (let i = 0; i < N; i++) y[i] = qmu(x[i], muBits, mu);
      let sl = 0, cl = 0, sq = 0, cq = 0;
      frames.forEach((e, j) => { const rel = 10 * Math.log10(e / fmax); if (rel < -50) return; let d = 0; for (let k = 0; k < FL; k++) d += (x[j * FL + k] - y[j * FL + k]) ** 2;
        const snr = 10 * Math.log10(e / (d + 1e-12)); if (rel > -10) { sl += snr; cl++; } else if (rel < -25) { sq += snr; cq++; } });
      const inside = levels.filter(v => Math.abs(v) < 0.1).length;
      $('muRead').innerHTML = `<div>levels within \\(\\pm 0.1\\) (red dashes)<b>${inside} of ${2 * L}</b></div><div>SNR, loud frames<b>${(sl / cl).toFixed(1)} dB</b></div><div>SNR, quiet frames (25+ dB down)<b>${(sq / cq).toFixed(1)} dB</b></div><div>bit rate at 8 kHz<b>${8 * muBits} kbit/s</b></div>`;
      $('muNote').innerHTML = mu < 1e-3 ? 'Uniform: quiet frames get far worse SNR than loud ones.' : mu > 100 ? 'Strong compression: quiet frames gain hugely, loud frames give up a little. A good trade, because the ear notices noise most in quiet passages.' : 'Moderate compression: watch the quiet-frame SNR climb as \\(\\mu\\) grows.';
      tex($('muRead')); tex($('muNote'));
      $('muPlay').onclick = () => play(y);
    }
    $('muMu').addEventListener('input', muUpdate);
    document.querySelectorAll('#muBits button').forEach(b => b.addEventListener('click', () => { muBits = +b.dataset.b; document.querySelectorAll('#muBits button').forEach(z => z.classList.toggle('on', z === b)); muUpdate(); }));
    window.addEventListener('resize', muUpdate);
    muUpdate();
  }
})();
