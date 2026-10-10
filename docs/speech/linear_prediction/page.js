(() => {
  const $ = id => document.getElementById(id);
  const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
  const SIGS = window.LP_SIGNALS, SR = window.LP_SR, T0 = window.LP_T0;
  const AUDIO = { noise: 'white_noise.wav', tone: 'tone_200.wav', aa: 'arctic_ah.wav', iy: 'arctic_see.wav', s: 'arctic_see.wav' };
  const COLOR = { noise: '--accent-4', tone: '--teal', aa: '--accent-2', iy: '--accent', s: '--accent-3' };
  const SHORT = { noise: 'white noise', tone: 'tone', aa: 'vowel /ɑ/', iy: 'vowel /i/', s: 'fricative /s/' };

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
  function segButtons(id, keys, initial, onPick) {
    const seg = $(id);
    keys.forEach(k => {
      const b = document.createElement('button'); b.textContent = SHORT[k]; b.dataset.k = k; if (k === initial) b.classList.add('on');
      b.addEventListener('click', () => { seg.querySelectorAll('button').forEach(x => x.classList.toggle('on', x === b)); onPick(k); });
      seg.appendChild(b);
    });
  }

  // ---------------- maths ----------------
  function corr(x, k) {
    let num = 0, den = 0;
    for (let n = 0; n < x.length; n++) { den += x[n] * x[n]; if (n + k < x.length) num += x[n] * x[n + k]; }
    return num / den;
  }
  // least-squares predictor of order p (covariance method), errors evaluated from n = start
  function solve(A, b) {
    const n = b.length, M = A.map((row, i) => row.concat([b[i]]));
    for (let i = 0; i < n; i++) {
      let piv = i; for (let r = i + 1; r < n; r++) if (Math.abs(M[r][i]) > Math.abs(M[piv][i])) piv = r;
      [M[i], M[piv]] = [M[piv], M[i]];
      const d = M[i][i] || 1e-12;
      for (let r = i + 1; r < n; r++) { const f = M[r][i] / d; for (let c = i; c <= n; c++) M[r][c] -= f * M[i][c]; }
    }
    const x = new Array(n).fill(0);
    for (let i = n - 1; i >= 0; i--) { let s = M[i][n]; for (let c = i + 1; c < n; c++) s -= M[i][c] * x[c]; x[i] = s / (M[i][i] || 1e-12); }
    return x;
  }
  function lsPredictor(x, p, start) {
    if (p === 0) return [];
    const A = [], b = [];
    for (let i = 1; i <= p; i++) {
      const row = [];
      for (let k = 1; k <= p; k++) { let s = 0; for (let n = start; n < x.length; n++) s += x[n - i] * x[n - k]; row.push(s + (i === k ? 1e-9 : 0)); }
      A.push(row);
      let s = 0; for (let n = start; n < x.length; n++) s += x[n] * x[n - i]; b.push(s);
    }
    return solve(A, b);
  }
  function predict(x, w) {
    const y = new Float64Array(x.length);
    for (let n = 0; n < x.length; n++) { let s = 0; for (let k = 1; k <= w.length; k++) if (n - k >= 0) s += w[k - 1] * x[n - k]; y[n] = s; }
    return y;
  }
  const PMAX = 24, START = PMAX;
  const cache = {};
  function analysis(key) {
    if (cache[key]) return cache[key];
    const x = SIGS[key].x, W = [], G = [];
    let ex = 0; for (let n = START; n < x.length; n++) ex += x[n] * x[n];
    for (let p = 0; p <= PMAX; p++) {
      const w = lsPredictor(x, p, START), y = predict(x, w);
      let ee = 0; for (let n = START; n < x.length; n++) ee += (x[n] - y[n]) ** 2;
      W.push(w); G.push(10 * Math.log10(ex / Math.max(ee, ex * 1e-12)));
    }
    return (cache[key] = { W, G });
  }

  // ---------------- 1. guessing game ----------------
  const GAME_KEYS = ['tone', 'aa', 's', 'noise'], SHOW = 30, GAME_P = 12;
  const game = { key: 'tone', n0: 0, guess: null, revealed: false, score: {} };
  GAME_KEYS.forEach(k => game.score[k] = { you: 0, pred: 0, n: 0 });
  function newRound() {
    const x = SIGS[game.key].x;
    game.n0 = START + SHOW + Math.floor(Math.random() * (x.length - START - SHOW - 1));
    game.guess = null; game.revealed = false;
    $('gameMsg').textContent = 'Click inside the dashed column to make your guess.';
    drawGame();
  }
  function gameGeom(W, H) { const x0 = 40, x1 = W - 16, dx = (x1 - x0) / (SHOW + 1); return { x0, x1, dx, yc: H / 2, sy: (H / 2 - 16) / 1.1 }; }
  function drawGame() {
    const cv = $('gameCanvas'), { c, W, H } = setupCanvas(cv), g = gameGeom(W, H), x = SIGS[game.key].x;
    const col = css(COLOR[game.key]), muted = css('--muted'), line = css('--line');
    c.strokeStyle = line; c.beginPath(); c.moveTo(g.x0, g.yc); c.lineTo(g.x1, g.yc); c.stroke();
    const X = i => g.x0 + (i + 0.5) * g.dx, Y = v => g.yc - v * g.sy;
    // target column
    c.setLineDash([4, 4]); c.strokeStyle = muted; c.strokeRect(X(SHOW) - g.dx / 2, 6, g.dx, H - 12); c.setLineDash([]);
    c.fillStyle = muted; c.font = '10px Inter, sans-serif'; c.textAlign = 'center'; c.fillText('next?', X(SHOW), H - 2);
    c.textAlign = 'left'; c.fillText(`the last ${SHOW} samples (${(SHOW / SR * 1000).toFixed(1)} ms)`, g.x0, 12);
    c.strokeStyle = col; c.lineWidth = 1.6; c.beginPath();
    for (let i = 0; i < SHOW; i++) { const v = x[game.n0 - SHOW + i]; i ? c.lineTo(X(i), Y(v)) : c.moveTo(X(i), Y(v)); }
    c.stroke();
    c.fillStyle = col; for (let i = 0; i < SHOW; i++) { c.beginPath(); c.arc(X(i), Y(x[game.n0 - SHOW + i]), 2.6, 0, 2 * Math.PI); c.fill(); }
    if (game.guess != null) { c.fillStyle = css('--ink'); c.beginPath(); c.arc(X(SHOW), Y(game.guess), 6, 0, 2 * Math.PI); c.fill(); }
    if (game.revealed) {
      const truth = x[game.n0], w = analysis(game.key).W[GAME_P];
      let pr = 0; for (let k = 1; k <= GAME_P; k++) pr += w[k - 1] * x[game.n0 - k];
      c.strokeStyle = col; c.lineWidth = 3; c.beginPath(); c.arc(X(SHOW), Y(truth), 8, 0, 2 * Math.PI); c.stroke();
      c.fillStyle = css('--accent'); c.save(); c.translate(X(SHOW), Y(pr)); c.rotate(Math.PI / 4); c.fillRect(-5, -5, 10, 10); c.restore();
      c.font = '11px Inter, sans-serif'; c.textAlign = 'left';
      c.fillStyle = css('--ink'); c.fillText('● you', g.x0, H - 10);
      c.fillStyle = col; c.fillText('○ true value', g.x0 + 48, H - 10);
      c.fillStyle = css('--accent'); c.fillText('◆ formula (12 past samples)', g.x0 + 132, H - 10);
    }
  }
  function drawScore() {
    $('gameScore').innerHTML = GAME_KEYS.filter(k => game.score[k].n).map(k => {
      const s = game.score[k];
      return `<div>${SHORT[k]} · ${s.n} round${s.n > 1 ? 's' : ''}<b>you ${(s.you / s.n).toFixed(2)} · formula ${(s.pred / s.n).toFixed(2)}</b>average miss</div>`;
    }).join('');
  }
  $('gameCanvas').addEventListener('click', ev => {
    if (game.revealed) { newRound(); return; }
    const cv = $('gameCanvas'), r = cv.getBoundingClientRect(), g = gameGeom(r.width, +cv.dataset.h);
    const xi = (ev.clientX - r.left - g.x0) / g.dx - 0.5;
    if (Math.abs(xi - SHOW) > 1.2) { $('gameMsg').textContent = 'Click inside the dashed column (the next sample).'; return; }
    game.guess = Math.max(-1.1, Math.min(1.1, (g.yc - (ev.clientY - r.top)) / g.sy));
    game.revealed = true;
    const x = SIGS[game.key].x, truth = x[game.n0], w = analysis(game.key).W[GAME_P];
    let pr = 0; for (let k = 1; k <= GAME_P; k++) pr += w[k - 1] * x[game.n0 - k];
    const s = game.score[game.key]; s.n++; s.you += Math.abs(game.guess - truth); s.pred += Math.abs(pr - truth);
    $('gameMsg').textContent = `You missed by ${Math.abs(game.guess - truth).toFixed(2)}, the formula by ${Math.abs(pr - truth).toFixed(2)}. Click the plot or “Next round” to play again.`;
    drawGame(); drawScore();
  });
  $('gameNext').addEventListener('click', newRound);
  segButtons('gameSig', GAME_KEYS, 'tone', k => { game.key = k; newRound(); });

  // ---------------- 2. scatter ----------------
  const SC_KEYS = ['noise', 'tone', 'aa', 'iy', 's'];
  let scKey = 'aa';
  function scDraw() {
    const x = SIGS[scKey].x, k = +$('scK').value, r = corr(x, k), col = css(COLOR[scKey]);
    $('scKv').textContent = `${k} sample${k > 1 ? 's' : ''} = ${(k / SR * 1000).toFixed(2)} ms`;
    $('scR').textContent = (r >= 0 ? '+' : '−') + Math.abs(r).toFixed(2);
    { const { c, W, H } = setupCanvas($('scCanvas'));
      const S = Math.min(W, H) - 34, x0 = 28 + (W - 28 - S) / 2, y0 = 6, X = v => x0 + (v + 1.05) / 2.1 * S, Y = v => y0 + (1.05 - v) / 2.1 * S;
      c.strokeStyle = css('--line'); c.strokeRect(x0, y0, S, S);
      c.beginPath(); c.moveTo(X(0), y0); c.lineTo(X(0), y0 + S); c.moveTo(x0, Y(0)); c.lineTo(x0 + S, Y(0)); c.stroke();
      c.fillStyle = col; c.globalAlpha = 0.35;
      for (let n = 0; n + k < x.length; n++) c.fillRect(X(x[n]) - 1.5, Y(x[n + k]) - 1.5, 3, 3);
      c.globalAlpha = 1;
      if ($('scLine').checked) { c.strokeStyle = css('--ink'); c.lineWidth = 2; c.beginPath(); c.moveTo(X(-1.05), Y(-1.05 * r)); c.lineTo(X(1.05), Y(1.05 * r)); c.stroke(); }
      c.fillStyle = css('--muted'); c.font = '10px Inter, sans-serif'; c.textAlign = 'center';
      c.fillText('x[n]', x0 + S / 2, y0 + S + 14);
      c.save(); c.translate(x0 - 14, y0 + S / 2); c.rotate(-Math.PI / 2); c.fillText('x[n+k]', 0, 0); c.restore(); c.textAlign = 'left'; }
    { const { c, W, H } = setupCanvas($('scCurve'));
      const pl = 30, pw = W - pl - 8, pt = 6, ph = H - 24, K = 200, X = kk => pl + kk / K * pw, Y = v => pt + (1 - v) / 2 * ph;
      c.strokeStyle = css('--line'); c.beginPath(); c.moveTo(pl, Y(0)); c.lineTo(pl + pw, Y(0)); c.stroke();
      c.fillStyle = css('--muted'); c.font = '10px Inter, sans-serif';
      c.fillText('+1', 4, Y(1) + 4); c.fillText('0', 12, Y(0) + 3); c.fillText('−1', 6, Y(-1));
      for (let ms = 0; ms <= 12; ms += 2) { const kk = ms * SR / 1000; if (kk <= K) c.fillText(ms + ' ms', X(kk) - 8, H - 4); }
      if (scKey === 'aa' && T0 <= K) { c.strokeStyle = css('--accent-2'); c.setLineDash([3, 3]); c.beginPath(); c.moveTo(X(T0), pt); c.lineTo(X(T0), pt + ph); c.stroke(); c.setLineDash([]); c.fillStyle = css('--accent-2'); c.fillText('pitch period', X(T0) + 3, pt + 10); }
      c.strokeStyle = col; c.lineWidth = 1.6; c.beginPath();
      for (let kk = 0; kk <= K; kk++) { const v = kk ? corr(x, kk) : 1; kk ? c.lineTo(X(kk), Y(v)) : c.moveTo(X(kk), Y(v)); }
      c.stroke();
      c.fillStyle = css('--ink'); c.beginPath(); c.arc(X(k), Y(r), 4.5, 0, 2 * Math.PI); c.fill(); }
    const notes = {
      noise: 'White noise: a round blob at every lag. Nothing to predict.',
      tone: 'Pure tone: always a perfect ellipse/line. \\(r(k) = \\cos(2\\pi \\cdot 200\\,k/16000)\\) swings smoothly between \\(+1\\) and \\(-1\\) every 5 ms.',
      aa: k < 3 ? 'Neighbouring samples almost equal: a tight rising line.' : (k === T0 || Math.abs(k - T0) < 4) ? 'One pitch period apart: the waveform has repeated, so the line is back.' : 'Formant ringing makes \\(r\\) swing up and down as \\(k\\) grows; it peaks again at the pitch period (dashed).',
      iy: 'Vowel /i/: very predictable at small lags; its high \\(F_2\\) adds faster wiggles to \\(r(k)\\).',
      s: k === 1 ? 'Fricative /s/ at \\(k = 1\\): negative. Its high-frequency hiss flips sign from sample to sample.' : 'Fricative /s/: weak correlations at every lag. Mostly unpredictable noise.',
    };
    setMath($('scNote'), notes[scKey]);
    $('scPlayers').innerHTML = `<div class="player">listen: ${SIGS[scKey].label}<audio controls preload="none" src="linear_prediction/audio/${AUDIO[scKey]}"></audio></div>`;
  }
  segButtons('scSig', SC_KEYS, scKey, k => { scKey = k; scDraw(); });
  $('scK').addEventListener('input', scDraw);
  $('scLine').addEventListener('change', scDraw);

  // ---------------- 4. order slider ----------------
  let lpKey = 'aa';
  function lpDraw() {
    const p = +$('lpP').value, x = SIGS[lpKey].x, A = analysis(lpKey), w = A.W[p], y = predict(x, w), col = css(COLOR[lpKey]);
    $('lpPv').textContent = p;
    const g = A.G[p];
    if (g > 60) setMath($('lpG'), '\\(&gt; 60\\) dB (perfect)'); else $('lpG').textContent = g.toFixed(1) + ' dB';
    if (g > 60) setMath($('lpE'), '\\(\\approx 0\\) %'); else $('lpE').textContent = (100 * Math.pow(10, -g / 10)).toFixed(g > 20 ? 2 : 1) + ' %';
    { const { c, W, H } = setupCanvas($('lpWave'));
      const n0 = START, L = 400, X = i => 6 + i / L * (W - 12), half = (H - 16) / 4, yA = 8 + half, yE = 8 + 3 * half;
      const Y = (v, yc) => yc - v * (half - 4);
      c.strokeStyle = css('--line'); c.beginPath(); c.moveTo(6, yA); c.lineTo(W - 6, yA); c.moveTo(6, yE); c.lineTo(W - 6, yE); c.stroke();
      c.lineWidth = 3.2; c.strokeStyle = css('--line-strong'); c.beginPath(); for (let i = 0; i < L; i++) { const v = Y(x[n0 + i], yA); i ? c.lineTo(X(i), v) : c.moveTo(X(i), v); } c.stroke();
      c.lineWidth = 1.3; c.strokeStyle = css('--accent'); c.beginPath(); for (let i = 0; i < L; i++) { const v = Y(y[n0 + i], yA); i ? c.lineTo(X(i), v) : c.moveTo(X(i), v); } c.stroke();
      c.strokeStyle = css('--accent-2'); c.beginPath(); for (let i = 0; i < L; i++) { const v = Y(x[n0 + i] - y[n0 + i], yE); i ? c.lineTo(X(i), v) : c.moveTo(X(i), v); } c.stroke();
      c.fillStyle = css('--muted'); c.font = '10px Inter, sans-serif'; c.fillText('signal & prediction', 8, 12); c.fillText('error', 8, yE - half + 10); }
    { const { c, W, H } = setupCanvas($('lpGain'));
      const pl = 30, pw = W - pl - 8, pt = 8, ph = H - 26, GM = 40, X = q => pl + q / PMAX * pw, Y = v => pt + (1 - Math.min(GM, Math.max(0, v)) / GM) * ph;
      c.fillStyle = css('--muted'); c.font = '10px Inter, sans-serif'; c.strokeStyle = css('--line');
      [0, 10, 20, 30, 40].forEach(v => { c.beginPath(); c.moveTo(pl, Y(v)); c.lineTo(pl + pw, Y(v)); c.stroke(); c.fillText(v + (v === 40 ? '+' : ''), 6, Y(v) + 3); });
      [0, 4, 8, 12, 16, 20, 24].forEach(q => c.fillText(q, X(q) - 3, H - 6));
      c.fillText('dB', 6, 10);
      SC_KEYS.forEach(kk => {
        if (kk === lpKey) return;
        const G2 = analysis(kk).G; c.strokeStyle = css(COLOR[kk]); c.globalAlpha = 0.25; c.lineWidth = 1.2; c.beginPath(); G2.forEach((v, q) => q ? c.lineTo(X(q), Y(v)) : c.moveTo(X(q), Y(v))); c.stroke(); c.globalAlpha = 1;
      });
      c.strokeStyle = col; c.lineWidth = 2.2; c.beginPath(); A.G.forEach((v, q) => q ? c.lineTo(X(q), Y(v)) : c.moveTo(X(q), Y(v))); c.stroke();
      c.fillStyle = css('--ink'); c.beginPath(); c.arc(X(p), Y(A.G[p]), 4.5, 0, 2 * Math.PI); c.fill(); }
    { const { c, W, H } = setupCanvas($('lpW'));
      const pl = 30, pw = W - pl - 8, yc = H / 2 - 6, sc = (H / 2 - 18) / Math.max(1, ...w.map(Math.abs)), X = k => pl + (k - 0.5) / PMAX * pw;
      c.strokeStyle = css('--line'); c.beginPath(); c.moveTo(pl, yc); c.lineTo(pl + pw, yc); c.stroke();
      c.fillStyle = css('--muted'); c.font = '10px Inter, sans-serif';
      [1, 4, 8, 12, 16, 20, 24].forEach(k => c.fillText(k, X(k) - 3, H - 6));
      c.fillText('k', 8, H - 6);
      w.forEach((v, i) => { const xx = X(i + 1); c.strokeStyle = col; c.lineWidth = 2; c.beginPath(); c.moveTo(xx, yc); c.lineTo(xx, yc - v * sc); c.stroke();
        c.fillStyle = col; c.beginPath(); c.arc(xx, yc - v * sc, 3, 0, 2 * Math.PI); c.fill(); });
      if (!w.length) { c.fillStyle = css('--muted'); c.fillText('p = 0: no weights; the prediction is always 0', pl + 6, yc - 8); }
      else if (w.length <= 3) { c.fillStyle = css('--ink'); c.fillText(w.map((v, i) => `a${i + 1} = ${v.toFixed(2)}`).join('   '), pl + 6, 14); } }
    const notes = {
      tone: p === 2 ? 'Two weights predict a sinusoid perfectly: the weights are \\(2\\cos\\omega \\approx 1.994\\) and \\(-1\\), exactly as the identity says.' : p > 2 ? 'Still perfect, but the extra weights aren’t zero: with \\(p &gt; 2\\) there are infinitely many perfect weight sets for a pure tone, and which one the solver lands on depends on tiny details (see the box below).' : 'One weight isn’t enough to follow the curve of a sinusoid.',
      aa: p <= 2 ? 'The first weights capture the smooth, slow part of the ringing.' : p < 10 ? 'Each extra pair of weights captures another resonance: watch the gain climb.' : 'Gain has levelled off: what remains is mostly the glottal kicks, which are new each period.',
      iy: '/i/ starts even higher than /ɑ/ (its very low \\(F_1\\) makes neighbouring samples almost equal), and its high \\(F_2/F_3\\) keep the gain climbing until about \\(p = 18\\).',
      s: 'About 8 dB at most: the hiss is mostly unpredictable. The weights still capture its spectral shape (where the noise energy sits).',
      noise: 'Zero gain at every order: there is nothing in the past to exploit.',
    };
    setMath($('lpNote'), notes[lpKey]);
  }
  segButtons('lpSig', SC_KEYS, lpKey, k => { lpKey = k; lpDraw(); });
  $('lpP').addEventListener('input', lpDraw);

  window.addEventListener('resize', () => { drawGame(); scDraw(); lpDraw(); });
  newRound(); scDraw(); lpDraw();
  // ---------------- Yule's pendulum pelted with peas ----------------
  const SUN = window.LP_SUNSPOTS;
  if (SUN && document.getElementById('yuleCanvas')) {
    let seed = 1927;
    const rnd = () => { seed = (seed * 1664525 + 1013904223) >>> 0; return seed / 4294967296; };
    const gauss = () => { let u = 0; while (!u) u = rnd(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rnd()); };
    const mean = SUN.x.reduce((s, v) => s + v, 0) / SUN.x.length;
    let peas = [];
    const newPeas = () => { peas = SUN.x.map(() => gauss()); };
    function yuleDraw() {
      const scale = +$('yuleShock').value / 100, n = SUN.x.length, [a1, a2] = SUN.a;
      if (scale === 0) $('yuleShockV').textContent = 'no peas'; else setMath($('yuleShockV'), scale.toFixed(1) + ' \\(\\times\\) Yule&rsquo;s estimate');
      const y = new Float64Array(n); y[0] = 60; y[1] = 60;                    // start displaced, as if just let go
      for (let t = 2; t < n; t++) y[t] = a1 * y[t - 1] + a2 * y[t - 2] + scale * SUN.shock_sd * peas[t];
      const { c, W, H } = setupCanvas($('yuleCanvas'));
      const pl = 34, pw = W - pl - 8, pt = 8, ph = H - 26, lo = -90, hi = 170;
      const X = i => pl + i / (n - 1) * pw, Y = v => pt + (hi - v) / (hi - lo) * ph;
      c.strokeStyle = css('--line'); c.fillStyle = css('--muted'); c.font = '10px Inter, sans-serif';
      [0, 50, 100, 150].forEach(v => { c.beginPath(); c.moveTo(pl, Y(v)); c.lineTo(pl + pw, Y(v)); c.stroke(); c.fillText(v, 4, Y(v) + 3); });
      [1750, 1800, 1850, 1900].forEach(yr => { const i = SUN.years.indexOf(yr); if (i >= 0) c.fillText(yr, X(i) - 12, H - 4); });
      if ($('yuleReal').checked) { c.strokeStyle = css('--accent-3'); c.globalAlpha = 0.55; c.lineWidth = 2; c.beginPath(); SUN.x.forEach((v, i) => i ? c.lineTo(X(i), Y(v)) : c.moveTo(X(i), Y(v))); c.stroke(); c.globalAlpha = 1; }
      c.strokeStyle = css('--accent'); c.lineWidth = 1.6; c.beginPath(); y.forEach((v, i) => { const vv = v + mean; i ? c.lineTo(X(i), Y(vv)) : c.moveTo(X(i), Y(vv)); }); c.stroke();
      c.fillStyle = css('--accent'); c.fillText('pelted pendulum (simulated)', pl + 6, pt + 10);
      if ($('yuleReal').checked) { c.fillStyle = css('--accent-3'); c.fillText('real sunspots', pl + 160, pt + 10); }
      $('yuleNote').textContent = scale === 0 ? 'No disturbances: the pendulum just rings down to rest at its 10.6-year period.'
        : 'Same two weights every time, new random peas each click: irregular cycles of varying length and height.';
    }
    newPeas();
    $('yuleNew').addEventListener('click', () => { newPeas(); yuleDraw(); });
    $('yuleShock').addEventListener('input', yuleDraw);
    $('yuleReal').addEventListener('change', yuleDraw);
    window.addEventListener('resize', yuleDraw);
    yuleDraw();
  }
})();
