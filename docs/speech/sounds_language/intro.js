// Demos for sounds_language.html (Module 2, page 1; each block runs only if its element is on the page):
//   #lexDemo   one ksp sentence sound by sound: waveform + spectrogram, phone tier, word tier; click to play (data: sentence.js)
//   #mpDemo    eight minimal pairs spoken by ksp, with a measurement per token (data: intro_data.js, audio/*.wav)
//   #tqDemo    one phoneme or two? English vs Hindi, answers computed from PHOIBLE median inventories
//   #ipaDemo   the IPA as a character set: consonant chart and vowel quadrilateral coloured by English / Hindi use
//   #featDemo  two phonemes as PHOIBLE feature vectors, differing features and Hamming distance
//   #g2pDemo   spelling vs sound: graphemes aligned with the phonemes they stand for (hand-written examples)
// Everything except the g2p examples is written by make_intro.py.
(function () {
  const $ = id => document.getElementById(id);
  const esc = s => String(s).replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' })[c]);
  const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
  const AUD = 'sounds_language/audio/';
  const D = window.INTRO;

  // ---------------------------------------------------------------- audio helpers
  let actx = null, curSrc = null, curEl = null;
  function stopAll() {
    if (curSrc) { try { curSrc.stop(); } catch (e) { /* already stopped */ } curSrc = null; }
    if (curEl) { try { curEl.pause(); } catch (e) { /* ignore */ } curEl = null; }
  }
  function playFile(f) {
    stopAll();
    try { curEl = new Audio(AUD + f); const p = curEl.play(); if (p && p.catch) p.catch(() => {}); } catch (e) { console.warn(e); }
  }
  function playBuf(y, sr, onend) {
    stopAll();
    try {
      actx = actx || new (window.AudioContext || window.webkitAudioContext)();
      if (actx.state === 'suspended') actx.resume();
      const buf = actx.createBuffer(1, y.length, sr); buf.getChannelData(0).set(y);
      const s = actx.createBufferSource(); s.buffer = buf; s.connect(actx.destination);
      if (onend) s.onended = onend;
      s.start(); curSrc = s;
      return actx.currentTime;
    } catch (e) { console.warn(e); return null; }
  }
  function setupCanvas(cv) {
    if (!cv.dataset.h) { cv.dataset.h = cv.getAttribute('height'); cv.style.height = cv.dataset.h + 'px'; }
    const dpr = window.devicePixelRatio || 1, W = cv.clientWidth || 600, H = +cv.dataset.h;
    cv.width = Math.round(W * dpr); cv.height = Math.round(H * dpr);
    const c = cv.getContext('2d'); c.setTransform(dpr, 0, 0, dpr, 0, 0); c.clearRect(0, 0, W, H);
    return { c, W, H, dpr };
  }
  function hexRGB(h) {
    h = (h || '').replace('#', '');
    if (h.length === 3) h = h.split('').map(ch => ch + ch).join('');
    const n = parseInt(h, 16);
    return isNaN(n) ? [128, 128, 128] : [(n >> 16) & 255, (n >> 8) & 255, n & 255];
  }
  function buttons(id, items, label, on) {
    $(id).innerHTML = items.map(x => `<button type="button">${label(x)}</button>`).join('');
    $(id).querySelectorAll('button').forEach((b, i) => b.addEventListener('click', () => on(i)));
  }
  const CLS_VAR = { vowel: '--accent', stop: '--accent-2', fricative: '--accent-3', nasal: '--teal', approximant: '--accent-4', affricate: '--accent-4' };

  // ---------------------------------------------------------------- 1. a sentence, sound by sound
  const L = window.LEX_SENT;
  if (L && $('lexDemo')) {
    const SR = L.sr, bin = atob(L.b64), N = bin.length >> 1, x = new Float32Array(N);
    for (let i = 0; i < N; i++) { let v = bin.charCodeAt(2 * i) | (bin.charCodeAt(2 * i + 1) << 8); if (v >= 32768) v -= 65536; x[i] = v / 32768; }
    const DUR = N / SR;
    const PH = L.phones.map(p => ({ s: p[0], e: p[1], arpa: p[2], ipa: p[3], cls: p[4], w: p[5] }));
    const WD = L.words.map((w, i) => ({ s: w[0], e: w[1], word: w[2], i }));
    WD.forEach(w => { w.ipa = PH.filter(p => p.w === w.i).map(p => p.ipa).join(''); });
    const BAND = { wave: [0, 54], spec: [58, 182], ph: [186, 216], wd: [220, 248] };
    const cv = $('lexCv');
    let sel = null, base = null, play0 = null, playSpan = null, raf = 0;

    function clip(a, b) {
      const i0 = Math.max(0, Math.round(a * SR)), i1 = Math.min(N, Math.round(b * SR)), y = x.slice(i0, i1), n = Math.min(64, y.length >> 1);
      for (let i = 0; i < n; i++) { const g = i / n; y[i] *= g; y[y.length - 1 - i] *= g; }
      return y;
    }
    // radix-2 FFT (in place)
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
    // wideband spectrogram: 8 ms Hann frames, 256-point FFT, one frame per pixel column, 70 dB range
    function spectrogram(cols) {
      const NF = 256, WL = Math.round(0.008 * SR), NB = 128, win = new Float32Array(WL), S = new Float32Array(cols * NB);
      for (let i = 0; i < WL; i++) win[i] = 0.5 - 0.5 * Math.cos(2 * Math.PI * i / (WL - 1));
      let mx = -1e9;
      const re = new Float32Array(NF), im = new Float32Array(NF);
      for (let c = 0; c < cols; c++) {
        const ctr = Math.round((c + 0.5) / cols * N), i0 = ctr - (WL >> 1);
        re.fill(0); im.fill(0);
        for (let i = 0; i < WL; i++) { const k = i0 + i; re[i] = (k >= 0 && k < N ? x[k] - (k > 0 ? 0.95 * x[k - 1] : 0) : 0) * win[i]; }
        fft(re, im);
        for (let b = 0; b < NB; b++) { const v = 10 * Math.log10(re[b] * re[b] + im[b] * im[b] + 1e-12); S[c * NB + b] = v; if (v > mx) mx = v; }
      }
      return { S, NB, mx };
    }
    function phoneAt(t) { return PH.find(p => t >= p.s && t < p.e && p.cls !== 'pause'); }
    function wordAt(t) { return WD.find(w => t >= w.s && t < w.e); }

    function drawBase() {
      const { c, W, H, dpr } = setupCanvas(cv);
      const tx = t => t / DUR * W;
      const ink = css('--ink'), muted = css('--muted'), line = css('--line');
      // waveform
      const [w0, w1] = BAND.wave, mid = (w0 + w1) / 2, amp = (w1 - w0) / 2;
      c.strokeStyle = css('--accent'); c.lineWidth = 1; c.beginPath();
      for (let px = 0; px < W; px++) {
        const i0 = Math.floor(px / W * N), i1 = Math.max(i0 + 1, Math.floor((px + 1) / W * N));
        let lo = 0, hi = 0; for (let i = i0; i < i1; i++) { if (x[i] < lo) lo = x[i]; if (x[i] > hi) hi = x[i]; }
        c.moveTo(px + 0.5, mid - hi * amp); c.lineTo(px + 0.5, mid - lo * amp + 0.5);
      }
      c.stroke();
      // spectrogram
      const cols = Math.max(50, Math.min(1200, Math.round(W))), sp = spectrogram(cols), rows = BAND.spec[1] - BAND.spec[0];
      const off = document.createElement('canvas'); off.width = cols; off.height = sp.NB;
      const oc = off.getContext('2d'), img = oc.createImageData(cols, sp.NB);
      const c0 = hexRGB(css('--surface')), c1 = hexRGB(css('--ink'));
      for (let cc = 0; cc < cols; cc++) for (let b = 0; b < sp.NB; b++) {
        let v = (sp.S[cc * sp.NB + b] - (sp.mx - 70)) / 70; v = Math.max(0, Math.min(1, v)); v = Math.pow(v, 1.3);
        const o = ((sp.NB - 1 - b) * cols + cc) * 4;
        img.data[o] = c0[0] + v * (c1[0] - c0[0]); img.data[o + 1] = c0[1] + v * (c1[1] - c0[1]); img.data[o + 2] = c0[2] + v * (c1[2] - c0[2]); img.data[o + 3] = 255;
      }
      oc.putImageData(img, 0, 0);
      c.imageSmoothingEnabled = true;
      c.drawImage(off, 0, BAND.spec[0], W, rows);
      c.fillStyle = muted; c.font = '10px Inter, sans-serif'; c.textBaseline = 'top';
      c.fillText('8 kHz', 3, BAND.spec[0] + 1); c.textBaseline = 'bottom'; c.fillText('0', 3, BAND.spec[1] - 1);
      // tiers
      for (const [p, band, label] of [[PH, BAND.ph, 'ipa'], [WD, BAND.wd, 'word']]) {
        for (const it of p) {
          const a = tx(it.s), b = tx(it.e), y0 = band[0], h = band[1] - band[0];
          if (it.cls === 'pause') continue;
          const col = label === 'ipa' ? css(CLS_VAR[it.cls] || '--accent-4') : css('--accent-4');
          c.globalAlpha = 0.16; c.fillStyle = col; c.fillRect(a, y0, b - a, h); c.globalAlpha = 1;
          c.strokeStyle = col; c.lineWidth = 1; c.strokeRect(a + 0.5, y0 + 0.5, Math.max(1, b - a - 1), h - 1);
          const txt = label === 'ipa' ? it.ipa : it.word;
          c.font = label === 'ipa' ? '15px "Noto Sans", "DejaVu Sans", sans-serif' : '12px Inter, sans-serif';
          if (c.measureText(txt).width < b - a - 2) { c.fillStyle = ink; c.textAlign = 'center'; c.textBaseline = 'middle'; c.fillText(txt, (a + b) / 2, y0 + h / 2 + 1); c.textAlign = 'left'; }
        }
      }
      // boundaries through the signal bands
      c.strokeStyle = line; c.lineWidth = 1;
      for (const p of PH) { const a = Math.round(tx(p.s)) + 0.5; c.beginPath(); c.moveTo(a, BAND.wave[0]); c.lineTo(a, BAND.spec[1]); c.stroke(); }
      base = document.createElement('canvas'); base.width = cv.width; base.height = cv.height;
      base.getContext('2d').drawImage(cv, 0, 0);
      overlay();
    }
    function overlay(tcur) {
      if (!base) return;
      const c = cv.getContext('2d'), dpr = cv.width / (cv.clientWidth || 1), W = cv.clientWidth || 600, H = +cv.dataset.h;
      c.setTransform(1, 0, 0, 1, 0, 0); c.clearRect(0, 0, cv.width, cv.height); c.drawImage(base, 0, 0);
      c.setTransform(dpr, 0, 0, dpr, 0, 0);
      if (sel) {
        c.globalAlpha = 0.18; c.fillStyle = css('--accent-3'); c.fillRect(sel.s / DUR * W, 0, (sel.e - sel.s) / DUR * W, H); c.globalAlpha = 1;
      }
      if (tcur !== undefined) { c.strokeStyle = css('--accent-2'); c.lineWidth = 1.5; c.beginPath(); c.moveTo(tcur / DUR * W, 0); c.lineTo(tcur / DUR * W, H); c.stroke(); }
    }
    function play(a, b, item) {
      sel = item ? { s: a, e: b } : null;
      markChips(item);
      cancelAnimationFrame(raf);
      const t0 = playBuf(clip(a, b), SR, () => { play0 = null; overlay(); });
      play0 = t0; playSpan = [a, b];
      const tick = () => {
        if (play0 === null || !actx) { overlay(); return; }
        const t = playSpan[0] + (actx.currentTime - play0);
        if (t > playSpan[1]) { overlay(); return; }
        overlay(t); raf = requestAnimationFrame(tick);
      };
      if (t0 !== null) raf = requestAnimationFrame(tick); else overlay();
    }
    function markChips(item) {
      document.querySelectorAll('#lexPhones .chip, #lexWords .chip').forEach(ch => ch.classList.toggle('on', !!item && ch._item === item));
    }
    cv.addEventListener('click', ev => {
      const r = cv.getBoundingClientRect(), t = (ev.clientX - r.left) / r.width * DUR, y = ev.clientY - r.top;
      const it = (y >= BAND.ph[0] && y < BAND.ph[1]) ? phoneAt(t) : wordAt(t);
      if (it) play(it.s, it.e, it);
    });
    cv.style.cursor = 'pointer';
    $('lexPlay').addEventListener('click', () => play(0, DUR, null));
    // chips
    const ph = $('lexPhones');
    PH.filter(p => p.cls !== 'pause').forEach(p => {
      const b = document.createElement('button'); b.type = 'button'; b.className = 'chip ipx c-' + p.cls; b._item = p;
      b.innerHTML = `${esc(p.ipa)}<small>${esc(p.arpa)}</small>`; b.title = p.cls;
      b.addEventListener('click', () => play(p.s, p.e, p)); ph.appendChild(b);
    });
    const wd = $('lexWords');
    WD.forEach(w => {
      const b = document.createElement('button'); b.type = 'button'; b.className = 'chip w'; b._item = w;
      b.innerHTML = `${esc(w.word)} <span class="ipx" style="color:var(--muted)">/${esc(w.ipa)}/</span>`;
      b.addEventListener('click', () => play(w.s, w.e, w)); wd.appendChild(b);
    });
    $('lexLegend').innerHTML = ['vowel', 'stop', 'fricative', 'nasal', 'approximant'].map(k =>
      `<span><i style="background:var(${CLS_VAR[k]})"></i>${k}</span>`).join('') + '<span>top: waveform; middle: spectrogram (0&ndash;8 kHz); then phones and words</span>';
    drawBase();
    let rt = 0;
    window.addEventListener('resize', () => { clearTimeout(rt); rt = setTimeout(drawBase, 150); });
  }

  if (!D) return;

  // ---------------------------------------------------------------- 2. minimal pairs
  if ($('mpGrid')) {
    $('mpGrid').innerHTML = D.mp.map((m, i) => `<div class="mpcard"><div class="ct ipx">/${esc(m.contrast.replace(' · ', '/ · /'))}/</div><div class="what">${esc(m.what)}</div>` +
      m.words.map((w, j) => `<div class="mpword"><button type="button" data-f="${esc(w.file)}" aria-label="play ${esc(w.w)}">▶</button><b>${esc(w.w)}</b>` +
        `<span class="ipx">${esc(w.ipa)}</span><span class="m" title="${esc(w.text)} (utterance ${esc(w.utt)})">${esc(w.m)} &middot; ${esc(w.utt)}</span></div>`).join('') + '</div>').join('');
    $('mpGrid').querySelectorAll('button[data-f]').forEach(b => b.addEventListener('click', () => playFile(b.dataset.f)));
  }

  // ---------------------------------------------------------------- 4. one phoneme or two?
  if ($('tqDemo')) {
    const Q = D.quiz, guess = Q.map(() => ({ eng: null, hind: null }));
    const LN = { eng: 'English', hind: 'Hindi' };
    let cur = 0;
    function score() {
      let n = 0, ok = 0;
      Q.forEach((q, i) => ['eng', 'hind'].forEach(k => { if (guess[i][k]) { n++; if (guess[i][k] === q[k].v) ok++; } }));
      $('tqScore').innerHTML = `<div>your answers<b>${ok} / ${n} right</b></div><div>still to answer<b>${2 * Q.length - n} of ${2 * Q.length}</b></div>`;
    }
    function side(q, k) {
      const a = q[k], g = guess[cur][k], src = D.inv[k];
      let h = `<div class="side"><div class="lang">${LN[k]}</div><div class="src">PHOIBLE inventory ${src.id} (${esc(src.src)})</div>` +
        `<div class="seg" style="margin-top:6px;"><button type="button" data-k="${k}" data-v="one"${g === 'one' ? ' class="on"' : ''}>one phoneme</button>` +
        `<button type="button" data-k="${k}" data-v="two"${g === 'two' ? ' class="on"' : ''}>two phonemes</button></div>`;
      if (g) {
        const listed = a.listed.length ? a.listed.map(s => `/${esc(s)}/`).join(' ') : 'neither symbol';
        h += `<div class="ans"><span class="${g === a.v ? 'ok' : 'bad'}">${g === a.v ? '✓ right' : '✗ not quite'}: ${a.v === 'two' ? 'two phonemes' : 'one phoneme'}${a.marg ? ' (one of them marginal)' : ''}.</span> ` +
          `${a.ex}<br><span class="warn">Inventory lists: <span class="ipx">${listed}</span>. ${a.agree} of ${a.n} ${LN[k]} inventories in PHOIBLE give the same answer.</span></div>`;
      }
      return h + '</div>';
    }
    function show(i) {
      cur = i;
      $('tqBtns').querySelectorAll('button').forEach((b, k) => b.classList.toggle('on', k === i));
      const q = Q[i];
      $('tqOut').innerHTML = `<div class="tqhead ipx">[${esc(q.a)}] &nbsp;vs&nbsp; [${esc(q.b)}] <span class="warn">${esc(q.what)}</span></div><div class="tqgrid">${side(q, 'eng')}${side(q, 'hind')}</div>`;
      $('tqOut').querySelectorAll('button[data-k]').forEach(b => b.addEventListener('click', () => { guess[cur][b.dataset.k] = b.dataset.v; show(cur); score(); }));
    }
    buttons('tqBtns', Q, q => `<span class="ipx">${esc(q.a)} · ${esc(q.b)}</span>`, show);
    show(0); score();
  }

  // ---------------------------------------------------------------- 5. IPA chart
  if ($('ipaDemo')) {
    const C = D.chart, E = C.langs.eng, Hd = C.langs.hind;
    const MODES = [['both', 'English and Hindi'], ['eng', 'English only'], ['hind', 'Hindi only']];
    let mode = 'both';
    const isMarg = arr => arr && arr.length && arr.every(p => p.endsWith('*'));
    const clean = arr => (arr || []).map(p => p.replace('*', ' (marginal)')).join(', ');
    function cls(inE, inH) {
      if (mode === 'eng') return inE ? 'en' : '';
      if (mode === 'hind') return inH ? 'hi' : '';
      return inE && inH ? 'both' : inE ? 'en' : inH ? 'hi' : '';
    }
    function symHTML(s, eArr, hArr) {
      const k = cls(!!eArr, !!hArr), mg = (k === 'en' && isMarg(eArr)) || (k === 'hi' && isMarg(hArr)) || (k === 'both' && isMarg(eArr) && isMarg(hArr));
      const tip = `${s}: English ${eArr ? clean(eArr) : '—'}; Hindi ${hArr ? clean(hArr) : '—'}`;
      return `<span class="sym ${k}${mg ? ' mg' : ''}" title="${esc(tip)}">${esc(s)}</span>`;
    }
    function draw() {
      $('ipaView').querySelectorAll('button').forEach((b, k) => b.classList.toggle('on', MODES[k][0] === mode));
      let h = '<thead><tr><th></th>' + C.places.map(p => `<th>${esc(p)}</th>`).join('') + '</tr></thead><tbody>';
      C.rows.forEach(([m, cells]) => {
        h += `<tr><th>${esc(m)}</th>` + cells.map(cell => cell ? `<td>${cell.map(s => symHTML(s, E.c[s], Hd.c[s])).join(' ')}</td>` : '<td class="na"></td>').join('') + '</tr>';
      });
      $('ipaCons').innerHTML = h + '</tbody>';
      // vowel quadrilateral (SVG)
      const ROUND = new Set('yʉuʏʊøɵoœɞɔɶɒ'.split(''));
      const X = (h_, b) => { const y = 22 + h_ * (210 / 6), left = 46 + (y - 22) / 210 * 110; return [left + b / 4 * (390 - left), y]; };
      let s = '<svg class="vquad" viewBox="0 0 430 250" role="img" aria-label="Vowel chart"><polygon points="' +
        [X(0, 0), X(0, 4), X(6, 4), X(6, 0)].map(p => p.join(',')).join(' ') + '" fill="none" stroke="var(--line-strong)"/>';
      [2, 4].forEach(r => { const a = X(r, 0), b = X(r, 4); s += `<line x1="${a[0]}" y1="${a[1]}" x2="${b[0]}" y2="${b[1]}" stroke="var(--line)"/>`; });
      { const a = X(0, 2), b = X(6, 2); s += `<line x1="${a[0]}" y1="${a[1]}" x2="${b[0]}" y2="${b[1]}" stroke="var(--line)"/>`; }
      C.vowels.forEach(([v, hh, bb]) => {
        let [px, py] = X(hh, bb);
        if (v !== 'ə') px += ROUND.has(v) ? 11 : -11;
        const k = cls(!!E.v[v], !!Hd.v[v]);
        const fill = { en: 'color-mix(in srgb, var(--accent) 30%, transparent)', hi: 'color-mix(in srgb, var(--accent-2) 30%, transparent)', both: 'color-mix(in srgb, var(--accent-3) 50%, transparent)' }[k] || 'none';
        const tip = `${v}: English ${E.v[v] ? clean(E.v[v]) : '—'}; Hindi ${Hd.v[v] ? clean(Hd.v[v]) : '—'}`;
        s += `<g><title>${esc(tip)}</title><circle cx="${px}" cy="${py}" r="10" fill="${fill}"/>` +
          `<text x="${px}" y="${py + 5}" text-anchor="middle" font-size="15" font-family="Noto Sans, DejaVu Sans, sans-serif" fill="${k ? 'var(--ink)' : 'var(--muted)'}">${esc(v)}</text></g>`;
      });
      s += '<text x="46" y="14" font-size="10" fill="var(--muted)">front</text><text x="370" y="14" font-size="10" fill="var(--muted)">back</text>' +
        '<text x="400" y="26" font-size="10" fill="var(--muted)">close</text><text x="400" y="236" font-size="10" fill="var(--muted)">open</text></svg>';
      $('ipaVow').innerHTML = s;
      $('ipaNote').innerHTML = `English: PHOIBLE inventory ${E.id} (${esc(E.src)}${E.dial ? ', ' + esc(E.dial) : ''}); its diphthongs, not placed: <span class="ipx">${E.diph.map(esc).join(' ') || '—'}</span>. ` +
        `Hindi: inventory ${Hd.id} (${esc(Hd.src)}${Hd.dial ? ', ' + esc(Hd.dial) : ''}). Long, nasalised, aspirated and breathy variants are pooled in the cell of their base symbol; ` +
        `<span class="ipx">s̪|s</span>-type entries (dental or alveolar, undecided) go to the alveolar cell.`;
    }
    buttons('ipaView', MODES, m => m[1], i => { mode = MODES[i][0]; draw(); });
    draw();
  }

  // ---------------------------------------------------------------- 6. feature bit strips
  if ($('featDemo')) {
    const F = D.feat, BY = {};
    F.syms.forEach(s => { BY[s.s] = s; });
    const lab = s => `${s.s}${s.s === 'b̤' ? ' (bʱ)' : ''}${s.s === 'ɡ̤' ? ' (ɡʱ)' : ''}${s.s === 'd̪̤' ? ' (d̪ʱ)' : ''}${s.s === 'ɖ̤' ? ' (ɖʱ)' : ''}${s.en || s.hi ? '  ' + (s.en ? 'E' : '') + (s.hi ? 'H' : '') : ''}`;
    ['featA', 'featB'].forEach(id => { $(id).innerHTML = F.syms.map(s => `<option value="${esc(s.s)}">${esc(lab(s))}</option>`).join(''); });
    const QUICK = [['p', 'b'], ['p', 'pʰ'], ['t̪', 'ʈ'], ['b', 'b̤'], ['s', 'ʃ'], ['i', 'ɪ'], ['m', 'b'], ['n', 'ŋ'], ['v', 'w'], ['p', 'a']];
    buttons('featQuick', QUICK, q => `<span class="ipx">${esc(q[0])} · ${esc(q[1])}</span>`, i => { $('featA').value = QUICK[i][0]; $('featB').value = QUICK[i][1]; draw(); });
    const sign = v => v === '+' ? 'p' : v === '0' ? 'z' : '';
    function draw() {
      const A = BY[$('featA').value], B = BY[$('featB').value];
      $('featQuick').querySelectorAll('button').forEach((b, k) => b.classList.toggle('on', QUICK[k][0] === A.s && QUICK[k][1] === B.s));
      const diff = F.feats.map((_, i) => A.v[i] !== B.v[i]);
      let h = '<thead><tr><th style="writing-mode:horizontal-tb;transform:none;"></th>' + F.feats.map((f, i) => `<th title="PHOIBLE: ${esc(F.keys[i])}">${esc(f)}</th>`).join('') + '</tr></thead><tbody>';
      [A, B].forEach(S => { h += `<tr><td class="rowh ipx">/${esc(S.s)}/</td>` + S.v.map((v, i) => `<td class="${sign(v)}${diff[i] ? ' d' : ''}">${esc(v)}</td>`).join('') + '</tr>'; });
      $('featTable').innerHTML = h + '</tbody>';
      const nd = diff.filter(Boolean).length, flips = F.feats.filter((_, i) => diff[i] && A.v[i] !== '0' && B.v[i] !== '0').length;
      const names = F.feats.filter((_, i) => diff[i]);
      $('featOut').innerHTML = `<div>Hamming distance<b>${nd}</b></div><div>of which + / &minus; flips<b>${flips}</b></div>` +
        `<div>differing features<b style="font-family:Inter,sans-serif;font-size:0.82rem;">${names.length ? names.map(esc).join(', ') : 'none: same vector'}</b></div>` +
        `<div>PHOIBLE rows behind the vectors<b>${A.n} / ${B.n}</b></div>`;
    }
    $('featA').addEventListener('change', draw); $('featB').addEventListener('change', draw);
    $('featA').value = 'p'; $('featB').value = 'b'; draw();
  }

  // ---------------------------------------------------------------- 7. spelling vs sound (hand-written; English in Southern British transcription)
  const WORDS = [
    { w: 'though', lang: 'English', map: [['th', 'ð'], ['ough', 'əʊ']], note: 'The same ⟨ough⟩ has at least five pronunciations: try <em>through</em> and <em>tough</em>.' },
    { w: 'through', lang: 'English', map: [['th', 'θ'], ['r', 'r'], ['ough', 'uː']], note: 'The same ⟨ough⟩ as <em>though</em>, a different vowel; ⟨th⟩ is voiceless here.' },
    { w: 'tough', lang: 'English', map: [['t', 't'], ['ou', 'ʌ'], ['gh', 'f']], note: '⟨gh⟩ is /f/ here and silent in <em>though</em>.' },
    { w: 'knight', lang: 'English', map: [['k', ''], ['n', 'n'], ['igh', 'aɪ'], ['t', 't']], note: 'The ⟨k⟩ and ⟨gh⟩ were once pronounced; the spelling kept them.' },
    { w: 'city', lang: 'English', map: [['c', 's'], ['i', 'ɪ'], ['t', 't'], ['y', 'i']], note: '⟨c⟩ is /s/ here and /k/ in <em>cat</em>.' },
    { w: 'phone', lang: 'English', map: [['ph', 'f'], ['o', 'əʊ'], ['n', 'n'], ['e', '']], note: 'The final ⟨e⟩ is silent but changes how the ⟨o⟩ is read.' },
    { w: 'कमल', lang: 'Hindi', gloss: 'lotus', map: [['क', 'k ə'], ['म', 'm ə'], ['ल', 'l']], note: 'Each letter carries the inherent vowel /ə/, but the last one is not pronounced (schwa deletion).' },
    { w: 'रचना', lang: 'Hindi', gloss: 'creation', map: [['र', 'r ə'], ['च', 't͡ʃ'], ['ना', 'n aː']], note: 'Here the <em>middle</em> inherent vowel is dropped: written <em>ra-ca-nā</em>, said /rət͡ʃnaː/.' },
    { w: 'फल', lang: 'Hindi', gloss: 'fruit', map: [['फ', 'pʰ ə'], ['ल', 'l']], note: 'One letter for the aspirated stop /pʰ/, which English would need two letters for.' },
    { w: 'অসম', lang: 'Assamese', gloss: 'Assam', map: [['অ', 'ɔ'], ['স', 'x ɔ'], ['ম', 'm']], note: 'The letter ⟨স⟩ (an <em>s</em> in most scripts) is said /x/ in Assamese; the inherent vowel is /ɔ/.' },
    { w: 'தங்கம்', lang: 'Tamil', gloss: 'gold', map: [['த', 't̪ a'], ['ங்', 'ŋ'], ['க', 'ɡ a'], ['ம்', 'm']], note: 'The letter <span lang="ta">க</span> is said [ɡ] after a nasal; the dot (<em>puḷḷi</em>) marks a consonant with no vowel.' },
  ];
  function showWord(i) {
    const W = WORDS[i];
    $('g2pBtns').querySelectorAll('button').forEach((b, k) => b.classList.toggle('on', k === i));
    const tiles = W.map.map(([g, ph]) => `<div class="tile"><span class="g">${esc(g)}</span>` +
      `<span class="p ipx${ph ? '' : ' silent'}">${ph ? esc(ph.replace(/ /g, '')) : '∅'}</span></div>`).join('');
    const phonemes = W.map.flatMap(([, ph]) => ph ? ph.split(' ') : []);
    const letters = W.lang === 'English' ? `${W.w.length} letters, ` : '';
    $('g2pOut').innerHTML = `<div class="pairhead">${esc(W.w)} <span class="warn">${W.lang}${W.gloss ? ' &middot; &lsquo;' + W.gloss + '&rsquo;' : ''}</span></div>
      <div class="tiles">${tiles}</div>
      <div class="g2pstats">${letters}${W.map.length} graphemes, ${phonemes.length} phonemes /${esc(phonemes.join(' '))}/. ${W.note}</div>`;
  }
  if ($('g2pDemo')) { buttons('g2pBtns', WORDS, W => esc(W.w), showWord); showWord(0); }
})();
