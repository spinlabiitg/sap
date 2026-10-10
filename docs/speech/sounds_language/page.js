// Live demo for phoneme_inventories.html (Module 2, page 2): pick a language and one of its PHOIBLE inventories, see it as a consonant chart and a vowel chart.
// Data: sounds_language/inventories.js (window.PHOIBLE_DEMO), written by make_figures.py from PHOIBLE 2.0 + Glottolog (CC BY 4.0).
(() => {
  const D = window.PHOIBLE_DEMO;
  const $ = id => document.getElementById(id);
  if (!D || !$('phLang')) return;
  const esc = s => String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  const HEIGHTS = ['close', 'near-close', 'close-mid', 'mid', 'open-mid', 'near-open', 'open'];
  const BACK = ['front', 'central', 'back'];
  const GROUPS = ['Indo-Aryan', 'Dravidian', 'Austroasiatic', 'Sino-Tibetan', 'Other', 'World'];
  const GLABEL = { 'Other': 'Other families (India)', 'World': 'For comparison (outside India)' };
  const QUICK = ['hind1269', 'beng1280', 'assa1263', 'mara1378', 'tami1289', 'telu1262', 'nucl1305', 'mala1464', 'sant1410',
    'mund1320', 'khas1269', 'mani1292', 'bodo1269', 'stan1293', 'mand1415', 'hawa1245'];
  const byG = {};
  D.langs.forEach((l, i) => { byG[l.g] = i; });

  // language picker, grouped by family
  const sel = $('phLang');
  GROUPS.forEach(gname => {
    const og = document.createElement('optgroup');
    og.label = GLABEL[gname] || gname;
    D.langs.forEach((l, i) => {
      if (l.fam !== gname) return;
      const o = document.createElement('option');
      o.value = i; o.textContent = `${l.name} (${l.inv.length})`;
      og.appendChild(o);
    });
    if (og.children.length) sel.appendChild(og);
  });
  // quick picks
  const qp = $('phQuick');
  QUICK.forEach(g => {
    if (!(g in byG)) return;
    const b = document.createElement('button');
    b.type = 'button'; b.textContent = D.langs[byG[g]].name.replace(' Chinese', '').replace('Standard ', '');
    b.dataset.i = byG[g];
    b.addEventListener('click', () => { sel.value = b.dataset.i; pickLang(+b.dataset.i); });
    qp.appendChild(b);
  });

  let cur = null;
  function pickLang(i) {
    cur = D.langs[i];
    [...qp.children].forEach(b => b.classList.toggle('on', +b.dataset.i === i));
    const row = $('phSrc');
    row.innerHTML = '';
    cur.inv.forEach(inv => {
      const b = document.createElement('button');
      b.type = 'button';
      b.innerHTML = `${esc(inv.s)} <span class="cnt">${inv.C}C · ${inv.V}V${inv.T ? ' · ' + inv.T + 'T' : ''}</span>` + (inv.id === cur.pick ? ' <span class="star" title="the inventory used for the statistics on this page">★</span>' : '');
      b.addEventListener('click', () => show(inv));
      b.dataset.id = inv.id;
      row.appendChild(b);
    });
    show(cur.inv.find(v => v.id === cur.pick) || cur.inv[0]);
  }

  const cell = (g, m) => m ? `<span class="marg" title="marginal (loanwords, rare, or uncertain) in this source">(${esc(g)})</span>` : `<span>${esc(g)}</span>`;

  function show(inv) {
    [...$('phSrc').children].forEach(b => b.classList.toggle('on', +b.dataset.id === inv.id));
    // info line
    const fam = cur.fam === 'World' || cur.fam === 'Other' ? cur.family : cur.fam === cur.family ? cur.fam : `${cur.fam} (${cur.family})`;
    $('phInfo').innerHTML =
      `<div>language<b>${esc(cur.name)}</b></div><div>family (Glottolog)<b>${esc(fam)}</b></div>` +
      `<div>consonants · vowels · tones<b>${inv.C} · ${inv.V} · ${inv.T}</b></div>` +
      `<div>countries (Glottolog)<b>${esc((cur.ctry || '—').replace(/;/g, ', '))}</b></div>`;
    $('phSrcName').innerHTML = `<b>Source:</b> ${esc(D.src[inv.s] || inv.s)} · PHOIBLE inventory <a href="https://phoible.org/inventories/view/${inv.id}" target="_blank" rel="noopener">${inv.id}</a>` +
      ` · listed as &ldquo;${esc(inv.n)}&rdquo;${inv.d ? ', ' + esc(inv.d) : ''}`;
    // feature badges
    $('phBadges').innerHTML = D.feats.map((f, k) => `<span class="badge ${inv.f[k] ? 'yes' : 'no'}">${inv.f[k] ? '✓' : '✗'} ${esc(f)}</span>`).join('');
    // consonant chart: drop empty rows and columns to stay compact
    const grid = {};
    inv.c.forEach(([g, mi, pi, m]) => { (grid[mi + ',' + pi] = grid[mi + ',' + pi] || []).push(cell(g, m)); });
    const cols = D.places.map((_, pi) => pi).filter(pi => D.manners.some((_, mi) => grid[mi + ',' + pi]));
    const rows = D.manners.map((_, mi) => mi).filter(mi => cols.some(pi => grid[mi + ',' + pi]));
    let h = '<table class="ipa"><thead><tr><th></th>' + cols.map(pi => `<th class="${D.places[pi] === 'Retroflex' ? 'rx' : ''}">${D.places[pi]}</th>`).join('') + '</tr></thead><tbody>';
    rows.forEach(mi => {
      h += `<tr><th>${D.manners[mi]}</th>` + cols.map(pi => `<td class="${D.places[pi] === 'Retroflex' ? 'rx' : ''}">${(grid[mi + ',' + pi] || []).join(' ')}</td>`).join('') + '</tr>';
    });
    h += '</tbody></table>';
    if (inv.o.length) h += `<p class="ipanote">Other consonants (clicks, double articulations, unusual symbols): <span class="ipaglyphs">${inv.o.map(([g, m]) => cell(g, m)).join(' ')}</span></p>`;
    $('phCons').innerHTML = h;
    // vowel chart
    const vg = {};
    inv.v.forEach(([g, hi, bi, m]) => { (vg[hi + ',' + bi] = vg[hi + ',' + bi] || []).push(cell(g, m)); });
    const vrows = HEIGHTS.map((_, hi) => hi).filter(hi => BACK.some((_, bi) => vg[hi + ',' + bi]));
    let v = '<table class="ipa vow"><thead><tr><th></th>' + BACK.map(b => `<th>${b}</th>`).join('') + '</tr></thead><tbody>';
    vrows.forEach(hi => { v += `<tr><th>${HEIGHTS[hi]}</th>` + BACK.map((_, bi) => `<td>${(vg[hi + ',' + bi] || []).join(' ')}</td>`).join('') + '</tr>'; });
    v += '</tbody></table>';
    if (inv.x.length) v += `<p class="ipanote">Diphthongs and other vowels: <span class="ipaglyphs">${inv.x.map(([g, m]) => cell(g, m)).join(' ')}</span></p>`;
    v += `<p class="ipanote">Tones: ${inv.t.length ? '<span class="ipaglyphs">' + inv.t.map(([g, m]) => cell(g, m)).join(' ') + '</span>' : (['ea', 'upsid', 'saphon', 'er'].includes(inv.s) ? 'none listed (this source never lists tones)' : 'none listed')}</p>`;
    $('phVow').innerHTML = v;
  }

  sel.addEventListener('change', () => pickLang(+sel.value));
  const start = byG['assa1263'] !== undefined ? byG['assa1263'] : 0;
  sel.value = start;
  pickLang(start);
})();
