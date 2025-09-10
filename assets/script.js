// assets/script.js — debug-friendly version
function el(tag, attrs = {}, ...kids) {
  const n = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === 'class') n.className = v;
    else if (k === 'html') n.innerHTML = v;
    else n.setAttribute(k, v);
  }
  kids.forEach(k => k != null && n.append(k));
  return n;
}

function banner(msg) {
  const b = el('div', { class: 'error-banner', style: 'background:#fee2e2;color:#991b1b;padding:10px 14px;margin:12px;border-radius:8px;border:1px solid #fecaca' }, msg);
  document.body.prepend(b);
}

async function loadJSON(path) {
  try {
    const r = await fetch(path + '?v=' + Date.now(), { cache: 'no-store' });
    if (!r.ok) throw new Error(`HTTP ${r.status} for ${path}`);
    return await r.json();
  } catch (e) {
    console.error('Failed to fetch JSON:', e);
    banner(`Could not load ${path}. See console for details.`);
    return null;
  }
}

function setHeader(meta) {
  document.getElementById('title').textContent = meta?.title || 'Project Supplementary Materials';
  const bits = [];
  if (meta?.authors) bits.push(meta.authors);
  if (meta?.conference) bits.push(meta.conference);
  document.getElementById('sub').textContent = bits.join(' · ');
  const nav = document.getElementById('nav'); nav.innerHTML = '';
  if (meta?.paper_link) nav.append(el('a', { href: meta.paper_link, target: '_blank', rel: 'noopener' }, 'Paper'));
  if (meta?.code_link) nav.append(el('a', { href: meta.code_link, target: '_blank', rel: 'noopener' }, 'Code'));
  if (meta?.doi_link) nav.append(el('a', { href: meta.doi_link, target: '_blank', rel: 'noopener' }, 'DOI/Dataset'));
}

function guessMime(file) {
  const f = (file || '').toLowerCase();
  if (f.endsWith('.wav')) return 'audio/wav';
  if (f.endsWith('.mp3')) return 'audio/mpeg';
  if (f.endsWith('.ogg')) return 'audio/ogg';
  if (f.endsWith('.au'))  return 'audio/basic';
  if (f.endsWith('.png')) return 'image/png';
  if (f.endsWith('.jpg') || f.endsWith('.jpeg')) return 'image/jpeg';
  return '';
}

function render(sections) {
  const host = document.getElementById('content'); host.innerHTML = '';
  if (!Array.isArray(sections)) {
    banner('Manifest loaded but has no "sections" array.');
    console.warn('Manifest "sections" missing or not an array:', sections);
    return;
  }
  sections.forEach(sec => {
    const s = el('section', { class: 'section' });
    s.append(el('h2', {}, sec.title || 'Section'));
    if (sec.description) s.append(el('p', { class: 'desc' }, sec.description));
    const grid = el('div', { class: 'grid' });

    (sec.items || []).forEach(item => {
      const card = el('div', { class: 'card' });
      card.append(el('h3', {}, item.title || (item.type === 'audio' ? 'Audio sample' : item.type === 'plot' ? 'Interactive plot' : 'Figure')));

      if (item.type === 'audio') {
        const mime = item.mime || guessMime(item.file) || 'audio/wav';
        const a = el('audio', { controls: '', preload: 'none' });
        a.append(el('source', { src: item.file, type: mime }));
        a.append('Your browser does not support the audio element.');
        card.append(a);

      } else if (item.type === 'plot') {
        const f = el('iframe', { class: 'plot', src: item.file, loading: 'lazy', title: item.title || 'Interactive plot' });
        card.append(f);

      } else if (item.type === 'image') {
        card.append(el('img', { src: item.file, alt: item.title || 'figure', style: 'max-width:100%;border-radius:10px;display:block' }));
      }

      if (item.caption) card.append(el('p', {}, item.caption));
      grid.append(card);
    });

    s.append(grid); host.append(s);
  });
}

(async () => {
  const m = await loadJSON('data/manifest.json');
  if (!m) return;                   // already bannered
  setHeader(m.project || {});
  render(m.sections || []);
})();
