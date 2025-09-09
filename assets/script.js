
async function loadManifest() {
  const res = await fetch('data/manifest.json');
  if (!res.ok) { console.warn('No manifest.json found or invalid path.'); return null; }
  return await res.json();
}

function el(tag, attrs={}, ...children) {
  const node = document.createElement(tag);
  for (const [k,v] of Object.entries(attrs)) {
    if (k === 'class') node.className = v;
    else if (k.startsWith('on') && typeof v === 'function') node.addEventListener(k.substring(2), v);
    else node.setAttribute(k, v);
  }
  for (const c of children) node.append(c);
  return node;
}

function renderHeader(project) {
  const title = project?.title || "Project Supplementary Materials";
  const authors = project?.authors ? ` · ${project.authors}` : "";
  const conf = project?.conference ? ` — ${project.conference}` : "";
  document.getElementById('title').textContent = title;
  document.getElementById('sub').textContent = (authors + conf).trim();
  const nav = document.getElementById('nav');
  if (project?.paper_link) nav.append(el('a', {href: project.paper_link, target: '_blank', rel: 'noopener'}, 'Paper'));
  if (project?.code_link) nav.append(el('a', {href: project.code_link, target: '_blank', rel: 'noopener'}, 'Code'));
  if (project?.doi_link) nav.append(el('a', {href: project.doi_link, target: '_blank', rel: 'noopener'}, 'DOI / Dataset'));
}

function renderSections(sections=[]) {
  const container = document.getElementById('content');
  container.innerHTML = '';
  sections.forEach(sec => {
    const secEl = el('section', {class: 'section'});
    secEl.append(el('h2', {}, sec.title || 'Section'));
    if (sec.description) secEl.append(el('p', {class: 'desc'}, sec.description));
    const grid = el('div', {class: 'grid'});
    (sec.items || []).forEach(item => {
      const card = el('div', {class: 'card'});
      card.append(el('h3', {}, item.title || (item.type === 'audio' ? 'Audio sample' : 'Interactive plot')));
      if (item.type === 'audio') {
        const audio = el('audio', {controls: '', preload: 'none'});
        const src = el('source', {src: item.file, type: 'audio/wav'});
        audio.append(src);
        card.append(audio);
      } else if (item.type === 'plot') {
        const iframe = el('iframe', {class: 'plot', src: item.file, loading: 'lazy'});
        card.append(iframe);
      }
      if (item.caption) card.append(el('p', {}, item.caption));
      grid.append(card);
    });
    secEl.append(grid);
    container.append(secEl);
  });
}

(async () => {
  const manifest = await loadManifest();
  if (!manifest) {
    renderHeader({title: 'Project Supplementary Materials'});
    renderSections([]);
    return;
  }
  renderHeader(manifest.project || {});
  renderSections(manifest.sections || []);
})();
