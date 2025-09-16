/* ICASSP 2026 Results — robust loader + accordion (no filters) */

/* ---------- helpers ---------- */
function el(tag, attrs={}, ...kids){
  const n = document.createElement(tag);
  for (const [k,v] of Object.entries(attrs||{})){
    if (k === 'class') n.className = v;
    else if (k === 'html') n.innerHTML = v;
    else n.setAttribute(k, v);
  }
  kids.forEach(c => c!=null && n.append(c));
  return n;
}
function guessMime(f){
  const x=(f||'').toLowerCase();
  if (x.endsWith('.wav'))  return 'audio/wav';
  if (x.endsWith('.mp3'))  return 'audio/mpeg';
  if (x.endsWith('.ogg'))  return 'audio/ogg';
  if (x.endsWith('.au'))   return 'audio/basic';
  if (x.endsWith('.png'))  return 'image/png';
  if (x.endsWith('.jpg') || x.endsWith('.jpeg')) return 'image/jpeg';
  if (x.endsWith('.pdf'))  return 'application/pdf';
  return '';
}

/* Try multiple manifest locations and surface a friendly error */
async function loadManifest(){
  const candidates = [
    'data/manifest.json',
    './data/manifest.json',
    'manifest.json',
    './manifest.json'
  ];
  for (const p of candidates){
    try{
      const r = await fetch(p + '?v=' + Date.now(), { cache:'no-store' });
      if (r.ok){
        console.log('[results] Loaded manifest from', p);
        return await r.json();
      }
    }catch(e){
      console.warn('[results] Fetch failed for', p, e);
    }
  }
  return null;
}

/* ---------- header / overview ---------- */
function setHeader(p){
  document.getElementById('title').textContent =
    p?.title || 'Project Supplementary Materials';

  const bits=[];
  if (p?.authors)    bits.push(p.authors);
  if (p?.conference) bits.push(p.conference);
  document.querySelector('.subtle').textContent = bits.join(' · ');

  const nav=document.getElementById('nav');
  nav.innerHTML='';
  if(p?.paper_link) nav.append(el('a',{href:p.paper_link,target:'_blank',rel:'noopener'},'Paper'));
  if(p?.code_link)  nav.append(el('a',{href:p.code_link, target:'_blank',rel:'noopener'},'Code'));
  if(p?.doi_link)   nav.append(el('a',{href:p.doi_link,  target:'_blank',rel:'noopener'},'DOI/Dataset'));
}
function renderOverview(p){
  const main=document.getElementById('content');
  if(p?.abstract || (p?.notes && p.notes.length)){
    const sec=el('section',{class:'section', id:'overview'});
    sec.append(el('h2',{},'Overview'));
    if(p.abstract) sec.append(el('p',{},p.abstract));
    if(Array.isArray(p.notes) && p.notes.length){
      const box=el('div',{class:'callout'});
      box.append(el('strong',{},'Notes:'));
      const ul=el('ul',{}); p.notes.forEach(t=>ul.append(el('li',{},t)));
      box.append(ul);
      sec.append(box);
    }
    main.append(sec);
  }
}

/* ---------- ToC ---------- */
function buildTOC(sections){
  const toc=document.getElementById('toc');
  toc.innerHTML='';
  toc.append(el('h3',{},'On this page'));
  if (document.getElementById('overview')) toc.append(el('a',{href:'#overview'},'Overview'));
  (sections||[]).forEach((s,i)=>{
    const id='sec-'+(i+1);
    toc.append(el('a',{href:'#'+id}, s.title || ('Section '+(i+1))));
  });
}

/* ---------- items & sections ---------- */
function makeDescBlock(desc){
  if (!desc) return null;
  const box = el('div',{class:'desc'});
  if (typeof desc === 'string'){ box.append(el('p',{},desc)); return box; }
  if (desc.what?.length){
    box.append(el('h4',{},'What this is'));
    const ul=el('ul',{}); desc.what.forEach(t=>ul.append(el('li',{},t))); box.append(ul);
  }
  if (desc.look?.length){
    box.append(el('h4',{},'What to look/listen for'));
    const ul=el('ul',{}); desc.look.forEach(t=>ul.append(el('li',{},t))); box.append(ul);
  }
  if (desc.method?.length){
    box.append(el('h4',{},'Method'));
    const ul=el('ul',{}); desc.method.forEach(t=>ul.append(el('li',{},t))); box.append(ul);
  }
  return box;
}
function badgeForType(item){
  if (Array.isArray(item.badges) && item.badges.length) return item.badges;
  const f=(item.file||'').toLowerCase();
  if (f.endsWith('.wav')) return ['WAV'];
  if (f.endsWith('.mp3')) return ['MP3'];
  if (f.endsWith('.ogg')) return ['OGG'];
  if (f.endsWith('.png')) return ['PNG'];
  if (f.endsWith('.jpg') || f.endsWith('.jpeg')) return ['JPG'];
  if (f.endsWith('.html')) return ['HTML','Interactive'];
  if (f.endsWith('.pdf')) return ['PDF'];
  if (item.type) return [String(item.type).toUpperCase()];
  return [];
}
function renderItemCard(item){
  const card = el('div',{class:'item'}); // removed data-tags to avoid filters completely

  const head = el('div',{class:'item-head'});
  head.append(el('span',{class:'item-title'}, item.title || 'Item'));
  badgeForType(item).forEach(b=> head.append(el('span',{class:'badge'}, b)));
  card.append(head);

  const media = el('div',{class:'item-media'});
  const f=(item.file||'').toLowerCase();
  if (item.type==='audio' || /\.(wav|mp3|ogg|au)$/i.test(f)){
    const a=el('audio',{controls:'',preload:'none'});
    a.append(el('source',{src:item.file,type:guessMime(item.file)||'audio/wav'}));
    a.append('Your browser does not support the audio element.');
    media.append(a);
  } else if (item.type==='plot' || f.endsWith('.html')){
    media.append(el('iframe',{class:'plot',src:item.file,loading:'lazy',title:item.title||'Interactive plot'}));
  } else if (item.type==='pdf' || f.endsWith('.pdf')){
    media.append(el('iframe',{class:'pdf',src:item.file,loading:'lazy',title:item.title||'PDF preview'}));
  } else if (item.type==='image' || /\.(png|jpg|jpeg)$/i.test(f)){
    media.append(el('img',{src:item.file,alt:item.title||'figure'}));
  } else {
    media.append(el('p',{class:'item-caption'}, 'Unsupported item type.'));
  }

  const descNode = makeDescBlock(item.desc);
  if (descNode) media.append(descNode);
  card.append(media);
  return card;
}
function renderSections(sections){
  const host=document.getElementById('content');

  (sections||[]).forEach((sec,i)=>{
    const id='sec-'+(i+1);
    const details = el('details',{class:'accordion', id});
    if (i===0) details.setAttribute('open','open');

    const summary = el('summary',{}, sec.title || 'Section');
    if (sec.description) summary.append(el('span',{class:'summary-note'}, ' — '+sec.description));
    details.append(summary);

    const body = el('div',{class:'section-body'});
    const cols = Number(sec.cols||0);
    const grid = el('div',{class: (cols?`grid cols-${cols}`:'grid cols-1')}); // default to full-width single column

    (sec.items||[]).forEach(item=>{
      grid.append( renderItemCard(item) );
    });

    body.append(grid);
    details.append(body);
    host.append(details);
  });

  if (location.hash){
    const target = document.querySelector(location.hash);
    if (target && target.tagName.toLowerCase()==='details') target.setAttribute('open','open');
  }
}

/* ---------- boot ---------- */
(async()=>{
  const content = document.getElementById('content');
  const m = await loadManifest();

  if (!m){
    content.innerHTML = `
      <div class="callout">
        <strong>Setup error:</strong> Could not load <code>manifest.json</code>.<br>
        Place it at <code>/data/manifest.json</code> (recommended) or at the site root as <code>/manifest.json</code>.
      </div>`;
    return;
  }

  setHeader(m.project || {});
  renderOverview(m.project || {});
  buildTOC(m.sections || []);
  renderSections(m.sections || []);
})();
