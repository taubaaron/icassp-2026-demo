/* ICASSP 2026 Results — dynamic rendering (accordion + filters) */

async function loadJSON(path){
  try{
    const r = await fetch(path + '?v=' + Date.now(), {cache:'no-store'});
    return r.ok ? await r.json() : null;
  }catch(e){
    console.error('JSON load failed', e);
    return null;
  }
}

/* ---------- small helpers ---------- */
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
function badgeForType(item){
  // Try explicit badges first
  if (Array.isArray(item.badges) && item.badges.length) return item.badges;
  // Otherwise infer: WAV/PNG/HTML/PDF
  const f = (item.file||'').toLowerCase();
  if (f.endsWith('.wav')) return ['WAV'];
  if (f.endsWith('.mp3')) return ['MP3'];
  if (f.endsWith('.ogg')) return ['OGG'];
  if (f.endsWith('.png')) return ['PNG'];
  if (f.endsWith('.jpg') || f.endsWith('.jpeg')) return ['JPG'];
  if (f.endsWith('.html')) return ['HTML','Interactive'];
  if (f.endsWith('.pdf')) return ['PDF'];
  // Fallback to item.type
  if (item.type) return [String(item.type).toUpperCase()];
  return [];
}

/* ---------- header / overview ---------- */
function setHeader(p){
  document.getElementById('title').textContent =
    p?.title || 'Project Supplementary Materials';

  const bits=[];
  if (p?.authors)   bits.push(p.authors);
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

/* ---------- table of contents ---------- */
function buildTOC(sections){
  const toc=document.getElementById('toc');
  toc.innerHTML='';
  toc.append(el('h3',{},'On this page'));
  // Only show Overview link if it exists
  const hasOverview = !!document.getElementById('overview');
  if (hasOverview) toc.append(el('a',{href:'#overview'},'Overview'));
  sections.forEach((s,i)=>{
    const id='sec-'+(i+1);
    toc.append(el('a',{href:'#'+id}, s.title || ('Section '+(i+1))));
  });
}

/* ---------- filters (chips) ---------- */
function buildFilters(allTags){
  if (!allTags.size) return;
  // Ensure a container exists; create it just before sections if not present.
  let filtersHost = document.getElementById('filters');
  if (!filtersHost){
    filtersHost = el('div',{id:'filters', class:'filters'});
    const content = document.getElementById('content');
    content.prepend(filtersHost);
  } else {
    filtersHost.classList.add('filters');
  }
  filtersHost.innerHTML = '';
  [...allTags].sort((a,b)=>a.localeCompare(b)).forEach(tag=>{
    filtersHost.append(el('div',{class:'filter-chip', 'data-tag':tag}, tag));
  });

  const chips = Array.from(filtersHost.querySelectorAll('.filter-chip'));
  const items = Array.from(document.querySelectorAll('[data-tags]'));
  function applyFilter(){
    const active = chips.filter(c=>c.classList.contains('active')).map(c=>c.dataset.tag);
    if (!active.length){ items.forEach(el=>el.style.display=''); return; }
    items.forEach(el=>{
      const tags = (el.dataset.tags || '').split(',').map(s=>s.trim()).filter(Boolean);
      // AND logic: show item only if it contains ALL active tags
      const show = active.every(a=>tags.includes(a));
      el.style.display = show ? '' : 'none';
    });
  }
  chips.forEach(chip=>{
    chip.addEventListener('click', ()=>{
      chip.classList.toggle('active');
      applyFilter();
    });
  });
}

/* ---------- sections & items ---------- */
function makeDescBlock(desc){
  // desc can be string OR {what:[], look:[], method:[]}
  if (!desc) return null;
  const box = el('div',{class:'desc'});
  if (typeof desc === 'string'){
    box.append(el('p',{}, desc));
    return box;
  }
  if (desc.what && desc.what.length){
    box.append(el('h4',{},'What this is'));
    const ul=el('ul',{}); desc.what.forEach(t=>ul.append(el('li',{},t))); box.append(ul);
  }
  if (desc.look && desc.look.length){
    box.append(el('h4',{},'What to look/listen for'));
    const ul=el('ul',{}); desc.look.forEach(t=>ul.append(el('li',{},t))); box.append(ul);
  }
  if (desc.method && desc.method.length){
    box.append(el('h4',{},'Method'));
    const ul=el('ul',{}); desc.method.forEach(t=>ul.append(el('li',{},t))); box.append(ul);
  }
  return box;
}

function renderItemCard(item){
  const card = el('div',{class:'item', 'data-tags': (item.tags||[]).join(',') });

  // Header
  const head = el('div',{class:'item-head'});
  head.append(el('span',{class:'item-title'}, item.title || (
    item.type==='audio' ? 'Audio sample' :
    item.type==='plot'  ? 'Interactive plot' :
    item.type==='pdf'   ? 'PDF' :
    item.type==='image' ? 'Figure' : 'Item'
  )));
  const badges = badgeForType(item);
  badges.forEach(b=> head.append(el('span',{class:'badge'}, b)));
  card.append(head);

  // Media
  const media = el('div',{class:'item-media'});
  const f = (item.file||'').toLowerCase();
  if (item.type==='audio' || f.endsWith('.wav') || f.endsWith('.mp3') || f.endsWith('.ogg') || f.endsWith('.au')){
    const mime=item.mime||guessMime(item.file)||'audio/wav';
    const a=el('audio',{controls:'',preload:'none'});
    a.append(el('source',{src:item.file,type:mime}));
    a.append('Your browser does not support the audio element.');
    media.append(a);
  } else if (item.type==='plot' || f.endsWith('.html')){
    media.append(el('iframe',{class:'plot',src:item.file,loading:'lazy',title:item.title||'Interactive plot'}));
  } else if (item.type==='pdf' || f.endsWith('.pdf')){
    media.append(el('iframe',{class:'pdf',src:item.file,loading:'lazy',title:item.title||'PDF preview'}));
  } else if (item.type==='image' || f.endsWith('.png') || f.endsWith('.jpg') || f.endsWith('.jpeg')){
    media.append(el('img',{src:item.file,alt:item.title||'figure'}));
  } else {
    media.append(el('p',{class:'item-caption'}, 'Unsupported item type.'));
  }

  // Explanation / caption
  const descNode = makeDescBlock(item.desc);
  if (descNode) media.append(descNode);
  if (item.caption && !descNode){
    card.append(el('div',{class:'item-caption'}, item.caption));
  }

  card.append(media);
  return card;
}

function renderSections(sections){
  const host=document.getElementById('content');
  const allTags = new Set();

  sections.forEach((sec,i)=>{
    const id='sec-'+(i+1);

    const details = el('details',{class:'accordion', id});
    if (i===0) details.setAttribute('open','open');

    const summary = el('summary',{}, sec.title || 'Section');
    if (sec.description) summary.append(el('span',{class:'summary-note'}, ' — '+sec.description));
    details.append(summary);

    const body = el('div',{class:'section-body'});

    const cols = Number(sec.cols||0);
    const gridClass = cols ? `grid cols-${cols}` : 'grid auto';
    const grid = el('div',{class:gridClass});

    (sec.items||[]).forEach(item=>{
      // accumulate tags for filter chips
      (item.tags||[]).forEach(t=> allTags.add(t));
      grid.append( renderItemCard(item) );
    });

    body.append(grid);
    details.append(body);
    host.append(details);
  });

  // Build filter chips once we know all unique tags
  buildFilters(allTags);

  // If URL has a hash matching a section, open it
  if (location.hash){
    const target = document.querySelector(location.hash);
    if (target && target.tagName.toLowerCase()==='details') target.setAttribute('open','open');
  }
}

/* ---------- boot ---------- */
(async()=>{
  const m = await loadJSON('data/manifest.json');
  setHeader(m?.project||{});
  renderOverview(m?.project||{});
  buildTOC(m?.sections||[]);
  renderSections(m?.sections||[]);
})();
