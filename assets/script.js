async function loadJSON(path){
  try{ const r = await fetch(path + '?v=' + Date.now(), {cache:'no-store'}); return r.ok ? await r.json() : null; }
  catch(e){ console.error('JSON load failed', e); return null; }
}
function el(tag, attrs={}, ...kids){
  const n=document.createElement(tag);
  for(const [k,v] of Object.entries(attrs||{})){
    if(k==='class') n.className=v; else if(k==='html') n.innerHTML=v;
    else n.setAttribute(k,v);
  }
  kids.forEach(c=>c!=null && n.append(c));
  return n;
}
function guessMime(f){
  const x=(f||'').toLowerCase();
  if(x.endsWith('.wav')) return 'audio/wav';
  if(x.endsWith('.mp3')) return 'audio/mpeg';
  if(x.endsWith('.ogg')) return 'audio/ogg';
  if(x.endsWith('.au'))  return 'audio/basic';
  if(x.endsWith('.png')) return 'image/png';
  if(x.endsWith('.jpg')||x.endsWith('.jpeg')) return 'image/jpeg';
  return '';
}
function setHeader(p){
  document.getElementById('title').textContent = p?.title || 'Project Supplementary Materials';
  const bits=[]; if(p?.authors) bits.push(p.authors); if(p?.conference) bits.push(p.conference);
  document.querySelector('.subtle').textContent = bits.join(' · ');
  const nav=document.getElementById('nav'); nav.innerHTML='';
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
      const ul=el('ul',{}); p.notes.forEach(t=>ul.append(el('li',{},t))); box.append(ul);
      sec.append(box);
    }
    main.append(sec);
  }
}
function buildTOC(sections){
  const toc=document.getElementById('toc');
  toc.innerHTML='';
  toc.append(el('h3',{},'On this page'));
  toc.append(el('a',{href:'#overview'},'Overview'));
  sections.forEach((s,i)=>{
    const id='sec-'+(i+1);
    toc.append(el('a',{href:'#'+id}, s.title || ('Section '+(i+1))));
  });
}
function renderSections(sections){
  const host=document.getElementById('content');
  sections.forEach((sec,i)=>{
    const id='sec-'+(i+1);
    const s=el('section',{class:'section', id});
    s.append(el('h2',{},sec.title||'Section'));
    if(sec.description) s.append(el('p',{class:'desc'},sec.description));
    const cols = Number(sec.cols||0);
    const grid=el('div',{class: (cols?`grid cols-${cols}`:'grid')});
    (sec.items||[]).forEach(item=>{
      const card=el('div',{class:'card'});
      card.append(el('h3',{}, item.title || (item.type==='audio'?'Audio sample': item.type==='plot'?'Interactive plot':'Figure')));
      if(item.type==='audio'){
        const mime=item.mime||guessMime(item.file)||'audio/wav';
        const a=el('audio',{controls:'',preload:'none'});
        a.append(el('source',{src:item.file,type:mime}));
        a.append('Your browser does not support the audio element.'); card.append(a);
      }else if(item.type==='plot'){
        card.append(el('iframe',{class:'plot',src:item.file,loading:'lazy',title:item.title||'Interactive plot'}));
      }else if(item.type==='image'){
        card.append(el('img',{src:item.file,alt:item.title||'figure'}));
      }
      if(item.caption) card.append(el('p',{},item.caption));
      grid.append(card);
    });
    s.append(grid); host.append(s);
  });
}
(async()=>{
  const m=await loadJSON('data/manifest.json');
  setHeader(m?.project||{});
  renderOverview(m?.project||{});
  buildTOC(m?.sections||[]);
  renderSections(m?.sections||[]);
})();
