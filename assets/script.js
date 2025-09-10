<script>
// Helper: fetch JSON
async function loadJSON(path){
  try{ const r = await fetch(path, {cache:'no-store'}); return r.ok ? await r.json() : null; }
  catch(e){ console.warn('JSON load failed', e); return null; }
}

// Helper: element factory
function el(tag, attrs={}, ...kids){
  const n = document.createElement(tag);
  for(const [k,v] of Object.entries(attrs||{})){
    if(k==='class') n.className=v;
    else if(k==='html') n.innerHTML=v;
    else if(k.startsWith('on') && typeof v==='function') n.addEventListener(k.slice(2),v);
    else n.setAttribute(k,v);
  }
  for(const c of kids) if(c!==null && c!==undefined) n.append(c);
  return n;
}

// Guess a reasonable MIME by file extension
function guessMime(file){
  const f = file.toLowerCase();
  if(f.endsWith('.wav')) return 'audio/wav';
  if(f.endsWith('.mp3')) return 'audio/mpeg';
  if(f.endsWith('.ogg')) return 'audio/ogg';
  if(f.endsWith('.au'))  return 'audio/basic';
  if(f.endsWith('.png')) return 'image/png';
  if(f.endsWith('.jpg') || f.endsWith('.jpeg')) return 'image/jpeg';
  return '';
}

function setHeader(meta){
  document.getElementById('title').textContent = meta?.title || 'Project Supplementary Materials';
  const bits=[]; if(meta?.authors) bits.push(meta.authors); if(meta?.conference) bits.push(meta.conference);
  document.getElementById('sub').textContent = bits.join(' · ');
  const nav = document.getElementById('nav'); nav.innerHTML='';
  if(meta?.paper_link) nav.append(el('a',{href:meta.paper_link,target:'_blank',rel:'noopener'},'Paper'));
  if(meta?.code_link)  nav.append(el('a',{href:meta.code_link, target:'_blank',rel:'noopener'},'Code'));
  if(meta?.doi_link)   nav.append(el('a',{href:meta.doi_link,  target:'_blank',rel:'noopener'},'DOI/Dataset'));
}

function render(sections){
  const host = document.getElementById('content'); host.innerHTML='';
  (sections||[]).forEach(sec=>{
    const s = el('section',{class:'section'});
    s.append(el('h2',{},sec.title||'Section'));
    if(sec.description) s.append(el('p',{class:'desc'},sec.description));
    const grid = el('div',{class:'grid'});

    (sec.items||[]).forEach(item=>{
      const card = el('div',{class:'card'});
      card.append(el('h3',{}, item.title || (item.type==='audio'?'Audio sample': item.type==='plot'?'Interactive plot':'Figure')));

      if(item.type==='audio'){
        const mime = item.mime || guessMime(item.file) || 'audio/wav';
        const a = el('audio',{controls:'',preload:'none'});
        a.append(el('source',{src:item.file,type:mime}));
        a.append('Your browser does not support the audio element.');
        card.append(a);

      } else if(item.type==='plot'){
        const f = el('iframe',{class:'plot',src:item.file,loading:'lazy',title:item.title||'Interactive plot'});
        card.append(f);

      } else if(item.type==='image'){
        const img = el('img',{src:item.file, alt:item.title||'figure', style:'max-width:100%;border-radius:10px;display:block'});
        card.append(img);
      }

      if(item.caption) card.append(el('p',{},item.caption));
      grid.append(card);
    });

    s.append(grid); host.append(s);
  });
}

(async()=>{
  const m = await loadJSON('data/manifest.json');
  setHeader(m?.project||{});
  render(m?.sections||[]);
})();
</script>
