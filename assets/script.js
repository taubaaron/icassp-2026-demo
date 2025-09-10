
async function loadJSON(path){ try{ const r=await fetch(path,{cache:'no-store'}); if(!r.ok) return null; return await r.json(); }catch(e){console.warn('JSON load failed',e); return null;} }

function el(tag, attrs={}, ...kids){
  const n=document.createElement(tag);
  for(const [k,v] of Object.entries(attrs||{})){
    if(k==='class') n.className=v;
    else if(k==='html') n.innerHTML=v;
    else if(k.startsWith('on') && typeof v==='function') n.addEventListener(k.slice(2),v);
    else n.setAttribute(k,v);
  }
  for(const c of kids) if(c!==null && c!==undefined) n.append(c);
  return n;
}

function setHeader(meta){
  document.getElementById('title').textContent = meta?.title || 'Project Supplementary Materials';
  const bits = [];
  if(meta?.authors) bits.push(meta.authors);
  if(meta?.conference) bits.push(meta.conference);
  document.getElementById('sub').textContent = bits.join(' · ');
  const nav=document.getElementById('nav');
  nav.innerHTML='';
  if(meta?.paper_link) nav.append(el('a',{href:meta.paper_link,target:'_blank',rel:'noopener'},'Paper'));
  if(meta?.code_link) nav.append(el('a',{href:meta.code_link,target:'_blank',rel:'noopener'},'Code'));
  if(meta?.doi_link) nav.append(el('a',{href:meta.doi_link,target:'_blank',rel:'noopener'},'DOI/Dataset'));
  if(meta?.contact_link) nav.append(el('a',{href:meta.contact_link,target:'_blank',rel:'noopener'},'Contact'));
}

function render(sections){
  const host=document.getElementById('content'); host.innerHTML='';
  (sections||[]).forEach(sec=>{
    const s=el('section',{class:'section'});
    s.append(el('h2',{},sec.title||'Section'));
    if(sec.description) s.append(el('p',{class:'desc'},sec.description));
    const grid=el('div',{class:'grid'});
    (sec.items||[]).forEach(item=>{
      const card=el('div',{class:'card'});
      const heading=item.title || (item.type==='audio'?'Audio sample':'Interactive plot');
      card.append(el('h3',{},heading));
      if(item.type==='audio'){
        const a=el('audio',{controls:'',preload:'none'});
        a.append(el('source',{src:item.file,type:item.mime||'audio/wav'}));
        a.append('Your browser does not support the audio element.');
        card.append(a);
      } else if(item.type==='plot'){
        const f=el('iframe',{class:'plot',src:item.file,loading:'lazy',title:item.title||'Interactive plot'});
        card.append(f);
      }
      if(item.caption) card.append(el('p',{},item.caption));
      if(item.tags && item.tags.length){
        const kv=el('div',{class:'kv'});
        item.tags.forEach(t=>kv.append(el('span',{},t)));
        card.append(kv);
      }
      grid.append(card);
    });
    s.append(grid);
    host.append(s);
  });
}

(async()=>{
  const m=await loadJSON('data/manifest.json');
  setHeader(m?.project||{});
  render(m?.sections||[]);
})();
