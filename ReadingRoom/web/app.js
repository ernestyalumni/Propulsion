import {BookReader} from './reader.js';

const $ = (selector) => document.querySelector(selector);
const esc = (s) => String(s ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const palette = {nr: ['#525d84','#ecedf4'], wie: ['#346c64','#e7eee8'], sutton: ['#a46b45','#f2e9de']};
const names = {nr: 'Numerical Recipes', wie: 'Space Vehicle Dynamics & Control', sutton: 'Rocket Propulsion Elements'};
let data, active, reader, saveTimer, dirty = false, saving = false, rendering = false;
let editVersion = 0, saveChain = Promise.resolve(), routeGeneration = 0;
let currentHash = location.hash || '#library';
const app = $('#app');

function notice(message) { $('#notice').textContent = message; $('#notice').hidden = !message; }
function style(book) { const p = palette[book]; return `--accent:${p[0]};--wash:${p[1]}`; }
function section(book, id) { return book.sections.find(s => s.id === id) || book.sections.find(s => s.id === book.start) || book.sections[0]; }
function saved(book) { return data.state.books[book.id] || {sections:{}}; }
function checkedCount(book) { return Object.values(saved(book).sections || {}).filter(s => s.checks?.read).length; }
function defaultMark(book) { const s = section(book, book.start); return {section:s.id,page:s.pdf_page,zoom:1,scroll:0}; }
function readLink(book, sec = '') { return `#read/${book}${sec ? '/' + encodeURIComponent(sec) : ''}`; }
function openButton(book, sec, label, cls = '') { return `<button class="${cls}" data-read="${book}" data-section="${esc(sec || '')}">${label}</button>`; }

function art(book = 'wie', large = false) {
  const stroke = palette[book][0];
  const inside = book === 'nr'
    ? '<path d="M20 112 Q40 32 61 84 T105 54 T155 26"/><path d="M20 112 L155 112 M20 112 L20 20" opacity=".35"/><circle cx="61" cy="84" r="4"/><circle cx="105" cy="54" r="4"/><path d="M20 100 Q65 105 94 65 T155 35" stroke-dasharray="3 5" opacity=".4"/>'
    : book === 'sutton'
      ? '<path d="M25 25 Q55 29 70 65 Q82 79 98 70 L150 29 M25 117 Q55 113 70 77 Q82 63 98 72 L150 113"/><path d="M20 71 H166" stroke-dasharray="3 5" opacity=".5"/><path d="M32 43 Q63 46 73 68 Q85 76 146 48 M32 100 Q63 97 73 76 Q85 67 146 97" opacity=".35"/><path d="M29 57 Q66 59 73 70 L155 63 M29 86 Q66 82 73 74 L155 81" opacity=".35"/>'
      : '<ellipse cx="90" cy="71" rx="69" ry="29" transform="rotate(-27 90 71)"/><ellipse cx="90" cy="71" rx="69" ry="29" transform="rotate(45 90 71)" opacity=".5"/><circle cx="90" cy="71" r="46" opacity=".3"/><path d="M90 18 V125 M32 71 H149" stroke-dasharray="3 6" opacity=".4"/><circle cx="145" cy="47" r="4" fill="'+stroke+'"/><circle cx="90" cy="71" r="5"/>';
  return `<svg ${large ? 'class="orbit-art"' : ''} viewBox="0 0 180 142" fill="none" stroke="${stroke}" stroke-width="1.2" aria-hidden="true">${inside}</svg>`;
}

function intro(kicker, title, text, right = '') {
  return `<div class="intro"><div><div class="eyebrow">${kicker}</div><h1>${title}</h1><p>${text}</p></div>${right}</div>`;
}

function library() {
  const books = data.books;
  if (!books.length) { app.innerHTML = '<div class="empty">No reading bundles found. Check the exports path shown in the server instructions.</div>'; return; }
  const last = books.find(b => b.id === data.state.last_book) || books.find(b => b.id === 'wie') || books[0];
  const mark = saved(last).bookmark;
  const next = section(last, mark?.section || last.start);
  const page = mark?.page || next.pdf_page;
  const count = books.reduce((n,b) => n + checkedCount(b), 0);
  app.innerHTML = intro('A PLACE TO THINK, BUILD & RETURN TO', 'The reading room.',
    'Follow the physics from the page to a working simulation.<br>Your books, questions, and next steps, all in one place.',
    `<div class="counter"><b>${String(count).padStart(2,'0')}</b>sections marked read</div>`) +
    `<section class="continue-card"><div><div class="eyebrow">${mark ? 'PICK UP WHERE YOU LEFT OFF' : 'A GOOD PLACE TO BEGIN'}</div><h2>${esc(next.title)}</h2><p>${esc(last.short)} · §${esc(next.number)} · ${mark ? 'Saved PDF page '+page : 'Printed p. '+next.printed_page+' / '+(next.exact?'':'≈ ')+'PDF p. '+page}</p><div class="actions">${openButton(last.id, '', mark ? 'Resume reading ↗' : 'Start reading ↗', 'primary')}<button data-action="roadmap">Explore the roadmap</button></div></div>${art(last.id,true)}</section>` +
    `<div class="section-heading"><h2>On your shelf</h2><span>${books.length} books · Original PDFs + indexed text</span></div><div class="books-grid">` + books.map(book => {
      const n = checkedCount(book), total = book.sections.length;
      const record = saved(book), resume = section(book, record.bookmark?.section || book.start);
      return `<article class="book-card" style="${style(book.id)}"><div class="book-visual"><div class="book-spine">${book.id==='nr'?'Numerical<br>Recipes':book.id==='wie'?'Space Vehicle<br>Dynamics':'Rocket<br>Propulsion'}<small>${book.id==='nr'?'THIRD EDITION':book.id==='wie'?'SECOND EDITION':'NINTH EDITION'}</small></div>${art(book.id)}</div><div class="book-content"><div class="eyebrow">${esc(book.domain)}</div><h3>${names[book.id]}</h3><div class="author">${esc(book.authors)}</div><div class="progress-line"><span>${n} / ${total} indexed entries read</span><span>${Math.round(n/total*100)}%</span></div><div class="progress-track"><span style="width:${n/total*100}%"></span></div><div class="book-footer"><span>§${esc(resume.number)} · ${record.bookmark?'Your bookmark':'Suggested start'}</span>${openButton(book.id,'',record.bookmark?'Continue ↗':'Open book ↗')}</div><div class="mini-history">Historical ledger: ${book.snapshot} · tracked separately</div></div></article>`;
    }).join('') + '</div>' +
    '<div class="bottom-strip"><p><strong>Read a little. Make it tangible.</strong><br>Understand an assumption, work through a derivation, then put it to the test.</p><a href="#labs">Visit the simulation labs ↗</a></div>';
}

const milestones = {
  nr: [['01 · NUMERICAL FOUNDATION','17.1','Propagate the state','Runge–Kutta methods and error control support both orbit and attitude propagation.'],['02 · FIND THE SOLUTION','9.3','Solve an implicit relation','Root finding connects nozzle area ratios, Kepler’s equation, and targeting.'],['03 · EXPLORE UNCERTAINTY','7.4','Disperse the inputs','Random variables and covariance turn one simulation into an ensemble.']],
  wie: [['01 · REPRESENT ATTITUDE','5.4','Get the conventions right','Connect quaternions to the existing convention lab before integrating rotations.'],['02 · FOLLOW THE MOTION','6.2','A freely rotating body','Build from Euler’s equations; compare energy and angular momentum.'],['03 · CLOSE THE LOOP','7.3','Control the attitude','Study feedback, then explore how gains affect the response.']],
  sutton: [['01 · UNDERSTAND THRUST','3.3','From chamber to nozzle','Read the ideal flow relations alongside your symbolic nozzle code.'],['02 · FLY THE VEHICLE','4.1','Turn thrust into ascent','Connect propulsion performance with mass change and an ODE solver.'],['03 · CONNECT THE SYSTEMS','18.1','Point the thrust','Link propulsion and attitude through thrust-vector control.']]
};

function roadmap() {
  app.innerHTML = intro('THREE THREADS, ONE DIRECTION','A roadmap into the physics.',
    'A proposed route through the books, organized around things we can build.<br>Use the full ranked lists below whenever you want to take a different path.') +
    '<div class="roadmap-grid">' + data.books.map(b => `<section class="roadmap-track" style="${style(b.id)}"><h3>${esc(b.short)} / ${esc(b.domain)}</h3>` + milestones[b.id].map(([step,sid,title,description]) => {
      const s = b.sections.find(s => s.id === sid);
      const read = saved(b).sections?.[sid]?.checks?.read;
      return `<article class="milestone"><div class="step">${step}${read?' · READ':''}</div><h3>${title}</h3><p>${description}</p>${s?openButton(b.id,sid,`Read §${sid} ↗`):'<span class="pill">Locator unavailable</span>'}</article>`;
    }).join('') + '</section>').join('') + '</div>' +
    '<div class="convergence"><div class="eyebrow">WHERE THE THREADS MEET · FUTURE LAB</div><h2>A coupled vehicle simulation</h2><p>Numerical integration + rigid-body dynamics + propulsion<br>Each connection earns its place through a derivation and a check.</p></div>' +
    data.books.map(b => `<details class="ranked-list"><summary>${esc(b.short)} — full ranked reading list (${b.chapters.length} entries)</summary>` + b.chapters.map(ch => {
      const numbers = String(ch.number).match(/\d+/g) || [];
      const sec = b.sections.find(s => numbers.some(n => s.number === n || s.number === n+'.0')) || b.sections.find(s => numbers.some(n => s.number.startsWith(n+'.')));
      return `<div class="rank-row"><span class="rank">${ch.rank}</span><div><h4>Chapter ${esc(ch.number)} · ${esc(ch.title)}</h4><p>${esc(ch.why)}</p><p>Target: ${esc(ch.module)} · ${esc(ch.language)}</p><small>Snapshot ${b.snapshot}: ${esc(ch.historical_status)}${ch.historical_notes?' — '+esc(ch.historical_notes):''}</small></div>${sec?openButton(b.id,sec.id,'Open chapter ↗'):''}</div>`;
    }).join('') + `<div class="rank-row"><span></span><div class="actions"><button data-doc="${b.id}/roadmap" data-title="${esc(b.short)} · Original roadmap">Full roadmap notes</button><button data-doc="${b.id}/ledger" data-title="${esc(b.short)} · Historical ledger">Historical ledger</button></div></div></details>`).join('');
}

function labs() {
  app.innerHTML = intro('FROM EQUATION TO EXPERIMENT','Put the material in motion.',
    'Start with what is already in Propulsion. One interactive lab and two source-code companions.<br>Availability is checked locally; these labels do not claim that tests have passed.') +
    '<div class="lab-grid">' + data.labs.map((lab,i) => `<article class="lab-card"><div class="lab-icon">${['q ≡ −q','A / A*','Δt'][i]}</div><span class="eyebrow">${esc(lab.kind)}</span><h2>${esc(lab.title)}</h2><p>${esc(lab.description)}</p><div class="actions">${lab.available?`<a class="primary" href="${lab.url}" target="_blank" rel="noopener">${lab.kind==='Interactive lab'?'Launch lab':'View code'} ↗</a>`:'<span class="pill">Unavailable here</span>'}${data.books.some(b=>b.id===lab.book)?openButton(lab.book,lab.section,'Read source section'):''}</div><small>${esc(lab.path)}<br>${lab.available?'Present in this checkout · not revalidated':'Missing from this checkout'}</small></article>`).join('') + '</div>' +
    '<div class="bottom-strip"><p><strong>Our next lab can start with a question.</strong><br>Save an experiment idea in the reader. It will be waiting in the session handoff.</p><a href="/api/handoff" download="Propulsion-reading-session.md">Export session ↗</a></div>';
}

function cleanMarkdown(text) {
  const math = [];
  const protectedText = text.replace(/\$\$[\s\S]*?\$\$|\\\[[\s\S]*?\\\]|\\\([\s\S]*?\\\)|(?<!\\)\$(?!\s)[^$\n]+?\$/g, value => {
    const id = math.push(value)-1; return `MATHPLACEHOLDER${id}END`;
  });
  let html = DOMPurify.sanitize(marked.parse(protectedText), {FORBID_TAGS:['img','iframe','style','video','audio','source'], FORBID_ATTR:['style']});
  html = html.replace(/MATHPLACEHOLDER(\d+)END/g, (_,i) => {
    const value = math[Number(i)], display = value.startsWith('$$') || value.startsWith('\\[');
    const trim = value.startsWith('\\') || display ? 2 : 1;
    return katex.renderToString(value.slice(trim,-trim), {displayMode:display, throwOnError:false, trust:false, strict:false});
  });
  return html;
}

async function documentView(key, title) {
  $('#document-title').textContent = title;
  $('#document-body').textContent = 'Loading…';
  $('#document-dialog').showModal();
  try {
    const response = await fetch('/book/'+key);
    if (!response.ok) throw Error('This document is unavailable.');
    $('#document-body').innerHTML = cleanMarkdown(await response.text());
    // Exported relative links refer to the source machine. Keep them inert here.
    $('#document-body').querySelectorAll('a').forEach(a => {
      a.removeAttribute('href'); a.title = 'Reference in the original exported document';
    });
  } catch (error) { $('#document-body').textContent = error.message; }
}

function toc(book, query = '') {
  const filtered = book.sections.filter(s => `${s.number} ${s.title}`.toLowerCase().includes(query.toLowerCase()));
  $('#toc-items').innerHTML = filtered.map(s => `<button class="toc-item ${s.id===active.section?'active':''}" data-read="${book.id}" data-section="${esc(s.id)}"><span class="sec-num">${esc(s.number || '·')}</span><span>${esc(s.title)}<small>p. ${esc(s.printed_page)} · ${s.exact?'':'≈ '}PDF ${s.pdf_page}${saved(book).sections?.[s.id]?.checks?.read?' · read':''}</small></span></button>`).join('') || '<p class="reader-help">No matching sections.</p>';
}

async function openReader(book, requested, generation) {
  const mark = requested ? {...defaultMark(book),section:requested,page:section(book,requested).pdf_page} : (saved(book).bookmark || defaultMark(book));
  const sec = section(book,mark.section);
  const checks = {read:false,discussed:false,derived:false,implemented:false};
  const record = saved(book).sections?.[sec.id] || {notes:'',questions:'',next:'',checks};
  active = {book,section:sec.id,page:mark.page,zoom:mark.zoom,scroll:mark.scroll,mode:'pdf'};
  app.innerHTML = `<div class="reader-shell"><div class="reader-header"><div><div class="eyebrow">${esc(book.short)} / ${esc(book.domain)}</div><h1>${esc(sec.title)}</h1><p>§${esc(sec.number)} · Section starts at printed p. ${sec.printed_page} / ${sec.exact?'':'≈ '}PDF p. ${sec.pdf_page}</p></div><span class="pill green">Your local reading session</span></div><div class="reader-layout"><aside class="toc-panel"><div class="panel-title">CONTENTS</div><input id="section-search" type="search" placeholder="Find a topic or section…" aria-label="Find a section"><div id="toc-items" class="toc-items"></div></aside><section class="reader-center"><div class="reader-toolbar"><button id="previous-page" aria-label="Previous PDF page">←</button><label>PDF <input id="page-number" type="number" min="1" max="${book.pages}" value="${mark.page}" aria-label="PDF page"></label><span class="muted" id="page-total" style="font-size:10px">/ ${book.pages}</span><button id="next-page" aria-label="Next PDF page">→</button><select id="zoom" aria-label="Page zoom"><option value="0.75">75%</option><option value="1">Fit width</option><option value="1.25">125%</option><option value="1.5">150%</option><option value="2">200%</option></select><button id="text-toggle">Parsed text</button></div><div id="reader-viewport" class="reader-viewport"><div class="loading">Opening the original PDF…</div></div><div class="reader-status"><span id="page-description">Original source · selectable text</span><span id="bookmark-status">Opening…</span></div><div class="reader-links"><a href="/book/${book.id}/pdf#page=${mark.page}" id="original-pdf" target="_blank" rel="noopener">Open original PDF ↗</a><button data-doc="${book.id}/roadmap" data-title="${esc(book.short)} · Roadmap">Roadmap</button><button data-doc="${book.id}/ledger" data-title="${esc(book.short)} · Historical ledger">Historical ledger</button></div><p class="reader-help">${esc(book.ocr_note)}</p></section><aside class="notes-panel"><h3>Make it your own.</h3><p class="note-section">Notes for §${esc(sec.number)} · ${esc(sec.title)}</p><div class="learning-checks">${Object.keys(checks).map(key=>`<label><input type="checkbox" data-check="${key}" ${record.checks[key]?'checked':''}>${key==='implemented'?'Implemented':key[0].toUpperCase()+key.slice(1)}</label>`).join('')}</div><label class="field">What clicked?<textarea id="notes" maxlength="40000" placeholder="An insight, an assumption, a derivation…">${esc(record.notes)}</textarea></label><label class="field">Still wondering<textarea class="short" id="questions" maxlength="40000" placeholder="Questions to work through together…">${esc(record.questions)}</textarea></label><label class="field">Next experiment<textarea class="short" id="next" maxlength="40000" placeholder="What would make this idea tangible?">${esc(record.next)}</textarea></label><div class="save-row"><span id="save-status" role="status">No unsaved changes</span><button id="save-notes">Save notes</button></div><p class="reader-help">Checks are your record of learning, not automatic test results. Notes save after you pause typing.</p><div class="reader-links"><button id="copy-session">Copy session context ↗</button></div></aside></div></div>`;
  $('#zoom').value = String(mark.zoom);
  toc(book);
  const selectedToc = $('#toc-items .active');
  if (selectedToc) $('.toc-panel').scrollTop = selectedToc.offsetTop - $('.toc-panel').offsetTop - 100;
  $('#section-search').addEventListener('input', e => toc(book,e.target.value));
  $('#notes').addEventListener('input', changed);
  $('#questions').addEventListener('input', changed);
  $('#next').addEventListener('input', changed);
  document.querySelectorAll('[data-check]').forEach(el => el.addEventListener('change', changed));
  $('#save-notes').addEventListener('click', () => flush());
  $('#copy-session').addEventListener('click', copySession);
  $('#previous-page').addEventListener('click', () => goPage(active.page-1));
  $('#next-page').addEventListener('click', () => goPage(active.page+1));
  $('#page-number').addEventListener('change', e => goPage(Number(e.target.value)));
  $('#zoom').addEventListener('change', async e => { active.zoom=Number(e.target.value); active.scroll=0; await renderPage(); changed(); });
  $('#text-toggle').addEventListener('click', toggleText);
  $('#reader-viewport').addEventListener('scroll', () => {
    if (rendering || !active || active.mode !== 'pdf') return;
    const el=$('#reader-viewport');
    active.scroll=el.scrollTop/Math.max(1,el.scrollHeight-el.clientHeight);
    changed();
  });
  reader = new BookReader($('#reader-viewport'));
  const localReader = reader;
  setReaderDisabled(true);
  try {
    const pages = await localReader.load(book.id);
    if (generation !== routeGeneration) return;
    $('#page-total').textContent = '/ '+pages;
    await renderPage();
    if (generation !== routeGeneration) return;
    setReaderDisabled(false);
    changed(); // Opening a document records a bookmark, never completion.
  } catch (error) {
    if (generation !== routeGeneration) return;
    notice('Could not open the PDF: '+error.message+' You can still use the original PDF link.');
    $('#bookmark-status').textContent = 'PDF unavailable';
  }
}

function setReaderDisabled(disabled) {
  ['#previous-page','#next-page','#page-number','#zoom','#text-toggle'].forEach(id=>{if($(id))$(id).disabled=disabled;});
}

async function renderPage() {
  const current = active;
  rendering = true;
  try {
    await reader.render(current.page,current.zoom,current.scroll);
    if (active !== current) return;
    $('#page-number').value = active.page;
    $('#original-pdf').href = `/book/${active.book.id}/pdf#page=${active.page}`;
    $('#page-description').textContent = `PDF page ${active.page} · Original source`;
    $('#bookmark-status').textContent = 'Position ready';
  } finally { if (active === current) rendering = false; }
}

async function goPage(page) {
  if (rendering) return;
  if (!Number.isInteger(page) || page<1 || page>active.book.pages) {
    $('#page-number').value=active.page; return;
  }
  active.page=page;active.scroll=0;
  if (active.mode==='text') { active.mode='pdf'; $('#text-toggle').textContent='Parsed text'; }
  try { await renderPage(); changed(); }
  catch (error) { notice('Could not render this page: '+error.message); }
}

async function toggleText() {
  if (rendering) return;
  if (active.mode === 'text') {
    active.mode='pdf';$('#text-toggle').textContent='Parsed text';await renderPage();return;
  }
  active.mode='text';$('#text-toggle').textContent='Original PDF';
  const current=active;
  $('#reader-viewport').innerHTML='<div class="loading">Opening parsed text…</div>';
  try {
    const book=active.book;
    let html;
    if(book.id==='nr') {
      const response=await fetch(`/api/text/nr?page=${active.page}`);
      if(!response.ok)throw Error('Page text unavailable');
      html='<pre>'+esc(await response.text())+'</pre>';
    } else {
      const chapter=String(Number(active.section.split('.')[0])).padStart(3,'0');
      const key='chapter-'+chapter;
      if(!book.text_chapters.includes(key))throw Error('No chapter text for this entry; use the original PDF.');
      const response=await fetch(`/book/${book.id}/${key}`);
      if(!response.ok)throw Error('Chapter text unavailable');
      html=cleanMarkdown(await response.text());
    }
    if(active!==current||active.mode!=='text')return;
    $('#reader-viewport').innerHTML='<article class="text-page prose">'+html+'</article>';
    $('#reader-viewport').querySelectorAll('a').forEach(a=>a.removeAttribute('href'));
    $('#page-description').textContent=book.id==='nr'?'Extracted page text · check math in PDF':'Parsed chapter · check equations in PDF';
    $('#bookmark-status').textContent='PDF bookmark retained';
  } catch(error) { if(active===current)$('#reader-viewport').innerHTML='<div class="empty">'+esc(error.message)+'</div>'; }
}

function changed() {
  dirty=true;editVersion++;
  if($('#save-status'))$('#save-status').textContent='Unsaved changes…';
  clearTimeout(saveTimer);
  saveTimer=setTimeout(()=>flush(),800);
}

function flush() {
  clearTimeout(saveTimer);
  if (!active || !dirty) return saveChain.then(()=>true);
  const version=editVersion, book=active.book.id;
  const checks={};document.querySelectorAll('[data-check]').forEach(el=>checks[el.dataset.check]=el.checked);
  const patch={bookmark:{section:active.section,page:active.page,zoom:active.zoom,scroll:active.scroll},
    section:{id:active.section,notes:$('#notes').value,questions:$('#questions').value,next:$('#next').value,checks}};
  saveChain=saveChain.catch(()=>false).then(async()=>{
    saving=true;
    if($('#save-status'))$('#save-status').textContent='Saving…';
    try {
      const response=await fetch('/api/progress',{method:'POST',headers:{'Content-Type':'application/json','X-Reading-Token':data.token},body:JSON.stringify({book,patch,revision:data.state.revision})});
      const result=await response.json();
      if(!response.ok)throw Error(result.error || 'Save failed');
      data.state=result.state;
      if(editVersion===version){dirty=false;if($('#save-status'))$('#save-status').textContent='Saved to disk';}
      if($('#bookmark-status'))$('#bookmark-status').textContent='Bookmark saved';
      if(result.warning)notice(result.warning);
      return true;
    } catch(error) {
      notice(error.message+' Keep this tab open or copy your notes before reloading.');
      if($('#save-status'))$('#save-status').textContent='Not saved · retry';
      return false;
    } finally { saving=false; }
  });
  return saveChain;
}

async function copySession() {
  if(!await flush())return;
  const response=await fetch('/api/handoff');
  const text=await response.text();
  try { await navigator.clipboard.writeText(text);$('#copy-session').textContent='Session context copied ✓'; }
  catch { $('#document-title').textContent='Session context — select and copy';$('#document-body').textContent=text;$('#document-dialog').showModal(); }
}

async function exportSession() {
  if(!await flush())return;
  const response=await fetch('/api/handoff');
  if(!response.ok){notice('Could not export the saved session.');return;}
  const url=URL.createObjectURL(new Blob([await response.text()],{type:'text/markdown'}));
  const link=document.createElement('a');link.href=url;link.download='Propulsion-reading-session.md';
  link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
}

async function navigate(hash) {
  if(!await flush())return;
  if(location.hash===hash)await route();
  else location.hash=hash;
}

async function route() {
  const requested=location.hash||'#library';
  if(!await flush()) { history.replaceState(null,'',currentHash);return; }
  const generation=++routeGeneration;
  currentHash=requested;
  reader?.destroy();reader=null;active=null;dirty=false;
  const [view,id,sec] = requested.slice(1).split('/');
  document.querySelectorAll('[data-nav]').forEach(el=>el.classList.toggle('active',el.dataset.nav===(view==='read'?'library':view)));
  $('#breadcrumb').textContent={library:'Reading room',roadmap:'Learning roadmap',labs:'Simulation labs',read:'At the reading desk'}[view]||'Reading room';
  if(view==='roadmap')roadmap();
  else if(view==='labs')labs();
  else if(view==='read') {
    const book=data.books.find(b=>b.id===id);
    if(!book){app.innerHTML='<div class="empty">This book is unavailable. Return to the reading room to choose another.</div>';return;}
    await openReader(book,sec?decodeURIComponent(sec):null,generation);
  } else library();
}

document.addEventListener('click',async event=>{
  const download=event.target.closest('a[href="/api/handoff"][download]');
  if(download){event.preventDefault();await exportSession();return;}
  const button=event.target.closest('[data-read],[data-action],[data-doc]');
  if(button?.dataset.read)await navigate(readLink(button.dataset.read,button.dataset.section));
  else if(button?.dataset.action)await navigate('#'+button.dataset.action);
  else if(button?.dataset.doc)await documentView(button.dataset.doc,button.dataset.title);
  const anchor=event.target.closest('a[href^="#"]');
  if(anchor){event.preventDefault();await navigate(anchor.getAttribute('href'));}
});
$('#close-dialog').addEventListener('click',()=>$('#document-dialog').close());
window.addEventListener('beforeunload',event=>{if(dirty||saving){event.preventDefault();event.returnValue='';}});
window.addEventListener('hashchange',()=>route().catch(error=>notice(error.message)));
window.addEventListener('keydown',event=>{
  if(!active||event.target.closest('input,textarea,select')||$('#document-dialog').open||event.metaKey||event.ctrlKey)return;
  if(event.key==='ArrowRight'){event.preventDefault();goPage(active.page+1);}
  if(event.key==='ArrowLeft'){event.preventDefault();goPage(active.page-1);}
});

try {
  const response=await fetch('/api/bootstrap');
  if(!response.ok)throw Error('The local reading server could not load your library.');
  data=await response.json();
  $('#shelf-nav').innerHTML=data.books.map(b=>`<a class="shelf-link" href="${readLink(b.id)}" style="${style(b.id)}"><span class="dot"></span>${b.id==='nr'?'Numerical Recipes':b.id==='wie'?'Space Vehicle Dynamics':'Rocket Propulsion'}</a>`).join('');
  if(data.warnings.length)notice(data.warnings.map(w=>w.book+': '+w.message).join('\n'));
  await route();
} catch(error) {notice(error.message);app.innerHTML='<div class="empty">The reading room could not start. Check the local server and reload.</div>';}
