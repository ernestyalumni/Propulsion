const esc = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const names = {nr:'Numerical Recipes', wie:'Wie', sutton:'Sutton', hp:'Hill & Peterson'};

function assetLink(asset) {
  const status = asset.kind === 'PDF' && asset.available ? (asset.build_current ? ' · current build' : ' · stale or unrecorded build') : '';
  return `<div class="resource-link"><small>${esc(asset.kind)}${status}</small>${asset.available ? `<a href="${esc(asset.url)}" target="_blank" rel="noopener">${esc(asset.title)} ↗</a>` : `<span>${esc(asset.title)} · unavailable here</span>`}</div>`;
}

export function notesMarkup(catalog, book, section) {
  const topics = catalog.topics.filter(t => !book || t.locators.some(l => l.book === book && (!section || l.section === section)));
  return `<div class="intro"><div><div class="eyebrow">DERIVATION → COMPUTATION → EVIDENCE</div><h1>Notes & worked solutions.</h1><p>One growing body of mathematics, shared by the companions and the master.<br>Read the assumptions, follow the derivation, then inspect what was checked.</p></div></div>
    <div class="notes-filters"><a href="#notes">All topics</a>${Object.entries(names).map(([id,name])=>`<a href="#notes/${id}">${name}</a>`).join('')}</div>
    ${book ? `<p class="notes-scope">Related to ${esc(names[book] || book)}${section?' §'+esc(section):''} · <a href="#read/${esc(book)}${section?'/'+encodeURIComponent(section):''}">Return to the book ↗</a></p>` : ''}
    <details class="notes-volumes" ${book?'':'open'}><summary>Master & book companions · portrait / screen / source</summary><div class="document-grid">${catalog.documents.map(assetLink).join('')}</div></details>
    <label class="topic-search-label">Find a topic or manuscript<input id="topic-search" type="search" placeholder="Try Euler, nozzle, geometry, or Sage…"></label>
    <div class="notes-grid">${topics.map(topic => {
      const v = topic.verification;
      const status = v.recorded && !v.current ? 'Evidence is stale — rerun after source changes' : v.symbolic && v.numerical ? 'Sage checked · numerically tested' : v.numerical ? 'Numerically tested · Sage check not recorded' : 'No current computation evidence';
      return `<article class="topic-card" id="${esc(topic.id)}" data-topic-search="${esc([topic.title,topic.summary,...topic.assets.map(a=>a.title)].join(' ').toLowerCase())}"><div class="eyebrow">${esc(topic.id)}</div><h2>${esc(topic.title)}</h2><p>${esc(topic.summary)}</p><div class="evidence-status ${v.current&&v.symbolic&&v.numerical?'checked':''}">${esc(status)}</div>${v.created?`<small>Recorded ${esc(v.created)}${v.sage_version?' · '+esc(v.sage_version):''}</small>`:''}<p class="reader-help">${esc(topic.status)} These checks do not change your learning progress.</p><div class="topic-locators">${topic.locators.map(l=>l.available?`<a href="#read/${l.book}/${encodeURIComponent(l.section)}">${names[l.book]} §${esc(l.section)} · p. ${l.printed_page} / ${l.exact?'':'≈ '}PDF ${l.pdf_page} ↗</a>`:`<span>${names[l.book]} §${esc(l.section)} · book unavailable</span>`).join('')}</div><div class="topic-resources">${topic.assets.map(assetLink).join('')}</div></article>`;
    }).join('')}</div>${topics.length?'':'<p>No topic is registered for this section yet. Browse the book’s topics or the full collection.</p>'}`;
}

export function attachNotesSearch() {
  document.querySelector('#topic-search').addEventListener('input', event => {
    const query = event.target.value.trim().toLowerCase();
    document.querySelectorAll('[data-topic-search]').forEach(card => {card.hidden = !card.dataset.topicSearch.includes(query);});
  });
}
