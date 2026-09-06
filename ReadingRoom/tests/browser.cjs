// Real Chrome acceptance checks. The server and all progress use a fresh temp dir.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const {spawn} = require('node:child_process');
const {once} = require('node:events');
const modulePath = process.env.PLAYWRIGHT_MODULE || path.resolve(__dirname, '../../../../tools/playwright-runner/node_modules/playwright');
const {chromium} = require(modulePath);
const room = path.resolve(__dirname, '..');
const state = fs.mkdtempSync(path.join(os.tmpdir(), 'propulsion-browser-'));
const errors = [], external = [];
let server, browser;

async function startServer() {
  server = spawn('python3', ['-B', 'server.py', '--port', '0', '--state-dir', state], {cwd: room});
  return new Promise((resolve,reject) => {
    let output='';
    server.stdout.on('data', chunk=>{
      output+=chunk;
      const match=output.match(/http:\/\/127\.0\.0\.1:\d+/);
      if(match)resolve(match[0]);
    });
    server.stderr.on('data', chunk=>process.stderr.write(chunk));
    server.on('error',reject);
    server.on('exit', code=>reject(Error('Server exited: '+code)));
  });
}

async function stopServer() {
  if(server && server.exitCode===null){server.kill('SIGINT');await once(server,'exit');}
}

async function waitSaved(page) {
  await page.waitForFunction(()=>document.querySelector('#save-status')?.textContent==='Saved to disk');
}

async function rendered(page) {
  await page.waitForFunction(()=>document.querySelector('.textLayer')?.childElementCount>0);
  await waitSaved(page);
}

(async()=>{
  let base=await startServer();
  browser=await chromium.launch({channel:'chrome',headless:true});
  const context=await browser.newContext({viewport:{width:1440,height:1000}});
  context.on('page',page=>{
    page.on('pageerror',e=>errors.push(e.message));
    page.on('request',r=>{if(!/^(http:\/\/127\.0\.0\.1:|blob:|data:)/.test(r.url()))external.push(r.url());});
  });
  const page=await context.newPage();
  await page.goto(base);
  await page.waitForSelector('.book-card');
  assert.equal(await page.locator('.book-card').count(),3);
  assert.equal(await page.locator('.counter b').innerText(),'00');
  await page.screenshot({path:path.join(state,'home.png'),fullPage:true});
  await page.getByRole('button',{name:'Start reading ↗',exact:true}).click();
  await rendered(page);
  assert.equal(await page.locator('#page-number').inputValue(),'352');
  assert.equal(await page.locator('[data-check="read"]').isChecked(),false);
  await page.locator('#notes').fill('The representation changes; the physical attitude does not.');
  await page.locator('#questions').fill('How do the active/passive conventions compare?');
  await page.locator('#next').fill('Compare q and −q in the quaternion lab.');
  await page.locator('[data-check="read"]').check();
  await page.locator('#zoom').selectOption('1.5');
  await rendered(page);
  await page.locator('#reader-viewport').evaluate(el=>{el.scrollTop=(el.scrollHeight-el.clientHeight)*0.45;});
  await page.waitForTimeout(1200);
  await waitSaved(page);
  let disk=JSON.parse(fs.readFileSync(path.join(state,'progress.json'),'utf8'));
  assert.ok(disk.books.wie.bookmark.scroll>0.3);
  assert.equal(disk.books.wie.bookmark.zoom,1.5);
  assert.equal(disk.books.wie.sections['5.4'].checks.implemented,false);
  assert.ok(fs.readFileSync(path.join(state,'HANDOFF.md'),'utf8').includes('active/passive'));

  // Export flushes a just-typed note before producing the handoff.
  await page.locator('#next').fill('Compare q and −q in the quaternion lab. Export immediately.');
  const downloadPromise=page.waitForEvent('download');
  await page.getByRole('link',{name:'Export session ↗',exact:true}).click();
  const download=await downloadPromise;
  assert.ok(fs.readFileSync(await download.path(),'utf8').includes('Export immediately.'));

  // Reload and restart the entire server: state is not tied to browser storage.
  await stopServer();base=await startServer();
  await page.goto(base+'/#read/wie');await rendered(page);
  assert.equal(await page.locator('#notes').inputValue(),'The representation changes; the physical attitude does not.');
  assert.equal(await page.locator('#zoom').inputValue(),'1.5');
  assert.equal(await page.locator('[data-check="read"]').isChecked(),true);
  assert.ok(await page.locator('#reader-viewport').evaluate(el=>el.scrollTop/(el.scrollHeight-el.clientHeight))>0.3);
  await page.getByRole('button',{name:'Next PDF page',exact:true}).click();
  await page.waitForFunction(()=>document.querySelector('#page-number').value==='353');await waitSaved(page);

  // Parsed math and original PDF switching preserve the PDF bookmark.
  await page.getByRole('button',{name:'Parsed text',exact:true}).click();
  await page.waitForSelector('.text-page .katex');
  assert.ok(await page.locator('.text-page .katex').count()>0);
  await page.getByRole('button',{name:'Original PDF',exact:true}).click();
  await rendered(page);
  assert.equal(await page.locator('#page-number').inputValue(),'353');
  await page.screenshot({path:path.join(state,'reader.png'),fullPage:true});

  // Book and section notes remain independent. All PDFs render locally.
  await page.locator('#section-search').fill('5.5');
  await page.locator('.toc-item').click();
  await page.waitForFunction(()=>document.querySelector('.note-section')?.textContent.startsWith('Notes for §5.5'));
  await rendered(page);
  assert.equal(await page.locator('#notes').inputValue(),'');
  await page.goto(base+'/#read/sutton/3.3');await rendered(page);
  assert.equal(await page.locator('#page-number').inputValue(),'75');
  await page.getByRole('button',{name:'Parsed text',exact:true}).click();
  await page.waitForSelector('.text-page .katex');
  await page.goto(base+'/#read/nr/17.1');await rendered(page);
  assert.equal(await page.locator('#page-number').inputValue(),'931');
  await page.getByRole('button',{name:'Parsed text',exact:true}).click();
  await page.waitForSelector('.text-page pre');
  assert.ok((await page.locator('.text-page pre').innerText()).length>100);

  await page.goto(base+'/#roadmap');await page.waitForSelector('.milestone');
  assert.equal(await page.locator('.milestone').count(),9);
  await page.locator('.ranked-list summary').first().click();
  assert.ok(await page.locator('.rank-row').count()>30);
  await page.screenshot({path:path.join(state,'roadmap.png'),fullPage:true});
  await page.getByRole('button',{name:'Historical ledger',exact:true}).first().click();
  await page.waitForSelector('dialog[open] .prose table');
  await page.getByRole('button',{name:'Close document',exact:true}).click();
  await page.goto(base+'/#labs');await page.waitForSelector('.lab-card');
  assert.equal(await page.locator('.lab-card').count(),3);
  const popupPromise=page.waitForEvent('popup');
  await page.getByRole('link',{name:'Launch lab ↗',exact:true}).click();
  const popup=await popupPromise;
  await popup.waitForSelector('canvas');
  await popup.close();

  // A second tab cannot overwrite a newer section note silently.
  await page.goto(base+'/#read/wie/5.4');await rendered(page);
  const second=await context.newPage();await second.goto(base+'/#read/wie/5.4');await rendered(second);
  await second.locator('#notes').fill('Newer tab note');await waitSaved(second);
  await page.locator('#notes').fill('Unsaved older tab note');
  await page.waitForFunction(()=>document.querySelector('#notice').textContent.includes('another tab'));
  assert.equal(await page.locator('#notes').inputValue(),'Unsaved older tab note');
  assert.equal(JSON.parse(fs.readFileSync(path.join(state,'progress.json'),'utf8')).books.wie.sections['5.4'].notes,'Newer tab note');
  await page.getByRole('link',{name:'Reading room',exact:false}).first().click();
  assert.ok(page.url().includes('#read/wie'));
  page.on('dialog',d=>d.accept());await page.close();await second.close();

  // The layout remains usable on a phone-size viewport, with no horizontal overflow.
  const mobile=await context.newPage();await mobile.setViewportSize({width:390,height:844});
  await mobile.goto(base);await mobile.waitForSelector('.book-card');
  assert.ok(await mobile.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1));
  await mobile.screenshot({path:path.join(state,'mobile.png'),fullPage:true});
  await mobile.goto(base+'/#read/wie/5.4');await rendered(mobile);
  assert.ok(await mobile.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1));
  assert.deepEqual(errors,[]);
  assert.deepEqual(external,[]);
  console.log('PASS: bookshelf, three PDFs, math text, bookmarks, scroll/zoom resume, notes, server restart, roadmap, lab, conflict protection, mobile, no external requests.');
  console.log('Screenshots and isolated test state: '+state);
})().catch(error=>{console.error(error);process.exitCode=1;}).finally(async()=>{
  await browser?.close();await stopServer();
});
