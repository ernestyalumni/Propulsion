import {getDocument, GlobalWorkerOptions, TextLayer} from '/vendor/pdfjs-dist/build/pdf.mjs';

GlobalWorkerOptions.workerSrc = '/vendor/pdfjs-dist/build/pdf.worker.mjs';

export class BookReader {
  constructor(viewport) {
    this.viewport = viewport;
    this.generation = 0;
    this.renderTask = null;
    this.textLayer = null;
  }
  async load(book) {
    this.loadingTask = getDocument({url: `/book/${book}/pdf`,
      cMapUrl: '/vendor/pdfjs-dist/cmaps/', cMapPacked: true,
      standardFontDataUrl: '/vendor/pdfjs-dist/standard_fonts/',
      wasmUrl: '/vendor/pdfjs-dist/wasm/', isEvalSupported: false});
    this.pdf = await this.loadingTask.promise;
    return this.pdf.numPages;
  }
  async render(number, zoom, scroll = 0) {
    const gen = ++this.generation;
    this.renderTask?.cancel();
    this.textLayer?.cancel();
    const page = await this.pdf.getPage(number);
    if (gen !== this.generation) return;
    const natural = page.getViewport({scale: 1});
    const scale = Math.max(200, this.viewport.clientWidth - 32) / natural.width * zoom;
    const view = page.getViewport({scale});
    const ratio = Math.min(window.devicePixelRatio || 1, 2);
    const sheet = document.createElement('div');
    sheet.className = 'pdf-sheet';
    sheet.style.width = `${view.width}px`;
    sheet.style.height = `${view.height}px`;
    sheet.style.setProperty('--scale-factor', scale);
    sheet.style.setProperty('--total-scale-factor', scale);
    const canvas = document.createElement('canvas');
    canvas.setAttribute('aria-label', `Original PDF, page ${number}`);
    canvas.width = Math.floor(view.width * ratio);
    canvas.height = Math.floor(view.height * ratio);
    canvas.style.width = `${view.width}px`;
    canvas.style.height = `${view.height}px`;
    sheet.append(canvas);
    this.viewport.replaceChildren(sheet);
    this.renderTask = page.render({canvasContext: canvas.getContext('2d'), viewport: view,
      transform: ratio === 1 ? null : [ratio, 0, 0, ratio, 0, 0]});
    try { await this.renderTask.promise; }
    catch (error) { if (error.name === 'RenderingCancelledException') return; throw error; }
    if (gen !== this.generation) return;
    const layer = document.createElement('div');
    layer.className = 'textLayer';
    sheet.append(layer);
    this.textLayer = new TextLayer({textContentSource: page.streamTextContent(), container: layer, viewport: view});
    await this.textLayer.render();
    if (gen === this.generation) this.viewport.scrollTop = scroll * Math.max(0, this.viewport.scrollHeight - this.viewport.clientHeight);
  }
  destroy() {
    this.generation++;
    this.renderTask?.cancel();
    this.textLayer?.cancel();
    this.loadingTask?.destroy();
  }
}
