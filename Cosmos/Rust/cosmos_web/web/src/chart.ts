// A small log-scale time-series chart on a 2D canvas: one y axis, decade
// grid lines, direct labels at the line ends, a crosshair tooltip on hover.
// Samples live in preallocated ring buffers; nothing is allocated per frame.

import { token } from "./theme";

export interface SeriesSpec
{
  name: string;
  /** The direct label at the line's end; short enough for the margin. */
  label: string;
  colorToken: string;
}

export interface ChartSpec
{
  title: string;
  /** Formats a value for the tooltip. */
  format: (value: number) => string;
  /** The smallest value shown; zeros and smaller clamp to it. */
  floor: number;
  series: SeriesSpec[];
}

const MARGIN = { left: 44, right: 78, top: 8, bottom: 22 };

export class LogChart
{
  private readonly canvas: HTMLCanvasElement;
  private readonly context: CanvasRenderingContext2D;
  private readonly tooltip: HTMLDivElement;
  private readonly times: Float64Array;
  private readonly values: Float64Array[];
  private count = 0;
  private head = 0;
  private hoverX: number | null = null;
  private dirty = true;

  constructor(
    host: HTMLElement,
    private readonly spec: ChartSpec,
    private readonly capacity = 2400)
  {
    const caption = document.createElement("figcaption");
    caption.textContent = spec.title;
    this.canvas = document.createElement("canvas");
    this.canvas.setAttribute("role", "img");
    this.canvas.setAttribute("aria-label",
      `${spec.title}, log scale, one line per integrator; numbers in the table above`);
    this.tooltip = document.createElement("div");
    this.tooltip.className = "tooltip";
    this.tooltip.hidden = true;
    host.append(caption, this.canvas, this.tooltip);
    this.context = this.canvas.getContext("2d")!;
    this.times = new Float64Array(capacity);
    this.values = spec.series.map(() => new Float64Array(capacity));

    this.canvas.addEventListener("pointermove", (event) =>
    {
      const bounds = this.canvas.getBoundingClientRect();
      this.hoverX = event.clientX - bounds.left;
      this.dirty = true;
    });
    this.canvas.addEventListener("pointerleave", () =>
    {
      this.hoverX = null;
      this.tooltip.hidden = true;
      this.dirty = true;
    });
  }

  clear(): void
  {
    this.count = 0;
    this.head = 0;
    this.dirty = true;
  }

  push(time: number, values: ArrayLike<number>): void
  {
    this.times[this.head] = time;
    for (let s = 0; s < this.values.length; s++)
    {
      this.values[s][this.head] = values[s];
    }
    this.head = (this.head + 1) % this.capacity;
    this.count = Math.min(this.count + 1, this.capacity);
    this.dirty = true;
  }

  invalidate(): void { this.dirty = true; }

  private index(i: number): number
  {
    return (this.head - this.count + i + this.capacity) % this.capacity;
  }

  private logValue(value: number): number
  {
    const magnitude = Math.abs(value);
    return Math.log10(Number.isFinite(magnitude)
      ? Math.max(magnitude, this.spec.floor) : this.spec.floor);
  }

  draw(): void
  {
    if (!this.dirty) return;
    this.dirty = false;
    const ratio = Math.min(window.devicePixelRatio || 1, 2);
    const width = this.canvas.clientWidth;
    const height = this.canvas.clientHeight;
    if (width === 0 || height === 0) return;
    if (this.canvas.width !== Math.round(width * ratio)
      || this.canvas.height !== Math.round(height * ratio))
    {
      this.canvas.width = Math.round(width * ratio);
      this.canvas.height = Math.round(height * ratio);
    }
    const c = this.context;
    c.setTransform(ratio, 0, 0, ratio, 0, 0);
    c.clearRect(0, 0, width, height);
    const ink = token("--ink");
    const muted = token("--ink-muted");
    const rule = token("--rule");
    const surface = token("--surface");
    const plotWidth = width - MARGIN.left - MARGIN.right;
    const plotHeight = height - MARGIN.top - MARGIN.bottom;
    c.font = `11px ${token("--font-data")}`;

    let low = Infinity;
    let high = -Infinity;
    for (let s = 0; s < this.values.length; s++)
    {
      for (let i = 0; i < this.count; i++)
      {
        const v = this.logValue(this.values[s][this.index(i)]);
        low = Math.min(low, v);
        high = Math.max(high, v);
      }
    }
    if (!Number.isFinite(low))
    {
      low = Math.log10(this.spec.floor);
      high = low + 1;
    }
    low = Math.floor(low);
    high = Math.max(Math.ceil(high), low + 1);
    const t0 = this.count > 0 ? this.times[this.index(0)] : 0;
    const t1 = this.count > 1 ? this.times[this.index(this.count - 1)] : t0 + 1;
    const x = (t: number) => MARGIN.left + ((t - t0) / Math.max(t1 - t0, 1e-9)) * plotWidth;
    const y = (logValue: number) =>
      MARGIN.top + (1 - (logValue - low) / (high - low)) * plotHeight;

    // Decade grid and labels: every label is a value the axis reaches.
    const decadeStep = Math.max(1, Math.ceil((high - low) / 5));
    c.lineWidth = 1;
    c.textAlign = "right";
    c.textBaseline = "middle";
    for (let d = low; d <= high; d += decadeStep)
    {
      const py = Math.round(y(d)) + 0.5;
      c.strokeStyle = rule;
      c.beginPath();
      c.moveTo(MARGIN.left, py);
      c.lineTo(MARGIN.left + plotWidth, py);
      c.stroke();
      c.fillStyle = muted;
      c.fillText(d === 0 ? "1" : `1e${d}`, MARGIN.left - 6, py);
    }
    c.textAlign = "center";
    c.textBaseline = "top";
    c.fillText(`t = ${t0.toFixed(1)} s`, MARGIN.left + 26, MARGIN.top + plotHeight + 6);
    c.fillText(`${t1.toFixed(1)} s`, MARGIN.left + plotWidth - 14, MARGIN.top + plotHeight + 6);

    // Lines, 2px, in series order.
    const ends: { y: number; name: string; color: string }[] = [];
    for (let s = 0; s < this.values.length; s++)
    {
      const color = token(this.spec.series[s].colorToken);
      c.strokeStyle = color;
      c.lineWidth = 2;
      c.lineJoin = "round";
      c.beginPath();
      for (let i = 0; i < this.count; i++)
      {
        const k = this.index(i);
        const px = x(this.times[k]);
        const py = y(this.logValue(this.values[s][k]));
        if (i === 0) c.moveTo(px, py); else c.lineTo(px, py);
      }
      c.stroke();
      if (this.count > 0)
      {
        const last = this.index(this.count - 1);
        ends.push({ y: y(this.logValue(this.values[s][last])),
          name: this.spec.series[s].label, color });
      }
    }

    // Direct labels at the right end, pushed apart to avoid collisions.
    ends.sort((a, b) => a.y - b.y);
    for (let i = 1; i < ends.length; i++)
    {
      ends[i].y = Math.max(ends[i].y, ends[i - 1].y + 13);
    }
    const overflow = ends.length > 0
      ? ends[ends.length - 1].y - (MARGIN.top + plotHeight) : 0;
    c.textAlign = "left";
    c.textBaseline = "middle";
    for (const end of ends)
    {
      const ly = overflow > 0 ? end.y - overflow : end.y;
      c.fillStyle = end.color;
      c.fillRect(MARGIN.left + plotWidth + 6, ly - 3, 6, 6);
      c.fillStyle = ink;
      c.fillText(end.name, MARGIN.left + plotWidth + 16, ly);
    }

    // Crosshair and tooltip.
    if (this.hoverX !== null && this.count > 1)
    {
      const fraction = (this.hoverX - MARGIN.left) / plotWidth;
      const target = t0 + Math.min(Math.max(fraction, 0), 1) * (t1 - t0);
      let nearest = 0;
      let best = Infinity;
      for (let i = 0; i < this.count; i++)
      {
        const distance = Math.abs(this.times[this.index(i)] - target);
        if (distance < best) { best = distance; nearest = i; }
      }
      const k = this.index(nearest);
      const px = Math.round(x(this.times[k])) + 0.5;
      c.strokeStyle = muted;
      c.lineWidth = 1;
      c.beginPath();
      c.moveTo(px, MARGIN.top);
      c.lineTo(px, MARGIN.top + plotHeight);
      c.stroke();
      const rows = this.spec.series.map((series, s) =>
      {
        const py = y(this.logValue(this.values[s][k]));
        c.fillStyle = token(series.colorToken);
        c.beginPath();
        c.arc(px, py, 4, 0, 2 * Math.PI);
        c.fill();
        c.strokeStyle = surface;
        c.lineWidth = 2;
        c.stroke();
        return `<div class="row"><span class="swatch" style="background:var(${series.colorToken})"></span>`
          + `${series.name}: ${this.spec.format(this.values[s][k])}</div>`;
      });
      this.tooltip.innerHTML = `<div>t = ${this.times[k].toFixed(2)} s</div>${rows.join("")}`;
      this.tooltip.hidden = false;
      const left = Math.min(px + 12, width - this.tooltip.offsetWidth - 4);
      this.tooltip.style.left = `${Math.max(left, 0)}px`;
      this.tooltip.style.top = `${MARGIN.top + 22}px`;
    }
  }
}
