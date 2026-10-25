// Cosmos Integrator Lab: the clock, the input and the frame loop.
//
// Rust (cosmos_web, compiled to WebAssembly) owns the simulation and is
// called once per frame with the simulation time to show. Simulation time
// advances by the wall-clock frame time, clamped so a stalled tab cannot ask
// for minutes of catch-up (the "spiral of death" guard), times a user time
// scale. Rendering is three.js WebGPURenderer, which falls back to WebGL 2
// where WebGPU is unavailable and asks for the discrete GPU.

import * as THREE from "three/webgpu";

import "./style.css";
import { LogChart } from "./chart";
import { OrbitFrame, RigidFrame } from "./layout";
import { OrbitView } from "./orbitView";
import { RigidBodyView } from "./rigidBodyView";
import { SERIES_TOKENS, onThemeChange } from "./theme";
import { OrbitLab, RigidBodyLab, WasmView, loadSimulation } from "./wasm";

const MAXIMUM_FRAME_SECONDS = 0.1;
const RIGID_SAMPLE_INTERVAL = 0.05;
const ORBIT_SAMPLE_INTERVAL = 0.02;
const TORQUE = 0.6;
const TORQUE_DURATION = 0.5;

const element = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;

function scientific(value: number, digits = 1): string
{
  if (!Number.isFinite(value)) return "—";
  if (value === 0) return "0";
  return value.toExponential(digits).replace("e-", "e−");
}

function powerOfTen(exponent: number): string
{
  return `1e${exponent}`.replace("e-", "e−");
}

class Clock
{
  time = 0;
  playing = true;
  scale = 1;

  advance(wallSeconds: number): void
  {
    if (this.playing)
    {
      this.time += Math.min(wallSeconds, MAXIMUM_FRAME_SECONDS) * this.scale;
    }
  }
}

function showFailure(message: string): void
{
  const failure = element<HTMLParagraphElement>("failure");
  failure.textContent = message;
  failure.hidden = false;
}

async function main(): Promise<void>
{
  await loadSimulation();
  const canvas = element<HTMLCanvasElement>("scene");
  // `?webgl` forces the WebGL 2 backend, for comparison or old drivers.
  const renderer = new THREE.WebGPURenderer({
    canvas,
    antialias: true,
    powerPreference: "high-performance",
    forceWebGL: new URLSearchParams(location.search).has("webgl"),
  });
  await renderer.init();
  const backend = renderer.backend as { isWebGPUBackend?: boolean };
  const backendChip = element<HTMLSpanElement>("backend");
  backendChip.textContent = backend.isWebGPUBackend ? "WebGPU" : "WebGL 2 fallback";
  backendChip.classList.add("good");
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));

  const bodyCount = RigidBodyLab.bodyCount();
  const rigidStride = RigidBodyLab.frameStride();
  const orbitStride = OrbitLab.frameStride();
  const rigidNames = Array.from({ length: bodyCount }, (_, i) => RigidBodyLab.methodName(i));
  const orbitNames = ["DOPRI5", "DOP853"];
  const shortNames = rigidNames.map((name) => name.replace("Implicit midpoint", "Midpoint"));

  // ---------- rigid body ----------
  const rigidView = new RigidBodyView(bodyCount);
  const rigidClock = new Clock();
  const rigidFrame = new WasmView();
  const energyChart = new LogChart(element("chart-energy"), {
    title: "Energy error |ΔE/E| vs reference",
    format: (v) => scientific(Math.abs(v), 2),
    floor: 1e-16,
    series: rigidNames.map((name, i) => ({ name, label: shortNames[i], colorToken: SERIES_TOKENS[i] })),
  });
  const attitudeChart = new LogChart(element("chart-attitude"), {
    title: "Attitude error vs reference (rad)",
    format: (v) => scientific(v, 2),
    floor: 1e-14,
    series: rigidNames.map((name, i) => ({ name, label: shortNames[i], colorToken: SERIES_TOKENS[i] })),
  });
  let rigidLab: RigidBodyLab | null = null;
  let torqueUntil = -1;
  let lastRigidSample = -Infinity;
  const sampleBuffer = new Float64Array(bodyCount);

  const scenario = element<HTMLSelectElement>("scenario");
  const timeStep = element<HTMLInputElement>("time-step");
  const tolerance = element<HTMLInputElement>("tolerance");
  const timeScale = element<HTMLInputElement>("time-scale");
  const timeStepValue = () => Math.pow(10, Number(timeStep.value));
  const toleranceValue = () => Math.pow(10, Number(tolerance.value));

  const labels = element<HTMLDivElement>("labels-rigid");
  const labelDetails: HTMLSpanElement[] = [];
  const labelBadges: HTMLSpanElement[] = [];
  rigidNames.forEach((name, i) =>
  {
    const cell = document.createElement("div");
    cell.className = "quadrant";
    cell.innerHTML = `<span class="name"><span class="swatch" style="background:var(${SERIES_TOKENS[i]})"></span>${name}</span>`;
    const detail = document.createElement("span");
    detail.className = "detail mono";
    const badge = document.createElement("span");
    badge.className = "badge";
    badge.textContent = "step failed";
    badge.hidden = true;
    cell.append(detail, badge);
    labels.append(cell);
    labelDetails.push(detail);
    labelBadges.push(badge);
  });

  const rigidRows = rigidNames.map((name, i) =>
  {
    const row = document.createElement("tr");
    row.innerHTML = `<td><span class="swatch" style="display:inline-block;background:var(${SERIES_TOKENS[i]})"></span> ${name}</td><td></td><td></td><td></td><td></td>`;
    element<HTMLTableElement>("stats-rigid").tBodies[0].append(row);
    return row;
  });

  function rebuildRigid(): void
  {
    rigidLab?.free();
    rigidLab = new RigidBodyLab(Number(scenario.value), timeStepValue(), toleranceValue());
    rigidClock.time = 0;
    torqueUntil = -1;
    lastRigidSample = -Infinity;
    energyChart.clear();
    attitudeChart.clear();
    element("time-step-value").textContent = `${timeStepValue().toFixed(3)} s`;
    element("tolerance-value").textContent = powerOfTen(Number(tolerance.value));
  }

  function updateTimeScale(input: HTMLInputElement, output: string, clock: Clock): void
  {
    clock.scale = Math.pow(2, Number(input.value));
    element(output).textContent = `${clock.scale}×`;
  }

  scenario.addEventListener("change", rebuildRigid);
  timeStep.addEventListener("input", rebuildRigid);
  tolerance.addEventListener("input", rebuildRigid);
  timeScale.addEventListener("input", () => updateTimeScale(timeScale, "time-scale-value", rigidClock));
  element("restart").addEventListener("click", rebuildRigid);
  const playButton = element<HTMLButtonElement>("play");
  playButton.addEventListener("click", () =>
  {
    rigidClock.playing = !rigidClock.playing;
    playButton.textContent = rigidClock.playing ? "Pause" : "Play";
  });
  element("torque").addEventListener("click", () =>
  {
    rigidLab?.setTorque(0, 0, TORQUE);
    torqueUntil = rigidClock.time + TORQUE_DURATION;
  });
  rebuildRigid();
  updateTimeScale(timeScale, "time-scale-value", rigidClock);

  // ---------- orbit ----------
  const orbitView = new OrbitView(OrbitLab.mu(), orbitNames.length);
  const orbitClock = new Clock();
  const orbitFrame = new WasmView();
  const trailViews = orbitNames.map(() => new WasmView());
  const markerViews = orbitNames.map(() => new WasmView());
  const stepChart = new LogChart(element("chart-step"), {
    title: "Accepted step size h",
    format: (v) => scientific(v, 2),
    floor: 1e-8,
    series: orbitNames.map((name, i) => ({ name, label: name, colorToken: SERIES_TOKENS[i] })),
  }, 3000);
  let orbitLab: OrbitLab | null = null;
  let lastOrbitSample = -Infinity;
  const orbitTolerance = element<HTMLInputElement>("orbit-tolerance");
  const orbitTimeScale = element<HTMLInputElement>("orbit-time-scale");

  const orbitRows = orbitNames.map((name, i) =>
  {
    const row = document.createElement("tr");
    row.innerHTML = `<td><span class="swatch" style="display:inline-block;background:var(${SERIES_TOKENS[i]})"></span> ${name}</td><td></td><td></td><td></td><td></td>`;
    element<HTMLTableElement>("stats-orbit").tBodies[0].append(row);
    return row;
  });

  const orbitKey = element<HTMLDivElement>("legend-orbit");
  orbitKey.innerHTML = orbitNames
    .map((name, i) => `<span><span class="swatch" style="background:var(${SERIES_TOKENS[i]})"></span>${name}</span>`)
    .join("") + `<span>● Earth (left)</span><span>● Moon (right)</span>`;

  function rebuildOrbit(): void
  {
    orbitLab?.free();
    orbitLab = new OrbitLab(Math.pow(10, Number(orbitTolerance.value)));
    orbitClock.time = 0;
    lastOrbitSample = -Infinity;
    stepChart.clear();
    element("orbit-tolerance-value").textContent = powerOfTen(Number(orbitTolerance.value));
  }
  orbitTolerance.addEventListener("input", rebuildOrbit);
  orbitTimeScale.addEventListener("input", () => updateTimeScale(orbitTimeScale, "orbit-time-scale-value", orbitClock));
  element("orbit-restart").addEventListener("click", rebuildOrbit);
  const orbitPlay = element<HTMLButtonElement>("orbit-play");
  orbitPlay.addEventListener("click", () =>
  {
    orbitClock.playing = !orbitClock.playing;
    orbitPlay.textContent = orbitClock.playing ? "Pause" : "Play";
  });
  rebuildOrbit();
  updateTimeScale(orbitTimeScale, "orbit-time-scale-value", orbitClock);

  // ---------- tabs, resize, theme ----------
  let active: "rigid" | "orbit" = "rigid";
  const tabs = { rigid: element<HTMLButtonElement>("tab-rigid"), orbit: element<HTMLButtonElement>("tab-orbit") };
  function select(tab: "rigid" | "orbit"): void
  {
    active = tab;
    tabs.rigid.setAttribute("aria-selected", String(tab === "rigid"));
    tabs.orbit.setAttribute("aria-selected", String(tab === "orbit"));
    element("panel-rigid").hidden = tab !== "rigid";
    element("panel-orbit").hidden = tab !== "orbit";
    labels.hidden = tab !== "rigid";
    orbitKey.hidden = tab !== "orbit";
    energyChart.invalidate();
    attitudeChart.invalidate();
    stepChart.invalidate();
    try { localStorage.setItem("cosmos-lab-tab", tab); } catch { /* storage unavailable */ }
  }
  tabs.rigid.addEventListener("click", () => select("rigid"));
  tabs.orbit.addEventListener("click", () => select("orbit"));
  let remembered: string | null = null;
  try { remembered = localStorage.getItem("cosmos-lab-tab"); } catch { /* storage unavailable */ }
  select(location.hash === "#orbit" || remembered === "orbit" ? "orbit" : "rigid");

  const viewport = canvas.parentElement!;
  new ResizeObserver(() =>
  {
    const width = viewport.clientWidth;
    const height = viewport.clientHeight;
    renderer.setSize(width, height, false);
    rigidView.resize(width, height);
    orbitView.resize(width, height);
    energyChart.invalidate();
    attitudeChart.invalidate();
    stepChart.invalidate();
  }).observe(viewport);

  onThemeChange(() =>
  {
    rigidView.applyTheme();
    orbitView.applyTheme();
    energyChart.invalidate();
    attitudeChart.invalidate();
    stepChart.invalidate();
  });

  // ---------- frame loop ----------
  const clockChip = element("clock");
  const costChip = element("cost");
  let previous = performance.now();
  let lastTable = 0;
  let smoothedCost = 0;

  renderer.setAnimationLoop(() =>
  {
    const now = performance.now();
    const wall = (now - previous) / 1000;
    previous = now;
    const refreshTable = now - lastTable > 100;
    if (refreshTable) lastTable = now;
    let cost = 0;

    if (active === "rigid" && rigidLab !== null)
    {
      rigidClock.advance(wall);
      if (torqueUntil >= 0 && rigidClock.time >= torqueUntil)
      {
        rigidLab.advanceTo(torqueUntil);
        rigidLab.setTorque(0, 0, 0);
        torqueUntil = -1;
      }
      const start = performance.now();
      rigidLab.advanceTo(rigidClock.time);
      cost = performance.now() - start;
      const frame = rigidFrame.read(rigidLab.framePointer(), rigidLab.frameLength());
      rigidView.update(frame, rigidStride);

      if (rigidClock.time - lastRigidSample >= RIGID_SAMPLE_INTERVAL)
      {
        lastRigidSample = rigidClock.time;
        for (let i = 0; i < bodyCount; i++) sampleBuffer[i] = frame[i * rigidStride + RigidFrame.energyError];
        energyChart.push(rigidClock.time, sampleBuffer);
        for (let i = 0; i < bodyCount; i++) sampleBuffer[i] = frame[i * rigidStride + RigidFrame.attitudeError];
        attitudeChart.push(rigidClock.time, sampleBuffer);
      }
      if (refreshTable)
      {
        for (let i = 0; i < bodyCount; i++)
        {
          const base = i * rigidStride;
          const failed = frame[base + RigidFrame.failed] !== 0;
          const cells = rigidRows[i].cells;
          const evaluations = frame[base + RigidFrame.evaluations];
          cells[1].textContent = String(frame[base + RigidFrame.accepted]);
          cells[2].textContent = evaluations > 0 ? String(evaluations) : "—";
          cells[3].textContent = failed ? "failed" : scientific(Math.abs(frame[base + RigidFrame.energyError]));
          cells[4].textContent = failed ? "failed" : scientific(frame[base + RigidFrame.attitudeError]);
          cells[3].classList.toggle("failed", failed);
          cells[4].classList.toggle("failed", failed);
          labelBadges[i].hidden = !failed;
          labelDetails[i].textContent = i < 2
            ? `fixed Δt ${timeStepValue().toFixed(3)} s`
            : `tol ${powerOfTen(Number(tolerance.value))} · h ${frame[base + RigidFrame.stepSize].toFixed(4)} s`;
        }
        clockChip.textContent = `t = ${rigidClock.time.toFixed(2)} s`;
      }
      rigidView.camera.updateMatrixWorld();
      renderer.render(rigidView.scene, rigidView.camera);
      energyChart.draw();
      attitudeChart.draw();
    }
    else if (active === "orbit" && orbitLab !== null)
    {
      orbitClock.advance(wall);
      const start = performance.now();
      orbitLab.advanceTo(orbitClock.time);
      cost = performance.now() - start;
      const frame = orbitFrame.read(orbitLab.framePointer(), orbitNames.length * orbitStride);
      const lab = orbitLab;
      const trails = trailViews.map((view, i) => view.read(lab.trailPointer(i), lab.trailLength(i)));
      const markers = markerViews.map((view, i) => view.read(lab.markersPointer(i), lab.markersLength(i)));
      orbitView.update(frame, orbitStride, trails, markers);

      if (orbitClock.time - lastOrbitSample >= ORBIT_SAMPLE_INTERVAL)
      {
        lastOrbitSample = orbitClock.time;
        for (let i = 0; i < orbitNames.length; i++) sampleBuffer[i] = frame[i * orbitStride + OrbitFrame.stepSize];
        stepChart.push(orbitClock.time, sampleBuffer.subarray(0, orbitNames.length));
      }
      if (refreshTable)
      {
        for (let i = 0; i < orbitNames.length; i++)
        {
          const base = i * orbitStride;
          const cells = orbitRows[i].cells;
          const periods = frame[base + OrbitFrame.periods];
          cells[1].textContent = String(frame[base + OrbitFrame.accepted]);
          cells[2].textContent = String(frame[base + OrbitFrame.rejected]);
          cells[3].textContent = String(frame[base + OrbitFrame.evaluations]);
          cells[4].textContent = periods > 0
            ? `${scientific(frame[base + OrbitFrame.closingError])} (${periods} T)`
            : "after 1 T";
        }
        clockChip.textContent = `t = ${orbitClock.time.toFixed(2)} (T = ${OrbitLab.period().toFixed(3)})`;
      }
      renderer.render(orbitView.scene, orbitView.camera);
      stepChart.draw();
    }

    smoothedCost = 0.9 * smoothedCost + 0.1 * cost;
    if (refreshTable) costChip.textContent = `wasm ${smoothedCost.toFixed(2)} ms/frame`;
  });
}

main().catch((error: unknown) =>
{
  const message = error instanceof Error ? error.message : String(error);
  showFailure(`The simulation could not start: ${message}. It needs WebAssembly and either WebGPU or WebGL 2.`);
  console.error(error);
});
