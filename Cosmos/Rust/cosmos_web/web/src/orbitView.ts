// The Arenstorf orbit in the rotating Earth–Moon frame, top down. Each
// integrator draws a trail of its dense output (one point per frame) and a
// dot at every accepted step. Positions arrive as f64 from Rust and are
// narrowed to f32 for the GPU (WGSL has no f64); orbit coordinates are O(1),
// so single precision is far below a pixel.

import * as THREE from "three/webgpu";

import { MARKER_CAPACITY, OrbitFrame, TRAIL_CAPACITY } from "./layout";
import { SERIES_TOKENS, token } from "./theme";

const BOUNDS = { left: -1.35, right: 1.2, bottom: -1.2, top: 1.2 };
const MARKER_RADIUS = 0.0085;

interface Track
{
  trail: THREE.Line;
  trailPositions: Float32Array;
  markers: THREE.InstancedMesh;
  markerMaterial: THREE.MeshBasicMaterial;
  trailMaterial: THREE.LineBasicMaterial;
  satellite: THREE.Mesh;
  satelliteMaterial: THREE.MeshBasicMaterial;
  copiedMarkers: number;
  firstMarker: number;
}

export class OrbitView
{
  readonly scene = new THREE.Scene();
  readonly camera = new THREE.OrthographicCamera(-1, 1, 1, -1, -10, 10);
  private readonly tracks: Track[] = [];
  private readonly earthMaterial = new THREE.MeshBasicMaterial();
  private readonly moonMaterial = new THREE.MeshBasicMaterial();

  constructor(mu: number, count: number)
  {
    const earth = new THREE.Mesh(new THREE.CircleGeometry(0.06, 48), this.earthMaterial);
    earth.position.set(-mu, 0, -0.1);
    const moon = new THREE.Mesh(new THREE.CircleGeometry(0.025, 32), this.moonMaterial);
    moon.position.set(1 - mu, 0, -0.1);
    this.scene.add(earth, moon);

    const markerGeometry = new THREE.CircleGeometry(MARKER_RADIUS, 12);
    for (let i = 0; i < count; i++)
    {
      const trailPositions = new Float32Array(TRAIL_CAPACITY * 3);
      const geometry = new THREE.BufferGeometry();
      geometry.setAttribute("position",
        new THREE.BufferAttribute(trailPositions, 3).setUsage(THREE.DynamicDrawUsage));
      geometry.setDrawRange(0, 0);
      const trailMaterial = new THREE.LineBasicMaterial({ transparent: true, opacity: 0.85 });
      const trail = new THREE.Line(geometry, trailMaterial);
      trail.frustumCulled = false;

      const markerMaterial = new THREE.MeshBasicMaterial({ transparent: true, opacity: 0.75 });
      const markers = new THREE.InstancedMesh(markerGeometry, markerMaterial, MARKER_CAPACITY);
      markers.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
      markers.count = 0;
      markers.frustumCulled = false;
      markers.position.z = 0.01 * (i + 1);

      const satelliteMaterial = new THREE.MeshBasicMaterial();
      const satellite = new THREE.Mesh(new THREE.CircleGeometry(0.022, 24), satelliteMaterial);
      satellite.position.z = 0.1;
      this.scene.add(trail, markers, satellite);
      this.tracks.push({ trail, trailPositions, markers, markerMaterial, trailMaterial,
        satellite, satelliteMaterial, copiedMarkers: 0, firstMarker: NaN });
    }
    this.applyTheme();
  }

  applyTheme(): void
  {
    this.scene.background = new THREE.Color(token("--scene"));
    this.earthMaterial.color.set(token("--series-1"));
    this.earthMaterial.color.lerp(new THREE.Color(token("--ink-muted")), 0.55);
    this.moonMaterial.color.set(token("--ink-muted"));
    this.tracks.forEach((track, i) =>
    {
      const color = new THREE.Color(token(SERIES_TOKENS[i]));
      track.trailMaterial.color.copy(color);
      track.markerMaterial.color.copy(color);
      track.satelliteMaterial.color.copy(color);
    });
  }

  resize(width: number, height: number): void
  {
    const aspect = width / Math.max(height, 1);
    const spanX = BOUNDS.right - BOUNDS.left;
    const spanY = BOUNDS.top - BOUNDS.bottom;
    const centreX = (BOUNDS.left + BOUNDS.right) / 2;
    const centreY = (BOUNDS.top + BOUNDS.bottom) / 2;
    const halfY = Math.max(spanY, spanX / aspect) / 2;
    this.camera.left = centreX - halfY * aspect;
    this.camera.right = centreX + halfY * aspect;
    this.camera.top = centreY + halfY;
    this.camera.bottom = centreY - halfY;
    this.camera.updateProjectionMatrix();
  }

  /** Screen position (CSS pixels) of an orbit-plane point, for labels. */
  project(x: number, y: number, width: number, height: number): [number, number]
  {
    const u = (x - this.camera.left) / (this.camera.right - this.camera.left);
    const v = (this.camera.top - y) / (this.camera.top - this.camera.bottom);
    return [u * width, v * height];
  }

  update(frame: Float64Array, stride: number, trails: Float64Array[], markers: Float64Array[]): void
  {
    this.tracks.forEach((track, i) =>
    {
      const base = i * stride;
      track.satellite.visible = frame[base + OrbitFrame.failed] === 0;
      track.satellite.position.x = frame[base + OrbitFrame.position];
      track.satellite.position.y = frame[base + OrbitFrame.position + 1];

      const trail = trails[i];
      const points = trail.length / 2;
      for (let p = 0; p < points; p++)
      {
        track.trailPositions[3 * p] = trail[2 * p];
        track.trailPositions[3 * p + 1] = trail[2 * p + 1];
      }
      const attribute = track.trail.geometry.getAttribute("position") as THREE.BufferAttribute;
      attribute.needsUpdate = true;
      track.trail.geometry.setDrawRange(0, points);

      // Markers only grow, except when Rust drops the oldest half; then the
      // first marker changes and everything is copied again.
      const values = markers[i];
      const count = values.length / 3;
      if (count < track.copiedMarkers || (count > 0 && values[0] !== track.firstMarker))
      {
        track.copiedMarkers = 0;
      }
      const matrices = track.markers.instanceMatrix.array as Float32Array;
      for (let m = track.copiedMarkers; m < count; m++)
      {
        const o = 16 * m;
        matrices[o] = 1; matrices[o + 1] = 0; matrices[o + 2] = 0; matrices[o + 3] = 0;
        matrices[o + 4] = 0; matrices[o + 5] = 1; matrices[o + 6] = 0; matrices[o + 7] = 0;
        matrices[o + 8] = 0; matrices[o + 9] = 0; matrices[o + 10] = 1; matrices[o + 11] = 0;
        matrices[o + 12] = values[3 * m]; matrices[o + 13] = values[3 * m + 1];
        matrices[o + 14] = 0; matrices[o + 15] = 1;
      }
      if (count !== track.copiedMarkers)
      {
        track.markers.instanceMatrix.needsUpdate = true;
      }
      track.copiedMarkers = count;
      track.firstMarker = count > 0 ? values[0] : NaN;
      track.markers.count = count;
    });
  }
}
