// Four copies of one T-handle, one per integrator, each in its own quadrant
// of an orthographic view. Each body's inertial frame is tilted the same way
// for viewing; inside it the body carries its attitude quaternion. The
// grey arrow is the inertial angular momentum L (fixed when no torque acts
// and the integrator conserves it); the colored arrow is ω, drawn in body
// coordinates.

import * as THREE from "three/webgpu";

import { RigidFrame } from "./layout";
import { SERIES_TOKENS, token } from "./theme";

interface Body
{
  container: THREE.Group;
  attitude: THREE.Group;
  momentum: THREE.ArrowHelper;
  angularVelocity: THREE.ArrowHelper;
  knob: THREE.MeshStandardMaterial;
  metal: THREE.MeshStandardMaterial;
}

const VIEW_TILT = new THREE.Euler(0.5, -0.65, 0.0);
const BODY_SCALE = 0.3;

export class RigidBodyView
{
  readonly scene = new THREE.Scene();
  readonly camera = new THREE.OrthographicCamera(-1, 1, 1, -1, -10, 10);
  private readonly bodies: Body[] = [];
  private readonly direction = new THREE.Vector3();
  private readonly hemisphere = new THREE.HemisphereLight(0xffffff, 0x444444, 1.6);
  private readonly sun = new THREE.DirectionalLight(0xffffff, 2.2);

  constructor(count: number)
  {
    this.sun.position.set(-1.5, 2.5, 3.0);
    this.scene.add(this.hemisphere, this.sun);
    const crossbar = new THREE.CylinderGeometry(0.08, 0.08, 1.6, 24);
    crossbar.rotateZ(Math.PI / 2);
    const shaft = new THREE.CylinderGeometry(0.06, 0.06, 1.0, 20);
    shaft.translate(0, 0.5, 0);
    const knob = new THREE.SphereGeometry(0.15, 28, 18);
    knob.translate(0, 1.0, 0);

    for (let i = 0; i < count; i++)
    {
      const metal = new THREE.MeshStandardMaterial({ metalness: 0.35, roughness: 0.4 });
      const knobMaterial = new THREE.MeshStandardMaterial({ metalness: 0.1, roughness: 0.35 });
      const attitude = new THREE.Group();
      attitude.add(
        new THREE.Mesh(crossbar, metal),
        new THREE.Mesh(shaft, metal),
        new THREE.Mesh(knob, knobMaterial));
      const angularVelocity = new THREE.ArrowHelper(
        new THREE.Vector3(0, 1, 0), new THREE.Vector3(), 1.35, 0xffffff, 0.16, 0.08);
      attitude.add(angularVelocity);
      const container = new THREE.Group();
      container.rotation.copy(VIEW_TILT);
      const momentum = new THREE.ArrowHelper(
        new THREE.Vector3(0, 1, 0), new THREE.Vector3(), 1.6, 0xffffff, 0.18, 0.09);
      container.add(attitude, momentum);
      container.scale.setScalar(BODY_SCALE);
      this.scene.add(container);
      this.bodies.push({ container, attitude, momentum, angularVelocity, knob: knobMaterial, metal });
    }
    this.applyTheme();
  }

  applyTheme(): void
  {
    this.scene.background = new THREE.Color(token("--scene"));
    const ink = new THREE.Color(token("--ink-secondary"));
    const metal = new THREE.Color(token("--body-metal"));
    this.bodies.forEach((body, i) =>
    {
      const series = new THREE.Color(token(SERIES_TOKENS[i]));
      body.knob.color.copy(series);
      body.metal.color.copy(metal);
      body.angularVelocity.setColor(series);
      body.momentum.setColor(ink);
    });
  }

  /** Quadrant centres in a 2 × 2 grid that fills the canvas. */
  resize(width: number, height: number): void
  {
    const aspect = width / Math.max(height, 1);
    this.camera.left = -aspect;
    this.camera.right = aspect;
    this.camera.top = 1;
    this.camera.bottom = -1;
    this.camera.updateProjectionMatrix();
    const scale = BODY_SCALE * Math.min(aspect, 1.0);
    this.bodies.forEach((body, i) =>
    {
      const column = i % 2;
      const row = Math.floor(i / 2);
      body.container.position.set((column - 0.5) * aspect, (0.5 - row) * 1.0 - 0.06, 0);
      body.container.scale.setScalar(scale);
    });
  }

  update(frame: Float64Array, stride: number): void
  {
    this.bodies.forEach((body, i) =>
    {
      const base = i * stride;
      if (frame[base + RigidFrame.failed] !== 0)
      {
        return;
      }
      const q = RigidFrame.quaternion;
      body.attitude.quaternion.set(
        frame[base + q + 1], frame[base + q + 2], frame[base + q + 3], frame[base + q]);
      const w = RigidFrame.angularVelocity;
      this.direction.set(frame[base + w], frame[base + w + 1], frame[base + w + 2]);
      if (this.direction.lengthSq() > 0)
      {
        body.angularVelocity.setDirection(this.direction.normalize());
      }
      const l = RigidFrame.angularMomentumInertial;
      this.direction.set(frame[base + l], frame[base + l + 1], frame[base + l + 2]);
      if (this.direction.lengthSq() > 0)
      {
        body.momentum.setDirection(this.direction.normalize());
      }
    });
  }
}
