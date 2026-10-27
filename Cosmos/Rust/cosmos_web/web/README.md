# Cosmos Integrator Lab

A real-time viewer for the Rust integrators in `cosmos_numerical`: four
integrators on one rigid body (RK4, implicit midpoint, DOPRI5, DOP853 against a
DOP853 reference at 1e-13), and the Arenstorf orbit of HNW I §II.0 under DOPRI5
and DOP853 with every accepted step drawn.

## Run it

One command from a clone, on Linux or macOS:

```bash
Cosmos/Rust/cosmos_web/run-lab.sh
```

It builds the lab and opens http://127.0.0.1:4173 in the default browser. It
uses the local toolchain when it is complete, and Docker otherwise, so Docker
alone is enough. `--docker` or `--local` forces a route, `--port N` changes the
port, and Ctrl-C stops the server. Append `?webgl` to the URL to force the
WebGL 2 backend.

To launch with a double-click:

- **macOS:** double-click `Cosmos Integrator Lab.command` in Finder.
- **Linux:** run `Cosmos/Rust/cosmos_web/run-lab.sh --install-launcher` once.
  "Cosmos Integrator Lab" then appears in the application menu (Activities on
  GNOME) and opens a terminal that runs the lab.

Docker by hand, from the repository root:

```bash
docker build -f Cosmos/Rust/cosmos_web/Dockerfile -t cosmos-integrator-lab .
docker run --rm -p 4173:80 cosmos-integrator-lab
```

The build context is the repository root because `cosmos_numerical` depends on
`Stunticons/wildrider`; `Dockerfile.dockerignore` sends only those two
directories. `.github/workflows/cosmos-integrator-lab.yml` builds the same
Dockerfile's `dist` stage and publishes it to GitHub Pages on pushes to
`master` (one-time setup: Settings → Pages → Source: GitHub Actions).

## Develop

```bash
cd Cosmos/Rust/cosmos_web/web
npm install
npm run dev        # builds the wasm, then serves on http://localhost:5173
npm run build      # dist/ (any static host) and dist/artifact.html
npx vite preview   # serves dist/ on http://localhost:4173
```

Prerequisites: `rustup target add wasm32-unknown-unknown` and
`cargo install wasm-bindgen-cli --version 0.2.129 --locked` (the version must
equal the `wasm-bindgen` pin in `cosmos_web/Cargo.toml`; `scripts/build-wasm.sh`
checks). `npm run build:web` is the TypeScript half alone, for when
`src/generated/` already holds the bindings.

## Architecture

Rust is the simulation; TypeScript is the clock, the input and the pixels.

- `cosmos_web` (Rust, `cdylib`) exposes two classes through `wasm-bindgen`:
  `RigidBodyLab` and `OrbitLab`. The viewer calls `advanceTo(t)` once per frame.
  The lab logic is ordinary Rust in `rigid_body_lab.rs` / `orbit_lab.rs` with
  native `cargo test`s; `lib.rs` only adapts it to JavaScript.
- Results are read without copying: Rust writes into buffers it allocated once
  (`framePointer()`, `trailPointer()`, `markersPointer()`), and `src/wasm.ts`
  wraps them in `Float64Array` views over the module's memory, rebuilt only if
  the memory grows (growth detaches the old `ArrayBuffer`).
- Rendering is three.js `WebGPURenderer` (`three/webgpu`), which uses WebGPU
  where the browser has it and falls back to WebGL 2.

## Practices taken from real-time rendering

- **Decouple the simulation clock from the frame rate.** The fixed-step
  integrators step with their own Δt and the renderer shows a blend of the last
  two states (slerp for the attitude), the scheme of Glenn Fiedler's
  ["Fix Your Timestep!"](https://gafferongames.com/post/fix_your_timestep/). The
  adaptive integrators do better than a blend: they return their own dense
  output (HNW I §II.6) at the frame time.
- **Guard against the spiral of death.** A frame never advances more than 0.1 s
  of wall time, so a backgrounded tab cannot demand minutes of catch-up; Rust
  also caps fixed steps per frame.
- **No per-frame allocation, persistent buffers.** GPU buffers (trails,
  instanced step markers) are allocated at capacity once and updated in place;
  markers are appended incrementally. This is the same advice as for Vulkan
  ([Arm](https://developer.arm.com/community/arm-community-blogs/b/mobile-graphics-and-gaming-blog/posts/vulkan-descriptor-and-buffer-management),
  [NVIDIA's Vulkan dos and don'ts](https://developer.nvidia.com/blog/vulkan-dos-donts)):
  allocate big, keep it, avoid per-frame creation.
- **f64 in the simulation, f32 on the GPU.** WGSL has no f64; positions are
  narrowed when copied into vertex buffers. Orbit coordinates are O(1), so the
  rounding is far below a pixel. A scene at planetary scale would need
  camera-relative coordinates before narrowing.
- **Ask for the discrete GPU.** The renderer requests
  `powerPreference: "high-performance"`. It is a hint
  ([MDN](https://developer.mozilla.org/docs/Web/API/GPU/requestAdapter)); on
  hybrid laptops Chrome may still pick the integrated GPU
  ([Chrome WebGPU troubleshooting](https://developer.chrome.com/docs/web-platform/webgpu/troubleshooting-tips)).
- **Cap the device-pixel ratio at 2** and let the browser's `requestAnimationFrame`
  pause the loop in hidden tabs.

## What the numbers mean

The rigid-body errors are against the reference run at the same time: relative
kinetic-energy error, relative `|L|` error, and attitude error
`2 arccos |⟨q, q_ref⟩|`. With no torque, energy and `|L|` are invariants;
implicit midpoint keeps both to round-off (GNI §IV.2), RK4 drifts, and DOPRI5 /
DOP853 drift at about their tolerance. A torque pulse switches on at a frame
time; the adaptive runs restart there from their dense output
(`EmbeddedRungeKutta::restart_at`).

The orbit's return error is `max_i |y_i(kT) − y_i(0)|` after each period: the
global error the tolerance actually bought.
