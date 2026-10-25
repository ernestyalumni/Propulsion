// Loads the Rust simulation (cosmos_web) and reads its results in place.
//
// Rust writes each frame into buffers it allocated once; TypeScript reads
// them through Float64Array views over the module's linear memory, so no
// frame data is copied or serialized across the boundary. A view goes stale
// when WebAssembly memory grows (the old ArrayBuffer is detached), so
// `WasmView` rebuilds it only when the buffer, the address or the length
// changes, and never allocates on an ordinary frame.

import initialize, { OrbitLab, RigidBodyLab } from "./generated/cosmos_web.js";

export { OrbitLab, RigidBodyLab };

let memory: WebAssembly.Memory | null = null;

export async function loadSimulation(): Promise<void>
{
  const exports = await initialize();
  memory = exports.memory;
}

export class WasmView
{
  private view = new Float64Array(0);
  private buffer: ArrayBufferLike | null = null;
  private pointer = -1;

  read(pointer: number, length: number): Float64Array
  {
    if (memory === null)
    {
      throw new Error("the simulation module is not loaded");
    }
    if (memory.buffer !== this.buffer || pointer !== this.pointer
      || length !== this.view.length)
    {
      this.buffer = memory.buffer;
      this.pointer = pointer;
      this.view = new Float64Array(memory.buffer, pointer, length);
    }
    return this.view;
  }
}
