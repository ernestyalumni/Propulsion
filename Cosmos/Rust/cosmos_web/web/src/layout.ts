// Offsets into the frame buffers written by cosmos_web (Rust). These mirror
// `rigid_body_lab::frame` and `orbit_lab::frame` and must change with them.

export const RigidFrame = {
  time: 0,
  quaternion: 1,
  angularVelocity: 5,
  angularMomentumInertial: 8,
  kineticEnergy: 11,
  energyError: 12,
  momentumError: 13,
  attitudeError: 14,
  quaternionNormError: 15,
  stepSize: 16,
  accepted: 17,
  rejected: 18,
  evaluations: 19,
  failed: 20,
} as const;

export const OrbitFrame = {
  time: 0,
  position: 1,
  velocity: 3,
  accepted: 5,
  rejected: 6,
  evaluations: 7,
  stepSize: 8,
  failed: 9,
  closingError: 10,
  periods: 11,
} as const;

/** `orbit_lab::TRAIL_CAPACITY` and `MARKER_CAPACITY`. */
export const TRAIL_CAPACITY = 8192;
export const MARKER_CAPACITY = 8192;
