/**
 * One Euro Filter — adaptive low-pass filter for noisy signals.
 *
 * Used by MediaPipe for landmark smoothing. The key insight: slow movements
 * get heavy smoothing (removes jitter), fast movements get light smoothing
 * (preserves responsiveness).
 *
 * Reference: https://gery.casiez.net/1euro/
 */

interface LowPassFilter {
  value: number;
  initialized: boolean;
}

function lowPassUpdate(f: LowPassFilter, value: number, alpha: number): number {
  if (!f.initialized) {
    f.value = value;
    f.initialized = true;
    return value;
  }
  f.value = alpha * value + (1 - alpha) * f.value;
  return f.value;
}

function smoothingFactor(te: number, cutoff: number): number {
  const r = 2 * Math.PI * cutoff * te;
  return r / (r + 1);
}

interface OneEuroState {
  x: LowPassFilter;
  dx: LowPassFilter;
  lastTime: number;
}

function createState(): OneEuroState {
  return {
    x: { value: 0, initialized: false },
    dx: { value: 0, initialized: false },
    lastTime: -1,
  };
}

function oneEuroFilter(
  state: OneEuroState,
  value: number,
  timestamp: number,
  minCutoff: number,
  beta: number,
  dCutoff: number,
): number {
  const te = state.lastTime < 0 ? 1 / 30 : timestamp - state.lastTime;
  state.lastTime = timestamp;

  // Estimate derivative
  const alphaD = smoothingFactor(te, dCutoff);
  const dValue = state.x.initialized ? (value - state.x.value) / te : 0;
  const edValue = lowPassUpdate(state.dx, dValue, alphaD);

  // Adaptive cutoff based on speed
  const cutoff = minCutoff + beta * Math.abs(edValue);
  const alpha = smoothingFactor(te, cutoff);

  return lowPassUpdate(state.x, value, alpha);
}

/**
 * Landmark smoother using One Euro Filter.
 *
 * Applies independent filtering to each (x, y, z) of 21 landmarks.
 * Matches MediaPipe's VelocityFilter behavior.
 */
export interface LandmarkSmoother {
  /** Apply smoothing to landmarks. Returns new smoothed landmark array. */
  apply(landmarks: Array<{ x: number; y: number; z: number }>, timestamp?: number): Array<{ x: number; y: number; z: number }>;
  /** Reset filter state (e.g., when hand is lost and re-detected) */
  reset(): void;
}

export interface SmootherOptions {
  /** Minimum cutoff frequency (Hz). Lower = more smoothing. Default: 1.0 */
  minCutoff?: number;
  /** Speed coefficient. Higher = less lag during fast movement. Default: 0.0 */
  beta?: number;
  /** Derivative cutoff frequency (Hz). Default: 1.0 */
  dCutoff?: number;
}

export function createLandmarkSmoother(options: SmootherOptions = {}): LandmarkSmoother {
  const {
    // MediaPipe defaults for hand landmarks (from their config)
    minCutoff = 1.0,
    beta = 10.0,
    dCutoff = 1.0,
  } = options;

  // 21 landmarks × 3 channels (x, y, z)
  let states: OneEuroState[] = [];

  function ensureStates(count: number) {
    if (states.length !== count) {
      states = Array.from({ length: count }, () => createState());
    }
  }

  function apply(
    landmarks: Array<{ x: number; y: number; z: number }>,
    timestamp?: number,
  ): Array<{ x: number; y: number; z: number }> {
    const t = timestamp ?? performance.now() / 1000;
    const numChannels = landmarks.length * 3;
    ensureStates(numChannels);

    return landmarks.map((lm, i) => ({
      x: oneEuroFilter(states[i * 3]!, lm.x, t, minCutoff, beta, dCutoff),
      y: oneEuroFilter(states[i * 3 + 1]!, lm.y, t, minCutoff, beta, dCutoff),
      z: oneEuroFilter(states[i * 3 + 2]!, lm.z, t, minCutoff, beta, dCutoff),
    }));
  }

  function reset() {
    states = [];
  }

  return { apply, reset };
}
