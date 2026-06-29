/**
 * Calculate the L2 (Euclidean) norm of a vector (supports number[] and Float32Array).
 * Logic adapted from mathjs norm.js (_vectorNorm with p=2, simplified)
 */
export function norm(vec: Float32Array | number[]): number {
  if (vec.length === 0) return 0;

  let sum = 0;
  const len = vec.length;
  for (let i = 0; i < len; i++) {
    sum += vec[i] * vec[i];
  }
  return Math.sqrt(sum);
}

/**
 * Normalize a vector to unit length and return as a Float32Array.
 */
export function normalize(vec: Float32Array | number[]): Float32Array {
  // Always copy: callers (VectorStore) may reuse their input arrays, and
  // in-place mutation would corrupt test fixtures and indexed embeddings.
  const arr = new Float32Array(vec);
  const mag = norm(arr);
  if (mag > 0) {
    const len = arr.length;
    for (let i = 0; i < len; i++) {
      arr[i] /= mag;
    }
  }
  return arr;
}