/**
 * Calculate the dot product of two vectors (supports number[] and Float32Array).
 * Logic adapted from mathjs dot.js (dense path, simplified)
 */
export function dot(a: Float32Array | number[], b: Float32Array | number[]): number {
  if (a.length !== b.length) {
    throw new Error(`Vectors must have equal length (${a.length} != ${b.length})`);
  }
  if (a.length === 0) {
    throw new Error('Cannot calculate the dot product of empty vectors');
  }

  let result = 0;
  const len = a.length;
  for (let i = 0; i < len; i++) {
    result += a[i] * b[i];
  }
  return result;
}