/**
 * Calculate the dot product of two vectors.
 * Logic adapted from mathjs dot.js (dense path, simplified)
 */
export function dot(a: number[], b: number[]): number {
  if (a.length !== b.length) {
    throw new Error(`Vectors must have equal length (${a.length} != ${b.length})`);
  }
  if (a.length === 0) {
    throw new Error('Cannot calculate the dot product of empty vectors');
  }

  let result = 0;
  for (let i = 0; i < a.length; i++) {
    result += a[i] * b[i];
  }
  return result;
}
