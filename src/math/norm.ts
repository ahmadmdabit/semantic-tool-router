/**
 * Calculate the L2 (Euclidean) norm of a vector.
 * Logic adapted from mathjs norm.js (_vectorNorm with p=2, simplified)
 */
export function norm(vec: number[]): number {
  if (vec.length === 0) return 0;

  let sum = 0;
  for (let i = 0; i < vec.length; i++) {
    sum += vec[i] * vec[i];
  }
  return Math.sqrt(sum);
}
