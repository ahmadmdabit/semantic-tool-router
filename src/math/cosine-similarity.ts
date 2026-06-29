import { dot } from './dot.js';
import { norm } from './norm.js';

/**
 * Calculate cosine similarity between two vectors.
 * Uses adapted mathjs logic for dot product and L2 norm.
 */
export function cosineSimilarity(a: number[], b: number[]): number {
  const magA = norm(a);
  const magB = norm(b);

  if (magA === 0 || magB === 0) return 0;

  return dot(a, b) / (magA * magB);
}
