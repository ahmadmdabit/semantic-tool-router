import { describe, it, expect } from 'vitest';
import { dot } from '../../src/math/dot.js';
import { norm, normalize } from '../../src/math/norm.js';
import { cosineSimilarity } from '../../src/math/cosine-similarity.js';

describe('dot', () => {
  it('computes the inner product of two equal-length vectors', () => {
    expect(dot([1, 2, 3], [4, 5, 6])).toBe(32); // 4+10+18
  });

  it('is commutative', () => {
    const a = [1.5, -2, 0];
    const b = [3, 4, 5];
    expect(dot(a, b)).toBe(dot(b, a));
  });

  it('returns 0 for orthogonal vectors', () => {
    expect(dot([1, 0, 0], [0, 1, 0])).toBe(0);
  });

  it('throws when lengths differ', () => {
    expect(() => dot([1, 2], [1, 2, 3])).toThrow(/equal length/);
  });

  it('throws on empty vectors', () => {
    expect(() => dot([], [])).toThrow(/empty/);
  });
});

describe('norm', () => {
  it('returns 0 for the zero vector', () => {
    expect(norm([0, 0, 0])).toBe(0);
  });

  it('computes the L2 euclidean length', () => {
    expect(norm([3, 4])).toBe(5);
    expect(norm([1, 1, 1, 1])).toBe(2);
  });

  it('returns 0 for an empty vector', () => {
    expect(norm([])).toBe(0);
  });

  it('works on Float32Array input', () => {
    expect(norm(new Float32Array([3, 4]))).toBe(5);
  });
});

describe('normalize', () => {
  it('returns a unit-length vector', () => {
    const out = normalize([3, 4]);
    expect(out[0]).toBeCloseTo(0.6, 5);
    expect(out[1]).toBeCloseTo(0.8, 5);
    expect(norm(out)).toBeCloseTo(1, 5);
  });

  it('returns a zero-magnitude vector unchanged (avoids NaN)', () => {
    const out = normalize([0, 0, 0]);
    expect(out).toEqual(Float32Array.from([0, 0, 0]));
  });

  it('always returns a Float32Array', () => {
    expect(normalize([1, 2, 3])).toBeInstanceOf(Float32Array);
    expect(normalize(new Float32Array([1, 2, 3]))).toBeInstanceOf(Float32Array);
  });

  it('does NOT mutate the input array (regression: copy semantics)', () => {
    const input = [3, 4, 0];
    const snapshot = [...input];
    normalize(input);
    expect(input).toEqual(snapshot);
  });

  it('does NOT mutate Float32Array input (defensive copy)', () => {
    const input = new Float32Array([3, 4, 0]);
    const snapshot = Buffer.from(input);
    normalize(input);
    expect(Buffer.from(input)).toEqual(snapshot);
  });
});

describe('cosineSimilarity', () => {
  it('is 1.0 for identical direction', () => {
    expect(cosineSimilarity([1, 2, 3], [2, 4, 6])).toBeCloseTo(1, 5);
  });

  it('is -1.0 for opposite direction', () => {
    expect(cosineSimilarity([1, 0], [-1, 0])).toBeCloseTo(-1, 5);
  });

  it('is 0.0 for orthogonal vectors', () => {
    expect(cosineSimilarity([1, 0, 0], [0, 1, 0])).toBe(0);
  });

  it('returns 0 (not NaN) when the query vector is zero', () => {
    expect(cosineSimilarity([0, 0, 0], [1, 2, 3])).toBe(0);
  });

  it('returns 0 (not NaN) when the document vector is zero', () => {
    expect(cosineSimilarity([1, 2, 3], [0, 0, 0])).toBe(0);
  });

  it('returns 0 when both vectors are zero', () => {
    expect(cosineSimilarity([0, 0], [0, 0])).toBe(0);
  });
});
