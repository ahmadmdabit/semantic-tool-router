import { describe, it, expect, beforeEach } from 'vitest';
import { VectorStore } from '../../src/vector/vector-store.js';
import { Tool } from '../../src/types.js';

function makeTool(name: string, description: string): Tool {
  return { name, description, parameters: { type: 'object', properties: {} } };
}

// Two tools whose embeddings point in clearly different directions in a
// 3-dimensional space. We can reason about cosine: a query aligned with a ToolA
// vector should score ToolA higher.
const ToolA = makeTool('alpha', 'first tool');
const ToolB = makeTool('beta', 'second tool');

// Unit vectors along orthogonal axes.
const EMBED_A = [1, 0, 0];
const EMBED_B = [0, 1, 0];

let store: VectorStore;

beforeEach(() => {
  // In-memory store: constructor does not touch disk, so a default path is safe.
  store = new VectorStore();
});

describe('add + size', () => {
  it('starts empty', () => {
    expect(store.size()).toBe(0);
  });

  it('reports size after adding tools', async () => {
    await store.add([ToolA, ToolB], [EMBED_A, EMBED_B]);
    expect(store.size()).toBe(2);
  });

  it('throws when tools and embeddings have different lengths', async () => {
    await expect(store.add([ToolA, ToolB], [EMBED_A])).rejects.toThrow(/length mismatch/);
  });

  it('upserts by tool name on second add', async () => {
    await store.add([ToolA], [EMBED_A]);
    await store.add([ToolA], [EMBED_B]); // overwrite
    expect(store.size()).toBe(1);
    const tools = store.tools();
    expect(tools[0].name).toBe('alpha');
  });
});

describe('search', () => {
  beforeEach(async () => {
    await store.add([ToolA, ToolB], [EMBED_A, EMBED_B]);
  });

  it('returns the closest tool first for an aligned query', async () => {
    const results = await store.search([1, 0, 0], 5);
    expect(results[0].tool.name).toBe('alpha');
    expect(results[0].score).toBeCloseTo(1, 5);
  });

  it('ranks an orthogonal query as closer to the matching axis', async () => {
    const results = await store.search([0, 1, 0], 5);
    expect(results[0].tool.name).toBe('beta');
  });

  it('returns at most k results', async () => {
    const results = await store.search([1, 0, 0], 1);
    expect(results.length).toBe(1);
  });

  it('returns an empty list when the store is empty', async () => {
    const empty = new VectorStore();
    const results = await empty.search([1, 0, 0], 5);
    expect(results).toEqual([]);
  });

  it('applies a boost multiplier to tools in boostTools', async () => {
    // Query aligned with beta. Without boost, beta wins (cosine 1.0 vs 0.0).
    // With boost on alpha, alpha's 0.0 * 1.15 = 0.0 — still loses. Use a tie:
    // a diagonal query [1,1,0] gives both cosine ~0.707, then boost flips it.
    const noBoost = await store.search([1, 1, 0], 5);
    const topNoBoost = noBoost[0];

    const withBoost = await store.search([1, 1, 0], 5, {
      boostTools: new Set(['alpha']),
      boost: 1.5,
    });
    const topWithBoost = withBoost[0];

    expect(topNoBoost.tool.name).toBe('alpha'); // tie broken by insertion order-ish
    expect(topWithBoost.tool.name).toBe('alpha');
    // The boosted alpha score should be strictly higher than the unboosted one.
    const alphaWithBoost = withBoost.find((s) => s.tool.name === 'alpha')!;
    const alphaNoBoost = noBoost.find((s) => s.tool.name === 'alpha')!;
    expect(alphaWithBoost.score).toBeGreaterThan(alphaNoBoost.score);
  });
});

describe('tools + toolTexts', () => {
  it('exposes the catalog via tools()', async () => {
    await store.add([ToolA, ToolB], [EMBED_A, EMBED_B]);
    const names = store.tools().map((t) => t.name).sort();
    expect(names).toEqual(['alpha', 'beta']);
  });

  it('composes canonical text via toolTexts()', async () => {
    const rich = makeTool('gamma', 'desc gamma');
    rich.intent = 'my intent';
    rich.examples = ['ex one', 'ex two'];
    rich.whenToUse = ['use when X'];
    rich.whenNotToUse = ['not when Y'];
    await store.add([rich], [[0, 0, 1]]);

    const texts = store.toolTexts();
    const text = texts.get('gamma')!;
    expect(text).toContain('gamma');
    expect(text).toContain('my intent');
    expect(text).toContain('ex one');
    expect(text).toContain('use when X');
    expect(text).toContain('NOT: not when Y');
  });

  it('returns a fresh array from tools() (no internal mutation leak)', async () => {
    await store.add([ToolA], [EMBED_A]);
    const t1 = store.tools();
    t1.pop(); // mutate the returned copy
    expect(store.size()).toBe(1); // internal state unaffected
  });
});
