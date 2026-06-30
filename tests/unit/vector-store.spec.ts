import { describe, it, expect, beforeEach, vi } from 'vitest';
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
const EMBEDa = [1, 0, 0];
const EMBEDb = [0, 1, 0];

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
    await store.add([ToolA, ToolB], [EMBEDa, EMBEDb]);
    expect(store.size()).toBe(2);
  });

  it('throws when tools and embeddings have different lengths', async () => {
    await expect(store.add([ToolA, ToolB], [EMBEDa])).rejects.toThrow(/length mismatch/);
  });

  it('upserts by tool name on second add', async () => {
    await store.add([ToolA], [EMBEDa]);
    await store.add([ToolA], [EMBEDb]); // overwrite
    expect(store.size()).toBe(1);
    const tools = store.tools();
    expect(tools[0].name).toBe('alpha');
  });
});

describe('search', () => {
  beforeEach(async () => {
    await store.add([ToolA, ToolB], [EMBEDa, EMBEDb]);
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

  it('subtracts the negative-prototype cosine when negVec is present', async () => {
    // Two tools with identical positive vectors but different negative vectors.
    // A query aligned with BOTH the positive and ToolA's negative vector should
    // score ToolA lower than ToolB (which has no negative signal).
    const pos = [1, 0, 0];
    const negA = [1, 0, 0]; // query-aligned negative → penalises ToolA
    const zeroNeg = new Float32Array(0); // no negative → no penalty
    const ToolA = makeTool('alpha', 'first tool');
    const ToolB = makeTool('beta', 'second tool');
    await store.add([ToolA, ToolB], [pos, pos], [negA, zeroNeg]);

    const results = await store.search([1, 0, 0], 5);
    // ToolB (no neg penalty) must outrank ToolA (neg penalty applied).
    expect(results[0].tool.name).toBe('beta');
    expect(results[1].tool.name).toBe('alpha');
  });

  it('treats a missing negVec as zero (no penalty)', async () => {
    // Backward-compat: add without negEmbeddings → negVec is zero-length.
    const ToolA = makeTool('alpha', 'first tool');
    await store.add([ToolA], [[1, 0, 0]]);
    const results = await store.search([1, 0, 0], 5);
    expect(results[0].score).toBeCloseTo(1, 5); // pure positive cosine, no subtraction
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

describe('search + searchTools', () => {
  it('treats an all-zero negVec as no penalty (isZeroVec branch)', async () => {
    // A negVec of [0,0,0] is non-zero-length but all-zero — isZeroVec must
    // return true so the polarity subtraction is skipped (same as zero-length).
    const ToolA = makeTool('alpha', 'first tool');
    await store.add([ToolA], [[1, 0, 0]], [[0, 0, 0]]);
    const results = await store.search([1, 0, 0], 5);
    // cosine(q=[1,0,0], pos=[1,0,0]) = 1.0, no penalty because isZeroVec=true.
    expect(results[0].score).toBeCloseTo(1, 5);
  });

  it('searchTools() returns tools without scores (back-compat shim)', async () => {
    await store.add([ToolA, ToolB], [EMBEDa, EMBEDb]);
    const tools = await store.searchTools([1, 0, 0], 5);
    expect(tools).toHaveLength(2);
    expect(tools[0].name).toBe('alpha');
    // searchTools drops scores — the returned objects are plain Tool records.
    expect((tools[0] as any).score).toBeUndefined();
  });
});

describe('tools + toolTexts', () => {
  it('exposes the catalog via tools()', async () => {
    await store.add([ToolA, ToolB], [EMBEDa, EMBEDb]);
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
    await store.add([ToolA], [EMBEDa]);
    const t1 = store.tools();
    t1.pop(); // mutate the returned copy
    expect(store.size()).toBe(1); // internal state unaffected
  });
});

describe('metadata + size + error paths', () => {
  it('stores and retrieves metadata via setMetadata/getMetadata', () => {
    expect(store.getMetadata()).toBeNull();
    store.setMetadata({ model: 'nomic-embed-text:latest', dimensions: 768, indexedAt: '2026-06-30T00:00:00.000Z' });
    expect(store.getMetadata()).toEqual({ model: 'nomic-embed-text:latest', dimensions: 768, indexedAt: '2026-06-30T00:00:00.000Z' });
  });

  it('reports size as the number of stored vectors', async () => {
    expect(store.size()).toBe(0);
    await store.add([ToolA], [EMBEDa]);
    expect(store.size()).toBe(1);
  });

  it('throws when negEmbeddings length does not match tools', async () => {
    await expect(store.add([ToolA, ToolB], [EMBEDa, EMBEDb], [[1, 0, 0]])).rejects.toThrow(/negEmbeddings length mismatch/);
  });
});

describe('save + load round-trip (JSONL)', () => {
  const tmpPath = `tmp/test-store-${process.pid}.jsonl`;

  it('persists posVec and negVec and reloads them', async () => {
    const store1 = new VectorStore(tmpPath);
    const ToolA = makeTool('alpha', 'first tool');
    await store1.add([ToolA], [[1, 0, 0]], [[0, 1, 0]]);
    store1.setMetadata({ model: 'test-model', dimensions: 3, indexedAt: 'now' });
    await store1.save();

    const store2 = new VectorStore(tmpPath);
    await store2.load();
    expect(store2.size()).toBe(1);
    expect(store2.getMetadata()?.model).toBe('test-model');

    // The negative prototype must survive the round-trip: a query aligned
    // with the negative vector should be penalised on reload.
    const results = await store2.search([0, 1, 0], 5);
    expect(results[0].tool.name).toBe('alpha');
    // cosine(q=[0,1,0], pos=[1,0,0]) - 0.3*cosine(q=[0,1,0], neg=[0,1,0])
    // = 0 - 0.3*1 = -0.3
    expect(results[0].score).toBeCloseTo(-0.3, 5);
  });

  it('loads a legacy single-vector index (embedding field, no negVec)', async () => {
    // Hand-write an old-format JSONL record.
    const fs = await import('node:fs');
    const legacy = JSON.stringify({
      type: 'vector',
      data: { tool: { name: 'legacy', description: 'old format', parameters: {} }, embedding: [1, 0, 0] },
    });
    fs.writeFileSync(tmpPath, legacy + '\n');

    const store = new VectorStore(tmpPath);
    await store.load();
    expect(store.size()).toBe(1);
    // No negVec → no penalty → pure positive cosine.
    const results = await store.search([1, 0, 0], 5);
    expect(results[0].score).toBeCloseTo(1, 5);
  });

  it('skips a vector record missing both posVec and embedding', async () => {
    const fs = await import('node:fs');
    const bad = JSON.stringify({ type: 'vector', data: { tool: { name: 'broken', description: 'x', parameters: {} } } });
    fs.writeFileSync(tmpPath, bad + '\n');

    const store = new VectorStore(tmpPath);
    await store.load();
    expect(store.size()).toBe(0);
  });

  it('skips malformed JSON lines without throwing', async () => {
    const fs = await import('node:fs');
    fs.writeFileSync(tmpPath, '{"type":"vector","data":{}\n'); // truncated JSON

    const store = new VectorStore(tmpPath);
    await store.load();
    expect(store.size()).toBe(0);
  });

  it('save omits negVec when it carries no signal', async () => {
    const fs = await import('node:fs');
    const store = new VectorStore(tmpPath);
    const ToolA = makeTool('alpha', 'first tool');
    // No negEmbeddings supplied → negVec is zero-length → not persisted.
    await store.add([ToolA], [[1, 0, 0]]);
    await store.save();

    const raw = fs.readFileSync(tmpPath, 'utf-8');
    const record = JSON.parse(raw.trim());
    expect(record.data.posVec).toBeDefined();
    expect(record.data.negVec).toBeUndefined();
  });

  it('load() with a non-existent file yields an empty store', async () => {
    const fs = await import('node:fs');
    const missingPath = `tmp/missing-store-${process.pid}.jsonl`;
    // Ensure the file does not exist.
    if (fs.existsSync(missingPath)) fs.unlinkSync(missingPath);

    const store = new VectorStore(missingPath);
    await store.load();
    expect(store.size()).toBe(0);
    expect(store.getMetadata()).toBeNull();
    // An empty store must still be searchable (returns []).
    const results = await store.search([1, 0, 0], 5);
    expect(results).toEqual([]);
  });

  it('skips a vector record that has neither posVec nor embedding', async () => {
    // A vector record missing both fields triggers the warn-and-skip branch.
    const fs = await import('node:fs');
    const record = JSON.stringify({
      type: 'vector',
      data: { tool: { name: 'orphan', description: 'x', parameters: {} } },
    });
    fs.writeFileSync(tmpPath, record + '\n');

    const warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const store = new VectorStore(tmpPath);
    await store.load();
    expect(store.size()).toBe(0);
    expect(warnSpy).toHaveBeenCalledWith(
      expect.stringContaining('missing posVec/embedding'),
    );
    warnSpy.mockRestore();
  });
});
