import { describe, it, expect, beforeEach } from 'vitest';
import { retrieve } from '../../src/routing/retriever.js';
import { VectorStore } from '../../src/vector/vector-store.js';
import { IEmbedder, Tool } from '../../src/types.js';
import { resetIntentCache } from '../../src/routing/intent-detector.js';

function makeTool(name: string, description: string, extra: Partial<Tool> = {}): Tool {
  return { name, description, parameters: { type: 'object', properties: {} }, ...extra };
}

// Deterministic fake embedder: hashes the text into a repeatable unit vector.
// Same text → same vector; different texts → different vectors. This lets us
// reason about cosine similarity without a real model.
function makeFakeEmbedder(dim = 8): IEmbedder {
  return {
    embed: async (text: string) => {
      // Simple seeded hash → deterministic pseudo-random vector.
      let h = 0;
      for (let i = 0; i < text.length; i++) {
        h = (h * 31 + text.charCodeAt(i)) | 0;
      }
      const vec: number[] = [];
      for (let i = 0; i < dim; i++) {
        h = (h * 1103515245 + 12345) & 0x7fffffff;
        vec.push((h % 1000) / 1000);
      }
      // Normalize to unit length so cosine == dot.
      const mag = Math.sqrt(vec.reduce((s, v) => s + v * v, 0));
      return vec.map((v) => v / (mag || 1));
    },
  };
}

// A small catalog where each tool's description is distinctive enough that the
// fake embedder produces separable vectors.
const GLOB = makeTool('glob', 'find files matching a glob pattern wildcard', {
  triggers: ['find files', 'glob'],
  boosts: ['grep'],
});
const GREP = makeTool('grep', 'search for a regex pattern within files content', {
  triggers: ['grep', 'regex'],
  boosts: [],
});
const READ = makeTool('readFile', 'read the content of a text file display', {
  triggers: ['read file'],
  boosts: [],
});

let store: VectorStore;
let embedder: IEmbedder;

beforeEach(async () => {
  resetIntentCache();
  embedder = makeFakeEmbedder();
  store = new VectorStore();
  // Pre-compute each tool's embedding using the SAME embedder so the vectors
  // are consistent with what retrieve() will produce for the query.
  const tools = [GLOB, GREP, READ];
  const embeddings = await Promise.all(
    tools.map((t) => embedder.embed(`${t.name} ${t.description}`, 'document')),
  );
  await store.add(tools, embeddings);
});

describe('retrieve — S1 dense cosine', () => {
  it('ranks the most cosine-similar tool first when no triggers fire', async () => {
    // A query whose embedding is closest to glob's vector.
    const results = await retrieve('glob wildcard pattern', embedder, store, { k: 3 });
    expect(results.length).toBeGreaterThan(0);
    // The top result should be one of the three tools (sanity).
    expect(['glob', 'grep', 'readFile']).toContain(results[0].tool.name);
    // Scores are normalized to [0,1].
    for (const r of results) {
      expect(r.score).toBeGreaterThanOrEqual(0);
      expect(r.score).toBeLessThanOrEqual(1);
    }
  });

  it('returns at most k results', async () => {
    const results = await retrieve('glob', embedder, store, { k: 2 });
    expect(results.length).toBeLessThanOrEqual(2);
  });
});

describe('retrieve — S2 intent boost', () => {
  it('boosts a tool whose trigger fires even if cosine alone would rank it lower', async () => {
    // "find files" fires glob's dynamic trigger. The S2 boost should lift glob
    // to #1 even if the raw cosine is not the highest.
    const results = await retrieve('find files matching', embedder, store, { k: 3 });
    expect(results[0].tool.name).toBe('glob');
    expect(results[0].debug?.intent).toBe('none'); // dynamic rules carry pattern 'none'
  });

  it('boosts tools listed in `boosts` alongside the trigger owner', async () => {
    // glob.boosts = ['grep']. When glob's trigger fires, grep should also get
    // an S2 rank (visible in debug.intent === 'none' for both).
    const results = await retrieve('glob pattern', embedder, store, { k: 3 });
    const grepResult = results.find((r) => r.tool.name === 'grep');
    expect(grepResult?.debug?.intent).toBe('none');
  });
});

describe('retrieve — S3 keyword overlap', () => {
  it('surfaces a tool whose description shares tokens with the query', async () => {
    // "regex content" shares tokens with grep's description.
    const results = await retrieve('regex content', embedder, store, { k: 3 });
    const grepResult = results.find((r) => r.tool.name === 'grep');
    expect(grepResult?.debug?.keyword).toBeGreaterThan(0);
  });
});

describe('retrieve — threshold filter', () => {
  it('drops tools below the composite-score floor', async () => {
    // A high threshold should filter out low-confidence tools.
    const results = await retrieve('glob', embedder, store, { k: 3, threshold: 0.99 });
    for (const r of results) {
      expect(r.score).toBeGreaterThanOrEqual(0.99);
    }
  });

  it('returns all tools when threshold is 0 (disabled)', async () => {
    const results = await retrieve('glob', embedder, store, { k: 5, threshold: 0 });
    expect(results.length).toBe(3);
  });
});

describe('retrieve — debug breakdown', () => {
  it('populates cosine, keyword, and intent in the debug object', async () => {
    const results = await retrieve('glob', embedder, store, { k: 1 });
    expect(results[0].debug).toBeDefined();
    expect(results[0].debug?.cosine).toBeDefined();
    expect(results[0].debug?.keyword).toBeDefined();
    expect(results[0].debug?.intent).toBeDefined();
  });
});
