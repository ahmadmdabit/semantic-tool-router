// Benchmark: routing latency across catalog sizes.
//
// Measures the hot path — retrieve() end-to-end — with a deterministic mock
// embedder (no Ollama dependency). Each bench variant builds an in-memory
// VectorStore with N synthetic tools, then times `retrieve()` against a
// representative query. Use this to catch regressions when changing the
// retrieval pipeline (new signals, polarity encoding, RRF fusion, etc.).

import { bench, describe } from 'vitest';
import { retrieve } from '../../src/routing/retriever.js';
import { VectorStore } from '../../src/vector/vector-store.js';
import { Tool } from '../../src/types.js';

// --- Deterministic mock embedder --------------------------------------------
// Produces a unit vector from a simple hash of the text. Same text always
// yields the same vector, so benchmarks are stable across runs.
function mockEmbed(text: string, dims = 64): number[] {
  let h = 0;
  for (let i = 0; i < text.length; i++) h = (h * 31 + text.charCodeAt(i)) | 0;
  const vec: number[] = [];
  for (let i = 0; i < dims; i++) {
    h = (h * 1103515245 + 12345) & 0x7fffffff;
    vec.push((h % 1000) / 1000);
  }
  const mag = Math.sqrt(vec.reduce((s, v) => s + v * v, 0));
  return vec.map((v) => v / (mag || 1));
}

// Mock embedder matching the IEmbedder interface.
const mockEmbedder = {
  embed: async (text: string): Promise<number[]> => mockEmbed(text),
};

// --- Synthetic catalog generator --------------------------------------------
// Creates N tools with realistic-ish fields so the intent detector and
// keyword-overlap scorer have signal to work with.
function makeTool(index: number): Tool {
  const names = ['glob', 'read', 'write', 'edit', 'move', 'rename', 'list', 'scan', 'grep', 'fetch', 'run', 'skill'];
  const name = names[index % names.length] + (index >= names.length ? `-${index}` : '');
  return {
    name,
    description: `Perform ${name} operation on files or data.`,
    intent: `${name} intent`,
    examples: [`Use ${name} to process data`, `Run ${name} on the target`],
    whenToUse: [`When you need to ${name} something`],
    whenNotToUse: [`Do not use ${name} for unrelated tasks`],
    triggers: [name, `${name} data`],
    boosts: [],
    parameters: {
      type: 'object',
      properties: {
        target: { description: `Target path or pattern for ${name}` },
      },
    },
  };
}

async function buildStore(n: number): Promise<VectorStore> {
  const tools = Array.from({ length: n }, (_, i) => makeTool(i));
  const store = new VectorStore();
  const posEmbeddings = tools.map((t) => mockEmbed(`${t.name} ${t.description} ${t.intent} ${t.examples.join(' ')} ${t.whenToUse.join(' ')}`));
  const negEmbeddings = tools.map((t) => mockEmbed(t.whenNotToUse.join(' ')));
  await store.add(tools, posEmbeddings, negEmbeddings);
  return store;
}

//  ---------------------------------------------------------------------------
// Edge-case helpers
//  ---------------------------------------------------------------------------

// All tools share IDENTICAL positive embeddings — worst-case for sort// The dense pass yields a perfect tie, so RRF + slice must resolve ordering by
// insertion order / keyword signal. Reveals O(N log N) sort cost at scale.
async function buildHomogeneousStore(n: number): Promise<VectorStore> {
  const tools = Array.from({ length: n }, (_, i) => makeTool(i));
  const sharedPos = mockEmbed('identical positive text for all tools');
  const store = new VectorStore();
  const posEmbeddings = tools.map(() => [...sharedPos]);
  const negEmbeddings = tools.map(() => []);
  await store.add(tools, posEmbeddings, negEmbeddings);
  return store;
}

// All tools share IDENTICAL whenNotToUse text — any query matching it will trigger
// the polarity subtraction path (negVec present, cosine penalty applied).
async function buildStoreWithBoundaries(n: number): Promise<VectorStore> {
  const tools = Array.from({ length: n }, (_, i) => makeTool(i));
  const store = new VectorStore();
  const posEmbeddings = tools.map((t) => mockEmbed(`${t.name} ${t.description} ${t.intent} ${t.examples.join(' ')} ${t.whenToUse.join(' ')}`));
  const sharedNeg = mockEmbed('do not use this tool for unrelated tasks or general conversation');
  const negEmbeddings = tools.map(() => [...sharedNeg]);
  await store.add(tools, posEmbeddings, negEmbeddings);
  return store;
}

// --- Benchmarks -------------------------------------------------------------
describe('routing latency', () => {
  bench('retrieve() with 14-tool catalog (current)', async () => {
    const store = await buildStore(14);
    await retrieve('find all TypeScript files in src/', mockEmbedder, store, { k: 5 });
  });

  bench('retrieve() with 50-tool catalog (growth target)', async () => {
    const store = await buildStore(50);
    await retrieve('find all TypeScript files in src/', mockEmbedder, store, { k: 5 });
  });

  bench('retrieve() with 200-tool catalog (stress)', async () => {
    const store = await buildStore(200);
    await retrieve('find all TypeScript files in src/', mockEmbedder, store, { k: 5 });
  });

  // --- Edge cases (see DEEP analysis) ---------------------------------------

  bench('empty catalog — early return', async () => {
    const emptyStore = new VectorStore();
    await retrieve('anything', mockEmbedder, emptyStore, { k: 5 });
  });

  bench('k=1 from 14-tool catalog', async () => {
    const store = await buildStore(14);
    await retrieve('find files', mockEmbedder, store, { k: 1 });
  });

  bench('k=50 from 200-tool catalog', async () => {
    const store = await buildStore(200);
    await retrieve('find files', mockEmbedder, store, { k: 50 });
  });

  bench('high threshold (0.99) — heavy filtering', async () => {
    const store = await buildStore(50);
    await retrieve('tell me a joke', mockEmbedder, store, { k: 50, threshold: 0.99 });
  });

  bench('intent-heavy query — S2 path', async () => {
    const store = await buildStore(50);
    await retrieve('find files matching *.ts pattern', mockEmbedder, store, { k: 5 });
  });

  bench('homogeneous catalog — worst-case sort', async () => {
    const store = await buildHomogeneousStore(100);
    await retrieve('anything', mockEmbedder, store, { k: 5 });
  });

  bench('query matching whenNotToUse — polarity penalty', async () => {
    const store = await buildStoreWithBoundaries(50);
    await retrieve('do not use this for unrelated tasks', mockEmbedder, store, { k: 5 });
  });
});
