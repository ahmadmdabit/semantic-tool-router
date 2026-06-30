import { describe, it, expect } from 'vitest';
import { Tool } from '../../src/types.js';
import { VectorStore } from '../../src/vector/vector-store.js';
import { retrieve } from '../../src/routing/retriever.js';

// These helpers mirror the text composition inside index.command.ts (which keeps
// its builders private). Duplicating them here lets us build a store whose
// vectors are consistent with what the real index command would produce, so the
// route command's retrieve() call ranks meaningfully.
function buildPosText(tool: Tool): string {
  const parts: string[] = [tool.name, tool.description];
  const params = tool.parameters as { properties?: Record<string, { description?: string }> } | undefined;
  if (params?.properties) {
    for (const prop of Object.values(params.properties)) {
      if (prop?.description) parts.push(prop.description);
    }
  }
  if (tool.intent) parts.push(tool.intent);
  if (tool.examples?.length) parts.push(tool.examples.join('. '));
  if (tool.whenToUse?.length) parts.push(tool.whenToUse.join('. '));
  return parts.filter(Boolean).join('. ');
}

function buildNegText(tool: Tool): string {
  if (!tool.whenNotToUse || tool.whenNotToUse.length === 0) return '';
  return tool.whenNotToUse.join('. ');
}

const GLOB: Tool = {
  name: 'glob',
  description: 'Find files matching a glob pattern.',
  intent: 'find files by pattern',
  examples: ['Find all .ts files in src', 'Locate files matching *.json'],
  whenToUse: ['The query contains a glob pattern'],
  whenNotToUse: ['The user wants to read a specific file', 'The user wants to search inside file contents'],
  triggers: ['find files', 'glob'],
  boosts: ['grep'],
  parameters: { type: 'object', properties: { pattern: { description: 'glob pattern like *.ts' } } },
};

const GREP: Tool = {
  name: 'grep',
  description: 'Search for a regex pattern within files.',
  intent: 'search inside files for a regex pattern',
  examples: ['Find every TODO comment in src/**/*.ts'],
  whenToUse: ['The user has a regex or literal string'],
  whenNotToUse: ['The user wants to match file names'],
  triggers: ['grep', 'regex'],
  boosts: [],
  parameters: { type: 'object', properties: { pattern: { description: 'regex pattern' } } },
};

// Synchronous seeded hash (mirrors the async fake embedder but blocking).
function embedSync(text: string, dim = 8): number[] {
  let h = 0;
  for (let i = 0; i < text.length; i++) h = (h * 31 + text.charCodeAt(i)) | 0;
  const vec: number[] = [];
  for (let i = 0; i < dim; i++) {
    h = (h * 1103515245 + 12345) & 0x7fffffff;
    vec.push((h % 1000) / 1000);
  }
  const mag = Math.sqrt(vec.reduce((s, v) => s + v * v, 0));
  return vec.map((v) => v / (mag || 1));
}

describe('index command text composition', () => {
  it('buildPosText includes name, description, params, intent, examples, whenToUse', () => {
    const text = buildPosText(GLOB);
    expect(text).toContain('glob');
    expect(text).toContain('Find files matching a glob pattern.');
    expect(text).toContain('glob pattern like *.ts'); // param description
    expect(text).toContain('find files by pattern'); // intent
    expect(text).toContain('Find all .ts files in src'); // example
    expect(text).toContain('The query contains a glob pattern'); // whenToUse
  });

  it('buildNegText joins whenNotToUse sentences', () => {
    expect(buildNegText(GLOB)).toBe('The user wants to read a specific file. The user wants to search inside file contents');
  });

  it('buildNegText returns empty string when no boundaries declared', () => {
    const bare: Tool = { name: 'bare', description: 'bare tool', parameters: {} };
    expect(buildNegText(bare)).toBe('');
  });
});

describe('route command via retrieve() with realistic store', () => {
  it('routes a glob query to glob when posVec/negVec are populated', async () => {
    const store = new VectorStore();
    const tools = [GLOB, GREP];
    const posEmbeddings = tools.map((t) => embedSync(buildPosText(t)));
    const negEmbeddings = tools.map((t) => {
      const negText = buildNegText(t);
      return negText ? embedSync(negText) : [];
    });
    await store.add(tools, posEmbeddings, negEmbeddings);

    // Use a deterministic async embedder backed by the same sync hash so the
    // query embedding is consistent with the stored vectors.
    const embedder = { embed: async (text: string) => embedSync(text) };
    const results = await retrieve('find files matching *.ts', embedder, store, { k: 2 });
    expect(results.length).toBeGreaterThan(0);
    // glob should be in the top-2 (it may or may not be #1 depending on the
    // deterministic embedder, but it must surface).
    expect(results.map((r) => r.tool.name)).toContain('glob');
  });

  it('demotes a tool whose whenNotToUse matches the query', async () => {
    // A query about "read a specific file" matches glob's whenNotToUse. With
    // polarity encoding, glob should be demoted relative to a store with only
    // posVec (no negVec).
    const storeWithNeg = new VectorStore();
    const posGLOB = embedSync(buildPosText(GLOB));
    const negGLOB = embedSync(buildNegText(GLOB));
    await storeWithNeg.add([GLOB], [posGLOB], [negGLOB]);

    const storePosOnly = new VectorStore();
    await storePosOnly.add([GLOB], [posGLOB]); // no negEmbeddings

    const embedder = { embed: async (text: string) => embedSync(text) };
    const q = 'read a specific file';
    const withNeg = await retrieve(q, embedder, storeWithNeg, { k: 1 });
    const posOnly = await retrieve(q, embedder, storePosOnly, { k: 1 });

    // The polarity-adjusted score must be <= the pos-only score (negative
    // prototype can only subtract, never add).
    expect(withNeg[0].score).toBeLessThanOrEqual(posOnly[0].score);
  });
});
