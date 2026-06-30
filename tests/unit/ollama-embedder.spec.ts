import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { OllamaEmbedder } from '../../src/embeddings/ollama-embedder.js';

// A 10-dimensional deterministic vector we control.
const FAKE_VECTOR = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];

function mockFetch(response: { ok: boolean; statusText?: string; body?: any; failWith?: Error }) {
  global.fetch = vi.fn(async () => {
    if (response.failWith) throw response.failWith;
    return {
      ok: response.ok,
      statusText: response.statusText ?? '',
      json: async () => response.body ?? { embedding: FAKE_VECTOR },
    } as Response;
  });
}

describe('OllamaEmbedder', () => {
  const originalFetch = global.fetch;

  afterEach(() => {
    global.fetch = originalFetch;
    vi.restoreAllMocks();
  });

  it('prepends search_query: for query type and search_document: for document type', async () => {
    const embedder = new OllamaEmbedder('http://x', 'm', 10); // 10 matches FAKE_VECTOR length
    let capturedPrompt = '';

    global.fetch = vi.fn(async (_url, opts: any) => {
      capturedPrompt = JSON.parse(opts.body).prompt;
      return { ok: true, statusText: '', json: async () => ({ embedding: FAKE_VECTOR }) } as Response;
    });

    await embedder.embed('find files', 'query');
    expect(capturedPrompt).toBe('search_query: find files');

    await embedder.embed('glob pattern', 'document');
    expect(capturedPrompt).toBe('search_document: glob pattern');
  });

  it('defaults to document prefix when type is omitted', async () => {
    const embedder = new OllamaEmbedder('http://x', 'm', 10);
    let capturedPrompt = '';

    global.fetch = vi.fn(async (_url, opts: any) => {
      capturedPrompt = JSON.parse(opts.body).prompt;
      return { ok: true, statusText: '', json: async () => ({ embedding: FAKE_VECTOR }) } as Response;
    });

    await embedder.embed('glob pattern');
    expect(capturedPrompt).toBe('search_document: glob pattern');
  });

  it('throws when the model returns fewer dimensions than requested', async () => {
    mockFetch({ ok: true, body: { embedding: [0.1, 0.2] } }); // only 2 dims
    const embedder = new OllamaEmbedder('http://x', 'm', 768);
    await expect(embedder.embed('test', 'document')).rejects.toThrow(/returned 2 dimensions/);
  });

  it('truncates to requested dimensions (Matryoshka)', async () => {
    mockFetch({ ok: true, body: { embedding: FAKE_VECTOR } }); // 10 dims
    const embedder = new OllamaEmbedder('http://x', 'm', 5);
    const result = await embedder.embed('test', 'document');
    expect(result).toHaveLength(5);
    expect(result).toEqual([0.1, 0.2, 0.3, 0.4, 0.5]);
  });

  it('returns the full vector when dimensions match exactly', async () => {
    mockFetch({ ok: true, body: { embedding: FAKE_VECTOR } }); // 10 dims
    const embedder = new OllamaEmbedder('http://x', 'm', 10);
    const result = await embedder.embed('test', 'document');
    expect(result).toHaveLength(10);
    expect(result).toEqual(FAKE_VECTOR);
  });

  it('throws a descriptive error on non-ok response', async () => {
    mockFetch({ ok: false, statusText: 'Bad Request' });
    const embedder = new OllamaEmbedder('http://x', 'm', 768);
    await expect(embedder.embed('test', 'document')).rejects.toThrow(/Bad Request/);
  });

  it('throws a connection-failed error when fetch rejects', async () => {
    mockFetch({ ok: true, failWith: new Error('ECONNREFUSED') });
    const embedder = new OllamaEmbedder('http://x', 'm', 768);
    await expect(embedder.embed('test', 'document')).rejects.toThrow(/connection failed/);
  });

  it('throws a timeout error when the AbortController fires', async () => {
    // Simulate the AbortController firing (what happens after 30s of no
    // response) by having fetch reject with an AbortError. This exercises the
    // catch branch that converts AbortError into a "timed out" message.
    const abortError = new Error('The operation was aborted');
    abortError.name = 'AbortError';
    global.fetch = vi.fn(async () => { throw abortError; });
    const embedder = new OllamaEmbedder('http://x', 'm', 10);
    await expect(embedder.embed('test', 'document')).rejects.toThrow(/timed out/);
  });
});
