import { writeFileSync, readFileSync, existsSync } from 'node:fs';
import { Tool, IVectorStore, VectorStoreMetadata, ScoredTool, SearchHints } from '../types.js';
import { cosineSimilarity } from '../math/cosine-similarity.js';
import { normalize } from '../math/norm.js';

interface StoredVector {
  tool: Tool;
  embedding: Float32Array;
}

export class VectorStore implements IVectorStore {
  private vectors: StoredVector[] = [];
  private metadata: VectorStoreMetadata | null = null;
  private readonly storePath: string;

  constructor(storePath = 'vector-store.jsonl') {
    this.storePath = storePath;
  }

  setMetadata(metadata: VectorStoreMetadata): void {
    this.metadata = metadata;
  }

  getMetadata(): VectorStoreMetadata | null {
    return this.metadata;
  }

  async add(tools: Tool[], embeddings: number[][]): Promise<void> {
    if (tools.length !== embeddings.length) {
      throw new Error('Tools and embeddings length mismatch');
    }

    for (let i = 0; i < tools.length; i++) {
      const normEmbedding = normalize(embeddings[i]);
      const existingIndex = this.vectors.findIndex((v) => v.tool.name === tools[i].name);
      const record = { tool: tools[i], embedding: normEmbedding };
      if (existingIndex >= 0) {
        this.vectors[existingIndex] = record;
      } else {
        this.vectors.push(record);
      }
    }
  }

  // Returns scored tools so callers (the retriever, the CLI) can surface the
  // confidence or fuse with other signals. The zero-norm guard on
  // cosineSimilarity means an all-zero upstream vector yields a safe 0 instead
  // of NaN poisoning the rank.
  async search(queryEmbedding: number[], k: number, hints?: SearchHints): Promise<ScoredTool[]> {
    if (this.vectors.length === 0) return [];

    // Stored vectors are already unit length normalising the query turns the
    // dot product into cosine similarity, exactly what cosineSimilarity gives
    // us except we keep the cheap implicit form for the bulk pass.
    const normQuery = normalize(queryEmbedding);
    const boost = hints?.boost ?? 1.15;
    const boostTools = hints?.boostTools;

    const scores: ScoredTool[] = this.vectors.map((stored) => {
      let score = cosineSimilarity(normQuery, stored.embedding);
      if (boostTools && boostTools.has(stored.tool.name)) {
        score *= boost;
      }
      return { tool: stored.tool, score };
    });

    scores.sort((a, b) => b.score - a.score);

    return scores.slice(0, k);
  }

  // Back-compat shim: returns tools only (drops scores) for any caller that
  // predates scoring exposure. Kept narrow; the retriever uses search()
  // directly.
  async searchTools(queryEmbedding: number[], k: number, hints?: SearchHints): Promise<Tool[]> {
    return (await this.search(queryEmbedding, k, hints)).map((s) => s.tool);
  }

  async load(): Promise<void> {
    if (!existsSync(this.storePath)) {
      this.vectors = [];
      this.metadata = null;
      return;
    }

    const content = readFileSync(this.storePath, 'utf-8');
    const lines = content.split(/\r?\n/);
    this.vectors = [];
    this.metadata = null;

    for (const line of lines) {
      if (!line.trim()) continue;
      try {
        const entry = JSON.parse(line);
        if (entry.type === 'metadata') {
          this.metadata = entry.data;
        } else if (entry.type === 'vector') {
          this.vectors.push({
            tool: entry.data.tool,
            embedding: new Float32Array(entry.data.embedding),
          });
        }
      } catch (error) {
        console.warn(`Warning: Failed to parse line in vector store. Skipping line.`);
      }
    }
  }

  async save(): Promise<void> {
    const lines: string[] = [];
    if (this.metadata) {
      lines.push(JSON.stringify({ type: 'metadata', data: this.metadata }));
    }
    for (const stored of this.vectors) {
      lines.push(JSON.stringify({
        type: 'vector',
        data: {
          tool: stored.tool,
          embedding: Array.from(stored.embedding),
        }
      }));
    }
    writeFileSync(this.storePath, lines.join('\n') + '\n');
  }

  size(): number {
    return this.vectors.length;
  }

  // Exposes the in-memory tool catalog so the retriever can compute keyword
  // overlap (S3) over the same set the dense search saw. Returns a shallow
  // copy to keep the store's internal state encapsulated.
  tools(): Tool[] {
    return this.vectors.map((v) => v.tool);
  }

  // Composes the canonical per-tool text using the same rules as
  // buildEmbeddingText() in index.command.ts. Lets keyword-overlap (S3) bias
  // on the same vocabulary the dense pass indexed — param descriptions,
  // intent, examples, whenToUse, and whenNotToUse — without duplicating the
  // rule across modules.
  toolTexts(): Map<string, string> {
    const out = new Map<string, string>();
    for (const v of this.vectors) {
      const tool = v.tool;
      const parts: string[] = [tool.name, tool.description];
      const properties = (tool.parameters as { properties?: Record<string, { description?: string }> } | undefined)?.properties;
      if (properties) {
        for (const prop of Object.values(properties)) {
          if (prop?.description) parts.push(prop.description);
        }
      }
      if (tool.intent) parts.push(tool.intent);
      if (tool.examples && tool.examples.length > 0) parts.push(tool.examples.join('. '));
      if (tool.whenToUse && tool.whenToUse.length > 0) parts.push(tool.whenToUse.join('. '));
      if (tool.whenNotToUse && tool.whenNotToUse.length > 0) {
        parts.push(tool.whenNotToUse.map((s) => `NOT: ${s}`).join('. '));
      }
      out.set(tool.name, parts.filter(Boolean).join('. '));
    }
    return out;
  }
}
