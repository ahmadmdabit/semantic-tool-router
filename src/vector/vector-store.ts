import { writeFileSync, readFileSync, existsSync } from 'node:fs';
import { Tool, IVectorStore, VectorStoreMetadata } from '../types.js';
import { cosineSimilarity } from '../math/cosine-similarity.js';

interface StoredVector {
  tool: Tool;
  embedding: number[];
}

interface VectorStoreData {
  metadata: VectorStoreMetadata | null;
  vectors: StoredVector[];
}

export class VectorStore implements IVectorStore {
  private vectors: StoredVector[] = [];
  private metadata: VectorStoreMetadata | null = null;
  private readonly storePath: string;

  constructor(storePath = 'vector-store.json') {
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
      const existingIndex = this.vectors.findIndex((v) => v.tool.name === tools[i].name);
      const record = { tool: tools[i], embedding: embeddings[i] };
      if (existingIndex >= 0) {
        this.vectors[existingIndex] = record; // Upsert
      } else {
        this.vectors.push(record);
      }
    }
  }

  async search(queryEmbedding: number[], k: number): Promise<Tool[]> {
    if (this.vectors.length === 0) return [];

    const scores = this.vectors.map((stored) => ({
      tool: stored.tool,
      score: cosineSimilarity(queryEmbedding, stored.embedding),
    }));

    scores.sort((a, b) => b.score - a.score);

    return scores.slice(0, k).map((item) => item.tool);
  }

  async load(): Promise<void> {
    if (!existsSync(this.storePath)) {
      this.vectors = [];
      this.metadata = null;
      return;
    }

    const content = readFileSync(this.storePath, 'utf-8');
    try {
      const data: VectorStoreData = JSON.parse(content);
      this.metadata = data.metadata || null;
      this.vectors = data.vectors || data as any;
    } catch (error) {
      console.warn(`Warning: Failed to parse vector store at ${this.storePath}. Initializing empty store.`);
      this.vectors = [];
      this.metadata = null;
    }
  }

  async save(): Promise<void> {
    const data: VectorStoreData = {
      metadata: this.metadata,
      vectors: this.vectors,
    };
    writeFileSync(this.storePath, JSON.stringify(data, null, 2));
  }

  size(): number {
    return this.vectors.length;
  }
}
