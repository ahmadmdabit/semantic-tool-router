import { writeFileSync, readFileSync, existsSync } from 'node:fs';
import { Tool, IVectorStore, VectorStoreMetadata } from '../types.js';
import { dot } from '../math/dot.js';
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

  async search(queryEmbedding: number[], k: number): Promise<Tool[]> {
    if (this.vectors.length === 0) return [];

    const normQuery = normalize(queryEmbedding);
    const scores = this.vectors.map((stored) => ({
      tool: stored.tool,
      score: dot(normQuery, stored.embedding),
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
}