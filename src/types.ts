export interface Tool {
  name: string;
  description: string;
  parameters: Record<string, any>;
}

export interface IEmbedder {
  embed(text: string): Promise<number[]>;
}

export interface VectorStoreMetadata {
  model: string;
  dimensions: number;
  indexedAt: string;
}

export interface IVectorStore {
  add(tools: Tool[], embeddings: number[][]): Promise<void>;
  search(queryEmbedding: number[], k: number): Promise<Tool[]>;
  load(): Promise<void>;
  save(): Promise<void>;
  size(): number;
  getMetadata(): VectorStoreMetadata | null;
  setMetadata(metadata: VectorStoreMetadata): void;
}
