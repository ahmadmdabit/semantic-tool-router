export interface Tool {
  name: string;
  description: string;
  parameters: Record<string, any>;
  // Optional enrichment fields for retrieval quality (all optional for backward-compat).
  intent?: string;
  examples?: string[];       // positive natural-language usages
  whenToUse?: string[];      // positive inclusion criteria (1-3 bullets)
  whenNotToUse?: string[];   // negative exclusion criteria (1-2 bullets, usually "use X instead")
  // S2 enrichment — optional, fully backward-compat. `triggers` are matched
  // against the lower-cased query; when one fires, this tool and every tool
  // named in `boosts` gets the S2 score bump. Lets the intent detector scale
  // to any catalog without hard-coding tool names in source.
  triggers?: string[];
  boosts?: string[];
  strict?: boolean;
}

export type EmbedType = 'query' | 'document';

export interface IEmbedder {
  embed(text: string, type?: EmbedType): Promise<number[]>;
}

export interface VectorStoreMetadata {
  model: string;
  dimensions: number;
  indexedAt: string;
}

// Vector-search result including the cosine score so callers (route command,
// fusion retriever, tests) can expose confidence or blend with other signals.
export interface ScoredTool {
  tool: Tool;
  score: number;
}

// Structural hints from the intent pre-classifier. `boostTools` get their
// cosine score multiplied by `boost` inside VectorStore.search().
export interface SearchHints {
  boostTools?: Set<string>;
  boost?: number;
}

export interface IVectorStore {
  add(tools: Tool[], embeddings: number[][]): Promise<void>;
  search(queryEmbedding: number[], k: number, hints?: SearchHints): Promise<ScoredTool[]>;
  load(): Promise<void>;
  save(): Promise<void>;
  size(): number;
  getMetadata(): VectorStoreMetadata | null;
  setMetadata(metadata: VectorStoreMetadata): void;
}
