import { IEmbedder, EmbedType } from '../types.js';

export class OllamaEmbedder implements IEmbedder {
  private readonly baseUrl: string;
  private readonly model: string;
  private readonly dimensions: number;

  constructor(
    baseUrl = 'http://localhost:11434',
    model = 'nomic-embed-text:latest',
    dimensions = 768
  ) {
    this.baseUrl = baseUrl;
    this.model = model;
    this.dimensions = dimensions;
  }

  // nomic-embed-text v1.5+ expects retrieval-task prefixes. Default 'document'
  // preserves backward-calls that omit the arg.
  async embed(text: string, type: EmbedType = 'document'): Promise<number[]> {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 30000); // 30s timeout

    // Prefix keeps query and document embeddings on-manifold for nomic models;
    // non-nomic models tolerate it because the trailing payload still carries
    // the original text verbatim.
    const prompt = type === 'query' ? `search_query: ${text}` : `search_document: ${text}`;

    let response: Response;
    try {
      response = await fetch(`${this.baseUrl}/api/embeddings`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model: this.model,
          prompt,
        }),
        signal: controller.signal,
      });
      clearTimeout(timeoutId);
      if (!response.ok) {
        throw new Error(`Ollama embedding failed: ${response.statusText}`);
      }
    } catch (error: any) {
      clearTimeout(timeoutId);
      if (error.name === 'AbortError') throw new Error(`Ollama embedding timed out after 30s at ${this.baseUrl}`);
      throw new Error(`Ollama connection failed at ${this.baseUrl}. Ensure Ollama is running. Details: ${error.message}`);
    }

    const data = await response.json();
    const embedding: number[] = data.embedding;

    // Support Matryoshka embeddings: truncate if model returns more dimensions
    if (embedding.length < this.dimensions) {
      throw new Error(
        `Model returned ${embedding.length} dimensions, but ${this.dimensions} requested. Cannot expand dimensions.`
      );
    }

    // Truncate to requested dimensions (Matryoshka capability)
    return embedding.slice(0, this.dimensions);
  }
}
