import { IEmbedder } from '../types.js';

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

  async embed(text: string): Promise<number[]> {
    const response = await fetch(`${this.baseUrl}/api/embeddings`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model: this.model,
        prompt: text,
      }),
    });

    if (!response.ok) {
      throw new Error(`Ollama embedding failed: ${response.statusText}`);
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
