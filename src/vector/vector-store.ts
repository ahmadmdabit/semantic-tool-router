import { writeFileSync, readFileSync, existsSync } from 'node:fs';
import { Tool, IVectorStore, VectorStoreMetadata, ScoredTool, SearchHints } from '../types.js';
import { cosineSimilarity } from '../math/cosine-similarity.js';
import { normalize } from '../math/norm.js';

interface StoredVector {
  tool: Tool;
  // Positive prototype: what the tool IS. Embedded from name + description +
  // params + intent + examples + whenToUse.
  posVec: Float32Array;
  // Negative prototype: what the tool is NOT. Embedded from whenNotToUse (the
  // sentences prefixed with NOT: at index time). Empty Float32Array when the
  // tool declares no negative boundaries — polarity subtraction then contributes
  // nothing and the score falls back to pure positive cosine.
  negVec: Float32Array;
}

// Weight applied to the negative-prototype cosine before subtracting it from
// the positive score. The negative text is dense (every token is about what the
// tool is NOT) so its cosine with a matching query runs high per-token; this
// dampens it to a meaningful-but-not-dominant penalty. Override with
// $POLARITY_ALPHA during tuning.
const PolarityAlpha = Number(process.env.POLARITY_ALPHA ?? 0.3);

// Length of a zero-length negVec — used to detect "no negative prototype" and
// skip the polarity subtraction for tools that haven't declared boundaries.
const NEG_VEC_ZERO = new Float32Array(0);

function isZeroVec(v: Float32Array): boolean {
  for (let i = 0; i < v.length; i++) if (v[i] !== 0) return false;
  return true;
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

  async add(tools: Tool[], embeddings: number[][], negEmbeddings?: number[][]): Promise<void> {
    if (tools.length !== embeddings.length) {
      throw new Error('Tools and embeddings length mismatch');
    }
    if (negEmbeddings && negEmbeddings.length !== tools.length) {
      throw new Error('Tools and negEmbeddings length mismatch');
    }

    for (let i = 0; i < tools.length; i++) {
      const posVec = normalize(embeddings[i]);
      // When no negative embeddings are supplied (older callers, tests), fall
      // back to a zero-length negative vector — polarity subtraction then adds
      // nothing and behaviour is identical to the single-vector design.
      const negVec = negEmbeddings
        ? (negEmbeddings[i]?.length ? normalize(negEmbeddings[i]) : NEG_VEC_ZERO)
        : NEG_VEC_ZERO;
      const existingIndex = this.vectors.findIndex((v) => v.tool.name === tools[i].name);
      const record = { tool: tools[i], posVec, negVec };
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
      // Polarity-adjusted S1: how well the query matches what the tool IS minus
      // how well it matches what the tool is NOT. When the query shares tokens
      // with a tool's whenNotToUse boundaries, the negative-prototype cosine
      // rises and the net score drops — this is the mechanism the NOT: prefix
      // was always trying to provide but couldn't achieve inside a single
      // centroid (where the negative text only ADDED to cosine).
      let score = cosineSimilarity(normQuery, stored.posVec);
      if (stored.negVec.length > 0 && !isZeroVec(stored.negVec)) {
        score -= PolarityAlpha * cosineSimilarity(normQuery, stored.negVec);
      }
      // S2 boost (multiplicative) is applied on top of the polarity-adjusted
      // score so a structural signal can still rescue a tool whose cosine sits
      // just below a near-tie.
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
          const data = entry.data;
          // posVec is the new name for the single-vector `embedding` field.
          // Accept both so indexes written before this change load without
          // re-indexing (negVec defaults to zero → no polarity penalty).
          const pos = data.posVec ?? data.embedding;
          if (!pos) {
            console.warn('Warning: vector record missing posVec/embedding. Skipping.');
            continue;
          }
          const neg = data.negVec && (data.negVec as number[]).length > 0
            ? new Float32Array(data.negVec as number[])
            : NEG_VEC_ZERO;
          this.vectors.push({
            tool: data.tool,
            posVec: new Float32Array(pos),
            negVec: neg,
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
      const record: Record<string, unknown> = {
        tool: stored.tool,
        posVec: Array.from(stored.posVec),
      };
      // Only persist the negative vector when it carries signal; keeps the
      // JSONL lean for tools that declare no negative boundaries.
      if (stored.negVec.length > 0 && !isZeroVec(stored.negVec)) {
        record.negVec = Array.from(stored.negVec);
      }
      lines.push(JSON.stringify({ type: 'vector', data: record }));
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
