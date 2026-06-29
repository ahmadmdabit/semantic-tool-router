# Semantic Tool Router CLI

A pure TypeScript and Node.js implementation of a Semantic Tool Router. This CLI replaces static, full-catalog loading with Just-in-Time (JIT) Context Injection to avoid the "Fat Agent" architecture trap.

## Overview

When an AI agent's tool catalog exceeds 50 tools, injecting all tool schemas into the model's prompt leads to "lost in the middle" syndrome, where high context volume causes the model to ignore relevant instructions. The Semantic Tool Router solves this by dynamically selecting and injecting only the top-K most relevant tool schemas per request using semantic vector search.

## Core Features

- **Just-in-Time Context Injection**: Dynamically injects only the top-K most relevant tool schemas per request.
- **Token Reduction**: Reduces tool context tokens by up to 99%.
- **Low Latency**: Improves Time-to-First-Token (TTFT) by minimizing prompt size.
- **Semantic Routing**: Uses vector embeddings to match user intent with tool descriptions accurately.
- **Scalability**: Maintains consistent accuracy regardless of catalog size, supporting hundreds of tools.
- **Matryoshka Embedding Support**: Truncates model embeddings to the requested dimension for flexible performance tuning.

## Project Scope

This phase is strictly minimal and focuses on two core CLI commands:

- `index`: Builds or updates the vector index from a directory of tool JSON files.
- `route`: Given a user query, returns the Top-K most relevant tools.

Advanced features such as evaluation, fallback logic, and observability are out of scope for this phase (YAGNI).

## CLI Commands

The CLI exposes two strictly minimal commands:

### `index`

Builds or updates the vector index from a directory of tool JSON files.

```bash
semantic-tool-router index <tools-directory> [options]
```

Options:

| Option             | Default                   | Description                                           |
| ------------------ | ------------------------- | ----------------------------------------------------- |
| `-o, --output`     | `vector-store.json`       | Output path for vector store                          |
| `-m, --model`      | `nomic-embed-text:latest` | Ollama embedding model                                |
| `-d, --dimensions` | `768`                     | Embedding dimensions (supports Matryoshka truncation) |

### `route`

Given a user query, returns the Top-K most relevant tools from the index.

```bash
semantic-tool-router route <query> [options]
```

Options:

| Option             | Default                   | Description                               |
| ------------------ | ------------------------- | ----------------------------------------- |
| `-k, --top-k`      | `5`                       | Number of tools to return                 |
| `-s, --store`      | `vector-store.json`       | Path to vector store                      |
| `-m, --model`      | `nomic-embed-text:latest` | Ollama embedding model (must match index) |
| `-d, --dimensions` | `768`                     | Embedding dimensions (must match index)   |

## Technology Decisions

| Area            | Choice                                           | Rationale                                       |
| --------------- | ------------------------------------------------ | ----------------------------------------------- |
| Language        | TypeScript (strict)                              | Type safety without complexity.                 |
| CLI Framework   | `commander`                                      | Standard, minimal dependency.                   |
| Embedding       | Ollama (`nomic-embed-text:latest`, configurable) | Local, specified embedding model.               |
| Vector Store    | In-memory + JSON file persistence                | Simple, zero external database dependencies.    |
| Similarity      | Cosine similarity (manual implementation)        | No extra mathematical dependencies.             |
| Package Manager | yarn                                             | Standard ecosystem, deterministic lockfile.     |
| Embedding Dims  | 768 default (configurable via `--dimensions`)    | Matryoshka truncation for flexible performance. |

## Matryoshka Embeddings

The embedder supports Matryoshka (Russian doll) embedding models. When a model returns more dimensions than requested, the output is truncated to the specified dimension. This enables:

- **Performance tuning**: Use fewer dimensions for faster search at the cost of some accuracy.
- **Migration flexibility**: Re-index with different dimensions without changing models.
- **Validation**: A hard error is thrown if the model returns fewer dimensions than requested (dimensions cannot be expanded).

When using the `route` command, the vector store metadata is checked to ensure the requested dimensions match the indexed dimensions. A mismatch produces an error to prevent incorrect similarity calculations.

## File Structure

```text
semantic-tool-router/
├── src/
│   ├── cli.ts                    # Entry point (Commander)
│   ├── types.ts                  # Shared interfaces only
│   ├── commands/
│   │   ├── index.command.ts      # `index` command
│   │   └── route.command.ts      # `route` command
│   ├── embeddings/
│   │   └── ollama-embedder.ts    # Single responsibility: embedding
│   ├── math/
│   │   ├── cosine-similarity.ts  # Cosine similarity function
│   │   ├── dot.ts                # Dot product function
│   │   └── norm.ts               # L2 norm function
│   ├── tools/
│   │   └── tool-loader.ts        # Loads tools from directory
│   └── vector/
│       └── vector-store.ts       # In-memory + persistence
├── tools/                        # User-provided tool catalog (JSON files)
├── dist/                         # Compiled JavaScript output (gitignored)
├── vector-store.json             # Generated index (should be gitignored)
├── package.json
├── tsconfig.json
├── .yarnrc.yml
├── yarn.lock
└── README.md
```

## Architecture

### 1. Catalog Design (Offline)

Centralize all tool definitions. Each entry includes:

- Unique `name`
- Descriptive `description` in natural language (compatible with OpenAI Tool Call specification)
- `parameters` following JSON Schema conventions

### 2. Vector Indexing (Offline)

- Embed tool descriptions using an embedding model.
- Store vectors in an in-memory index persisted to a JSON file, with metadata including model name and dimensions.

### 3. Router Runtime (Per Request)

- Embed the user query using the same model and dimensions as the index.
- Validate metadata compatibility (model name warning, dimension mismatch error).
- Perform Nearest Neighbor Search against the vector index to retrieve the Top-K tools (default: K=5).
- Inject only these K schemas into the LLM call.

## Design Principles

- **YAGNI**: No evaluation mode, metrics, or fallback strategies in v1.
- **KISS**: Manual math implementations and JSON persistence avoid heavy external dependencies.
- **SOLID**: Each module has a single responsibility, and commands depend on abstractions (`IEmbedder`, `IVectorStore`).
- **No Premature Optimization**: No caching, batching, or parallel embedding unless required later.

## SOLID Alignment

| Principle                 | Implementation                                                                          |
| ------------------------- | --------------------------------------------------------------------------------------- |
| **S**ingle Responsibility | Each file has one clear job (Embedder, VectorStore, ToolLoader, Commands).              |
| **O**pen/Closed           | VectorStore and Embedder can be swapped later via interface without changing consumers. |
| **L**iskov                | Not applicable yet (no inheritance).                                                    |
| **I**nterface Segregation | Small, focused interfaces (`IEmbedder`, `IVectorStore`).                                |
| **D**ependency Inversion  | Commands depend on abstractions (`IEmbedder`, `IVectorStore`).                          |
| Object Schema             | Tool uses `description`/`parameters` fields matching OpenAI Tool Call specification.    |

## Core Interfaces

```typescript
// types.ts (OpenAI Tool Call compatible)
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
```

## Implementation Order

1. **Setup**: `package.json`, `tsconfig.json`, basic CLI skeleton.
2. **Tool Loader**: Read directory of JSON files.
3. **Ollama Embedder**: Simple HTTP call to Ollama.
4. **Vector Store**: In-memory + JSON persistence + cosine similarity.
5. **CLI Commands**: `index` and `route`.

## Mathematical Implementation

To avoid heavy dependencies like `mathjs`, the core mathematical logic for cosine similarity is implemented as clean, pure, dependency-free TypeScript functions.

### Dot Product

```typescript
/**
 * Calculate the dot product of two vectors.
 */
export function dot(a: number[], b: number[]): number {
  if (a.length !== b.length) {
    throw new Error(
      `Vectors must have equal length (${a.length} != ${b.length})`,
    );
  }
  if (a.length === 0) {
    throw new Error("Cannot calculate the dot product of empty vectors");
  }

  let result = 0;
  for (let i = 0; i < a.length; i++) {
    result += a[i] * b[i];
  }
  return result;
}
```

### L2 Norm

```typescript
/**
 * Calculate the L2 (Euclidean) norm of a vector.
 */
export function norm(vec: number[]): number {
  if (vec.length === 0) return 0;

  let sum = 0;
  for (let i = 0; i < vec.length; i++) {
    sum += vec[i] * vec[i];
  }
  return Math.sqrt(sum);
}
```

### Cosine Similarity

```typescript
import { dot } from "./dot";
import { norm } from "./norm";

export function cosineSimilarity(a: number[], b: number[]): number {
  const magA = norm(a);
  const magB = norm(b);

  if (magA === 0 || magB === 0) return 0;

  return dot(a, b) / (magA * magB);
}
```

## Configuration and Best Practices

### Evaluation

Test different values of K (e.g., 3, 5, 10) against a benchmark set to find the optimal balance between accuracy and latency.

### Tool Descriptions

Crucial for routing accuracy. Use the vocabulary users actually employ, including explicit intent, action, and key entities.

### Observability

Log the selected tools, final tool call, and fallbacks. This is critical for identifying router misses and re-indexing when tools change.

### Fallback Strategy

If the model fails a task, implement logic to widen K or trigger a secondary, broader retrieval pass.

### Dimension Consistency

When using `route`, the vector store metadata is validated. The model name triggers a warning on mismatch. The dimension count triggers a hard error on mismatch, preventing incorrect similarity calculations.

## Getting Started

### Prerequisites

- Node.js (v18 or higher)
- Ollama running locally with the `nomic-embed-text:latest` model pulled (`ollama pull nomic-embed-text:latest`)

### Installation

```bash
yarn install
```

### Build

```bash
yarn run build
```

### Usage

1. Index your tool catalog:

```bash
yarn start -- index ./tools
```

With custom model and dimensions:

```bash
yarn start -- index ./tools --model nomic-embed-text:latest --dimensions 256
```

2. Route a query:

```bash
yarn start -- route "List all files in the current directory"
```

For development with hot-reloading:

```bash
yarn run dev
```

## License

[MIT](LICENSE)
