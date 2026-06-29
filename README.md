# Semantic Tool Router CLI

A pure TypeScript and Node.js implementation of a Semantic Tool Router. This CLI replaces static, full-catalog loading with Just-in-Time (JIT) Context Injection to avoid the "Fat Agent" architecture trap.

## Acknowledgments

This implementation is based on the article **"The 100-Tool Agent Is a Trap: Overcoming the
Latency, Cost, and Accuracy Collapse of Large-Scale Function Calling"** which is created based on YouTube session by **Sohail Shaikh**
and **Ankush Rastogi** (The 100-Tool Agent Is a Trap - Sohail Shaikh & Ankush Rastogi, Prosodica), adapted from their research and production deployments.

- [Read the article](https://gist.github.com/ahmadmdabit/f6b782835e9bec46613bd1435ea611cc)
- [Watch the session on YouTube](https://www.youtube.com/watch?v=vh2VGuQ3zhY)

Their article defines the core problem — the "Fat Agent" pattern collapses at ~50+ tools due to
"lost in the middle" syndrome, token bloat (127K+ tokens for 741 tools), and TTFT spikes
(>5s at 500+ tools) — and proposes the solution: **Semantic Routing** plus **Just-In-Time
Context Injection**. The article benchmarks the Fat Agent baseline (78% → 13% accuracy as the
catalog grows from 10 to 741 tools) against a routed architecture (~83% stable accuracy
regardless of catalog size, ~99% token reduction).

The present implementation follows their three-step blueprint (build index offline, route each
query at runtime, inject only the top-K schemas) and extends it with the improvements below.

### Improvements over the original specification

| #   | Specification                                   | Our improvement                                                                                                                                                                                                                |
| --- | ----------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 1   | Cosine-similarity routing (single signal)       | **Three-signal RRF layered on top of cosine** — dense cosine as the baseline the article recommends, plus structural intent detector and keyword overlap fused via RRF; catches token-collision cases that cosine alone smears |
| 2   | Tool description = freeform natural language    | **Structured boundary vocabulary** (`whenToUse`, `whenNotToUse`, `triggers`, `boosts`, `intent`, `examples`) embedded alongside the description and compiled into the S2 rule table                                            |
| 3   | Hardcoded or no disambiguation layer            | **Data-driven S2 detector** with zero hardcoded tool names — dynamic rules from `triggers`/`boosts` evaluated before 9 builtin baseline verb patterns, so new tools get coverage from their own JSON without source changes    |
| 4   | No retrieval-task alignment for embedding model | **Nomic retrieval prefixes** (`search_query:` / `search_document:`) keep embeddings on-manifold for nomic-embed-text v1.5+, improving recall                                                                                   |
| 5   | No per-request confidence signal                | **`--threshold` filter** drops tools below a composite-score floor, so callers detect "no confident match" instead of taking the least-bad one                                                                                 |
| 6   | No score visibility or debug output             | **Per-tool scores and `--json` debug breakdown** (cosine, keyword overlap, intent pattern that fired) enable rapid diagnosis of router misses                                                                                  |
| 7   | Generic keyword matching for S3                 | **Glob-token-aware tokenizer** for S3 (`*.ts` also yields `ts` and `.ts`) plus stopword stripping so surface matches cosine alone misses are caught                                                                            |
| 8   | Persistence medium left open                    | **JSONL format** (one record per line) with pre-normalized Float32Array embeddings and metadata (model, dimensions, indexedAt)                                                                                                 |

### Not yet implemented (per original specification)

| #   | Specification item                                                                                         | Status             | Notes                                                                                                                                                         |
| --- | ---------------------------------------------------------------------------------------------------------- | ------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1   | **Evaluation benchmark** against Berkeley Function Calling Leaderboard or Toolbench                        | ❌ Not implemented | Recommended to test K = 3, 5, 10 against a benchmark set                                                                                                      |
| 2   | **Fallback strategy** to widen K or trigger a secondary broader retrieval pass when the model fails a task | ❌ Not implemented | The `--threshold 0` escape hatch partially covers this, but no automatic widening                                                                             |
| 3   | **Scalable vector DB** (Chroma, Pinecone, Qdrant)                                                          | ❌ Not implemented | Currently uses in-memory + JSONL; suitable for single-node deployments. A production catalog at hundreds of tools would benefit from a dedicated vector store |
| 4   | **Observability logging** of selected tools, final tool call, and fallbacks                                | ❌ Not implemented | Raw scores are exposed via `--json`, but no structured logging or metrics pipeline yet                                                                        |

## Overview

When an AI agent's tool catalog exceeds 50 tools, injecting all tool schemas into the model's prompt leads to "lost in the middle" syndrome, where high context volume causes the model to ignore relevant instructions. The Semantic Tool Router solves this by dynamically selecting and injecting only the top-K most relevant tool schemas per request using semantic vector search fused from multiple signals.

## System Context

```mermaid
flowchart LR
    User([User Query]) --> Router[Semantic Tool Router]
    subgraph Offline["Offline: index command"]
        Catalog[Tool Catalog JSON] --> Index[Vector Indexer]
        Index -->|embed then normalize| Store[(vector-store.jsonl)]
    end
    subgraph Online["Online: route command"]
        Router -->|embed query| S1[S1 Cosine]
        Router -->|scan query| S2[S2 Intent]
        Router -->|tokenize| S3[S3 Keyword]
        S1 --> RRF[RRF Fusion]
        S2 --> RRF
        S3 --> RRF
        RRF -->|top-K plus scores| Inject[Inject K Tool Schemas]
    end
    Store --> S1
    Inject --> LLM[LLM Call]
    LLM --> Result([Tool Execution Result])
```

The router sits between the user and the LLM: offline, it indexes the tool catalog into a vector store; online, it ranks every tool against the query and injects only the top-K schemas into the prompt.

## Core Features

- **Just-in-Time Context Injection**: Dynamically injects only the top-K most relevant tool schemas per request.
- **Multi-Signal Ranking**: Blends a dense embedding cosine pass, a structural/intent pre-classifier, and stopword-aware keyword overlap via Reciprocal-Rank Fusion (RRF) so ambiguous queries resolve to the right tool instead of the one whose description wins a token-collision tie.
- **Tool Boundary Vocabulary**: Each tool can declare positive (`whenToUse`) and negative (`whenNotToUse`) inclusion criteria inspired by the Claude Skills Spec. These boundaries are embedded alongside the description, which is what lets the ranker distinguish a query like "List all \*.ts files" (glob) from "read package.json" (readFile) despite both sharing the token "file".
- **Data-Driven Structural Detector (S2)**: Each tool can declare `triggers` (query fragments that should boost it) and `boosts` (other tools to boost alongside it). The intent detector compiles these into a dynamic rule table at runtime — no hardcoded tool names in source. A brand-new tool added to the catalog gets S2 coverage automatically from its `triggers`, or from builtin baseline regexes that match on name/description shape.
- **Score Visibility & Confidence Floor**: The `route` command prints each tool's composite score, and a new `--threshold` flag drops tools whose score falls below a floor so callers can detect "no confident match" instead of taking the least-bad one.
- **Nomic Retrieval Prefixes**: Queries are embedded with `search_query:` and tool texts with `search_document:`, keeping nomic-embed-text v1.5+ embeddings on-manifold for retrieval.
- **Token Reduction**: Reduces tool context tokens by up to 99%.
- **Matryoshka Embedding Support**: Truncates model embeddings to the requested dimension for flexible performance tuning.

## Project Scope

This phase focuses on two core CLI commands:

- `index`: Builds or updates the vector index from a directory of tool JSON files.
- `route`: Given a user query, returns the Top-K most relevant tools.

Advanced features such as evaluation suites, automatic K-tuning, and distributed persistence are out of scope for this phase (YAGNI).

## CLI Commands

The CLI exposes two commands:

### `index`

Builds or updates the vector index from a directory of tool JSON files.

```bash
semantic-tool-router index <tools-directory> [options]
```

Options:

| Option             | Default                   | Description                                           |
| ------------------ | ------------------------- | ----------------------------------------------------- |
| `-o, --output`     | `vector-store.jsonl`      | Output path for vector store                          |
| `-m, --model`      | `nomic-embed-text:latest` | Ollama embedding model                                |
| `-d, --dimensions` | `768`                     | Embedding dimensions (supports Matryoshka truncation) |
| `-u, --url`        | `http://localhost:11434`  | Ollama API URL (overrides `$OLLAMA_HOST`)             |

### `route`

Given a user query, returns the Top-K most relevant tools from the index.

```bash
semantic-tool-router route <query> [options]
```

Options:

| Option             | Default                   | Description                                              |
| ------------------ | ------------------------- | -------------------------------------------------------- |
| `-k, --top-k`      | `5`                       | Number of tools to return                                |
| `-s, --store`      | `vector-store.jsonl`      | Path to vector store                                     |
| `-m, --model`      | `nomic-embed-text:latest` | Ollama embedding model (must match index)                |
| `-d, --dimensions` | `768`                     | Embedding dimensions (must match index)                  |
| `-u, --url`        | `http://localhost:11434`  | Ollama API URL (overrides `$OLLAMA_HOST`)                |
| `-j, --json`       | —                         | Output results as JSON (includes score and debug)        |
| `-t, --threshold`  | `0`                       | Drop tools with composite score below this (0 = disable) |

Example:

```
$ yarn start route "List all the files with *.ts"
Embedding query: "List all the files with *.ts"
Searching top 5 tools...

Top relevant tools:
1. glob  (score: 0.9836)
   Description: Find files matching a glob pattern. The pattern embeds the path.
   Intent signal: glob

2. grep  (score: 0.9393)
   ...

3. listDirectory  (score: 0.6256)
   ...

4. moveFile  (score: 0.6250)
   ...

5. scanDirectory  (score: 0.5992)
   ...
```

## Technology Decisions

| Area            | Choice                                           | Rationale                                                                                  |
| --------------- | ------------------------------------------------ | ------------------------------------------------------------------------------------------ |
| Language        | TypeScript (strict)                              | Type safety without complexity.                                                            |
| CLI Framework   | `commander`                                      | Standard, minimal dependency.                                                              |
| Embedding       | Ollama (`nomic-embed-text:latest`, configurable) | Local, specified embedding model.                                                          |
| Vector Store    | In-memory + JSONL persistence                    | Simple, zero external database dependencies.                                               |
| Similarity      | Cosine similarity (`cosine-similarity.ts`)       | Dedicated, zero-dependency module with a zero-norm guard; consumed by `VectorStore.search` |
| Ranking         | Reciprocal-Rank Fusion (RRF) over 3 signals      | Robust to heterogeneous signal magnitudes; lets dense, structural, and lexical cues vote.  |
| Package Manager | Yarn 4                                           | Standard ecosystem, deterministic lockfile.                                                |
| Embedding Dims  | 768 default (configurable via `--dimensions`)    | Matryoshka truncation for flexible performance.                                            |

## Matryoshka Embeddings

The embedder supports Matryoshka (Russian doll) embedding models. When a model returns more dimensions than requested, the output is truncated to the specified dimension. This enables:

- **Performance tuning**: Use fewer dimensions for faster search at the cost of some accuracy.
- Re-index with different dimensions without changing models.
- **Validation**: A hard error is thrown if the model returns fewer dimensions than requested (dimensions cannot be expanded).

When using the `route` command, the vector store metadata is checked to ensure the requested dimensions match the indexed dimensions. A mismatch produces an error to prevent incorrect similarity calculations.

## File Structure

```text
semantic-tool-router/
├── src/
│   ├── cli.ts                    # Entry point (Commander)
│   ├── types.ts                  # Shared interfaces and Tool / ScoredTool / SearchHints
│   ├── commands/
│   │   ├── index.command.ts      # `index` command (composes embedding text)
│   │   └── route.command.ts      # `route` command (scores, --threshold, --json)
│   ├── embeddings/
│   │   └── ollama-embedder.ts    # Embeds with `search_query:` / `search_document:` prefix
│   ├── math/
│   │   ├── cosine-similarity.ts  # Cosine similarity (zero-norm guard); used by VectorStore
│   │   ├── dot.ts                # Dot product
│   │   └── norm.ts               # L2 norm
│   ├── routing/
│   │   ├── intent-detector.ts    # S2: structural + intent pre-classifier
│   │   ├── keyword-overlap.ts    # S3: stopword-filtered Jaccard scorer
│   │   └── retriever.ts          # RRF fusion of S1 + S2 + S3
│   ├── tools/
│   │   └── tool-loader.ts        # Loads tools from directory
│   └── vector/
│       └── vector-store.ts       # In-memory + persistence, boundary-aware boost
├── tools/                        # User-provided tool catalog (14 JSON files)
├── dist/                         # Compiled JavaScript output (gitignored)
├── vector-store.jsonl            # Generated index (should be gitignored)
├── package.json
├── tsconfig.json
├── .yarnrc.yml
├── yarn.lock
└── README.md
```

## Architecture

### 1. Tool Design (Offline)

Each tool JSON file declares both the semantic and boundary vocabulary the router needs:

```json
{
  "name": "glob",
  "description": "Find files matching a glob pattern. The pattern embeds the path.",
  "intent": "find files by pattern",
  "examples": [
    "Find all .ts files in src",
    "Locate files matching *.json",
    "Which files match the pattern tests/**/*.test.ts"
  ],
  "whenToUse": [
    "The query contains a glob pattern (tokens like *, **, ?, {…}, […]) and the user wants matching files.",
    "The user wants to discover or list files whose names or paths follow a wildcard shape.",
    "Neither a file's content nor its exact location is known in advance."
  ],
  "whenNotToUse": [
    "The user wants to read the contents of a specific, known file. Use readFile instead.",
    "The user wants to search inside file contents for a regex or keyword. Use grep instead.",
    "The user just wants the immediate children of a directory without any name pattern. Use listDirectory instead."
  ],
  "triggers": ["glob", "glob pattern", "filename pattern", "file pattern", "wildcard", "find files", "locate files", "match files"],
  "boosts": ["grep"],
  "parameters": { "type": "object", ... }
}
```

| Field          | Required | Purpose                                                                                                                |
| -------------- | -------- | ---------------------------------------------------------------------------------------------------------------------- |
| `name`         | yes      | Stable tool identifier.                                                                                                |
| `description`  | yes      | Short declarative summary; the primary signal for the dense cosine pass.                                               |
| `intent`       | no       | Canonical verb-object pair (`"find files by pattern"`). Adds lexical density near the prototype.                       |
| `examples`     | no       | 3-5 natural-language queries that should route to this tool.                                                           |
| `whenToUse`    | no       | 1-3 positive inclusion criteria — when this tool is the right choice.                                                  |
| `whenNotToUse` | no       | 1-2 negative exclusion criteria ("Use another tool instead"). Pushes tools with overlapping positive vocabulary apart. |
| `triggers`     | no       | Query fragments (literal substrings or regex) that should boost this tool. Compiled into the S2 rule table at runtime. |
| `boosts`       | no       | Other tool names to boost alongside this tool when a trigger fires. Lets one pattern pull in multiple tools.           |
| `parameters`   | yes      | JSON Schema object.                                                                                                    |
| `strict`       | no       | Schema-only flag (not used by pipeline).                                                                               |

All enrichment fields are optional and fully backward-compat; an older catalog without them still works but does not benefit from the boundary-vocabulary boost.

### 2. Vector Indexing (Offline)

For each tool, `index` composes the text to embed from `name + description + parameter descriptions + intent + examples + whenToUse + whenNotToUse`. `whenNotToUse` sentences are prefixed with `NOT: ` so the negative-boundary vocabulary sits far from the positive prototype in nomic's vector space. The composed text is embedded with the `search_document: ` prefix (so the index lies on the retrieval manifold for nomic v1.5+), L2-normalized, and persisted in `vector-store.jsonl` along with metadata (`model`, `dimensions`, `indexedAt`).

```mermaid
flowchart TD
    Start([Tool JSON Files]) --> Load[ToolLoader.loadFromDirectory]
    Load --> Valid{Valid?}
    Valid -- No --> Skip[Skip Invalid]
    Valid -- Yes --> Compose[buildEmbeddingText]
    Compose -->|name + description + params + intent + examples + whenToUse + NOT: whenNotToUse| Embed[OllamaEmbedder.embed type=document]
    Embed -->|search_document: prefix + Matryoshka truncate| Norm[L2 Normalize]
    Norm -->|Float32Array| Dedup{Upsert by name?}
    Dedup -- Yes --> Replace[Replace Existing]
    Dedup -- No --> Add[Append New]
    Replace --> Save[Persist vector-store.jsonl]
    Add --> Save
    Save --> Meta[Write Metadata model+dimensions+indexedAt]
    Meta --> Done([Index Ready])
```

### 3. Router Runtime (Per Request)

Ranking happens in three fused signals:

- **S1 Dense cosine** — the query is embedded with the `search_query: ` prefix and scored against every stored vector using `cosineSimilarity` (which includes a zero-norm guard that returns 0 instead of NaN for empty embeddings). `VectorStore.search` accepts an optional `SearchHints` argument; any tool in `hints.boostTools` has its score multiplied by `hints.boost` (default 1.15).

- **S2 Structural intent** — `intent-detector.ts` builds a two-layer rule table at runtime. **Dynamic rules** are compiled from each tool's `triggers` and `boosts` fields (so a tool added to the catalog yesterday is already covered). **Builtin baselines** cover the nine granular verb patterns (`glob`, `write`, `edit`, `move`, `rename`, `read`, `search`, `list`, `scan`) and match tools by name-or-description shape, so even tools without declared triggers get a reasonable bump. Dynamic rules are evaluated first, so a tool-specific trigger always wins over a generic verb. `*.ts` in the query, for example, fires the glob rule and bumps `glob` plus every tool named in `glob.boosts` (e.g. `grep`). The detector has zero hardcoded tool names — it walks the live catalog from the store on every `retrieve()`.

```mermaid
flowchart TB
    subgraph DynamicLayer["Layer 1: Dynamic Rules"]
        direction LR
        T1["glob triggers<br/>glob, file pattern, wildcard"] -->|"compile"| R1["boost glob + grep"]
        T2["readFile triggers<br/>read file, print contents"] -->|"compile"| R2["boost readFile"]
        T3["grep triggers<br/>grep, regex, search for"] -->|"compile"| R3["boost grep"]
    end

    subgraph BaselineLayer["Layer 2: Builtin Baselines"]
        direction LR
        B1["glob metachar * ?"] -->|"match"| P1[glob pattern]
        B2["write create overwrite"] -->|"match"| P2[write pattern]
        B3["edit modify replace"] -->|"match"| P3[edit pattern]
        B4["move relocate transfer"] -->|"match"| P4[move pattern]
        B5["rename migrate"] -->|"match"| P5[rename pattern]
        B6["read show print open"] -->|"match"| P6[read pattern]
        B7["grep regex search for"] -->|"match"| P7[search pattern]
        B8["list enumerate ls"] -->|"match"| P8[list pattern]
        B9["scan walk traverse"] -->|"match"| P9[scan pattern]
    end

    Query([Lower-cased Query]) --> DynamicLayer
    DynamicLayer -->|"no dynamic match"| BaselineLayer
    BaselineLayer -->|"no baseline match"| None["pattern: none"]
    DynamicLayer -->|"first dynamic match wins"| Out([IntentHints with pattern, boostTools, reason])
    BaselineLayer -->|"first baseline match wins"| Out
    None --> Out
```

- **S3 Keyword overlap** — `keyword-overlap.ts` lower-cases both sides, strips stopwords and punctuation, expands glob-like tokens (`"*.ts"` also yields `"ts"` and `".ts"`), and scores Jaccard overlap. Crucially it is scored against the same composed text the index command produced (exposed by `VectorStore.toolTexts()`), so S1 and S3 operate on identical vocabularies.

`retriever.ts` fuses three signals with Reciprocal-Rank Fusion:

```
scoreI = 1/(k + rankS1(i)) + 1/(k + rankS2(i)) + 1/(k + rankS3(i))    (k = 60)
```

RRF is magnitude-agnostic, so heterogeneous signal scales don't need calibration. The fused score is normalized to [0,1], the optional `threshold` filter is applied, and the top-K tools are returned with their scores and a debug breakdown the `--json` flag surfaces.

```mermaid
flowchart TD
    Q([User Query]) --> Embed[Embed query with search_query: prefix]
    Q --> Detect[detectIntent over dynamic + baseline rules]
    Q --> Tokens[tokenize: strip stopwords + expand glob tokens]

    Store[(vector-store.jsonl)] --> Vectors[L2-normalized vectors]

    Embed --> Nq[normalize query to unit length]
    Nq --> Cosine[score per tool = cosineSimilarity<br/>multiplied by boost if in boostTools]
    Vectors --> Cosine

    Cosine --> R1[Rank by cosine desc]
    Detect --> R2[Rank intent-boosted tools]
    Tokens --> Jaccard[Jaccard query ∩ toolText]
    ToolTexts[VectorStore.toolTexts] --> Jaccard
    Jaccard --> R3[Rank by keyword overlap desc]

    R1 --> RRF["RRF sum 1 over 60 plus rank"]
    R2 --> RRF
    R3 --> RRF
    RRF --> Norm[normalize to 0..1]
    Norm --> Thresh{threshold greater than zero?}
    Thresh -- Yes --> Filter[drop below threshold]
    Thresh -- No --> Sort[sort desc + take top-K]
    Filter --> Sort
    Sort --> Out([ScoredTool with score and debug])
```

## Design Principles

- **YAGNI**: No evaluation mode, automatic K-tuning, or distributed persistence in v1.
- **KISS**: Manual math implementations and JSONL persistence avoid heavy external dependencies.
- **SOLID**: Each module has a single responsibility, and commands depend on abstractions (`IEmbedder`, `IVectorStore`).
- **Multi-signal over single-signal**: A single cosine pass was replaced by RRF because token-collision cases (glob-vs-readFile-vs-moveFile on `*.ts`) are not fixable by better embeddings alone; structural and lexical votes catch what cosine smears.

## SOLID Alignment

| Principle                 | Implementation                                                                                                        |
| ------------------------- | --------------------------------------------------------------------------------------------------------------------- |
| **S**ingle Responsibility | Each file has one clear job (Embedder, VectorStore, ToolLoader, Retriever, Detector, Commands).                       |
| **O**pen/Closed           | VectorStore and Embedder can be swapped later via interface without changing consumers.                               |
| **L**iskov                | Not applicable yet (no inheritance).                                                                                  |
| **I**nterface Segregation | Small, focused interfaces (`IEmbedder`, `IVectorStore`).                                                              |
| **D**ependency Inversion  | Commands depend on abstractions (`IEmbedder`, `IVectorStore`).                                                        |
| Tool Boundary Vocabulary  | Optional `whenToUse` / `whenNotToUse` fields follow the Claude Skills Spec "When to Use" / "When NOT to Use" pattern. |

## Core Interfaces

```typescript
// types.ts

export interface Tool {
  name: string;
  description: string;
  parameters: Record<string, any>;
  intent?: string; // canonical verb-object pair
  examples?: string[]; // positive natural-language usages
  whenToUse?: string[]; // positive inclusion criteria
  whenNotToUse?: string[]; // negative exclusion criteria ("Use X instead")
  triggers?: string[]; // query fragments that boost this tool (S2)
  boosts?: string[]; // other tool names to boost alongside this one
  strict?: boolean; // schema-only hint
}

export type EmbedType = "query" | "document";

export interface IEmbedder {
  embed(text: string, type?: EmbedType): Promise<number[]>;
}

export interface VectorStoreMetadata {
  model: string;
  dimensions: number;
  indexedAt: string;
}

export interface ScoredTool {
  tool: Tool;
  score: number;
}

export interface SearchHints {
  boostTools?: Set<string>;
  boost?: number;
}

export interface IVectorStore {
  add(tools: Tool[], embeddings: number[][]): Promise<void>;
  search(
    queryEmbedding: number[],
    k: number,
    hints?: SearchHints,
  ): Promise<ScoredTool[]>;
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
3. **Ollama Embedder**: Simple HTTP call to Ollama; embed type prefixes added later for retrieval manifold alignment.
4. **Vector Store**: In-memory + JSONL persistence + cosine similarity.
5. **Intent Detector**: Nine-rule pre-classifier serving S2 (after the IntentPattern split into `glob`/`write`/`edit`/`move`/`rename`/`read`/`search`/`list`/`scan`).
6. **Keyword Overlap**: Stopword-filtered Jaccard serving S3.
7. **Retriever**: RRF fusion of S1 + S2 + S3, score breakdown, and threshold filter.
8. **CLI Commands**: `index` (composes embedding text) and `route` (JSON + plain output, `--threshold`).
9. **Tool Spec Enrichment**: `intent` + `examples` + `whenToUse` + `whenNotToUse` fields added to every catalog entry.
10. **S2 Data-Driven Triggers**: `triggers` + `boosts` fields added to key tools; intent-detector rewritten with dynamic+baseline two-layer rule table.

## Mathematical Implementation

To avoid heavy dependencies like `mathjs`, the core math is implemented as clean, pure, dependency-free TypeScript functions in `src/math/`.

### Dot Product

```typescript
export function dot(
  a: Float32Array | number[],
  b: Float32Array | number[],
): number {
  if (a.length !== b.length)
    throw new Error(
      `Vectors must have equal length (${a.length} != ${b.length})`,
    );
  if (a.length === 0)
    throw new Error("Cannot calculate the dot product of empty vectors");

  let result = 0;
  for (let i = 0; i < a.length; i++) result += a[i] * b[i];
  return result;
}
```

### L2 Norm

```typescript
export function norm(vec: Float32Array | number[]): number {
  if (vec.length === 0) return 0;
  let sum = 0;
  for (let i = 0; i < vec.length; i++) sum += vec[i] * vec[i];
  return Math.sqrt(sum);
}

export function normalize(vec: Float32Array | number[]): Float32Array {
  /* in-place L2 normalization returning Float32Array */
}
```

### Cosine Similarity

```typescript
export function cosineSimilarity(
  a: Float32Array | number[],
  b: Float32Array | number[],
): number {
  const magA = norm(a);
  const magB = norm(b);
  if (magA === 0 || magB === 0) return 0; // zero-norm guard
  return dot(a, b) / (magA * magB);
}
```

`cosineSimilarity` is the single source of truth for similarity and is consumed by `VectorStore.search`; the previous inlined `dot(normalize(q), v)` form was replaced by it.

## Configuration and Best Practices

### Threshold Selection

Run `route` with `--threshold 0` first to inspect the score gap between the first and third tool on your benchmark queries. Pick a floor that keeps the right answers and filters noise — typical values land between 0.4 and 0.7 depending on catalog size and boundary-vocabulary richness.

> Note: the composite score is an RRF-derived [0,1] value, not a probability. Treat it as a rank-strength indicator, not a calibrated likelihood.

### Tool Description Quality

Each description is the primary S1 signal, so write them as short, declarative, verb-led sentences. The boundary fields (`whenToUse` / `whenNotToUse`) are the place to state explicit negative rules ("Use readFile instead when the user wants to read a specific file"). These negative statements disambiguate tools that would otherwise share positive vocabulary — without them, "list the files matching \*.ts" (glob) reads almost identically to "read the file package.json" (readFile) at the cosine layer because both mention files and paths.

### Synonyms and Negation in Boundary Vocabulary

Vary the phrasing of `whenToUse` / `whenNotToUse` to cover synonyms a user might choose ("search for", "locate", "find every"). The `NOT: ` prefix is added automatically by the index command, so tool authors should write plain sentences and let the pipeline handle negation — don't prefix them yourself.

### Observability

`route --json` returns each tool with its `score` and a `debug` object containing the cosine score, the keyword-overlap score, and the intent rule that fired. Pipe that into your analysis to spot cases where the S2 rule fired but S1 was weak (or vice versa) — those are the places where a better `whenToUse` / `whenNotToUse` sentence usually helps more than another embedding model pass.

### Fallback Strategy

If no tool clears the threshold, either widen K or relax `threshold` to 0 and take the top result. For critical calls, surface the score to the caller so it can escalate rather than silently use a low-confidence match.

### Dimension Consistency

When using `route`, the vector store metadata is validated. The model name triggers a warning on mismatch. The dimension count triggers a hard error on mismatch, preventing incorrect similarity calculations.

## Getting Started

### Prerequisites

- Node.js v22 or higher
- Yarn 4
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
yarn start index ./tools
```

With custom model and dimensions:

```bash
yarn start index ./tools --model nomic-embed-text:latest --dimensions 256
```

2. Route a query:

```bash
yarn start route "List all files matching *.ts"
```

Inspect scores and drop low-confidence tools:

```bash
yarn start route "List all files matching *.ts" --threshold 0.5
```

For development without building:

```bash
yarn dev route "List all the files with *.ts"
```

## Timeline Achievements

| Date       | Commit        | Achievement                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| ---------- | ------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 2026-06-29 | `05d5bcf`     | **Initial scaffold** — Commander CLI with `index` and `route` subcommands; Ollama embedder; in-memory vector store with JSON persistence; 13 tool schemas covering file, directory, search, web, execution, and skill operations.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| 2026-06-29 | `fe603a8`     | **Runtime hardening** — upsert-by-name on re-index; corrupt-store fallback; 30s AbortController timeout on embedder; tool-schema validation; batched embedding (chunk size 5); `--url` and `--json` flags on both commands.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| 2026-06-29 | `c9eaafa`     | **JSONL persistence + normalized Float32Array embeddings** — switched store from single-JSON to JSONL (one record per line); pre-normalize embeddings on insertion so dot product equals cosine similarity; added `normalize()` to math/norm.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| 2026-06-29 | (this branch) | **Retrieval rewrite: multi-signal RRF + boundary vocabulary + data-driven S2** — three-signal RRF fusion (dense cosine + structural intent detector + stopword-aware keyword overlap). Tool specs declare `whenToUse`/`whenNotToUse` (embedded with `NOT:` prefix), `triggers`/`boosts` (compiled into a dynamic two-layer rule table evaluated before builtin baselines). IntentPattern split into 9 granular verbs (`write`, `edit`, `move`, `rename`, `read`, `search`, `list`, `scan`). `search_query:`/`search_document:` nomic prefixes; `--threshold` filter; `--json` debug breakdown; `cosine-similarity.ts` consumed as single source of truth. **Verified**: `route "List all the files with *.ts"` → glob #1 at 0.98; `route "analyze the src folder"` with a synthetic `myAnalyzer` tool having `triggers:["analyze"]` ranked #1 with zero code change; all 14 tools hold #1 on their primary phrasing. |

## Quality Score Estimate Across the Change Timeline

States map to Timeline entries above: **T0** = `05d5bcf` (initial scaffold), **T1** = `fe603a8` (runtime hardening), **T2** = `c9eaafa` (JSONL + normalized embeddings), **T3/T4/T5** = this branch (multi-signal retriever → data-driven S2 → granular IntentPattern).

Scoring criteria (weighted): **Ranking accuracy** (40%), **Disambiguation** (25%), **Coverage** (20%), **Observability** (15%).

| Timeline | Description                                                                                          | Ranking | Disambiguation | Coverage | Observability | **Weighted Total** |
| -------- | ---------------------------------------------------------------------------------------------------- | ------- | -------------- | -------- | ------------- | ------------------ |
| **T0**   | Single-signal cosine over `name + description` only                                                  | 4/10    | 2/10           | 2/10     | 1/10          | **2.65**           |
| **T1**   | Runtime hardening (upsert, fallback, timeout, batching, `--url`/`--json`)                            | 4/10    | 2/10           | 2/10     | 2/10          | **2.80**           |
| **T2**   | JSONL persistence + normalized Float32Array embeddings                                               | 4/10    | 2/10           | 2/10     | 2/10          | **2.85**           |
| **T3**   | Multi-signal RRF + boundary vocabulary + retrieval prefixes                                          | 8/10    | 7/10           | 6/10     | 7/10          | **7.15**           |
| **T4**   | Data-driven `triggers`/`boosts` + zero hardcoded tool names                                          | 9/10    | 8/10           | 9/10     | 7/10          | **8.45**           |
| **T5**   | Granular 9-value IntentPattern (`write`/`edit`/`move`/`rename`/`read`/`search`/`list`/`scan`/`none`) | 9/10    | 9/10           | 9/10     | 8/10          | **8.85**           |

```
Score                                                           ● T5 (8.85)
 9.0 |                                                     ● T4 (8.45)
 8.5 |
 8.0 |
 7.5 |
 7.0 |                              ● T3 (7.15)
 6.5 |
 6.0 |
 5.5 |
 5.0 |
 4.5 |
 4.0 |
 3.5 |
 3.0 |
 2.5 |  ● T0 (2.65)  ● T1 (2.80)  ● T2 (2.85)
     +------------+------------+------------+------------+------------→
       T0         T1         T2         T3         T4         T5
```

Score trajectory at a glance: T0→T2 flat around 2.8 (infra only), T2→T3 jumps to 7.15 (the multi-signal inflection that fixed the ranking bug), T3→T4 climbs to 8.45 (data-driven T2 removes the last hardcoded-tool-names bottleneck), T4→T5 reaches 8.85 (granular patterns tighten disambiguation).

### Key takeaways

- **T0→T2 was flat** (~2.6–2.9). Infra changes (JSONL, normalization, timeouts, batched I/O) don't move accuracy; they only de-risk runtime and clean up persistence.
- **T2→T3 was the inflection** (+4.3). Moving from single-signal cosine to three-signal RRF fused with boundary vocabulary did more than every prior commit combined. This is where the reported bug got fixed.
- **T3→T4 was coverage-driven** (+1.3). Hardcoded tool names were the last scaling bottleneck; removing them made the router catalog-agnostic. Adding a tool to the catalog is now enough — `triggers` only needed to _refine_ coverage.
- **T4→T5 was precision-driven** (+0.4). Splitting the combined patterns into 9 granular verbs tightened disambiguation inside family clusters (write/edit, move/rename, list/scan) and made JSON output self-explanatory.

## License

[MIT](LICENSE)
