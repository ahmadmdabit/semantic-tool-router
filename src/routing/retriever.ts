// Retriever: fuses S1 (dense cosine), S2 (structural intent), and S3 (keyword
// overlap) into a single ranked list via Reciprocal-Rank Fusion.
//
// RRF is used instead of weighted linear fusion because the three signals
// operate on completely different scales (cosine on [0,1], normalized boost
// multipliers, Jaccard on [0,1]) and RRF is robust to those calibration
// differences. scoreI = ΣSignals 1/(k + rankSignalI) where k=60 is the
// typical dampening constant.
//
// The retriever is the single call site the CLI uses to rank tools; it owns
// the loop reading the store, calling the detector, calling the embedder, and
// combining results.

import { Tool, IEmbedder, EmbedType, ScoredTool, SearchHints } from '../types.js';
import { VectorStore } from '../vector/vector-store.js';
import { detectIntent, IntentHints } from './intent-detector.js';
import { keywordScoreText } from './keyword-overlap.js';

// Should match the dampening constant used in RRF literature.
const RRFk = 60;
// Max number of candidates to pull from the dense pass before fusing. Larger
// than the final `k` so S2 boosts can still rescue a tool that's outside the
// raw top-k.
const CandidateMultiplier = 2;

export interface RetrieveOptions {
  // Final number of tools to return. Defaults to the route command's --top-k.
  k?: number;
  // Drop tools with composite score below this floor. 0 disables the filter.
  threshold?: number;
}

export interface RetrievedTool extends ScoredTool {
  // Breakdown of how each signal scored this tool — printed by the CLI when
  // --threshold or --json exposes the composite.
  debug?: {
    cosine?: number;
    keyword?: number;
    intent?: IntentHints['pattern'];
  };
}

interface SignalContribution {
  rank: number;
}

// Normalize the raw RRF sum to [0,1] for display. The theoretical maximum is
// 3/RRFk (a tool ranked #1 by all three signals), so we divide by that to
// get a relatable 0..1 figure.
function normalizeRRF(rrf: number): number {
  return Math.min(1, rrf / (3 / RRFk));
}

export async function retrieve(
  query: string,
  embedder: IEmbedder,
  store: VectorStore,
  options: RetrieveOptions = {},
): Promise<RetrievedTool[]> {
  const k = options.k ?? 5;
  const threshold = options.threshold ?? 0;
  const candidateK = Math.max(k * CandidateMultiplier, k + 5);

  // Catalog is needed by S2 (intent detector) and S3 (keyword overlap) —
  // pull it once and reuse.
  const tools = store.tools();
  if (tools.length === 0) return [];

  // S2: structural pre-classification. Pass the live catalog so the detector
  // can build its rule table from each tool's `triggers` / `boosts` fields
  // instead of relying on hardcoded tool names.
  const hints: IntentHints = detectIntent(query, tools);
  const searchHints: SearchHints | undefined = hints.boostTools.size
    ? { boostTools: hints.boostTools, boost: hints.boost }
    : undefined;

  // S1: dense cosine pass with S2 boost baked into the vector store.
  const queryEmbedType: EmbedType = 'query';
  const queryEmbedding = await embedder.embed(query, queryEmbedType);
  const denseScored: ScoredTool[] = await store.search(queryEmbedding, candidateK, searchHints);

  const toolTexts = store.toolTexts();

  // S3: lexical overlap scored against the canonical per-tool text — same
  // composition (name + description + params + intent + examples +
  // whenToUse + whenNotToUse) that the dense pass indexed, so S1 and S3
  // operate over consistent vocabularies. S3 catches surface matches the
  // cosine pass smears, e.g. the "*.ts" token in glob's description.
  const keywordByTool = new Map<string, number>();
  for (const tool of tools) {
    const text = toolTexts.get(tool.name) ?? '';
    keywordByTool.set(tool.name, keywordScoreText(query, text));
  }

  // Convert each signal into a 1-based rank. RRF cares only about relative
  // order, so we sort each signal descending and record position.
  const cosineRanks = new Map<string, SignalContribution>();
  {
    let rank = 1;
    for (const s of denseScored) cosineRanks.set(s.tool.name, { rank: rank++ });
  }

  const keywordRanks = new Map<string, SignalContribution>(rankMap(keywordByTool));

  // Dense scoring recovers best from S1, but when the S2 classifier fires
  // ("glob", "write", "edit", …) we also let that signal vote directly so a tool
  // the feature clearly points to gets a small boost even if cosine places it
  // just outside the candidate window.
  const intentRanks = new Map<string, SignalContribution>();
  if (hints.boostTools.size > 0) {
    let rank = 1;
    for (const name of hints.boostTools) intentRanks.set(name, { rank: rank++ });
  }

  // Fuse with RRF.
  const fused = new Map<string, RetrievedTool>();
  for (const tool of tools) {
    const cosine = cosineRanks.get(tool.name);
    const keyword = keywordRanks.get(tool.name);
    const intent = intentRanks.get(tool.name);

    let rrf = 0;
    if (cosine) rrf += 1 / (RRFk + cosine.rank);
    if (keyword) rrf += 1 / (RRFk + keyword.rank);
    if (intent) rrf += 1 / (RRFk + intent.rank);

    fused.set(tool.name, {
      tool,
      score: normalizeRRF(rrf),
      debug: {
        cosine: denseScored.find((s) => s.tool.name === tool.name)?.score,
        keyword: keywordByTool.get(tool.name),
        intent: intent ? hints.pattern : 'none',
      },
    });
  }

  // Sort by fused score descending, apply threshold, trim to k.
  const all = [...fused.values()].sort((a, b) => b.score - a.score);
  const aboveThreshold = threshold > 0
    ? all.filter((s) => s.score >= threshold)
    : all;

  return aboveThreshold.slice(0, k);
}

// Helper: sort a map by value descending and produce a 1-based rank map.
function rankMap(m: Map<string, number>): Map<string, SignalContribution> {
  const entries = [...m.entries()].sort((a, b) => b[1] - a[1]);
  const ranks = new Map<string, SignalContribution>();
  let rank = 1;
  for (const [name] of entries) ranks.set(name, { rank: rank++ });
  return ranks;
}
