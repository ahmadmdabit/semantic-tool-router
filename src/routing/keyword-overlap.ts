// Keyword surface-overlap scorer (S3).
//
// Complements the dense cosine pass with cheap lexical evidence. The useful
// tokens for tool retrieval are the technical, domain, and
// identifier-like ones ("glob", "*.ts", "src", "regex", "write"), so we
// strip stopwords and punctuation and compute a Jaccard between the remaining
// query and tool tokens. This catches shared terms the cosine pass can lose
// when a query's intent collides with many tool descriptions ("list" matches
// several tools before any would match "glob").

import { Tool } from '../types.js';

// Conservative English stopword list plus verbs that dominate
// user-prompt phrasing but say little about tool choice. Tuned to queries
// like "find all *.ts files in src/", where the signal tokens are
// {find, *.ts, files, src} after dropping noisy words.
const STOPWORDS = new Set([
  'a', 'an', 'the', 'and', 'or', 'not', 'no',
  'is', 'are', 'was', 'were', 'be', 'been', 'being',
  'am', 'do', 'does', 'did', 'have', 'has', 'had',
  'i', 'me', 'my', 'mine', 'we', 'us', 'our', 'you', 'your', 'he', 'she', 'it', 'its', 'they', 'them', 'their',
  'this', 'that', 'these', 'those', 'there', 'here', 'what', 'which', 'who', 'whom', 'how', 'when', 'where', 'why',
  'to', 'of', 'in', 'on', 'at', 'by', 'from', 'with', 'as', 'into', 'about', 'for', 'per',
  'can', 'could', 'will', 'would', 'shall', 'should', 'may', 'might',
  'please', 'the', 'just', 'also', 'any', 'all', 'every', 'each',
  'me', 'very', 'on', 'up', 'out', 'over',
  'a', 's', 'd', 'll', 're', 've',
  // common user verbs that carry little information
  'want', 'make', 'made', 'let', 'use', 'run', 'get', 'got', 'give', 'tell',
  'need', 'know', 'keep', 'put', 'see', 'set', 'try', 'look', 'going',
]);

// Public entry points ----------------------------------------------------------------

// Scores a full tool: composes its canonical text (name, description,
// param descriptions, intent, examples, whenToUse, whenNotToUse) and
// computes Jaccard overlap against the query. Used in tests and ad-hoc
// callers where the store-level helper isn't in scope.
export function keywordScore(query: string, tool: Tool): number {
  return keywordScoreText(query, collectToolText(tool));
}

// Scores a pre-composed tool text. Used by the retriever so S1 and S3
// operate over identical vocabulary (same text that was embedded at
// index time).
export function keywordScoreText(query: string, toolText: string): number {
  return keywordScoreTokens(query, tokenize(toolText));
}

// Lowerscases on input; keeps technical substrings like "*.ts" intact rather
// than stripping them into meaningless punctuation.
function tokenize(text: string): Set<string> {
  const tokens = new Set<string>();
  for (const raw of text.toLowerCase().split(/[\s,;:()]+/)) {
    if (!raw) continue;
    const t = raw.replace(/[\'\"]/g, '').replace(/^[\W_]+|[\W_]+$/g, '');
    if (!t || STOPWORDS.has(t)) continue;
    tokens.add(t);

    // Expand "*.ts" -> also index {*.ts, ts, .ts} so keyword overlap still
    // catches when the query is "list *.ts files" and the tool side embeds
    // an example like "Glob pattern with e.g. '*.ts'".
    if (/^[*?]/.test(t) || /^.{0,4}\.[a-z0-9]+$/i.test(t)) {
      const ext = t.replace(/^[*?]+\./, '').replace(/^\./, '');
      if (ext && ext !== t) tokens.add(ext);
    }
  }
  return tokens;
}

function collectToolText(tool: Tool): string {
  // Same fields the index command embeds so S1 and S3 operate over
  // consistent vocabularies.
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
  return parts.filter(Boolean).join('. ');
}

// Jaccard overlap of the token sets, normalized to [0,1].
function keywordScoreTokens(query: string, toolTokens: Set<string>): number {
  const q = tokenize(query);
  const t = toolTokens;
  if (q.size === 0 || t.size === 0) return 0;

  let intersection = 0;
  for (const tok of q) {
    if (t.has(tok)) intersection += 1;
  }
  // Use the smaller set in the denominator so surface-level tools (few
  // description words) don't get punished against the richer queries.
  const denom = Math.min(q.size, t.size);
  return intersection / denom;
}
