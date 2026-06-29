// Structural / intent pre-classifier (S2).
//
// Cheap, deterministic, zero-dependency rule layer that scans the raw query
// for surface markers of the user's *intent* and maps them to a set of tools
// that should be boosted in the dense ranking. This is not a replacement for
// the cosine pass — it catches cases where the embedding nearest-neighbour is
// ambiguous (e.g. "list all *.ts files" leaking into moveFile/readFile while
// glob sits at the right answer) by up-weighting candidates whose tool name
// the structural signal already predicts.
//
// The rule table has two layers:
//
//   1. Dynamic rules, built from each tool's optional `triggers` and `boosts`
//      fields. When a trigger matches, that tool plus every tool named in
//      `boosts` is bumped. These come FIRST so a tool-specific trigger always
//      wins over a generic verb.
//
//   2. Builtin baseline rules, kept as a safety net for catalogs that don't
//      declare triggers. They cover the universal file-operation verbs
//      (glob, write/edit, move/rename, read, search, list/scan) and boost
//      every tool in the catalog whose name or description matches the verb's
//      shape — so a brand-new tool called `archiveFiles` still gets the
//      move/rename bump even though nobody told us about it.
//
// Tools that declare neither `triggers` nor `boosts` still participate via
// the builtin layer; tools that declare only one of the two still work. The
// detector is fully catalog-aware and has zero hardcoded tool names.

export type IntentPattern =
  | 'glob'
  | 'write'
  | 'edit'
  | 'move'
  | 'rename'
  | 'read'
  | 'search'
  | 'list'
  | 'scan'
  | 'none';

export interface IntentHints {
  pattern: IntentPattern;
  // Tools whose cosine scores get multiplied by `boost` inside VectorStore.search.
  boostTools: Set<string>;
  boost: number;
  // Which rule fired — handy for --json debug output.
  reason?: string;
}

// Multiplicative boost applied to the cosine score of tools matched by the
// active intent. 1.15 nudges without drowning S1 — enough to rank glob ahead
// of near-ties on "list *.ts" but not enough to overtake a genuinely stronger
// match on a clean query.
const DefaultBoost = 1.15;

interface Rule {
  pattern: IntentPattern;
  // RegExp tested against the lower-cased query.
  match: RegExp;
  // Tool names to boost when this rule fires. Empty for builtin baselines,
  // which instead use `matchNames` / `matchDescription` to pick targets.
  boostTools: string[];
  // For builtin baselines: match a tool's name or description against these
  // regexes to decide which tools to boost. Lets a new tool ride the right
  // verb without anyone declaring it.
  matchNames?: RegExp;
  matchDescription?: RegExp;
  // Stable identifier for debug output.
  reason: string;
  // Length of the trigger string that produced this rule. Dynamic rules with
  // longer triggers are more specific; when several dynamic rules match the
  // same query we pick the longest (maximum-specificity) winner instead of
  // the first in catalog order. Baselines set this to 0.
  triggerLength: number;
}

import { Tool } from '../types.js';

// Builtin baselines — regex only, no hardcoded tool names. At rule-evaluation
// time we walk the catalog and boost any tool whose name or description
// matches `matchNames` / `matchDescription`. Order matters: the first rule
// that matches wins, so keep the most specific signals (glob, then
// destructive write/move) before broad verbs like read/list.
const BaselineRules: Rule[] = [
  {
    pattern: 'glob',
    match: /[*?]|\*\*|\[[^\]]*\]|\{[^}]*\}/,
    boostTools: [],
    // NOTE: trailing \b removed from matchNames so camelCase tool names like
    // `globFiles` or `wildcardMatch` still match. Descriptions are natural
    // language (real word boundaries), so matchDescription keeps \b.
    matchNames: /\b(glob|globber|wildcard|pattern)/i,
    matchDescription: /\b(glob|wildcard|pattern|filename pattern|file pattern)\b/i,
    reason: 'builtin:glob',
    triggerLength: 0,
  },
  {
    pattern: 'write',
    match: /\b(write|create|overwrite|new\s+file|make\s+a|make|generate\s+a)\b/,
    boostTools: [],
    matchNames: /\b(write|create|generate|save)/i,
    matchDescription: /\b(write|create|overwrite|generate|save)\b/i,
    reason: 'builtin:write',
    triggerLength: 0,
  },
  {
    pattern: 'edit',
    match: /\b(edit|modify|change|replace|patch|delete\s+file)\b/,
    boostTools: [],
    matchNames: /\b(edit|modify|patch|update|replace)/i,
    matchDescription: /\b(edit|modify|change|replace|patch|update)\b/i,
    reason: 'builtin:edit',
    triggerLength: 0,
  },
  {
    pattern: 'move',
    match: /\b(move|relocate|transfer|shift)\b.*(?:\.+[\\/]|\/)|(\bfrom\b|\bto\b).*\//,
    boostTools: [],
    matchNames: /\b(move|relocate|transfer|shift)/i,
    matchDescription: /\b(move|relocate|transfer|shift)\b/i,
    reason: 'builtin:move',
    triggerLength: 0,
  },
  {
    pattern: 'rename',
    match: /\b(rename|migrate)\b/,
    boostTools: [],
    matchNames: /\b(rename|renaming)/i,
    matchDescription: /\b(rename|renaming)\b/i,
    reason: 'builtin:rename',
    triggerLength: 0,
  },
  {
    pattern: 'search',
    match: /\b(grep|regex|search\s+for|find\s+keyword|match\s+pattern|find\s+every|look\s+for\s+the\s+string|look\s+for)\b/,
    boostTools: [],
    // matchNames: kept tight — only tools whose NAME is genuinely search-family.
    // Generic verbs like "find"/"match"/"query" appear in many non-search tool
    // names (e.g. "findFiles") and cause cross-contamination.
    matchNames: /\b(grep|search|locate|lookup)/i,
    // matchDescription: "regex" and "search" are the reliable signals.
    // "find"/"match"/"query" are too broad — editFile's description mentions
    // "Regex" (it supports regex replacement), which wrongly pulled editFile
    // into the search cluster for queries like "find every place where...".
    matchDescription: /\b(search|grep|locate|lookup)\b/i,
    reason: 'builtin:search',
    triggerLength: 0,
  },
  {
    pattern: 'read',
    match: /\b(read|show\s+(?:me\s+)?(?:the\s+)?(?:content|contents|text|body)|print\s+(?:the\s+)?(?:file|contents?)|open\s+(?:the\s+)?(?:file|document)|open\s+and\s+display|display\s+(?:the\s+)?(?:file|document|contents?))\b/,
    boostTools: [],
    matchNames: /\b(read|open|show|print|display|cat|head|tail|dump)/i,
    matchDescription: /\b(read|show|print|display|open|view|dump)\b/i,
    reason: 'builtin:read',
    triggerLength: 0,
  },
  {
    pattern: 'list',
    match: /\b(list|enumerate|ls|directory listing|folder listing)\b/,
    boostTools: [],
    matchNames: /\b(list|enumerate|ls|dir)/i,
    matchDescription: /\b(list|enumerate|directory listing|folder listing)\b/i,
    reason: 'builtin:list',
    triggerLength: 0,
  },
  {
    pattern: 'scan',
    match: /\b(scan|walk|traverse|crawl|catalog|index)\b/,
    boostTools: [],
    matchNames: /\b(scan|walk|traverse|crawl|catalog|index)/i,
    matchDescription: /\b(scan|walk|traverse|crawl|catalog|index)\b/i,
    reason: 'builtin:scan',
    triggerLength: 0,
  },
];

// Compiles the dynamic rule layer from a tool's `triggers` and `boosts`.
// Each trigger becomes one Rule whose `boostTools` is the tool itself plus
// any names listed in `boosts`. The trigger string is treated as a literal
// substring match (escaped) unless it contains regex metacharacters, in
// which case it is compiled as-is so advanced users can pass real regexes.
function buildDynamicRules(tools: Tool[]): Rule[] {
  const rules: Rule[] = [];
  for (const tool of tools) {
    if (!tool.triggers || tool.triggers.length === 0) continue;
    const boosts = tool.boosts ?? [];
    const boostTools = [tool.name, ...boosts];
    for (const raw of tool.triggers) {
      const pattern = compileTrigger(raw);
      rules.push({
        pattern: 'none',
        match: pattern,
        boostTools,
        reason: `dynamic:${tool.name}:${raw}`,
        triggerLength: raw.length,
      });
    }
  }
  return rules;
}

// Decides whether a trigger string is a literal substring or a regex.
// If it contains characters that are special in regex (other than letters,
// digits, whitespace, and a few safe punctuation), compile it as regex.
// Otherwise escape it for a literal substring match.
function compileTrigger(raw: string): RegExp {
  const needsRegex = /[^a-zA-Z0-9\s._-]/.test(raw);
  const body = needsRegex ? raw : escapeRegex(raw);
  return new RegExp(body, 'i');
}

function escapeRegex(s: string): string {
  return s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

// Builds the full rule table for a catalog: dynamic rules first (so a
// tool-specific trigger always wins), then builtin baselines as a safety net.
// The result is cached on the detector so repeated queries against the same
// catalog don't recompile.
let cachedToolsKey: string | null = null;
let cachedRules: Rule[] | null = null;

function buildRuleTable(tools: Tool[]): Rule[] {
  // Cheap invalidation key: tool names joined. Good enough because triggers
  // and boosts live inside the tool JSON, so any catalog change alters names
  // or the user re-indexes (which reloads the store).
  const key = tools.map((t) => t.name).join('|');
  if (key === cachedToolsKey && cachedRules) return cachedRules;

  const dynamic = buildDynamicRules(tools);
  const table = [...dynamic, ...BaselineRules];
  cachedToolsKey = key;
  cachedRules = table;
  return table;
}

// Resets the rule-table cache. Exposed for tests and for callers that swap
// the catalog at runtime.
export function resetIntentCache(): void {
  cachedToolsKey = null;
  cachedRules = null;
}

// Resolves a matching rule into the set of tools to boost, or null if the
// rule has no effect against this catalog (builtin baselines with no matching
// tools produce an empty boost set and are skipped).
function resolveBoostTools(rule: Rule, tools: Tool[]): Set<string> | null {
  if (rule.boostTools.length > 0) {
    return new Set(rule.boostTools);
  }
  const boosted = new Set<string>();
  const nameRe = rule.matchNames;
  const descRe = rule.matchDescription;
  for (const tool of tools) {
    if (nameRe && nameRe.test(tool.name)) boosted.add(tool.name);
    if (descRe && descRe.test(tool.description)) boosted.add(tool.name);
  }
  return boosted.size > 0 ? boosted : null;
}

// Main entry point. `tools` is the live catalog from the store. Among all
// rules that match the query and produce a non-empty boost set, the one with
// the LONGEST trigger wins (maximum-specificity ordering). This prevents an
// early-catalog tool with a broad trigger ("change the file") from shadowing a
// later tool with a more specific trigger ("change the file extension") for
// the same query. Builtin baselines (triggerLength 0) only fire when no
// dynamic rule matched.
export function detectIntent(query: string, tools: Tool[]): IntentHints {
  const q = query.toLowerCase();
  const rules = buildRuleTable(tools);

  let bestRule: Rule | null = null;
  let bestBoost: Set<string> | null = null;

  for (const rule of rules) {
    if (!rule.match.test(q)) continue;
    const boost = resolveBoostTools(rule, tools);
    if (!boost) continue;
    // Prefer the rule with the longest trigger. On a tie, the first in
    // catalog order wins (stable).
    if (!bestRule || rule.triggerLength > bestRule.triggerLength) {
      bestRule = rule;
      bestBoost = boost;
    }
  }

  if (!bestRule) {
    return { pattern: 'none', boostTools: new Set(), boost: 1 };
  }

  return {
    pattern: bestRule.pattern,
    boostTools: bestBoost!,
    boost: DefaultBoost,
    reason: bestRule.reason,
  };
}

// Kept for backward-compat with any caller that still uses the no-catalog
// signature. Falls back to the builtin baselines only.
export function detectIntentLegacy(query: string): IntentHints {
  return detectIntent(query, []);
}

// Re-export so the retriever can build the table without re-importing.
export type { Rule };
