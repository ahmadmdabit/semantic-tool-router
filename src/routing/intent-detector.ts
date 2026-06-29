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
    matchNames: /\b(glob|globber|wildcard|pattern)\b/i,
    matchDescription: /\b(glob|wildcard|pattern|filename pattern|file pattern)\b/i,
    reason: 'builtin:glob',
  },
  {
    pattern: 'write',
    match: /\b(write|create|overwrite|new\s+file)\b/,
    boostTools: [],
    matchNames: /\b(write|create|generate|save)\b/i,
    matchDescription: /\b(write|create|overwrite|generate|save)\b/i,
    reason: 'builtin:write',
  },
  {
    pattern: 'edit',
    match: /\b(edit|modify|change|replace|patch|delete\s+file)\b/,
    boostTools: [],
    matchNames: /\b(edit|modify|patch|update|replace)\b/i,
    matchDescription: /\b(edit|modify|change|replace|patch|update)\b/i,
    reason: 'builtin:edit',
  },
  {
    pattern: 'move',
    match: /\b(move|relocate|transfer|shift)\b.*(?:\.+[\\/]|\/)|(\bfrom\b|\bto\b).*\//,
    boostTools: [],
    matchNames: /\b(move|relocate|transfer|shift)\b/i,
    matchDescription: /\b(move|relocate|transfer|shift)\b/i,
    reason: 'builtin:move',
  },
  {
    pattern: 'rename',
    match: /\b(rename|migrate)\b/,
    boostTools: [],
    matchNames: /\b(rename|renaming)\b/i,
    matchDescription: /\b(rename|renaming)\b/i,
    reason: 'builtin:rename',
  },
  {
    pattern: 'search',
    match: /\b(grep|regex|search\s+for|find\s+keyword|match\s+pattern)\b/,
    boostTools: [],
    matchNames: /\b(grep|search|find|locate|lookup|query|match)\b/i,
    matchDescription: /\b(search|grep|regex|find|locate|match|query|lookup)\b/i,
    reason: 'builtin:search',
  },
  {
    pattern: 'read',
    match: /\b(read|show\s+(?:me\s+)?(?:the\s+)?(?:content|contents|text|body)|print\s+(?:the\s+)?(?:file|contents?)|open\s+(?:the\s+)?(?:file|document))\b/,
    boostTools: [],
    matchNames: /\b(read|open|show|print|display|cat|head|tail|dump)\b/i,
    matchDescription: /\b(read|show|print|display|open|view|dump)\b/i,
    reason: 'builtin:read',
  },
  {
    pattern: 'list',
    match: /\b(list|enumerate|ls|directory listing|folder listing)\b/,
    boostTools: [],
    matchNames: /\b(list|enumerate|ls|dir)\b/i,
    matchDescription: /\b(list|enumerate|directory listing|folder listing)\b/i,
    reason: 'builtin:list',
  },
  {
    pattern: 'scan',
    match: /\b(scan|walk|traverse|crawl|catalog|index)\b/,
    boostTools: [],
    matchNames: /\b(scan|walk|traverse|crawl|catalog|index)\b/i,
    matchDescription: /\b(scan|walk|traverse|crawl|catalog|index)\b/i,
    reason: 'builtin:scan',
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

// Main entry point. `tools` is the live catalog from the store. Returns the
// first matching rule's hints (dynamic wins over builtin), or a no-op hint
// set when nothing fires.
export function detectIntent(query: string, tools: Tool[]): IntentHints {
  const q = query.toLowerCase();
  const rules = buildRuleTable(tools);

  for (const rule of rules) {
    if (!rule.match.test(q)) continue;

    let boostTools: Set<string>;
    if (rule.boostTools.length > 0) {
      // Dynamic rule: boost the declared tools.
      boostTools = new Set(rule.boostTools);
    } else {
      // Builtin baseline: walk the catalog and boost any tool whose name or
      // description matches the verb's shape.
      boostTools = new Set<string>();
      const nameRe = rule.matchNames;
      const descRe = rule.matchDescription;
      for (const tool of tools) {
        if (nameRe && nameRe.test(tool.name)) boostTools.add(tool.name);
        if (descRe && descRe.test(tool.description)) boostTools.add(tool.name);
      }
    }

    if (boostTools.size === 0) continue;

    return {
      pattern: rule.pattern,
      boostTools,
      boost: DefaultBoost,
      reason: rule.reason,
    };
  }

  return { pattern: 'none', boostTools: new Set(), boost: 1 };
}

// Kept for backward-compat with any caller that still uses the no-catalog
// signature. Falls back to the builtin baselines only.
export function detectIntentLegacy(query: string): IntentHints {
  return detectIntent(query, []);
}

// Re-export so the retriever can build the table without re-importing.
export type { Rule };
