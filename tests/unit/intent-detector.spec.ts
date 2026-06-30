import { describe, it, expect, beforeEach } from 'vitest';
import {
  detectIntent,
  detectIntentLegacy,
  resetIntentCache,
  IntentPattern,
} from '../../src/routing/intent-detector.js';
import { Tool } from '../../src/types.js';

// Minimal catalog representing each verb family. Tool NAMES embed the verb so
// the builtin baselines (which match on name-or-description shape) have a
// target to boost. `triggers` are wired so we can also exercise the dynamic
// rule layer (Layer 1) from the same fixture.
//
// IMPORTANT: a builtin baseline only "fires" (returns a non-none pattern) if
// the catalog contains at least one tool whose name or description matches the
// verb's shape regex. A catalog of generic names would make every baseline
// return 'none' — so each tool here carries its verb in its name.
const CATALOG: Tool[] = [
  {
    name: 'glob',
    description: 'Find files matching a glob pattern.',
    parameters: { type: 'object', properties: {} },
    triggers: ['find files', 'locate files', 'glob'],
    boosts: ['grep'],
  },
  {
    name: 'grep',
    description: 'Search for a regex pattern within files.',
    parameters: { type: 'object', properties: {} },
    triggers: ['grep', 'regex'],
    boosts: [],
  },
  {
    name: 'readFile',
    description: 'Read the content of a text file.',
    parameters: { type: 'object', properties: {} },
    triggers: ['read file', 'print contents'],
    boosts: [],
  },
  {
    name: 'writeFile',
    description: 'Write content to a file.',
    parameters: { type: 'object', properties: {} },
    triggers: ['write file', 'create file'],
    boosts: ['editFile'],
  },
  {
    name: 'editFile',
    description: 'Apply a targeted text replacement.',
    parameters: { type: 'object', properties: {} },
    triggers: ['edit file', 'replace text'],
    boosts: ['writeFile'],
  },
  {
    name: 'moveFile',
    description: 'Move a file to a new path.',
    parameters: { type: 'object', properties: {} },
    triggers: ['move file'],
    boosts: ['renameFile'],
  },
  {
    name: 'renameFile',
    description: 'Rename an existing file.',
    parameters: { type: 'object', properties: {} },
    triggers: ['rename file'],
    boosts: ['moveFile'],
  },
  {
    name: 'listDirectory',
    description: 'List files and directories in a given path.',
    parameters: { type: 'object', properties: {} },
    triggers: ['list files'],
    boosts: ['scanDirectory'],
  },
  {
    name: 'scanDirectory',
    description: 'Scan directory for text files.',
    parameters: { type: 'object', properties: {} },
    triggers: ['scan directory'],
    boosts: ['listDirectory'],
  },
  {
    name: 'useSkill',
    description: 'Load a skill file.',
    parameters: { type: 'object', properties: {} },
    // Deliberately no triggers — this tool only gets coverage via builtin baselines.
    triggers: [],
    boosts: [],
  },
];

// A second catalog with overlapping names but DIFFERENT triggers, used to prove
// resetIntentCache() actually rebuilds the rule table.
const OTHER_CATALOG: Tool[] = [
  {
    name: 'myAnalyzer',
    description: 'Analyze a folder.',
    parameters: { type: 'object', properties: {} },
    triggers: ['analyze'],
    boosts: [],
  },
];

beforeEach(() => {
  resetIntentCache();
});

describe('detectIntent — dynamic rules (Layer 1)', () => {
  it('fires a trigger and boosts the declaring tool', () => {
    const hints = detectIntent('locate files matching src/**/*.ts', CATALOG);
    expect(hints.boostTools.has('glob')).toBe(true);
    expect(hints.pattern).toBe('none'); // dynamic rules carry pattern 'none'
    expect(hints.reason).toContain('dynamic:glob');
  });

  it('boosts every tool listed in `boosts` when a trigger fires', () => {
    const hints = detectIntent('glob pattern', CATALOG);
    expect(hints.boostTools.has('glob')).toBe(true);
    expect(hints.boostTools.has('grep')).toBe(true); // glob.boosts = ['grep']
    expect(hints.boostTools.size).toBe(2);
  });

  it('treats a trigger as a literal substring when it has no metacharacters', () => {
    const hints = detectIntent('something with regex inside', CATALOG);
    expect(hints.boostTools.has('grep')).toBe(true);
  });

  it('stops at the FIRST matching trigger (order = catalog order)', () => {
    // "grep for files" — glob's triggers are ['find files','locate files','glob']:
    // none match this query. grep's 'grep' trigger matches, so grep wins.
    const hints = detectIntent('grep for files', CATALOG);
    expect(hints.reason).toContain('dynamic:grep');
  });

  it('compiles a trigger as a regex when it contains metacharacters', () => {
    // A trigger with regex metacharacters (e.g. a dot) must be compiled as a
    // real regex, not escaped as a literal substring. This exercises the
    // compileTrigger branch where needsRegex is true. Note: the trigger must
    // still be a VALID regex — "file.ts" contains a metacharacter (.) and
    // matches both "file.ts" and "file-ts" via the regex dot.
    const regexCatalog: Tool[] = [
      {
        name: 'grep',
        description: 'search files',
        parameters: { type: 'object', properties: {} },
        triggers: ['file.ts'], // the dot is a regex metacharacter
        boosts: [],
      },
    ];
    resetIntentCache();
    // "find file.ts usage" contains "file.ts" — the dot in the trigger regex
    // matches the literal dot in the query.
    const hints = detectIntent('find file.ts usage', regexCatalog);
    expect(hints.boostTools.has('grep')).toBe(true);
    expect(hints.reason).toContain('dynamic:grep:file.ts');
  });

  it('picks the longest trigger when two tools match the same query', () => {
    // Both tools match "change the file extension" but renameFile's trigger is
    // longer. Longest-trigger-wins must select renameFile.
    const catalog: Tool[] = [
      {
        name: 'editFile',
        description: 'edit',
        parameters: { type: 'object', properties: {} },
        triggers: ['change the file'], // length 14
        boosts: [],
      },
      {
        name: 'renameFile',
        description: 'rename',
        parameters: { type: 'object', properties: {} },
        triggers: ['change the file extension'], // length 24
        boosts: [],
      },
    ];
    resetIntentCache();
    const hints = detectIntent('change the file extension from .js to .ts', catalog);
    expect(hints.boostTools.has('renameFile')).toBe(true);
    expect(hints.reason).toContain('dynamic:renameFile');
  });
});

describe('detectIntent — builtin baselines (Layer 2)', () => {
  it('fires the glob baseline on wildcard metacharacters', () => {
    const hints = detectIntent('list *.ts files', CATALOG);
    expect(hints.pattern).toBe('glob');
    expect(hints.reason).toContain('builtin:glob');
  });

  it('fires the write baseline on create-verb phrasing', () => {
    const hints = detectIntent('create a new file', CATALOG);
    expect(hints.pattern).toBe('write');
  });

  it('fires the read baseline on read-verb phrasing', () => {
    const hints = detectIntent('read the readme', CATALOG);
    expect(hints.pattern).toBe('read');
  });

  it('fires the move baseline on move-verb + path-shape phrasing', () => {
    // The move baseline regex requires either a path extension or a from/to
    // construction — "move the folder to /tmp" satisfies the path-shape arm.
    const hints = detectIntent('move the folder to /tmp', CATALOG);
    expect(hints.pattern).toBe('move');
  });

  it('returns none + empty boostTools when nothing matches', () => {
    const hints = detectIntent('tell me a joke', CATALOG);
    expect(hints.pattern).toBe('none');
    expect(hints.boostTools.size).toBe(0);
    expect(hints.boost).toBe(1);
  });
});

describe('detectIntent — layer priority', () => {
  it('a dynamic trigger wins over a builtin baseline even if both could fire', () => {
    // "print contents of the file" matches readFile's 'print contents'
    // dynamic trigger (Layer 1) AND the read builtin baseline (Layer 2) via
    // "print ... contents". The dynamic rule must win.
    const hints = detectIntent('print contents of the file', CATALOG);
    expect(hints.reason).toContain('dynamic:readFile');
    expect(hints.pattern).toBe('none'); // dynamic rules are tagged 'none'
  });
});

describe('detectIntentLegacy', () => {
  it('falls back to builtin baselines only (no dynamic rules), but still needs a catalog to boost', () => {
    // detectIntentLegacy passes an EMPTY catalog. The edit baseline regex
    // matches "edit", but with no catalog tools to walk, boostTools is empty
    // and the rule is skipped -> pattern falls through to 'none'.
    // This documents the legacy limitation: it cannot boost without a catalog.
    const hints = detectIntentLegacy('edit the readme');
    expect(hints.pattern).toBe('none');
    expect(hints.boostTools.size).toBe(0);
  });

  it('returns none for an unrecognized query', () => {
    expect(detectIntentLegacy('unrecognized').pattern).toBe('none');
  });
});

describe('resetIntentCache', () => {
  it('isolates two catalogs called in sequence within the same process', () => {
    const h1 = detectIntent('analyze the folder', OTHER_CATALOG);
    expect(h1.boostTools.has('myAnalyzer')).toBe(true);

    // Without a reset, a naive cache keyed only on names could reuse the old
    // rule table. The call below must rebuild.
    resetIntentCache();
    const h2 = detectIntent('locate files in src', CATALOG);
    expect(h2.boostTools.has('glob')).toBe(true);
  });

  it('rebuilds an equivalent table after reset (no stale entries)', () => {
    detectIntent('analyze', OTHER_CATALOG);
    resetIntentCache();
    // 'analyze' is NOT a trigger in CATALOG, so it should no longer match.
    const h = detectIntent('analyze', CATALOG);
    expect(h.pattern).toBe('none');
    expect(h.boostTools.size).toBe(0);
  });
});

describe('IntentPattern contract', () => {
  it('every baseline exposes a known pattern value', () => {
    // Each query is chosen to (a) match the corresponding builtin baseline
    // regex and (b) have a target tool in CATALOG whose name/description
    // matches the verb's shape. Queries that would match a dynamic trigger
    // are avoided so the builtin layer is what fires.
    const patterns: IntentPattern[] = [
      detectIntent('*.ts', CATALOG).pattern,             // glob: metachar
      detectIntent('overwrite output.log', CATALOG).pattern, // write: overwrite + path
      detectIntent('patch the file', CATALOG).pattern,   // edit: patch
      detectIntent('shift dir to /x', CATALOG).pattern,  // move: shift + path
      detectIntent('migrate the component', CATALOG).pattern, // rename: migrate
      detectIntent('open the document', CATALOG).pattern, // read: open + document
      detectIntent('match pattern', CATALOG).pattern,    // search: match pattern
      detectIntent('enumerate the items', CATALOG).pattern, // list: enumerate
      detectIntent('crawl the tree', CATALOG).pattern,   // scan: crawl
      detectIntent('nope', CATALOG).pattern,             // none
    ];
    const expected: IntentPattern[] = [
      'glob', 'write', 'edit', 'move', 'rename', 'read', 'search', 'list', 'scan', 'none',
    ];
    expect(patterns).toEqual(expected);
  });
});
