import { describe, it, expect } from 'vitest';
import { keywordScore, keywordScoreText } from '../../src/routing/keyword-overlap.js';
import { Tool } from '../../src/types.js';

// A small synthetic tool whose canonical text we control tightly.
function makeTool(name: string, description: string, extra: Partial<Tool> = {}): Tool {
  return {
    name,
    description,
    parameters: { type: 'object', properties: {} },
    ...extra,
  };
}

const GLOB_TOOL = makeTool('glob', 'Find files matching a glob pattern.', {
  examples: ['Find all .ts files in src', 'Locate files matching *.json'],
  whenToUse: ['The query contains a glob pattern'],
  whenNotToUse: ['The user wants to read a specific file'],
});

describe('keywordScoreText', () => {
  it('returns 1.0 when the query shares all tokens with the tool text', () => {
    // "glob pattern" — both tokens appear in GLOB_TOOL's canonical text.
    const score = keywordScoreText('glob pattern', 'glob pattern find files');
    expect(score).toBeCloseTo(1, 5);
  });

  it('returns 0.0 when the query shares no tokens with the tool text', () => {
    const score = keywordScoreText('quantum entanglement', GLOB_TOOL.description);
    expect(score).toBe(0);
  });

  it('returns a value in (0,1) for partial overlap', () => {
    // "find the files matching" vs "find files matching a glob pattern":
    // query tokens {find, files, matching}, tool tokens {find, files, matching, glob, pattern}.
    // intersection=3, denom = min(3, 5) = 3 -> 1.0 ... use a deeper mismatch instead.
    const score = keywordScoreText('glob regex', 'glob pattern find files');
    // query tokens {glob, regex}, tool tokens {glob, pattern, find, files}.
    // intersection=1 (glob), denom = min(2,4) = 2 -> 0.5
    expect(score).toBeGreaterThan(0);
    expect(score).toBeLessThan(1);
    expect(score).toBeCloseTo(0.5, 5);
  });

  it('is case-insensitive', () => {
    const a = keywordScoreText('GLOB PATTERN', 'glob pattern');
    const b = keywordScoreText('glob pattern', 'glob pattern');
    expect(a).toBeCloseTo(b, 5);
  });
});

describe('keywordScore — stopword handling', () => {
  it('ignores common stopwords that carry little tool-choice signal', () => {
    // "the" and "a" are stopwords; the only signal token is "glob".
    const withStopwords = keywordScore('the a glob', GLOB_TOOL);
    const withoutStopwords = keywordScore('glob', GLOB_TOOL);
    expect(withStopwords).toBeCloseTo(withoutStopwords, 5);
  });

  it('does not inflate the score when the query is all stopwords', () => {
    const score = keywordScore('the a an is are was', GLOB_TOOL);
    expect(score).toBe(0);
  });
});

describe('keywordScore — glob-token expansion', () => {
  it('matches the extension token "ts" when the query uses "*.ts"', () => {
    // The tool text embeds "*.ts" (via examples). The query "ts files" should
    // overlap because "*.ts" is expanded to also index "ts".
    const score = keywordScore('ts files', GLOB_TOOL);
    expect(score).toBeGreaterThan(0);
  });

  it('matches ".ts" as well as "ts" for a "*.ts" tool text', () => {
    const score = keywordScore('.ts', GLOB_TOOL);
    expect(score).toBeGreaterThan(0);
  });

  it('does not match an unrelated extension', () => {
    // Isolate the fixture: a tool whose canonical text contains no token
    // shared with the query except the extension "ts" it declares.
    const tsOnly = makeTool('tsScanner', 'ts language scanner.', {
      examples: ['scan *.ts'],
    });
    const score = keywordScore('py extension', tsOnly);
    expect(score).toBe(0);
  });
});

describe('keywordScore — composition contract', () => {
  it('scores against the same text the index command embeds (whenToUse + NOT: whenNotToUse)', () => {
    // The canonical text includes "NOT: The user wants to read a specific file".
    // A query about reading a file should therefore overlap with glob's
    // whenNotToUse — this is the intended behavior (S3 surfaces the boundary).
    const score = keywordScore('read a specific file', GLOB_TOOL);
    expect(score).toBeGreaterThan(0);
  });

  it('is deterministic for the same inputs', () => {
    const a = keywordScore('find files', GLOB_TOOL);
    const b = keywordScore('find files', GLOB_TOOL);
    expect(a).toBe(b);
  });
});
