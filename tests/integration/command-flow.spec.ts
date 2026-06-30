import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { writeFileSync, mkdirSync, rmSync, readFileSync } from 'node:fs';
import { join } from 'node:path';

const TMPDIR = `test-cmd-tools-${process.pid}`;
const TMPSTORE = `test-cmd-store-${process.pid}.jsonl`;

function writeTool(filename: string, tool: unknown) {
  writeFileSync(join(TMPDIR, filename), JSON.stringify(tool));
}

function validTool(name: string) {
  return {
    name,
    description: `${name} description`,
    intent: `${name} intent`,
    examples: [`example for ${name}`],
    whenToUse: [`use ${name} when X`],
    whenNotToUse: [`do not use ${name} for Y`],
    triggers: [name],
    boosts: [],
    parameters: { type: 'object', properties: { path: { description: `${name} path` } } },
  };
}

// Deterministic sync hash → unit vector (8 dims).
function embedSync(text: string, dim = 8): number[] {
  let h = 0;
  for (let i = 0; i < text.length; i++) h = (h * 31 + text.charCodeAt(i)) | 0;
  const vec: number[] = [];
  for (let i = 0; i < dim; i++) {
    h = (h * 1103515245 + 12345) & 0x7fffffff;
    vec.push((h % 1000) / 1000);
  }
  const mag = Math.sqrt(vec.reduce((s, v) => s + v * v, 0));
  return vec.map((v) => v / (mag || 1));
}

// Top-level mock: every import of OllamaEmbedder gets our deterministic double.
// The command does `new OllamaEmbedder(...)` and then calls `embedder.embed()`,
// so the mock must be a constructable function whose instances carry `embed`.
const MockEmbedder = vi.fn(function(this: any) {
  this.embed = vi.fn(async (text: string) => embedSync(text));
});
vi.mock('../../src/embeddings/ollama-embedder.js', () => ({
  OllamaEmbedder: MockEmbedder,
}));

beforeEach(() => {
  mkdirSync(TMPDIR, { recursive: true });
});

afterEach(() => {
  rmSync(TMPDIR, { recursive: true, force: true });
  rmSync(TMPSTORE, { force: true });
  vi.restoreAllMocks();
});

describe('index command', () => {
  it('loads tools, embeds both prototypes, and writes a JSONL index', async () => {
    writeTool('glob.json', validTool('glob'));

    const { createIndexCommand } = await import('../../src/commands/index.command.js');
    const cmd = createIndexCommand();

    // createIndexCommand returns the `index` subcommand, so the args after
    // parsing are just the positional + options (no leading `index` token).
    await cmd.parseAsync(['node', 'cli', TMPDIR, '--dimensions', '8', '--output', TMPSTORE]);

    const raw = readFileSync(TMPSTORE, 'utf-8').trim().split('\n');
    const records = raw.map((l) => JSON.parse(l));
    const vectors = records.filter((r: any) => r.type === 'vector');
    expect(vectors.length).toBe(1);
    // Each vector record must carry a posVec.
    expect(vectors[0].data.posVec).toBeDefined();
  });

  it('rejects a non-positive --dimensions and exits with code 1', async () => {
    writeTool('glob.json', validTool('glob'));

    const { createIndexCommand } = await import('../../src/commands/index.command.js');
    const cmd = createIndexCommand();

    // Commander routes validation failures to process.exit; mock it to throw
    // so the test can assert on the exit code instead of killing the runner.
    const exitSpy = vi.spyOn(process, 'exit').mockImplementation(((code?: number) => {
      throw new Error(`EXIT_${code}`);
    }) as typeof process.exit);

    await expect(
      cmd.parseAsync(['node', 'cli', TMPDIR, '--dimensions', '-1']),
    ).rejects.toThrow('EXIT_1');

    expect(exitSpy).toHaveBeenCalledWith(1);
  });
});

describe('index + route end-to-end (mocked embedder)', () => {
  it('indexes a catalog then routes a query to the expected tool', async () => {
    writeTool('glob.json', validTool('glob'));
    writeTool('grep.json', validTool('grep'));

    const { createIndexCommand } = await import('../../src/commands/index.command.js');
    const { createRouteCommand } = await import('../../src/commands/route.command.js');

    // Step 1: index.
    const indexCmd = createIndexCommand();
    await indexCmd.parseAsync(['node', 'cli', TMPDIR, '--dimensions', '8', '--output', TMPSTORE]);

    const rawLines = readFileSync(TMPSTORE, 'utf-8').trim().split('\n');
    const vectors = rawLines.map((l) => JSON.parse(l)).filter((r: any) => r.type === 'vector');
    expect(vectors.length).toBe(2);
    expect(vectors[0].data.posVec).toBeDefined();

    // Step 2: route a query that should resolve to glob.
    const routeCmd = createRouteCommand();
    const logs: string[] = [];
    const origLog = console.log;
    console.log = (...args: any[]) => { logs.push(args.map(String).join(' ')); };

    await routeCmd.parseAsync(['node', 'cli', 'glob wildcard pattern', '--store', TMPSTORE, '--dimensions', '8']);

    console.log = origLog;
    const output = logs.join('\n');
    // glob should be the top-ranked tool.
    expect(output).toMatch(/1\. glob/);
  });

  it('route prints no-tools message when nothing clears the threshold', async () => {
    writeTool('glob.json', validTool('glob'));

    const { createIndexCommand } = await import('../../src/commands/index.command.js');
    const { createRouteCommand } = await import('../../src/commands/route.command.js');

    const indexCmd = createIndexCommand();
    await indexCmd.parseAsync(['node', 'cli', TMPDIR, '--dimensions', '8', '--output', TMPSTORE]);

    const routeCmd = createRouteCommand();
    const logs: string[] = [];
    const origLog = console.log;
    console.log = (...args: any[]) => { logs.push(args.map(String).join(' ')); };

    // An extremely high threshold should drop every tool.
    await routeCmd.parseAsync(['node', 'cli', 'glob pattern', '--store', TMPSTORE, '--dimensions', '8', '--threshold', '0.9999']);

    console.log = origLog;
    expect(logs.join('\n')).toMatch(/no tools matched/);
  });
});

// Commander's built-in option validation (parseInt + range checks) calls
// process.exit(1) BEFORE the .action() callback fires. Vitest intercepts
// process.exit and throws "process.exit unexpectedly called with \"1\"" — we
// assert on that message to confirm the validation branch rejected the input.
describe('route command — input validation', () => {
  it('rejects a non-numeric --top-k during Commander validation', async () => {
    const { createRouteCommand } = await import('../../src/commands/route.command.js');
    const cmd = createRouteCommand();
    await expect(
      cmd.parseAsync(['node', 'cli', 'query', '--top-k', 'abc']),
    ).rejects.toThrow(/process.exit unexpectedly called with "1"/);
  });

  it('rejects a negative --dimensions during Commander validation', async () => {
    const { createRouteCommand } = await import('../../src/commands/route.command.js');
    const cmd = createRouteCommand();
    await expect(
      cmd.parseAsync(['node', 'cli', 'query', '--dimensions', '-5']),
    ).rejects.toThrow(/process.exit unexpectedly called with "1"/);
  });

  it('rejects a --threshold above 1 during Commander validation', async () => {
    const { createRouteCommand } = await import('../../src/commands/route.command.js');
    const cmd = createRouteCommand();
    await expect(
      cmd.parseAsync(['node', 'cli', 'query', '--threshold', '1.5']),
    ).rejects.toThrow(/process.exit unexpectedly called with "1"/);
  });
});

describe('index command — negative prototype', () => {
  it('embeds an empty negVec when a tool has no whenNotToUse', async () => {
    // Tool without whenNotToUse → buildNegText returns '' → embedder is never
    // called for the negative side, and the store records a zero-length negVec.
    const noNegTool = {
      name: 'simple',
      description: 'a tool without boundaries',
      intent: 'simple intent',
      examples: ['do a thing'],
      whenToUse: ['when doing a thing'],
      // no whenNotToUse
      triggers: ['simple'],
      boosts: [],
      parameters: { type: 'object', properties: {} },
    };
    writeTool('simple.json', noNegTool);

    const { createIndexCommand } = await import('../../src/commands/index.command.js');
    const cmd = createIndexCommand();
    await cmd.parseAsync(['node', 'cli', TMPDIR, '--dimensions', '8', '--output', TMPSTORE]);

    // Load the store and verify the negVec is zero-length.
    const { VectorStore } = await import('../../src/vector/vector-store.js');
    const store = new VectorStore(TMPSTORE);
    await store.load();
    const tools = store.tools();
    expect(tools[0].name).toBe('simple');

    // Search for the tool — score should be a pure positive cosine (no
    // penalty) since negVec is zero-length. The exact value depends on the
    // mock hash, but it must be positive and the tool must rank #1.
    const results = await store.search([1, 0, 0, 0, 0, 0, 0, 0], 5);
    expect(results[0].tool.name).toBe('simple');
    expect(results[0].score).toBeGreaterThan(0); // pure cosine, no subtraction

    // Verify the JSONL on disk has no negVec field (empty negVec is not persisted).
    const fs = await import('node:fs');
    const lines = fs.readFileSync(TMPSTORE, 'utf-8').trim().split('\n');
    const vectorLine = lines.find((l) => JSON.parse(l).type === 'vector');
    const record = JSON.parse(vectorLine!);
    expect(record.data.posVec).toBeDefined();
    expect(record.data.negVec).toBeUndefined();
  });
});

describe('route command — edge cases', () => {
  it('exits with code 1 when the vector store is empty', async () => {
    // Point --store at a non-existent file → load() yields an empty store.
    const { createRouteCommand } = await import('../../src/commands/route.command.js');
    const cmd = createRouteCommand();
    const exitSpy = vi.spyOn(process, 'exit').mockImplementation(((code?: number) => {
      throw new Error(`EXIT_${code}`);
    }) as typeof process.exit);

    await expect(
      cmd.parseAsync(['node', 'cli', 'glob pattern', '--store', 'nonexistent-store.jsonl']),
    ).rejects.toThrow('EXIT_1');
    expect(exitSpy).toHaveBeenCalledWith(1);
  });

  it('warns when the indexed model differs from the query model', async () => {
    writeTool('glob.json', validTool('glob'));
    const { createIndexCommand } = await import('../../src/commands/index.command.js');
    const indexCmd = createIndexCommand();
    await indexCmd.parseAsync(['node', 'cli', TMPDIR, '--dimensions', '8', '--output', TMPSTORE]);

    // Load the store, rewrite metadata to a different model, save it back.
    const fs = await import('node:fs');
    const rawLines = fs.readFileSync(TMPSTORE, 'utf-8').trim().split('\n');
    const records = rawLines.map((l) => JSON.parse(l));
    const metaRecord = records.find((r: any) => r.type === 'metadata');
    if (metaRecord) metaRecord.data.model = 'other-model:latest';
    fs.writeFileSync(TMPSTORE, records.map((r) => JSON.stringify(r)).join('\n') + '\n');

    const { createRouteCommand } = await import('../../src/commands/route.command.js');
    const cmd = createRouteCommand();
    const warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => {});

    await cmd.parseAsync(['node', 'cli', 'glob pattern', '--store', TMPSTORE, '--dimensions', '8']);

    const warnings = warnSpy.mock.calls.map((c) => c.join(' ')).join('\n');
    expect(warnings).toMatch(/Index was built with model "other-model:latest"/);
    warnSpy.mockRestore();
  });

  it('warns when the vector store has no metadata', async () => {
    // Hand-write a JSONL store with only a vector record (no metadata).
    const fs = await import('node:fs');
    const vectorOnly = JSON.stringify({
      type: 'vector',
      data: {
        tool: { name: 'glob', description: 'g', parameters: { type: 'object', properties: {} } },
        posVec: embedSync('glob'),
      },
    });
    fs.writeFileSync(TMPSTORE, vectorOnly + '\n');

    const { createRouteCommand } = await import('../../src/commands/route.command.js');
    const cmd = createRouteCommand();
    const warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => {});

    await cmd.parseAsync(['node', 'cli', 'glob pattern', '--store', TMPSTORE, '--dimensions', '8']);

    const warnings = warnSpy.mock.calls.map((c) => c.join(' ')).join('\n');
    expect(warnings).toMatch(/no metadata/);
    warnSpy.mockRestore();
  });

  it('prints the intent signal when a structural rule fires', async () => {
    writeTool('glob.json', validTool('glob'));
    const { createIndexCommand } = await import('../../src/commands/index.command.js');
    const indexCmd = createIndexCommand();
    await indexCmd.parseAsync(['node', 'cli', TMPDIR, '--dimensions', '8', '--output', TMPSTORE]);

    const { createRouteCommand } = await import('../../src/commands/route.command.js');
    const cmd = createRouteCommand();
    const logs: string[] = [];
    const origLog = console.log;
    console.log = (...args: any[]) => { logs.push(args.map(String).join(' ')); };

    // A query containing actual glob characters (*) fires the builtin glob
    // baseline rule (pattern: 'glob'), which surfaces as a non-'none' intent.
    await cmd.parseAsync(['node', 'cli', 'find all *.ts files', '--store', TMPSTORE, '--dimensions', '8']);

    console.log = origLog;
    const output = logs.join('\n');
    expect(output).toMatch(/Intent signal: glob/);
  });
});

describe('route command — output modes', () => {
  it('emits JSON array when --json is passed', async () => {
    writeTool('glob.json', validTool('glob'));
    const { createIndexCommand } = await import('../../src/commands/index.command.js');
    const indexCmd = createIndexCommand();
    await indexCmd.parseAsync(['node', 'cli', TMPDIR, '--dimensions', '8', '--output', TMPSTORE]);

    const { createRouteCommand } = await import('../../src/commands/route.command.js');
    const cmd = createRouteCommand();
    const logs: string[] = [];
    const origLog = console.log;
    console.log = (...args: any[]) => { logs.push(args.map(String).join(' ')); };

    await cmd.parseAsync(['node', 'cli', 'glob pattern', '--store', TMPSTORE, '--dimensions', '8', '--json']);

    console.log = origLog;
    const output = logs.join('\n');
    // JSON output must be parseable and contain a scored tool.
    const parsed = JSON.parse(output);
    expect(Array.isArray(parsed)).toBe(true);
    expect(parsed[0].tool.name).toBe('glob');
    expect(typeof parsed[0].score).toBe('number');
  });

  it('exits with code 1 when indexed dimensions differ from query dimensions', async () => {
    writeTool('glob.json', validTool('glob'));
    const { createIndexCommand } = await import('../../src/commands/index.command.js');
    const indexCmd = createIndexCommand();
    // Build the index at 8 dimensions.
    await indexCmd.parseAsync(['node', 'cli', TMPDIR, '--dimensions', '8', '--output', TMPSTORE]);

    const { createRouteCommand } = await import('../../src/commands/route.command.js');
    const cmd = createRouteCommand();
    const exitSpy = vi.spyOn(process, 'exit').mockImplementation(((code?: number) => {
      throw new Error(`EXIT_${code}`);
    }) as typeof process.exit);

    // Query at a different dimension count — triggers the metadata mismatch branch.
    await expect(
      cmd.parseAsync(['node', 'cli', 'glob pattern', '--store', TMPSTORE, '--dimensions', '16']),
    ).rejects.toThrow('EXIT_1');
    expect(exitSpy).toHaveBeenCalledWith(1);
  });
});
