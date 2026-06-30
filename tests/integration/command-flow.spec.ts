import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { writeFileSync, mkdirSync, rmSync, readFileSync } from 'node:fs';
import { join } from 'node:path';

const TMP_DIR = `test-cmd-tools-${process.pid}`;
const TMP_STORE = `test-cmd-store-${process.pid}.jsonl`;

function writeTool(filename: string, tool: unknown) {
  writeFileSync(join(TMP_DIR, filename), JSON.stringify(tool));
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
  mkdirSync(TMP_DIR, { recursive: true });
});

afterEach(() => {
  rmSync(TMP_DIR, { recursive: true, force: true });
  rmSync(TMP_STORE, { force: true });
  vi.restoreAllMocks();
});

describe('index command', () => {
  it('loads tools, embeds both prototypes, and writes a JSONL index', async () => {
    writeTool('glob.json', validTool('glob'));

    const { createIndexCommand } = await import('../../src/commands/index.command.js');
    const cmd = createIndexCommand();

    // createIndexCommand returns the `index` subcommand, so the args after
    // parsing are just the positional + options (no leading `index` token).
    await cmd.parseAsync(['node', 'cli', TMP_DIR, '--dimensions', '8', '--output', TMP_STORE]);

    const raw = readFileSync(TMP_STORE, 'utf-8').trim().split('\n');
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
      cmd.parseAsync(['node', 'cli', TMP_DIR, '--dimensions', '-1']),
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
    await indexCmd.parseAsync(['node', 'cli', TMP_DIR, '--dimensions', '8', '--output', TMP_STORE]);

    const rawLines = readFileSync(TMP_STORE, 'utf-8').trim().split('\n');
    const vectors = rawLines.map((l) => JSON.parse(l)).filter((r: any) => r.type === 'vector');
    expect(vectors.length).toBe(2);
    expect(vectors[0].data.posVec).toBeDefined();

    // Step 2: route a query that should resolve to glob.
    const routeCmd = createRouteCommand();
    const logs: string[] = [];
    const origLog = console.log;
    console.log = (...args: any[]) => { logs.push(args.map(String).join(' ')); };

    await routeCmd.parseAsync(['node', 'cli', 'glob wildcard pattern', '--store', TMP_STORE, '--dimensions', '8']);

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
    await indexCmd.parseAsync(['node', 'cli', TMP_DIR, '--dimensions', '8', '--output', TMP_STORE]);

    const routeCmd = createRouteCommand();
    const logs: string[] = [];
    const origLog = console.log;
    console.log = (...args: any[]) => { logs.push(args.map(String).join(' ')); };

    // An extremely high threshold should drop every tool.
    await routeCmd.parseAsync(['node', 'cli', 'glob pattern', '--store', TMP_STORE, '--dimensions', '8', '--threshold', '0.9999']);

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

describe('route command — output modes', () => {
  it('emits JSON array when --json is passed', async () => {
    writeTool('glob.json', validTool('glob'));
    const { createIndexCommand } = await import('../../src/commands/index.command.js');
    const indexCmd = createIndexCommand();
    await indexCmd.parseAsync(['node', 'cli', TMP_DIR, '--dimensions', '8', '--output', TMP_STORE]);

    const { createRouteCommand } = await import('../../src/commands/route.command.js');
    const cmd = createRouteCommand();
    const logs: string[] = [];
    const origLog = console.log;
    console.log = (...args: any[]) => { logs.push(args.map(String).join(' ')); };

    await cmd.parseAsync(['node', 'cli', 'glob pattern', '--store', TMP_STORE, '--dimensions', '8', '--json']);

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
    await indexCmd.parseAsync(['node', 'cli', TMP_DIR, '--dimensions', '8', '--output', TMP_STORE]);

    const { createRouteCommand } = await import('../../src/commands/route.command.js');
    const cmd = createRouteCommand();
    const exitSpy = vi.spyOn(process, 'exit').mockImplementation(((code?: number) => {
      throw new Error(`EXIT_${code}`);
    }) as typeof process.exit);

    // Query at a different dimension count — triggers the metadata mismatch branch.
    await expect(
      cmd.parseAsync(['node', 'cli', 'glob pattern', '--store', TMP_STORE, '--dimensions', '16']),
    ).rejects.toThrow('EXIT_1');
    expect(exitSpy).toHaveBeenCalledWith(1);
  });
});
