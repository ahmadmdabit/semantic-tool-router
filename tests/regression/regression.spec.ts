import { describe, it, expect, beforeAll } from 'vitest';
import { OllamaEmbedder } from '../../src/embeddings/ollama-embedder.js';
import { retrieve } from '../../src/routing/retriever.js';
import { VectorStore } from '../../src/vector/vector-store.js';
import { ToolLoader } from '../../src/tools/tool-loader.js';

// Full regression suite ported from test-routes.cmd. Each query declares the
// expected #1 tool. The spec uses the REAL OllamaEmbedder against a running
// Ollama instance (nomic-embed-text:latest), so it validates the actual
// pipeline the CLI uses — not a mock.
//
// Prerequisite: Ollama running locally with nomic-embed-text:latest pulled, and
// an up-to-date index is NOT required (retrieve() embeds tools on the fly via
// the fake embedder path — wait, no: the real embedder embeds the query, but
// the store is populated by embedding each tool's canonical text at setup time
// using the SAME real embedder).
//
// If Ollama is offline, beforeAll fails fast with a clear message instead of
// producing 45 opaque per-query failures.

const TOOLSDIR = 'tools';

interface Case {
  query: string;
  expected: string;
  // true negatives accept any result below a score floor.
  negative?: boolean;
}

const SUITE: Case[] = [
  // Glob family
  { query: 'List all the files with *.ts', expected: 'glob' },
  { query: 'Find every *.json file under src/', expected: 'glob' },
  { query: 'Which files match tests/**/*.test.ts', expected: 'glob' },
  { query: 'Locate all markdown files in the repo', expected: 'glob' },

  // Read family
  { query: 'read package.json', expected: 'readFile' },
  { query: 'show me the contents of README.md', expected: 'readFile' },
  { query: 'print the file src/cli.ts', expected: 'readFile' },
  { query: 'open and display tsconfig.json', expected: 'readFile' },

  // Write family
  { query: 'write a new hello.txt file', expected: 'writeFile' },
  { query: 'create a fresh .env from the example', expected: 'writeFile' },
  { query: 'overwrite output.log with an empty file', expected: 'writeFile' },
  { query: 'make a new component called Button.tsx', expected: 'writeFile' },

  // Edit family
  { query: 'edit the README to change the title', expected: 'editFile' },
  { query: 'replace all occurrences of foo with bar in src/index.ts', expected: 'editFile' },
  { query: 'update the version field in package.json', expected: 'editFile' },
  { query: 'modify the header in the landing page', expected: 'editFile' },

  // Move family
  { query: 'move src/old.ts to src/new.ts', expected: 'moveFile' },
  { query: 'relocate the assets folder into public/', expected: 'moveDirectory' },
  { query: 'transfer logs/ to /tmp/archive', expected: 'moveDirectory' },

  // Rename family
  { query: 'rename src/old.ts to src/new.ts', expected: 'renameFile' },
  { query: 'migrate the component to a new name', expected: 'renameDirectory' },
  // Target: renameFile. Fix applied — added "change file extension" triggers and
  // a whenToUse entry that captures extension-as-rename semantics.
  { query: 'change the file extension from .js to .ts', expected: 'renameFile' },

  // List family
  { query: 'list files in src/', expected: 'listDirectory' },
  { query: 'show me what is inside the tools directory', expected: 'listDirectory' },
  { query: 'enumerate the children of dist/', expected: 'listDirectory' },
  { query: 'ls the components folder', expected: 'listDirectory' },

  // Scan family
  { query: 'scan the src directory', expected: 'scanDirectory' },
  { query: 'walk the project tree', expected: 'scanDirectory' },
  // Target: scanDirectory. Fix applied — added a whenNotToUse entry to
  // listDirectory steering recursive-structure queries to scanDirectory.
  { query: 'show the directory structure recursively', expected: 'scanDirectory' },
  { query: 'traverse the folder and print every path', expected: 'scanDirectory' },

  // Search / grep family
  { query: 'search for TODO comments in all source files', expected: 'grep' },
  { query: 'grep for process.env across the repo', expected: 'grep' },
  // Target: grep. Fix applied — added prototype examples to grep.json
  // ("Search for the string TODO in markdown files", "Find every place where...")
  // so S1 cosine and S3 keyword overlap rise for these phrasings.
  { query: 'find every place where cosineSimilarity is called', expected: 'grep' },
  { query: 'look for the string TODO in markdown files', expected: 'grep' },

  // Shell / execution family
  { query: 'run npm test', expected: 'runCommand' },
  { query: 'execute the build script', expected: 'runCommand' },
  { query: 'run a shell command to check git status', expected: 'runCommand' },
  { query: 'install dependencies with yarn', expected: 'runCommand' },

  // Web / fetch family
  { query: 'fetch the prosodica.ai homepage', expected: 'webFetch' },
  { query: 'download the JSON from https://api.example.com/data', expected: 'webFetch' },
  { query: 'get the latest release notes from GitHub', expected: 'webFetch' },

  // Skill family
  { query: 'invoke the code-reviewer skill', expected: 'useSkill' },
  { query: 'use the test-generator skill on src/math', expected: 'useSkill' },
  { query: 'run the documentation skill', expected: 'useSkill' },

  // True negatives — no confident match expected.
  { query: 'tell me a joke', expected: '', negative: true },
  { query: 'what is the weather like', expected: '', negative: true },
  { query: 'explain quantum entanglement', expected: '', negative: true },
  { query: 'the meaning of life', expected: '', negative: true },
];

const NegativeFloor = 0.7; // true negatives should stay below this

describe('regression (real embedder, real catalog)', () => {
  const embedder = new OllamaEmbedder();
  let store: VectorStore;

  beforeAll(async () => {
    // Health check: fail fast with a clear message if Ollama is unreachable.
    const ctrl = new AbortController();
    const timeout = setTimeout(() => ctrl.abort(), 5000);
    try {
      const res = await fetch(`${(embedder as any).baseUrl}/api/tags`, { signal: ctrl.signal });
      if (!res.ok) throw new Error(`Ollama returned ${res.status}`);
    } catch (err: any) {
      throw new Error(
        `Ollama is offline — regression suite requires a running Ollama with nomic-embed-text:latest pulled. (${err.message})`,
      );
    } finally {
      clearTimeout(timeout);
    }

    // Build an in-memory index from the real catalog using the real embedder.
    // Mirrors index.command.ts: positive and negative prototypes are embedded
    // separately so the S1 score can subtract the negative-prototype cosine.
    const tools = ToolLoader.loadFromDirectory(TOOLSDIR);
    store = new VectorStore();
    const posEmbeddings: number[][] = [];
    const negEmbeddings: number[][] = [];
    for (const tool of tools) {
      // Positive prototype: what the tool IS.
      const posParts = [tool.name, tool.description];
      if (tool.intent) posParts.push(tool.intent);
      if (tool.examples?.length) posParts.push(tool.examples.join('. '));
      if (tool.whenToUse?.length) posParts.push(tool.whenToUse.join('. '));
      posEmbeddings.push(await embedder.embed(posParts.filter(Boolean).join('. '), 'document'));

      // Negative prototype: what the tool is NOT (whenNotToUse). Embedded as
      // plain sentences (no NOT: prefix) into a separate negVec.
      if (tool.whenNotToUse?.length) {
        negEmbeddings.push(await embedder.embed(tool.whenNotToUse.join('. '), 'document'));
      } else {
        negEmbeddings.push([]); // → zero-length negVec, no polarity penalty
      }
    }
    await store.add(tools, posEmbeddings, negEmbeddings);
  }, 120000); // embedding 14 tools (pos + neg) can take a few seconds

  for (const { query, expected, negative } of SUITE) {
    it(`routes "${query}" → ${negative ? `(negative, <${NegativeFloor})` : expected}`, async () => {
      const results = await retrieve(query, embedder, store, { k: 5 });

      if (negative) {
        // True negative: no tool should clear the confidence floor.
        for (const r of results) {
          expect(r.score).toBeLessThan(NegativeFloor);
        }
      } else {
        expect(results.length).toBeGreaterThan(0);
        expect(results[0].tool.name).toBe(expected);
      }
    });
  }
});
