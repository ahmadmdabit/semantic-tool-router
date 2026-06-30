import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import { ToolLoader } from '../../src/tools/tool-loader.js';
import { Tool } from '../../src/types.js';
import { writeFileSync, mkdirSync, rmSync, existsSync } from 'node:fs';
import { join } from 'node:path';

const TMP_DIR = `test-tools-${process.pid}`;

function writeTool(dir: string, filename: string, content: string | unknown) {
  const data = typeof content === 'string' ? content : JSON.stringify(content);
  writeFileSync(join(dir, filename), data);
}

function validTool(name: string): Tool {
  return { name, description: `${name} description`, parameters: { type: 'object', properties: {} } };
}

beforeEach(() => {
  mkdirSync(TMP_DIR, { recursive: true });
});

afterEach(() => {
  rmSync(TMP_DIR, { recursive: true, force: true });
});

describe('ToolLoader.loadFromDirectory', () => {
  it('loads valid JSON tool files', () => {
    writeTool(TMP_DIR, 'alpha.json', validTool('alpha'));
    writeTool(TMP_DIR, 'beta.json', validTool('beta'));
    const tools = ToolLoader.loadFromDirectory(TMP_DIR);
    expect(tools.map((t) => t.name).sort()).toEqual(['alpha', 'beta']);
  });

  it('skips non-JSON files', () => {
    writeTool(TMP_DIR, 'alpha.json', validTool('alpha'));
    writeTool(TMP_DIR, 'readme.txt', 'not a tool');
    const tools = ToolLoader.loadFromDirectory(TMP_DIR);
    expect(tools).toHaveLength(1);
  });

  it('skips files with malformed JSON and continues loading', () => {
    writeTool(TMP_DIR, 'alpha.json', validTool('alpha'));
    writeTool(TMP_DIR, 'broken.json', '{ not valid json');
    writeTool(TMP_DIR, 'beta.json', validTool('beta'));
    const tools = ToolLoader.loadFromDirectory(TMP_DIR);
    expect(tools.map((t) => t.name).sort()).toEqual(['alpha', 'beta']);
  });

  it('skips files missing required fields (name, description, parameters)', () => {
    writeTool(TMP_DIR, 'alpha.json', validTool('alpha'));
    writeTool(TMP_DIR, 'no-name.json', { description: 'x', parameters: {} }); // no name
    writeTool(TMP_DIR, 'no-desc.json', { name: 'node', parameters: {} }); // no description
    writeTool(TMP_DIR, 'no-params.json', { name: 'noparams', description: 'x' }); // no parameters
    writeTool(TMP_DIR, 'bad-params.json', { name: 'badparams', description: 'x', parameters: null }); // null parameters
    const tools = ToolLoader.loadFromDirectory(TMP_DIR);
    expect(tools).toHaveLength(1);
    expect(tools[0].name).toBe('alpha');
  });

  it('returns an empty array for an empty directory', () => {
    const tools = ToolLoader.loadFromDirectory(TMP_DIR);
    expect(tools).toEqual([]);
  });
});
