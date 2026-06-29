import { readdirSync, readFileSync } from 'node:fs';
import { join } from 'node:path';
import { Tool } from '../types.js';

export class ToolLoader {
  static loadFromDirectory(dirPath: string): Tool[] {
    const files = readdirSync(dirPath).filter((f) => f.endsWith('.json'));
    const tools: Tool[] = [];

    for (const file of files) {
      const content = readFileSync(join(dirPath, file), 'utf-8');
      const tool = JSON.parse(content) as Tool;

      if (!tool.name || !tool.description) {
        console.warn(`Skipping invalid tool file: ${file}`);
        continue;
      }

      tools.push(tool);
    }

    return tools;
  }
}
