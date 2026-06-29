import { Command } from 'commander';
import { OllamaEmbedder } from '../embeddings/ollama-embedder.js';
import { VectorStore } from '../vector/vector-store.js';
import { ToolLoader } from '../tools/tool-loader.js';

export function createIndexCommand(): Command {
  const command = new Command('index');

  command
    .description('Build vector index from tool catalog')
    .argument('<toolsDir>', 'Directory containing tool JSON files')
    .option('-o, --output <path>', 'Output path for vector store', 'vector-store.json')
    .option('-m, --model <name>', 'Ollama embedding model', 'nomic-embed-text:latest')
    .option('-d, --dimensions <number>', 'Embedding dimensions (supports Matryoshka truncation)', '768')
    .action(async (toolsDir: string, options) => {
      const dimensions = parseInt(options.dimensions, 10);

      if (isNaN(dimensions) || dimensions <= 0) {
        console.error('Error: --dimensions must be a positive integer');
        process.exit(1);
      }

      console.log(`Loading tools from ${toolsDir}...`);
      const tools = ToolLoader.loadFromDirectory(toolsDir);
      console.log(`Loaded ${tools.length} tools`);

      const embedder = new OllamaEmbedder('http://localhost:11434', options.model, dimensions);
      const store = new VectorStore(options.output);

      console.log(`Generating embeddings (model: ${options.model}, dimensions: ${dimensions})...`);
      const embeddings: number[][] = [];

      for (const tool of tools) {
        const text = `${tool.name} ${tool.description}`;
        const embedding = await embedder.embed(text);
        embeddings.push(embedding);
        console.log(`  Embedded: ${tool.name}`);
      }

      store.setMetadata({
        model: options.model,
        dimensions,
        indexedAt: new Date().toISOString(),
      });

      await store.add(tools, embeddings);
      await store.save();

      console.log(`Index saved to ${options.output} (${store.size()} vectors)`);
      console.log(`Metadata: model=${options.model}, dimensions=${dimensions}`);
    });

  return command;
}
