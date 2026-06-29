import { Command } from 'commander';
import { OllamaEmbedder } from '../embeddings/ollama-embedder.js';
import { VectorStore } from '../vector/vector-store.js';
import { ToolLoader } from '../tools/tool-loader.js';
import { Tool, EmbedType } from '../types.js';

// Builds the text that gets embedded for a tool. Enriched vs. the old
// `name + description` form: includes parameter descriptions, the optional
// `intent` tag, and declared `examples`. Each of these gives the embedder a
// larger, more lexically-varied surface — which is what lets R1/R7 signals
// resolve queries like "list all *.ts files" toward glob despite token
// collision against listDirectory/readFile/moveFile.
function buildEmbeddingText(tool: Tool): string {
  const parts: string[] = [tool.name, tool.description];

  // Flatten the parameter descriptions so glob's `*.ts`, grep's
  // `src/**/*.ts`, etc. land inside the vector — the strongest S1 signal
  // for the failing query was never being embedded before.
  const params = tool.parameters;
  if (params && typeof params === 'object' && 'properties' in params) {
    const properties = (params as { properties?: Record<string, { description?: string }> }).properties;
    if (properties) {
      for (const prop of Object.values(properties)) {
        if (prop?.description) parts.push(prop.description);
      }
    }
  }

  if (tool.intent) parts.push(tool.intent);
  if (tool.examples && tool.examples.length > 0) parts.push(tool.examples.join('. '));

  // Boundary vocabulary: positive inclusion criteria and explicit
  // exclusion criteria. The "NOT: " prefix on whenNotToUse pushes negative
  // sentences away from the positive-prototype region in nomic's vector
  // space, so "do NOT use glob to read a file" sits far from the glob
  // prototype and near the readFile one.
  if (tool.whenToUse && tool.whenToUse.length > 0) parts.push(tool.whenToUse.join('. '));
  if (tool.whenNotToUse && tool.whenNotToUse.length > 0) {
    parts.push(tool.whenNotToUse.map((s) => `NOT: ${s}`).join('. '));
  }

  return parts.filter(Boolean).join('. ');
}

export function createIndexCommand(): Command {
  const command = new Command('index');

  command
    .description('Build vector index from tool catalog')
    .argument('<toolsDir>', 'Directory containing tool JSON files')
    .option('-o, --output <path>', 'Output path for vector store', 'vector-store.jsonl')
    .option('-m, --model <name>', 'Ollama embedding model', 'nomic-embed-text:latest')
    .option('-d, --dimensions <number>', 'Embedding dimensions (supports Matryoshka truncation)', '768')
    .option('-u, --url <url>', 'Ollama API URL', process.env.OLLAMA_HOST || 'http://localhost:11434')
    .action(async (toolsDir: string, options) => {
      const dimensions = parseInt(options.dimensions, 10);

      if (isNaN(dimensions) || dimensions <= 0) {
        console.error('Error: --dimensions must be a positive integer');
        process.exit(1);
      }

      console.log(`Loading tools from ${toolsDir}...`);
      const tools = ToolLoader.loadFromDirectory(toolsDir);
      console.log(`Loaded ${tools.length} tools`);

      const embedder = new OllamaEmbedder(options.url, options.model, dimensions);
      const store = new VectorStore(options.output);

      console.log(`Generating embeddings (model: ${options.model}, dimensions: ${dimensions})...`);
      const embeddings: number[][] = [];
      const batchSize = 5;
      const embedType: EmbedType = 'document';

      for (let i = 0; i < tools.length; i += batchSize) {
        const batch = tools.slice(i, i + batchSize);
        const batchEmbeddings = await Promise.all(batch.map(async (tool) => {
          const text = buildEmbeddingText(tool);
          const embedding = await embedder.embed(text, embedType);
          console.log(`  Embedded: ${tool.name}`);
          return embedding;
        }));
        embeddings.push(...batchEmbeddings);
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
