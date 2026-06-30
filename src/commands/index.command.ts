import { Command } from 'commander';
import { OllamaEmbedder } from '../embeddings/ollama-embedder.js';
import { VectorStore } from '../vector/vector-store.js';
import { ToolLoader } from '../tools/tool-loader.js';
import { Tool, EmbedType } from '../types.js';

// Builds the POSITIVE prototype text: everything that describes what the tool
// IS and when it should be used. This is embedded into posVec, the vector the
// query is compared against for the "match" half of the S1 score.
function buildPosText(tool: Tool): string {
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
  if (tool.whenToUse && tool.whenToUse.length > 0) parts.push(tool.whenToUse.join('. '));

  return parts.filter(Boolean).join('. ');
}

// Builds the NEGATIVE prototype text: what the tool is NOT for. This is
// embedded into negVec, the vector that gets SUBTRACTED from the S1 score
// when the query shares tokens with the tool's exclusion criteria. Returns an
// empty string when the tool declares no boundaries — the caller then stores a
// zero-length negVec and polarity contributes nothing.
function buildNegText(tool: Tool): string {
  if (!tool.whenNotToUse || tool.whenNotToUse.length === 0) return '';
  return tool.whenNotToUse.join('. ');
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
      const posEmbeddings: number[][] = [];
      const negEmbeddings: number[][] = [];
      const batchSize = 5;
      const embedType: EmbedType = 'document';

      for (let i = 0; i < tools.length; i += batchSize) {
        const batch = tools.slice(i, i + batchSize);
        const [batchPos, batchNeg] = await Promise.all([
          Promise.all(batch.map((tool) => embedder.embed(buildPosText(tool), embedType))),
          // Negative text is embedded with the same document prefix so the
          // negVec lives on the same manifold as posVec — polarity subtraction
          // is only meaningful when both vectors are in the same space.
          Promise.all(batch.map(async (tool) => {
            const negText = buildNegText(tool);
            return negText ? embedder.embed(negText, embedType) : [];
          })),
        ]);
        batchPos.forEach((emb, idx) => {
          posEmbeddings.push(emb);
          negEmbeddings.push(batchNeg[idx]);
          console.log(`  Embedded: ${batch[idx].name}`);
        });
      }

      store.setMetadata({
        model: options.model,
        dimensions,
        indexedAt: new Date().toISOString(),
      });

      await store.add(tools, posEmbeddings, negEmbeddings);
      await store.save();

      console.log(`Index saved to ${options.output} (${store.size()} vectors)`);
      console.log(`Metadata: model=${options.model}, dimensions=${dimensions}`);
    });

  return command;
}
