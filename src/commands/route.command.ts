import { Command } from 'commander';
import { OllamaEmbedder } from '../embeddings/ollama-embedder.js';
import { VectorStore } from '../vector/vector-store.js';

export function createRouteCommand(): Command {
  const command = new Command('route');

  command
    .description('Find top-K relevant tools for a query')
    .argument('<query>', 'User query')
    .option('-k, --top-k <number>', 'Number of tools to return', '5')
    .option('-s, --store <path>', 'Path to vector store', 'vector-store.json')
    .option('-m, --model <name>', 'Ollama embedding model (must match index)', 'nomic-embed-text:latest')
    .option('-d, --dimensions <number>', 'Embedding dimensions (must match index)', '768')
    .action(async (query: string, options) => {
      const k = parseInt(options.topK, 10);
      const dimensions = parseInt(options.dimensions, 10);

      if (isNaN(k) || k <= 0) {
        console.error('Error: --top-k must be a positive integer');
        process.exit(1);
      }

      if (isNaN(dimensions) || dimensions <= 0) {
        console.error('Error: --dimensions must be a positive integer');
        process.exit(1);
      }

      const embedder = new OllamaEmbedder('http://localhost:11434', options.model, dimensions);
      const store = new VectorStore(options.store);

      await store.load();

      if (store.size() === 0) {
        console.error('Vector store is empty. Run "index" command first.');
        process.exit(1);
      }

      // Validate metadata match
      const metadata = store.getMetadata();
      if (metadata) {
        if (metadata.model !== options.model) {
          console.warn(`Warning: Index was built with model "${metadata.model}" but querying with "${options.model}"`);
          console.warn('Results may be inaccurate. Consider re-indexing or using the same model.');
        }
        if (metadata.dimensions !== dimensions) {
          console.error(`Error: Index was built with ${metadata.dimensions} dimensions but querying with ${dimensions}`);
          console.error('Dimension mismatch will produce incorrect results. Please use matching dimensions.');
          process.exit(1);
        }
      } else {
        console.warn('Warning: Vector store has no metadata. Cannot validate model/dimensions compatibility.');
      }

      console.log(`Embedding query: "${query}"`);
      const queryEmbedding = await embedder.embed(query);

      console.log(`Searching top ${k} tools...`);
      const results = await store.search(queryEmbedding, k);

      console.log('\nTop relevant tools:');
      results.forEach((tool, index) => {
        console.log(`${index + 1}. ${tool.name}`);
        console.log(`   Description: ${tool.description}`);
        console.log('');
      });
    });

  return command;
}
