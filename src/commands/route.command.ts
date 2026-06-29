import { Command } from 'commander';
import { OllamaEmbedder } from '../embeddings/ollama-embedder.js';
import { VectorStore } from '../vector/vector-store.js';
import { retrieve } from '../routing/retriever.js';

export function createRouteCommand(): Command {
  const command = new Command('route');

  command
    .description('Find top-K relevant tools for a query')
    .argument('<query>', 'User query')
    .option('-k, --top-k <number>', 'Number of tools to return', '5')
    .option('-s, --store <path>', 'Path to vector store', 'vector-store.jsonl')
    .option('-m, --model <name>', 'Ollama embedding model (must match index)', 'nomic-embed-text:latest')
    .option('-d, --dimensions <number>', 'Embedding dimensions (must match index)', '768')
    .option('-u, --url <url>', 'Ollama API URL', process.env.OLLAMA_HOST || 'http://localhost:11434')
    .option('-j, --json', 'Output results as JSON')
    .option('-t, --threshold <number>', 'Drop tools with composite score below this (0 disables)', '0')
    .action(async (query: string, options) => {
      const k = parseInt(options.topK, 10);
      const dimensions = parseInt(options.dimensions, 10);
      const threshold = parseFloat(options.threshold);

      if (isNaN(k) || k <= 0) {
        console.error('Error: --top-k must be a positive integer');
        process.exit(1);
      }

      if (isNaN(dimensions) || dimensions <= 0) {
        console.error('Error: --dimensions must be a positive integer');
        process.exit(1);
      }

      if (isNaN(threshold) || threshold < 0 || threshold > 1) {
        console.error('Error: --threshold must be a number in the range [0,1]');
        process.exit(1);
      }

      const embedder = new OllamaEmbedder(options.url, options.model, dimensions);
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

      if (!options.json) console.log(`Embedding query: "${query}"`);
      if (!options.json) console.log(`Searching top ${k} tools...`);

      // retrieve() embeds the query with the 'query' retrieval prefix, then
      // fuses cosine (S1), structural intent (S2), and keyword overlap (S3)
      // via RRF. Threshold (if set) is applied inside retrieve().
      const results = await retrieve(query, embedder, store, {
        k,
        threshold,
      });

      if (options.json) {
        console.log(JSON.stringify(results, null, 2));
      } else {
        console.log('\nTop relevant tools:');
        if (results.length === 0) {
          console.log('  (no tools matched the threshold)');
        }
        results.forEach((r, index) => {
          console.log(`${index + 1}. ${r.tool.name}  (score: ${r.score.toFixed(4)})`);
          console.log(`   Description: ${r.tool.description}`);
          if (r.debug?.intent && r.debug.intent !== 'none') {
            console.log(`   Intent signal: ${r.debug.intent}`);
          }
          console.log('');
        });
      }
    });

  return command;
}
