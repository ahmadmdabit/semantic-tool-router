import { defineConfig } from 'vitest/config';

export default defineConfig({
  test: {
    // The intent-detector module caches its rule table keyed on tool names.
    // Running specs in separate workers is fine (each gets its own module
    // instance), but within a single file the cache leaks between tests —
    // specs call resetIntentCache() in beforeEach to isolate. Disabling
    // cross-file parallelism keeps total runtime predictable for the stress
    // spec which embeds the real model.
    fileParallelism: false,
    include: ['tests/**/*.spec.ts'],
    coverage: {
      provider: 'v8',
      include: ['src/**/*.ts'],
      exclude: ['src/example.ts'],
    },
    benchmark: {
      include: ['tests/**/*.bench.ts'],
    },
    environment: 'node',
  },
});
