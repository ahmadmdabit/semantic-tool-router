/**
 * Separate vitest config for benchmark mode.
 *
 * Uses the default pool (pool:'forks' is omitted here) because
 * tinybench's high-resolution performance.now() timer loses resolution when
 * the bench runner communicates across forked worker processes — resulting in
 * all-zero sample counts even when the benchmarked function completes normally.
 * Threads share the same V8 isolate timing clock and do not have this issue.
 */
import { defineConfig } from 'vitest/config';

export default defineConfig({
  test: {
    environment: 'node',
    benchmark: {
      include: ['tests/**/*.bench.ts'],
    },
  },
});
