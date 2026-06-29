import { defineConfig } from 'vite';
import { resolve } from 'node:path';
import { readFileSync } from 'node:fs';

// Read package.json to automatically externalize dependencies
const pkg = JSON.parse(readFileSync('./package.json', 'utf-8'));
const external = [
    ...Object.keys(pkg.dependencies || {}),
    ...Object.keys(pkg.peerDependencies || {}),
    /^node:/, // Externalize Node.js built-in modules
];

export default defineConfig({
    build: {
        lib: {
            entry: resolve(process.cwd(), 'src/cli.ts'),
            formats: ['es'], // Use 'es' for ESM, or 'cjs' for CommonJS
            fileName: () => 'cli.js',
        },
        rollupOptions: {
            external,
        },
        target: 'node22', // Set to your minimum supported Node version
        minify: false,    // Keep false for CLIs to make debugging/stack traces readable
        sourcemap: true,
    },
});
