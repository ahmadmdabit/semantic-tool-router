#!/usr/bin/env node

import { Command } from 'commander';
import { createIndexCommand } from './commands/index.command.js';
import { createRouteCommand } from './commands/route.command.js';

const program = new Command();

program
  .name('semantic-tool-router')
  .description('Semantic Tool Router CLI')
  .version('0.1.0');

program.addCommand(createIndexCommand());
program.addCommand(createRouteCommand());

program.parse(process.argv);
