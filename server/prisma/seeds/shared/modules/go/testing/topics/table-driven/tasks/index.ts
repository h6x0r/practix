/**
 * Table-Driven Tests - Tasks Index
 * Exports all tasks in order
 */

import { task as simpleTable } from './01-simple-table';
import { task as subtests } from './02-subtests';
import { task as parallelSubtests } from './03-parallel-subtests';
import { task as errorCases } from './04-error-cases';
import { task as complexTable } from './05-complex-table';

export const tasks = [
	simpleTable,
	subtests,
	parallelSubtests,
	errorCases,
	complexTable,
];
