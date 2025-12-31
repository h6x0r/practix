/**
 * Benchmarking and Performance - Tasks Index
 * Exports all tasks in order
 */

import { task as basicBenchmark } from './01-basic-benchmark';
import { task as memoryAllocation } from './02-memory-allocation';
import { task as parallelBenchmark } from './03-parallel-benchmark';
import { task as subBenchmarks } from './04-sub-benchmarks';
import { task as benchmarkComparison } from './05-benchmark-comparison';

export const tasks = [
	basicBenchmark,
	memoryAllocation,
	parallelBenchmark,
	subBenchmarks,
	benchmarkComparison,
];
