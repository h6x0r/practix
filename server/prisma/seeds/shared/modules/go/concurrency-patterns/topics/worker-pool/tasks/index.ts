/**
 * Worker Pool - Tasks Index
 * Exports all tasks in order
 */

import { task as runSequential } from './01-run-sequential';
import { task as runSequentialLimited } from './02-run-sequential-limited';
import { task as runParallel } from './03-run-parallel';
import { task as runParallelBounded } from './04-run-parallel-bounded';
import { task as runPool } from './05-run-pool';
import { task as runPoolCancelOnError } from './06-run-pool-cancel-on-error';
import { task as makeJobQueue } from './07-make-job-queue';
import { task as throttleJobSubmission } from './08-throttle-job-submission';
import { task as runPoolWithPanicHandler } from './09-run-pool-with-panic-handler';
import { task as runPoolWithResults } from './10-run-pool-with-results';

export const tasks = [
	runSequential,
	runSequentialLimited,
	runParallel,
	runParallelBounded,
	runPool,
	runPoolCancelOnError,
	makeJobQueue,
	throttleJobSubmission,
	runPoolWithPanicHandler,
	runPoolWithResults,
];
