/**
 * Context Management - Tasks Index
 * Exports all tasks in order
 */

import { task as doWithTimeout } from './01-do-with-timeout';
import { task as doWithDeadline } from './02-do-with-deadline';
import { task as doWithCancel } from './03-do-with-cancel';
import { task as notifyCancel } from './04-notify-cancel';
import { task as waitForSignal } from './05-wait-for-signal';
import { task as waitAll } from './06-wait-all';
import { task as waitAny } from './07-wait-any';
import { task as runUntil } from './08-run-until';
import { task as retryWithContext } from './09-retry-with-context';
import { task as heartbeat } from './10-heartbeat';

export const tasks = [
	doWithTimeout,
	doWithDeadline,
	doWithCancel,
	notifyCancel,
	waitForSignal,
	waitAll,
	waitAny,
	runUntil,
	retryWithContext,
	heartbeat,
];
