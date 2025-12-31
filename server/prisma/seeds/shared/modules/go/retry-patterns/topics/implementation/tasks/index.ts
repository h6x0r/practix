/**
 * Retry Patterns Implementation - Tasks Index
 * Exports all tasks in order
 */

import { task as backoff } from './01-backoff';
import { task as doRetry } from './02-do';
import { task as retryUntil } from './03-retry-until';
import { task as backoffSequence } from './04-backoff-sequence';
import { task as sleepContext } from './05-sleep-context';

export const tasks = [
	backoff,
	doRetry,
	retryUntil,
	backoffSequence,
	sleepContext,
];
