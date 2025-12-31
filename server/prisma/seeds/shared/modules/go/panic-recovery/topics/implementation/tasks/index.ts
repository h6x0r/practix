import { task as safeCalls } from './01-safe-calls';
import { task as safeGoroutines } from './02-safe-goroutines';
import { task as panicToError } from './03-panic-to-error';
import { task as retryOnPanic } from './04-retry-on-panic';

export const tasks = [
	safeCalls,
	safeGoroutines,
	panicToError,
	retryOnPanic,
];
