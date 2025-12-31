/**
 * Circuit Breaker Implementation - Tasks Index
 * Exports all tasks in order
 */

import { task as newConstructor } from './01-new-constructor';
import { task as doClosedState } from './02-do-closed-state';
import { task as doOpenState } from './03-do-open-state';
import { task as doHalfOpenState } from './04-do-halfopen-state';
import { task as stateMethod } from './05-state-method';
import { task as resetMethod } from './06-reset-method';
import { task as remainingHalfOpen } from './07-remaining-halfopen';

export const tasks = [
	newConstructor,
	doClosedState,
	doOpenState,
	doHalfOpenState,
	stateMethod,
	resetMethod,
	remainingHalfOpen,
];
