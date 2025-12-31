/**
 * Rate Limiting Implementation - Tasks Index
 * Exports all tasks in order
 */

import { task as limiterStructure } from './01-limiter-structure';
import { task as newConstructor } from './02-new-constructor';
import { task as reserveMethod } from './03-reserve-method';
import { task as allowMethod } from './04-allow-method';
import { task as allowWithin } from './05-allow-within';
import { task as wrapMethod } from './06-wrap-method';

export const tasks = [
	limiterStructure,
	newConstructor,
	reserveMethod,
	allowMethod,
	allowWithin,
	wrapMethod,
];
