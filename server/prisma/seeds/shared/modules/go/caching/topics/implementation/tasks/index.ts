/**
 * Caching Implementation - Tasks Index
 * Exports all tasks in order
 */

import { task as ttlcacheStructure } from './01-ttlcache-structure';
import { task as setMethod } from './02-set-method';
import { task as getMethod } from './03-get-method';
import { task as cleanupNow } from './04-cleanup-now';
import { task as deleteMethod } from './05-delete-method';
import { task as lenMethod } from './06-len-method';

export const tasks = [
	ttlcacheStructure,
	setMethod,
	getMethod,
	cleanupNow,
	deleteMethod,
	lenMethod,
];
