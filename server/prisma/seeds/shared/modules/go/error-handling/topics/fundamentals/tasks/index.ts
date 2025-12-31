/**
 * Error Handling Fundamentals - Tasks Index
 * Exports all tasks in order
 */

import { task as sentinelErrors } from './01-sentinel-errors';
import { task as customErrorType } from './02-custom-error-type';
import { task as errorUnwrap } from './03-error-unwrap';
import { task as errorWrap } from './04-error-wrap';
import { task as errorE } from './05-error-e';
import { task as isNotFound } from './06-is-not-found';
import { task as formatNotFound } from './07-format-not-found';

export const tasks = [
	sentinelErrors,
	customErrorType,
	errorUnwrap,
	errorWrap,
	errorE,
	isNotFound,
	formatNotFound,
];
