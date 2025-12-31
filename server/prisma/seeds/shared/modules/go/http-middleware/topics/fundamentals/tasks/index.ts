/**
 * HTTP Middleware Fundamentals - Tasks Index
 * Basic HTTP middleware patterns (easy/medium difficulty)
 */

import { task as requestId } from './01-request-id';
import { task as setHeader } from './02-set-header';
import { task as requireMethod } from './03-require-method';
import { task as requireHeader } from './04-require-header';
import { task as logger } from './05-logger';
import { task as stripPrefix } from './06-strip-prefix';
import { task as requireQueryParam } from './07-require-query-param';
import { task as headerToContext } from './08-header-to-context';
import { task as recover } from './09-recover';

export const tasks = [
	requestId,
	setHeader,
	requireMethod,
	requireHeader,
	logger,
	stripPrefix,
	requireQueryParam,
	headerToContext,
	recover,
];
