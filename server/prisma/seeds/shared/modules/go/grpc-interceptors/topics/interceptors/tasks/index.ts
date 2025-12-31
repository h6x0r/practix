/**
 * gRPC Interceptors - Tasks Index
 * Unary server interceptor implementations
 */

import { task as loggingInterceptor } from './01-logging-interceptor';
import { task as timeoutInterceptor } from './02-timeout-interceptor';
import { task as chain } from './03-chain';
import { task as retryInterceptor } from './04-retry-interceptor';
import { task as contextValueInterceptor } from './05-context-value-interceptor';

export const tasks = [
	loggingInterceptor,
	timeoutInterceptor,
	chain,
	retryInterceptor,
	contextValueInterceptor,
];
