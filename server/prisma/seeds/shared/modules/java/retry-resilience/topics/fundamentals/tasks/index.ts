import { Task } from '../../../../../../types';
import { task as retryBasics } from './01-retry-basics';
import { task as exponentialBackoff } from './02-exponential-backoff';
import { task as resilience4jRetry } from './03-resilience4j-retry';
import { task as circuitBreaker } from './04-circuit-breaker';

export const tasks: Task[] = [
    retryBasics,
    exponentialBackoff,
    resilience4jRetry,
    circuitBreaker,
];
