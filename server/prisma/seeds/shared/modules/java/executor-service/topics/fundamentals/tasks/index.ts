import { Task } from '../../../../../../types';
import { task as executorBasics } from './01-executor-basics';
import { task as threadPools } from './02-thread-pools';
import { task as callableFuture } from './03-callable-future';

export const tasks: Task[] = [
    executorBasics,
    threadPools,
    callableFuture,
];
