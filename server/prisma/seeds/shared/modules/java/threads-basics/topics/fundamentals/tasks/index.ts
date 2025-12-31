import { Task } from '../../../../../../types';
import threadCreation from './01-thread-creation';
import threadLifecycle from './02-thread-lifecycle';
import synchronizedKeyword from './03-synchronized-keyword';
import waitNotify from './04-wait-notify';
import threadSafety from './05-thread-safety';
import volatileKeyword from './06-volatile-keyword';

export const tasks: Task[] = [
    threadCreation,
    threadLifecycle,
    synchronizedKeyword,
    waitNotify,
    threadSafety,
    volatileKeyword,
];
