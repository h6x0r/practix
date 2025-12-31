import { Task } from '../../../../../../types';
import { task as completableBasics } from './01-completable-basics';
import { task as thenMethods } from './02-then-methods';
import { task as combiningFutures } from './03-combining-futures';
import { task as exceptionHandling } from './04-exception-handling';

export const tasks: Task[] = [
    completableBasics,
    thenMethods,
    combiningFutures,
    exceptionHandling,
];
