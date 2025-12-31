import { Task } from '../../../../../../types';
import { task as reentrantLock } from './01-reentrant-lock';
import { task as readWriteLock } from './02-read-write-lock';
import { task as stampedLock } from './03-stamped-lock';

export const tasks: Task[] = [
    reentrantLock,
    readWriteLock,
    stampedLock,
];
