import { Task } from '../../../../../../types';
import { task as queueBasics } from './01-queue-basics';
import { task as dequeOperations } from './02-deque-operations';
import { task as priorityQueue } from './03-priority-queue';
import { task as arraydeque } from './04-arraydeque';
import { task as queuePatterns } from './05-queue-patterns';

export const tasks: Task[] = [
    queueBasics,
    dequeOperations,
    priorityQueue,
    arraydeque,
    queuePatterns,
];
