import { Task } from '../../../../../../types';
import { task as concurrentHashmap } from './01-concurrent-hashmap';
import { task as copyOnWrite } from './02-copy-on-write';
import { task as blockingQueue } from './03-blocking-queue';

export const tasks: Task[] = [
    concurrentHashmap,
    copyOnWrite,
    blockingQueue,
];
