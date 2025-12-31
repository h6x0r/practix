import { Task } from '../../../../../../types';
import { task as atomicInteger } from './01-atomic-integer';
import { task as atomicReference } from './02-atomic-reference';
import { task as atomicArrays } from './03-atomic-arrays';
import { task as casOperations } from './04-cas-operations';

export const tasks: Task[] = [
    atomicInteger,
    atomicReference,
    atomicArrays,
    casOperations,
];
