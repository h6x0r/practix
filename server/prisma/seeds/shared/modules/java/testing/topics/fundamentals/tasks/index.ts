import { Task } from '../../../../../../types';
import { task as junitBasics } from './01-junit-basics';
import { task as assertions } from './02-assertions';
import { task as testLifecycle } from './03-test-lifecycle';

export const tasks: Task[] = [
    junitBasics,
    assertions,
    testLifecycle,
];
