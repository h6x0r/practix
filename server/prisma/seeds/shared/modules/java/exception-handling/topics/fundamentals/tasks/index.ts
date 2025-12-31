import { Task } from '../../../../../../types';
import { task as tryCatchBasics } from './01-try-catch-basics';
import { task as exceptionHierarchy } from './02-exception-hierarchy';
import { task as customExceptions } from './03-custom-exceptions';
import { task as tryWithResources } from './04-try-with-resources';
import { task as exceptionChaining } from './05-exception-chaining';
import { task as bestPractices } from './06-best-practices';

export const tasks: Task[] = [
    tryCatchBasics,
    exceptionHierarchy,
    customExceptions,
    tryWithResources,
    exceptionChaining,
    bestPractices,
];