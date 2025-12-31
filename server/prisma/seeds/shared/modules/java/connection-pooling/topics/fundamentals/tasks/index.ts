import { Task } from '../../../../../../types';
import { task as poolBasics } from './01-pool-basics';
import { task as hikaricpSetup } from './02-hikaricp-setup';

export const tasks: Task[] = [
    poolBasics,
    hikaricpSetup,
];
