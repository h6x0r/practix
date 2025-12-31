import { Task } from '../../../../../../types';
import { task as streamBasics } from './01-stream-basics';
import { task as intermediateOperations } from './02-intermediate-operations';
import { task as terminalOperations } from './03-terminal-operations';
import { task as collectors } from './04-collectors';

export const tasks: Task[] = [
    streamBasics,
    intermediateOperations,
    terminalOperations,
    collectors,
];
