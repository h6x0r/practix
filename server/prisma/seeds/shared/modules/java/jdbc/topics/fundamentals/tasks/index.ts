import { Task } from '../../../../../../types';
import { task as connectionBasics } from './01-connection-basics';
import { task as statementQuery } from './02-statement-query';
import { task as preparedStatement } from './03-prepared-statement';

export const tasks: Task[] = [
    connectionBasics,
    statementQuery,
    preparedStatement,
];
