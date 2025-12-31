import { Task } from '../../../../types';
import { task as queryRow } from './01-query-row';
import { task as queryMultiple } from './02-query-multiple';
import { task as execInsert } from './03-exec-insert';
import { task as preparedStatement } from './04-prepared-statement';
import { task as nullHandling } from './05-null-handling';
import { task as contextTimeout } from './06-context-timeout';

export const tasks: Task[] = [
    queryRow,
    queryMultiple,
    execInsert,
    preparedStatement,
    nullHandling,
    contextTimeout
];
