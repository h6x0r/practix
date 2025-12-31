import { Task } from '../../../../types';
import { task as basicTransaction } from './01-basic-transaction';
import { task as rollbackOnError } from './02-rollback-on-error';
import { task as savepoint } from './03-savepoint';
import { task as transactionOptions } from './04-transaction-options';
import { task as retrySerialization } from './05-retry-serialization';

export const tasks: Task[] = [
    basicTransaction,
    rollbackOnError,
    savepoint,
    transactionOptions,
    retrySerialization
];
