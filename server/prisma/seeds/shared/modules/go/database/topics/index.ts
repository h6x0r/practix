import { Topic } from '../../../../types';
import { topic as sqlBasics } from './sql-basics/topic';
import { topic as transactions } from './transactions/topic';
import { topic as connectionPool } from './connection-pool/topic';

export const topics: Topic[] = [
    sqlBasics,
    transactions,
    connectionPool
];
