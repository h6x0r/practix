import { Task } from '../../../../../../types';
import { task as inputOutputStream } from './01-input-output-stream';
import { task as fileStreams } from './02-file-streams';
import { task as bufferedStreams } from './03-buffered-streams';
import { task as dataStreams } from './04-data-streams';
import { task as objectStreams } from './05-object-streams';

export const tasks: Task[] = [
    inputOutputStream,
    fileStreams,
    bufferedStreams,
    dataStreams,
    objectStreams,
];
