import { Task } from '../../../../../../types';
import { task as pathFiles } from './01-path-files';
import { task as fileOperations } from './02-file-operations';
import { task as directoryOperations } from './03-directory-operations';
import { task as bytebuffer } from './04-bytebuffer';
import { task as fileChannel } from './05-file-channel';

export const tasks: Task[] = [
    pathFiles,
    fileOperations,
    directoryOperations,
    bytebuffer,
    fileChannel,
];
