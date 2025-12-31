import { Topic } from '../../../../types';
import { tasks } from './tasks';

export const fundamentals: Topic = {
    slug: 'java-nio-fundamentals',
    title: 'NIO Fundamentals',
    description: 'Learn Path, Files, ByteBuffer, FileChannel, and WatchService in Java NIO',
    order: 0,
    tasks,
    translations: {
        ru: {
            title: 'Основы NIO',
            description: 'Изучите Path, Files, ByteBuffer, FileChannel и WatchService в Java NIO',
        },
        uz: {
            title: 'NIO Asoslari',
            description: 'Java NIO da Path, Files, ByteBuffer, FileChannel va WatchService ni o\'rganing',
        },
    },
};
