import { Module } from '../../../../types';
import { topics } from './topics';

export const module: Module = {
    slug: 'java-nio',
    title: 'NIO and NIO.2',
    description: 'Master Java NIO: Path, Files, ByteBuffer, FileChannel, and WatchService.',
    order: 19,
    topics,
    translations: {
        ru: {
            title: 'NIO и NIO.2',
            description: 'Освойте Java NIO: Path, Files, ByteBuffer, FileChannel и WatchService.',
        },
        uz: {
            title: 'NIO va NIO.2',
            description: 'Java NIO ni o\'rganing: Path, Files, ByteBuffer, FileChannel va WatchService.',
        },
    },
};
