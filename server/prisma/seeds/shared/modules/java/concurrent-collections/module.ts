import { Module } from '../../../../types';
import { topics } from './topics';

export const module: Module = {
    slug: 'java-concurrent-collections',
    title: 'Concurrent Collections',
    description: 'Master thread-safe collections: ConcurrentHashMap, BlockingQueue, and synchronizers.',
    order: 14,
    topics,
    translations: {
        ru: {
            title: 'Параллельные коллекции',
            description: 'Освойте потокобезопасные коллекции: ConcurrentHashMap, BlockingQueue и синхронизаторы.',
        },
        uz: {
            title: 'Parallel kolleksiyalar',
            description: 'Oqim xavfsiz kolleksiyalarni o\'zlashtiring: ConcurrentHashMap, BlockingQueue va sinxronizatorlar.',
        },
    },
};
