import { Module } from '../../../../types';
import { topics } from './topics';

export const module: Module = {
    slug: 'java-stream-api',
    title: 'Stream API',
    description: 'Master Java Stream API: stream operations, collectors, parallel processing, and best practices.',
    order: 9,
    topics,
    translations: {
        ru: {
            title: 'Stream API',
            description: 'Освойте Java Stream API: потоковые операции, коллекторы, параллельную обработку и лучшие практики.',
        },
        uz: {
            title: 'Stream API',
            description: 'Java Stream API ni o\'rganing: oqim operatsiyalari, kollektorlar, parallel ishlov berish va eng yaxshi amaliyotlar.',
        },
    },
};
