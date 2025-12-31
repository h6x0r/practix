import { Topic } from '../../../../types';
import { tasks } from './tasks';

export const topic: Topic = {
    slug: 'stream-api-fundamentals',
    title: 'Stream API Fundamentals',
    description: 'Learn Java Stream API operations, collectors, and parallel processing.',
    order: 1,
    tasks,
    translations: {
        ru: {
            title: 'Основы Stream API',
            description: 'Изучите операции Java Stream API, коллекторы и параллельную обработку.',
        },
        uz: {
            title: 'Stream API Asoslari',
            description: 'Java Stream API operatsiyalari, kollektorlar va parallel ishlov berishni o\'rganing.',
        },
    },
};
