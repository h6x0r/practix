import { Topic } from '../../../../types';
import { tasks } from './tasks';

export const fundamentals: Topic = {
    slug: 'java-http-clients-fundamentals',
    title: 'HTTP Client Fundamentals',
    description: 'Learn Java HTTP Client API for synchronous and asynchronous HTTP requests',
    order: 0,
    tasks,
    translations: {
        ru: {
            title: 'Основы HTTP клиента',
            description: 'Изучите Java HTTP Client API для синхронных и асинхронных HTTP запросов',
        },
        uz: {
            title: 'HTTP Klient Asoslari',
            description: 'Sinxron va asinxron HTTP so\'rovlari uchun Java HTTP Client API ni o\'rganing',
        },
    },
};
