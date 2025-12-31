import { Topic } from '../../../../types';
import { tasks } from './tasks';

export const fundamentals: Topic = {
    slug: 'java-concurrent-collections-fundamentals',
    title: 'Concurrent Collections Fundamentals',
    description: 'Learn thread-safe collections, blocking queues, and synchronization utilities',
    order: 0,
    tasks,
    translations: {
        ru: {
            title: 'Основы конкурентных коллекций',
            description: 'Изучите потокобезопасные коллекции, блокирующие очереди и утилиты синхронизации',
        },
        uz: {
            title: 'Parallel Kolleksiyalar Asoslari',
            description: 'Thread-safe kolleksiyalar, blokirovkalovchi navbatlar va sinxronizatsiya utilitalarini o\'rganing',
        },
    },
};
