import { Topic } from '../../../../types';
import { tasks } from './tasks';

export const topic: Topic = {
    slug: 'java-caching-fundamentals',
    title: 'Caching Fundamentals',
    description: 'Master Caffeine caching: basic operations, cache loading, eviction policies, statistics, async caching, and common cache patterns.',
    order: 0,
    tasks,
    translations: {
        ru: {
            title: 'Основы кэширования',
            description: 'Освойте кэширование с Caffeine: базовые операции, загрузка кэша, политики вытеснения, статистика, асинхронное кэширование и распространенные паттерны.',
        },
        uz: {
            title: 'Keshlash Asoslari',
            description: 'Caffeine bilan keshlashni o\'rganing: asosiy operatsiyalar, kesh yuklash, chiqarish siyosatlari, statistika, asinxron keshlash va keng tarqalgan namunalar.',
        },
    },
};
