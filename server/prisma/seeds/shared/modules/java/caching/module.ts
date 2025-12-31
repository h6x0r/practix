import { Module } from '../../../../types';
import * as topics from './topics';

export const module: Module = {
    slug: 'java-caching',
    title: 'Caching with Caffeine',
    description: 'Master high-performance caching with Caffeine: cache creation, loading strategies, eviction policies, statistics, async operations, and cache patterns.',
    order: 33,
    topics: [
        topics.fundamentals,
    ],
    translations: {
        ru: {
            title: 'Кэширование с Caffeine',
            description: 'Освойте высокопроизводительное кэширование с Caffeine: создание кэша, стратегии загрузки, политики вытеснения, статистика, асинхронные операции и паттерны кэширования.',
        },
        uz: {
            title: 'Caffeine bilan keshlash',
            description: 'Caffeine bilan yuqori samarali keshlashni o\'rganing: kesh yaratish, yuklash strategiyalari, chiqarish siyosatlari, statistika, asinxron operatsiyalar va keshlash naqshlari.',
        },
    },
};
