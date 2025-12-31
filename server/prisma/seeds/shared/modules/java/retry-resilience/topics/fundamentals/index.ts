import { Topic } from '../../../../../types';
import { tasks } from './tasks';

export const fundamentals: Topic = {
    slug: 'java-retry-resilience-fundamentals',
    title: 'Retry & Resilience Fundamentals',
    description: 'Learn retry patterns, exponential backoff, and circuit breaker implementations.',
    order: 1,
    tasks,
    translations: {
        ru: {
            title: 'Основы повторов и устойчивости',
            description: 'Изучите паттерны повторов, экспоненциальную задержку и реализации circuit breaker.',
        },
        uz: {
            title: 'Retry va Chidamlilik Asoslari',
            description: 'Retry patternlari, eksponensial kechikish va circuit breaker amalga oshirishlarini o\'rganing.',
        },
    },
};
