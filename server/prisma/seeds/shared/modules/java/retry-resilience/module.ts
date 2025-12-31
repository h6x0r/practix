import { Module } from '../../../../types';
import { topics } from './topics';

export const module: Module = {
    title: 'Retry & Resilience Patterns',
    description: 'Master retry patterns, circuit breakers, and resilience strategies for robust Java applications.',
    section: 'production',
    order: 33,
    topics,
    translations: {
        ru: {
            title: 'Паттерны повторных попыток и устойчивости',
            description: 'Освойте паттерны повторных попыток, автоматические выключатели и стратегии устойчивости для надёжных Java-приложений.',
        },
        uz: {
            title: 'Qayta urinish va barqarorlik naqshlari',
            description: 'Qayta urinish naqshlari, circuit breaker va ishonchli Java ilovalari uchun barqarorlik strategiyalarini o\'rganing.',
        },
    },
};
