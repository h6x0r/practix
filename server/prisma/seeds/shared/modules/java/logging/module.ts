import { Module } from '../../../../types';
import { topics } from './topics';

export const module: Module = {
    slug: 'java-logging',
    title: 'Logging with SLF4J',
    description: 'Master Java logging with SLF4J, Logback, and best practices.',
    order: 28,
    topics,
    translations: {
        ru: {
            title: 'Логирование с SLF4J',
            description: 'Освойте логирование в Java с SLF4J, Logback и лучшие практики.',
        },
        uz: {
            title: 'SLF4J bilan loglash',
            description: 'Java da SLF4J, Logback bilan loglashtirish va eng yaxshi amaliyotlarni o\'rganing.',
        },
    },
};
