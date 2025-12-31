import { Module } from '../../../../types';
import { topics } from './topics';

export const module: Module = {
    slug: 'java-virtual-threads',
    title: 'Virtual Threads (Java 21+)',
    description: 'Master virtual threads: lightweight concurrency with Thread.ofVirtual, executors, structured concurrency, and migration patterns for modern Java applications.',
    order: 25,
    topics,
    translations: {
        ru: {
            title: 'Виртуальные потоки (Java 21+)',
            description: 'Освойте виртуальные потоки: легковесный параллелизм с Thread.ofVirtual, исполнители, структурированный параллелизм и паттерны миграции для современных Java-приложений.',
        },
        uz: {
            title: 'Virtual oqimlar (Java 21+)',
            description: 'Virtual oqimlarni o\'zlashtiring: Thread.ofVirtual bilan yengil parallellik, ijrochilar, strukturalashtirilgan parallellik va zamonaviy Java ilovalari uchun migratsiya patternlari.',
        },
    },
};
