import { Topic } from '../../../../types';
import { tasks } from './tasks';

export const topic: Topic = {
    slug: 'java-virtual-threads-fundamentals',
    title: 'Virtual Threads Fundamentals',
    description: 'Master virtual threads in Java 21+: lightweight concurrency, executors, structured concurrency, and migration patterns.',
    order: 1,
    tasks,
    translations: {
        ru: {
            title: 'Основы виртуальных потоков',
            description: 'Освойте виртуальные потоки в Java 21+: легковесная конкурентность, executors, структурированная конкурентность и паттерны миграции.',
        },
        uz: {
            title: 'Virtual Threads Asoslari',
            description: 'Java 21+ da virtual threads ni o\'rganing: yengil parallellik, executors, tuzilgan parallellik va migratsiya namunalari.',
        },
    },
};
