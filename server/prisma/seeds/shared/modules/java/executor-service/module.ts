import { Module } from '../../../../types';
import { topics } from './topics';

export const module: Module = {
    slug: 'java-executor-service',
    title: 'Executor Service',
    description: 'Master executor service framework: thread pools, callable/future, scheduled executors, and custom thread pool configurations.',
    order: 13,
    topics,
    translations: {
        ru: {
            title: 'Executor Service',
            description: 'Освойте фреймворк executor service: пулы потоков, callable/future, запланированные исполнители и настраиваемые конфигурации пулов потоков.',
        },
        uz: {
            title: 'Executor Service',
            description: 'Executor service freymvorkini o\'zlashtiring: oqim pullari, callable/future, rejalashtirilgan ijrochilar va maxsus oqim pullari konfiguratsiyalari.',
        },
    },
};
