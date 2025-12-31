import { Topic } from '../../../../types';
import { tasks } from './tasks';

export const topic: Topic = {
    slug: 'java-executor-fundamentals',
    title: 'Executor Service Fundamentals',
    description: 'Learn ExecutorService framework, thread pools, callable/future pattern, scheduled executors, and custom thread pool configurations.',
    order: 1,
    tasks,
    translations: {
        ru: {
            title: 'Основы ExecutorService',
            description: 'Изучите фреймворк ExecutorService, пулы потоков, паттерн callable/future, планировщики и настройку пулов потоков.',
        },
        uz: {
            title: 'ExecutorService Asoslari',
            description: 'ExecutorService frameworki, thread havzalari, callable/future namunasi, rejalashtiruvchilar va thread havzasi sozlamalarini o\'rganing.',
        },
    },
};
