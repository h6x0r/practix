import { Topic } from '../../../../types';
import { tasks } from './tasks';

export const fundamentals: Topic = {
    slug: 'java-logging-fundamentals',
    title: 'Logging Fundamentals',
    description: 'Learn SLF4J, Logback, and logging best practices in Java.',
    order: 0,
    tasks,
    translations: {
        ru: {
            title: 'Основы логирования',
            description: 'Изучите SLF4J, Logback и лучшие практики логирования в Java.',
        },
        uz: {
            title: 'Logging Asoslari',
            description: 'SLF4J, Logback va Java-da logging eng yaxshi amaliyotlarini o\'rganing.',
        },
    },
};
