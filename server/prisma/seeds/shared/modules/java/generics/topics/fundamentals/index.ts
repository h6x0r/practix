import { Topic } from '../../../../types';
import { tasks } from './tasks';

export const fundamentals: Topic = {
    slug: 'java-generics-fundamentals',
    title: 'Java Generics Fundamentals',
    description: 'Learn generic classes, methods, wildcards, bounds, type erasure, and patterns',
    order: 0,
    tasks,
    translations: {
        ru: {
            title: 'Основы Java Generics',
            description: 'Изучите обобщенные классы, методы, wildcards, границы, стирание типов и паттерны',
        },
        uz: {
            title: 'Java Generics Asoslari',
            description: 'Umumiy sinflar, metodlar, wildcards, chegaralar, tip o\'chirish va namunalarni o\'rganing',
        },
    },
};
