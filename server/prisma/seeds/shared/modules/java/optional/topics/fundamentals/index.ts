import { Topic } from '../../../../types';
import { tasks } from './tasks';

export const fundamentals: Topic = {
    slug: 'java-optional-fundamentals',
    title: 'Optional Fundamentals',
    description: 'Learn the Optional class: creation, methods, transformations, patterns, and anti-patterns.',
    order: 1,
    tasks,
    translations: {
        ru: {
            title: 'Основы Optional',
            description: 'Изучите класс Optional: создание, методы, трансформации, паттерны и анти-паттерны.',
        },
        uz: {
            title: 'Optional Asoslari',
            description: 'Optional klassini o\'rganing: yaratish, metodlar, transformatsiyalar, namunalar va anti-namunalar.',
        },
    },
};
