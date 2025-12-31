import { Topic } from '../../../../types';
import { tasks } from './tasks';

export const topic: Topic = {
    slug: 'java-fundamentals',
    title: 'Java Fundamentals',
    description: 'Core Java syntax: variables, operators, control flow, and strings.',
    order: 0,
    tasks,
    translations: {
        ru: {
            title: 'Основы Java',
            description: 'Основной синтаксис Java: переменные, операторы, управление потоком и строки.',
        },
        uz: {
            title: 'Java asoslari',
            description: 'Asosiy Java sintaksisi: o\'zgaruvchilar, operatorlar, boshqaruv oqimi va stringlar.',
        },
    },
};
