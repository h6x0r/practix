import { Topic } from '../../../../types';
import { tasks } from './tasks';

export const topic: Topic = {
    slug: 'java-lambda-fundamentals',
    title: 'Lambda Expressions Fundamentals',
    description: 'Master lambda expressions, functional interfaces, method references, and best practices.',
    order: 1,
    tasks,
    translations: {
        ru: {
            title: 'Основы лямбда-выражений',
            description: 'Освойте лямбда-выражения, функциональные интерфейсы, ссылки на методы и лучшие практики.',
        },
        uz: {
            title: 'Lambda Ifodalar Asoslari',
            description: 'Lambda ifodalari, funksional interfeyslar, metod havolalari va eng yaxshi amaliyotlarni o\'rganing.',
        },
    },
};
