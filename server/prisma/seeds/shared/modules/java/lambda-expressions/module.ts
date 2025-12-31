import { Module } from '../../../../types';
import { topics } from './topics';

export const module: Module = {
    slug: 'java-lambda-expressions',
    title: 'Lambda Expressions',
    description: 'Master lambda expressions: syntax, functional interfaces, and method references.',
    order: 8,
    topics,
    translations: {
        ru: {
            title: 'Лямбда-выражения',
            description: 'Освойте лямбда-выражения: синтаксис, функциональные интерфейсы и ссылки на методы.',
        },
        uz: {
            title: 'Lambda Ifodalari',
            description: 'Lambda ifodalarini o\'rganing: sintaksis, funksional interfeyslari va metodlarga havolalar.',
        },
    },
};
