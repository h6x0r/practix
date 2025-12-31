import { Module } from '../../../../types';
import { topics } from './topics';

export const module: Module = {
    slug: 'java-sealed-classes',
    title: 'Sealed Classes (Java 17+)',
    description: 'Master sealed classes and interfaces for controlled inheritance hierarchies and pattern matching.',
    order: 23,
    topics,
    translations: {
        ru: {
            title: 'Запечатанные классы (Java 17+)',
            description: 'Освойте запечатанные классы и интерфейсы для контролируемых иерархий наследования и сопоставления с образцом.',
        },
        uz: {
            title: 'Muhrlangan Klasslar (Java 17+)',
            description: 'Nazorat qilinadigan meros ierarxiyalari va namuna moslashtirish uchun muhrlangan klasslar va interfeyslari o\'rganing.',
        },
    },
};
