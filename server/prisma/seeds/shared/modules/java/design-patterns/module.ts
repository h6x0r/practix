import { Module } from '../../../../types';
import { topics } from './topics';

export const module: Module = {
    title: 'Design Patterns',
    description: 'Master essential design patterns in Java: creational, structural, and behavioral patterns for production code.',
    section: 'production',
    order: 28,
    topics,
    translations: {
        ru: {
            title: 'Паттерны проектирования',
            description: 'Освойте основные паттерны проектирования в Java: порождающие, структурные и поведенческие паттерны для production-кода.',
        },
        uz: {
            title: 'Dizayn naqshlari',
            description: 'Java da muhim dizayn naqshlarini o\'rganing: yaratuvchi, strukturaviy va xulq-atvor naqshlari production kod uchun.',
        },
    },
};
