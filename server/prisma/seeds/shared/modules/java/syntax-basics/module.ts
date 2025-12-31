import { Module } from '../../../../types';
import { topics } from './topics';

export const module: Module = {
    slug: 'java-syntax-basics',
    title: 'Java Syntax Basics',
    description: 'Master Java fundamentals: variables, types, operators, and control flow.',
    order: 0,
    topics,
    translations: {
        ru: {
            title: 'Основы синтаксиса Java',
            description: 'Освойте основы Java: переменные, типы, операторы и управление потоком.',
        },
        uz: {
            title: 'Java sintaksisi asoslari',
            description: 'Java asoslarini o\'zlashtiring: o\'zgaruvchilar, turlar, operatorlar va boshqaruv oqimi.',
        },
    },
};
