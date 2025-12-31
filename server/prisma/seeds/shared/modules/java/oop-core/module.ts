import { Module } from '../../../../types';
import * as topics from './topics';

export const module: Module = {
    slug: 'java-oop-core',
    title: 'Object-Oriented Programming Core',
    description: 'Master OOP fundamentals: classes, inheritance, polymorphism, and encapsulation.',
    order: 1,
    topics: [
        topics.fundamentals,
    ],
    translations: {
        ru: {
            title: 'Основы объектно-ориентированного программирования',
            description: 'Освойте основы ООП: классы, наследование, полиморфизм и инкапсуляцию.',
        },
        uz: {
            title: 'Obyektga yo\'naltirilgan dasturlash asoslari',
            description: 'OOP asoslarini o\'zlashtiring: sinflar, meros, polimorfizm va inkapsulyatsiya.',
        },
    },
};