import { Topic } from '../../../../types';
import { tasks } from './tasks';

export const topic: Topic = {
    slug: 'java-oop-fundamentals',
    title: 'OOP Fundamentals',
    description: 'Master core object-oriented programming concepts in Java: classes, objects, inheritance, polymorphism, encapsulation, and abstraction.',
    order: 0,
    tasks,
    translations: {
        ru: {
            title: 'Основы ООП',
            description: 'Освойте основные концепции объектно-ориентированного программирования в Java: классы, объекты, наследование, полиморфизм, инкапсуляция и абстракция.',
        },
        uz: {
            title: 'OOP Asoslari',
            description: 'Java-da obyektga yo\'naltirilgan dasturlashning asosiy tushunchalarini o\'rganing: sinflar, obyektlar, meros, polimorfizm, inkapsulyatsiya va abstraktsiya.',
        },
    },
};