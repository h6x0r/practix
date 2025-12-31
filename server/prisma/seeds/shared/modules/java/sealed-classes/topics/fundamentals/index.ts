import { Topic } from '../../../../types';
import { tasks } from './tasks';

export const fundamentals: Topic = {
    slug: 'java-sealed-classes-fundamentals',
    title: 'Sealed Classes Fundamentals',
    description: 'Learn sealed classes, interfaces, hierarchies, records integration, and pattern matching',
    order: 0,
    tasks,
    translations: {
        ru: {
            title: 'Основы Sealed классов',
            description: 'Изучите sealed классы, интерфейсы, иерархии, интеграцию с records и pattern matching',
        },
        uz: {
            title: 'Sealed Sinflar Asoslari',
            description: 'Sealed sinflar, interfeyslar, ierarxiyalar, records integratsiyasi va pattern matching ni o\'rganing',
        },
    },
};
