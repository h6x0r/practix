import { Topic } from '../../../../types';
import { tasks } from './tasks';

export const topic: Topic = {
    slug: 'fundamentals',
    title: 'Generics Fundamentals',
    description: 'Learn Go generics with type parameters, constraints, and reusable generic functions and data structures.',
    order: 0,
    tasks,
    translations: {
        ru: {
            title: 'Основы дженериков',
            description: 'Изучите дженерики Go с параметрами типов, ограничениями и повторно используемыми обобщенными функциями и структурами данных.'
        },
        uz: {
            title: 'Generiklar asoslari',
            description: 'Tip parametrlari, cheklovlar va qayta foydalaniladigan generik funksiyalar va ma\'lumotlar tuzilmalari bilan Go generiklarini o\'rganing.'
        }
    }
};
