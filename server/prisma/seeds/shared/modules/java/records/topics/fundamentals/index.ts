import { Topic } from '../../../../types';
import { tasks } from './tasks';

export const fundamentals: Topic = {
    slug: 'java-records-fundamentals',
    title: 'Records Fundamentals',
    description: 'Learn Java records: declaration, compact constructors, customization, patterns, and limitations.',
    order: 1,
    tasks,
    translations: {
        ru: {
            title: 'Основы Records',
            description: 'Изучите Java records: объявление, компактные конструкторы, настройка, паттерны и ограничения.',
        },
        uz: {
            title: 'Records Asoslari',
            description: 'Java records ni o\'rganing: e\'lon qilish, ixcham konstruktorlar, sozlash, namunalar va cheklovlar.',
        },
    },
};
