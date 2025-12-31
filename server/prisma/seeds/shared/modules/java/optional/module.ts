import { Module } from '../../../../types';
import { topics } from './topics';

export const module: Module = {
    slug: 'java-optional',
    title: 'Optional Class',
    description: 'Master the Optional class: creating, using, transforming, and avoiding null pointer exceptions.',
    order: 10,
    topics,
    translations: {
        ru: {
            title: 'Класс Optional',
            description: 'Освойте класс Optional: создание, использование, преобразование и предотвращение исключений нулевого указателя.',
        },
        uz: {
            title: 'Optional Klassi',
            description: 'Optional klassini o\'rganing: yaratish, ishlatish, o\'zgartirish va null pointer istisnolaridan qochish.',
        },
    },
};
