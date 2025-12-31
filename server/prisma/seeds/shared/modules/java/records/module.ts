import { Module } from '../../../../types';
import { topics } from './topics';

export const module: Module = {
    slug: 'java-records',
    title: 'Records (Java 16+)',
    description: 'Master Java records: creating immutable data carriers, compact constructors, customization, and best practices.',
    order: 22,
    topics,
    translations: {
        ru: {
            title: 'Записи (Java 16+)',
            description: 'Освойте записи Java: создание неизменяемых носителей данных, компактные конструкторы, настройка и лучшие практики.',
        },
        uz: {
            title: 'Yozuvlar (Java 16+)',
            description: 'Java yozuvlarini o\'rganing: o\'zgarmas ma\'lumot tashuvchilarni yaratish, ixcham konstruktorlar, sozlash va eng yaxshi amaliyotlar.',
        },
    },
};
