import { Module } from '../../../../types';
import { topics } from './topics';

export const module: Module = {
    slug: 'java-connection-pooling',
    title: 'Connection Pooling',
    description: 'Master database connection pooling: HikariCP setup, pool configuration, monitoring, and best practices for efficient resource management.',
    order: 21,
    topics,
    translations: {
        ru: {
            title: 'Пулы подключений',
            description: 'Освойте пулы подключений к базам данных: настройка HikariCP, конфигурация пула, мониторинг и лучшие практики для эффективного управления ресурсами.',
        },
        uz: {
            title: 'Ulanish pullari',
            description: 'Ma\'lumotlar bazasiga ulanish pullarini o\'rganing: HikariCP sozlash, pull konfiguratsiyasi, monitoring va resurslarni samarali boshqarish uchun eng yaxshi amaliyotlar.',
        },
    },
};
