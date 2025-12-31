import { Module } from '../../../../types';
import { topics } from './topics';

export const module: Module = {
    slug: 'database',
    title: 'Database Patterns',
    description: 'Master Go database patterns: SQL queries, transactions, connection pooling, and production-ready data access.',
    difficulty: 'medium',
    estimatedTime: '2.5h',
    order: 25,
    isPremium: false,
    section: 'production',
    topics,
    translations: {
        ru: {
            title: 'Паттерны работы с базами данных',
            description: 'Освойте паттерны работы с БД в Go: SQL-запросы, транзакции, пулы соединений и промышленный доступ к данным.',
        },
        uz: {
            title: 'Ma\'lumotlar bazasi patternlari',
            description: 'Go da ma\'lumotlar bazasi patternlarini o\'rganing: SQL so\'rovlar, tranzaksiyalar, connection pooling va ishlab chiqarishga tayyor ma\'lumotlarga kirish.',
        },
    },
};
