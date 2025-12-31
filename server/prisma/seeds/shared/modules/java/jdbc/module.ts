import { Module } from '../../../../types';
import * as topics from './topics';

export const module: Module = {
    slug: 'java-jdbc',
    title: 'JDBC Database Access',
    description: 'Master database connectivity with JDBC: connections, statements, transactions, and best practices.',
    order: 20,
    topics: [
        topics.fundamentals,
    ],
    translations: {
        ru: {
            title: 'Доступ к базам данных JDBC',
            description: 'Освойте подключение к базам данных с JDBC: соединения, операторы, транзакции и лучшие практики.',
        },
        uz: {
            title: 'JDBC ma\'lumotlar bazasiga kirish',
            description: 'JDBC bilan ma\'lumotlar bazasiga ulanishni o\'rganing: ulanishlar, so\'rovlar, tranzaksiyalar va eng yaxshi amaliyotlar.',
        },
    },
};
