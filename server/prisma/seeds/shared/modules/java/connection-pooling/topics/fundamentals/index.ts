import { Topic } from '../../../../types';
import { tasks } from './tasks';

export const topic: Topic = {
    slug: 'java-connection-pooling-fundamentals',
    title: 'Connection Pooling Fundamentals',
    description: 'Learn database connection pooling, HikariCP configuration, and best practices',
    order: 0,
    tasks,
    translations: {
        ru: {
            title: 'Основы пула соединений',
            description: 'Изучите пулы соединений с базой данных, настройку HikariCP и лучшие практики',
        },
        uz: {
            title: 'Ulanish Havzasi Asoslari',
            description: 'Ma\'lumotlar bazasi ulanish havzalari, HikariCP sozlash va eng yaxshi amaliyotlarni o\'rganing',
        },
    },
};
