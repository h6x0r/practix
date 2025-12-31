import { Topic } from '../../../../types';
import { tasks } from './tasks';

export const topic: Topic = {
    slug: 'java-jdbc-fundamentals',
    title: 'JDBC Fundamentals',
    description: 'Master Java Database Connectivity: connections, statements, prepared statements, result sets, transactions, and batch operations.',
    order: 0,
    tasks,
    translations: {
        ru: {
            title: 'Основы JDBC',
            description: 'Освойте Java Database Connectivity: соединения, statements, prepared statements, result sets, транзакции и пакетные операции.',
        },
        uz: {
            title: 'JDBC Asoslari',
            description: 'Java Database Connectivity ni o\'rganing: ulanishlar, statements, prepared statements, result sets, tranzaksiyalar va paketli operatsiyalar.',
        },
    },
};
