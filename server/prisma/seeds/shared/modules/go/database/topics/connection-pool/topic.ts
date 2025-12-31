import { Topic } from '../../../../types';
import { tasks } from './tasks';

export const topic: Topic = {
    slug: 'connection-pool',
    title: 'Connection Pool',
    description: 'Configure and manage database connection pools: limits, lifetimes, health checks, and graceful shutdown patterns.',
    order: 2,
    tasks,
    translations: {
        ru: {
            title: 'Пул соединений',
            description: 'Настройка и управление пулами подключений к базе данных: лимиты, время жизни, проверки работоспособности и паттерны корректного завершения.'
        },
        uz: {
            title: 'Ulanish havzasi',
            description: 'Ma\'lumotlar bazasi ulanish havzalarini sozlash va boshqarish: cheklovlar, yashash muddati, sog\'likni tekshirish va to\'g\'ri to\'xtatish namunalari.'
        }
    }
};
