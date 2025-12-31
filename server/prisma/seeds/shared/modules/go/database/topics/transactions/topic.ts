import { Topic } from '../../../../types';
import { tasks } from './tasks';

export const topic: Topic = {
    slug: 'transactions',
    title: 'Transactions',
    description: 'Master transaction patterns: commit/rollback flows, isolation levels, savepoints, and handling serialization errors.',
    order: 1,
    tasks,
    translations: {
        ru: {
            title: 'Транзакции',
            description: 'Освоение паттернов транзакций: потоки commit/rollback, уровни изоляции, точки сохранения и обработка ошибок сериализации.'
        },
        uz: {
            title: 'Tranzaksiyalar',
            description: 'Tranzaksiya namunalarini o\'zlashtirish: commit/rollback oqimlari, izolyatsiya darajalari, saqlash nuqtalari va serializatsiya xatolarini boshqarish.'
        }
    }
};
