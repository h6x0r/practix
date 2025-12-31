import { Topic } from '../../../../types';
import { tasks } from './tasks';

export const fundamentals: Topic = {
    slug: 'java-atomic-fundamentals',
    title: 'Atomic Operations Fundamentals',
    description: 'Learn atomic variables, compare-and-swap, accumulators, and field updaters for lock-free programming',
    order: 0,
    tasks,
    translations: {
        ru: {
            title: 'Основы атомарных операций',
            description: 'Изучите атомарные переменные, compare-and-swap, аккумуляторы и field updaters для lock-free программирования',
        },
        uz: {
            title: 'Atomik Operatsiyalar Asoslari',
            description: 'Atomik o\'zgaruvchilar, compare-and-swap, akkumulyatorlar va lock-free dasturlash uchun field updaters ni o\'rganing',
        },
    },
};
