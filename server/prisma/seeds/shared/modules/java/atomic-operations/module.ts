import { Module } from '../../../../types';
import { topics } from './topics';

export const module: Module = {
    slug: 'java-atomic-operations',
    title: 'Atomic Operations',
    description: 'Master atomic operations and lock-free programming with java.util.concurrent.atomic package.',
    order: 17,
    topics,
    translations: {
        ru: {
            title: 'Атомарные операции',
            description: 'Освойте атомарные операции и программирование без блокировок с пакетом java.util.concurrent.atomic.',
        },
        uz: {
            title: 'Atom operatsiyalari',
            description: 'Atom operatsiyalari va blokirovkasiz dasturlashni java.util.concurrent.atomic paketi bilan o\'zlashtiring.',
        },
    },
};
