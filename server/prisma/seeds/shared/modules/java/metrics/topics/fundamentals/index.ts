import { Topic } from '../../../../types';
import { tasks } from './tasks';

export const fundamentals: Topic = {
    slug: 'java-metrics-fundamentals',
    title: 'Metrics Fundamentals',
    description: 'Master metrics fundamentals with Micrometer',
    order: 0,
    tasks,
    translations: {
        ru: {
            title: 'Основы метрик',
            description: 'Освойте основы метрик с Micrometer',
        },
        uz: {
            title: 'Metrikalar Asoslari',
            description: 'Micrometer bilan metrikalar asoslarini o\'rganing',
        },
    },
};
