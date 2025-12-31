import { Module } from '../../../../types';
import * as topics from './topics';

export const module: Module = {
    slug: 'java-metrics',
    title: 'Metrics with Micrometer',
    description: 'Master application metrics and monitoring with Micrometer: counters, gauges, timers, and best practices.',
    order: 29,
    topics: [
        topics.fundamentals,
    ],
    translations: {
        ru: {
            title: 'Метрики с Micrometer',
            description: 'Освойте метрики и мониторинг приложений с Micrometer: счётчики, индикаторы, таймеры и лучшие практики.',
        },
        uz: {
            title: 'Micrometer bilan metrikalar',
            description: 'Micrometer bilan ilova metrikalarini va monitoringni o\'rganing: hisoblagichlar, o\'lchagichlar, taymerlar va eng yaxshi amaliyotlar.',
        },
    },
};
