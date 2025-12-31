import { Module } from '../../../../types';
import { topics } from './topics';

export const module: Module = {
    slug: 'java-completable-future',
    title: 'CompletableFuture',
    description: 'Master asynchronous programming with CompletableFuture: async operations, composition, and error handling.',
    order: 15,
    topics,
    translations: {
        ru: {
            title: 'CompletableFuture',
            description: 'Освойте асинхронное программирование с CompletableFuture: асинхронные операции, композиция и обработка ошибок.',
        },
        uz: {
            title: 'CompletableFuture',
            description: 'CompletableFuture bilan asinxron dasturlashni o\'zlashtiring: asinxron operatsiyalar, kompozitsiya va xatolarni qayta ishlash.',
        },
    },
};
