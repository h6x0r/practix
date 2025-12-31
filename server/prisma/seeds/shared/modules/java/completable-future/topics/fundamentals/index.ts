import { Topic } from '../../../../types';
import { tasks } from './tasks';

export const topic: Topic = {
    slug: 'java-completable-fundamentals',
    title: 'CompletableFuture Fundamentals',
    description: 'Master CompletableFuture for asynchronous programming, composition, error handling, and best practices.',
    order: 1,
    tasks,
    translations: {
        ru: {
            title: 'Основы CompletableFuture',
            description: 'Освойте CompletableFuture для асинхронного программирования, композиции, обработки ошибок и лучших практик.',
        },
        uz: {
            title: 'CompletableFuture Asoslari',
            description: 'Asinxron dasturlash, kompozitsiya, xatolarni boshqarish va eng yaxshi amaliyotlar uchun CompletableFuture ni o\'rganing.',
        },
    },
};
