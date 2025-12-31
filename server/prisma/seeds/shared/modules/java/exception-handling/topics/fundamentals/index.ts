import { Topic } from '../../../../types';
import { tasks } from './tasks';

export const topic: Topic = {
    slug: 'java-exception-fundamentals',
    title: 'Exception Handling Fundamentals',
    description: 'Master exception handling in Java: try-catch, custom exceptions, resources, and best practices.',
    order: 1,
    tasks,
    translations: {
        ru: {
            title: 'Основы обработки исключений',
            description: 'Освойте обработку исключений в Java: try-catch, пользовательские исключения, ресурсы и лучшие практики.',
        },
        uz: {
            title: 'Istisno Holatlari Asoslari',
            description: 'Java-da istisno holatlarini boshqarishni o\'rganing: try-catch, maxsus istisnolar, resurslar va eng yaxshi amaliyotlar.',
        },
    },
};