import { Module } from '../../../../types';
import { topics } from './topics';

export const module: Module = {
    slug: 'java-exception-handling',
    title: 'Exception Handling',
    description: 'Master exception handling: try-catch, custom exceptions, and best practices.',
    order: 3,
    topics,
    translations: {
        ru: {
            title: 'Обработка исключений',
            description: 'Освойте обработку исключений: try-catch, пользовательские исключения и лучшие практики.',
        },
        uz: {
            title: 'Istisno holatlarni boshqarish',
            description: 'Istisno holatlarni boshqarishni o\'zlashtiring: try-catch, maxsus istisnolar va eng yaxshi amaliyotlar.',
        },
    },
};