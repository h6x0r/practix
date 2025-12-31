import { Module } from '../../../../types';
import { topics } from './topics';

export const module: Module = {
    slug: 'java-date-time',
    title: 'Date and Time API',
    description: 'Master the modern Java Date and Time API: LocalDate, LocalDateTime, ZonedDateTime, formatting, and temporal operations.',
    order: 11,
    topics,
    translations: {
        ru: {
            title: 'API даты и времени',
            description: 'Освойте современный API даты и времени Java: LocalDate, LocalDateTime, ZonedDateTime, форматирование и временные операции.',
        },
        uz: {
            title: 'Sana va Vaqt API',
            description: 'Zamonaviy Java Sana va Vaqt API ni o\'rganing: LocalDate, LocalDateTime, ZonedDateTime, formatlash va vaqtinchalik operatsiyalar.',
        },
    },
};
