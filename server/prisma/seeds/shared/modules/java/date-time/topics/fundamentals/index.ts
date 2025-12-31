import { Topic } from '../../../../types';
import { tasks } from './tasks';

export const topic: Topic = {
    slug: 'date-time-fundamentals',
    title: 'Date and Time Fundamentals',
    description: 'Learn the Java Date and Time API fundamentals: LocalDate, LocalDateTime, ZonedDateTime, formatting, and temporal operations.',
    order: 1,
    tasks,
    translations: {
        ru: {
            title: 'Основы даты и времени',
            description: 'Изучите основы Java Date and Time API: LocalDate, LocalDateTime, ZonedDateTime, форматирование и временные операции.',
        },
        uz: {
            title: 'Sana va Vaqt Asoslari',
            description: 'Java Date and Time API asoslarini o\'rganing: LocalDate, LocalDateTime, ZonedDateTime, formatlash va vaqt operatsiyalari.',
        },
    },
};
