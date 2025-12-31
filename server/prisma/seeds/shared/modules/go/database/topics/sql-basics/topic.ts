import { Topic } from '../../../../types';
import { tasks } from './tasks';

export const topic: Topic = {
    slug: 'sql-basics',
    title: 'SQL Basics',
    description: 'Learn fundamental database/sql patterns: querying rows, handling results, prepared statements, and context management.',
    order: 0,
    tasks,
    translations: {
        ru: {
            title: 'Основы SQL',
            description: 'Изучение фундаментальных паттернов работы с базой данных/SQL: запросы строк, обработка результатов, подготовленные выражения и управление контекстом.'
        },
        uz: {
            title: 'SQL asoslari',
            description: 'Ma\'lumotlar bazasi/SQL bilan ishlashning asosiy namunalarini o\'rganish: qatorlarni so\'rov qilish, natijalarni qayta ishlash, tayyorlangan so\'rovlar va kontekstni boshqarish.'
        }
    }
};
