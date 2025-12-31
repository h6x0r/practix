import { Topic } from '../../../../../types';
import { tasks } from './tasks';

export const fundamentals: Topic = {
    slug: 'java-design-patterns-fundamentals',
    title: 'Design Patterns Fundamentals',
    description: 'Learn essential design patterns: Singleton, Builder, Factory, and Strategy patterns.',
    order: 1,
    tasks,
    translations: {
        ru: {
            title: 'Основы паттернов проектирования',
            description: 'Изучите основные паттерны проектирования: Singleton, Builder, Factory и Strategy.',
        },
        uz: {
            title: 'Dizayn Patternlari Asoslari',
            description: 'Asosiy dizayn patternlarini o\'rganing: Singleton, Builder, Factory va Strategy.',
        },
    },
};
