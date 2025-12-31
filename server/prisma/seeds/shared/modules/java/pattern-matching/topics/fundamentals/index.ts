import { Topic } from '../../../../types';
import { tasks } from './tasks';

export const fundamentals: Topic = {
    slug: 'java-pattern-matching-fundamentals',
    title: 'Pattern Matching Fundamentals',
    description: 'Learn modern pattern matching in Java: instanceof patterns, switch expressions, record patterns, guarded patterns, and best practices.',
    order: 1,
    tasks,
    translations: {
        ru: {
            title: 'Основы Pattern Matching',
            description: 'Изучите современный pattern matching в Java: instanceof patterns, switch expressions, record patterns, guarded patterns и лучшие практики.',
        },
        uz: {
            title: 'Pattern Matching Asoslari',
            description: 'Java-da zamonaviy pattern matching ni o\'rganing: instanceof patterns, switch expressions, record patterns, guarded patterns va eng yaxshi amaliyotlar.',
        },
    },
};
