import { Module } from '../../../../types';
import { topics } from './topics';

export const module: Module = {
    slug: 'java-pattern-matching',
    title: 'Pattern Matching',
    description: 'Master modern pattern matching in Java: instanceof patterns, switch patterns, record patterns, and guarded patterns.',
    order: 24,
    topics,
    translations: {
        ru: {
            title: 'Сопоставление с образцом',
            description: 'Освойте современное сопоставление с образцом в Java: шаблоны instanceof, switch, записей и защищенные шаблоны.',
        },
        uz: {
            title: 'Namuna Moslashtirish',
            description: 'Java da zamonaviy namuna moslashtirishni o\'rganing: instanceof naqshlari, switch naqshlari, yozuv naqshlari va himoyalangan naqshlar.',
        },
    },
};
