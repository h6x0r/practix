import { Module } from '../../../../types';
import { topics } from './topics';

export const module: Module = {
    slug: 'generics',
    title: 'Go Generics',
    description: 'Master Go generics: type parameters, constraints, and generic data structures for reusable code.',
    difficulty: 'medium',
    estimatedTime: '1.5h',
    order: 24,
    isPremium: false,
    section: 'core',
    topics,
    translations: {
        ru: {
            title: 'Обобщённые типы в Go',
            description: 'Освойте обобщённые типы в Go: параметры типов, ограничения и обобщённые структуры данных для повторно используемого кода.'
        },
        uz: {
            title: 'Go da Generiklar',
            description: 'Go generiklari: tip parametrlari, cheklovlar va qayta foydalaniladigan kod uchun umumiy ma\'lumotlar tuzilmalarini o\'rganing.'
        }
    }
};
