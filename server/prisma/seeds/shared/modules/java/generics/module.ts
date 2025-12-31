import { Module } from '../../../../types';
import { topics } from './topics';

export const module: Module = {
    slug: 'java-generics',
    title: 'Java Generics',
    description: 'Master generics: type parameters, wildcards, bounds, and type erasure.',
    order: 7,
    topics,
    translations: {
        ru: {
            title: 'Обобщения Java',
            description: 'Освойте обобщения: параметры типов, подстановочные знаки, границы и стирание типов.',
        },
        uz: {
            title: 'Java umumlashtirish',
            description: 'Umumlashtirishni o\'zlashtiring: tur parametrlari, belgilar, chegaralar va tur o\'chirish.',
        },
    },
};
