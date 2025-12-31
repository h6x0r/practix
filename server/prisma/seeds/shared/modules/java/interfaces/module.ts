import { Module } from '../../../../types';
import { topics } from './topics';

export const module: Module = {
    slug: 'java-interfaces',
    title: 'Java Interfaces',
    description: 'Master interfaces: contracts, default methods, and functional interfaces.',
    order: 2,
    topics,
    translations: {
        ru: {
            title: 'Интерфейсы Java',
            description: 'Освойте интерфейсы: контракты, методы по умолчанию и функциональные интерфейсы.',
        },
        uz: {
            title: 'Java interfeyslari',
            description: 'Interfeyslarni o\'zlashtiring: shartnomalar, standart metodlar va funksional interfeyslar.',
        },
    },
};