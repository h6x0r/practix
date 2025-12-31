import { Module } from '../../../../types';
import { topics } from './topics';

export const module: Module = {
    slug: 'java-collections-list',
    title: 'Collections: List Interface',
    description: 'Master List collections: ArrayList, LinkedList, iteration, and sorting.',
    order: 4,
    topics,
    translations: {
        ru: {
            title: 'Коллекции: интерфейс List',
            description: 'Освойте коллекции List: ArrayList, LinkedList, итерация и сортировка.',
        },
        uz: {
            title: 'Kolleksiyalar: List interfeysi',
            description: 'List kolleksiyalarini o\'zlashtiring: ArrayList, LinkedList, iteratsiya va saralash.',
        },
    },
};
