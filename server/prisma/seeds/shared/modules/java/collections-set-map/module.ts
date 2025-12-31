import { Module } from '../../../../types';
import { topics } from './topics';

export const module: Module = {
    slug: 'java-collections-set-map',
    title: 'Collections: Set and Map',
    description: 'Master Set and Map collections: HashSet, TreeSet, HashMap, TreeMap.',
    order: 5,
    topics,
    translations: {
        ru: {
            title: 'Коллекции: Set и Map',
            description: 'Освойте коллекции Set и Map: HashSet, TreeSet, HashMap, TreeMap.',
        },
        uz: {
            title: 'Kolleksiyalar: Set va Map',
            description: 'Set va Map kolleksiyalarini o\'zlashtiring: HashSet, TreeSet, HashMap, TreeMap.',
        },
    },
};
