import { Topic } from '../../../../types';
import { tasks } from './tasks';

export const fundamentals: Topic = {
    slug: 'java-set-map-fundamentals',
    title: 'Set and Map Fundamentals',
    description: 'Master Set and Map collections: HashSet, TreeSet, HashMap, TreeMap, LinkedHashMap, and LRU Cache implementation.',
    order: 1,
    tasks,
    translations: {
        ru: {
            title: 'Основы Set и Map',
            description: 'Освойте коллекции Set и Map: HashSet, TreeSet, HashMap, TreeMap, LinkedHashMap и реализацию LRU кэша.',
        },
        uz: {
            title: 'Set va Map Asoslari',
            description: 'Set va Map kolleksiyalarini o\'rganing: HashSet, TreeSet, HashMap, TreeMap, LinkedHashMap va LRU keshi amalga oshirilishi.',
        },
    },
};
