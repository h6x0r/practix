import { Module } from '../../../../types';
import { topics } from './topics';

export const module: Module = {
    slug: 'java-collections-queue',
    title: 'Collections: Queue and Deque',
    description: 'Master Queue collections: Queue, Deque, PriorityQueue, ArrayDeque.',
    order: 6,
    topics,
    translations: {
        ru: {
            title: 'Коллекции: Queue и Deque',
            description: 'Освойте коллекции Queue: Queue, Deque, PriorityQueue, ArrayDeque.',
        },
        uz: {
            title: 'Kolleksiyalar: Queue va Deque',
            description: 'Queue kolleksiyalarini o\'zlashtiring: Queue, Deque, PriorityQueue, ArrayDeque.',
        },
    },
};
