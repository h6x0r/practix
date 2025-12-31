import { Topic } from '../../../../types';
import { tasks } from './tasks';

export const topic: Topic = {
    slug: 'java-queue-fundamentals',
    title: 'Queue and Deque Fundamentals',
    description: 'Master Queue, Deque, PriorityQueue, ArrayDeque, and queue processing patterns.',
    order: 0,
    tasks,
    translations: {
        ru: {
            title: 'Основы Queue и Deque',
            description: 'Освойте Queue, Deque, PriorityQueue, ArrayDeque и паттерны обработки очередей.',
        },
        uz: {
            title: 'Queue va Deque Asoslari',
            description: 'Queue, Deque, PriorityQueue, ArrayDeque va navbat ishlov berish namunalarini o\'rganing.',
        },
    },
};
