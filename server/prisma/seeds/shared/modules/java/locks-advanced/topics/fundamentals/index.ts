import { Topic } from '../../../../types';
import { tasks } from './tasks';

export const topic: Topic = {
    slug: 'java-locks-advanced-fundamentals',
    title: 'Advanced Locks Fundamentals',
    description: 'Master ReentrantLock, ReadWriteLock, StampedLock, Condition variables, and lock patterns.',
    order: 1,
    tasks,
    translations: {
        ru: {
            title: 'Основы продвинутых блокировок',
            description: 'Освойте ReentrantLock, ReadWriteLock, StampedLock, условные переменные и паттерны блокировок.',
        },
        uz: {
            title: 'Murakkab Qulflar Asoslari',
            description: 'ReentrantLock, ReadWriteLock, StampedLock, Condition o\'zgaruvchilari va qulf namunalarini o\'rganing.',
        },
    },
};
