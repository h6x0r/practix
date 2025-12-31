import { Module } from '../../../../types';
import { topics } from './topics';

export const module: Module = {
    slug: 'java-locks-advanced',
    title: 'Advanced Locks',
    description: 'Master advanced locking mechanisms: ReentrantLock, ReadWriteLock, StampedLock, and lock patterns.',
    order: 16,
    topics,
    translations: {
        ru: {
            title: 'Продвинутые блокировки',
            description: 'Освойте продвинутые механизмы блокировки: ReentrantLock, ReadWriteLock, StampedLock и паттерны блокировки.',
        },
        uz: {
            title: 'Ilg\'or blokirovkalar',
            description: 'Ilg\'or blokirovka mexanizmlarini o\'zlashtiring: ReentrantLock, ReadWriteLock, StampedLock va blokirovka patternlari.',
        },
    },
};
