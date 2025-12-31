import { Module } from '../../../../types';
import { topics } from './topics';

export const module: Module = {
    slug: 'java-io-streams',
    title: 'I/O Streams',
    description: 'Master Java I/O streams: byte streams, character streams, buffered I/O, and serialization.',
    order: 18,
    topics,
    translations: {
        ru: {
            title: 'Потоки ввода-вывода',
            description: 'Освойте потоки ввода-вывода Java: байтовые потоки, символьные потоки, буферизованный ввод-вывод и сериализация.',
        },
        uz: {
            title: 'Kirish-chiqish oqimlari',
            description: 'Java I/O oqimlarini o\'rganing: bayt oqimlari, belgi oqimlari, buferli I/O va seriyalizatsiya.',
        },
    },
};
