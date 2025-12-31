import { Topic } from '../../../../types';
import { tasks } from './tasks';

export const fundamentals: Topic = {
    slug: 'java-io-streams-fundamentals',
    title: 'Java I/O Streams Fundamentals',
    description: 'Learn byte streams, character streams, buffered I/O, data streams, object serialization',
    order: 0,
    tasks,
    translations: {
        ru: {
            title: 'Основы Java I/O потоков',
            description: 'Изучите байтовые потоки, символьные потоки, буферизованный I/O, потоки данных, сериализацию объектов',
        },
        uz: {
            title: 'Java I/O Oqimlari Asoslari',
            description: 'Bayt oqimlari, belgi oqimlari, buferlangan I/O, ma\'lumot oqimlari, obyekt serializatsiyasini o\'rganing',
        },
    },
};
