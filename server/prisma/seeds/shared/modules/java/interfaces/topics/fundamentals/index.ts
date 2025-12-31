import { Topic } from '../../../../types';
import { tasks } from './tasks';

export const topic: Topic = {
    slug: 'java-interface-fundamentals',
    title: 'Interface Fundamentals',
    description: 'Master Java interfaces from basics to advanced features',
    order: 1,
    tasks,
    translations: {
        ru: {
            title: 'Основы интерфейсов',
            description: 'Освойте интерфейсы Java от основ до продвинутых возможностей',
        },
        uz: {
            title: 'Interfeys Asoslari',
            description: 'Java interfeyslarini asoslardan ilg\'or xususiyatlargacha o\'rganing',
        },
    },
};