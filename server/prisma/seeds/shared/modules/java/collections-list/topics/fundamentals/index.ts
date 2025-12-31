import { Topic } from '../../../../types';
import { tasks } from './tasks';

export const fundamentals: Topic = {
    slug: 'java-list-fundamentals',
    title: 'List Interface Fundamentals',
    description: 'Learn ArrayList, LinkedList, iteration, sorting, and list operations',
    order: 0,
    tasks,
    translations: {
        ru: {
            title: 'Основы интерфейса List',
            description: 'Изучите ArrayList, LinkedList, итерацию, сортировку и операции со списками',
        },
        uz: {
            title: 'List Interfeysi Asoslari',
            description: 'ArrayList, LinkedList, iteratsiya, saralash va ro\'yxat operatsiyalarini o\'rganing',
        },
    },
};
