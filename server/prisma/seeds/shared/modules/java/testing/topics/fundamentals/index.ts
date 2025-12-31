import { Topic } from '../../../../types';
import { tasks } from './tasks';

export const topic: Topic = {
    slug: 'java-testing-fundamentals',
    title: 'Testing Fundamentals',
    description: 'Master unit testing in Java with JUnit 5 and Mockito: write effective tests, use assertions, manage test lifecycle, and mock dependencies.',
    order: 0,
    tasks,
    translations: {
        ru: {
            title: 'Основы тестирования',
            description: 'Освойте модульное тестирование в Java с JUnit 5 и Mockito: пишите эффективные тесты, используйте assertions, управляйте жизненным циклом тестов и мокайте зависимости.',
        },
        uz: {
            title: 'Testlash Asoslari',
            description: 'JUnit 5 va Mockito bilan Java-da unit testlashni o\'rganing: samarali testlar yozing, assertions dan foydalaning, test hayot siklini boshqaring va bog\'liqliklarni mock qiling.',
        },
    },
};
