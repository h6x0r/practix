import { Module } from '../../../../types';
import * as topics from './topics';

export const module: Module = {
    slug: 'java-testing',
    title: 'Testing with JUnit',
    description: 'Master unit testing in Java with JUnit 5 and Mockito: assertions, test lifecycle, parameterized tests, and mocking.',
    order: 27,
    topics: [
        topics.fundamentals,
    ],
    translations: {
        ru: {
            title: 'Тестирование с JUnit',
            description: 'Освойте модульное тестирование в Java с JUnit 5 и Mockito: утверждения, жизненный цикл тестов, параметризованные тесты и моки.',
        },
        uz: {
            title: 'JUnit bilan test qilish',
            description: 'Java da JUnit 5 va Mockito bilan birlik testlarini o\'rganing: assertlar, test hayot tsikli, parametrli testlar va moklash.',
        },
    },
};
