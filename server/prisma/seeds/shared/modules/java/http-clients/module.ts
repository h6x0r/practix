import { Module } from '../../../../types';
import { topics } from './topics';

export const module: Module = {
    slug: 'java-http-clients',
    title: 'HTTP Clients',
    description: 'Master Java HTTP Client API for making HTTP requests, handling responses, and working with modern HTTP features.',
    order: 34,
    topics,
    translations: {
        ru: {
            title: 'HTTP-клиенты',
            description: 'Освойте Java HTTP Client API для выполнения HTTP-запросов, обработки ответов и работы с современными функциями HTTP.',
        },
        uz: {
            title: 'HTTP mijozlar',
            description: 'Java HTTP Client API ni o\'rganing: HTTP so\'rovlarini yuborish, javoblarni qayta ishlash va zamonaviy HTTP xususiyatlari bilan ishlash.',
        },
    },
};
