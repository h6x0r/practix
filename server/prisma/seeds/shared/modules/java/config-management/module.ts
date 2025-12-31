import { Module } from '../../../../types';
import { topics } from './topics';

export const module: Module = {
    slug: 'java-config-management',
    title: 'Configuration Management',
    description: 'Master Java configuration management: properties files, YAML, environment variables, and configuration patterns.',
    order: 30,
    topics,
    translations: {
        ru: {
            title: 'Управление конфигурацией',
            description: 'Освойте управление конфигурацией в Java: файлы свойств, YAML, переменные окружения и паттерны конфигурации.',
        },
        uz: {
            title: 'Konfiguratsiyani boshqarish',
            description: 'Java da konfiguratsiyani boshqarishni o\'rganing: xususiyatlar fayllari, YAML, muhit o\'zgaruvchilari va konfiguratsiya naqshlari.',
        },
    },
};
