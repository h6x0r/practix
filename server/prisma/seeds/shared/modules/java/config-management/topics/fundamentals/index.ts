import { Topic } from '../../../../types';
import { tasks } from './tasks';

export const fundamentals: Topic = {
    slug: 'java-config-management-fundamentals',
    title: 'Configuration Management Fundamentals',
    description: 'Learn properties files, YAML configuration, environment variables, configuration patterns, and profiles',
    order: 0,
    tasks,
    translations: {
        ru: {
            title: 'Основы управления конфигурацией',
            description: 'Изучите properties файлы, YAML конфигурацию, переменные окружения, паттерны конфигурации и профили',
        },
        uz: {
            title: 'Konfiguratsiya Boshqaruvi Asoslari',
            description: 'Properties fayllar, YAML konfiguratsiya, muhit o\'zgaruvchilari, konfiguratsiya namunalari va profillarni o\'rganing',
        },
    },
};
