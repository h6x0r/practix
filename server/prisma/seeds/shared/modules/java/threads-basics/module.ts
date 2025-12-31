import { Module } from '../../../types';
import { topics } from './topics';

export const module: Module = {
  slug: 'java-threads-basics',
  title: 'Threads Fundamentals',
  description: 'Learn the fundamentals of multi-threading in Java including thread creation, lifecycle management, synchronization mechanisms, and best practices for writing thread-safe code.',
  order: 12,
  topics,
  translations: {
    ru: {
      title: 'Основы потоков',
      description: 'Изучите основы многопоточности в Java, включая создание потоков, управление жизненным циклом, механизмы синхронизации и лучшие практики написания потокобезопасного кода.',
    },
    uz: {
      title: 'Oqimlar asoslari',
      description: 'Java da ko\'p oqimlilik asoslarini o\'rganing, shu jumladan oqimlarni yaratish, hayotiy tsiklni boshqarish, sinxronizatsiya mexanizmlari va oqim xavfsiz kod yozishning eng yaxshi amaliyotlari.',
    },
  },
};
