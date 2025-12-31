import { Topic } from '../../../../types';
import { tasks } from './tasks';

export const fundamentals: Topic = {
  slug: 'java-threads-fundamentals',
  title: 'Threads Fundamentals',
  description: 'Master the fundamentals of multi-threading in Java including thread creation, lifecycle, synchronization, and thread safety.',
  order: 1,
  tasks,
  translations: {
    ru: {
      title: 'Основы потоков',
      description: 'Освойте основы многопоточности в Java, включая создание потоков, жизненный цикл, синхронизацию и потокобезопасность.',
    },
    uz: {
      title: 'Oqimlar asoslari',
      description: 'Java da ko\'p oqimlilik asoslarini o\'rganing, shu jumladan oqimlarni yaratish, hayotiy tsikl, sinxronizatsiya va oqim xavfsizligi.',
    },
  },
};
