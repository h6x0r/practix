import { Module } from '../../../../types';
import { topics } from './topics';

export const synchronizationModule: Module = {
	title: 'Synchronization Primitives',
	description: 'Master advanced synchronization with RWMutex, condition variables, semaphores, and concurrent data structures.',
	section: 'concurrency',
	order: 16,
	topics,
	translations: {
		ru: {
			title: 'Примитивы синхронизации',
			description: 'Освойте продвинутую синхронизацию с RWMutex, условными переменными, семафорами и конкурентными структурами данных.'
		},
		uz: {
			title: 'Sinxronizatsiya primitivlari',
			description: 'RWMutex, shartli o\'zgaruvchilar, semaforlar va parallel ma\'lumotlar tuzilmalari bilan ilg\'or sinxronizatsiyani o\'zlashtiring.'
		}
	}
};
