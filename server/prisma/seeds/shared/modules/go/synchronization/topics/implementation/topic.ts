import { Topic } from '../../../../types';
import * as taskImports from './tasks';

const tasks = Object.values(taskImports);

export const topic: Topic = {
	title: 'Synchronization Primitives',
	description: 'Master Go synchronization with RWMutex, Cond, semaphores, and advanced concurrent data structures.',
	difficulty: 'medium',
	estimatedTime: '45m',	order: 1,
	tasks,
	translations: {
		ru: {
			title: 'Примитивы синхронизации',
			description: 'Освойте синхронизацию в Go с RWMutex, Cond, семафорами и продвинутыми параллельными структурами данных.'
		},
		uz: {
			title: 'Sinxronizatsiya primitivlari',
			description: 'Go dasturlashda RWMutex, Cond, semaforlar va ilg\'or parallel ma\'lumot strukturalari bilan sinxronizatsiyani o\'rganing.'
		}
	}
};
