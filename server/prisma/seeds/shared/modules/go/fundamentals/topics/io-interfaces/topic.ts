/**
 * IO Interfaces Topic
 * Working with Go io.Reader and io.Writer interfaces
 */

import { Topic } from '../../../../types';
import * as tasks from './tasks';

export const topic: Topic = {
	slug: 'io-interfaces',
	title: 'IO Interfaces',
	description: 'Master Go io.Reader and io.Writer interfaces for efficient I/O operations',
	difficulty: 'medium',
	estimatedTime: '20m',	order: 1,
	tasks: Object.values(tasks),
	translations: {
		ru: {
			title: 'IO Интерфейсы',
			description: 'Освойте интерфейсы io.Reader и io.Writer для эффективных I/O операций'
		},
		uz: {
			title: 'IO Interfeyslar',
			description: 'Samarali I/O operatsiyalari uchun io.Reader va io.Writer interfeyslarini o\'rganing'
		}
	}
};
