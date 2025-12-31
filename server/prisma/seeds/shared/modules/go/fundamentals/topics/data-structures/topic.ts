/**
 * Data Structures Topic
 * Maps, slices, and strings manipulation
 */

import { Topic } from '../../../../types';
import * as tasks from './tasks';

export const topic: Topic = {
	slug: 'data-structures',
	title: 'Data Structures',
	description: 'Master Go data structures: maps, slices, and strings with generics',
	difficulty: 'medium',
	estimatedTime: '2h',	order: 0,
	tasks: Object.values(tasks),
	translations: {
		ru: {
			title: 'Структуры данных',
			description: 'Освойте структуры данных Go: maps, slices и strings с использованием generics'
		},
		uz: {
			title: "Ma'lumotlar tuzilmalari",
			description: "Go ma'lumotlar tuzilmalarini o'rganing: maps, slices va strings generics bilan"
		}
	}
};
