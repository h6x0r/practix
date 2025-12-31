/**
 * Constructors Topic
 * Go constructor patterns and functional options
 */

import { Topic } from '../../../../types';
import * as tasks from './tasks';

export const topic: Topic = {
	slug: 'constructors',
	title: 'Constructor Patterns',
	description: 'Learn Go constructor patterns including functional options for flexible, extensible APIs',
	difficulty: 'medium',
	estimatedTime: '25m',	order: 2,
	tasks: Object.values(tasks),
	translations: {
		ru: {
			title: 'Паттерны конструкторов',
			description: 'Изучите паттерны конструкторов Go включая функциональные опции для гибких, расширяемых API'
		},
		uz: {
			title: 'Konstruktor naqshlari',
			description: 'Moslashuvchan, kengaytiriladigan API lar uchun funksional variantlar bilan Go konstruktor naqshlarini o\'rganing'
		}
	}
};
