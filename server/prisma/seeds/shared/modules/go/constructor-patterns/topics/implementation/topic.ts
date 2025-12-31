import { Topic } from '../../../../types';
import { tasks } from './tasks';

export const topic: Topic = {
	title: 'Constructor Patterns Implementation',
	description: 'Master functional options pattern and constructor validation for flexible, type-safe API design.',
	difficulty: 'medium',
	estimatedTime: '55m',	order: 1,
	tasks,
	translations: {
		ru: {
			title: 'Реализация паттернов конструкторов',
			description: 'Освоение функционального паттерна опций и валидации конструкторов для гибкого и типобезопасного проектирования API.'
		},
		uz: {
			title: 'Konstruktor patternlarini joriy qilish',
			description: 'Funksional parametrlar patterni va konstruktor tekshiruvini o\'rganish orqali moslashuvchan va tipdan xavfsiz API dizaynini yaratish.'
		}
	}
};
