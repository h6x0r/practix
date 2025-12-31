import { Module } from '../../../../types';
import { topics } from './topics';

export const module: Module = {
	slug: 'data-structures',
	title: 'Data Structures',
	description: 'Master essential data structure operations with Go generics including maps, slices, deduplication, and efficient string manipulation.',
	difficulty: 'easy',
	estimatedTime: '40m',
	order: 20,
	isPremium: false,
	section: 'core',
	topics,
	translations: {
		ru: {
			title: 'Структуры данных',
			description: 'Освойте основные операции со структурами данных с использованием Go generics, включая карты, слайсы, дедупликацию и эффективную работу со строками.'
		},
		uz: {
			title: 'Ma\'lumotlar strukturalari',
			description: 'Go generics yordamida maplar, slicelar, dublikatlarni olib tashlash va samarali string manipulyatsiyani o\'z ichiga olgan muhim ma\'lumotlar strukturalari operatsiyalarini o\'rganing.'
		}
	}
};
