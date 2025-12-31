import { Topic } from '../../../../../types';
import { tasks } from './basics/tasks';

export const basicsTopic: Topic = {
	slug: 'algo-strings-basics',
	title: 'String Operations',
	description: 'Essential string algorithms: palindromes, anagrams, pattern matching, and character manipulation.',
	difficulty: 'easy',
	estimatedTime: '5h',
	order: 1,
	tasks,
	translations: {
		ru: {
			title: 'Операции со строками',
			description: 'Основные строковые алгоритмы: палиндромы, анаграммы, поиск паттернов и работа с символами.',
		},
		uz: {
			title: 'Satr operatsiyalari',
			description: 'Asosiy satr algoritmlari: palindromlar, anagrammalar, pattern qidirish va belgilar bilan ishlash.',
		},
	},
};
