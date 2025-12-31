import { Module } from '../../../../types';
import { basicsTopic } from './topics';

export const stringsModule: Module = {
	slug: 'algo-strings',
	title: 'Strings',
	description: 'Master string manipulation: pattern matching, anagram detection, palindrome checking, and common string algorithms.',
	section: 'algorithms',
	order: 2,
	difficulty: 'easy',
	estimatedTime: '5h',
	topics: [basicsTopic],
	translations: {
		ru: {
			title: 'Строки',
			description: 'Освойте работу со строками: поиск паттернов, определение анаграмм, проверка палиндромов и основные строковые алгоритмы.',
		},
		uz: {
			title: 'Satrlar',
			description: 'Satrlar bilan ishlashni o\'rganing: pattern qidirish, anagramma aniqlash, palindrom tekshirish va asosiy satr algoritmlari.',
		},
	},
};
