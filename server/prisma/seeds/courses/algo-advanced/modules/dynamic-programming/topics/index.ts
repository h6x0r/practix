import { Topic } from '../../../../../types';
import { tasks } from './techniques/tasks';

export const techniquesTopic: Topic = {
	slug: 'dp-techniques',
	title: 'DP Techniques',
	description: 'Master core Dynamic Programming patterns: memoization, tabulation, space optimization, and classic problems like Fibonacci, Knapsack, LCS, and grid DP.',
	difficulty: 'hard',
	estimatedTime: '8h',
	order: 1,
	tasks,
	translations: {
		ru: {
			title: 'Техники DP',
			description: 'Освойте основные паттерны динамического программирования: мемоизация, табуляция, оптимизация памяти и классические задачи: Фибоначчи, рюкзак, LCS и сеточное DP.',
		},
		uz: {
			title: 'DP texnikalari',
			description: 'Dinamik dasturlashning asosiy patternlarini o\'rganing: memoizatsiya, tabulyatsiya, xotira optimallashtirish va klassik masalalar: Fibonachchi, ryukzak, LCS va to\'rli DP.',
		},
	},
};
