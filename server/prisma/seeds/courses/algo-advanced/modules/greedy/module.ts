import { Module } from '../../../../types';
import { techniquesTopic } from './topics';

export const greedyModule: Module = {
	slug: 'algo-greedy',
	title: 'Greedy Algorithms',
	description: 'Master greedy algorithms for optimization problems: interval scheduling, resource allocation, and making locally optimal choices',
	section: 'algorithms',
	order: 4,
	difficulty: 'medium',
	estimatedTime: '8h',
	topics: [techniquesTopic],
	translations: {
		ru: {
			title: 'Жадные алгоритмы',
			description: 'Освойте жадные алгоритмы для задач оптимизации: планирование интервалов, распределение ресурсов и локально оптимальные решения'
		},
		uz: {
			title: 'Greedy algoritmlar',
			description: "Optimallashtirish masalalari uchun greedy algoritmlarni o'rganing: interval rejalashtirish, resurslarni taqsimlash va lokal optimal qarorlar"
		}
	}
};

export default greedyModule;
