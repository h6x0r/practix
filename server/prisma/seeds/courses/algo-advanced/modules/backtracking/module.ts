import { Module } from '../../../../types';
import { techniquesTopic } from './topics';

export const backtrackingModule: Module = {
	slug: 'algo-backtracking',
	title: 'Backtracking',
	description: 'Master backtracking algorithms for exhaustive search, combinations, permutations, and constraint satisfaction problems',
	section: 'algorithms',
	order: 3,
	difficulty: 'medium',
	estimatedTime: '8h',
	topics: [techniquesTopic],
	translations: {
		ru: {
			title: 'Бэктрекинг',
			description: 'Освойте алгоритмы бэктрекинга для полного перебора, комбинаций, перестановок и задач удовлетворения ограничений'
		},
		uz: {
			title: 'Backtracking',
			description: "To'liq qidiruv, kombinatsiyalar, permutatsiyalar va cheklovlarni qondirish masalalari uchun backtracking algoritmlarini o'rganing"
		}
	}
};

export default backtrackingModule;
