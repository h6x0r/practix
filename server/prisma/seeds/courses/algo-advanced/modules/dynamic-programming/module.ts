import { Module } from '../../../../types';
import { techniquesTopic } from './topics';

export const dynamicProgrammingModule: Module = {
	slug: 'algo-dynamic-programming',
	title: 'Dynamic Programming',
	description: 'Master the art of breaking problems into overlapping subproblems. Learn memoization, tabulation, and space optimization techniques for FAANG interviews.',
	section: 'algorithms',
	order: 1,
	difficulty: 'hard',
	estimatedTime: '8h',
	topics: [techniquesTopic],
	translations: {
		ru: {
			title: 'Динамическое программирование',
			description: 'Освойте искусство разбиения задач на перекрывающиеся подзадачи. Изучите мемоизацию, табуляцию и оптимизацию памяти для FAANG-интервью.',
		},
		uz: {
			title: 'Dinamik dasturlash',
			description: 'Masalalarni bir-biriga o\'xshash kichik masalalarga bo\'lish san\'atini o\'rganing. FAANG suhbatlari uchun memoizatsiya, tabulyatsiya va xotira optimallashtirish texnikalarini o\'rganing.',
		},
	},
};
