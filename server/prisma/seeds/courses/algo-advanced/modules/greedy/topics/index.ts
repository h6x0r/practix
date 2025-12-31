import { Topic } from '../../../../../types';
import { tasks } from './techniques/tasks';

export const techniquesTopic: Topic = {
	slug: 'greedy-techniques',
	title: 'Greedy Techniques',
	description: 'Master greedy algorithms for optimization problems including scheduling, intervals, and resource allocation',
	order: 1,
	tasks,
	translations: {
		ru: {
			title: 'Жадные техники',
			description: 'Освойте жадные алгоритмы для задач оптимизации: планирование, интервалы и распределение ресурсов'
		},
		uz: {
			title: 'Greedy texnikalari',
			description: "Optimallashtirish masalalari uchun greedy algoritmlarni o'rganing: rejalashtirish, intervallar va resurslarni taqsimlash"
		}
	}
};

export const topics = [techniquesTopic];

export default topics;
