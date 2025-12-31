import { Topic } from '../../../../../types';
import { tasks } from './techniques/tasks';

export const techniquesTopic: Topic = {
	slug: 'backtracking-techniques',
	title: 'Backtracking Techniques',
	description: 'Master backtracking patterns for exhaustive search and constraint satisfaction problems',
	order: 1,
	tasks,
	translations: {
		ru: {
			title: 'Техники бэктрекинга',
			description: 'Освойте паттерны бэктрекинга для полного перебора и задач удовлетворения ограничений'
		},
		uz: {
			title: 'Backtracking texnikalari',
			description: "To'liq qidiruv va cheklovlarni qondirish masalalari uchun backtracking patternlarini o'rganing"
		}
	}
};

export const topics = [techniquesTopic];

export default topics;
