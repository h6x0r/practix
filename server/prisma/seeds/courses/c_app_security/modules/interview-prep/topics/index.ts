import { Topic } from '../../../../../types';
import { tasks } from './fundamentals/tasks';

export const fundamentalsTopic: Topic = {
	slug: 'sec-interview-questions',
	title: 'Security Interview Questions',
	description: 'Common security interview questions, scenario-based problems, and practical demonstrations.',
	difficulty: 'medium',
	estimatedTime: '4h',
	order: 1,
	tasks,
	translations: {
		ru: {
			title: 'Вопросы на собеседованиях по безопасности',
			description: 'Общие вопросы на собеседованиях по безопасности, сценарные задачи и практические демонстрации.',
		},
		uz: {
			title: 'Xavfsizlik intervyu savollari',
			description: 'Umumiy xavfsizlik intervyu savollari, stsenariyga asoslangan muammolar va amaliy namoyishlar.',
		},
	},
};
