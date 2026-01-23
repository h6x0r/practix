import { Module } from '../../../../types';
import { fundamentalsTopic } from './topics';

export const interviewPrepModule: Module = {
	slug: 'sec-interview-prep',
	title: 'Interview Preparation',
	description: 'Prepare for security interviews: common questions, scenario-based problems, and practical demonstrations.',
	section: 'security',
	order: 7,
	difficulty: 'medium',
	estimatedTime: '4h',
	topics: [fundamentalsTopic],
	translations: {
		ru: {
			title: 'Подготовка к собеседованиям',
			description: 'Подготовьтесь к собеседованиям по безопасности: общие вопросы, сценарные задачи и практические демонстрации.'
		},
		uz: {
			title: 'Intervyuga tayyorgarlik',
			description: 'Xavfsizlik intervyulariga tayyorlaning: umumiy savollar, stsenariyga asoslangan muammolar va amaliy namoyishlar.'
		}
	}
};
