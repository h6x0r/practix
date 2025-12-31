import { Topic } from '../../../../../../types';

const topic: Topic = {
	slug: 'extractive-qa',
	title: 'Extractive QA',
	description: 'Extract answers from context passages using span extraction techniques.',
	order: 1,
	isPremium: true,
	translations: {
		ru: {
			title: 'Извлекающий QA',
			description:
				'Извлечение ответов из контекстных пассажей с использованием техник извлечения спанов.',
		},
		uz: {
			title: 'Ajratib oluvchi QA',
			description:
				"Span ajratib olish texnikalaridan foydalanib kontekst parchalaridan javoblarni ajratib olish.",
		},
	},
};

export default topic;
