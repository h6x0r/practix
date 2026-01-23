import { Module } from '../../../../types';
import { basicsTopic } from './topics';

export const fundamentalsModule: Module = {
	slug: 'pe-fundamentals',
	title: 'Fundamentals',
	description:
		'Learn the core principles of prompt engineering: clear instructions, structured output, and iterative refinement.',
	section: 'prompt-engineering',
	order: 1,
	difficulty: 'easy',
	estimatedTime: '2h',
	isPremium: true,
	topics: [basicsTopic],
	translations: {
		ru: {
			title: 'Основы',
			description:
				'Изучите основные принципы prompt engineering: четкие инструкции, структурированный вывод и итеративное улучшение.',
		},
		uz: {
			title: 'Asoslar',
			description:
				"Prompt engineering asosiy tamoyillarini o'rganing: aniq ko'rsatmalar, strukturalangan chiqish va iterativ takomillashtirish.",
		},
	},
};
