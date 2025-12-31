import { CourseMeta } from '../../types';

const course: CourseMeta = {
	slug: 'python-llm',
	title: 'LLMs & Transformers',
	description:
		'Master Large Language Models from attention mechanisms to fine-tuning and deployment. Learn transformers, BERT, GPT, and modern NLP techniques.',
	category: 'language',
	icon: '🤖',
	estimatedTime: '30h',
	order: 22,
	translations: {
		ru: {
			title: 'LLM и трансформеры',
			description:
				'Освойте большие языковые модели от механизмов внимания до fine-tuning и развертывания. Изучите трансформеры, BERT, GPT и современные NLP техники.',
		},
		uz: {
			title: 'LLM va transformerlar',
			description:
				"Katta til modellarini diqqat mexanizmlaridan fine-tuning va joylashtirishgacha o'rganing. Transformerlar, BERT, GPT va zamonaviy NLP texnikalarini o'rganing.",
		},
	},
};

export default course;
