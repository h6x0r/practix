import { ModuleMeta } from '../../../../types';

const module: ModuleMeta = {
	slug: 'question-answering',
	title: 'Question Answering',
	description: 'Build question answering systems using various NLP techniques.',
	order: 7,
	isPremium: true,
	translations: {
		ru: {
			title: 'Ответы на вопросы',
			description:
				'Создайте системы ответов на вопросы с использованием различных NLP техник.',
		},
		uz: {
			title: 'Savollarga javob',
			description:
				"Turli NLP texnikalaridan foydalanib savollarga javob tizimlarini yarating.",
		},
	},
};

export default module;
