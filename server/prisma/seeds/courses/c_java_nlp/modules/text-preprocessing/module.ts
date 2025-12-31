import { ModuleMeta } from '../../../../types';

const module: ModuleMeta = {
	slug: 'text-preprocessing',
	title: 'Text Preprocessing',
	description: 'Clean and prepare text data for NLP tasks.',
	order: 1,
	isPremium: false,
	translations: {
		ru: {
			title: 'Предобработка текста',
			description: 'Очистка и подготовка текстовых данных для NLP задач.',
		},
		uz: {
			title: 'Matn oldindan qayta ishlash',
			description: "Matn ma'lumotlarini NLP vazifalari uchun tozalash va tayyorlash.",
		},
	},
};

export default module;
