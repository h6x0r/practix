import { ModuleMeta } from '../../../../types';

const module: ModuleMeta = {
	slug: 'serving',
	title: 'Model Serving',
	description: 'Build production-ready ML inference servers in Go.',
	order: 4,
	isPremium: true,
	translations: {
		ru: {
			title: 'Сервинг моделей',
			description: 'Создание production-ready серверов ML инференса на Go.',
		},
		uz: {
			title: 'Model serving',
			description: "Go da production-ready ML inference serverlarini yaratish.",
		},
	},
};

export default module;
