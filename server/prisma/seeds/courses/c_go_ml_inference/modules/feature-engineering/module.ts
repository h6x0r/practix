import { ModuleMeta } from '../../../../types';

const module: ModuleMeta = {
	slug: 'feature-engineering',
	title: 'Feature Engineering',
	description: 'Prepare and transform data for ML inference in Go.',
	order: 2,
	isPremium: false,
	translations: {
		ru: {
			title: 'Инженерия признаков',
			description: 'Подготовка и преобразование данных для ML инференса в Go.',
		},
		uz: {
			title: 'Feature engineering',
			description: "Go da ML inference uchun ma'lumotlarni tayyorlash va o'zgartirish.",
		},
	},
};

export default module;
