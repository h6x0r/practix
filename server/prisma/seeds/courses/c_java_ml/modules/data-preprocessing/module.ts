import { ModuleMeta } from '../../../../types';

const module: ModuleMeta = {
	slug: 'data-preprocessing',
	title: 'Data Preprocessing',
	description: 'Prepare and transform data for ML models using DataVec',
	order: 5,
	isPremium: false,
	translations: {
		ru: {
			title: 'Предобработка данных',
			description: 'Подготовка и трансформация данных для ML моделей с DataVec',
		},
		uz: {
			title: 'Ma\'lumotlarni oldindan qayta ishlash',
			description: 'DataVec bilan ML modellari uchun ma\'lumotlarni tayyorlash va o\'zgartirish',
		},
	},
};

export default module;
