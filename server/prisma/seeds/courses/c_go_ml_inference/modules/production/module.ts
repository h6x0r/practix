import { ModuleMeta } from '../../../../types';

const module: ModuleMeta = {
	slug: 'production',
	title: 'Production Deployment',
	description: 'Deploy and operate ML inference systems in production.',
	order: 7,
	isPremium: true,
	translations: {
		ru: {
			title: 'Production деплой',
			description: 'Деплой и эксплуатация систем ML инференса в продакшене.',
		},
		uz: {
			title: 'Production deploy',
			description: "Production da ML inference tizimlarini deploy qilish va boshqarish.",
		},
	},
};

export default module;
