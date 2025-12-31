import { ModuleMeta } from '../../../../types';

const module: ModuleMeta = {
	slug: 'monitoring',
	title: 'Monitoring & Observability',
	description: 'Implement comprehensive monitoring for ML inference systems.',
	order: 5,
	isPremium: true,
	translations: {
		ru: {
			title: 'Мониторинг и наблюдаемость',
			description: 'Реализация комплексного мониторинга для систем ML инференса.',
		},
		uz: {
			title: 'Monitoring va kuzatuvchanlik',
			description: "ML inference tizimlari uchun keng qamrovli monitoringni amalga oshirish.",
		},
	},
};

export default module;
