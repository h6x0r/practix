import { CourseMeta } from '../../types';

const course: CourseMeta = {
	slug: 'java-ml',
	title: 'Java Machine Learning',
	description: 'Build ML applications in Java using DL4J, Tribuo, and production-ready techniques. From data processing to model deployment.',
	category: 'language',
	icon: '☕',
	estimatedTime: '25h',
	order: 23,
	translations: {
		ru: {
			title: 'Machine Learning на Java',
			description: 'Создавайте ML приложения на Java с DL4J, Tribuo и production-ready техниками. От обработки данных до деплоя моделей.',
		},
		uz: {
			title: 'Java Machine Learning',
			description: "DL4J, Tribuo va production-ready texnikalar bilan Java da ML ilovalar yarating. Ma'lumotlarni qayta ishlashdan model deployigacha.",
		},
	},
};

export default course;
