import { CourseMeta } from '../../types';

const course: CourseMeta = {
	slug: 'math-for-ds',
	title: 'Math for Data Science',
	description:
		'Master essential mathematics for machine learning and data science. Learn linear algebra, calculus, statistics, and optimization through hands-on Python implementations.',
	category: 'cs',
	icon: '📐',
	estimatedTime: '30h',
	order: 6,
	translations: {
		ru: {
			title: 'Математика для Data Science',
			description:
				'Освойте основную математику для машинного обучения и data science. Изучите линейную алгебру, матанализ, статистику и оптимизацию через практические задачи на Python.',
		},
		uz: {
			title: 'Data Science uchun matematika',
			description:
				"Mashinali o'rganish va data science uchun zarur matematikani o'rganing. Chiziqli algebra, matematik tahlil, statistika va optimizatsiyani Python amaliyotlari orqali o'zlashtiring.",
		},
	},
};

export default course;
