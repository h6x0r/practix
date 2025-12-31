import { CourseMeta } from '../../types';

const course: CourseMeta = {
	slug: 'python-ml-fundamentals',
	title: 'Python ML Fundamentals',
	description:
		'Master the foundations of Machine Learning with Python. Learn NumPy, Pandas, scikit-learn, and data visualization through hands-on tasks.',
	category: 'language',
	icon: '🐍',
	estimatedTime: '25h',
	order: 20,
	translations: {
		ru: {
			title: 'Основы ML на Python',
			description:
				'Освойте основы машинного обучения с Python. Изучите NumPy, Pandas, scikit-learn и визуализацию данных через практические задачи.',
		},
		uz: {
			title: 'Python ML Asoslari',
			description:
				"Python bilan mashinali o'rganish asoslarini o'rganing. NumPy, Pandas, scikit-learn va ma'lumotlarni vizualizatsiya qilishni amaliy topshiriqlar orqali o'rganing.",
		},
	},
};

export default course;
