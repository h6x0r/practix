import { CourseMeta } from '../../types';

const course: CourseMeta = {
	slug: 'python-fundamentals',
	title: 'Python Fundamentals',
	description:
		'Master Python programming from the ground up. Learn syntax, data structures, functions, OOP, and essential programming concepts through hands-on coding challenges.',
	category: 'cs',
	icon: '🐍',
	estimatedTime: '25h',
	order: 5,
	translations: {
		ru: {
			title: 'Основы Python',
			description:
				'Освойте программирование на Python с нуля. Изучите синтаксис, структуры данных, функции, ООП и основные концепции программирования через практические задачи.',
		},
		uz: {
			title: 'Python asoslari',
			description:
				"Python dasturlashni noldan o'rganing. Sintaksis, ma'lumotlar tuzilmalari, funksiyalar, OOP va asosiy dasturlash tushunchalarini amaliy mashqlar orqali o'rganing.",
		},
	},
};

export default course;
