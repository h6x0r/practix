import { Topic } from '../../../../../../types';
import tasks from './tasks';

export const topic: Topic = {
	slug: 'basics',
	title: 'Python Basics',
	description: 'Variables, data types, and basic operations in Python.',
	order: 1,
	tasks: tasks,
	translations: {
		ru: {
			title: 'Основы Python',
			description: 'Переменные, типы данных и базовые операции в Python.',
		},
		uz: {
			title: 'Python asoslari',
			description: "Python da o'zgaruvchilar, ma'lumot turlari va asosiy operatsiyalar.",
		},
	},
};
