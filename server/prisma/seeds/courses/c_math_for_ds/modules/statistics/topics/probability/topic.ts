import { Topic } from '../../../../../../types';
import tasks from './tasks';

export const topic: Topic = {
	slug: 'probability',
	title: 'Probability & Distributions',
	description: 'Learn probability theory, statistical distributions, and measures for data science.',
	order: 1,
	tasks: tasks,
	translations: {
		ru: {
			title: 'Вероятность и распределения',
			description: 'Изучите теорию вероятностей, статистические распределения и меры для data science.',
		},
		uz: {
			title: 'Ehtimollik va taqsimotlar',
			description: 'Data science uchun ehtimollik nazariyasi, statistik taqsimotlar va o\'lchovlarni o\'rganing.',
		},
	},
};
