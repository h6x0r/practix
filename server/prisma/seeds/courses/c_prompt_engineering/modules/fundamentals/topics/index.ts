import { Topic } from '../../../../../types';
import tasks from './basics/tasks';

export const basicsTopic: Topic = {
	slug: 'pe-basics',
	title: 'Prompt Basics',
	description:
		'Master the fundamental techniques for writing clear, effective prompts that guide AI to produce desired outputs.',
	difficulty: 'easy',
	estimatedTime: '2h',
	order: 1,
	tasks,
	translations: {
		ru: {
			title: 'Основы промптов',
			description:
				'Освойте фундаментальные техники написания четких и эффективных промптов, которые направляют AI к желаемому результату.',
		},
		uz: {
			title: 'Prompt asoslari',
			description:
				"AIni kerakli natijalarga yo'naltiruvchi aniq va samarali promptlarni yozishning asosiy texnikalarini o'rganing.",
		},
	},
};
