import { Topic } from '../../../../../types';
import { tasks } from './techniques/tasks';

export const techniquesTopic: Topic = {
	slug: 'divide-conquer-techniques',
	title: 'Divide and Conquer Techniques',
	description: 'Master the divide and conquer paradigm: sorting, searching, and solving problems by breaking them into smaller subproblems',
	order: 1,
	tasks,
	translations: {
		ru: {
			title: 'Техники "разделяй и властвуй"',
			description: 'Освойте парадигму "разделяй и властвуй": сортировка, поиск и решение задач разбиением на подзадачи'
		},
		uz: {
			title: '"Bo\'l va hukmronlik qil" texnikalari',
			description: '"Bo\'l va hukmronlik qil" paradigmasini o\'rganing: saralash, qidirish va masalalarni kichik qismlarga bo\'lish'
		}
	}
};

export const topics = [techniquesTopic];

export default topics;
