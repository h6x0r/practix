import { Topic } from '../../../../../types';
import { tasks } from './principles/tasks';

export const principlesTopic: Topic = {
	slug: 'se-anti-patterns-principles',
	title: 'Common Anti-patterns',
	description: 'Identify and refactor common anti-patterns: God Object, Spaghetti Code, Golden Hammer, Copy-Paste Programming, Magic Numbers, and Premature Optimization.',
	difficulty: 'medium',
	estimatedTime: '5h',
	order: 1,
	tasks,
	translations: {
		ru: {
			title: 'Распространённые антипаттерны',
			description: 'Определяйте и рефакторьте распространённые антипаттерны: God Object, Spaghetti Code, Golden Hammer, Copy-Paste Programming, Magic Numbers и Premature Optimization.',
		},
		uz: {
			title: 'Umumiy anti-patternlar',
			description: 'Umumiy anti-patternlarni aniqlang va refaktoring qiling: God Object, Spaghetti Code, Golden Hammer, Copy-Paste Programming, Magic Numbers va Premature Optimization.',
		},
	},
};
