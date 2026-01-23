import { Topic } from '../../../../../types';
import { tasks } from './fundamentals/tasks';

export const fundamentalsTopic: Topic = {
	slug: 'sec-fundamentals-core',
	title: 'Core Security Concepts',
	description: 'CIA Triad, Defense in Depth, Least Privilege, Threat Modeling, and Security by Design principles.',
	difficulty: 'easy',
	estimatedTime: '4h',
	order: 1,
	tasks,
	translations: {
		ru: {
			title: 'Базовые концепции безопасности',
			description: 'Триада CIA, глубокая защита, минимальные привилегии, моделирование угроз и принципы безопасности по дизайну.',
		},
		uz: {
			title: 'Asosiy xavfsizlik tushunchalari',
			description: 'CIA Triad, chuqur himoya, minimal huquqlar, tahdidlarni modellashtirish va dizayn bo\'yicha xavfsizlik prinsiplari.',
		},
	},
};
