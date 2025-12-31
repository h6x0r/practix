import { Topic } from '../../../../types';
import * as taskImports from './tasks';

const tasks = Object.values(taskImports);

export const topic: Topic = {
	title: 'Performance Profiling and Optimization',
	description: 'Master Go profiling tools and optimization techniques for memory and CPU performance.',
	difficulty: 'medium',
	estimatedTime: '40m',	order: 1,
	tasks,
	translations: {
		ru: {
			title: 'Профилирование и оптимизация производительности',
			description: 'Освоение инструментов профилирования Go и методов оптимизации для повышения производительности памяти и процессора.'
		},
		uz: {
			title: 'Ishlash samaradorligini profillash va optimallashtirish',
			description: 'Go profillash vositalari va xotira hamda protsessor samaradorligini oshirish usullarini o\'zlashtirish.'
		}
	}
};
