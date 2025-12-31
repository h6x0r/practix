import { Topic } from '../../../../types';
import * as taskImports from './tasks';

const tasks = Object.values(taskImports);

export const topic: Topic = {
	title: 'Goroutine Lifecycle Management',
	description: 'Master goroutine lifecycle patterns with context awareness, proper cleanup, and leak prevention.',
	difficulty: 'medium',
	estimatedTime: '40m',	order: 1,
	tasks,
	translations: {
		ru: {
			title: 'Управление жизненным циклом горутин',
			description: 'Освойте паттерны жизненного цикла горутин с учетом контекста, правильной очисткой и предотвращением утечек.'
		},
		uz: {
			title: 'Goroutine hayot tsiklini boshqarish',
			description: 'Kontekstni hisobga olgan holda goroutine hayot tsikli patternlarini, to\'g\'ri tozalashni va oqish oldini olishni o\'rganing.'
		}
	}
};
