import { Topic } from '../../../../types';
import * as taskImports from './tasks';

const tasks = Object.values(taskImports);

export const topic: Topic = {
	title: 'Channel Patterns Implementation',
	description: 'Master production channel patterns including fan-in, fan-out, and worker pools for concurrent processing.',
	difficulty: 'medium',
	estimatedTime: '50m',	order: 1,
	tasks,
	translations: {
		ru: {
			title: 'Реализация паттернов каналов',
			description: 'Освойте промышленные паттерны каналов, включая fan-in, fan-out и пулы воркеров для параллельной обработки.'
		},
		uz: {
			title: 'Kanal patternlarini amalga oshirish',
			description: 'Fan-in, fan-out va worker poollar kabi ishlab chiqarish kanal patternlarini parallel ishlash uchun o\'rganing.'
		}
	}
};
