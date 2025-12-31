import { Topic } from '../../../../../types';
import { tasks } from './techniques/tasks';

export const techniquesTopic: Topic = {
	slug: 'graph-techniques',
	title: 'Graph Algorithms',
	description: 'Master essential graph algorithms: BFS, DFS, shortest paths, MST, topological sort, and cycle detection. Prepare for FAANG-level graph problems.',
	difficulty: 'hard',
	estimatedTime: '10h',
	order: 1,
	tasks,
	translations: {
		ru: {
			title: 'Алгоритмы на графах',
			description: 'Освойте основные алгоритмы на графах: BFS, DFS, кратчайшие пути, MST, топологическая сортировка и обнаружение циклов. Подготовка к задачам FAANG.',
		},
		uz: {
			title: 'Graf algoritmlari',
			description: 'Asosiy graf algoritmlarini o\'rganing: BFS, DFS, eng qisqa yo\'llar, MST, topologik saralash va sikl aniqlash. FAANG darajasidagi masalalarga tayyorgarlik.',
		},
	},
};
