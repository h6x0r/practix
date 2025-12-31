import { Module } from '../../../../types';
import { techniquesTopic } from './topics';

export const graphsModule: Module = {
	slug: 'algo-graphs',
	title: 'Graph Algorithms',
	description: 'Master graph traversal (BFS/DFS), shortest paths (Dijkstra), minimum spanning trees, topological sort, and cycle detection. Essential for FAANG interviews.',
	section: 'algorithms',
	order: 2,
	difficulty: 'hard',
	estimatedTime: '10h',
	topics: [techniquesTopic],
	translations: {
		ru: {
			title: 'Алгоритмы на графах',
			description: 'Освойте обход графов (BFS/DFS), кратчайшие пути (Дейкстра), минимальные остовные деревья, топологическую сортировку и обнаружение циклов. Необходимо для FAANG-интервью.',
		},
		uz: {
			title: 'Graf algoritmlari',
			description: 'Graf aylanishini (BFS/DFS), eng qisqa yo\'llarni (Dijkstra), minimal qoplovchi daraxtlarni, topologik saralash va sikl aniqlashni o\'rganing. FAANG suhbatlari uchun zarur.',
		},
	},
};
