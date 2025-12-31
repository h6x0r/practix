import { Topic } from '../../../../../types';
import { tasks } from './principles/tasks';

export const principlesTopic: Topic = {
	slug: 'se-grasp-principles',
	title: 'GRASP Principles',
	description: 'Information Expert, Creator, Controller, Low Coupling, High Cohesion, Polymorphism, Pure Fabrication, Indirection, and Protected Variations.',
	difficulty: 'medium',
	estimatedTime: '8h',
	order: 1,
	tasks,
	translations: {
		ru: {
			title: 'Принципы GRASP',
			description: 'Information Expert, Creator, Controller, Low Coupling, High Cohesion, Polymorphism, Pure Fabrication, Indirection и Protected Variations.',
		},
		uz: {
			title: 'GRASP Printsiplari',
			description: 'Information Expert, Creator, Controller, Low Coupling, High Cohesion, Polymorphism, Pure Fabrication, Indirection va Protected Variations.',
		},
	},
};
