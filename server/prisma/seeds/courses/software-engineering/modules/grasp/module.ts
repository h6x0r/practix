import { Module } from '../../../../types';
import { principlesTopic } from './topics';

export const graspModule: Module = {
	slug: 'se-grasp',
	title: 'GRASP Principles',
	description: 'Learn General Responsibility Assignment Software Patterns for object-oriented design.',
	section: 'software-engineering',
	order: 4,
	difficulty: 'medium',
	estimatedTime: '8h',
	topics: [principlesTopic],
};
