import { Module } from '../../../../types';
import { principlesTopic } from './topics';

export const antiPatternsModule: Module = {
	slug: 'se-anti-patterns',
	title: 'Anti-patterns',
	description: 'Learn to recognize and avoid common anti-patterns in software development.',
	section: 'software-engineering',
	order: 5,
	difficulty: 'medium',
	estimatedTime: '5h',
	topics: [principlesTopic],
};
