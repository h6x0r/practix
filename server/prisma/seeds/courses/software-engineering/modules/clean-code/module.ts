import { Module } from '../../../../types';
import { principlesTopic } from './topics';

export const cleanCodeModule: Module = {
	slug: 'se-clean-code',
	title: 'Clean Code',
	description: 'Master clean code principles and practices for writing maintainable, readable code.',
	section: 'software-engineering',
	order: 2,
	difficulty: 'medium',
	estimatedTime: '8h',
	topics: [principlesTopic],
};
