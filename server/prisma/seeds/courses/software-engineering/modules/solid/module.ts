import { Module } from '../../../../types';
import { principlesTopic } from './topics';

export const solidModule: Module = {
	slug: 'se-solid',
	title: 'SOLID Principles',
	description: 'Master the five fundamental principles of object-oriented design that make software more maintainable, flexible, and scalable.',
	section: 'software-engineering',
	order: 1,
	difficulty: 'medium',
	estimatedTime: '5h',
	topics: [principlesTopic],
};
