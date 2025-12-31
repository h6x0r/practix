import { Module } from '../../../../types';
import { principlesTopic } from './topics';

export const apiDesignModule: Module = {
	slug: 'se-api-design',
	title: 'API Design Principles',
	description: 'Master RESTful API design principles including resource naming, HTTP methods, error handling, and versioning strategies.',
	section: 'software-engineering',
	order: 6,
	difficulty: 'medium',
	estimatedTime: '5h',
	topics: [principlesTopic],
};
