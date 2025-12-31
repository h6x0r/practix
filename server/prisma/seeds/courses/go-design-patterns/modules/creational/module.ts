import { Module } from '../../../../types';
import { patternsTopic } from './topics';

export const creationalModule: Module = {
	slug: 'go-dp-creational',
	title: 'Creational Patterns',
	description: 'Learn creational design patterns that deal with object creation mechanisms.',
	section: 'design-patterns',
	order: 1,
	difficulty: 'medium',
	estimatedTime: '6h',
	topics: [patternsTopic],
};
