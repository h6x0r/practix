import { Module } from '../../../../types';
import { patternsTopic } from './topics';

export const structuralModule: Module = {
	slug: 'go-dp-structural',
	title: 'Structural Patterns',
	description: 'Learn structural design patterns that deal with object composition.',
	section: 'design-patterns',
	order: 2,
	difficulty: 'medium',
	estimatedTime: '7h',
	topics: [patternsTopic],
};
