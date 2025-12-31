import { Module } from '../../../../types';
import { patternsTopic } from './topics';

export const behavioralModule: Module = {
	slug: 'go-dp-behavioral',
	title: 'Behavioral Patterns',
	description: 'Learn behavioral design patterns that deal with object collaboration.',
	section: 'design-patterns',
	order: 3,
	difficulty: 'hard',
	estimatedTime: '8h',
	topics: [patternsTopic],
};
