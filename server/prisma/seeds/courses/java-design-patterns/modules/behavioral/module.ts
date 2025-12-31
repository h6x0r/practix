import { Module } from '../../../../types';
import { patternsTopic } from './topics';

export const behavioralModule: Module = {
	slug: 'java-dp-behavioral',
	title: 'Behavioral Patterns',
	description: 'Patterns that deal with object interaction and responsibility.',
	estimatedTime: '9h',
	order: 2,
	topics: [patternsTopic],
};
