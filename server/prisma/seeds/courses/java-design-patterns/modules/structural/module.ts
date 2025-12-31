import { Module } from '../../../../types';
import { patternsTopic } from './topics';

export const structuralModule: Module = {
	slug: 'java-dp-structural',
	title: 'Structural Patterns',
	description: 'Patterns that deal with object composition.',
	estimatedTime: '7h',
	order: 1,
	topics: [patternsTopic],
};
