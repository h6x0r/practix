import { Module } from '../../../../types';
import { patternsTopic } from './topics';

export const creationalModule: Module = {
	slug: 'java-dp-creational',
	title: 'Creational Patterns',
	description: 'Patterns that deal with object creation mechanisms.',
	estimatedTime: '5h',
	order: 0,
	topics: [patternsTopic],
};
