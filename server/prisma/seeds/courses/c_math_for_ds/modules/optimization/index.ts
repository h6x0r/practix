import { Module } from '../../../../types';
import moduleMeta from './module';
import { topics } from './topics';

export const optimizationModule: Module = {
	...moduleMeta,
	topics,
};
