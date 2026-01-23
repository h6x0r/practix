import { Module } from '../../../../types';
import moduleMeta from './module';
import { topics } from './topics';

export const dataStructuresModule: Module = {
	...moduleMeta,
	topics,
};
