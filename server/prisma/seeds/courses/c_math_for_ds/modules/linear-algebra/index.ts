import { Module } from '../../../../types';
import moduleMeta from './module';
import { topics } from './topics';

export const linearAlgebraModule: Module = {
	...moduleMeta,
	topics,
};
