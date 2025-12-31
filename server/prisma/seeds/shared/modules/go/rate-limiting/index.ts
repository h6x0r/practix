import { Module } from '../../../../types';
import { moduleMeta } from './module';
import { topics } from './topics';

export const rateLimitingModule: Module = {
	...moduleMeta,
	topics,
};
