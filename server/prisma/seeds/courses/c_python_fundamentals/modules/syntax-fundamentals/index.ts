import { Module } from '../../../../types';
import moduleMeta from './module';
import { topics } from './topics';

export const syntaxFundamentalsModule: Module = {
	...moduleMeta,
	topics,
};
