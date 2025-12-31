import moduleMeta from './module';
import topics from './topics';
import { Module } from '../../../../types';

const module: Module = {
	...moduleMeta,
	topics,
};

export default module;
