import { task as singleton } from './01-singleton';
import { task as factoryMethod } from './02-factory-method';
import { task as abstractFactory } from './03-abstract-factory';
import { task as builder } from './04-builder';
import { task as prototype } from './05-prototype';

export const tasks = [
	singleton,
	factoryMethod,
	abstractFactory,
	builder,
	prototype,
];
