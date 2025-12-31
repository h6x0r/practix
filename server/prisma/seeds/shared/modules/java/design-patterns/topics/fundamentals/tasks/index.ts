import { Task } from '../../../../../../types';
import { task as singleton } from './01-singleton';
import { task as builder } from './02-builder';
import { task as factory } from './03-factory';
import { task as strategy } from './04-strategy';

export const tasks: Task[] = [
    singleton,
    builder,
    factory,
    strategy,
];
