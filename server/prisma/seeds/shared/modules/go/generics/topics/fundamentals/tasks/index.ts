import { Task } from '../../../../types';
import { task as genericFunction } from './01-generic-function';
import { task as typeConstraints } from './02-type-constraints';
import { task as genericStruct } from './03-generic-struct';
import { task as comparableConstraint } from './04-comparable-constraint';
import { task as customConstraint } from './05-custom-constraint';
import { task as genericSliceOperations } from './06-generic-slice-operations';
import { task as genericMapOperations } from './07-generic-map-operations';
import { task as genericResultType } from './08-generic-result-type';

export const tasks: Task[] = [
    genericFunction,
    typeConstraints,
    genericStruct,
    comparableConstraint,
    customConstraint,
    genericSliceOperations,
    genericMapOperations,
    genericResultType
];
