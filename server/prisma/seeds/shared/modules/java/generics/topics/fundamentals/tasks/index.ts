import { Task } from '../../../../../../types';
import { task as genericClasses } from './01-generic-classes';
import { task as genericMethods } from './02-generic-methods';
import { task as boundedTypes } from './03-bounded-types';
import { task as wildcards } from './04-wildcards';
import { task as wildcardBounds } from './05-wildcard-bounds';
import { task as typeErasure } from './06-type-erasure';
import { task as genericPatterns } from './07-generic-patterns';

export const tasks: Task[] = [
    genericClasses,
    genericMethods,
    boundedTypes,
    wildcards,
    wildcardBounds,
    typeErasure,
    genericPatterns,
];
