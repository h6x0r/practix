import { Task } from '../../../../../../types';
import { task as sealedBasics } from './01-sealed-basics';
import { task as sealedInterfaces } from './02-sealed-interfaces';
import { task as sealedHierarchies } from './03-sealed-hierarchies';

export const tasks: Task[] = [
    sealedBasics,
    sealedInterfaces,
    sealedHierarchies,
];
