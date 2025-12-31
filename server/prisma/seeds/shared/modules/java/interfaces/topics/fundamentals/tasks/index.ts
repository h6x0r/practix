import { Task } from '../../../../../../types';
import { task as interfaceBasics } from './01-interface-basics';
import { task as multipleInterfaces } from './02-multiple-interfaces';
import { task as defaultMethods } from './03-default-methods';
import { task as staticMethods } from './04-static-methods';
import { task as functionalInterfaces } from './05-functional-interfaces';
import { task as interfaceInheritance } from './06-interface-inheritance';

export const tasks: Task[] = [
    interfaceBasics,
    multipleInterfaces,
    defaultMethods,
    staticMethods,
    functionalInterfaces,
    interfaceInheritance,
];