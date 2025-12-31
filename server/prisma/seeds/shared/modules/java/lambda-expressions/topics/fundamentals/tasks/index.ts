import { Task } from '../../../../../../types';
import { task as lambdaSyntax } from './01-lambda-syntax';
import { task as functionalInterfaces } from './02-functional-interfaces';
import { task as methodReferences } from './03-method-references';
import { task as builtinInterfaces } from './04-builtin-interfaces';
import { task as lambdaScope } from './05-lambda-scope';
import { task as comparatorLambdas } from './06-comparator-lambdas';
import { task as lambdaBestPractices } from './07-lambda-best-practices';

export const tasks: Task[] = [
    lambdaSyntax,
    functionalInterfaces,
    methodReferences,
    builtinInterfaces,
    lambdaScope,
    comparatorLambdas,
    lambdaBestPractices,
];
