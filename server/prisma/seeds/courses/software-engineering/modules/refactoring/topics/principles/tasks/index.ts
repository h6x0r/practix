import { task as extractMethodLong } from './01-extract-method-long';
import { task as extractMethodCalc } from './02-extract-method-calc';
import { task as extractVariableMagic } from './03-extract-variable-magic';
import { task as extractVariableComplex } from './04-extract-variable-complex';
import { task as renameVariables } from './05-rename-variables';
import { task as renameFunctions } from './06-rename-functions';
import { task as moveMethodEnvy } from './07-move-method-envy';
import { task as moveMethodResponsibility } from './08-move-method-responsibility';
import { task as replaceConditionalType } from './09-replace-conditional-type';
import { task as replaceConditionalState } from './10-replace-conditional-state';
import { task as introduceParamObjectLong } from './11-introduce-param-object-long';
import { task as introduceParamObjectClumps } from './12-introduce-param-object-clumps';

export const tasks = [
	extractMethodLong,
	extractMethodCalc,
	extractVariableMagic,
	extractVariableComplex,
	renameVariables,
	renameFunctions,
	moveMethodEnvy,
	moveMethodResponsibility,
	replaceConditionalType,
	replaceConditionalState,
	introduceParamObjectLong,
	introduceParamObjectClumps,
];
