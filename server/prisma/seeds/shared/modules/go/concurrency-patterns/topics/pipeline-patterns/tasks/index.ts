/**
 * Pipeline Patterns - Tasks Index
 * Exports all tasks in order
 */

import { task as gen } from './01-gen';
import { task as genWithContext } from './02-gen-with-context';
import { task as square } from './03-square';
import { task as squareStage } from './04-square-stage';
import { task as multiplyStage } from './05-multiply-stage';
import { task as filterStage } from './06-filter-stage';
import { task as takeStage } from './07-take-stage';
import { task as fanIn } from './08-fan-in';
import { task as sum } from './09-sum';
import { task as buildPipeline } from './10-build-pipeline';

export const tasks = [
	gen,
	genWithContext,
	square,
	squareStage,
	multiplyStage,
	filterStage,
	takeStage,
	fanIn,
	sum,
	buildPipeline,
];
