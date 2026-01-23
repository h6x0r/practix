import { Course } from '../../types';
import courseMeta from './course';
import { syntaxFundamentalsModule } from './modules/syntax-fundamentals';
import { controlFlowModule } from './modules/control-flow';
import { dataStructuresModule } from './modules/data-structures';
import { functionsModule } from './modules/functions';
import { oopBasicsModule } from './modules/oop-basics';

const modules = [
	syntaxFundamentalsModule,
	controlFlowModule,
	dataStructuresModule,
	functionsModule,
	oopBasicsModule,
];

export const pythonFundamentalsCourse: Course = {
	...courseMeta,
	modules,
};

export default pythonFundamentalsCourse;
