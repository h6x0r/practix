import { Course } from '../../types';
import courseMeta from './course';
import { linearAlgebraModule } from './modules/linear-algebra';
import { calculusModule } from './modules/calculus';
import { statisticsModule } from './modules/statistics';
import { optimizationModule } from './modules/optimization';

const modules = [
	linearAlgebraModule,
	calculusModule,
	statisticsModule,
	optimizationModule,
];

export const mathForDsCourse: Course = {
	...courseMeta,
	modules,
};

export default mathForDsCourse;
