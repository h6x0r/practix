import { Course } from '../../types';
import { courseMeta } from './course';
import { modules } from './modules';

export const goConcurrencyCourse: Course = {
	...courseMeta,
	modules,
};
