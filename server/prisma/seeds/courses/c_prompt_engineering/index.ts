import { Course } from '../../types';
import courseMeta from './course';
import { modules } from './modules';

export const promptEngineeringCourse: Course = {
	...courseMeta,
	modules,
};

export default promptEngineeringCourse;
