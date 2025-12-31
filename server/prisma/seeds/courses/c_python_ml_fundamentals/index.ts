import courseMeta from './course';
import numpyEssentials from './modules/numpy-essentials';
import pandasMastery from './modules/pandas-mastery';
import dataVisualization from './modules/data-visualization';
import classicalMl from './modules/classical-ml';
import gradientBoosting from './modules/gradient-boosting';
import { Course } from '../../types';

const course: Course = {
	...courseMeta,
	modules: [
		numpyEssentials,
		pandasMastery,
		dataVisualization,
		classicalMl,
		gradientBoosting,
	],
};

export default course;
