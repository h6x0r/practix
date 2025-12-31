import moduleMeta from './module';
import arrayBasics from './topics/array-basics';
import arrayOperations from './topics/array-operations';
import linearAlgebra from './topics/linear-algebra';
import { Module } from '../../../../types';

const module: Module = {
	...moduleMeta,
	topics: [arrayBasics, arrayOperations, linearAlgebra],
};

export default module;
