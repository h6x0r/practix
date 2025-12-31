import { task as informationExpert } from './01-information-expert';
import { task as creator } from './02-creator';
import { task as controller } from './03-controller';
import { task as lowCoupling } from './04-low-coupling';
import { task as highCohesion } from './05-high-cohesion';
import { task as polymorphism } from './06-polymorphism';
import { task as pureFabrication } from './07-pure-fabrication';
import { task as indirection } from './08-indirection';
import { task as protectedVariations } from './09-protected-variations';

export const tasks = [
	informationExpert,
	creator,
	controller,
	lowCoupling,
	highCohesion,
	polymorphism,
	pureFabrication,
	indirection,
	protectedVariations,
];
