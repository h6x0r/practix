import { task as adapter } from './01-adapter';
import { task as bridge } from './02-bridge';
import { task as composite } from './03-composite';
import { task as decorator } from './04-decorator';
import { task as facade } from './05-facade';
import { task as flyweight } from './06-flyweight';
import { task as proxy } from './07-proxy';

export const tasks = [
	adapter,
	bridge,
	composite,
	decorator,
	facade,
	flyweight,
	proxy,
];
