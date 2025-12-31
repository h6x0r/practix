import { task as envParsing } from './01-env-parsing';
import { task as configValidation } from './02-config-validation';
import { task as configHotReload } from './03-config-hot-reload';
import { task as multiSourceConfig } from './04-multi-source-config';

export const tasks = [
	envParsing,
	configValidation,
	configHotReload,
	multiSourceConfig,
];
