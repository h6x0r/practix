import { Task } from '../../../../../../types';
import { task as propertiesFiles } from './01-properties-files';
import { task as yamlConfig } from './02-yaml-config';

export const tasks: Task[] = [
    propertiesFiles,
    yamlConfig,
];
