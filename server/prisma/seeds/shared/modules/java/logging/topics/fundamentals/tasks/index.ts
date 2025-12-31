import { Task } from '../../../../../../types';
import { task as slf4jBasics } from './01-slf4j-basics';
import { task as logLevels } from './02-log-levels';
import { task as parameterizedLogging } from './03-parameterized-logging';
import { task as mdcContext } from './04-mdc-context';
import { task as logbackConfiguration } from './05-logback-configuration';
import { task as loggingBestPractices } from './06-logging-best-practices';

export const tasks: Task[] = [
    slf4jBasics,
    logLevels,
    parameterizedLogging,
    mdcContext,
    logbackConfiguration,
    loggingBestPractices,
];
