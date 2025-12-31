import { Task } from '../../../../types';
import { task as poolConfiguration } from './01-pool-configuration';
import { task as connectionLifetime } from './02-connection-lifetime';
import { task as healthCheck } from './03-health-check';
import { task as statsMonitoring } from './04-stats-monitoring';
import { task as gracefulShutdown } from './05-graceful-shutdown';

export const tasks: Task[] = [
    poolConfiguration,
    connectionLifetime,
    healthCheck,
    statsMonitoring,
    gracefulShutdown
];
