import { Task } from '../../../../../../types';
import { task as micrometerBasics } from './01-micrometer-basics';
import { task as counters } from './02-counters';
import { task as gauges } from './03-gauges';
import { task as timers } from './04-timers';
import { task as distributionSummary } from './05-distribution-summary';

export const tasks: Task[] = [
    micrometerBasics,
    counters,
    gauges,
    timers,
    distributionSummary,
];
