import { Task } from '../../../../../../types';
import { task as localDateTime } from './01-local-datetime';
import { task as instantDuration } from './02-instant-duration';
import { task as zonedDateTime } from './03-zoned-datetime';
import { task as dateFormatting } from './04-date-formatting';
import { task as dateManipulation } from './05-date-manipulation';

export const tasks: Task[] = [
    localDateTime,
    instantDuration,
    zonedDateTime,
    dateFormatting,
    dateManipulation,
];
