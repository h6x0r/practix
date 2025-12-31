import { Task } from '../../../../../../types';
import { task as hashsetBasics } from './01-hashset-basics';
import { task as treeset } from './02-treeset';
import { task as hashmapBasics } from './03-hashmap-basics';
import { task as treemap } from './04-treemap';
import { task as linkedhashmap } from './05-linkedhashmap';
import { task as lruCache } from './06-lru-cache';

export const tasks: Task[] = [
    hashsetBasics,
    treeset,
    hashmapBasics,
    treemap,
    linkedhashmap,
    lruCache,
];
