/**
 * Unit Testing Fundamentals - Tasks Index
 * Exports all tasks in order
 */

import { task as basicTest } from './01-basic-test';
import { task as errorAssertions } from './02-error-assertions';
import { task as testHelper } from './03-test-helper';
import { task as httpHandlerTest } from './04-http-handler-test';
import { task as testMain } from './05-test-main';

export const tasks = [
	basicTest,
	errorAssertions,
	testHelper,
	httpHandlerTest,
	testMain,
];
