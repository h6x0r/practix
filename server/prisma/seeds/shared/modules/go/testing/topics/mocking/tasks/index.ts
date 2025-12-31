/**
 * Mocking and Test Doubles - Tasks Index
 * Exports all tasks in order
 */

import { task as interfaceMock } from './01-interface-mock';
import { task as mockHttpClient } from './02-mock-http-client';
import { task as mockDatabase } from './03-mock-database';
import { task as spyPattern } from './04-spy-pattern';
import { task as fakeImplementation } from './05-fake-implementation';

export const tasks = [
	interfaceMock,
	mockHttpClient,
	mockDatabase,
	spyPattern,
	fakeImplementation,
];
