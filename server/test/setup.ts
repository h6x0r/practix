/**
 * E2E Test Setup
 * Handles test database cleanup and environment configuration
 */

// Set test environment
process.env.NODE_ENV = 'test';

// Increase timeout for E2E tests
jest.setTimeout(30000);

// Suppress console logs during tests (optional)
// global.console.log = jest.fn();
// global.console.debug = jest.fn();
