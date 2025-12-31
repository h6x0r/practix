// Jest setup file for global mocks and configuration

// Increase test timeout for integration tests
jest.setTimeout(30000);

// Mock console.log to reduce noise in tests
// Comment this out if you need to debug tests
// global.console = {
//   ...console,
//   log: jest.fn(),
//   info: jest.fn(),
//   debug: jest.fn(),
// };
