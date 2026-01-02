import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { logger, createLogger } from './logger';

// Test for production mode logging
describe('logger in production mode', () => {
  it('should filter debug/info logs when minLevel is higher', async () => {
    // Reset modules to get fresh import
    vi.resetModules();

    // Stub import.meta.env.DEV to false (production mode)
    vi.stubEnv('DEV', false);

    // Create spies
    const consoleDebugSpy = vi.spyOn(console, 'debug').mockImplementation(() => {});
    const consoleInfoSpy = vi.spyOn(console, 'info').mockImplementation(() => {});
    const consoleWarnSpy = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const consoleErrorSpy = vi.spyOn(console, 'error').mockImplementation(() => {});

    // Import fresh module with production mode
    const { logger: prodLogger } = await import('./logger');

    // In production mode, minLevel='warn', so debug/info should NOT log
    prodLogger.debug('Test', 'Debug message');
    prodLogger.info('Test', 'Info message');
    prodLogger.warn('Test', 'Warning message');
    prodLogger.error('Test', 'Error message');

    // debug/info should NOT be called (level < minLevel)
    expect(consoleDebugSpy).not.toHaveBeenCalled();
    expect(consoleInfoSpy).not.toHaveBeenCalled();
    // warn/error should be called (level >= minLevel)
    expect(consoleWarnSpy).toHaveBeenCalled();
    expect(consoleErrorSpy).toHaveBeenCalled();

    // Clean up
    consoleDebugSpy.mockRestore();
    consoleInfoSpy.mockRestore();
    consoleWarnSpy.mockRestore();
    consoleErrorSpy.mockRestore();
    vi.unstubAllEnvs();
  });
});

describe('logger', () => {
  let consoleDebugSpy: any;
  let consoleInfoSpy: any;
  let consoleWarnSpy: any;
  let consoleErrorSpy: any;

  beforeEach(() => {
    consoleDebugSpy = vi.spyOn(console, 'debug').mockImplementation(() => {});
    consoleInfoSpy = vi.spyOn(console, 'info').mockImplementation(() => {});
    consoleWarnSpy = vi.spyOn(console, 'warn').mockImplementation(() => {});
    consoleErrorSpy = vi.spyOn(console, 'error').mockImplementation(() => {});
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  describe('logger.debug', () => {
    it('should log debug message with context', () => {
      logger.debug('TestContext', 'Debug message');

      expect(consoleDebugSpy).toHaveBeenCalled();
      const loggedMessage = consoleDebugSpy.mock.calls[0][0];
      expect(loggedMessage).toContain('[DEBUG]');
      expect(loggedMessage).toContain('[TestContext]');
      expect(loggedMessage).toContain('Debug message');
    });

    it('should include additional arguments', () => {
      logger.debug('Test', 'Message with data', { key: 'value' });

      expect(consoleDebugSpy).toHaveBeenCalledWith(
        expect.any(String),
        { key: 'value' }
      );
    });
  });

  describe('logger.info', () => {
    it('should log info message with context', () => {
      logger.info('TestContext', 'Info message');

      expect(consoleInfoSpy).toHaveBeenCalled();
      const loggedMessage = consoleInfoSpy.mock.calls[0][0];
      expect(loggedMessage).toContain('[INFO]');
      expect(loggedMessage).toContain('[TestContext]');
      expect(loggedMessage).toContain('Info message');
    });
  });

  describe('logger.warn', () => {
    it('should log warning message with context', () => {
      logger.warn('TestContext', 'Warning message');

      expect(consoleWarnSpy).toHaveBeenCalled();
      const loggedMessage = consoleWarnSpy.mock.calls[0][0];
      expect(loggedMessage).toContain('[WARN]');
      expect(loggedMessage).toContain('[TestContext]');
      expect(loggedMessage).toContain('Warning message');
    });
  });

  describe('logger.error', () => {
    it('should log error message with context', () => {
      logger.error('TestContext', 'Error message');

      expect(consoleErrorSpy).toHaveBeenCalled();
      const loggedMessage = consoleErrorSpy.mock.calls[0][0];
      expect(loggedMessage).toContain('[ERROR]');
      expect(loggedMessage).toContain('[TestContext]');
      expect(loggedMessage).toContain('Error message');
    });

    it('should include error object when provided', () => {
      const testError = new Error('Test error');
      logger.error('TestContext', 'Error occurred', testError);

      expect(consoleErrorSpy).toHaveBeenCalledWith(
        expect.any(String),
        testError
      );
    });

    it('should handle non-Error objects', () => {
      logger.error('TestContext', 'Error occurred', 'string error');

      expect(consoleErrorSpy).toHaveBeenCalledWith(
        expect.any(String),
        undefined
      );
    });
  });

  describe('timestamp format', () => {
    it('should include ISO timestamp in log message', () => {
      logger.info('Test', 'Message');

      const loggedMessage = consoleInfoSpy.mock.calls[0][0];
      // Check for ISO date format pattern
      expect(loggedMessage).toMatch(/\[\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}/);
    });
  });
});

describe('createLogger', () => {
  let consoleDebugSpy: any;
  let consoleInfoSpy: any;
  let consoleWarnSpy: any;
  let consoleErrorSpy: any;

  beforeEach(() => {
    consoleDebugSpy = vi.spyOn(console, 'debug').mockImplementation(() => {});
    consoleInfoSpy = vi.spyOn(console, 'info').mockImplementation(() => {});
    consoleWarnSpy = vi.spyOn(console, 'warn').mockImplementation(() => {});
    consoleErrorSpy = vi.spyOn(console, 'error').mockImplementation(() => {});
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('should create a scoped logger with context', () => {
    const log = createLogger('MyComponent');

    log.debug('Debug from component');
    log.info('Info from component');
    log.warn('Warning from component');
    log.error('Error from component');

    expect(consoleDebugSpy.mock.calls[0][0]).toContain('[MyComponent]');
    expect(consoleInfoSpy.mock.calls[0][0]).toContain('[MyComponent]');
    expect(consoleWarnSpy.mock.calls[0][0]).toContain('[MyComponent]');
    expect(consoleErrorSpy.mock.calls[0][0]).toContain('[MyComponent]');
  });

  it('should pass additional arguments', () => {
    const log = createLogger('TestComponent');
    const data = { id: 1, name: 'test' };

    log.debug('Message', data);

    expect(consoleDebugSpy).toHaveBeenCalledWith(
      expect.any(String),
      data
    );
  });

  it('should pass error object to error method', () => {
    const log = createLogger('TestComponent');
    const error = new Error('Test error');

    log.error('Something failed', error);

    expect(consoleErrorSpy).toHaveBeenCalledWith(
      expect.any(String),
      error
    );
  });

  it('should pass multiple arguments to info', () => {
    const log = createLogger('TestComponent');

    log.info('Message with multiple args', 'arg1', 42, { obj: true });

    expect(consoleInfoSpy).toHaveBeenCalledWith(
      expect.any(String),
      'arg1',
      42,
      { obj: true }
    );
  });

  it('should pass multiple arguments to warn', () => {
    const log = createLogger('TestComponent');

    log.warn('Warning with args', { warning: 'data' });

    expect(consoleWarnSpy).toHaveBeenCalledWith(
      expect.any(String),
      { warning: 'data' }
    );
  });
});

describe('logger log levels', () => {
  let consoleDebugSpy: any;
  let consoleInfoSpy: any;
  let consoleWarnSpy: any;
  let consoleErrorSpy: any;

  beforeEach(() => {
    consoleDebugSpy = vi.spyOn(console, 'debug').mockImplementation(() => {});
    consoleInfoSpy = vi.spyOn(console, 'info').mockImplementation(() => {});
    consoleWarnSpy = vi.spyOn(console, 'warn').mockImplementation(() => {});
    consoleErrorSpy = vi.spyOn(console, 'error').mockImplementation(() => {});
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('should log all levels in development', () => {
    // In test environment, isDevelopment is true, so all levels should log
    logger.debug('Test', 'Debug message');
    logger.info('Test', 'Info message');
    logger.warn('Test', 'Warn message');
    logger.error('Test', 'Error message');

    expect(consoleDebugSpy).toHaveBeenCalled();
    expect(consoleInfoSpy).toHaveBeenCalled();
    expect(consoleWarnSpy).toHaveBeenCalled();
    expect(consoleErrorSpy).toHaveBeenCalled();
  });

  it('should format message with correct level labels', () => {
    logger.debug('Ctx', 'msg');
    logger.info('Ctx', 'msg');
    logger.warn('Ctx', 'msg');
    logger.error('Ctx', 'msg');

    expect(consoleDebugSpy.mock.calls[0][0]).toContain('[DEBUG]');
    expect(consoleInfoSpy.mock.calls[0][0]).toContain('[INFO]');
    expect(consoleWarnSpy.mock.calls[0][0]).toContain('[WARN]');
    expect(consoleErrorSpy.mock.calls[0][0]).toContain('[ERROR]');
  });
});
