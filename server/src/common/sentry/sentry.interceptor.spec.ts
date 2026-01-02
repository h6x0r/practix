import { ExecutionContext, HttpException, HttpStatus } from '@nestjs/common';
import { of, throwError } from 'rxjs';
import { SentryInterceptor } from './sentry.interceptor';
import * as Sentry from '@sentry/node';

// Mock Sentry
jest.mock('@sentry/node', () => ({
  withScope: jest.fn((callback) => {
    const mockScope = {
      setExtra: jest.fn(),
      setUser: jest.fn(),
      setTag: jest.fn(),
    };
    callback(mockScope);
  }),
  captureException: jest.fn(),
}));

describe('SentryInterceptor', () => {
  let interceptor: SentryInterceptor;
  let mockExecutionContext: ExecutionContext;
  let mockCallHandler: any;
  let mockRequest: any;

  beforeEach(() => {
    interceptor = new SentryInterceptor();

    mockRequest = {
      url: '/api/test',
      method: 'POST',
      body: { data: 'test' },
      params: { id: '123' },
      query: { page: '1' },
      user: { id: 'user-123', email: 'test@example.com' },
    };

    mockExecutionContext = {
      switchToHttp: jest.fn().mockReturnValue({
        getRequest: () => mockRequest,
      }),
      getHandler: jest.fn().mockReturnValue({ name: 'testHandler' }),
      getClass: jest.fn().mockReturnValue({ name: 'TestController' }),
    } as unknown as ExecutionContext;

    mockCallHandler = {
      handle: jest.fn(),
    };

    jest.clearAllMocks();
  });

  it('should be defined', () => {
    expect(interceptor).toBeDefined();
  });

  describe('successful requests', () => {
    it('should pass through successful responses', (done) => {
      mockCallHandler.handle.mockReturnValue(of({ data: 'success' }));

      interceptor.intercept(mockExecutionContext, mockCallHandler).subscribe({
        next: (value) => {
          expect(value).toEqual({ data: 'success' });
          expect(Sentry.captureException).not.toHaveBeenCalled();
          done();
        },
      });
    });
  });

  describe('skipped HTTP exceptions', () => {
    const skipStatuses = [
      HttpStatus.BAD_REQUEST,
      HttpStatus.UNAUTHORIZED,
      HttpStatus.FORBIDDEN,
      HttpStatus.NOT_FOUND,
      HttpStatus.CONFLICT,
      HttpStatus.TOO_MANY_REQUESTS,
    ];

    skipStatuses.forEach((status) => {
      it(`should not capture ${status} errors in Sentry`, (done) => {
        const error = new HttpException('Error', status);
        mockCallHandler.handle.mockReturnValue(throwError(() => error));

        interceptor.intercept(mockExecutionContext, mockCallHandler).subscribe({
          error: (err) => {
            expect(err).toBe(error);
            expect(Sentry.captureException).not.toHaveBeenCalled();
            done();
          },
        });
      });
    });
  });

  describe('captured exceptions', () => {
    it('should capture 500 Internal Server Error', (done) => {
      const error = new HttpException('Server Error', HttpStatus.INTERNAL_SERVER_ERROR);
      mockCallHandler.handle.mockReturnValue(throwError(() => error));

      interceptor.intercept(mockExecutionContext, mockCallHandler).subscribe({
        error: (err) => {
          expect(err).toBe(error);
          expect(Sentry.captureException).toHaveBeenCalledWith(error);
          done();
        },
      });
    });

    it('should capture 503 Service Unavailable', (done) => {
      const error = new HttpException('Service Unavailable', HttpStatus.SERVICE_UNAVAILABLE);
      mockCallHandler.handle.mockReturnValue(throwError(() => error));

      interceptor.intercept(mockExecutionContext, mockCallHandler).subscribe({
        error: (err) => {
          expect(Sentry.captureException).toHaveBeenCalledWith(error);
          done();
        },
      });
    });

    it('should capture generic Error', (done) => {
      const error = new Error('Generic error');
      mockCallHandler.handle.mockReturnValue(throwError(() => error));

      interceptor.intercept(mockExecutionContext, mockCallHandler).subscribe({
        error: (err) => {
          expect(err).toBe(error);
          expect(Sentry.captureException).toHaveBeenCalledWith(error);
          done();
        },
      });
    });

    it('should add request context to Sentry scope', (done) => {
      const error = new Error('Test error');
      mockCallHandler.handle.mockReturnValue(throwError(() => error));

      interceptor.intercept(mockExecutionContext, mockCallHandler).subscribe({
        error: () => {
          expect(Sentry.withScope).toHaveBeenCalled();
          done();
        },
      });
    });

    it('should add user context when available', (done) => {
      const error = new Error('Test error');
      mockCallHandler.handle.mockReturnValue(throwError(() => error));

      interceptor.intercept(mockExecutionContext, mockCallHandler).subscribe({
        error: () => {
          expect(Sentry.withScope).toHaveBeenCalled();
          done();
        },
      });
    });

    it('should handle request without user context', (done) => {
      mockRequest.user = undefined;
      const error = new Error('Test error');
      mockCallHandler.handle.mockReturnValue(throwError(() => error));

      interceptor.intercept(mockExecutionContext, mockCallHandler).subscribe({
        error: () => {
          expect(Sentry.captureException).toHaveBeenCalledWith(error);
          done();
        },
      });
    });
  });

  describe('edge cases', () => {
    it('should handle TypeError', (done) => {
      const error = new TypeError('Cannot read property');
      mockCallHandler.handle.mockReturnValue(throwError(() => error));

      interceptor.intercept(mockExecutionContext, mockCallHandler).subscribe({
        error: (err) => {
          expect(Sentry.captureException).toHaveBeenCalledWith(error);
          done();
        },
      });
    });

    it('should handle ReferenceError', (done) => {
      const error = new ReferenceError('Variable not defined');
      mockCallHandler.handle.mockReturnValue(throwError(() => error));

      interceptor.intercept(mockExecutionContext, mockCallHandler).subscribe({
        error: (err) => {
          expect(Sentry.captureException).toHaveBeenCalledWith(error);
          done();
        },
      });
    });

    it('should rethrow error after capturing', (done) => {
      const error = new Error('Original error');
      mockCallHandler.handle.mockReturnValue(throwError(() => error));

      interceptor.intercept(mockExecutionContext, mockCallHandler).subscribe({
        error: (err) => {
          expect(err).toBe(error);
          expect(err.message).toBe('Original error');
          done();
        },
      });
    });
  });
});
