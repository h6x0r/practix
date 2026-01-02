import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { api, ApiError, isAbortError, setupInterceptors } from './client';

// Mock fetch globally
const mockFetch = vi.fn();
global.fetch = mockFetch;

// Mock logger
vi.mock('../logger', () => ({
  createLogger: () => ({
    info: vi.fn(),
    warn: vi.fn(),
    error: vi.fn(),
    debug: vi.fn(),
  }),
}));

// Mock env
vi.mock('../../config/env', () => ({
  ENV: {
    API_URL: 'http://test-api.com',
  },
  getHeaders: () => ({
    'Content-Type': 'application/json',
    'Authorization': 'Bearer test-token',
  }),
}));

describe('API Client', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  describe('api.get', () => {
    it('should make GET request with correct URL and headers', async () => {
      const mockData = { id: 1, name: 'Test' };
      mockFetch.mockResolvedValueOnce({
        ok: true,
        status: 200,
        json: () => Promise.resolve(mockData),
      });

      const result = await api.get('/users/1');

      expect(mockFetch).toHaveBeenCalledWith(
        'http://test-api.com/users/1',
        expect.objectContaining({
          method: 'GET',
          headers: {
            'Content-Type': 'application/json',
            'Authorization': 'Bearer test-token',
          },
        })
      );
      expect(result).toEqual(mockData);
    });

    it('should pass abort signal to fetch', async () => {
      const controller = new AbortController();
      mockFetch.mockResolvedValueOnce({
        ok: true,
        status: 200,
        json: () => Promise.resolve({}),
      });

      await api.get('/test', { signal: controller.signal });

      expect(mockFetch).toHaveBeenCalledWith(
        expect.any(String),
        expect.objectContaining({
          signal: controller.signal,
        })
      );
    });

    it('should throw ApiError on 404', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 404,
        statusText: 'Not Found',
        json: () => Promise.resolve({ message: 'Resource not found' }),
      });

      try {
        await api.get('/not-found');
        expect.fail('Should have thrown');
      } catch (error) {
        expect(error).toBeInstanceOf(ApiError);
        expect((error as ApiError).status).toBe(404);
      }
    });
  });

  describe('api.post', () => {
    it('should make POST request with body', async () => {
      const requestBody = { name: 'New User' };
      const responseData = { id: 1, name: 'New User' };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        status: 201,
        json: () => Promise.resolve(responseData),
      });

      const result = await api.post('/users', requestBody);

      expect(mockFetch).toHaveBeenCalledWith(
        'http://test-api.com/users',
        expect.objectContaining({
          method: 'POST',
          body: JSON.stringify(requestBody),
        })
      );
      expect(result).toEqual(responseData);
    });
  });

  describe('api.patch', () => {
    it('should make PATCH request with body', async () => {
      const requestBody = { name: 'Updated' };
      mockFetch.mockResolvedValueOnce({
        ok: true,
        status: 200,
        json: () => Promise.resolve({ id: 1, ...requestBody }),
      });

      await api.patch('/users/1', requestBody);

      expect(mockFetch).toHaveBeenCalledWith(
        'http://test-api.com/users/1',
        expect.objectContaining({
          method: 'PATCH',
          body: JSON.stringify(requestBody),
        })
      );
    });
  });

  describe('api.delete', () => {
    it('should make DELETE request', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        status: 200,
        json: () => Promise.resolve({ success: true }),
      });

      await api.delete('/users/1');

      expect(mockFetch).toHaveBeenCalledWith(
        'http://test-api.com/users/1',
        expect.objectContaining({
          method: 'DELETE',
        })
      );
    });
  });

  describe('response handling', () => {
    it('should handle 204 No Content', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        status: 204,
      });

      const result = await api.get('/resource');

      expect(result).toEqual({});
    });

    it('should throw on invalid JSON response', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        status: 200,
        json: () => Promise.reject(new Error('Invalid JSON')),
      });

      await expect(api.get('/test')).rejects.toThrow('Failed to parse response');
    });

    it('should throw on invalid JSON with error status', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 500,
        json: () => Promise.reject(new Error('Invalid JSON')),
      });

      await expect(api.get('/test')).rejects.toThrow('Invalid JSON response from server');
    });
  });

  describe('401 Unauthorized handling', () => {
    it('should call onUnauthorized callback on 401', async () => {
      const onUnauthorized = vi.fn();
      setupInterceptors(onUnauthorized);

      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 401,
        json: () => Promise.resolve({ message: 'Unauthorized' }),
      });

      await expect(api.get('/protected')).rejects.toThrow('Session expired');
      expect(onUnauthorized).toHaveBeenCalled();
    });

    it('should throw ApiError with 401 status', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 401,
        json: () => Promise.resolve({}),
      });

      try {
        await api.get('/protected');
      } catch (error) {
        expect(error).toBeInstanceOf(ApiError);
        expect((error as ApiError).status).toBe(401);
      }
    });
  });

  describe('retry logic', () => {
    it('should retry on network error', async () => {
      const networkError = new TypeError('Failed to fetch');

      mockFetch
        .mockRejectedValueOnce(networkError)
        .mockRejectedValueOnce(networkError)
        .mockResolvedValueOnce({
          ok: true,
          status: 200,
          json: () => Promise.resolve({ success: true }),
        });

      const result = await api.get('/test');

      expect(mockFetch).toHaveBeenCalledTimes(3);
      expect(result).toEqual({ success: true });
    }, 15000);

    it('should throw after max retries', async () => {
      const networkError = new TypeError('Failed to fetch');

      mockFetch
        .mockRejectedValueOnce(networkError)
        .mockRejectedValueOnce(networkError)
        .mockRejectedValueOnce(networkError);

      await expect(api.get('/test')).rejects.toThrow('Failed to fetch');
      expect(mockFetch).toHaveBeenCalledTimes(3);
    }, 15000);

    it('should not retry on non-network errors', async () => {
      const error = new Error('Some other error');
      mockFetch.mockRejectedValueOnce(error);

      await expect(api.get('/test')).rejects.toThrow('Some other error');
      expect(mockFetch).toHaveBeenCalledTimes(1);
    });

    it('should not retry aborted requests', async () => {
      const abortError = new DOMException('Aborted', 'AbortError');
      mockFetch.mockRejectedValueOnce(abortError);

      await expect(api.get('/test')).rejects.toThrow();
      expect(mockFetch).toHaveBeenCalledTimes(1);
    });
  });

  describe('ApiError', () => {
    it('should create ApiError with correct properties', () => {
      const error = new ApiError('Test error', 400, { field: 'value' });

      expect(error.message).toBe('Test error');
      expect(error.status).toBe(400);
      expect(error.data).toEqual({ field: 'value' });
      expect(error.name).toBe('ApiError');
    });
  });

  describe('isAbortError', () => {
    it('should return true for AbortError', () => {
      const abortError = new DOMException('Aborted', 'AbortError');
      expect(isAbortError(abortError)).toBe(true);
    });

    it('should return false for other errors', () => {
      expect(isAbortError(new Error('Regular error'))).toBe(false);
      expect(isAbortError(new TypeError('Type error'))).toBe(false);
      expect(isAbortError(null)).toBe(false);
      expect(isAbortError(undefined)).toBe(false);
    });
  });

  describe('error message extraction', () => {
    it('should use message field from error response', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 400,
        statusText: 'Bad Request',
        json: () => Promise.resolve({ message: 'Custom error message' }),
      });

      await expect(api.get('/test')).rejects.toMatchObject({
        message: 'Custom error message',
      });
    });

    it('should use error field if message not present', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 400,
        statusText: 'Bad Request',
        json: () => Promise.resolve({ error: 'Error field message' }),
      });

      await expect(api.get('/test')).rejects.toMatchObject({
        message: 'Error field message',
      });
    });

    it('should fallback to statusText', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 400,
        statusText: 'Bad Request',
        json: () => Promise.resolve({}),
      });

      await expect(api.get('/test')).rejects.toMatchObject({
        message: 'Bad Request',
      });
    });
  });

  describe('401 without callback', () => {
    it('should throw without calling callback if not set', async () => {
      // Reset modules to get fresh state without callback set
      vi.resetModules();

      // Re-mock fetch for the fresh module
      const freshMockFetch = vi.fn();
      vi.stubGlobal('fetch', freshMockFetch);

      freshMockFetch.mockResolvedValueOnce({
        ok: false,
        status: 401,
        json: () => Promise.resolve({}),
      });

      // Import fresh module - callback will be null
      const { api: freshApi } = await import('./client');

      await expect(freshApi.get('/protected')).rejects.toThrow('Session expired');

      // Restore original mock
      vi.stubGlobal('fetch', mockFetch);
    });
  });

  describe('network error detection', () => {
    it('should detect TypeError with "Failed to fetch" as network error', async () => {
      const networkError = new TypeError('Failed to fetch');
      mockFetch
        .mockRejectedValueOnce(networkError)
        .mockResolvedValueOnce({
          ok: true,
          status: 200,
          json: () => Promise.resolve({ success: true }),
        });

      const result = await api.get('/test');

      expect(mockFetch).toHaveBeenCalledTimes(2);
      expect(result).toEqual({ success: true });
    }, 15000);

    it('should detect TypeError with "network" in message as network error', async () => {
      const networkError = new TypeError('Network request failed');
      mockFetch
        .mockRejectedValueOnce(networkError)
        .mockResolvedValueOnce({
          ok: true,
          status: 200,
          json: () => Promise.resolve({ success: true }),
        });

      const result = await api.get('/test');

      expect(mockFetch).toHaveBeenCalledTimes(2);
      expect(result).toEqual({ success: true });
    }, 15000);

    it('should not retry regular TypeError', async () => {
      const typeError = new TypeError('Cannot read property x of undefined');
      mockFetch.mockRejectedValueOnce(typeError);

      await expect(api.get('/test')).rejects.toThrow('Cannot read property x of undefined');
      expect(mockFetch).toHaveBeenCalledTimes(1);
    });
  });

  describe('request methods with signals', () => {
    it('should pass signal to PATCH request', async () => {
      const controller = new AbortController();
      mockFetch.mockResolvedValueOnce({
        ok: true,
        status: 200,
        json: () => Promise.resolve({ updated: true }),
      });

      await api.patch('/resource/1', { name: 'Updated' }, { signal: controller.signal });

      expect(mockFetch).toHaveBeenCalledWith(
        expect.any(String),
        expect.objectContaining({
          signal: controller.signal,
        })
      );
    });

    it('should pass signal to DELETE request', async () => {
      const controller = new AbortController();
      mockFetch.mockResolvedValueOnce({
        ok: true,
        status: 200,
        json: () => Promise.resolve({ deleted: true }),
      });

      await api.delete('/resource/1', { signal: controller.signal });

      expect(mockFetch).toHaveBeenCalledWith(
        expect.any(String),
        expect.objectContaining({
          signal: controller.signal,
        })
      );
    });
  });

  describe('exponential backoff', () => {
    it('should use exponential backoff delays', async () => {
      const networkError = new TypeError('Failed to fetch');
      const startTime = Date.now();

      mockFetch
        .mockRejectedValueOnce(networkError) // 1st attempt
        .mockRejectedValueOnce(networkError) // 2nd attempt
        .mockResolvedValueOnce({
          ok: true,
          status: 200,
          json: () => Promise.resolve({ success: true }),
        });

      await api.get('/test');

      const elapsedTime = Date.now() - startTime;
      // Should have waited at least baseDelay (1000ms) + 2*baseDelay (2000ms) = ~3000ms
      expect(elapsedTime).toBeGreaterThanOrEqual(2900);
    }, 15000);
  });
});
