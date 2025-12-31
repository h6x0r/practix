import { ENV, getHeaders } from '../../config/env';
import { createLogger } from '../logger';

const log = createLogger('ApiClient');

// Retry configuration
const RETRY_CONFIG = {
  maxAttempts: 3,
  baseDelayMs: 1000, // 1s -> 2s -> 4s (exponential backoff)
  maxDelayMs: 10000,
};

export class ApiError extends Error {
  status: number;
  data: unknown;

  constructor(message: string, status: number, data?: unknown) {
    super(message);
    this.status = status;
    this.data = data;
    this.name = 'ApiError';
  }
}

const isNetworkError = (error: unknown): boolean => {
  return (
    error instanceof TypeError &&
    (error.message.includes('fetch') ||
      error.message.includes('network') ||
      error.message === 'Failed to fetch' ||
      error.message.includes('Network request failed'))
  );
};

const sleep = (ms: number): Promise<void> =>
  new Promise((resolve) => setTimeout(resolve, ms));

const isAbortError = (error: unknown): boolean => {
  return error instanceof DOMException && error.name === 'AbortError';
};

const fetchWithRetry = async (
  url: string,
  options: RequestInit
): Promise<Response> => {
  let lastError: unknown;

  for (let attempt = 1; attempt <= RETRY_CONFIG.maxAttempts; attempt++) {
    try {
      return await fetch(url, options);
    } catch (error) {
      lastError = error;

      // Don't retry aborted requests
      if (isAbortError(error)) {
        throw error;
      }

      if (!isNetworkError(error) || attempt === RETRY_CONFIG.maxAttempts) {
        throw error;
      }

      const delay = Math.min(
        RETRY_CONFIG.baseDelayMs * Math.pow(2, attempt - 1),
        RETRY_CONFIG.maxDelayMs
      );

      log.warn(`Network error, retry ${attempt}/${RETRY_CONFIG.maxAttempts} in ${delay}ms`);
      await sleep(delay);
    }
  }

  throw lastError;
};

// Global callback registry for auth failures (e.g. 401)
let onUnauthorizedCallback: (() => void) | null = null;

export const setupInterceptors = (onUnauthorized: () => void) => {
  onUnauthorizedCallback = onUnauthorized;
};

const handleResponse = async <T>(res: Response): Promise<T> => {
  if (res.status === 401) {
    if (onUnauthorizedCallback) {
      onUnauthorizedCallback();
    }
    throw new ApiError('Session expired', 401);
  }

  // Handle 204 No Content
  if (res.status === 204) {
    return {} as T;
  }

  let data: unknown;
  try {
    data = await res.json();
  } catch (error) {
    log.error('Failed to parse JSON response', error);
    if (!res.ok) {
      throw new ApiError('Invalid JSON response from server', res.status);
    }
    throw new ApiError('Failed to parse response', res.status);
  }

  if (!res.ok) {
    const errorData = data as { message?: string; error?: string };
    throw new ApiError(errorData.message || errorData.error || res.statusText, res.status, data);
  }

  return data as T;
};

interface RequestOptions {
  signal?: AbortSignal;
}

export const api = {
  get: async <T>(endpoint: string, options?: RequestOptions): Promise<T> => {
    const res = await fetchWithRetry(`${ENV.API_URL}${endpoint}`, {
      method: 'GET',
      headers: getHeaders(),
      signal: options?.signal,
    });
    return handleResponse<T>(res);
  },

  post: async <T>(endpoint: string, body: unknown, options?: RequestOptions): Promise<T> => {
    const res = await fetchWithRetry(`${ENV.API_URL}${endpoint}`, {
      method: 'POST',
      headers: getHeaders(),
      body: JSON.stringify(body),
      signal: options?.signal,
    });
    return handleResponse<T>(res);
  },

  patch: async <T>(endpoint: string, body: unknown, options?: RequestOptions): Promise<T> => {
    const res = await fetchWithRetry(`${ENV.API_URL}${endpoint}`, {
      method: 'PATCH',
      headers: getHeaders(),
      body: JSON.stringify(body),
      signal: options?.signal,
    });
    return handleResponse<T>(res);
  },

  delete: async <T>(endpoint: string, options?: RequestOptions): Promise<T> => {
    const res = await fetchWithRetry(`${ENV.API_URL}${endpoint}`, {
      method: 'DELETE',
      headers: getHeaders(),
      signal: options?.signal,
    });
    return handleResponse<T>(res);
  }
};

export { isAbortError };
