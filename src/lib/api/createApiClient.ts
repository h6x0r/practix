import { api } from './client';

/**
 * Creates a generic API client for CRUD operations
 * @param basePath - The base path for the resource (e.g., '/users', '/courses')
 * @returns An object with common CRUD methods
 */
export function createApiClient<T>(basePath: string) {
  return {
    get: (id: string) => api.get<T>(`${basePath}/${id}`),
    getAll: () => api.get<T[]>(basePath),
    create: (data: Partial<T>) => api.post<T>(basePath, data),
    update: (id: string, data: Partial<T>) => api.patch<T>(`${basePath}/${id}`, data),
    delete: (id: string) => api.delete(`${basePath}/${id}`),
  };
}
