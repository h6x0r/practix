# API Architecture Overview

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                      React Components                            │
│  (Pages, Features UI, etc.)                                      │
└──────────────┬──────────────────────────────────┬────────────────┘
               │                                  │
               │                                  │
    ┌──────────▼──────────┐          ┌───────────▼──────────┐
    │  Feature Services   │          │   Shared API Utils   │
    │  /features/*/api/   │          │   /lib/api/          │
    └──────────┬──────────┘          └───────────┬──────────┘
               │                                  │
               │  imports                         │
               └──────────────┬───────────────────┘
                              │
                   ┌──────────▼──────────┐
                   │  API Client (api)   │
                   │  - GET, POST, etc.  │
                   │  - Error Handling   │
                   │  - Interceptors     │
                   └──────────┬──────────┘
                              │
                   ┌──────────▼──────────┐
                   │   Backend API       │
                   │   (NestJS)          │
                   └─────────────────────┘
```

## Layer Responsibilities

### Layer 1: React Components
- **Location:** `/src/pages/`, `/src/features/*/ui/`
- **Responsibility:** UI rendering, user interaction
- **Imports from:** Feature Services only (never direct API calls)
- **Example:**
  ```typescript
  // In a React component
  import { courseService } from '@/features/courses/api/courseService';
  
  const courses = await courseService.getAllCourses();
  ```

### Layer 2A: Feature Services
- **Location:** `/src/features/{feature-name}/api/`
- **Responsibility:** Feature-specific business logic, data transformation
- **Imports from:** Shared API Utils
- **Features:**
  - `auth` - Authentication & user management
  - `courses` - Course data & structure
  - `tasks` - Task fetching & submissions
  - `ai` - AI tutor interactions
  - `playground` - Code execution
  - `analytics` - User analytics
  - `payments` - Payment processing
  - `roadmap` - Learning roadmaps
  - `config` - App configuration

### Layer 2B: Shared API Utils
- **Location:** `/src/lib/api/`
- **Responsibility:** Core HTTP client, error handling, interceptors
- **Provides:**
  - `api` - Base HTTP client
  - `ApiError` - Structured error class
  - `setupInterceptors` - Global error handlers
  - `createApiClient` - Generic CRUD factory

### Layer 3: API Client
- **Responsibility:** HTTP communication with backend
- **Features:**
  - Automatic JSON serialization
  - Auth token injection (via headers)
  - 401 handling with interceptors
  - 204 No Content handling
  - Error response parsing

### Layer 4: Backend API
- **Technology:** NestJS
- **Base URL:** Configured via `ENV.API_URL`
- **Authentication:** JWT tokens in headers

## Data Flow Examples

### Example 1: User Login
```
┌─────────────┐     login()      ┌──────────────────┐
│  Auth Page  │ ───────────────> │  authService     │
│  Component  │                  │  (feature API)   │
└─────────────┘                  └────────┬─────────┘
                                          │ api.post()
                                 ┌────────▼─────────┐
                                 │  API Client      │
                                 │  POST /auth/login│
                                 └────────┬─────────┘
                                          │
                                 ┌────────▼─────────┐
                                 │  Backend API     │
                                 │  Returns token   │
                                 └──────────────────┘
```

### Example 2: Fetching Courses
```
┌─────────────┐  getAllCourses() ┌──────────────────┐
│  Courses    │ ───────────────> │  courseService   │
│  Component  │                  │  (feature API)   │
└─────────────┘                  └────────┬─────────┘
                                          │ api.get()
                                 ┌────────▼─────────┐
                                 │  API Client      │
                                 │  GET /courses    │
                                 └────────┬─────────┘
                                          │
                                 ┌────────▼─────────┐
                                 │  Backend API     │
                                 │  Returns courses │
                                 └──────────────────┘
```

### Example 3: Error Handling Flow
```
┌─────────────┐                  ┌──────────────────┐
│  Component  │ ──── API call ──>│  Feature Service │
└─────┬───────┘                  └────────┬─────────┘
      │                                   │
      │                          ┌────────▼─────────┐
      │                          │  API Client      │
      │                          │  Makes request   │
      │                          └────────┬─────────┘
      │                                   │
      │                          ┌────────▼─────────┐
      │                          │  Backend returns │
      │                          │  401 Unauthorized│
      │                          └────────┬─────────┘
      │                                   │
      │                          ┌────────▼─────────┐
      │                          │  Interceptor     │
      │                          │  Calls callback  │
      │                          └────────┬─────────┘
      │                                   │
      │                          ┌────────▼─────────┐
      │                          │  Logout user     │
      │ <──── ApiError thrown ── │  Clear session   │
      │                          └──────────────────┘
      │
┌─────▼───────┐
│  Component  │
│  catches err│
│  shows toast│
└─────────────┘
```

## Service Organization Pattern

### Standard Feature API Service Structure
```typescript
// /src/features/{feature}/api/{feature}Service.ts

import { api } from '@/lib/api';
import { FeatureType } from '../model/types';

export const featureService = {
  getAll: async (): Promise<FeatureType[]> => {
    return api.get<FeatureType[]>('/endpoint');
  },
  
  getById: async (id: string): Promise<FeatureType> => {
    return api.get<FeatureType>(`/endpoint/${id}`);
  },
  
  create: async (data: Partial<FeatureType>): Promise<FeatureType> => {
    return api.post<FeatureType>('/endpoint', data);
  },
  
  update: async (id: string, data: Partial<FeatureType>): Promise<FeatureType> => {
    return api.patch<FeatureType>(`/endpoint/${id}`, data);
  },
  
  delete: async (id: string): Promise<void> => {
    return api.delete(`/endpoint/${id}`);
  }
};
```

### Using Generic Client Factory (Recommended)
```typescript
// /src/features/{feature}/api/{feature}Service.ts

import { createApiClient } from '@/lib/api';
import { FeatureType } from '../model/types';

const baseClient = createApiClient<FeatureType>('/endpoint');

export const featureService = {
  ...baseClient,
  
  // Add custom methods only
  getSpecialData: async () => {
    return api.get<FeatureType[]>('/endpoint/special');
  }
};
```

## Configuration

### Environment Setup
```typescript
// /src/config/env.ts
export const ENV = {
  API_URL: import.meta.env.VITE_API_URL || 'http://localhost:3001'
};

export const getHeaders = () => {
  const token = localStorage.getItem('token');
  return {
    'Content-Type': 'application/json',
    ...(token && { Authorization: `Bearer ${token}` })
  };
};
```

### Interceptor Setup
```typescript
// In App.tsx
import { setupInterceptors } from '@/lib/api';
import { authService } from '@/features/auth/api/authService';

setupInterceptors(() => {
  // This runs on any 401 response
  authService.logout();
  setUser(null);
  // Optionally redirect to login
});
```

## Best Practices Summary

1. **Component → Feature Service** (Always use feature services, never raw API)
2. **Feature Service → API Client** (Use shared client for HTTP calls)
3. **One feature, one service** (Keep services focused)
4. **Type everything** (Use TypeScript generics)
5. **Handle errors** (Always try-catch API calls)
6. **Use factories** (Reduce boilerplate with `createApiClient`)

## Migration from Old Pattern

### Before (Anti-pattern)
```typescript
// Component directly using API client
import { api } from '../services/api';

const MyComponent = () => {
  const data = await api.get('/users');
  // ❌ Bad: Business logic in component
};
```

### After (Correct pattern)
```typescript
// Component using feature service
import { userService } from '@/features/users/api/userService';

const MyComponent = () => {
  const data = await userService.getAll();
  // ✅ Good: Clean separation of concerns
};

// Feature service
import { api } from '@/lib/api';
export const userService = {
  getAll: () => api.get('/users')
};
```

---

*This architecture ensures clean separation of concerns, testability, and scalability.*
