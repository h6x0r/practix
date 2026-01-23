# API Client Guide & Architecture

> Complete guide to the frontend API layer: architecture, usage, and best practices.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                      React Components                            │
│  (Pages, Features UI, etc.)                                      │
└──────────────┬──────────────────────────────────────────────────┘
               │
    ┌──────────▼──────────┐          ┌───────────────────────┐
    │  Feature Services   │          │   Shared API Utils    │
    │  /features/*/api/   │◄─────────│   /lib/api/           │
    └──────────┬──────────┘          └───────────────────────┘
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

### Layer Responsibilities

| Layer | Location | Responsibility |
|-------|----------|----------------|
| **Components** | `/src/features/*/ui/` | UI rendering, user interaction |
| **Feature Services** | `/src/features/*/api/` | Business logic, data transformation |
| **Shared Utils** | `/src/lib/api/` | HTTP client, error handling, interceptors |
| **Backend** | NestJS | REST API, authentication, data persistence |

---

## Quick Reference

### Basic Imports

```typescript
// Shared API utilities
import { api, ApiError, setupInterceptors } from '@/lib/api';

// Generic API client factory
import { createApiClient } from '@/lib/api';

// Feature-specific services
import { authService } from '@/features/auth/api/authService';
import { courseService } from '@/features/courses/api/courseService';
import { taskService } from '@/features/tasks/api/taskService';
import { askAiTutor } from '@/features/ai/api/geminiService';
import { playgroundService } from '@/features/playground/api/playgroundService';
```

---

## Using the Base API Client

### GET Request
```typescript
import { api } from '@/lib/api';

interface User {
  id: string;
  name: string;
  email: string;
}

const user = await api.get<User>('/users/123');
```

### POST Request
```typescript
const newUser = await api.post<User>('/users', {
  name: 'John Doe',
  email: 'john@example.com'
});
```

### PATCH Request
```typescript
const updated = await api.patch<User>('/users/123', {
  name: 'Jane Doe'
});
```

### DELETE Request
```typescript
await api.delete('/users/123');
```

---

## Error Handling

### Using ApiError
```typescript
import { api, ApiError } from '@/lib/api';

try {
  const data = await api.get('/users/me');
} catch (error) {
  if (error instanceof ApiError) {
    console.error('API Error:', error.message);
    console.error('Status:', error.status);
    console.error('Data:', error.data);

    if (error.status === 401) {
      // Handle unauthorized
    } else if (error.status === 404) {
      // Handle not found
    }
  }
}
```

### Setting Up Interceptors

```typescript
// In App.tsx
import { setupInterceptors } from '@/lib/api';
import { authService } from '@/features/auth/api/authService';

setupInterceptors(() => {
  // This runs on any 401 response
  authService.logout();
  setUser(null);
});
```

---

## Generic API Client Factory

The `createApiClient` function creates a typed CRUD client for any resource.

### Basic Usage
```typescript
import { createApiClient } from '@/lib/api';

interface Product {
  id: string;
  name: string;
  price: number;
}

const productClient = createApiClient<Product>('/products');

// GET /products
const allProducts = await productClient.getAll();

// GET /products/123
const product = await productClient.get('123');

// POST /products
const newProduct = await productClient.create({
  name: 'New Product',
  price: 99.99
});

// PATCH /products/123
const updated = await productClient.update('123', {
  price: 89.99
});

// DELETE /products/123
await productClient.delete('123');
```

### Extending with Custom Methods
```typescript
// src/features/products/api/productService.ts
import { createApiClient, api } from '@/lib/api';
import { Product } from '../model/types';

const baseClient = createApiClient<Product>('/products');

export const productService = {
  ...baseClient,

  // Add custom methods
  getFeatured: async () => api.get<Product[]>('/products/featured'),
  searchByName: async (query: string) =>
    api.get<Product[]>(`/products/search?q=${encodeURIComponent(query)}`)
};
```

---

## Feature Service Examples

### Auth Service
```typescript
import { authService } from '@/features/auth/api/authService';

// Login
const { token, user } = await authService.login({
  email: 'user@example.com',
  password: 'password123'
});

// Register
const { token, user } = await authService.register({
  name: 'John Doe',
  email: 'john@example.com',
  password: 'password123'
});

// Get current user
const currentUser = await authService.getMe();

// Logout
authService.logout();
```

### Course Service
```typescript
import { courseService } from '@/features/courses/api/courseService';

const courses = await courseService.getAllCourses();
const course = await courseService.getCourseById('course-123');
const modules = await courseService.getCourseStructure('course-123');
```

### Task Service
```typescript
import { taskService } from '@/features/tasks/api/taskService';

const task = await taskService.fetchTask('two-sum');
const recentTasks = await taskService.getRecentTasks();
const submission = await taskService.submitCode('console.log("Hello")', 'task-123');
```

### AI Tutor Service
```typescript
import { askAiTutor } from '@/features/ai/api/geminiService';

const answer = await askAiTutor(
  'Two Sum Problem',
  'function twoSum(nums, target) { ... }',
  'How can I optimize this solution?',
  'javascript',
  'en'
);
```

### Playground Service
```typescript
import { playgroundService } from '@/features/playground/api/playgroundService';

const result = await playgroundService.runCode(
  'console.log("Hello World")',
  'javascript',
  '' // optional stdin
);

const { languages, default: defaultLang } = await playgroundService.getLanguages();
```

---

## Best Practices

### 1. Always Use Feature Services
```typescript
// Bad - direct API call in component
const user = await api.get('/users/me');

// Good - use feature service
const user = await authService.getMe();
```

### 2. Type Everything
```typescript
interface MyData {
  id: string;
  value: number;
}

const data = await api.get<MyData>('/endpoint');
// `data` is now typed as MyData
```

### 3. Handle Errors Properly
```typescript
try {
  const data = await api.get('/data');
} catch (error) {
  if (error instanceof ApiError) {
    // Handle API errors
  } else {
    // Handle network errors
  }
}
```

### 4. Use Generic Client for CRUD
```typescript
// Instead of:
export const userService = {
  getAll: () => api.get('/users'),
  get: (id: string) => api.get(`/users/${id}`),
  create: (data: User) => api.post('/users', data),
};

// Just do:
export const userService = createApiClient<User>('/users');
```

---

## Directory Structure

```
src/
├── lib/
│   └── api/
│       ├── client.ts          # Core API client
│       ├── createApiClient.ts # Generic CRUD factory
│       └── index.ts           # Public exports
└── features/
    ├── auth/
    │   └── api/
    │       └── authService.ts
    ├── courses/
    │   └── api/
    │       └── courseService.ts
    ├── tasks/
    │   └── api/
    │       └── taskService.ts
    └── ...
```

---

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

---

*Last updated: January 2026*
