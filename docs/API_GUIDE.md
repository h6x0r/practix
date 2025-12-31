# API Client Usage Guide

## Quick Reference

### Basic Imports

```typescript
// For shared API utilities (HTTP client, error handling, interceptors)
import { api, ApiError, setupInterceptors } from '@/lib/api';

// For generic API client factory
import { createApiClient } from '@/lib/api';

// For feature-specific services
import { authService } from '@/features/auth/api/authService';
import { courseService } from '@/features/courses/api/courseService';
import { taskService } from '@/features/tasks/api/taskService';
import { askAiTutor } from '@/features/ai/api/geminiService';
import { playgroundService } from '@/features/playground/api/playgroundService';
```

## Using the Base API Client

### Simple GET Request
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
import { api } from '@/lib/api';

const newUser = await api.post<User>('/users', {
  name: 'John Doe',
  email: 'john@example.com'
});
```

### PATCH Request
```typescript
import { api } from '@/lib/api';

const updated = await api.patch<User>('/users/123', {
  name: 'Jane Doe'
});
```

### DELETE Request
```typescript
import { api } from '@/lib/api';

await api.delete('/users/123');
```

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

## Setting Up Interceptors

### Global 401 Handler
```typescript
import { setupInterceptors } from '@/lib/api';
import { authService } from '@/features/auth/api/authService';

// In your App.tsx or main entry point
setupInterceptors(() => {
  console.warn("Session expired");
  authService.logout();
  // Redirect to login, etc.
});
```

## Using the Generic API Client Factory

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

### Creating a Feature Service with Generic Client
```typescript
// src/features/products/api/productService.ts
import { createApiClient } from '@/lib/api';
import { Product } from '../model/types';

const productClient = createApiClient<Product>('/products');

export const productService = {
  ...productClient,
  
  // Add custom methods specific to products
  getFeatured: async () => {
    return api.get<Product[]>('/products/featured');
  },
  
  searchByName: async (query: string) => {
    return api.get<Product[]>(`/products/search?q=${encodeURIComponent(query)}`);
  }
};
```

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

// Upgrade to premium
const upgradedUser = await authService.upgrade();
```

### Course Service
```typescript
import { courseService } from '@/features/courses/api/courseService';

// Get all courses
const courses = await courseService.getAllCourses();

// Get specific course
const course = await courseService.getCourseById('course-123');

// Get course structure with modules
const modules = await courseService.getCourseStructure('course-123');
```

### Task Service
```typescript
import { taskService } from '@/features/tasks/api/taskService';

// Fetch a task by slug
const task = await taskService.fetchTask('two-sum');

// Get recent tasks
const recentTasks = await taskService.getRecentTasks();

// Submit code for a task
const submission = await taskService.submitCode(
  'console.log("Hello")',
  'task-123'
);

// Check if task is completed (uses localStorage)
const isCompleted = taskService.isResourceCompleted('task-123', 'task');

// Mark task as completed (saves to localStorage)
taskService.markTaskAsCompleted('task-123');
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

// Run code
const result = await playgroundService.runCode(
  'console.log("Hello World")',
  'javascript',
  '' // optional stdin
);

// Get available languages
const { languages, default: defaultLang } = await playgroundService.getLanguages();

// Get judge system status
const status = await playgroundService.getJudgeStatus();
```

## Best Practices

### 1. Use Type Safety
Always provide TypeScript types for better IDE support and type checking:
```typescript
interface MyData {
  id: string;
  value: number;
}

const data = await api.get<MyData>('/endpoint');
// `data` is now typed as MyData
```

### 2. Handle Errors Properly
Always wrap API calls in try-catch blocks:
```typescript
try {
  const data = await api.get('/data');
  // Handle success
} catch (error) {
  if (error instanceof ApiError) {
    // Handle API errors
  } else {
    // Handle network errors
  }
}
```

### 3. Use Feature Services
Don't use the raw API client in components. Always create/use feature services:
```typescript
// Bad - directly in component
const user = await api.get('/users/me');

// Good - use feature service
const user = await authService.getMe();
```

### 4. Keep Services in Feature Directories
New services should go in `/src/features/[feature-name]/api/`:
```
src/
  features/
    products/
      api/
        productService.ts
      model/
        types.ts
      ui/
        ProductList.tsx
```

### 5. Reuse Generic Client When Possible
For simple CRUD operations, use `createApiClient`:
```typescript
// Instead of writing:
export const userService = {
  getAll: () => api.get('/users'),
  get: (id: string) => api.get(`/users/${id}`),
  create: (data: User) => api.post('/users', data),
  // ... etc
};

// Just do:
export const userService = createApiClient<User>('/users');
```

## Directory Structure

```
src/
  lib/
    api/
      client.ts          # Core API client
      createApiClient.ts # Generic CRUD factory
      index.ts           # Public exports
  features/
    auth/
      api/
        authService.ts
    courses/
      api/
        courseService.ts
    tasks/
      api/
        taskService.ts
    ai/
      api/
        geminiService.ts
    playground/
      api/
        playgroundService.ts
```

---

*For migration details, see [API_SERVICES_MIGRATION.md](./API_SERVICES_MIGRATION.md)*
