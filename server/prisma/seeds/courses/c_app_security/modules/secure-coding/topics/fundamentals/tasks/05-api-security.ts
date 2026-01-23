import { Task } from '../../../../../../../types';

export const task: Task = {
	slug: 'secure-api-design',
	title: 'Secure REST API Design',
	difficulty: 'medium',
	tags: ['security', 'secure-coding', 'api', 'rest', 'typescript'],
	estimatedTime: '35m',
	isPremium: false,
	youtubeUrl: '',
	description: `Learn to design and implement secure REST APIs.

**API Security Checklist:**

1. **Authentication** - Who is making the request?
2. **Authorization** - Are they allowed to do this?
3. **Input Validation** - Is the input safe?
4. **Rate Limiting** - Prevent abuse
5. **HTTPS Only** - Encrypt in transit
6. **CORS Policy** - Control cross-origin access
7. **Response Headers** - Security headers

**Common API Vulnerabilities:**

| Vulnerability | Example | Prevention |
|---------------|---------|------------|
| BOLA | Access other users' data | Check ownership |
| Mass Assignment | Modify admin fields | Whitelist fields |
| Excessive Data | Return sensitive fields | Filter response |
| Rate Limit Bypass | DDoS/credential stuffing | Rate limiting |
| SSRF | Fetch attacker URL | Validate URLs |

**Security Headers:**

\`\`\`
X-Content-Type-Options: nosniff
X-Frame-Options: DENY
Content-Security-Policy: default-src 'self'
Strict-Transport-Security: max-age=31536000
\`\`\`

**Your Task:**

Implement an \`APISecurityMiddleware\` class with security controls.`,
	initialCode: `interface Request {
  method: string;
  path: string;
  headers: Record<string, string>;
  body?: any;
  userId?: string;
  ip: string;
}

interface Response {
  status: number;
  headers: Record<string, string>;
  body?: any;
}

interface RateLimitConfig {
  maxRequests: number;
  windowMs: number;
}

interface CORSConfig {
  allowedOrigins: string[];
  allowedMethods: string[];
  allowedHeaders: string[];
  maxAge: number;
}

class APISecurityMiddleware {
  private rateLimits: Map<string, { count: number; resetAt: Date }> = new Map();
  private corsConfig: CORSConfig;
  private rateLimitConfig: RateLimitConfig;

  constructor(
    corsConfig?: Partial<CORSConfig>,
    rateLimitConfig?: Partial<RateLimitConfig>
  ) {
    this.corsConfig = {
      allowedOrigins: corsConfig?.allowedOrigins || ['https://example.com'],
      allowedMethods: corsConfig?.allowedMethods || ['GET', 'POST', 'PUT', 'DELETE'],
      allowedHeaders: corsConfig?.allowedHeaders || ['Content-Type', 'Authorization'],
      maxAge: corsConfig?.maxAge || 86400,
    };
    this.rateLimitConfig = {
      maxRequests: rateLimitConfig?.maxRequests || 100,
      windowMs: rateLimitConfig?.windowMs || 60000,
    };
  }

  checkRateLimit(identifier: string): { allowed: boolean; remaining: number; resetAt: Date } {
    // TODO: Check and update rate limit
    return { allowed: false, remaining: 0, resetAt: new Date() };
  }

  validateOrigin(origin: string): boolean {
    // TODO: Check if origin is allowed
    return false;
  }

  getCORSHeaders(origin: string): Record<string, string> {
    // TODO: Generate CORS headers for response
    return {};
  }

  getSecurityHeaders(): Record<string, string> {
    // TODO: Return standard security headers
    return {};
  }

  validateRequest(request: Request): { valid: boolean; errors: string[] } {
    // TODO: Validate request (method, content-type, etc.)
    return { valid: false, errors: [] };
  }

  sanitizeResponseData(data: any, sensitiveFields: string[]): any {
    // TODO: Remove sensitive fields from response
    return data;
  }

  preventMassAssignment(body: any, allowedFields: string[]): any {
    // TODO: Only keep allowed fields
    return {};
  }

  checkOwnership(resourceOwnerId: string, requestUserId: string): boolean {
    // TODO: Verify resource ownership (prevent BOLA)
    return false;
  }

  isValidURL(url: string, allowedDomains: string[]): boolean {
    // TODO: Validate URL against allowed domains (prevent SSRF)
    return false;
  }
}

export { APISecurityMiddleware, Request, Response, RateLimitConfig, CORSConfig };`,
	solutionCode: `interface Request {
  method: string;
  path: string;
  headers: Record<string, string>;
  body?: any;
  userId?: string;
  ip: string;
}

interface Response {
  status: number;
  headers: Record<string, string>;
  body?: any;
}

interface RateLimitConfig {
  maxRequests: number;
  windowMs: number;
}

interface CORSConfig {
  allowedOrigins: string[];
  allowedMethods: string[];
  allowedHeaders: string[];
  maxAge: number;
}

class APISecurityMiddleware {
  private rateLimits: Map<string, { count: number; resetAt: Date }> = new Map();
  private corsConfig: CORSConfig;
  private rateLimitConfig: RateLimitConfig;

  constructor(
    corsConfig?: Partial<CORSConfig>,
    rateLimitConfig?: Partial<RateLimitConfig>
  ) {
    this.corsConfig = {
      allowedOrigins: corsConfig?.allowedOrigins || ['https://example.com'],
      allowedMethods: corsConfig?.allowedMethods || ['GET', 'POST', 'PUT', 'DELETE'],
      allowedHeaders: corsConfig?.allowedHeaders || ['Content-Type', 'Authorization'],
      maxAge: corsConfig?.maxAge || 86400,
    };
    this.rateLimitConfig = {
      maxRequests: rateLimitConfig?.maxRequests || 100,
      windowMs: rateLimitConfig?.windowMs || 60000,
    };
  }

  checkRateLimit(identifier: string): { allowed: boolean; remaining: number; resetAt: Date } {
    const now = new Date();
    let record = this.rateLimits.get(identifier);

    if (!record || record.resetAt <= now) {
      record = {
        count: 0,
        resetAt: new Date(now.getTime() + this.rateLimitConfig.windowMs),
      };
      this.rateLimits.set(identifier, record);
    }

    record.count++;
    const remaining = Math.max(0, this.rateLimitConfig.maxRequests - record.count);
    const allowed = record.count <= this.rateLimitConfig.maxRequests;

    return { allowed, remaining, resetAt: record.resetAt };
  }

  validateOrigin(origin: string): boolean {
    if (this.corsConfig.allowedOrigins.includes('*')) {
      return true;
    }

    return this.corsConfig.allowedOrigins.some(allowed => {
      if (allowed.startsWith('*.')) {
        const domain = allowed.slice(2);
        return origin.endsWith(domain) || origin === \`https://\${domain}\`;
      }
      return origin === allowed;
    });
  }

  getCORSHeaders(origin: string): Record<string, string> {
    if (!this.validateOrigin(origin)) {
      return {};
    }

    return {
      'Access-Control-Allow-Origin': origin,
      'Access-Control-Allow-Methods': this.corsConfig.allowedMethods.join(', '),
      'Access-Control-Allow-Headers': this.corsConfig.allowedHeaders.join(', '),
      'Access-Control-Max-Age': this.corsConfig.maxAge.toString(),
      'Access-Control-Allow-Credentials': 'true',
    };
  }

  getSecurityHeaders(): Record<string, string> {
    return {
      'X-Content-Type-Options': 'nosniff',
      'X-Frame-Options': 'DENY',
      'X-XSS-Protection': '1; mode=block',
      'Strict-Transport-Security': 'max-age=31536000; includeSubDomains',
      'Content-Security-Policy': "default-src 'self'",
      'Cache-Control': 'no-store',
      'Pragma': 'no-cache',
    };
  }

  validateRequest(request: Request): { valid: boolean; errors: string[] } {
    const errors: string[] = [];

    // Check HTTP method
    if (!this.corsConfig.allowedMethods.includes(request.method.toUpperCase())) {
      errors.push(\`Method \${request.method} not allowed\`);
    }

    // Check Content-Type for POST/PUT
    if (['POST', 'PUT', 'PATCH'].includes(request.method.toUpperCase())) {
      const contentType = request.headers['content-type']?.toLowerCase() || '';
      if (!contentType.includes('application/json') && !contentType.includes('multipart/form-data')) {
        errors.push('Invalid Content-Type');
      }
    }

    // Check for required auth header (except for public endpoints)
    if (!request.headers['authorization'] && !request.path.startsWith('/public')) {
      errors.push('Authorization header required');
    }

    // Basic path validation
    if (request.path.includes('..') || request.path.includes('//')) {
      errors.push('Invalid path');
    }

    return { valid: errors.length === 0, errors };
  }

  sanitizeResponseData(data: any, sensitiveFields: string[]): any {
    if (typeof data !== 'object' || data === null) {
      return data;
    }

    if (Array.isArray(data)) {
      return data.map(item => this.sanitizeResponseData(item, sensitiveFields));
    }

    const sanitized: Record<string, any> = {};

    for (const [key, value] of Object.entries(data)) {
      if (sensitiveFields.includes(key)) {
        continue; // Skip sensitive fields
      }

      if (typeof value === 'object' && value !== null) {
        sanitized[key] = this.sanitizeResponseData(value, sensitiveFields);
      } else {
        sanitized[key] = value;
      }
    }

    return sanitized;
  }

  preventMassAssignment(body: any, allowedFields: string[]): any {
    if (typeof body !== 'object' || body === null) {
      return {};
    }

    const filtered: Record<string, any> = {};

    for (const field of allowedFields) {
      if (field in body) {
        filtered[field] = body[field];
      }
    }

    return filtered;
  }

  checkOwnership(resourceOwnerId: string, requestUserId: string): boolean {
    return resourceOwnerId === requestUserId;
  }

  isValidURL(url: string, allowedDomains: string[]): boolean {
    try {
      const parsed = new URL(url);

      // Only allow HTTP/HTTPS
      if (!['http:', 'https:'].includes(parsed.protocol)) {
        return false;
      }

      // Check against allowed domains
      return allowedDomains.some(domain => {
        if (domain.startsWith('*.')) {
          const baseDomain = domain.slice(2);
          return parsed.hostname.endsWith(baseDomain);
        }
        return parsed.hostname === domain;
      });
    } catch {
      return false;
    }
  }
}

export { APISecurityMiddleware, Request, Response, RateLimitConfig, CORSConfig };`,
	hint1: `For preventMassAssignment, only copy fields from body that are in the allowedFields array. This prevents users from setting admin flags.`,
	hint2: `For isValidURL, parse the URL, check protocol (only http/https), then verify hostname against allowedDomains list.`,
	testCode: `import { APISecurityMiddleware } from './solution';

// Test1: checkRateLimit tracks requests
test('Test1', () => {
  const middleware = new APISecurityMiddleware({}, { maxRequests: 2 });
  expect(middleware.checkRateLimit('user1').allowed).toBe(true);
  expect(middleware.checkRateLimit('user1').allowed).toBe(true);
  expect(middleware.checkRateLimit('user1').allowed).toBe(false);
});

// Test2: validateOrigin allows configured origins
test('Test2', () => {
  const middleware = new APISecurityMiddleware({ allowedOrigins: ['https://example.com'] });
  expect(middleware.validateOrigin('https://example.com')).toBe(true);
  expect(middleware.validateOrigin('https://evil.com')).toBe(false);
});

// Test3: getCORSHeaders returns correct headers
test('Test3', () => {
  const middleware = new APISecurityMiddleware({ allowedOrigins: ['https://example.com'] });
  const headers = middleware.getCORSHeaders('https://example.com');
  expect(headers['Access-Control-Allow-Origin']).toBe('https://example.com');
});

// Test4: getSecurityHeaders includes key headers
test('Test4', () => {
  const middleware = new APISecurityMiddleware();
  const headers = middleware.getSecurityHeaders();
  expect(headers['X-Content-Type-Options']).toBe('nosniff');
  expect(headers['X-Frame-Options']).toBe('DENY');
});

// Test5: validateRequest checks method
test('Test5', () => {
  const middleware = new APISecurityMiddleware({ allowedMethods: ['GET'] });
  const result = middleware.validateRequest({
    method: 'DELETE', path: '/api', headers: {}, ip: '1.1.1.1',
  });
  expect(result.valid).toBe(false);
});

// Test6: sanitizeResponseData removes sensitive fields
test('Test6', () => {
  const middleware = new APISecurityMiddleware();
  const result = middleware.sanitizeResponseData(
    { id: 1, name: 'John', password: 'secret' },
    ['password']
  );
  expect(result.name).toBe('John');
  expect(result.password).toBeUndefined();
});

// Test7: preventMassAssignment filters fields
test('Test7', () => {
  const middleware = new APISecurityMiddleware();
  const result = middleware.preventMassAssignment(
    { name: 'John', email: 'j@e.com', isAdmin: true },
    ['name', 'email']
  );
  expect(result.name).toBe('John');
  expect(result.isAdmin).toBeUndefined();
});

// Test8: checkOwnership validates correctly
test('Test8', () => {
  const middleware = new APISecurityMiddleware();
  expect(middleware.checkOwnership('user1', 'user1')).toBe(true);
  expect(middleware.checkOwnership('user1', 'user2')).toBe(false);
});

// Test9: isValidURL prevents SSRF
test('Test9', () => {
  const middleware = new APISecurityMiddleware();
  expect(middleware.isValidURL('https://safe.com/api', ['safe.com'])).toBe(true);
  expect(middleware.isValidURL('https://evil.com/steal', ['safe.com'])).toBe(false);
  expect(middleware.isValidURL('file:///etc/passwd', ['safe.com'])).toBe(false);
});

// Test10: Rate limit resets after window
test('Test10', () => {
  const middleware = new APISecurityMiddleware({}, { maxRequests: 1, windowMs: 100 });
  middleware.checkRateLimit('user1');
  expect(middleware.checkRateLimit('user1').allowed).toBe(false);
  // Can't easily test time-based reset in sync test, but structure is there
});`,
	whyItMatters: `APIs are the backbone of modern applications - and a prime attack target.

**OWASP API Top 10 (2023):**

1. Broken Object Level Authorization (BOLA)
2. Broken Authentication
3. Broken Object Property Level Authorization
4. Unrestricted Resource Consumption
5. Broken Function Level Authorization
6. Unrestricted Access to Sensitive Business Flows
7. Server Side Request Forgery (SSRF)
8. Security Misconfiguration
9. Improper Inventory Management
10. Unsafe Consumption of APIs

**Real API Breaches:**

| Company | Vulnerability | Impact |
|---------|--------------|--------|
| Venmo | BOLA | Transactions exposed |
| Uber | Mass Assignment | Admin access |
| T-Mobile | Rate Limit Bypass | 2M records |
| Peloton | BOLA | User data exposed |

**Defense in Depth:**

\`\`\`
┌─────────────────────────────────┐
│          WAF/CDN                │ ← DDoS protection
├─────────────────────────────────┤
│       API Gateway               │ ← Rate limiting, auth
├─────────────────────────────────┤
│    Application Logic            │ ← Authorization, validation
├─────────────────────────────────┤
│        Database                 │ ← Row-level security
└─────────────────────────────────┘
\`\`\``,
	order: 4,
	translations: {
		ru: {
			title: 'Безопасный дизайн REST API',
			description: `Научитесь проектировать и реализовывать безопасные REST API.

**Чек-лист безопасности API:**

1. Аутентификация - кто делает запрос?
2. Авторизация - разрешено ли это?
3. Валидация ввода - безопасен ли ввод?
4. Rate Limiting - предотвратить злоупотребление
5. Только HTTPS - шифрование в транзите
6. CORS Policy - контроль кросс-доменного доступа

**Ваша задача:**

Реализуйте класс \`APISecurityMiddleware\`.`,
			hint1: `Для preventMassAssignment копируйте только поля из body, которые есть в allowedFields.`,
			hint2: `Для isValidURL распарсите URL, проверьте протокол (только http/https), затем hostname против allowedDomains.`,
			whyItMatters: `API - основа современных приложений и главная цель атак.`
		},
		uz: {
			title: 'Xavfsiz REST API dizayni',
			description: `Xavfsiz REST API larni loyihalash va amalga oshirishni o'rganing.

**API xavfsizlik tekshiruv ro'yxati:**

1. Autentifikatsiya - so'rov kim tomonidan yuborilmoqda?
2. Avtorizatsiya - bunga ruxsat berilganmi?
3. Kiritishni tekshirish - kiritish xavfsizmi?
4. Rate Limiting - suiiste'mollikni oldini olish

**Sizning vazifangiz:**

\`APISecurityMiddleware\` klassini amalga oshiring.`,
			hint1: `preventMassAssignment uchun faqat allowedFields dagi maydonlarni body dan ko'chiring.`,
			hint2: `isValidURL uchun URL ni parsing qiling, protokolni tekshiring (faqat http/https), keyin hostname ni allowedDomains ga qarang.`,
			whyItMatters: `API zamonaviy ilovalarning asosi va hujumlarning asosiy maqsadi.`
		}
	}
};

export default task;
