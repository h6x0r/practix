import { Task } from '../../../../../../../types';

export const task: Task = {
	slug: 'sec-misconfig-audit',
	title: 'Security Misconfiguration: Hardening Your Application',
	difficulty: 'medium',
	tags: ['security', 'owasp', 'configuration', 'hardening', 'typescript'],
	estimatedTime: '35m',
	isPremium: false,
	youtubeUrl: '',
	description: `Learn to identify and fix security misconfigurations - a pervasive vulnerability in OWASP Top 10.

**What is Security Misconfiguration?**

Security misconfiguration is the most common vulnerability on the OWASP list. It can happen at any level: network, platform, web server, application, database, or framework.

**Common Misconfigurations:**

1. **Default Credentials** - admin/admin still used in production
2. **Unnecessary Features** - Debug mode, sample apps, unused endpoints
3. **Verbose Errors** - Stack traces exposed to users
4. **Missing Security Headers** - No CSP, HSTS, X-Frame-Options
5. **Outdated Software** - Unpatched vulnerabilities
6. **Open Cloud Storage** - Public S3 buckets, Firebase rules

**Your Task:**

Implement a \`SecurityConfigAuditor\` class that:

1. Audits HTTP security headers
2. Checks for debug mode/verbose errors
3. Validates environment configuration
4. Detects common misconfigurations

**Example Usage:**

\`\`\`typescript
const auditor = new SecurityConfigAuditor();

// Audit HTTP headers
auditor.auditHeaders({
  'X-Frame-Options': 'DENY',
  'Content-Security-Policy': "default-src 'self'",
});
// { secure: true, missing: [], recommendations: [] }

// Check environment
auditor.auditEnvironment({
  debug: false,
  environment: 'production',
  stackTraces: false,
});
// { secure: true }
\`\`\``,
	initialCode: `interface HeaderAuditResult {
  secure: boolean;
  missing: string[];
  weak: string[];
  recommendations: string[];
}

interface EnvAuditResult {
  secure: boolean;
  issues: string[];
}

interface ServerConfig {
  debug?: boolean;
  environment?: string;
  stackTraces?: boolean;
  adminPath?: string;
  defaultCredentials?: boolean;
  directoryListing?: boolean;
  httpsOnly?: boolean;
}

class SecurityConfigAuditor {
  // Required security headers
  private readonly REQUIRED_HEADERS = [
    'X-Frame-Options',
    'X-Content-Type-Options',
    'X-XSS-Protection',
    'Strict-Transport-Security',
    'Content-Security-Policy',
  ];

  auditHeaders(headers: Record<string, string>): HeaderAuditResult {
    // TODO: Check for required security headers
    // Identify missing and weak headers
    // Return recommendations
    return { secure: false, missing: [], weak: [], recommendations: [] };
  }

  auditEnvironment(config: ServerConfig): EnvAuditResult {
    // TODO: Check environment configuration
    // Debug mode, stack traces, admin paths, etc.
    return { secure: false, issues: [] };
  }

  validateCsp(cspValue: string): { valid: boolean; issues: string[] } {
    // TODO: Validate Content-Security-Policy
    // Check for unsafe-inline, unsafe-eval, wildcards
    return { valid: false, issues: [] };
  }

  checkDefaultCredentials(username: string, password: string): boolean {
    // TODO: Check if credentials are common defaults
    return false;
  }

  generateSecureHeaders(): Record<string, string> {
    // TODO: Generate recommended security headers
    return {};
  }

  auditCorsConfig(config: {
    origins: string[];
    credentials: boolean;
    methods: string[];
  }): { secure: boolean; issues: string[] } {
    // TODO: Audit CORS configuration
    return { secure: false, issues: [] };
  }
}

export { SecurityConfigAuditor, HeaderAuditResult, EnvAuditResult, ServerConfig };`,
	solutionCode: `interface HeaderAuditResult {
  secure: boolean;
  missing: string[];
  weak: string[];
  recommendations: string[];
}

interface EnvAuditResult {
  secure: boolean;
  issues: string[];
}

interface ServerConfig {
  debug?: boolean;
  environment?: string;
  stackTraces?: boolean;
  adminPath?: string;
  defaultCredentials?: boolean;
  directoryListing?: boolean;
  httpsOnly?: boolean;
}

class SecurityConfigAuditor {
  // Required security headers
  private readonly REQUIRED_HEADERS = [
    'X-Frame-Options',
    'X-Content-Type-Options',
    'X-XSS-Protection',
    'Strict-Transport-Security',
    'Content-Security-Policy',
  ];

  // Common default credentials
  private readonly DEFAULT_CREDENTIALS = [
    ['admin', 'admin'],
    ['admin', 'password'],
    ['root', 'root'],
    ['user', 'user'],
    ['test', 'test'],
    ['admin', '123456'],
    ['administrator', 'administrator'],
  ];

  // Audit HTTP security headers
  auditHeaders(headers: Record<string, string>): HeaderAuditResult {
    const normalizedHeaders: Record<string, string> = {};
    for (const [key, value] of Object.entries(headers)) {
      normalizedHeaders[key.toLowerCase()] = value;
    }

    const missing: string[] = [];
    const weak: string[] = [];
    const recommendations: string[] = [];

    for (const required of this.REQUIRED_HEADERS) {
      const value = normalizedHeaders[required.toLowerCase()];

      if (!value) {
        missing.push(required);
        continue;
      }

      // Check for weak configurations
      if (required === 'X-Frame-Options' && value.toUpperCase() === 'ALLOWALL') {
        weak.push(\`\${required}: ALLOWALL is not secure\`);
      }

      if (required === 'Strict-Transport-Security') {
        const maxAge = parseInt(value.match(/max-age=(\\d+)/i)?.[1] || '0');
        if (maxAge < 31536000) {
          weak.push('HSTS max-age should be at least 1 year (31536000)');
        }
        if (!value.includes('includeSubDomains')) {
          recommendations.push('Add includeSubDomains to HSTS');
        }
      }
    }

    // Additional recommendations
    if (!normalizedHeaders['referrer-policy']) {
      recommendations.push('Add Referrer-Policy header');
    }
    if (!normalizedHeaders['permissions-policy']) {
      recommendations.push('Add Permissions-Policy header');
    }

    return {
      secure: missing.length === 0 && weak.length === 0,
      missing,
      weak,
      recommendations,
    };
  }

  // Audit environment configuration
  auditEnvironment(config: ServerConfig): EnvAuditResult {
    const issues: string[] = [];

    // Debug mode in production
    if (config.debug && config.environment === 'production') {
      issues.push('Debug mode enabled in production');
    }

    // Stack traces exposed
    if (config.stackTraces && config.environment === 'production') {
      issues.push('Stack traces exposed in production');
    }

    // Predictable admin path
    if (config.adminPath && ['admin', '/admin', '/administrator', '/manage'].includes(config.adminPath)) {
      issues.push('Using predictable admin path');
    }

    // Default credentials
    if (config.defaultCredentials) {
      issues.push('Default credentials still in use');
    }

    // Directory listing enabled
    if (config.directoryListing) {
      issues.push('Directory listing enabled');
    }

    // HTTP allowed
    if (config.httpsOnly === false) {
      issues.push('HTTP (non-HTTPS) connections allowed');
    }

    return {
      secure: issues.length === 0,
      issues,
    };
  }

  // Validate Content-Security-Policy
  validateCsp(cspValue: string): { valid: boolean; issues: string[] } {
    const issues: string[] = [];

    // Check for unsafe directives
    if (cspValue.includes("'unsafe-inline'")) {
      issues.push("'unsafe-inline' allows inline scripts - XSS risk");
    }

    if (cspValue.includes("'unsafe-eval'")) {
      issues.push("'unsafe-eval' allows eval() - code injection risk");
    }

    // Check for overly permissive wildcards
    if (cspValue.includes('*') && !cspValue.includes('*.')) {
      issues.push('Wildcard * is too permissive');
    }

    // Check for data: URI
    if (cspValue.includes('data:') && cspValue.includes('script-src')) {
      issues.push("data: in script-src allows XSS");
    }

    // Check for missing default-src
    if (!cspValue.includes('default-src')) {
      issues.push('Missing default-src directive');
    }

    return {
      valid: issues.length === 0,
      issues,
    };
  }

  // Check if credentials are common defaults
  checkDefaultCredentials(username: string, password: string): boolean {
    const lowerUser = username.toLowerCase();
    const lowerPass = password.toLowerCase();

    return this.DEFAULT_CREDENTIALS.some(
      ([defUser, defPass]) => lowerUser === defUser && lowerPass === defPass
    );
  }

  // Generate recommended security headers
  generateSecureHeaders(): Record<string, string> {
    return {
      'X-Frame-Options': 'DENY',
      'X-Content-Type-Options': 'nosniff',
      'X-XSS-Protection': '1; mode=block',
      'Strict-Transport-Security': 'max-age=31536000; includeSubDomains; preload',
      'Content-Security-Policy': "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data: https:; font-src 'self'; connect-src 'self'; frame-ancestors 'none'",
      'Referrer-Policy': 'strict-origin-when-cross-origin',
      'Permissions-Policy': 'geolocation=(), microphone=(), camera=()',
    };
  }

  // Audit CORS configuration
  auditCorsConfig(config: {
    origins: string[];
    credentials: boolean;
    methods: string[];
  }): { secure: boolean; issues: string[] } {
    const issues: string[] = [];

    // Check for wildcard origin with credentials
    if (config.origins.includes('*') && config.credentials) {
      issues.push('Wildcard origin with credentials is not allowed by browsers and is insecure');
    }

    // Check for wildcard origin
    if (config.origins.includes('*')) {
      issues.push('Wildcard origin allows any domain');
    }

    // Check for null origin
    if (config.origins.includes('null')) {
      issues.push('null origin should not be allowed');
    }

    // Check for dangerous methods
    if (config.methods.includes('*')) {
      issues.push('Allowing all HTTP methods is overly permissive');
    }

    return {
      secure: issues.length === 0,
      issues,
    };
  }
}

export { SecurityConfigAuditor, HeaderAuditResult, EnvAuditResult, ServerConfig };`,
	hint1: `For auditHeaders, normalize header names to lowercase, then check if each REQUIRED_HEADER exists. Also validate their values for weak configurations like HSTS with short max-age.`,
	hint2: `For auditEnvironment, check each config property for insecure values in production: debug enabled, stack traces exposed, predictable admin paths, directory listing, etc.`,
	testCode: `import { SecurityConfigAuditor } from './solution';

// Test1: Missing security headers detected
test('Test1', () => {
  const auditor = new SecurityConfigAuditor();
  const result = auditor.auditHeaders({});
  expect(result.secure).toBe(false);
  expect(result.missing.length).toBeGreaterThan(0);
});

// Test2: All required headers present is secure
test('Test2', () => {
  const auditor = new SecurityConfigAuditor();
  const result = auditor.auditHeaders({
    'X-Frame-Options': 'DENY',
    'X-Content-Type-Options': 'nosniff',
    'X-XSS-Protection': '1; mode=block',
    'Strict-Transport-Security': 'max-age=31536000',
    'Content-Security-Policy': "default-src 'self'",
  });
  expect(result.secure).toBe(true);
});

// Test3: Debug mode in production detected
test('Test3', () => {
  const auditor = new SecurityConfigAuditor();
  const result = auditor.auditEnvironment({
    debug: true,
    environment: 'production',
  });
  expect(result.secure).toBe(false);
  expect(result.issues.some(i => i.includes('Debug'))).toBe(true);
});

// Test4: Secure environment config passes
test('Test4', () => {
  const auditor = new SecurityConfigAuditor();
  const result = auditor.auditEnvironment({
    debug: false,
    environment: 'production',
    stackTraces: false,
    httpsOnly: true,
  });
  expect(result.secure).toBe(true);
});

// Test5: CSP with unsafe-inline flagged
test('Test5', () => {
  const auditor = new SecurityConfigAuditor();
  const result = auditor.validateCsp("default-src 'self'; script-src 'unsafe-inline'");
  expect(result.valid).toBe(false);
  expect(result.issues.some(i => i.includes('unsafe-inline'))).toBe(true);
});

// Test6: checkDefaultCredentials detects admin/admin
test('Test6', () => {
  const auditor = new SecurityConfigAuditor();
  expect(auditor.checkDefaultCredentials('admin', 'admin')).toBe(true);
});

// Test7: checkDefaultCredentials passes for strong credentials
test('Test7', () => {
  const auditor = new SecurityConfigAuditor();
  expect(auditor.checkDefaultCredentials('johndoe', 'X#9kLm!pQ2$')).toBe(false);
});

// Test8: generateSecureHeaders returns required headers
test('Test8', () => {
  const auditor = new SecurityConfigAuditor();
  const headers = auditor.generateSecureHeaders();
  expect(headers['X-Frame-Options']).toBeDefined();
  expect(headers['Strict-Transport-Security']).toBeDefined();
});

// Test9: CORS wildcard with credentials flagged
test('Test9', () => {
  const auditor = new SecurityConfigAuditor();
  const result = auditor.auditCorsConfig({
    origins: ['*'],
    credentials: true,
    methods: ['GET', 'POST'],
  });
  expect(result.secure).toBe(false);
});

// Test10: Stack traces in production detected
test('Test10', () => {
  const auditor = new SecurityConfigAuditor();
  const result = auditor.auditEnvironment({
    stackTraces: true,
    environment: 'production',
  });
  expect(result.issues.some(i => i.includes('Stack traces'))).toBe(true);
});`,
	whyItMatters: `Security misconfiguration is the most commonly seen vulnerability and has caused major breaches.

**Real-World Misconfigurations:**

**1. Capital One S3 Bucket (2019)**
\`\`\`
Impact: 100 million customers affected
Cause: Misconfigured WAF + SSRF + overly permissive IAM
Data: SSNs, bank accounts, credit applications
Fine: $80 million
\`\`\`

**2. Facebook API Misconfiguration (2019)**
\`\`\`
Impact: 540 million records exposed
Cause: Third-party app databases publicly accessible
Data: Facebook IDs, comments, likes, account names
Platform: Elasticsearch on AWS without authentication
\`\`\`

**3. Microsoft Power Apps (2021)**
\`\`\`
Impact: 38 million records
Cause: Default "Table permissions" set to public
Data: COVID contact tracing, employee info
Affected: 47 organizations including government
\`\`\`

**Security Headers Cheat Sheet:**

| Header | Purpose | Recommended Value |
|--------|---------|-------------------|
| X-Frame-Options | Clickjacking | DENY or SAMEORIGIN |
| X-Content-Type-Options | MIME sniffing | nosniff |
| X-XSS-Protection | XSS filter | 1; mode=block |
| HSTS | Force HTTPS | max-age=31536000; includeSubDomains |
| CSP | Script control | default-src 'self' |
| Referrer-Policy | Info leakage | strict-origin-when-cross-origin |

**Common Hardening Steps:**

\`\`\`typescript
// Express.js security middleware
import helmet from 'helmet';

app.use(helmet({
  contentSecurityPolicy: {
    directives: {
      defaultSrc: ["'self'"],
      scriptSrc: ["'self'"],
      styleSrc: ["'self'", "'unsafe-inline'"],
      imgSrc: ["'self'", "data:", "https:"],
      connectSrc: ["'self'"],
      fontSrc: ["'self'"],
      frameAncestors: ["'none'"],
    },
  },
  hsts: {
    maxAge: 31536000,
    includeSubDomains: true,
    preload: true,
  },
}));

// Disable X-Powered-By header
app.disable('x-powered-by');

// Production error handling
if (process.env.NODE_ENV === 'production') {
  app.use((err, req, res, next) => {
    console.error(err);
    res.status(500).json({ error: 'Internal error' });
    // No stack trace!
  });
}
\`\`\``,
	order: 7,
	translations: {
		ru: {
			title: 'Неправильная конфигурация безопасности: Укрепление приложения',
			description: `Научитесь выявлять и исправлять неправильные конфигурации безопасности - повсеместную уязвимость в OWASP Top 10.

**Что такое Security Misconfiguration?**

Неправильная конфигурация - самая распространённая уязвимость в списке OWASP. Она может возникнуть на любом уровне: сеть, платформа, веб-сервер, приложение, БД, фреймворк.

**Ваша задача:**

Реализуйте класс \`SecurityConfigAuditor\`:

1. Аудит HTTP заголовков безопасности
2. Проверка режима отладки/подробных ошибок
3. Валидация конфигурации окружения
4. Обнаружение типичных неправильных настроек`,
			hint1: `Для auditHeaders нормализуйте имена заголовков в lowercase, затем проверьте наличие каждого REQUIRED_HEADER.`,
			hint2: `Для auditEnvironment проверьте каждое свойство config на небезопасные значения в production.`,
			whyItMatters: `Неправильная конфигурация безопасности - самая часто встречающаяся уязвимость.`
		},
		uz: {
			title: 'Xavfsizlik noto\'g\'ri sozlanishi: Ilovangizni mustahkamlash',
			description: `OWASP Top 10 da keng tarqalgan zaiflik - xavfsizlik noto'g'ri sozlanishini aniqlash va tuzatishni o'rganing.

**Security Misconfiguration nima?**

Noto'g'ri sozlash OWASP ro'yxatidagi eng keng tarqalgan zaiflik. U har qanday darajada yuz berishi mumkin: tarmoq, platforma, veb-server, ilova, DB, framework.

**Sizning vazifangiz:**

\`SecurityConfigAuditor\` klassini amalga oshiring:

1. HTTP xavfsizlik sarlavhalarini tekshirish
2. Debug rejimi/batafsil xatolarni tekshirish
3. Muhit konfiguratsiyasini tasdiqlash
4. Umumiy noto'g'ri sozlashlarni aniqlash`,
			hint1: `auditHeaders uchun sarlavha nomlarini lowercase ga normalizatsiya qiling, keyin har bir REQUIRED_HEADER mavjudligini tekshiring.`,
			hint2: `auditEnvironment uchun har bir config xususiyatini production da xavfsiz bo'lmagan qiymatlar uchun tekshiring.`,
			whyItMatters: `Xavfsizlik noto'g'ri sozlanishi eng ko'p uchraydigan zaiflik va katta buzilishlarga sabab bo'lgan.`
		}
	}
};

export default task;
