import { Task } from '../../../../../../../types';

export const task: Task = {
	slug: 'secure-logging',
	title: 'Secure Logging Practices',
	difficulty: 'medium',
	tags: ['security', 'secure-coding', 'logging', 'compliance', 'typescript'],
	estimatedTime: '30m',
	isPremium: false,
	youtubeUrl: '',
	description: `Learn to implement secure logging that aids debugging without creating security risks.

**Logging Security Risks:**

1. **Sensitive Data Exposure** - Passwords, tokens in logs
2. **Log Injection** - Attacker-controlled log entries
3. **PII Violations** - GDPR/HIPAA compliance issues
4. **Log Tampering** - Evidence destruction

**What NOT to Log:**

- Passwords (even hashed!)
- API keys and tokens
- Credit card numbers
- Social Security Numbers
- Session IDs
- Health information (PHI)
- Personal data (PII)

**What TO Log:**

- User ID (not email)
- Timestamp
- Request ID
- Action performed
- Success/failure
- Error codes
- Client IP (with consent)

**Your Task:**

Implement a \`SecureLogger\` class that sanitizes data before logging.`,
	initialCode: `type LogLevel = 'debug' | 'info' | 'warn' | 'error';

interface LogEntry {
  timestamp: Date;
  level: LogLevel;
  message: string;
  context?: Record<string, any>;
  requestId?: string;
  userId?: string;
}

interface LoggerConfig {
  sensitiveFields: string[];
  maxMessageLength: number;
  enableConsole: boolean;
}

class SecureLogger {
  private config: LoggerConfig;
  private logs: LogEntry[] = [];
  private defaultSensitiveFields = [
    'password', 'passwd', 'secret', 'token', 'apiKey', 'api_key',
    'authorization', 'auth', 'creditCard', 'credit_card', 'ccn',
    'ssn', 'socialSecurity', 'dob', 'dateOfBirth',
  ];

  constructor(config?: Partial<LoggerConfig>) {
    this.config = {
      sensitiveFields: config?.sensitiveFields || this.defaultSensitiveFields,
      maxMessageLength: config?.maxMessageLength || 1000,
      enableConsole: config?.enableConsole ?? false,
    };
  }

  log(level: LogLevel, message: string, context?: Record<string, any>): LogEntry {
    // TODO: Create sanitized log entry
    return {} as LogEntry;
  }

  debug(message: string, context?: Record<string, any>): LogEntry {
    return this.log('debug', message, context);
  }

  info(message: string, context?: Record<string, any>): LogEntry {
    return this.log('info', message, context);
  }

  warn(message: string, context?: Record<string, any>): LogEntry {
    return this.log('warn', message, context);
  }

  error(message: string, context?: Record<string, any>): LogEntry {
    return this.log('error', message, context);
  }

  sanitizeContext(context: Record<string, any>): Record<string, any> {
    // TODO: Recursively sanitize context object
    return {};
  }

  sanitizeMessage(message: string): string {
    // TODO: Sanitize log message
    return '';
  }

  isSensitiveField(fieldName: string): boolean {
    // TODO: Check if field name is sensitive
    return false;
  }

  maskValue(value: string): string {
    // TODO: Mask sensitive value (show first/last chars)
    return '';
  }

  preventLogInjection(message: string): string {
    // TODO: Prevent log injection attacks
    return '';
  }

  getLogs(filter?: { level?: LogLevel; since?: Date }): LogEntry[] {
    // TODO: Get filtered logs
    return [];
  }

  auditLog(userId: string, action: string, resource: string, success: boolean): LogEntry {
    // TODO: Create audit log entry
    return {} as LogEntry;
  }
}

export { SecureLogger, LogEntry, LogLevel, LoggerConfig };`,
	solutionCode: `type LogLevel = 'debug' | 'info' | 'warn' | 'error';

interface LogEntry {
  timestamp: Date;
  level: LogLevel;
  message: string;
  context?: Record<string, any>;
  requestId?: string;
  userId?: string;
}

interface LoggerConfig {
  sensitiveFields: string[];
  maxMessageLength: number;
  enableConsole: boolean;
}

class SecureLogger {
  private config: LoggerConfig;
  private logs: LogEntry[] = [];
  private defaultSensitiveFields = [
    'password', 'passwd', 'secret', 'token', 'apiKey', 'api_key',
    'authorization', 'auth', 'creditCard', 'credit_card', 'ccn',
    'ssn', 'socialSecurity', 'dob', 'dateOfBirth',
  ];

  constructor(config?: Partial<LoggerConfig>) {
    this.config = {
      sensitiveFields: config?.sensitiveFields || this.defaultSensitiveFields,
      maxMessageLength: config?.maxMessageLength || 1000,
      enableConsole: config?.enableConsole ?? false,
    };
  }

  log(level: LogLevel, message: string, context?: Record<string, any>): LogEntry {
    const sanitizedMessage = this.sanitizeMessage(this.preventLogInjection(message));
    const sanitizedContext = context ? this.sanitizeContext(context) : undefined;

    const entry: LogEntry = {
      timestamp: new Date(),
      level,
      message: sanitizedMessage.slice(0, this.config.maxMessageLength),
      context: sanitizedContext,
    };

    this.logs.push(entry);

    if (this.config.enableConsole) {
      console.log(JSON.stringify(entry));
    }

    return entry;
  }

  debug(message: string, context?: Record<string, any>): LogEntry {
    return this.log('debug', message, context);
  }

  info(message: string, context?: Record<string, any>): LogEntry {
    return this.log('info', message, context);
  }

  warn(message: string, context?: Record<string, any>): LogEntry {
    return this.log('warn', message, context);
  }

  error(message: string, context?: Record<string, any>): LogEntry {
    return this.log('error', message, context);
  }

  sanitizeContext(context: Record<string, any>): Record<string, any> {
    const sanitized: Record<string, any> = {};

    for (const [key, value] of Object.entries(context)) {
      if (this.isSensitiveField(key)) {
        sanitized[key] = typeof value === 'string' ? this.maskValue(value) : '[REDACTED]';
      } else if (typeof value === 'object' && value !== null) {
        if (Array.isArray(value)) {
          sanitized[key] = value.map(item =>
            typeof item === 'object' ? this.sanitizeContext(item) : item
          );
        } else {
          sanitized[key] = this.sanitizeContext(value);
        }
      } else {
        sanitized[key] = value;
      }
    }

    return sanitized;
  }

  sanitizeMessage(message: string): string {
    let sanitized = message;

    // Remove potential sensitive data patterns
    // Email addresses
    sanitized = sanitized.replace(/[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}/g, '[EMAIL]');

    // Credit card numbers
    sanitized = sanitized.replace(/\\b\\d{4}[\\s-]?\\d{4}[\\s-]?\\d{4}[\\s-]?\\d{4}\\b/g, '[CC]');

    // SSN
    sanitized = sanitized.replace(/\\b\\d{3}-\\d{2}-\\d{4}\\b/g, '[SSN]');

    // Long tokens/keys
    sanitized = sanitized.replace(/[a-zA-Z0-9]{32,}/g, '[TOKEN]');

    // Check for sensitive keywords and mask following values
    for (const field of this.config.sensitiveFields) {
      const pattern = new RegExp(\`(\${field})[\\\\s:=]+[^\\\\s]+\`, 'gi');
      sanitized = sanitized.replace(pattern, '$1=[REDACTED]');
    }

    return sanitized;
  }

  isSensitiveField(fieldName: string): boolean {
    const normalized = fieldName.toLowerCase().replace(/[_-]/g, '');
    return this.config.sensitiveFields.some(field =>
      normalized.includes(field.toLowerCase().replace(/[_-]/g, ''))
    );
  }

  maskValue(value: string): string {
    if (value.length <= 4) {
      return '****';
    }
    const visibleChars = Math.min(2, Math.floor(value.length / 4));
    return value.slice(0, visibleChars) + '****' + value.slice(-visibleChars);
  }

  preventLogInjection(message: string): string {
    return message
      .replace(/\\r/g, '\\\\r')
      .replace(/\\n/g, '\\\\n')
      .replace(/\\t/g, '\\\\t')
      .replace(/[\\x00-\\x1F\\x7F]/g, ''); // Remove control characters
  }

  getLogs(filter?: { level?: LogLevel; since?: Date }): LogEntry[] {
    let filtered = [...this.logs];

    if (filter?.level) {
      filtered = filtered.filter(log => log.level === filter.level);
    }

    if (filter?.since) {
      filtered = filtered.filter(log => log.timestamp >= filter.since!);
    }

    return filtered;
  }

  auditLog(userId: string, action: string, resource: string, success: boolean): LogEntry {
    const message = \`AUDIT: User \${userId} \${action} \${resource} - \${success ? 'SUCCESS' : 'FAILURE'}\`;
    return this.log('info', message, {
      type: 'audit',
      userId,
      action,
      resource,
      success,
    });
  }
}

export { SecureLogger, LogEntry, LogLevel, LoggerConfig };`,
	hint1: `For sanitizeContext, recursively traverse the object. If a key matches sensitiveFields, mask the value. Handle nested objects and arrays.`,
	hint2: `For preventLogInjection, escape newlines and carriage returns (\\n, \\r) to prevent attackers from injecting fake log entries.`,
	testCode: `import { SecureLogger } from './solution';

// Test1: log creates entry with timestamp
test('Test1', () => {
  const logger = new SecureLogger();
  const entry = logger.info('Test message');
  expect(entry.timestamp).toBeTruthy();
  expect(entry.level).toBe('info');
  expect(entry.message).toBe('Test message');
});

// Test2: sanitizeContext masks sensitive fields
test('Test2', () => {
  const logger = new SecureLogger();
  const result = logger.sanitizeContext({ username: 'john', password: 'secret123' });
  expect(result.username).toBe('john');
  expect(result.password).not.toBe('secret123');
  expect(result.password).toContain('****');
});

// Test3: sanitizeMessage removes emails
test('Test3', () => {
  const logger = new SecureLogger();
  const result = logger.sanitizeMessage('User test@example.com logged in');
  expect(result).toContain('[EMAIL]');
  expect(result).not.toContain('@example.com');
});

// Test4: isSensitiveField detects variations
test('Test4', () => {
  const logger = new SecureLogger();
  expect(logger.isSensitiveField('password')).toBe(true);
  expect(logger.isSensitiveField('user_password')).toBe(true);
  expect(logger.isSensitiveField('username')).toBe(false);
});

// Test5: maskValue shows partial value
test('Test5', () => {
  const logger = new SecureLogger();
  const result = logger.maskValue('secrettoken');
  expect(result).toContain('****');
  expect(result.length).toBeLessThan('secrettoken'.length);
});

// Test6: preventLogInjection escapes newlines
test('Test6', () => {
  const logger = new SecureLogger();
  const result = logger.preventLogInjection('Line1\\nFake: ADMIN logged in');
  expect(result).not.toContain('\\n');
  expect(result).toContain('\\\\n');
});

// Test7: getLogs filters by level
test('Test7', () => {
  const logger = new SecureLogger();
  logger.info('Info message');
  logger.error('Error message');
  const errors = logger.getLogs({ level: 'error' });
  expect(errors.length).toBe(1);
  expect(errors[0].level).toBe('error');
});

// Test8: auditLog creates audit entry
test('Test8', () => {
  const logger = new SecureLogger();
  const entry = logger.auditLog('user123', 'DELETE', '/api/data', true);
  expect(entry.message).toContain('AUDIT');
  expect(entry.message).toContain('user123');
  expect(entry.context?.success).toBe(true);
});

// Test9: Nested objects are sanitized
test('Test9', () => {
  const logger = new SecureLogger();
  const result = logger.sanitizeContext({
    user: { name: 'John', apiKey: 'secret' },
  });
  expect(result.user.name).toBe('John');
  expect(result.user.apiKey).toContain('****');
});

// Test10: sanitizeMessage handles credit cards
test('Test10', () => {
  const logger = new SecureLogger();
  const result = logger.sanitizeMessage('Card: 4111-1111-1111-1111');
  expect(result).toContain('[CC]');
  expect(result).not.toContain('4111');
});`,
	whyItMatters: `Logs are often overlooked attack vectors and compliance nightmares.

**Log Security Incidents:**

\`\`\`
Twitter (2018):
- Passwords logged in plaintext
- 330 million users affected
- Had to force password resets

Facebook (2019):
- Hundreds of millions of passwords
- Stored in plaintext logs
- Searchable by employees

GitHub (2018):
- Accidentally logged passwords
- During password reset flow
- Exposed in server logs
\`\`\`

**Log Injection Attack:**

\`\`\`
Input: "login\\nADMIN: Authorized access granted"

Vulnerable log output:
2024-01-15 10:00:00 INFO User login
ADMIN: Authorized access granted

Appears legitimate in log analysis!
\`\`\`

**GDPR/HIPAA Compliance:**

| Requirement | Solution |
|-------------|----------|
| No PII in logs | Mask/hash identifiers |
| Right to erasure | Log rotation, anonymization |
| Access controls | Centralized, secured logging |
| Audit trail | Immutable, timestamped |`,
	order: 3,
	translations: {
		ru: {
			title: 'Практики безопасного логирования',
			description: `Научитесь реализовывать безопасное логирование, которое помогает отладке без создания рисков безопасности.

**Риски безопасности логирования:**

1. Раскрытие чувствительных данных - пароли, токены в логах
2. Log Injection - контролируемые атакующим записи
3. Нарушения PII - проблемы соответствия GDPR
4. Подделка логов - уничтожение улик

**Что НЕ логировать:**
- Пароли, API ключи, токены
- Номера кредитных карт, SSN
- Персональные данные (PII)

**Ваша задача:**

Реализуйте класс \`SecureLogger\`.`,
			hint1: `Для sanitizeContext рекурсивно обходите объект. Если ключ в sensitiveFields, маскируйте значение.`,
			hint2: `Для preventLogInjection экранируйте \\n и \\r чтобы предотвратить внедрение фейковых записей в лог.`,
			whyItMatters: `Логи часто являются упущенным вектором атак и кошмаром для compliance.`
		},
		uz: {
			title: 'Xavfsiz logging amaliyotlari',
			description: `Xavfsizlik xavflarini yaratmasdan tuzatishga yordam beradigan xavfsiz logging ni amalga oshirishni o'rganing.

**Logging xavfsizlik xavflari:**

1. Sezgir ma'lumotlarning ochilishi - parollar, tokenlar loglarda
2. Log Injection - tajovuzkor tomonidan boshqariladigan yozuvlar
3. PII buzilishlari - GDPR muvofiqlik muammolari

**Sizning vazifangiz:**

\`SecureLogger\` klassini amalga oshiring.`,
			hint1: `sanitizeContext uchun ob'ektni rekursiv o'ting. Agar kalit sensitiveFields da bo'lsa, qiymatni maskalang.`,
			hint2: `preventLogInjection uchun \\n va \\r ni escape qiling soxta log yozuvlarini kiritishni oldini olish uchun.`,
			whyItMatters: `Loglar ko'pincha e'tibordan chetda qoladigan hujum vektorlari va compliance uchun dahshat.`
		}
	}
};

export default task;
