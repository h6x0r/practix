import { Task } from '../../../../../../../types';

export const task: Task = {
	slug: 'sec-logging-monitoring',
	title: 'Security Logging and Monitoring Failures',
	difficulty: 'medium',
	tags: ['security', 'owasp', 'logging', 'monitoring', 'typescript'],
	estimatedTime: '35m',
	isPremium: false,
	youtubeUrl: '',
	description: `Learn to implement effective security logging and monitoring - essential for detecting and responding to attacks.

**What are Logging and Monitoring Failures?**

Without proper logging and monitoring, breaches cannot be detected. This vulnerability was added to OWASP Top 10 in 2017 because organizations often don't know they've been breached until months later.

**Common Failures:**

1. **No logging** of critical security events
2. **Logs only stored locally** and not centralized
3. **No alerting** on suspicious activities
4. **Sensitive data in logs** (passwords, tokens)
5. **Insufficient log retention**
6. **No correlation** between events

**Your Task:**

Implement a \`SecurityLogger\` class that:

1. Logs security-relevant events with proper structure
2. Detects and redacts sensitive data before logging
3. Categorizes events by severity
4. Provides audit trail capabilities

**Example Usage:**

\`\`\`typescript
const logger = new SecurityLogger();

// Log authentication events
logger.logAuth('user123', 'LOGIN_SUCCESS', { ip: '192.168.1.1' });
logger.logAuth('user456', 'LOGIN_FAILED', { ip: '10.0.0.1', reason: 'invalid_password' });

// Log access events
logger.logAccess('user123', 'READ', '/api/users/456', { allowed: true });

// Sensitive data is automatically redacted
logger.logAuth('user123', 'LOGIN_FAILED', { password: 'secret123' });
// password is redacted to '***REDACTED***'

// Get audit trail
logger.getAuditTrail('user123'); // All events for user
\`\`\``,
	initialCode: `interface SecurityEvent {
  timestamp: Date;
  eventType: string;
  severity: 'info' | 'warning' | 'critical';
  userId?: string;
  action: string;
  resource?: string;
  details: Record<string, any>;
  ip?: string;
}

interface AuditQuery {
  userId?: string;
  eventType?: string;
  startDate?: Date;
  endDate?: Date;
  severity?: string;
}

class SecurityLogger {
  private events: SecurityEvent[] = [];
  private readonly SENSITIVE_FIELDS = ['password', 'token', 'secret', 'apiKey', 'authorization', 'cookie'];

  logAuth(userId: string, action: string, details: Record<string, any> = {}): void {
    // TODO: Log authentication event
    // Determine severity based on action
    // Redact sensitive fields
  }

  logAccess(userId: string, action: string, resource: string, details: Record<string, any> = {}): void {
    // TODO: Log access/authorization event
    // Include resource being accessed
  }

  logSecurity(severity: 'info' | 'warning' | 'critical', action: string, details: Record<string, any> = {}): void {
    // TODO: Log general security event
    // Used for security alerts, policy violations, etc.
  }

  redactSensitive(data: Record<string, any>): Record<string, any> {
    // TODO: Redact sensitive fields from data
    // Replace values of SENSITIVE_FIELDS with '***REDACTED***'
    return data;
  }

  getAuditTrail(query: AuditQuery): SecurityEvent[] {
    // TODO: Query events based on criteria
    // Filter by userId, eventType, date range, severity
    return [];
  }

  detectSuspiciousPatterns(): { detected: boolean; patterns: string[] } {
    // TODO: Analyze events for suspicious patterns
    // Multiple failed logins, unusual access times, etc.
    return { detected: false, patterns: [] };
  }

  exportLogs(format: 'json' | 'csv'): string {
    // TODO: Export logs in specified format
    return '';
  }
}

export { SecurityLogger, SecurityEvent, AuditQuery };`,
	solutionCode: `interface SecurityEvent {
  timestamp: Date;
  eventType: string;
  severity: 'info' | 'warning' | 'critical';
  userId?: string;
  action: string;
  resource?: string;
  details: Record<string, any>;
  ip?: string;
}

interface AuditQuery {
  userId?: string;
  eventType?: string;
  startDate?: Date;
  endDate?: Date;
  severity?: string;
}

class SecurityLogger {
  private events: SecurityEvent[] = [];
  private readonly SENSITIVE_FIELDS = ['password', 'token', 'secret', 'apiKey', 'authorization', 'cookie', 'creditCard', 'ssn'];

  // Log authentication event
  logAuth(userId: string, action: string, details: Record<string, any> = {}): void {
    // Determine severity based on action
    let severity: 'info' | 'warning' | 'critical' = 'info';

    if (action.includes('FAILED') || action.includes('LOCKED')) {
      severity = 'warning';
    }
    if (action.includes('BREACH') || action.includes('COMPROMISED')) {
      severity = 'critical';
    }

    const redactedDetails = this.redactSensitive(details);

    this.events.push({
      timestamp: new Date(),
      eventType: 'AUTH',
      severity,
      userId,
      action,
      details: redactedDetails,
      ip: details.ip,
    });
  }

  // Log access/authorization event
  logAccess(userId: string, action: string, resource: string, details: Record<string, any> = {}): void {
    let severity: 'info' | 'warning' | 'critical' = 'info';

    // Unauthorized access is warning
    if (details.allowed === false) {
      severity = 'warning';
    }

    // Sensitive resource access
    if (resource.includes('admin') || resource.includes('sensitive')) {
      severity = severity === 'warning' ? 'critical' : 'warning';
    }

    this.events.push({
      timestamp: new Date(),
      eventType: 'ACCESS',
      severity,
      userId,
      action,
      resource,
      details: this.redactSensitive(details),
      ip: details.ip,
    });
  }

  // Log general security event
  logSecurity(severity: 'info' | 'warning' | 'critical', action: string, details: Record<string, any> = {}): void {
    this.events.push({
      timestamp: new Date(),
      eventType: 'SECURITY',
      severity,
      action,
      details: this.redactSensitive(details),
      ip: details.ip,
      userId: details.userId,
    });
  }

  // Redact sensitive fields from data
  redactSensitive(data: Record<string, any>): Record<string, any> {
    const redacted: Record<string, any> = {};

    for (const [key, value] of Object.entries(data)) {
      const lowerKey = key.toLowerCase();

      // Check if field is sensitive
      const isSensitive = this.SENSITIVE_FIELDS.some(
        field => lowerKey.includes(field.toLowerCase())
      );

      if (isSensitive && value) {
        redacted[key] = '***REDACTED***';
      } else if (typeof value === 'object' && value !== null && !Array.isArray(value)) {
        // Recursively redact nested objects
        redacted[key] = this.redactSensitive(value);
      } else {
        redacted[key] = value;
      }
    }

    return redacted;
  }

  // Query events based on criteria
  getAuditTrail(query: AuditQuery): SecurityEvent[] {
    return this.events.filter(event => {
      if (query.userId && event.userId !== query.userId) {
        return false;
      }
      if (query.eventType && event.eventType !== query.eventType) {
        return false;
      }
      if (query.severity && event.severity !== query.severity) {
        return false;
      }
      if (query.startDate && event.timestamp < query.startDate) {
        return false;
      }
      if (query.endDate && event.timestamp > query.endDate) {
        return false;
      }
      return true;
    });
  }

  // Detect suspicious patterns
  detectSuspiciousPatterns(): { detected: boolean; patterns: string[] } {
    const patterns: string[] = [];

    // Check for multiple failed logins from same user
    const userFailedLogins = new Map<string, number>();
    const ipFailedLogins = new Map<string, number>();

    for (const event of this.events) {
      if (event.eventType === 'AUTH' && event.action.includes('FAILED')) {
        if (event.userId) {
          const count = (userFailedLogins.get(event.userId) || 0) + 1;
          userFailedLogins.set(event.userId, count);
          if (count >= 5) {
            patterns.push(\`Multiple failed logins for user: \${event.userId}\`);
          }
        }
        if (event.ip) {
          const count = (ipFailedLogins.get(event.ip) || 0) + 1;
          ipFailedLogins.set(event.ip, count);
          if (count >= 10) {
            patterns.push(\`Multiple failed logins from IP: \${event.ip}\`);
          }
        }
      }
    }

    // Check for unauthorized access attempts
    const unauthorizedAttempts = this.events.filter(
      e => e.eventType === 'ACCESS' && e.details.allowed === false
    ).length;

    if (unauthorizedAttempts >= 5) {
      patterns.push(\`High number of unauthorized access attempts: \${unauthorizedAttempts}\`);
    }

    // Check for critical events
    const criticalEvents = this.events.filter(e => e.severity === 'critical').length;
    if (criticalEvents > 0) {
      patterns.push(\`Critical security events detected: \${criticalEvents}\`);
    }

    return {
      detected: patterns.length > 0,
      patterns: [...new Set(patterns)], // Remove duplicates
    };
  }

  // Export logs in specified format
  exportLogs(format: 'json' | 'csv'): string {
    if (format === 'json') {
      return JSON.stringify(this.events, null, 2);
    }

    if (format === 'csv') {
      const headers = ['timestamp', 'eventType', 'severity', 'userId', 'action', 'resource', 'ip'];
      const rows = this.events.map(event =>
        headers.map(h => {
          const val = event[h as keyof SecurityEvent];
          if (val instanceof Date) return val.toISOString();
          return val || '';
        }).join(',')
      );
      return [headers.join(','), ...rows].join('\\n');
    }

    return '';
  }
}

export { SecurityLogger, SecurityEvent, AuditQuery };`,
	hint1: `For logAuth, determine severity based on action string - FAILED/LOCKED actions should be warnings, BREACH/COMPROMISED should be critical. Always call redactSensitive on details.`,
	hint2: `For redactSensitive, iterate through object keys and check if they contain any SENSITIVE_FIELDS (case-insensitive). Replace matching values with '***REDACTED***'.`,
	testCode: `import { SecurityLogger } from './solution';

// Test1: logAuth logs event with correct type
test('Test1', () => {
  const logger = new SecurityLogger();
  logger.logAuth('user1', 'LOGIN_SUCCESS', { ip: '1.2.3.4' });
  const trail = logger.getAuditTrail({ userId: 'user1' });
  expect(trail.length).toBe(1);
  expect(trail[0].eventType).toBe('AUTH');
});

// Test2: Failed login has warning severity
test('Test2', () => {
  const logger = new SecurityLogger();
  logger.logAuth('user1', 'LOGIN_FAILED', {});
  const trail = logger.getAuditTrail({ userId: 'user1' });
  expect(trail[0].severity).toBe('warning');
});

// Test3: redactSensitive hides password
test('Test3', () => {
  const logger = new SecurityLogger();
  const result = logger.redactSensitive({ password: 'secret123', username: 'john' });
  expect(result.password).toBe('***REDACTED***');
  expect(result.username).toBe('john');
});

// Test4: redactSensitive hides token
test('Test4', () => {
  const logger = new SecurityLogger();
  const result = logger.redactSensitive({ apiToken: 'abc123' });
  expect(result.apiToken).toBe('***REDACTED***');
});

// Test5: logAccess logs resource
test('Test5', () => {
  const logger = new SecurityLogger();
  logger.logAccess('user1', 'READ', '/api/users', { allowed: true });
  const trail = logger.getAuditTrail({ eventType: 'ACCESS' });
  expect(trail[0].resource).toBe('/api/users');
});

// Test6: getAuditTrail filters by userId
test('Test6', () => {
  const logger = new SecurityLogger();
  logger.logAuth('user1', 'LOGIN', {});
  logger.logAuth('user2', 'LOGIN', {});
  const trail = logger.getAuditTrail({ userId: 'user1' });
  expect(trail.length).toBe(1);
});

// Test7: detectSuspiciousPatterns finds multiple failed logins
test('Test7', () => {
  const logger = new SecurityLogger();
  for (let i = 0; i < 5; i++) {
    logger.logAuth('user1', 'LOGIN_FAILED', {});
  }
  const result = logger.detectSuspiciousPatterns();
  expect(result.detected).toBe(true);
  expect(result.patterns.some(p => p.includes('failed logins'))).toBe(true);
});

// Test8: exportLogs returns JSON
test('Test8', () => {
  const logger = new SecurityLogger();
  logger.logAuth('user1', 'LOGIN', {});
  const json = logger.exportLogs('json');
  expect(JSON.parse(json)).toBeDefined();
});

// Test9: logSecurity logs with correct severity
test('Test9', () => {
  const logger = new SecurityLogger();
  logger.logSecurity('critical', 'INTRUSION_DETECTED', { ip: '10.0.0.1' });
  const trail = logger.getAuditTrail({ severity: 'critical' });
  expect(trail.length).toBe(1);
});

// Test10: Unauthorized access flagged as warning
test('Test10', () => {
  const logger = new SecurityLogger();
  logger.logAccess('user1', 'READ', '/admin', { allowed: false });
  const trail = logger.getAuditTrail({ userId: 'user1' });
  expect(trail[0].severity).not.toBe('info');
});`,
	whyItMatters: `Without proper logging, you won't know you've been breached until it's too late.

**Real-World Detection Failures:**

**1. Marriott (2014-2018)**
\`\`\`
Breach Duration: 4 YEARS undetected
Impact: 500 million guest records
Cause: Inadequate monitoring of Starwood systems
Data: Passport numbers, payment cards
Fine: $123 million (GDPR)
\`\`\`

**2. Equifax (2017)**
\`\`\`
Breach Duration: 76 days undetected
Impact: 147 million records
Cause: SSL certificate expired, blocking security monitoring
Data: SSNs, driver's licenses, credit cards
Settlement: $700 million
\`\`\`

**3. Target (2013)**
\`\`\`
Alerts Ignored: Security tools detected but ignored
Impact: 40 million payment cards
Cause: Alerts dismissed as false positives
Cost: $162 million in breach costs
\`\`\`

**What to Log (Security Events):**

| Category | Events |
|----------|--------|
| Authentication | Login success/failure, logout, password reset |
| Authorization | Access denied, privilege changes |
| Session | Creation, invalidation, timeout |
| Data | Sensitive data access, exports, bulk downloads |
| System | Config changes, startup/shutdown |
| Security | WAF blocks, rate limits, anomalies |

**Logging Best Practices:**

\`\`\`typescript
// ❌ BAD: No logging
app.post('/login', async (req, res) => {
  const user = await authenticate(req.body);
  return res.json({ token: createToken(user) });
});

// ✅ GOOD: Comprehensive logging
app.post('/login', async (req, res) => {
  try {
    const user = await authenticate(req.body);

    logger.info('LOGIN_SUCCESS', {
      userId: user.id,
      ip: req.ip,
      userAgent: req.headers['user-agent'],
    });

    return res.json({ token: createToken(user) });
  } catch (error) {
    logger.warn('LOGIN_FAILED', {
      email: req.body.email, // NOT password!
      ip: req.ip,
      reason: error.message,
    });
    throw error;
  }
});

// ❌ BAD: Sensitive data in logs
logger.info('User login', { password: req.body.password });

// ✅ GOOD: Sensitive data redacted
logger.info('User login', {
  email: req.body.email,
  password: '***REDACTED***',
});
\`\`\`

**Key Principles:**
1. Log security events in real-time
2. Never log sensitive data (passwords, tokens, PII)
3. Centralize logs for correlation
4. Set up alerts for suspicious patterns
5. Retain logs for compliance (typically 1-7 years)
6. Make logs tamper-evident`,
	order: 8,
	translations: {
		ru: {
			title: 'Сбои логирования и мониторинга безопасности',
			description: `Научитесь внедрять эффективное логирование и мониторинг безопасности - необходимые для обнаружения атак и реагирования на них.

**Что такое Logging and Monitoring Failures?**

Без надлежащего логирования и мониторинга невозможно обнаружить взломы. Организации часто узнают о взломе через месяцы.

**Ваша задача:**

Реализуйте класс \`SecurityLogger\`:

1. Логирование событий безопасности со структурой
2. Обнаружение и редактирование чувствительных данных
3. Категоризация событий по важности
4. Возможности аудит-трейла`,
			hint1: `Для logAuth определяйте severity по строке action - FAILED/LOCKED должны быть warning, BREACH/COMPROMISED - critical.`,
			hint2: `Для redactSensitive пройдите по ключам объекта и проверьте, содержат ли они SENSITIVE_FIELDS (без учёта регистра).`,
			whyItMatters: `Без надлежащего логирования вы не узнаете о взломе, пока не станет слишком поздно.`
		},
		uz: {
			title: 'Xavfsizlik loglash va monitoring xatolari',
			description: `Samarali xavfsizlik loglash va monitoringni amalga oshirishni o'rganing - hujumlarni aniqlash va javob berish uchun zarur.

**Logging va Monitoring Failures nima?**

To'g'ri loglash va monitoringsiz buzilishlarni aniqlab bo'lmaydi. Tashkilotlar ko'pincha buzilishni oylar o'tgandan keyin bilishadi.

**Sizning vazifangiz:**

\`SecurityLogger\` klassini amalga oshiring:

1. Xavfsizlik hodisalarini tuzilish bilan loglash
2. Nozik ma'lumotlarni aniqlash va tahrirlash
3. Hodisalarni jiddiylik bo'yicha turkumlash
4. Audit trail imkoniyatlari`,
			hint1: `logAuth uchun action satridan severity ni aniqlang - FAILED/LOCKED warning bo'lishi kerak.`,
			hint2: `redactSensitive uchun ob'ekt kalitlari bo'ylab yuring va ular SENSITIVE_FIELDS ni o'z ichiga olishini tekshiring.`,
			whyItMatters: `To'g'ri loglashsiz siz buzilish haqida kech bilib qolasiz.`
		}
	}
};

export default task;
