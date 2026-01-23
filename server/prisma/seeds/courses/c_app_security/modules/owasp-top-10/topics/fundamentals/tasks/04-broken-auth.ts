import { Task } from '../../../../../../../types';

export const task: Task = {
	slug: 'sec-broken-auth',
	title: 'Broken Authentication: Secure Session Management',
	difficulty: 'hard',
	tags: ['security', 'owasp', 'authentication', 'typescript'],
	estimatedTime: '45m',
	isPremium: false,
	youtubeUrl: '',
	description: `Learn to identify and fix broken authentication vulnerabilities - consistently in OWASP Top 10.

**What is Broken Authentication?**

Broken authentication occurs when session management or credential handling is improperly implemented, allowing attackers to compromise passwords, keys, session tokens, or assume other users' identities.

**Common Vulnerabilities:**

1. **Weak Passwords** - No complexity requirements
2. **Credential Stuffing** - No rate limiting or account lockout
3. **Session Fixation** - Sessions not regenerated on login
4. **Session Hijacking** - Tokens exposed in URL or not secured
5. **Insecure Password Recovery** - Security questions, email-only

**Your Task:**

Implement an \`AuthSecurityChecker\` class that:

1. Validates password strength (entropy, patterns, breached lists)
2. Detects brute force attempts with rate limiting
3. Validates session security (tokens, expiry, binding)
4. Provides secure session configuration recommendations

**Example Usage:**

\`\`\`typescript
const checker = new AuthSecurityChecker();

// Check password strength
checker.checkPassword('123456');
// { strong: false, issues: ['Too common', 'No uppercase', ...] }

checker.checkPassword('Tr0ub4dor&3!');
// { strong: true, entropy: 65, issues: [] }

// Track login attempts
checker.recordLoginAttempt('user@email.com', false);
checker.isAccountLocked('user@email.com'); // false (first attempt)
// After 5 failures...
checker.isAccountLocked('user@email.com'); // true
\`\`\``,
	initialCode: `interface PasswordCheckResult {
  strong: boolean;
  entropy: number;
  issues: string[];
}

interface SessionConfig {
  httpOnly: boolean;
  secure: boolean;
  sameSite: 'Strict' | 'Lax' | 'None';
  maxAge: number;
  regenerateOnLogin: boolean;
}

interface LoginAttempt {
  email: string;
  success: boolean;
  timestamp: number;
  ipAddress?: string;
}

class AuthSecurityChecker {
  private loginAttempts: Map<string, LoginAttempt[]> = new Map();
  private readonly COMMON_PASSWORDS = [
    '123456', 'password', '12345678', 'qwerty', '123456789',
    'letmein', 'welcome', 'admin', 'iloveyou', 'monkey'
  ];
  private readonly MAX_ATTEMPTS = 5;
  private readonly LOCKOUT_DURATION = 900000; // 15 minutes

  checkPassword(password: string): PasswordCheckResult {
    // TODO: Check password strength
    // - Check minimum length (8+)
    // - Check for uppercase, lowercase, numbers, special chars
    // - Check against common passwords
    // - Calculate entropy
    return { strong: false, entropy: 0, issues: [] };
  }

  calculateEntropy(password: string): number {
    // TODO: Calculate password entropy
    // Entropy = length * log2(character_pool_size)
    return 0;
  }

  recordLoginAttempt(email: string, success: boolean, ipAddress?: string): void {
    // TODO: Record login attempt
    // Clear old attempts on success
    // Store failure for rate limiting
  }

  isAccountLocked(email: string): boolean {
    // TODO: Check if account is locked due to too many failed attempts
    // Consider LOCKOUT_DURATION for temporary lockout
    return false;
  }

  getSecureSessionConfig(): SessionConfig {
    // TODO: Return recommended secure session configuration
    return {
      httpOnly: false,
      secure: false,
      sameSite: 'None',
      maxAge: 0,
      regenerateOnLogin: false,
    };
  }

  validateSessionToken(token: string): { valid: boolean; issues: string[] } {
    // TODO: Validate session token format and characteristics
    // Check length, randomness, format
    return { valid: false, issues: [] };
  }
}

export { AuthSecurityChecker, PasswordCheckResult, SessionConfig };`,
	solutionCode: `interface PasswordCheckResult {
  strong: boolean;
  entropy: number;
  issues: string[];
}

interface SessionConfig {
  httpOnly: boolean;
  secure: boolean;
  sameSite: 'Strict' | 'Lax' | 'None';
  maxAge: number;
  regenerateOnLogin: boolean;
}

interface LoginAttempt {
  email: string;
  success: boolean;
  timestamp: number;
  ipAddress?: string;
}

class AuthSecurityChecker {
  private loginAttempts: Map<string, LoginAttempt[]> = new Map();
  private readonly COMMON_PASSWORDS = [
    '123456', 'password', '12345678', 'qwerty', '123456789',
    'letmein', 'welcome', 'admin', 'iloveyou', 'monkey',
    'dragon', 'master', 'sunshine', 'princess', 'football'
  ];
  private readonly MAX_ATTEMPTS = 5;
  private readonly LOCKOUT_DURATION = 900000; // 15 minutes

  // Check password strength comprehensively
  checkPassword(password: string): PasswordCheckResult {
    const issues: string[] = [];

    // Length check
    if (password.length < 8) {
      issues.push('Password must be at least 8 characters');
    }
    if (password.length < 12) {
      issues.push('Consider using 12+ characters for better security');
    }

    // Character type checks
    if (!/[A-Z]/.test(password)) {
      issues.push('Add uppercase letters');
    }
    if (!/[a-z]/.test(password)) {
      issues.push('Add lowercase letters');
    }
    if (!/[0-9]/.test(password)) {
      issues.push('Add numbers');
    }
    if (!/[!@#$%^&*()_+\\-=\\[\\]{};':\"\\\\|,.<>\\/?]/.test(password)) {
      issues.push('Add special characters');
    }

    // Common password check
    if (this.COMMON_PASSWORDS.includes(password.toLowerCase())) {
      issues.push('This password is too common');
    }

    // Pattern checks
    if (/(.)\\1{2,}/.test(password)) {
      issues.push('Avoid repeated characters');
    }
    if (/^(?:abc|123|qwerty)/i.test(password)) {
      issues.push('Avoid sequential patterns');
    }

    const entropy = this.calculateEntropy(password);

    return {
      strong: issues.filter(i => !i.includes('Consider')).length === 0 && entropy >= 50,
      entropy,
      issues,
    };
  }

  // Calculate password entropy
  calculateEntropy(password: string): number {
    let poolSize = 0;

    if (/[a-z]/.test(password)) poolSize += 26;
    if (/[A-Z]/.test(password)) poolSize += 26;
    if (/[0-9]/.test(password)) poolSize += 10;
    if (/[!@#$%^&*()_+\\-=\\[\\]{};':\"\\\\|,.<>\\/?]/.test(password)) poolSize += 32;

    if (poolSize === 0) return 0;

    // Entropy = length * log2(poolSize)
    return Math.round(password.length * Math.log2(poolSize));
  }

  // Record login attempt for rate limiting
  recordLoginAttempt(email: string, success: boolean, ipAddress?: string): void {
    const normalizedEmail = email.toLowerCase();

    if (success) {
      // Clear attempts on successful login
      this.loginAttempts.delete(normalizedEmail);
      return;
    }

    // Record failed attempt
    const attempts = this.loginAttempts.get(normalizedEmail) || [];
    attempts.push({
      email: normalizedEmail,
      success: false,
      timestamp: Date.now(),
      ipAddress,
    });

    this.loginAttempts.set(normalizedEmail, attempts);
  }

  // Check if account is locked
  isAccountLocked(email: string): boolean {
    const normalizedEmail = email.toLowerCase();
    const attempts = this.loginAttempts.get(normalizedEmail) || [];

    // Filter to recent attempts within lockout window
    const now = Date.now();
    const recentAttempts = attempts.filter(
      a => now - a.timestamp < this.LOCKOUT_DURATION
    );

    // Update stored attempts (cleanup old ones)
    if (recentAttempts.length !== attempts.length) {
      this.loginAttempts.set(normalizedEmail, recentAttempts);
    }

    return recentAttempts.length >= this.MAX_ATTEMPTS;
  }

  // Return secure session configuration
  getSecureSessionConfig(): SessionConfig {
    return {
      httpOnly: true,        // Prevent JavaScript access
      secure: true,          // HTTPS only
      sameSite: 'Strict',    // Prevent CSRF
      maxAge: 3600000,       // 1 hour
      regenerateOnLogin: true, // Prevent session fixation
    };
  }

  // Validate session token characteristics
  validateSessionToken(token: string): { valid: boolean; issues: string[] } {
    const issues: string[] = [];

    // Length check (should be at least 128 bits = 32 hex chars)
    if (token.length < 32) {
      issues.push('Token too short - should be at least 128 bits');
    }

    // Should be random-looking (hex or base64)
    if (!/^[a-zA-Z0-9+/=_-]+$/.test(token)) {
      issues.push('Token contains unexpected characters');
    }

    // Should not contain predictable patterns
    if (/^[0-9]+$/.test(token)) {
      issues.push('Token appears to be sequential/numeric only');
    }
    if (token.includes('user') || token.includes('session')) {
      issues.push('Token should not contain predictable strings');
    }

    return {
      valid: issues.length === 0,
      issues,
    };
  }
}

export { AuthSecurityChecker, PasswordCheckResult, SessionConfig };`,
	hint1: `For checkPassword, use regex to test for character types. Calculate entropy using the formula: length * log2(character_pool_size). Return issues as an array of improvement suggestions.`,
	hint2: `For isAccountLocked, filter attempts to only those within LOCKOUT_DURATION. Compare count against MAX_ATTEMPTS. Remember to clear attempts on successful login.`,
	testCode: `import { AuthSecurityChecker } from './solution';

// Test1: Weak password detected
test('Test1', () => {
  const checker = new AuthSecurityChecker();
  const result = checker.checkPassword('123456');
  expect(result.strong).toBe(false);
  expect(result.issues.length).toBeGreaterThan(0);
});

// Test2: Strong password passes
test('Test2', () => {
  const checker = new AuthSecurityChecker();
  const result = checker.checkPassword('Tr0ub4dor&3!xY');
  expect(result.strong).toBe(true);
});

// Test3: Entropy calculation works
test('Test3', () => {
  const checker = new AuthSecurityChecker();
  const entropy = checker.calculateEntropy('Password1!');
  expect(entropy).toBeGreaterThan(40);
});

// Test4: Common password detected
test('Test4', () => {
  const checker = new AuthSecurityChecker();
  const result = checker.checkPassword('password');
  expect(result.issues.some(i => i.toLowerCase().includes('common'))).toBe(true);
});

// Test5: Account not locked initially
test('Test5', () => {
  const checker = new AuthSecurityChecker();
  expect(checker.isAccountLocked('test@email.com')).toBe(false);
});

// Test6: Account locked after 5 failures
test('Test6', () => {
  const checker = new AuthSecurityChecker();
  for (let i = 0; i < 5; i++) {
    checker.recordLoginAttempt('test@email.com', false);
  }
  expect(checker.isAccountLocked('test@email.com')).toBe(true);
});

// Test7: Successful login clears attempts
test('Test7', () => {
  const checker = new AuthSecurityChecker();
  for (let i = 0; i < 4; i++) {
    checker.recordLoginAttempt('test@email.com', false);
  }
  checker.recordLoginAttempt('test@email.com', true);
  expect(checker.isAccountLocked('test@email.com')).toBe(false);
});

// Test8: Secure session config has httpOnly
test('Test8', () => {
  const checker = new AuthSecurityChecker();
  const config = checker.getSecureSessionConfig();
  expect(config.httpOnly).toBe(true);
  expect(config.secure).toBe(true);
});

// Test9: Session token validation - short token fails
test('Test9', () => {
  const checker = new AuthSecurityChecker();
  const result = checker.validateSessionToken('abc123');
  expect(result.valid).toBe(false);
  expect(result.issues.some(i => i.includes('short'))).toBe(true);
});

// Test10: Session token validation - good token passes
test('Test10', () => {
  const checker = new AuthSecurityChecker();
  const result = checker.validateSessionToken('a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6q7r8s9t0');
  expect(result.valid).toBe(true);
});`,
	whyItMatters: `Broken authentication is consistently in OWASP Top 10 and leads to account takeover.

**Major Breaches from Auth Failures:**

**1. Yahoo (2013-2014)**
\`\`\`
Impact: 3 billion accounts compromised
Cause: Weak password hashing, forged cookies
Method: MD5 hashing + state-sponsored attackers
Result: Largest data breach in history
\`\`\`

**2. LinkedIn (2012)**
\`\`\`
Impact: 165 million accounts
Cause: SHA-1 without salt for passwords
Method: Rainbow table attacks
Result: Passwords cracked within days
\`\`\`

**3. Dropbox (2012)**
\`\`\`
Impact: 68 million accounts
Cause: Employee password reuse
Method: Credential stuffing from other breaches
Result: Password reset for all users
\`\`\`

**Authentication Security Checklist:**

| Area | Best Practice |
|------|---------------|
| Passwords | bcrypt/Argon2, 12+ chars, breach check |
| Sessions | Random 128-bit tokens, httpOnly, secure |
| Rate Limiting | 5 attempts, exponential backoff, CAPTCHA |
| MFA | TOTP/WebAuthn, backup codes |
| Recovery | Time-limited tokens, not security questions |

**Secure Password Storage:**

\`\`\`typescript
// ❌ NEVER: Plain text or weak hash
const hash = md5(password);
const hash = sha1(password);

// ✅ ALWAYS: Adaptive hashing with salt
import * as bcrypt from 'bcrypt';
const hash = await bcrypt.hash(password, 12);
const valid = await bcrypt.compare(input, hash);

// ✅ BETTER: Argon2id (memory-hard)
import * as argon2 from 'argon2';
const hash = await argon2.hash(password, {
  type: argon2.argon2id,
  memoryCost: 65536,
  timeCost: 3,
  parallelism: 4,
});
\`\`\`

**Session Management:**

\`\`\`typescript
// Generate secure session ID
const sessionId = crypto.randomBytes(32).toString('hex');

// Secure cookie settings
res.cookie('sessionId', sessionId, {
  httpOnly: true,    // No JavaScript access
  secure: true,      // HTTPS only
  sameSite: 'strict', // CSRF protection
  maxAge: 3600000,   // 1 hour
  path: '/',
});

// Regenerate on authentication state change
req.session.regenerate(() => {
  req.session.userId = user.id;
});
\`\`\``,
	order: 3,
	translations: {
		ru: {
			title: 'Сломанная аутентификация: Безопасное управление сессиями',
			description: `Научитесь выявлять и исправлять уязвимости аутентификации - постоянно в OWASP Top 10.

**Что такое Broken Authentication?**

Сломанная аутентификация возникает при неправильной реализации управления сессиями или учётными данными.

**Частые уязвимости:**

1. Слабые пароли
2. Отсутствие ограничения попыток входа
3. Фиксация сессии
4. Небезопасное восстановление пароля

**Ваша задача:**

Реализуйте класс \`AuthSecurityChecker\`:

1. Проверка надёжности пароля
2. Обнаружение брутфорс-атак
3. Валидация безопасности сессий
4. Рекомендации по конфигурации`,
			hint1: `Для checkPassword используйте regex для проверки типов символов. Вычислите энтропию по формуле: длина * log2(размер_пула_символов).`,
			hint2: `Для isAccountLocked фильтруйте попытки в пределах LOCKOUT_DURATION. Сравните количество с MAX_ATTEMPTS.`,
			whyItMatters: `Сломанная аутентификация постоянно в OWASP Top 10 и ведёт к захвату аккаунтов.`
		},
		uz: {
			title: 'Buzilgan autentifikatsiya: Xavfsiz sessiya boshqaruvi',
			description: `Autentifikatsiya zaifliklarini aniqlash va tuzatishni o'rganing - doimiy OWASP Top 10 da.

**Broken Authentication nima?**

Buzilgan autentifikatsiya sessiya yoki hisob ma'lumotlarini noto'g'ri boshqarishda yuz beradi.

**Sizning vazifangiz:**

\`AuthSecurityChecker\` klassini amalga oshiring:

1. Parol kuchini tekshirish
2. Bruteforce hujumlarini aniqlash
3. Sessiya xavfsizligini tasdiqlash
4. Konfiguratsiya tavsiyalari`,
			hint1: `checkPassword uchun belgi turlarini tekshirish uchun regex dan foydalaning.`,
			hint2: `isAccountLocked uchun urinishlarni LOCKOUT_DURATION ichida filtrlang.`,
			whyItMatters: `Buzilgan autentifikatsiya doimiy ravishda OWASP Top 10 da va akkauntlarni egallab olishga olib keladi.`
		}
	}
};

export default task;
