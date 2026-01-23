import { Task } from '../../../../../../../types';

export const task: Task = {
	slug: 'sec-csrf-prevention',
	title: 'CSRF: Cross-Site Request Forgery Protection',
	difficulty: 'medium',
	tags: ['security', 'owasp', 'csrf', 'typescript'],
	estimatedTime: '35m',
	isPremium: false,
	youtubeUrl: '',
	description: `Learn to prevent Cross-Site Request Forgery (CSRF) attacks - one of the OWASP Top 10 vulnerabilities.

**What is CSRF?**

CSRF attacks force authenticated users to perform unwanted actions on a web application. The attacker tricks the victim's browser into making requests to a site where the victim is authenticated.

**Attack Scenario:**

\`\`\`html
<!-- Attacker's malicious site -->
<html>
<body>
  <!-- Hidden form auto-submits to victim's bank -->
  <form action="https://bank.com/transfer" method="POST" id="evil">
    <input type="hidden" name="to" value="attacker123"/>
    <input type="hidden" name="amount" value="10000"/>
  </form>
  <script>document.getElementById('evil').submit();</script>
</body>
</html>
<!-- If victim is logged into bank.com, money is transferred! -->
\`\`\`

**Your Task:**

Implement a \`CsrfProtection\` class that:

1. Generates cryptographically secure CSRF tokens
2. Validates tokens against stored session tokens
3. Checks request origin/referer headers
4. Provides middleware-style protection function

**Example Usage:**

\`\`\`typescript
const csrf = new CsrfProtection();

// Generate token for form
const token = csrf.generateToken('session-123');
// Returns: 'a1b2c3d4e5f6...' (random hex string)

// Validate on form submission
csrf.validateToken('session-123', token); // true
csrf.validateToken('session-123', 'wrong'); // false

// Check origin
csrf.validateOrigin('https://mysite.com', 'https://mysite.com/api/transfer'); // true
csrf.validateOrigin('https://evil.com', 'https://mysite.com/api/transfer'); // false
\`\`\``,
	initialCode: `interface CsrfConfig {
  tokenLength?: number;
  tokenExpiry?: number; // in milliseconds
  allowedOrigins?: string[];
}

interface StoredToken {
  token: string;
  createdAt: number;
  sessionId: string;
}

class CsrfProtection {
  private tokens: Map<string, StoredToken> = new Map();
  private config: Required<CsrfConfig>;

  constructor(config?: CsrfConfig) {
    this.config = {
      tokenLength: config?.tokenLength || 32,
      tokenExpiry: config?.tokenExpiry || 3600000, // 1 hour
      allowedOrigins: config?.allowedOrigins || [],
    };
  }

  generateToken(sessionId: string): string {
    // TODO: Generate a cryptographically secure random token
    // Store it with sessionId and timestamp
    // Return the token
    return '';
  }

  validateToken(sessionId: string, token: string): boolean {
    // TODO: Check if token exists and matches session
    // Check if token is not expired
    // Optionally invalidate token after use (one-time use)
    return false;
  }

  validateOrigin(requestOrigin: string, targetUrl: string): boolean {
    // TODO: Parse target URL to get origin
    // Check if requestOrigin matches target or is in allowedOrigins
    return false;
  }

  validateReferer(referer: string, targetUrl: string): boolean {
    // TODO: Similar to validateOrigin but for Referer header
    // Referer can be missing (some privacy settings block it)
    return false;
  }

  protect(sessionId: string, requestToken: string, origin?: string, targetUrl?: string): ValidationResult {
    // TODO: Comprehensive protection check
    // Return detailed result with reason for failure
    return { valid: false, reason: 'Not implemented' };
  }

  cleanExpiredTokens(): number {
    // TODO: Remove expired tokens from storage
    return 0;
  }
}

interface ValidationResult {
  valid: boolean;
  reason?: string;
}

export { CsrfProtection, CsrfConfig, ValidationResult };`,
	solutionCode: `interface CsrfConfig {
  tokenLength?: number;
  tokenExpiry?: number; // in milliseconds
  allowedOrigins?: string[];
}

interface StoredToken {
  token: string;
  createdAt: number;
  sessionId: string;
}

class CsrfProtection {
  private tokens: Map<string, StoredToken> = new Map();
  private config: Required<CsrfConfig>;

  constructor(config?: CsrfConfig) {
    this.config = {
      tokenLength: config?.tokenLength || 32,
      tokenExpiry: config?.tokenExpiry || 3600000, // 1 hour
      allowedOrigins: config?.allowedOrigins || [],
    };
  }

  // Generate cryptographically secure random token
  generateToken(sessionId: string): string {
    // Generate random bytes and convert to hex string
    const array = new Uint8Array(this.config.tokenLength);
    crypto.getRandomValues(array);
    const token = Array.from(array, byte => byte.toString(16).padStart(2, '0')).join('');

    // Store token with session info
    this.tokens.set(token, {
      token,
      createdAt: Date.now(),
      sessionId,
    });

    return token;
  }

  // Validate token for session
  validateToken(sessionId: string, token: string): boolean {
    const stored = this.tokens.get(token);

    if (!stored) {
      return false; // Token doesn't exist
    }

    if (stored.sessionId !== sessionId) {
      return false; // Token belongs to different session
    }

    // Check expiration
    const now = Date.now();
    if (now - stored.createdAt > this.config.tokenExpiry) {
      this.tokens.delete(token); // Clean up expired token
      return false;
    }

    // Token is valid - invalidate for one-time use
    this.tokens.delete(token);
    return true;
  }

  // Validate request origin against target
  validateOrigin(requestOrigin: string, targetUrl: string): boolean {
    if (!requestOrigin) {
      return false;
    }

    try {
      const targetOrigin = new URL(targetUrl).origin;

      // Same origin check
      if (requestOrigin === targetOrigin) {
        return true;
      }

      // Check allowed origins
      return this.config.allowedOrigins.includes(requestOrigin);
    } catch {
      return false; // Invalid URL
    }
  }

  // Validate referer header
  validateReferer(referer: string, targetUrl: string): boolean {
    // Referer may be missing due to privacy settings - be lenient
    if (!referer) {
      return true; // Accept missing referer (but log for monitoring)
    }

    try {
      const refererOrigin = new URL(referer).origin;
      return this.validateOrigin(refererOrigin, targetUrl);
    } catch {
      return false; // Invalid referer URL
    }
  }

  // Comprehensive protection check
  protect(sessionId: string, requestToken: string, origin?: string, targetUrl?: string): ValidationResult {
    // Check token first (most important)
    if (!this.validateToken(sessionId, requestToken)) {
      return { valid: false, reason: 'Invalid or expired CSRF token' };
    }

    // Check origin if provided
    if (origin && targetUrl && !this.validateOrigin(origin, targetUrl)) {
      return { valid: false, reason: 'Origin mismatch' };
    }

    return { valid: true };
  }

  // Clean up expired tokens
  cleanExpiredTokens(): number {
    const now = Date.now();
    let cleaned = 0;

    for (const [token, stored] of this.tokens) {
      if (now - stored.createdAt > this.config.tokenExpiry) {
        this.tokens.delete(token);
        cleaned++;
      }
    }

    return cleaned;
  }
}

interface ValidationResult {
  valid: boolean;
  reason?: string;
}

export { CsrfProtection, CsrfConfig, ValidationResult };`,
	hint1: `For generateToken, use crypto.getRandomValues() to generate random bytes, then convert to hex. Store the token in the Map with sessionId and timestamp.`,
	hint2: `For validateToken, check if token exists in Map, if sessionId matches, and if it's not expired. Delete token after successful validation for one-time use.`,
	testCode: `import { CsrfProtection } from './solution';

// Test1: generateToken returns string of correct length
test('Test1', () => {
  const csrf = new CsrfProtection({ tokenLength: 16 });
  const token = csrf.generateToken('session-1');
  expect(typeof token).toBe('string');
  expect(token.length).toBe(32); // 16 bytes = 32 hex chars
});

// Test2: generateToken creates unique tokens
test('Test2', () => {
  const csrf = new CsrfProtection();
  const token1 = csrf.generateToken('session-1');
  const token2 = csrf.generateToken('session-1');
  expect(token1).not.toBe(token2);
});

// Test3: validateToken returns true for valid token
test('Test3', () => {
  const csrf = new CsrfProtection();
  const token = csrf.generateToken('session-1');
  expect(csrf.validateToken('session-1', token)).toBe(true);
});

// Test4: validateToken returns false for wrong session
test('Test4', () => {
  const csrf = new CsrfProtection();
  const token = csrf.generateToken('session-1');
  expect(csrf.validateToken('session-2', token)).toBe(false);
});

// Test5: validateToken returns false for invalid token
test('Test5', () => {
  const csrf = new CsrfProtection();
  csrf.generateToken('session-1');
  expect(csrf.validateToken('session-1', 'invalid-token')).toBe(false);
});

// Test6: validateOrigin returns true for same origin
test('Test6', () => {
  const csrf = new CsrfProtection();
  expect(csrf.validateOrigin('https://example.com', 'https://example.com/api/test')).toBe(true);
});

// Test7: validateOrigin returns false for different origin
test('Test7', () => {
  const csrf = new CsrfProtection();
  expect(csrf.validateOrigin('https://evil.com', 'https://example.com/api/test')).toBe(false);
});

// Test8: validateOrigin accepts allowed origins
test('Test8', () => {
  const csrf = new CsrfProtection({ allowedOrigins: ['https://trusted.com'] });
  expect(csrf.validateOrigin('https://trusted.com', 'https://example.com/api/test')).toBe(true);
});

// Test9: protect returns valid for correct token
test('Test9', () => {
  const csrf = new CsrfProtection();
  const token = csrf.generateToken('session-1');
  const result = csrf.protect('session-1', token);
  expect(result.valid).toBe(true);
});

// Test10: validateToken invalidates after use (one-time use)
test('Test10', () => {
  const csrf = new CsrfProtection();
  const token = csrf.generateToken('session-1');
  csrf.validateToken('session-1', token); // First use
  expect(csrf.validateToken('session-1', token)).toBe(false); // Second use fails
});`,
	whyItMatters: `CSRF attacks have caused significant damage to major companies and users worldwide.

**Real-World CSRF Attacks:**

**1. Netflix (2006)**
\`\`\`
Attack: CSRF vulnerability in DVD queue
Impact: Attackers could add DVDs to victim's queue
Method: Hidden form on malicious website
Result: Embarrassing titles added to queues
\`\`\`

**2. ING Direct (2008)**
\`\`\`
Attack: CSRF in online banking
Impact: Could transfer funds between accounts
Method: Hidden form auto-submission
Result: Unauthorized fund transfers
\`\`\`

**3. YouTube (2008)**
\`\`\`
Attack: CSRF in comment/like system
Impact: Automated likes and comments
Method: XHR requests from malicious sites
Result: Manipulated video rankings
\`\`\`

**Defense Strategies:**

| Method | Effectiveness | Implementation |
|--------|---------------|----------------|
| CSRF Tokens | Best | Hidden form field + session storage |
| SameSite Cookies | Great | \`Set-Cookie: SameSite=Strict\` |
| Origin/Referer Check | Good | Server-side header validation |
| Double Submit Cookie | Good | Cookie + request parameter match |
| Custom Headers | Good | X-Requested-With for AJAX |

**Modern Best Practices:**

\`\`\`typescript
// Express.js with csurf
app.use(csrf({ cookie: true }));

// Form with CSRF token
<form action="/transfer" method="POST">
  <input type="hidden" name="_csrf" value="<%= csrfToken %>">
  <button>Transfer</button>
</form>

// SameSite Cookie
Set-Cookie: sessionId=abc123; SameSite=Strict; Secure; HttpOnly

// Fetch with credentials
fetch('/api/action', {
  method: 'POST',
  credentials: 'same-origin',
  headers: { 'X-CSRF-Token': token }
});
\`\`\``,
	order: 2,
	translations: {
		ru: {
			title: 'CSRF: Защита от подделки межсайтовых запросов',
			description: `Научитесь предотвращать CSRF-атаки (Cross-Site Request Forgery) - одну из уязвимостей OWASP Top 10.

**Что такое CSRF?**

CSRF-атаки заставляют аутентифицированных пользователей выполнять нежелательные действия. Злоумышленник обманывает браузер жертвы, заставляя его отправлять запросы на сайт, где жертва авторизована.

**Ваша задача:**

Реализуйте класс \`CsrfProtection\`:

1. Генерация криптографически безопасных CSRF-токенов
2. Валидация токенов для сессий
3. Проверка заголовков Origin/Referer
4. Комплексная функция защиты`,
			hint1: `Для generateToken используйте crypto.getRandomValues() для генерации случайных байтов, затем конвертируйте в hex.`,
			hint2: `Для validateToken проверьте существование токена, соответствие sessionId и срок действия. Удалите токен после использования.`,
			whyItMatters: `CSRF-атаки нанесли значительный ущерб крупным компаниям и пользователям по всему миру.`
		},
		uz: {
			title: 'CSRF: Saytlararo so\'rov soxtalashtirish himoyasi',
			description: `CSRF hujumlarini (Cross-Site Request Forgery) oldini olishni o'rganing - OWASP Top 10 zaifliklaridan biri.

**CSRF nima?**

CSRF hujumlari autentifikatsiyadan o'tgan foydalanuvchilarni istalmagan harakatlarni bajarishga majbur qiladi.

**Sizning vazifangiz:**

\`CsrfProtection\` klassini amalga oshiring:

1. Kriptografik xavfsiz CSRF tokenlarini yaratish
2. Sessiyalar uchun tokenlarni tasdiqlash
3. Origin/Referer sarlavhalarini tekshirish
4. Keng qamrovli himoya funksiyasi`,
			hint1: `generateToken uchun tasodifiy baytlarni yaratish uchun crypto.getRandomValues() dan foydalaning.`,
			hint2: `validateToken uchun token mavjudligini, sessionId mosligini va amal qilish muddatini tekshiring.`,
			whyItMatters: `CSRF hujumlari yirik kompaniyalar va foydalanuvchilarga katta zarar yetkazdi.`
		}
	}
};

export default task;
