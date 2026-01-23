import { Task } from '../../../../../../../types';

export const task: Task = {
	slug: 'sec-oauth2-security',
	title: 'OAuth 2.0 Security: Flows and Vulnerabilities',
	difficulty: 'hard',
	tags: ['security', 'oauth', 'authentication', 'authorization', 'typescript'],
	estimatedTime: '45m',
	isPremium: true,
	youtubeUrl: '',
	description: `Learn OAuth 2.0 security - the industry standard for authorization delegation.

**What is OAuth 2.0?**

OAuth 2.0 is an authorization framework that enables third-party applications to obtain limited access to a user's account on an HTTP service. It's used by Google, Facebook, GitHub, and countless other services.

**OAuth 2.0 Flows:**

1. **Authorization Code** (Web apps - most secure)
2. **Authorization Code + PKCE** (Mobile/SPA - recommended)
3. **Implicit** (Deprecated - insecure)
4. **Client Credentials** (Machine-to-machine)
5. **Resource Owner Password** (Legacy - avoid)

**Common OAuth Vulnerabilities:**

1. **CSRF in OAuth flow** - Missing state parameter
2. **Open Redirect** - Redirect URI manipulation
3. **Token Leakage** - Tokens in referrer headers
4. **Insufficient Redirect URI Validation** - Subdomain takeover

**Your Task:**

Implement an \`OAuthSecurityValidator\` class that:

1. Validates OAuth 2.0 requests for security issues
2. Generates and validates state parameters (CSRF protection)
3. Validates redirect URIs against registered patterns
4. Detects common OAuth attack patterns

**Example Usage:**

\`\`\`typescript
const validator = new OAuthSecurityValidator({
  allowedRedirectUris: ['https://myapp.com/callback'],
  requirePKCE: true,
});

// Validate authorization request
validator.validateAuthRequest({
  redirectUri: 'https://myapp.com/callback',
  state: 'random-state',
  codeChallenge: 'challenge',
  codeChallengeMethod: 'S256',
});
// { valid: true }

// Generate secure state
const state = validator.generateState();
validator.validateState(state, state); // true
\`\`\``,
	initialCode: `interface OAuthConfig {
  allowedRedirectUris: string[];
  requirePKCE?: boolean;
  allowedScopes?: string[];
}

interface AuthorizationRequest {
  responseType: string;
  clientId: string;
  redirectUri: string;
  scope?: string;
  state?: string;
  codeChallenge?: string;
  codeChallengeMethod?: string;
}

interface ValidationResult {
  valid: boolean;
  issues: string[];
}

class OAuthSecurityValidator {
  private config: OAuthConfig;
  private stateStore: Set<string> = new Set();

  constructor(config: OAuthConfig) {
    this.config = config;
  }

  validateAuthRequest(request: AuthorizationRequest): ValidationResult {
    // TODO: Validate OAuth authorization request
    // Check redirect URI, state, PKCE if required
    return { valid: false, issues: [] };
  }

  validateRedirectUri(redirectUri: string): boolean {
    // TODO: Validate redirect URI against allowed patterns
    // Exact match or pattern matching
    return false;
  }

  generateState(): string {
    // TODO: Generate cryptographically secure state parameter
    // Store for later validation
    return '';
  }

  validateState(expected: string, actual: string): boolean {
    // TODO: Validate state parameter (CSRF protection)
    // Should be timing-safe comparison
    return false;
  }

  validatePKCE(codeChallenge: string, codeVerifier: string, method: string): boolean {
    // TODO: Validate PKCE code challenge
    // Support S256 and plain methods
    return false;
  }

  detectAttacks(request: AuthorizationRequest): { detected: boolean; attacks: string[] } {
    // TODO: Detect common OAuth attacks
    // Open redirect, token leakage, etc.
    return { detected: false, attacks: [] };
  }

  generateCodeVerifier(): string {
    // TODO: Generate secure code verifier for PKCE
    return '';
  }

  generateCodeChallenge(verifier: string, method: string): string {
    // TODO: Generate code challenge from verifier
    // Support S256 (recommended) and plain
    return '';
  }
}

export { OAuthSecurityValidator, OAuthConfig, AuthorizationRequest, ValidationResult };`,
	solutionCode: `interface OAuthConfig {
  allowedRedirectUris: string[];
  requirePKCE?: boolean;
  allowedScopes?: string[];
}

interface AuthorizationRequest {
  responseType: string;
  clientId: string;
  redirectUri: string;
  scope?: string;
  state?: string;
  codeChallenge?: string;
  codeChallengeMethod?: string;
}

interface ValidationResult {
  valid: boolean;
  issues: string[];
}

class OAuthSecurityValidator {
  private config: OAuthConfig;
  private stateStore: Set<string> = new Set();

  constructor(config: OAuthConfig) {
    this.config = {
      ...config,
      requirePKCE: config.requirePKCE ?? true,
    };
  }

  // Validate OAuth authorization request
  validateAuthRequest(request: AuthorizationRequest): ValidationResult {
    const issues: string[] = [];

    // Validate response type
    if (!['code', 'token'].includes(request.responseType)) {
      issues.push('Invalid response_type');
    }

    // Warn about implicit flow
    if (request.responseType === 'token') {
      issues.push('Implicit flow (token) is deprecated - use Authorization Code with PKCE');
    }

    // Validate redirect URI
    if (!this.validateRedirectUri(request.redirectUri)) {
      issues.push('Redirect URI not in allowed list');
    }

    // Require state parameter (CSRF protection)
    if (!request.state) {
      issues.push('Missing state parameter (CSRF protection)');
    } else if (request.state.length < 16) {
      issues.push('State parameter too short (min 16 chars)');
    }

    // PKCE validation
    if (this.config.requirePKCE) {
      if (!request.codeChallenge) {
        issues.push('PKCE code_challenge required');
      }
      if (request.codeChallengeMethod !== 'S256') {
        issues.push('code_challenge_method must be S256');
      }
    }

    // Validate scopes if configured
    if (this.config.allowedScopes && request.scope) {
      const requestedScopes = request.scope.split(' ');
      for (const scope of requestedScopes) {
        if (!this.config.allowedScopes.includes(scope)) {
          issues.push(\`Scope '\${scope}' not allowed\`);
        }
      }
    }

    return {
      valid: issues.length === 0,
      issues,
    };
  }

  // Validate redirect URI against allowed patterns
  validateRedirectUri(redirectUri: string): boolean {
    try {
      const url = new URL(redirectUri);

      // Must be HTTPS (except localhost for development)
      if (url.protocol !== 'https:' && url.hostname !== 'localhost') {
        return false;
      }

      // Check against allowed URIs (exact match)
      return this.config.allowedRedirectUris.some(allowed => {
        // Exact match or pattern with *
        if (allowed.includes('*')) {
          const pattern = allowed.replace(/[.+?^\${}()|[\\]\\\\]/g, '\\\\$&').replace(/\\*/g, '.*');
          return new RegExp(\`^\${pattern}$\`).test(redirectUri);
        }
        return redirectUri === allowed || redirectUri.startsWith(allowed + '?');
      });
    } catch {
      return false;
    }
  }

  // Generate cryptographically secure state parameter
  generateState(): string {
    const array = new Uint8Array(32);
    crypto.getRandomValues(array);
    const state = Array.from(array, b => b.toString(16).padStart(2, '0')).join('');
    this.stateStore.add(state);
    return state;
  }

  // Validate state parameter (CSRF protection)
  validateState(expected: string, actual: string): boolean {
    if (!expected || !actual) {
      return false;
    }

    // Timing-safe comparison
    if (expected.length !== actual.length) {
      return false;
    }

    let result = 0;
    for (let i = 0; i < expected.length; i++) {
      result |= expected.charCodeAt(i) ^ actual.charCodeAt(i);
    }

    const valid = result === 0;

    // Remove used state
    if (valid) {
      this.stateStore.delete(expected);
    }

    return valid;
  }

  // Validate PKCE code challenge
  validatePKCE(codeChallenge: string, codeVerifier: string, method: string): boolean {
    if (method === 'plain') {
      return codeChallenge === codeVerifier;
    }

    if (method === 'S256') {
      const computed = this.generateCodeChallenge(codeVerifier, 'S256');
      return codeChallenge === computed;
    }

    return false;
  }

  // Detect common OAuth attacks
  detectAttacks(request: AuthorizationRequest): { detected: boolean; attacks: string[] } {
    const attacks: string[] = [];

    // Missing state (CSRF)
    if (!request.state) {
      attacks.push('csrf_no_state');
    }

    // Open redirect attempt
    try {
      const url = new URL(request.redirectUri);
      if (url.protocol !== 'https:' && url.hostname !== 'localhost') {
        attacks.push('open_redirect_http');
      }

      // Check for suspicious redirect targets
      if (url.hostname.includes('evil') || url.hostname.includes('attacker')) {
        attacks.push('suspicious_redirect_uri');
      }

      // Path traversal in redirect
      if (request.redirectUri.includes('..') || request.redirectUri.includes('%2e%2e')) {
        attacks.push('path_traversal');
      }
    } catch {
      attacks.push('malformed_redirect_uri');
    }

    // Implicit flow (deprecated)
    if (request.responseType === 'token') {
      attacks.push('deprecated_implicit_flow');
    }

    // Missing PKCE for public clients
    if (!request.codeChallenge && this.config.requirePKCE) {
      attacks.push('missing_pkce');
    }

    return {
      detected: attacks.length > 0,
      attacks,
    };
  }

  // Generate secure code verifier for PKCE
  generateCodeVerifier(): string {
    // 43-128 characters, URL-safe
    const array = new Uint8Array(32);
    crypto.getRandomValues(array);
    return Array.from(array, b => b.toString(16).padStart(2, '0')).join('');
  }

  // Generate code challenge from verifier
  generateCodeChallenge(verifier: string, method: string): string {
    if (method === 'plain') {
      return verifier;
    }

    if (method === 'S256') {
      // SHA-256 hash, then base64url encode
      // In browser, would use SubtleCrypto
      // Simplified version for demonstration
      const encoder = new TextEncoder();
      const data = encoder.encode(verifier);

      // This is a simplified hash - in production use crypto.subtle
      let hash = 0;
      for (let i = 0; i < data.length; i++) {
        hash = ((hash << 5) - hash) + data[i];
        hash = hash & hash;
      }

      return btoa(String(Math.abs(hash)))
        .replace(/\\+/g, '-')
        .replace(/\\//g, '_')
        .replace(/=/g, '');
    }

    return '';
  }
}

export { OAuthSecurityValidator, OAuthConfig, AuthorizationRequest, ValidationResult };`,
	hint1: `For validateAuthRequest, check each required field: response_type should be 'code', redirect_uri must be in allowed list, state should exist and be long enough, and PKCE should be present if required.`,
	hint2: `For validateState, use timing-safe comparison (XOR each character and accumulate). This prevents timing attacks that could leak the expected state value.`,
	testCode: `import { OAuthSecurityValidator } from './solution';

const config = {
  allowedRedirectUris: ['https://myapp.com/callback'],
  requirePKCE: true,
};

// Test1: Valid request passes validation
test('Test1', () => {
  const validator = new OAuthSecurityValidator(config);
  const result = validator.validateAuthRequest({
    responseType: 'code',
    clientId: 'client123',
    redirectUri: 'https://myapp.com/callback',
    state: 'randomstatevalue123',
    codeChallenge: 'challenge123',
    codeChallengeMethod: 'S256',
  });
  expect(result.valid).toBe(true);
});

// Test2: Missing state fails
test('Test2', () => {
  const validator = new OAuthSecurityValidator(config);
  const result = validator.validateAuthRequest({
    responseType: 'code',
    clientId: 'client123',
    redirectUri: 'https://myapp.com/callback',
    codeChallenge: 'challenge',
    codeChallengeMethod: 'S256',
  });
  expect(result.valid).toBe(false);
  expect(result.issues.some(i => i.includes('state'))).toBe(true);
});

// Test3: Invalid redirect URI fails
test('Test3', () => {
  const validator = new OAuthSecurityValidator(config);
  expect(validator.validateRedirectUri('https://evil.com/callback')).toBe(false);
});

// Test4: Valid redirect URI passes
test('Test4', () => {
  const validator = new OAuthSecurityValidator(config);
  expect(validator.validateRedirectUri('https://myapp.com/callback')).toBe(true);
});

// Test5: generateState creates unique values
test('Test5', () => {
  const validator = new OAuthSecurityValidator(config);
  const state1 = validator.generateState();
  const state2 = validator.generateState();
  expect(state1).not.toBe(state2);
  expect(state1.length).toBeGreaterThanOrEqual(32);
});

// Test6: validateState matches correctly
test('Test6', () => {
  const validator = new OAuthSecurityValidator(config);
  const state = validator.generateState();
  expect(validator.validateState(state, state)).toBe(true);
});

// Test7: validateState rejects mismatch
test('Test7', () => {
  const validator = new OAuthSecurityValidator(config);
  const state = validator.generateState();
  expect(validator.validateState(state, 'wrong')).toBe(false);
});

// Test8: detectAttacks finds missing PKCE
test('Test8', () => {
  const validator = new OAuthSecurityValidator(config);
  const result = validator.detectAttacks({
    responseType: 'code',
    clientId: 'client123',
    redirectUri: 'https://myapp.com/callback',
    state: 'state123',
  });
  expect(result.detected).toBe(true);
  expect(result.attacks).toContain('missing_pkce');
});

// Test9: detectAttacks finds implicit flow
test('Test9', () => {
  const validator = new OAuthSecurityValidator(config);
  const result = validator.detectAttacks({
    responseType: 'token',
    clientId: 'client123',
    redirectUri: 'https://myapp.com/callback',
    state: 'state123',
  });
  expect(result.attacks).toContain('deprecated_implicit_flow');
});

// Test10: generateCodeVerifier creates valid verifier
test('Test10', () => {
  const validator = new OAuthSecurityValidator(config);
  const verifier = validator.generateCodeVerifier();
  expect(verifier.length).toBeGreaterThanOrEqual(43);
});`,
	whyItMatters: `OAuth vulnerabilities have led to account takeovers at scale.

**Real-World OAuth Incidents:**

**1. Facebook (2018)**
\`\`\`
Attack: View As + OAuth token theft
Impact: 50 million access tokens stolen
Method: Chained bugs to extract tokens
Result: Largest Facebook breach
\`\`\`

**2. Slack (2017)**
\`\`\`
Attack: OAuth redirect manipulation
Impact: Could hijack Slack integrations
Method: Subdomain takeover + redirect bypass
Bounty: $3,000
\`\`\`

**3. Microsoft (2020)**
\`\`\`
Attack: OAuth app consent phishing
Impact: Corporate email compromise
Method: Malicious OAuth apps requesting permissions
Result: Widespread phishing campaigns
\`\`\`

**OAuth 2.0 Security Checklist:**

| Attack | Protection |
|--------|------------|
| CSRF | State parameter (required) |
| Code Interception | PKCE (required for public clients) |
| Token Leakage | Authorization Code flow |
| Open Redirect | Exact redirect URI matching |
| Replay | Short-lived codes, one-time use |

**Secure OAuth Implementation:**

\`\`\`typescript
// Authorization Request
// ✅ Secure
GET /authorize?
  response_type=code&
  client_id=CLIENT_ID&
  redirect_uri=https://app.com/callback&
  scope=read&
  state=RANDOM_STATE&
  code_challenge=CHALLENGE&
  code_challenge_method=S256

// ❌ Insecure (implicit flow)
GET /authorize?
  response_type=token&  // Token in URL fragment!
  client_id=CLIENT_ID&
  redirect_uri=https://app.com/callback

// Token Exchange
// ✅ Secure backend exchange
POST /token
Content-Type: application/x-www-form-urlencoded

grant_type=authorization_code&
code=AUTH_CODE&
redirect_uri=https://app.com/callback&
client_id=CLIENT_ID&
client_secret=SECRET&
code_verifier=VERIFIER

// State Validation
if (req.query.state !== session.oauthState) {
  throw new Error('CSRF detected');
}
\`\`\``,
	order: 1,
	translations: {
		ru: {
			title: 'Безопасность OAuth 2.0: Потоки и уязвимости',
			description: `Изучите безопасность OAuth 2.0 - отраслевой стандарт делегирования авторизации.

**Что такое OAuth 2.0?**

OAuth 2.0 - фреймворк авторизации, позволяющий сторонним приложениям получать ограниченный доступ к аккаунту пользователя.

**Ваша задача:**

Реализуйте класс \`OAuthSecurityValidator\`:

1. Валидация OAuth 2.0 запросов на проблемы безопасности
2. Генерация и валидация параметра state (защита от CSRF)
3. Валидация redirect URI по зарегистрированным паттернам
4. Обнаружение типичных OAuth атак`,
			hint1: `Для validateAuthRequest проверьте каждое обязательное поле: response_type должен быть 'code', redirect_uri в списке разрешённых, state должен существовать и быть достаточно длинным.`,
			hint2: `Для validateState используйте timing-safe сравнение (XOR каждого символа с накоплением). Это предотвращает timing атаки.`,
			whyItMatters: `Уязвимости OAuth привели к массовому захвату аккаунтов.`
		},
		uz: {
			title: 'OAuth 2.0 xavfsizligi: Oqimlar va zaifliklar',
			description: `OAuth 2.0 xavfsizligini o'rganing - avtorizatsiya delegatsiyasi uchun sanoat standarti.

**OAuth 2.0 nima?**

OAuth 2.0 - uchinchi tomon ilovalarga foydalanuvchi hisobiga cheklangan kirish imkonini beruvchi avtorizatsiya freymvorki.

**Sizning vazifangiz:**

\`OAuthSecurityValidator\` klassini amalga oshiring:

1. OAuth 2.0 so'rovlarini xavfsizlik muammolari uchun tasdiqlash
2. State parametrini yaratish va tasdiqlash (CSRF himoyasi)
3. Redirect URI ni ro'yxatdan o'tgan patternlarga qarshi tasdiqlash
4. Umumiy OAuth hujumlarini aniqlash`,
			hint1: `validateAuthRequest uchun har bir zarur maydonni tekshiring: response_type 'code' bo'lishi kerak, redirect_uri ruxsat etilgan ro'yxatda bo'lishi kerak.`,
			hint2: `validateState uchun timing-safe taqqoslashdan foydalaning (har bir belgini XOR qiling va yig'ing).`,
			whyItMatters: `OAuth zaifliklari keng miqyosda akkauntlarni egallab olishga olib keldi.`
		}
	}
};

export default task;
