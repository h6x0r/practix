import { Task } from '../../../../../../../types';

export const task: Task = {
	slug: 'auth-api-key-security',
	title: 'API Key Security Best Practices',
	difficulty: 'medium',
	tags: ['security', 'authentication', 'api-keys', 'typescript'],
	estimatedTime: '30m',
	isPremium: false,
	youtubeUrl: '',
	description: `Learn to securely generate, store, and validate API keys.

**API Keys vs Other Auth Methods:**

| Method | Use Case | Security Level |
|--------|----------|----------------|
| API Key | Server-to-server | Medium |
| OAuth 2.0 | User-delegated access | High |
| JWT | Stateless auth | High |
| Basic Auth | Simple internal | Low |

**API Key Best Practices:**

1. **Generation** - Cryptographically random, sufficient entropy
2. **Storage** - Hash keys, never store plain text
3. **Transmission** - HTTPS only, use headers not URLs
4. **Rotation** - Support key rotation without downtime
5. **Scoping** - Limit permissions per key
6. **Rate Limiting** - Prevent abuse
7. **Monitoring** - Log and alert on suspicious usage

**Key Format Options:**

\`\`\`
Prefix + Random:    sk_live_abc123xyz789...
UUID:               550e8400-e29b-41d4-a716-446655440000
Base64 encoded:     dGhpcyBpcyBhIHNlY3JldCBrZXk=
\`\`\`

**Your Task:**

Implement an \`APIKeyManager\` class for secure API key lifecycle management.`,
	initialCode: `interface APIKey {
  id: string;
  prefix: string;      // Visible part for identification
  hash: string;        // Hashed key for storage
  name: string;
  scopes: string[];
  createdAt: Date;
  expiresAt?: Date;
  lastUsedAt?: Date;
  rateLimit: number;   // Requests per minute
  isActive: boolean;
}

interface APIKeyValidation {
  valid: boolean;
  keyId?: string;
  scopes?: string[];
  reason?: string;
}

interface RateLimitStatus {
  allowed: boolean;
  remaining: number;
  resetAt: Date;
}

class APIKeyManager {
  private keys: Map<string, APIKey> = new Map();
  private usageCounters: Map<string, { count: number; resetAt: Date }> = new Map();

  generateKey(name: string, scopes: string[], options?: {
    expiresIn?: number; // Days
    rateLimit?: number; // Requests per minute
  }): { key: string; keyData: APIKey } {
    // TODO: Generate a new API key
    // Return both the raw key (to show once) and the stored data
    return { key: '', keyData: {} as APIKey };
  }

  private generateSecureRandom(length: number): string {
    // Generate cryptographically secure random string
    const chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789';
    let result = '';
    for (let i = 0; i < length; i++) {
      result += chars[Math.floor(Math.random() * chars.length)];
    }
    return result;
  }

  private hashKey(key: string): string {
    // Simple hash for demo (use bcrypt/argon2 in production)
    let hash = 0;
    for (let i = 0; i < key.length; i++) {
      const char = key.charCodeAt(i);
      hash = ((hash << 5) - hash) + char;
      hash = hash & hash;
    }
    return Math.abs(hash).toString(16).padStart(16, '0');
  }

  validateKey(rawKey: string): APIKeyValidation {
    // TODO: Validate an API key
    // Check: exists, not expired, is active
    return { valid: false };
  }

  checkRateLimit(keyId: string): RateLimitStatus {
    // TODO: Check if key is within rate limit
    return { allowed: false, remaining: 0, resetAt: new Date() };
  }

  recordUsage(keyId: string): void {
    // TODO: Record API key usage for rate limiting
  }

  revokeKey(keyId: string): boolean {
    // TODO: Revoke (deactivate) an API key
    return false;
  }

  rotateKey(keyId: string): { newKey: string; keyData: APIKey } | null {
    // TODO: Rotate key - generate new key with same settings
    return null;
  }

  hasScope(keyId: string, requiredScope: string): boolean {
    // TODO: Check if key has required scope
    return false;
  }

  getKeyByPrefix(prefix: string): APIKey | null {
    // TODO: Find key by its visible prefix
    return null;
  }

  cleanupExpiredKeys(): number {
    // TODO: Remove expired keys, return count removed
    return 0;
  }
}

export { APIKeyManager, APIKey, APIKeyValidation, RateLimitStatus };`,
	solutionCode: `interface APIKey {
  id: string;
  prefix: string;
  hash: string;
  name: string;
  scopes: string[];
  createdAt: Date;
  expiresAt?: Date;
  lastUsedAt?: Date;
  rateLimit: number;
  isActive: boolean;
}

interface APIKeyValidation {
  valid: boolean;
  keyId?: string;
  scopes?: string[];
  reason?: string;
}

interface RateLimitStatus {
  allowed: boolean;
  remaining: number;
  resetAt: Date;
}

class APIKeyManager {
  private keys: Map<string, APIKey> = new Map();
  private usageCounters: Map<string, { count: number; resetAt: Date }> = new Map();
  private keyHashIndex: Map<string, string> = new Map(); // hash -> keyId

  generateKey(name: string, scopes: string[], options?: {
    expiresIn?: number;
    rateLimit?: number;
  }): { key: string; keyData: APIKey } {
    const id = this.generateSecureRandom(8);
    const prefix = 'sk_' + this.generateSecureRandom(8);
    const secret = this.generateSecureRandom(32);
    const rawKey = prefix + '_' + secret;
    const hash = this.hashKey(rawKey);

    const keyData: APIKey = {
      id,
      prefix,
      hash,
      name,
      scopes,
      createdAt: new Date(),
      expiresAt: options?.expiresIn
        ? new Date(Date.now() + options.expiresIn * 24 * 60 * 60 * 1000)
        : undefined,
      rateLimit: options?.rateLimit || 60,
      isActive: true,
    };

    this.keys.set(id, keyData);
    this.keyHashIndex.set(hash, id);

    return { key: rawKey, keyData };
  }

  private generateSecureRandom(length: number): string {
    const chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789';
    let result = '';
    for (let i = 0; i < length; i++) {
      result += chars[Math.floor(Math.random() * chars.length)];
    }
    return result;
  }

  private hashKey(key: string): string {
    let hash = 0;
    for (let i = 0; i < key.length; i++) {
      const char = key.charCodeAt(i);
      hash = ((hash << 5) - hash) + char;
      hash = hash & hash;
    }
    return Math.abs(hash).toString(16).padStart(16, '0');
  }

  validateKey(rawKey: string): APIKeyValidation {
    const hash = this.hashKey(rawKey);
    const keyId = this.keyHashIndex.get(hash);

    if (!keyId) {
      return { valid: false, reason: 'Key not found' };
    }

    const key = this.keys.get(keyId);
    if (!key) {
      return { valid: false, reason: 'Key not found' };
    }

    if (!key.isActive) {
      return { valid: false, reason: 'Key has been revoked' };
    }

    if (key.expiresAt && key.expiresAt < new Date()) {
      return { valid: false, reason: 'Key has expired' };
    }

    // Update last used
    key.lastUsedAt = new Date();

    return {
      valid: true,
      keyId: key.id,
      scopes: key.scopes,
    };
  }

  checkRateLimit(keyId: string): RateLimitStatus {
    const key = this.keys.get(keyId);
    if (!key) {
      return { allowed: false, remaining: 0, resetAt: new Date() };
    }

    const now = new Date();
    let counter = this.usageCounters.get(keyId);

    // Reset counter if window has passed
    if (!counter || counter.resetAt <= now) {
      counter = {
        count: 0,
        resetAt: new Date(now.getTime() + 60000), // 1 minute window
      };
      this.usageCounters.set(keyId, counter);
    }

    const remaining = Math.max(0, key.rateLimit - counter.count);
    const allowed = remaining > 0;

    return { allowed, remaining, resetAt: counter.resetAt };
  }

  recordUsage(keyId: string): void {
    const status = this.checkRateLimit(keyId);
    const counter = this.usageCounters.get(keyId);

    if (counter) {
      counter.count++;
    }
  }

  revokeKey(keyId: string): boolean {
    const key = this.keys.get(keyId);
    if (!key) return false;

    key.isActive = false;
    return true;
  }

  rotateKey(keyId: string): { newKey: string; keyData: APIKey } | null {
    const oldKey = this.keys.get(keyId);
    if (!oldKey) return null;

    // Revoke old key
    oldKey.isActive = false;

    // Create new key with same settings
    return this.generateKey(oldKey.name, oldKey.scopes, {
      expiresIn: oldKey.expiresAt
        ? Math.ceil((oldKey.expiresAt.getTime() - Date.now()) / (24 * 60 * 60 * 1000))
        : undefined,
      rateLimit: oldKey.rateLimit,
    });
  }

  hasScope(keyId: string, requiredScope: string): boolean {
    const key = this.keys.get(keyId);
    if (!key) return false;

    // Support wildcard scopes
    if (key.scopes.includes('*')) return true;

    // Check exact match or prefix match (read:* matches read:users)
    return key.scopes.some(scope => {
      if (scope === requiredScope) return true;
      if (scope.endsWith(':*')) {
        const prefix = scope.slice(0, -1);
        return requiredScope.startsWith(prefix);
      }
      return false;
    });
  }

  getKeyByPrefix(prefix: string): APIKey | null {
    for (const key of this.keys.values()) {
      if (key.prefix === prefix) {
        return key;
      }
    }
    return null;
  }

  cleanupExpiredKeys(): number {
    const now = new Date();
    let removed = 0;

    for (const [id, key] of this.keys.entries()) {
      if (key.expiresAt && key.expiresAt < now) {
        this.keyHashIndex.delete(key.hash);
        this.keys.delete(id);
        this.usageCounters.delete(id);
        removed++;
      }
    }

    return removed;
  }
}

export { APIKeyManager, APIKey, APIKeyValidation, RateLimitStatus };`,
	hint1: `For generateKey, create a prefix (visible identifier) and secret part, concatenate them, hash the full key for storage, and store metadata with the hash (not the raw key).`,
	hint2: `For hasScope, support wildcard patterns like 'read:*' that match 'read:users', 'read:orders', etc.`,
	testCode: `import { APIKeyManager } from './solution';

// Test1: generateKey returns key and data
test('Test1', () => {
  const manager = new APIKeyManager();
  const { key, keyData } = manager.generateKey('test-key', ['read']);
  expect(key).toContain('sk_');
  expect(keyData.name).toBe('test-key');
  expect(keyData.isActive).toBe(true);
});

// Test2: validateKey returns valid for correct key
test('Test2', () => {
  const manager = new APIKeyManager();
  const { key } = manager.generateKey('test', ['read']);
  const result = manager.validateKey(key);
  expect(result.valid).toBe(true);
  expect(result.scopes).toContain('read');
});

// Test3: validateKey returns invalid for wrong key
test('Test3', () => {
  const manager = new APIKeyManager();
  manager.generateKey('test', ['read']);
  const result = manager.validateKey('wrong_key');
  expect(result.valid).toBe(false);
});

// Test4: revokeKey deactivates key
test('Test4', () => {
  const manager = new APIKeyManager();
  const { key, keyData } = manager.generateKey('test', ['read']);
  expect(manager.revokeKey(keyData.id)).toBe(true);
  const result = manager.validateKey(key);
  expect(result.valid).toBe(false);
  expect(result.reason).toContain('revoked');
});

// Test5: checkRateLimit tracks usage
test('Test5', () => {
  const manager = new APIKeyManager();
  const { keyData } = manager.generateKey('test', ['read'], { rateLimit: 2 });
  expect(manager.checkRateLimit(keyData.id).remaining).toBe(2);
  manager.recordUsage(keyData.id);
  expect(manager.checkRateLimit(keyData.id).remaining).toBe(1);
});

// Test6: hasScope with exact match
test('Test6', () => {
  const manager = new APIKeyManager();
  const { keyData } = manager.generateKey('test', ['read:users', 'write:orders']);
  expect(manager.hasScope(keyData.id, 'read:users')).toBe(true);
  expect(manager.hasScope(keyData.id, 'delete:users')).toBe(false);
});

// Test7: hasScope with wildcard
test('Test7', () => {
  const manager = new APIKeyManager();
  const { keyData } = manager.generateKey('admin', ['*']);
  expect(manager.hasScope(keyData.id, 'anything')).toBe(true);
});

// Test8: rotateKey creates new key
test('Test8', () => {
  const manager = new APIKeyManager();
  const { key: oldKey, keyData } = manager.generateKey('test', ['read']);
  const rotated = manager.rotateKey(keyData.id);
  expect(rotated).not.toBeNull();
  expect(rotated!.newKey).not.toBe(oldKey);
  expect(manager.validateKey(oldKey).valid).toBe(false);
  expect(manager.validateKey(rotated!.newKey).valid).toBe(true);
});

// Test9: getKeyByPrefix finds key
test('Test9', () => {
  const manager = new APIKeyManager();
  const { keyData } = manager.generateKey('test', ['read']);
  const found = manager.getKeyByPrefix(keyData.prefix);
  expect(found?.id).toBe(keyData.id);
});

// Test10: Expired key fails validation
test('Test10', () => {
  const manager = new APIKeyManager();
  const { key, keyData } = manager.generateKey('test', ['read'], { expiresIn: -1 });
  // Key is already expired
  const result = manager.validateKey(key);
  expect(result.valid).toBe(false);
  expect(result.reason).toContain('expired');
});`,
	whyItMatters: `API keys are everywhere - cloud providers, payment gateways, analytics services. Mishandling them leads to breaches.

**API Key Leaks:**

\`\`\`
Common leak sources:
- Committed to Git repositories
- Client-side JavaScript code
- Log files and error messages
- URL query parameters (visible in logs)
- Shared via email/chat

GitHub reports finding:
- 100,000+ API keys committed daily
- AWS, Google Cloud, Stripe most common
\`\`\`

**Defense Layers:**

1. **Generation**: High entropy (256 bits minimum)
2. **Storage**: Hash with bcrypt/argon2 (like passwords!)
3. **Transit**: Header only, HTTPS only
4. **Monitoring**: Alert on unusual patterns
5. **Rotation**: Automated, no downtime

**Key Design Pattern:**

\`\`\`
sk_xxxx_EXAMPLE_KEY_FORMAT_HERE
│   │    └─── 24+ char random secret
│   └─────── Environment (live/test)
└───────── Service identifier
\`\`\`

This format helps:
- Identify service quickly
- Distinguish prod/test
- Revoke by prefix if needed`,
	order: 5,
	translations: {
		ru: {
			title: 'Безопасность API ключей',
			description: `Научитесь безопасно генерировать, хранить и валидировать API ключи.

**API ключи vs Другие методы:**

| Метод | Применение | Безопасность |
|-------|------------|--------------|
| API Key | Server-to-server | Средняя |
| OAuth 2.0 | Делегированный доступ | Высокая |
| JWT | Stateless auth | Высокая |

**Best Practices:**

1. Генерация - криптографически случайные
2. Хранение - хешировать, не хранить в открытом виде
3. Передача - только HTTPS, только в заголовках
4. Ротация - поддержка без простоя

**Ваша задача:**

Реализуйте класс \`APIKeyManager\`.`,
			hint1: `Для generateKey создайте prefix (видимый идентификатор) и секретную часть, конкатенируйте их, хешируйте полный ключ для хранения.`,
			hint2: `Для hasScope поддержите wildcard паттерны типа 'read:*' которые соответствуют 'read:users', 'read:orders' и т.д.`,
			whyItMatters: `API ключи повсюду - облачные провайдеры, платёжные шлюзы, аналитика. Неправильное обращение ведёт к утечкам.`
		},
		uz: {
			title: 'API kalit xavfsizligi',
			description: `API kalitlarni xavfsiz yaratish, saqlash va tekshirishni o'rganing.

**API kalitlar vs Boshqa usullar:**

| Usul | Qo'llanilishi | Xavfsizlik |
|------|---------------|------------|
| API Key | Server-to-server | O'rtacha |
| OAuth 2.0 | Delegatsiya qilingan kirish | Yuqori |

**Sizning vazifangiz:**

\`APIKeyManager\` klassini amalga oshiring.`,
			hint1: `generateKey uchun prefix (ko'rinadigan identifikator) va sir qismini yarating, ularni birlashtiring, to'liq kalitni saqlash uchun xeshlang.`,
			hint2: `hasScope uchun 'read:users', 'read:orders' ga mos keladigan 'read:*' kabi wildcard patternlarni qo'llab-quvvatlang.`,
			whyItMatters: `API kalitlar hamma joyda - bulut provayderlar, to'lov shlyuzlari, analitika. Noto'g'ri boshqarish buzilishlarga olib keladi.`
		}
	}
};

export default task;
