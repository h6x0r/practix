import { Task } from '../../../../../../../types';

export const task: Task = {
	slug: 'sec-crypto-failures',
	title: 'Cryptographic Failures: Protecting Sensitive Data',
	difficulty: 'hard',
	tags: ['security', 'owasp', 'cryptography', 'encryption', 'typescript'],
	estimatedTime: '45m',
	isPremium: true,
	youtubeUrl: '',
	description: `Learn to prevent cryptographic failures - #2 in OWASP Top 10 2021 (formerly Sensitive Data Exposure).

**What are Cryptographic Failures?**

Cryptographic failures occur when sensitive data is not properly protected. This includes using weak algorithms, improper key management, missing encryption, or insecure protocols.

**Common Failures:**

1. **Weak Algorithms** - MD5, SHA1 for passwords, DES encryption
2. **No Encryption** - Sensitive data in plaintext
3. **Insecure Key Storage** - Keys in code or config files
4. **Protocol Downgrade** - Allowing HTTP, old TLS versions
5. **Missing Encryption in Transit** - Data over unencrypted channels

**Your Task:**

Implement a \`CryptoAuditor\` class that:

1. Detects usage of weak/deprecated cryptographic algorithms
2. Validates encryption configuration settings
3. Checks for secure password hashing practices
4. Identifies common cryptographic mistakes

**Example Usage:**

\`\`\`typescript
const auditor = new CryptoAuditor();

// Audit algorithm usage
auditor.auditAlgorithm('MD5');
// { secure: false, reason: 'MD5 is broken for security purposes' }

auditor.auditAlgorithm('SHA-256');
// { secure: true }

// Audit password hashing
auditor.auditPasswordHash('bcrypt', 12);
// { secure: true, strength: 'strong' }

auditor.auditPasswordHash('sha1', 1);
// { secure: false, reason: 'SHA1 unsuitable for passwords' }
\`\`\``,
	initialCode: `interface AlgorithmAuditResult {
  secure: boolean;
  reason?: string;
  recommendation?: string;
}

interface HashAuditResult {
  secure: boolean;
  strength: 'weak' | 'moderate' | 'strong';
  reason?: string;
  recommendation?: string;
}

interface TlsConfig {
  minVersion: string;
  cipherSuites: string[];
  allowHttp: boolean;
}

class CryptoAuditor {
  // Weak/deprecated algorithms
  private readonly WEAK_ALGORITHMS = {
    hash: ['MD5', 'SHA1', 'SHA-1'],
    encryption: ['DES', '3DES', 'RC4', 'RC2', 'BLOWFISH'],
    keyExchange: ['RSA-1024', 'DH-512', 'DH-1024'],
  };

  // Secure algorithms
  private readonly SECURE_ALGORITHMS = {
    hash: ['SHA-256', 'SHA-384', 'SHA-512', 'SHA-3'],
    encryption: ['AES-128', 'AES-256', 'ChaCha20'],
    passwordHash: ['bcrypt', 'argon2', 'scrypt', 'PBKDF2'],
  };

  auditAlgorithm(algorithm: string): AlgorithmAuditResult {
    // TODO: Check if algorithm is secure
    // Return result with reason if insecure
    return { secure: false };
  }

  auditPasswordHash(algorithm: string, costFactor: number): HashAuditResult {
    // TODO: Audit password hashing configuration
    // Check algorithm type and cost factor
    return { secure: false, strength: 'weak' };
  }

  auditTlsConfig(config: TlsConfig): { secure: boolean; issues: string[] } {
    // TODO: Audit TLS configuration
    // Check version, cipher suites, HTTP allowance
    return { secure: false, issues: [] };
  }

  detectWeakPatterns(code: string): string[] {
    // TODO: Detect weak crypto patterns in code
    // Look for MD5, weak random, hardcoded keys, etc.
    return [];
  }

  getSecureDefaults(): {
    hashAlgorithm: string;
    encryptionAlgorithm: string;
    keySize: number;
    passwordHashConfig: { algorithm: string; cost: number };
  } {
    // TODO: Return recommended secure defaults
    return {
      hashAlgorithm: '',
      encryptionAlgorithm: '',
      keySize: 0,
      passwordHashConfig: { algorithm: '', cost: 0 },
    };
  }
}

export { CryptoAuditor, AlgorithmAuditResult, HashAuditResult, TlsConfig };`,
	solutionCode: `interface AlgorithmAuditResult {
  secure: boolean;
  reason?: string;
  recommendation?: string;
}

interface HashAuditResult {
  secure: boolean;
  strength: 'weak' | 'moderate' | 'strong';
  reason?: string;
  recommendation?: string;
}

interface TlsConfig {
  minVersion: string;
  cipherSuites: string[];
  allowHttp: boolean;
}

class CryptoAuditor {
  // Weak/deprecated algorithms
  private readonly WEAK_ALGORITHMS = {
    hash: ['MD5', 'SHA1', 'SHA-1'],
    encryption: ['DES', '3DES', 'RC4', 'RC2', 'BLOWFISH'],
    keyExchange: ['RSA-1024', 'DH-512', 'DH-1024'],
  };

  // Secure algorithms
  private readonly SECURE_ALGORITHMS = {
    hash: ['SHA-256', 'SHA-384', 'SHA-512', 'SHA-3'],
    encryption: ['AES-128', 'AES-256', 'ChaCha20'],
    passwordHash: ['bcrypt', 'argon2', 'scrypt', 'PBKDF2'],
  };

  // Audit algorithm security
  auditAlgorithm(algorithm: string): AlgorithmAuditResult {
    const upper = algorithm.toUpperCase();

    // Check weak hash algorithms
    if (this.WEAK_ALGORITHMS.hash.some(a => upper === a.toUpperCase())) {
      return {
        secure: false,
        reason: \`\${algorithm} is cryptographically broken\`,
        recommendation: 'Use SHA-256 or SHA-3 for hashing',
      };
    }

    // Check weak encryption
    if (this.WEAK_ALGORITHMS.encryption.some(a => upper === a.toUpperCase())) {
      return {
        secure: false,
        reason: \`\${algorithm} has known vulnerabilities\`,
        recommendation: 'Use AES-256 or ChaCha20 for encryption',
      };
    }

    // Check weak key exchange
    if (this.WEAK_ALGORITHMS.keyExchange.some(a => upper.includes(a.toUpperCase()))) {
      return {
        secure: false,
        reason: \`\${algorithm} uses insufficient key size\`,
        recommendation: 'Use RSA-2048+ or ECDH with P-256+',
      };
    }

    // Check if secure algorithm
    const allSecure = [
      ...this.SECURE_ALGORITHMS.hash,
      ...this.SECURE_ALGORITHMS.encryption,
      ...this.SECURE_ALGORITHMS.passwordHash,
    ];

    if (allSecure.some(a => upper === a.toUpperCase())) {
      return { secure: true };
    }

    return {
      secure: false,
      reason: 'Algorithm not recognized or not recommended',
      recommendation: 'Use well-known, audited algorithms',
    };
  }

  // Audit password hashing configuration
  auditPasswordHash(algorithm: string, costFactor: number): HashAuditResult {
    const lower = algorithm.toLowerCase();

    // Check if using non-password hash
    if (['md5', 'sha1', 'sha-1', 'sha256', 'sha-256'].includes(lower)) {
      return {
        secure: false,
        strength: 'weak',
        reason: \`\${algorithm} is not suitable for password hashing\`,
        recommendation: 'Use bcrypt, argon2, or scrypt for passwords',
      };
    }

    // Check password-specific algorithms
    if (lower === 'bcrypt') {
      if (costFactor < 10) {
        return {
          secure: false,
          strength: 'weak',
          reason: 'Cost factor too low',
          recommendation: 'Use cost factor of 12 or higher',
        };
      }
      if (costFactor < 12) {
        return {
          secure: true,
          strength: 'moderate',
          recommendation: 'Consider increasing cost factor to 12+',
        };
      }
      return { secure: true, strength: 'strong' };
    }

    if (lower === 'argon2' || lower === 'argon2id') {
      if (costFactor < 3) {
        return {
          secure: false,
          strength: 'weak',
          reason: 'Iteration count too low for Argon2',
        };
      }
      return { secure: true, strength: 'strong' };
    }

    if (lower === 'scrypt') {
      return { secure: true, strength: costFactor >= 16384 ? 'strong' : 'moderate' };
    }

    if (lower === 'pbkdf2') {
      if (costFactor < 100000) {
        return {
          secure: false,
          strength: 'weak',
          reason: 'PBKDF2 requires at least 100,000 iterations',
        };
      }
      return { secure: true, strength: costFactor >= 600000 ? 'strong' : 'moderate' };
    }

    return {
      secure: false,
      strength: 'weak',
      reason: 'Unsupported password hashing algorithm',
    };
  }

  // Audit TLS configuration
  auditTlsConfig(config: TlsConfig): { secure: boolean; issues: string[] } {
    const issues: string[] = [];

    // Check minimum TLS version
    const insecureVersions = ['TLS1.0', 'TLSv1.0', 'TLS1.1', 'TLSv1.1', 'SSLv3', 'SSLv2'];
    if (insecureVersions.some(v => config.minVersion.toUpperCase() === v.toUpperCase())) {
      issues.push(\`TLS version \${config.minVersion} is deprecated. Use TLS 1.2+\`);
    }

    // Check cipher suites
    const weakCiphers = ['RC4', 'DES', '3DES', 'MD5', 'NULL', 'EXPORT', 'anon'];
    for (const cipher of config.cipherSuites) {
      if (weakCiphers.some(weak => cipher.toUpperCase().includes(weak))) {
        issues.push(\`Weak cipher suite: \${cipher}\`);
      }
    }

    // Check HTTP allowance
    if (config.allowHttp) {
      issues.push('HTTP (non-encrypted) traffic is allowed');
    }

    return {
      secure: issues.length === 0,
      issues,
    };
  }

  // Detect weak patterns in code
  detectWeakPatterns(code: string): string[] {
    const issues: string[] = [];

    // Check for weak algorithms in code
    if (/\bMD5\b/i.test(code)) {
      issues.push('MD5 usage detected - use SHA-256 or better');
    }
    if (/\bSHA1\b/i.test(code) || /sha-?1/i.test(code)) {
      issues.push('SHA1 usage detected - use SHA-256 or better');
    }

    // Check for Math.random() for security
    if (/Math\\.random\\(\\)/.test(code) && /password|token|key|secret/i.test(code)) {
      issues.push('Math.random() is not cryptographically secure');
    }

    // Check for hardcoded secrets
    if (/(?:password|secret|key|api_key)\\s*[=:]\\s*['"][^'"]{4,}/i.test(code)) {
      issues.push('Potential hardcoded secret detected');
    }

    // Check for base64 "encryption"
    if (/btoa\\(|atob\\(/.test(code) && /encrypt|secure|password/i.test(code)) {
      issues.push('Base64 is encoding, not encryption');
    }

    return issues;
  }

  // Return recommended secure defaults
  getSecureDefaults(): {
    hashAlgorithm: string;
    encryptionAlgorithm: string;
    keySize: number;
    passwordHashConfig: { algorithm: string; cost: number };
  } {
    return {
      hashAlgorithm: 'SHA-256',
      encryptionAlgorithm: 'AES-256-GCM',
      keySize: 256,
      passwordHashConfig: {
        algorithm: 'argon2id',
        cost: 3, // time cost for argon2
      },
    };
  }
}

export { CryptoAuditor, AlgorithmAuditResult, HashAuditResult, TlsConfig };`,
	hint1: `For auditAlgorithm, normalize the input to uppercase and check against WEAK_ALGORITHMS arrays. Return appropriate reason and recommendation for each category.`,
	hint2: `For auditPasswordHash, first reject non-password hashes (MD5, SHA1, SHA256). Then check algorithm-specific cost factors: bcrypt needs 12+, PBKDF2 needs 100k+ iterations.`,
	testCode: `import { CryptoAuditor } from './solution';

// Test1: MD5 detected as weak
test('Test1', () => {
  const auditor = new CryptoAuditor();
  const result = auditor.auditAlgorithm('MD5');
  expect(result.secure).toBe(false);
  expect(result.reason).toContain('broken');
});

// Test2: SHA-256 is secure
test('Test2', () => {
  const auditor = new CryptoAuditor();
  const result = auditor.auditAlgorithm('SHA-256');
  expect(result.secure).toBe(true);
});

// Test3: DES detected as weak encryption
test('Test3', () => {
  const auditor = new CryptoAuditor();
  const result = auditor.auditAlgorithm('DES');
  expect(result.secure).toBe(false);
});

// Test4: bcrypt with high cost is strong
test('Test4', () => {
  const auditor = new CryptoAuditor();
  const result = auditor.auditPasswordHash('bcrypt', 12);
  expect(result.secure).toBe(true);
  expect(result.strength).toBe('strong');
});

// Test5: SHA1 rejected for password hashing
test('Test5', () => {
  const auditor = new CryptoAuditor();
  const result = auditor.auditPasswordHash('sha1', 1);
  expect(result.secure).toBe(false);
  expect(result.reason).toContain('not suitable');
});

// Test6: bcrypt with low cost is weak
test('Test6', () => {
  const auditor = new CryptoAuditor();
  const result = auditor.auditPasswordHash('bcrypt', 8);
  expect(result.secure).toBe(false);
  expect(result.strength).toBe('weak');
});

// Test7: TLS 1.0 detected as insecure
test('Test7', () => {
  const auditor = new CryptoAuditor();
  const result = auditor.auditTlsConfig({
    minVersion: 'TLS1.0',
    cipherSuites: ['AES256-SHA'],
    allowHttp: false,
  });
  expect(result.secure).toBe(false);
  expect(result.issues.some(i => i.includes('deprecated'))).toBe(true);
});

// Test8: HTTP allowed is flagged
test('Test8', () => {
  const auditor = new CryptoAuditor();
  const result = auditor.auditTlsConfig({
    minVersion: 'TLS1.2',
    cipherSuites: ['AES256-GCM-SHA384'],
    allowHttp: true,
  });
  expect(result.issues.some(i => i.includes('HTTP'))).toBe(true);
});

// Test9: detectWeakPatterns finds MD5
test('Test9', () => {
  const auditor = new CryptoAuditor();
  const issues = auditor.detectWeakPatterns('const hash = crypto.MD5(data);');
  expect(issues.some(i => i.includes('MD5'))).toBe(true);
});

// Test10: getSecureDefaults returns argon2
test('Test10', () => {
  const auditor = new CryptoAuditor();
  const defaults = auditor.getSecureDefaults();
  expect(defaults.passwordHashConfig.algorithm).toBe('argon2id');
  expect(defaults.keySize).toBeGreaterThanOrEqual(256);
});`,
	whyItMatters: `Cryptographic failures expose sensitive data and have led to massive breaches.

**Major Cryptographic Failures:**

**1. Adobe (2013)**
\`\`\`
Impact: 153 million records
Failure: 3DES encryption with same key for all
Data: Passwords could be decoded
Cost: $1.2 million settlement + reputation damage
Why: Same key + ECB mode = patterns visible
\`\`\`

**2. Ashley Madison (2015)**
\`\`\`
Impact: 36 million accounts
Failure: Weak bcrypt implementation
Data: 15 million passwords cracked
Lesson: Cost factor matters (was too low)
\`\`\`

**3. Heartbleed (2014)**
\`\`\`
Impact: OpenSSL vulnerability
Failure: Memory leak exposed encryption keys
Data: Private keys, passwords in memory
Affected: 17% of all secure servers
\`\`\`

**Algorithm Security Cheat Sheet:**

| Purpose | Avoid | Use Instead |
|---------|-------|-------------|
| Hashing | MD5, SHA1 | SHA-256, SHA-3 |
| Passwords | SHA*, MD5 | bcrypt, Argon2 |
| Encryption | DES, 3DES, RC4 | AES-256-GCM |
| Key Exchange | RSA-1024, DH-1024 | ECDH P-256+, RSA-2048+ |
| TLS | 1.0, 1.1, SSLv3 | TLS 1.2, 1.3 |
| Random | Math.random() | crypto.randomBytes() |

**Secure Implementation Examples:**

\`\`\`typescript
// ❌ WRONG: Weak password hash
const hash = crypto.createHash('md5').update(password).digest('hex');

// ✅ RIGHT: Strong password hash
const hash = await bcrypt.hash(password, 12);

// ❌ WRONG: Insecure random
const token = Math.random().toString(36);

// ✅ RIGHT: Cryptographic random
const token = crypto.randomBytes(32).toString('hex');

// ❌ WRONG: ECB mode encryption
const cipher = crypto.createCipheriv('aes-256-ecb', key, '');

// ✅ RIGHT: GCM mode with IV
const iv = crypto.randomBytes(16);
const cipher = crypto.createCipheriv('aes-256-gcm', key, iv);

// ❌ WRONG: TLS 1.0
tls.createServer({ minVersion: 'TLSv1' });

// ✅ RIGHT: TLS 1.2+
tls.createServer({ minVersion: 'TLSv1.2' });
\`\`\`

**Key Principles:**
1. Use well-tested libraries (don't roll your own crypto)
2. Use appropriate algorithms for each use case
3. Use sufficient key sizes and work factors
4. Encrypt sensitive data at rest and in transit
5. Rotate keys periodically`,
	order: 6,
	translations: {
		ru: {
			title: 'Криптографические сбои: Защита конфиденциальных данных',
			description: `Научитесь предотвращать криптографические сбои - #2 в OWASP Top 10 2021.

**Что такое криптографические сбои?**

Криптографические сбои возникают при неправильной защите конфиденциальных данных: слабые алгоритмы, неправильное управление ключами, отсутствие шифрования.

**Ваша задача:**

Реализуйте класс \`CryptoAuditor\`:

1. Обнаружение слабых/устаревших криптографических алгоритмов
2. Валидация настроек шифрования
3. Проверка практик хеширования паролей
4. Выявление типичных криптографических ошибок`,
			hint1: `Для auditAlgorithm нормализуйте ввод в uppercase и проверьте против массивов WEAK_ALGORITHMS.`,
			hint2: `Для auditPasswordHash сначала отклоните non-password хеши (MD5, SHA1, SHA256). Затем проверьте cost factors: bcrypt нужно 12+, PBKDF2 нужно 100k+ итераций.`,
			whyItMatters: `Криптографические сбои раскрывают конфиденциальные данные и привели к масштабным утечкам.`
		},
		uz: {
			title: 'Kriptografik xatolar: Maxfiy ma\'lumotlarni himoya qilish',
			description: `Kriptografik xatolarni oldini olishni o'rganing - OWASP Top 10 2021 da #2.

**Kriptografik xatolar nima?**

Kriptografik xatolar maxfiy ma'lumotlar to'g'ri himoyalanmaganda yuz beradi: zaif algoritmlar, noto'g'ri kalit boshqaruvi, shifrlash yo'qligi.

**Sizning vazifangiz:**

\`CryptoAuditor\` klassini amalga oshiring:

1. Zaif/eskirgan kriptografik algoritmlarni aniqlash
2. Shifrlash sozlamalarini tasdiqlash
3. Parol xeshlash amaliyotlarini tekshirish
4. Umumiy kriptografik xatolarni aniqlash`,
			hint1: `auditAlgorithm uchun kirishni uppercase ga normalizatsiya qiling va WEAK_ALGORITHMS massivlariga qarshi tekshiring.`,
			hint2: `auditPasswordHash uchun avval parol bo'lmagan xeshlarni rad qiling (MD5, SHA1, SHA256).`,
			whyItMatters: `Kriptografik xatolar maxfiy ma'lumotlarni ochib beradi va katta buzilishlarga olib keldi.`
		}
	}
};

export default task;
