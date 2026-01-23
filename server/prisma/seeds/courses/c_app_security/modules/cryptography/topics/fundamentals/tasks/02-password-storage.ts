import { Task } from '../../../../../../../types';

export const task: Task = {
	slug: 'crypto-password-storage',
	title: 'Secure Password Storage',
	difficulty: 'medium',
	tags: ['security', 'cryptography', 'passwords', 'bcrypt', 'typescript'],
	estimatedTime: '30m',
	isPremium: false,
	youtubeUrl: '',
	description: `Learn how to securely store passwords using modern hashing algorithms with salts.

**Why Not Plain Hashes?**

Plain hashes are vulnerable to:
- **Rainbow Tables** - Pre-computed hash dictionaries
- **GPU Cracking** - Billions of hashes per second
- **Same Password = Same Hash** - Reveals password reuse

**The Solution: Salt + Slow Hash**

\`\`\`
salt = random_bytes(16)
hash = slow_hash(password + salt, cost_factor)
store: $algorithm$cost$salt$hash
\`\`\`

**Password Hashing Algorithms:**

| Algorithm | Work Factor | Memory | Status |
|-----------|-------------|--------|--------|
| bcrypt | CPU-hard | Low | Recommended |
| scrypt | CPU+Mem | High | Good |
| Argon2id | CPU+Mem | Configurable | Best |
| PBKDF2 | CPU-hard | Low | Acceptable |

**Your Task:**

Implement a \`PasswordManager\` class that:

1. Generates secure random salts
2. Simulates bcrypt-style hashing
3. Verifies passwords securely
4. Enforces password strength rules`,
	initialCode: `interface PasswordHash {
  algorithm: string;
  cost: number;
  salt: string;
  hash: string;
  formatted: string; // $algorithm$cost$salt$hash
}

interface PasswordStrength {
  score: number; // 0-4
  feedback: string[];
  isStrong: boolean;
}

interface VerificationResult {
  isValid: boolean;
  needsRehash: boolean;
  algorithm: string;
}

class PasswordManager {
  private readonly MIN_COST = 10;
  private readonly DEFAULT_COST = 12;
  private readonly SALT_LENGTH = 16;

  generateSalt(length: number = this.SALT_LENGTH): string {
    // TODO: Generate random hex salt
    return '';
  }

  simulateSlowHash(password: string, salt: string, cost: number): string {
    // Simulated slow hash - in real app use bcrypt/argon2
    let hash = password + salt;
    const iterations = Math.pow(2, cost);

    for (let i = 0; i < Math.min(iterations, 10000); i++) {
      let temp = 0;
      for (let j = 0; j < hash.length; j++) {
        temp = ((temp << 5) - temp) + hash.charCodeAt(j);
      }
      hash = temp.toString(16).padStart(8, '0') + hash.slice(0, 24);
    }

    return hash.slice(0, 32);
  }

  hashPassword(password: string, cost: number = this.DEFAULT_COST): PasswordHash {
    // TODO: Hash password with salt and return formatted result
    return { algorithm: '', cost: 0, salt: '', hash: '', formatted: '' };
  }

  verifyPassword(password: string, storedHash: string): VerificationResult {
    // TODO: Parse stored hash and verify password
    return { isValid: false, needsRehash: false, algorithm: '' };
  }

  parseStoredHash(formatted: string): { algorithm: string; cost: number; salt: string; hash: string } | null {
    // TODO: Parse $algorithm$cost$salt$hash format
    return null;
  }

  checkPasswordStrength(password: string): PasswordStrength {
    // TODO: Evaluate password strength (length, complexity, common patterns)
    return { score: 0, feedback: [], isStrong: false };
  }

  needsRehash(storedHash: string, newCost: number = this.DEFAULT_COST): boolean {
    // TODO: Check if hash should be upgraded to new cost factor
    return false;
  }

  isCommonPassword(password: string): boolean {
    // TODO: Check against common password list
    return false;
  }
}

export { PasswordManager, PasswordHash, PasswordStrength, VerificationResult };`,
	solutionCode: `interface PasswordHash {
  algorithm: string;
  cost: number;
  salt: string;
  hash: string;
  formatted: string;
}

interface PasswordStrength {
  score: number;
  feedback: string[];
  isStrong: boolean;
}

interface VerificationResult {
  isValid: boolean;
  needsRehash: boolean;
  algorithm: string;
}

class PasswordManager {
  private readonly MIN_COST = 10;
  private readonly DEFAULT_COST = 12;
  private readonly SALT_LENGTH = 16;

  private readonly COMMON_PASSWORDS = [
    'password', '123456', '12345678', 'qwerty', 'abc123',
    'password1', 'admin', 'letmein', 'welcome', 'monkey'
  ];

  generateSalt(length: number = this.SALT_LENGTH): string {
    const chars = '0123456789abcdef';
    let salt = '';
    for (let i = 0; i < length * 2; i++) {
      salt += chars[Math.floor(Math.random() * chars.length)];
    }
    return salt;
  }

  simulateSlowHash(password: string, salt: string, cost: number): string {
    let hash = password + salt;
    const iterations = Math.pow(2, cost);

    for (let i = 0; i < Math.min(iterations, 10000); i++) {
      let temp = 0;
      for (let j = 0; j < hash.length; j++) {
        temp = ((temp << 5) - temp) + hash.charCodeAt(j);
      }
      hash = temp.toString(16).padStart(8, '0') + hash.slice(0, 24);
    }

    return hash.slice(0, 32);
  }

  hashPassword(password: string, cost: number = this.DEFAULT_COST): PasswordHash {
    const actualCost = Math.max(cost, this.MIN_COST);
    const salt = this.generateSalt();
    const hash = this.simulateSlowHash(password, salt, actualCost);

    return {
      algorithm: 'bcrypt',
      cost: actualCost,
      salt,
      hash,
      formatted: \`$bcrypt$\${actualCost}$\${salt}$\${hash}\`,
    };
  }

  verifyPassword(password: string, storedHash: string): VerificationResult {
    const parsed = this.parseStoredHash(storedHash);

    if (!parsed) {
      return { isValid: false, needsRehash: false, algorithm: 'unknown' };
    }

    const computedHash = this.simulateSlowHash(password, parsed.salt, parsed.cost);
    const isValid = computedHash === parsed.hash;

    return {
      isValid,
      needsRehash: this.needsRehash(storedHash),
      algorithm: parsed.algorithm,
    };
  }

  parseStoredHash(formatted: string): { algorithm: string; cost: number; salt: string; hash: string } | null {
    const parts = formatted.split('$').filter(p => p);

    if (parts.length !== 4) {
      return null;
    }

    const [algorithm, costStr, salt, hash] = parts;
    const cost = parseInt(costStr, 10);

    if (isNaN(cost)) {
      return null;
    }

    return { algorithm, cost, salt, hash };
  }

  checkPasswordStrength(password: string): PasswordStrength {
    const feedback: string[] = [];
    let score = 0;

    // Length check
    if (password.length >= 8) score++;
    else feedback.push('Use at least 8 characters');

    if (password.length >= 12) score++;
    else if (password.length >= 8) feedback.push('Consider using 12+ characters');

    // Complexity checks
    if (/[A-Z]/.test(password)) score += 0.5;
    else feedback.push('Add uppercase letters');

    if (/[a-z]/.test(password)) score += 0.5;
    else feedback.push('Add lowercase letters');

    if (/[0-9]/.test(password)) score += 0.5;
    else feedback.push('Add numbers');

    if (/[^A-Za-z0-9]/.test(password)) score += 0.5;
    else feedback.push('Add special characters');

    // Common password check
    if (this.isCommonPassword(password)) {
      score = 0;
      feedback.unshift('This is a commonly used password');
    }

    return {
      score: Math.min(4, Math.floor(score)),
      feedback,
      isStrong: score >= 3 && !this.isCommonPassword(password),
    };
  }

  needsRehash(storedHash: string, newCost: number = this.DEFAULT_COST): boolean {
    const parsed = this.parseStoredHash(storedHash);

    if (!parsed) return true;
    if (parsed.algorithm !== 'bcrypt') return true;
    if (parsed.cost < newCost) return true;

    return false;
  }

  isCommonPassword(password: string): boolean {
    return this.COMMON_PASSWORDS.includes(password.toLowerCase());
  }
}

export { PasswordManager, PasswordHash, PasswordStrength, VerificationResult };`,
	hint1: `For hashPassword, generate a salt, compute the hash using simulateSlowHash, and format as $bcrypt$cost$salt$hash.`,
	hint2: `For checkPasswordStrength, check length (8+, 12+), character classes (upper, lower, digits, special), and common passwords list.`,
	testCode: `import { PasswordManager } from './solution';

// Test1: generateSalt returns hex string of correct length
test('Test1', () => {
  const pm = new PasswordManager();
  const salt = pm.generateSalt(16);
  expect(salt.length).toBe(32);
  expect(/^[0-9a-f]+$/.test(salt)).toBe(true);
});

// Test2: hashPassword returns formatted hash
test('Test2', () => {
  const pm = new PasswordManager();
  const result = pm.hashPassword('mypassword');
  expect(result.formatted.startsWith('$bcrypt$')).toBe(true);
  expect(result.algorithm).toBe('bcrypt');
  expect(result.cost).toBeGreaterThanOrEqual(10);
});

// Test3: verifyPassword returns true for correct password
test('Test3', () => {
  const pm = new PasswordManager();
  const hashed = pm.hashPassword('secret123');
  const result = pm.verifyPassword('secret123', hashed.formatted);
  expect(result.isValid).toBe(true);
});

// Test4: verifyPassword returns false for wrong password
test('Test4', () => {
  const pm = new PasswordManager();
  const hashed = pm.hashPassword('secret123');
  const result = pm.verifyPassword('wrong', hashed.formatted);
  expect(result.isValid).toBe(false);
});

// Test5: parseStoredHash extracts components
test('Test5', () => {
  const pm = new PasswordManager();
  const parsed = pm.parseStoredHash('$bcrypt$12$abcdef1234567890$hashvalue123');
  expect(parsed?.algorithm).toBe('bcrypt');
  expect(parsed?.cost).toBe(12);
  expect(parsed?.salt).toBe('abcdef1234567890');
});

// Test6: checkPasswordStrength rates weak password low
test('Test6', () => {
  const pm = new PasswordManager();
  const result = pm.checkPasswordStrength('123');
  expect(result.score).toBeLessThan(2);
  expect(result.isStrong).toBe(false);
});

// Test7: checkPasswordStrength rates strong password high
test('Test7', () => {
  const pm = new PasswordManager();
  const result = pm.checkPasswordStrength('MyStr0ng!Pass#2024');
  expect(result.score).toBeGreaterThanOrEqual(3);
  expect(result.isStrong).toBe(true);
});

// Test8: isCommonPassword detects common passwords
test('Test8', () => {
  const pm = new PasswordManager();
  expect(pm.isCommonPassword('password')).toBe(true);
  expect(pm.isCommonPassword('123456')).toBe(true);
  expect(pm.isCommonPassword('xK9#mP2$qL')).toBe(false);
});

// Test9: needsRehash detects old cost factor
test('Test9', () => {
  const pm = new PasswordManager();
  expect(pm.needsRehash('$bcrypt$8$salt$hash', 12)).toBe(true);
  expect(pm.needsRehash('$bcrypt$12$salt$hash', 12)).toBe(false);
});

// Test10: Same password produces different hashes (salt)
test('Test10', () => {
  const pm = new PasswordManager();
  const hash1 = pm.hashPassword('samepassword');
  const hash2 = pm.hashPassword('samepassword');
  expect(hash1.formatted).not.toBe(hash2.formatted);
});`,
	whyItMatters: `Password storage mistakes have exposed billions of accounts. Get it right.

**Timeline of Password Breaches:**

\`\`\`
2012: LinkedIn - 117M SHA1 unsalted hashes leaked
      → 90% cracked within days

2013: Adobe - 153M passwords with 3DES (not hashing!)
      → Patterns revealed identical passwords

2016: Yahoo - 3B accounts, MD5 hashed
      → Largest breach in history

2019: Facebook - 600M passwords stored in PLAIN TEXT
      → Searchable by employees
\`\`\`

**Why bcrypt/Argon2 Work:**

\`\`\`
MD5:      1,000,000,000 hashes/second (GPU)
SHA256:     500,000,000 hashes/second (GPU)
bcrypt:           10,000 hashes/second (cost=12)
Argon2id:          1,000 hashes/second (tuned)

Time to crack "password123":
MD5:     < 1 second
bcrypt:  ~3 hours
Argon2:  ~28 hours
\`\`\`

**Password Storage Checklist:**

1. Use bcrypt, scrypt, or Argon2id
2. Cost factor ≥ 12 (adjust for hardware)
3. Never store plain text (even in logs!)
4. Implement rate limiting
5. Support password rehashing on login`,
	order: 1,
	translations: {
		ru: {
			title: 'Безопасное хранение паролей',
			description: `Научитесь безопасно хранить пароли с использованием современных алгоритмов хеширования с солью.

**Почему не простые хеши?**

Простые хеши уязвимы к:
- **Rainbow Tables** - предвычисленные словари хешей
- **GPU-взлом** - миллиарды хешей в секунду
- **Одинаковый пароль = одинаковый хеш**

**Решение: Соль + Медленный хеш**

**Ваша задача:**

Реализуйте класс \`PasswordManager\`.`,
			hint1: `Для hashPassword сгенерируйте соль, вычислите хеш через simulateSlowHash и отформатируйте как $bcrypt$cost$salt$hash.`,
			hint2: `Для checkPasswordStrength проверьте длину (8+, 12+), классы символов и список распространённых паролей.`,
			whyItMatters: `Ошибки в хранении паролей скомпрометировали миллиарды аккаунтов.`
		},
		uz: {
			title: 'Parollarni xavfsiz saqlash',
			description: `Zamonaviy xeshlash algoritmlari va tuz bilan parollarni xavfsiz saqlashni o'rganing.

**Nima uchun oddiy xeshlar emas?**

Oddiy xeshlar quyidagilarga zaif:
- **Rainbow Tables** - oldindan hisoblangan xesh lug'atlari
- **GPU buzish** - sekundiga milliardlab xeshlar

**Sizning vazifangiz:**

\`PasswordManager\` klassini amalga oshiring.`,
			hint1: `hashPassword uchun tuz yarating, simulateSlowHash orqali xesh hisoblang va $bcrypt$cost$salt$hash formatida qaytaring.`,
			hint2: `checkPasswordStrength uchun uzunlik (8+, 12+), belgilar sinflarini va keng tarqalgan parollar ro'yxatini tekshiring.`,
			whyItMatters: `Parollarni saqlashdagi xatolar milliardlab akkauntlarni buzilishiga olib keldi.`
		}
	}
};

export default task;
