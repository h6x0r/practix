import { Task } from '../../../../../../../types';

export const task: Task = {
	slug: 'crypto-hashing-basics',
	title: 'Cryptographic Hashing Fundamentals',
	difficulty: 'easy',
	tags: ['security', 'cryptography', 'hashing', 'typescript'],
	estimatedTime: '25m',
	isPremium: false,
	youtubeUrl: '',
	description: `Learn cryptographic hashing - one-way functions that convert data into fixed-size fingerprints.

**What is Hashing?**

A cryptographic hash function takes input of any size and produces a fixed-size output (digest). Key properties:

1. **Deterministic** - Same input always produces same output
2. **One-way** - Cannot reverse hash to get original input
3. **Collision-resistant** - Hard to find two inputs with same hash
4. **Avalanche effect** - Small input change = completely different hash

**Common Hash Algorithms:**

| Algorithm | Output Size | Status |
|-----------|-------------|--------|
| MD5 | 128 bits | BROKEN - Don't use! |
| SHA-1 | 160 bits | DEPRECATED |
| SHA-256 | 256 bits | Recommended |
| SHA-3 | 256/512 bits | Latest standard |

**Your Task:**

Implement a \`HashUtility\` class that:

1. Simulates hashing with different algorithms
2. Validates hash properties
3. Detects hash algorithm by output length
4. Implements hash comparison`,
	initialCode: `interface HashResult {
  algorithm: string;
  input: string;
  hash: string;
  length: number;
}

interface HashVerification {
  isValid: boolean;
  algorithm: string;
  timingAttackSafe: boolean;
}

class HashUtility {
  private hashLengths: Record<string, number> = {
    'md5': 32,
    'sha1': 40,
    'sha256': 64,
    'sha512': 128,
  };

  // Simulated hash function (in real world, use crypto library)
  private simulateHash(input: string, algorithm: string): string {
    // Simple simulation - NOT real cryptographic hash
    let hash = '';
    const length = this.hashLengths[algorithm] || 64;
    const chars = '0123456789abcdef';

    // Create deterministic pseudo-hash based on input
    let seed = 0;
    for (let i = 0; i < input.length; i++) {
      seed = ((seed << 5) - seed) + input.charCodeAt(i);
      seed = seed & seed;
    }

    for (let i = 0; i < length; i++) {
      seed = (seed * 1103515245 + 12345) & 0x7fffffff;
      hash += chars[seed % 16];
    }

    return hash;
  }

  hash(input: string, algorithm: string = 'sha256'): HashResult {
    // TODO: Generate hash result with algorithm, input, hash, and length
    return { algorithm: '', input: '', hash: '', length: 0 };
  }

  detectAlgorithm(hash: string): string | null {
    // TODO: Detect algorithm based on hash length
    return null;
  }

  verify(input: string, expectedHash: string, algorithm?: string): HashVerification {
    // TODO: Verify that input produces expectedHash
    // Use timing-safe comparison
    return { isValid: false, algorithm: '', timingAttackSafe: false };
  }

  timingSafeCompare(a: string, b: string): boolean {
    // TODO: Implement timing-safe string comparison
    // Should take same time regardless of where strings differ
    return false;
  }

  isValidHashFormat(hash: string): boolean {
    // TODO: Check if hash is valid hexadecimal of known length
    return false;
  }

  demonstrateAvalanche(input: string): { original: string; modified: string; similarity: number } {
    // TODO: Show avalanche effect - change one char and compare hashes
    return { original: '', modified: '', similarity: 0 };
  }
}

export { HashUtility, HashResult, HashVerification };`,
	solutionCode: `interface HashResult {
  algorithm: string;
  input: string;
  hash: string;
  length: number;
}

interface HashVerification {
  isValid: boolean;
  algorithm: string;
  timingAttackSafe: boolean;
}

class HashUtility {
  private hashLengths: Record<string, number> = {
    'md5': 32,
    'sha1': 40,
    'sha256': 64,
    'sha512': 128,
  };

  // Simulated hash function (in real world, use crypto library)
  private simulateHash(input: string, algorithm: string): string {
    let hash = '';
    const length = this.hashLengths[algorithm] || 64;
    const chars = '0123456789abcdef';

    let seed = 0;
    for (let i = 0; i < input.length; i++) {
      seed = ((seed << 5) - seed) + input.charCodeAt(i);
      seed = seed & seed;
    }

    for (let i = 0; i < length; i++) {
      seed = (seed * 1103515245 + 12345) & 0x7fffffff;
      hash += chars[seed % 16];
    }

    return hash;
  }

  hash(input: string, algorithm: string = 'sha256'): HashResult {
    const normalizedAlg = algorithm.toLowerCase();
    const hashValue = this.simulateHash(input, normalizedAlg);

    return {
      algorithm: normalizedAlg,
      input,
      hash: hashValue,
      length: hashValue.length,
    };
  }

  detectAlgorithm(hash: string): string | null {
    const length = hash.length;

    for (const [alg, len] of Object.entries(this.hashLengths)) {
      if (len === length) {
        return alg;
      }
    }

    return null;
  }

  verify(input: string, expectedHash: string, algorithm?: string): HashVerification {
    const detectedAlg = algorithm || this.detectAlgorithm(expectedHash);

    if (!detectedAlg) {
      return { isValid: false, algorithm: 'unknown', timingAttackSafe: true };
    }

    const computedHash = this.simulateHash(input, detectedAlg);
    const isValid = this.timingSafeCompare(computedHash, expectedHash);

    return {
      isValid,
      algorithm: detectedAlg,
      timingAttackSafe: true,
    };
  }

  timingSafeCompare(a: string, b: string): boolean {
    if (a.length !== b.length) {
      return false;
    }

    let result = 0;
    for (let i = 0; i < a.length; i++) {
      result |= a.charCodeAt(i) ^ b.charCodeAt(i);
    }

    return result === 0;
  }

  isValidHashFormat(hash: string): boolean {
    // Check if valid hex string
    if (!/^[0-9a-fA-F]+$/.test(hash)) {
      return false;
    }

    // Check if known length
    const knownLengths = Object.values(this.hashLengths);
    return knownLengths.includes(hash.length);
  }

  demonstrateAvalanche(input: string): { original: string; modified: string; similarity: number } {
    const originalHash = this.simulateHash(input, 'sha256');

    // Modify one character
    const modifiedInput = input.slice(0, -1) + String.fromCharCode(input.charCodeAt(input.length - 1) + 1);
    const modifiedHash = this.simulateHash(modifiedInput, 'sha256');

    // Calculate similarity (should be ~50% for good hash)
    let matchingChars = 0;
    for (let i = 0; i < originalHash.length; i++) {
      if (originalHash[i] === modifiedHash[i]) {
        matchingChars++;
      }
    }

    const similarity = (matchingChars / originalHash.length) * 100;

    return {
      original: originalHash,
      modified: modifiedHash,
      similarity: Math.round(similarity),
    };
  }
}

export { HashUtility, HashResult, HashVerification };`,
	hint1: `For timingSafeCompare, use XOR on each character and OR the results. This ensures comparison takes constant time.`,
	hint2: `For detectAlgorithm, check the hash length against known lengths: MD5=32, SHA1=40, SHA256=64, SHA512=128.`,
	testCode: `import { HashUtility } from './solution';

// Test1: hash returns HashResult with correct structure
test('Test1', () => {
  const util = new HashUtility();
  const result = util.hash('test', 'sha256');
  expect(result.algorithm).toBe('sha256');
  expect(result.input).toBe('test');
  expect(result.hash.length).toBe(64);
});

// Test2: hash is deterministic
test('Test2', () => {
  const util = new HashUtility();
  const hash1 = util.hash('password', 'sha256');
  const hash2 = util.hash('password', 'sha256');
  expect(hash1.hash).toBe(hash2.hash);
});

// Test3: detectAlgorithm identifies MD5
test('Test3', () => {
  const util = new HashUtility();
  const md5Hash = 'd41d8cd98f00b204e9800998ecf8427e'; // 32 chars
  expect(util.detectAlgorithm(md5Hash)).toBe('md5');
});

// Test4: detectAlgorithm identifies SHA256
test('Test4', () => {
  const util = new HashUtility();
  const sha256Hash = 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855';
  expect(util.detectAlgorithm(sha256Hash)).toBe('sha256');
});

// Test5: verify returns true for matching hash
test('Test5', () => {
  const util = new HashUtility();
  const result = util.hash('secret', 'sha256');
  const verification = util.verify('secret', result.hash, 'sha256');
  expect(verification.isValid).toBe(true);
  expect(verification.timingAttackSafe).toBe(true);
});

// Test6: verify returns false for non-matching
test('Test6', () => {
  const util = new HashUtility();
  const result = util.hash('secret', 'sha256');
  const verification = util.verify('wrong', result.hash, 'sha256');
  expect(verification.isValid).toBe(false);
});

// Test7: timingSafeCompare works correctly
test('Test7', () => {
  const util = new HashUtility();
  expect(util.timingSafeCompare('abc', 'abc')).toBe(true);
  expect(util.timingSafeCompare('abc', 'abd')).toBe(false);
  expect(util.timingSafeCompare('abc', 'ab')).toBe(false);
});

// Test8: isValidHashFormat validates correctly
test('Test8', () => {
  const util = new HashUtility();
  expect(util.isValidHashFormat('d41d8cd98f00b204e9800998ecf8427e')).toBe(true);
  expect(util.isValidHashFormat('not-a-hash')).toBe(false);
  expect(util.isValidHashFormat('xyz')).toBe(false);
});

// Test9: demonstrateAvalanche shows different hashes
test('Test9', () => {
  const util = new HashUtility();
  const result = util.demonstrateAvalanche('hello');
  expect(result.original).not.toBe(result.modified);
  expect(typeof result.similarity).toBe('number');
});

// Test10: Different inputs produce different hashes
test('Test10', () => {
  const util = new HashUtility();
  const hash1 = util.hash('input1', 'sha256');
  const hash2 = util.hash('input2', 'sha256');
  expect(hash1.hash).not.toBe(hash2.hash);
});`,
	whyItMatters: `Hashing is fundamental to security - password storage, data integrity, digital signatures all rely on it.

**Password Storage Evolution:**

\`\`\`
1970s: Plain text passwords → Easily stolen
1980s: Simple encryption → Key compromise = all passwords
1990s: MD5/SHA1 hashing → Rainbow tables, GPU cracking
2000s: Salted hashes → Still fast to crack
2010s: bcrypt/scrypt/Argon2 → Purpose-built, slow by design
\`\`\`

**Real-World Breaches:**

| Company | Year | Issue | Passwords Exposed |
|---------|------|-------|-------------------|
| LinkedIn | 2012 | SHA1 unsalted | 117 million |
| Adobe | 2013 | 3DES encrypted | 153 million |
| Dropbox | 2012 | SHA1 + bcrypt | 68 million |

**Timing Attacks:**

\`\`\`typescript
// VULNERABLE - early return leaks information
function badCompare(a: string, b: string): boolean {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) {
    if (a[i] !== b[i]) return false; // Early return!
  }
  return true;
}

// SAFE - constant time comparison
function safeCompare(a: string, b: string): boolean {
  if (a.length !== b.length) return false;
  let result = 0;
  for (let i = 0; i < a.length; i++) {
    result |= a.charCodeAt(i) ^ b.charCodeAt(i);
  }
  return result === 0;
}
\`\`\`

Attackers can measure response time to guess characters one by one!`,
	order: 0,
	translations: {
		ru: {
			title: 'Основы криптографического хеширования',
			description: `Изучите криптографическое хеширование - односторонние функции, превращающие данные в "отпечатки" фиксированного размера.

**Что такое хеширование?**

Криптографическая хеш-функция принимает вход любого размера и выдаёт выход фиксированного размера. Ключевые свойства:

1. **Детерминированность** - одинаковый вход = одинаковый выход
2. **Односторонность** - нельзя восстановить вход по хешу
3. **Устойчивость к коллизиям** - сложно найти два входа с одинаковым хешем
4. **Лавинный эффект** - малое изменение входа = совершенно другой хеш

**Ваша задача:**

Реализуйте класс \`HashUtility\`.`,
			hint1: `Для timingSafeCompare используйте XOR для каждого символа и OR для накопления результата. Это обеспечивает постоянное время сравнения.`,
			hint2: `Для detectAlgorithm проверьте длину хеша: MD5=32, SHA1=40, SHA256=64, SHA512=128.`,
			whyItMatters: `Хеширование - основа безопасности: хранение паролей, целостность данных, цифровые подписи.`
		},
		uz: {
			title: 'Kriptografik xeshlash asoslari',
			description: `Kriptografik xeshlashni o'rganing - ma'lumotlarni belgilangan o'lchamdagi "barmoq izlari"ga aylantiradigan bir tomonlama funksiyalar.

**Xeshlash nima?**

Kriptografik xesh funksiyasi har qanday o'lchamdagi kirishni oladi va belgilangan o'lchamdagi chiqishni beradi.

**Sizning vazifangiz:**

\`HashUtility\` klassini amalga oshiring.`,
			hint1: `timingSafeCompare uchun har bir belgi uchun XOR va natijani yig'ish uchun OR dan foydalaning.`,
			hint2: `detectAlgorithm uchun xesh uzunligini tekshiring: MD5=32, SHA1=40, SHA256=64, SHA512=128.`,
			whyItMatters: `Xeshlash xavfsizlik asosi: parollarni saqlash, ma'lumotlar yaxlitligi, raqamli imzolar.`
		}
	}
};

export default task;
