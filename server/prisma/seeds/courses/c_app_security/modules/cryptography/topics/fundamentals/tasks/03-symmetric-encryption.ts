import { Task } from '../../../../../../../types';

export const task: Task = {
	slug: 'crypto-symmetric-encryption',
	title: 'Symmetric Encryption (AES)',
	difficulty: 'medium',
	tags: ['security', 'cryptography', 'aes', 'encryption', 'typescript'],
	estimatedTime: '35m',
	isPremium: false,
	youtubeUrl: '',
	description: `Learn symmetric encryption where the same key is used for encryption and decryption.

**What is Symmetric Encryption?**

Same key encrypts and decrypts data. Fast but requires secure key exchange.

\`\`\`
Encrypt: plaintext + key → ciphertext
Decrypt: ciphertext + key → plaintext
\`\`\`

**AES (Advanced Encryption Standard):**

| Key Size | Security Level | Use Case |
|----------|---------------|----------|
| AES-128 | 128 bits | General purpose |
| AES-192 | 192 bits | Higher security |
| AES-256 | 256 bits | Top secret, compliance |

**Encryption Modes:**

| Mode | Description | Use Case |
|------|-------------|----------|
| ECB | Each block independent | NEVER use! |
| CBC | Blocks chained with IV | File encryption |
| CTR | Counter mode, streamable | Network traffic |
| GCM | Authenticated encryption | Recommended! |

**Your Task:**

Implement a \`SymmetricCrypto\` class that:

1. Simulates AES encryption/decryption
2. Manages initialization vectors (IVs)
3. Demonstrates why ECB mode is insecure
4. Implements authenticated encryption concepts`,
	initialCode: `interface EncryptionResult {
  ciphertext: string;
  iv: string;
  mode: string;
  authTag?: string;
}

interface DecryptionResult {
  plaintext: string;
  authenticated: boolean;
}

type EncryptionMode = 'ECB' | 'CBC' | 'GCM';

class SymmetricCrypto {
  private readonly KEY_SIZES = { 'AES-128': 16, 'AES-192': 24, 'AES-256': 32 };

  generateKey(bits: 128 | 192 | 256 = 256): string {
    // TODO: Generate random hex key of appropriate length
    return '';
  }

  generateIV(): string {
    // TODO: Generate 16-byte (32 hex chars) initialization vector
    return '';
  }

  // Simulated block cipher (NOT real AES - for educational purposes)
  private simulateBlockCipher(block: string, key: string, encrypt: boolean): string {
    let result = '';
    for (let i = 0; i < block.length; i++) {
      const keyChar = key.charCodeAt(i % key.length);
      const blockChar = block.charCodeAt(i);
      if (encrypt) {
        result += String.fromCharCode((blockChar + keyChar) % 256);
      } else {
        result += String.fromCharCode((blockChar - keyChar + 256) % 256);
      }
    }
    return result;
  }

  encrypt(plaintext: string, key: string, mode: EncryptionMode = 'GCM'): EncryptionResult {
    // TODO: Encrypt plaintext using specified mode
    // For ECB: encrypt each block independently
    // For CBC: XOR with previous ciphertext (or IV for first block)
    // For GCM: add authentication tag
    return { ciphertext: '', iv: '', mode: '' };
  }

  decrypt(encrypted: EncryptionResult, key: string): DecryptionResult {
    // TODO: Decrypt based on mode used
    return { plaintext: '', authenticated: false };
  }

  demonstrateECBProblem(repeatingData: string, key: string): { ecb: string; cbc: string } {
    // TODO: Show why ECB is insecure with repeating patterns
    return { ecb: '', cbc: '' };
  }

  isValidKey(key: string): boolean {
    // TODO: Check if key is valid hex and correct length
    return false;
  }

  rotateKey(oldKey: string, data: string): { newKey: string; reEncrypted: EncryptionResult } {
    // TODO: Generate new key and re-encrypt data
    return { newKey: '', reEncrypted: { ciphertext: '', iv: '', mode: '' } };
  }

  computeAuthTag(ciphertext: string, key: string): string {
    // TODO: Compute simple authentication tag (HMAC simulation)
    return '';
  }
}

export { SymmetricCrypto, EncryptionResult, DecryptionResult, EncryptionMode };`,
	solutionCode: `interface EncryptionResult {
  ciphertext: string;
  iv: string;
  mode: string;
  authTag?: string;
}

interface DecryptionResult {
  plaintext: string;
  authenticated: boolean;
}

type EncryptionMode = 'ECB' | 'CBC' | 'GCM';

class SymmetricCrypto {
  private readonly KEY_SIZES = { 'AES-128': 16, 'AES-192': 24, 'AES-256': 32 };

  generateKey(bits: 128 | 192 | 256 = 256): string {
    const bytes = bits / 8;
    const chars = '0123456789abcdef';
    let key = '';
    for (let i = 0; i < bytes * 2; i++) {
      key += chars[Math.floor(Math.random() * 16)];
    }
    return key;
  }

  generateIV(): string {
    const chars = '0123456789abcdef';
    let iv = '';
    for (let i = 0; i < 32; i++) {
      iv += chars[Math.floor(Math.random() * 16)];
    }
    return iv;
  }

  private simulateBlockCipher(block: string, key: string, encrypt: boolean): string {
    let result = '';
    for (let i = 0; i < block.length; i++) {
      const keyChar = key.charCodeAt(i % key.length);
      const blockChar = block.charCodeAt(i);
      if (encrypt) {
        result += String.fromCharCode((blockChar + keyChar) % 256);
      } else {
        result += String.fromCharCode((blockChar - keyChar + 256) % 256);
      }
    }
    return result;
  }

  private xorStrings(a: string, b: string): string {
    let result = '';
    for (let i = 0; i < a.length; i++) {
      result += String.fromCharCode(a.charCodeAt(i) ^ b.charCodeAt(i % b.length));
    }
    return result;
  }

  private toHex(str: string): string {
    let hex = '';
    for (let i = 0; i < str.length; i++) {
      hex += str.charCodeAt(i).toString(16).padStart(2, '0');
    }
    return hex;
  }

  private fromHex(hex: string): string {
    let str = '';
    for (let i = 0; i < hex.length; i += 2) {
      str += String.fromCharCode(parseInt(hex.substr(i, 2), 16));
    }
    return str;
  }

  encrypt(plaintext: string, key: string, mode: EncryptionMode = 'GCM'): EncryptionResult {
    const iv = this.generateIV();
    let ciphertext = '';

    if (mode === 'ECB') {
      // ECB: Each block encrypted independently (INSECURE!)
      ciphertext = this.toHex(this.simulateBlockCipher(plaintext, key, true));
    } else if (mode === 'CBC') {
      // CBC: XOR with previous ciphertext block
      let previousBlock = this.fromHex(iv);
      let result = '';
      for (let i = 0; i < plaintext.length; i += 16) {
        const block = plaintext.substr(i, 16).padEnd(16, '\\0');
        const xored = this.xorStrings(block, previousBlock);
        const encrypted = this.simulateBlockCipher(xored, key, true);
        result += encrypted;
        previousBlock = encrypted;
      }
      ciphertext = this.toHex(result);
    } else {
      // GCM: Like CTR with authentication
      let result = '';
      for (let i = 0; i < plaintext.length; i += 16) {
        const block = plaintext.substr(i, 16);
        const counter = iv + i.toString(16).padStart(8, '0');
        const keystream = this.simulateBlockCipher(counter, key, true);
        result += this.xorStrings(block, keystream);
      }
      ciphertext = this.toHex(result);
    }

    const result: EncryptionResult = { ciphertext, iv, mode };

    if (mode === 'GCM') {
      result.authTag = this.computeAuthTag(ciphertext, key);
    }

    return result;
  }

  decrypt(encrypted: EncryptionResult, key: string): DecryptionResult {
    const { ciphertext, iv, mode, authTag } = encrypted;

    // Verify auth tag for GCM
    if (mode === 'GCM') {
      const expectedTag = this.computeAuthTag(ciphertext, key);
      if (authTag !== expectedTag) {
        return { plaintext: '', authenticated: false };
      }
    }

    let plaintext = '';

    if (mode === 'ECB') {
      plaintext = this.simulateBlockCipher(this.fromHex(ciphertext), key, false);
    } else if (mode === 'CBC') {
      const ciphertextRaw = this.fromHex(ciphertext);
      let previousBlock = this.fromHex(iv);
      for (let i = 0; i < ciphertextRaw.length; i += 16) {
        const block = ciphertextRaw.substr(i, 16);
        const decrypted = this.simulateBlockCipher(block, key, false);
        plaintext += this.xorStrings(decrypted, previousBlock);
        previousBlock = block;
      }
    } else {
      const ciphertextRaw = this.fromHex(ciphertext);
      for (let i = 0; i < ciphertextRaw.length; i += 16) {
        const block = ciphertextRaw.substr(i, 16);
        const counter = iv + i.toString(16).padStart(8, '0');
        const keystream = this.simulateBlockCipher(counter, key, true);
        plaintext += this.xorStrings(block, keystream);
      }
    }

    return {
      plaintext: plaintext.replace(/\\0+$/, ''),
      authenticated: mode === 'GCM',
    };
  }

  demonstrateECBProblem(repeatingData: string, key: string): { ecb: string; cbc: string } {
    const ecbResult = this.encrypt(repeatingData, key, 'ECB');
    const cbcResult = this.encrypt(repeatingData, key, 'CBC');

    return {
      ecb: ecbResult.ciphertext,
      cbc: cbcResult.ciphertext,
    };
  }

  isValidKey(key: string): boolean {
    // Check hex format
    if (!/^[0-9a-fA-F]+$/.test(key)) {
      return false;
    }

    // Check valid length (16, 24, or 32 bytes = 32, 48, or 64 hex chars)
    const validLengths = [32, 48, 64];
    return validLengths.includes(key.length);
  }

  rotateKey(oldKey: string, data: string): { newKey: string; reEncrypted: EncryptionResult } {
    const newKey = this.generateKey(256);
    const reEncrypted = this.encrypt(data, newKey, 'GCM');

    return { newKey, reEncrypted };
  }

  computeAuthTag(ciphertext: string, key: string): string {
    // Simple HMAC-like authentication tag
    let tag = 0;
    const combined = ciphertext + key;
    for (let i = 0; i < combined.length; i++) {
      tag = ((tag << 5) - tag) + combined.charCodeAt(i);
      tag = tag & tag;
    }
    return Math.abs(tag).toString(16).padStart(16, '0');
  }
}

export { SymmetricCrypto, EncryptionResult, DecryptionResult, EncryptionMode };`,
	hint1: `For encrypt with CBC, XOR each plaintext block with the previous ciphertext block (or IV for the first block) before encrypting.`,
	hint2: `For GCM mode, compute an authentication tag using computeAuthTag and include it in the result. Verify it during decryption.`,
	testCode: `import { SymmetricCrypto } from './solution';

// Test1: generateKey produces correct length
test('Test1', () => {
  const crypto = new SymmetricCrypto();
  expect(crypto.generateKey(128).length).toBe(32);
  expect(crypto.generateKey(256).length).toBe(64);
});

// Test2: generateIV produces 32 hex chars
test('Test2', () => {
  const crypto = new SymmetricCrypto();
  const iv = crypto.generateIV();
  expect(iv.length).toBe(32);
  expect(/^[0-9a-f]+$/.test(iv)).toBe(true);
});

// Test3: encrypt returns EncryptionResult structure
test('Test3', () => {
  const crypto = new SymmetricCrypto();
  const key = crypto.generateKey();
  const result = crypto.encrypt('hello', key, 'GCM');
  expect(result.ciphertext).toBeTruthy();
  expect(result.iv.length).toBe(32);
  expect(result.mode).toBe('GCM');
});

// Test4: decrypt recovers original plaintext
test('Test4', () => {
  const crypto = new SymmetricCrypto();
  const key = crypto.generateKey();
  const plaintext = 'Secret message!';
  const encrypted = crypto.encrypt(plaintext, key, 'CBC');
  const decrypted = crypto.decrypt(encrypted, key);
  expect(decrypted.plaintext).toBe(plaintext);
});

// Test5: GCM mode includes auth tag
test('Test5', () => {
  const crypto = new SymmetricCrypto();
  const key = crypto.generateKey();
  const result = crypto.encrypt('test', key, 'GCM');
  expect(result.authTag).toBeTruthy();
});

// Test6: GCM decryption verifies authentication
test('Test6', () => {
  const crypto = new SymmetricCrypto();
  const key = crypto.generateKey();
  const encrypted = crypto.encrypt('data', key, 'GCM');
  const decrypted = crypto.decrypt(encrypted, key);
  expect(decrypted.authenticated).toBe(true);
});

// Test7: isValidKey validates correctly
test('Test7', () => {
  const crypto = new SymmetricCrypto();
  expect(crypto.isValidKey('a'.repeat(32))).toBe(true); // AES-128
  expect(crypto.isValidKey('a'.repeat(64))).toBe(true); // AES-256
  expect(crypto.isValidKey('xyz')).toBe(false);
  expect(crypto.isValidKey('gg'.repeat(32))).toBe(false); // Invalid hex
});

// Test8: demonstrateECBProblem shows pattern leakage
test('Test8', () => {
  const crypto = new SymmetricCrypto();
  const key = crypto.generateKey();
  const repeating = 'AAAAAAAAAAAAAAAA'.repeat(3); // Same 16-byte blocks
  const result = crypto.demonstrateECBProblem(repeating, key);
  expect(result.ecb).toBeTruthy();
  expect(result.cbc).toBeTruthy();
  // ECB will have repeating patterns, CBC won't
});

// Test9: rotateKey generates new key and re-encrypts
test('Test9', () => {
  const crypto = new SymmetricCrypto();
  const oldKey = crypto.generateKey();
  const result = crypto.rotateKey(oldKey, 'sensitive data');
  expect(result.newKey).not.toBe(oldKey);
  expect(result.reEncrypted.ciphertext).toBeTruthy();
});

// Test10: Wrong key fails to decrypt correctly
test('Test10', () => {
  const crypto = new SymmetricCrypto();
  const key1 = crypto.generateKey();
  const key2 = crypto.generateKey();
  const encrypted = crypto.encrypt('secret', key1, 'CBC');
  const decrypted = crypto.decrypt(encrypted, key2);
  expect(decrypted.plaintext).not.toBe('secret');
});`,
	whyItMatters: `Symmetric encryption protects data at rest and in transit. Understanding modes prevents catastrophic mistakes.

**The ECB Penguin:**

A famous demonstration shows why ECB mode is dangerous:

\`\`\`
Original Image:        ECB Encrypted:       CBC Encrypted:
  🐧 (penguin)          🐧 (visible!)        ░░░ (random)

ECB encrypts identical blocks identically,
so image patterns remain visible!
\`\`\`

**Real-World Symmetric Encryption:**

| Use Case | Algorithm | Mode |
|----------|-----------|------|
| Disk encryption | AES-256 | XTS |
| HTTPS traffic | AES-128 | GCM |
| Database encryption | AES-256 | GCM |
| File encryption | AES-256 | CBC (with HMAC) |

**Key Management Best Practices:**

1. Never hardcode keys in source code
2. Use key derivation functions (KDF) from passwords
3. Rotate keys periodically
4. Store keys in HSM or key vault
5. Different keys for different purposes`,
	order: 2,
	translations: {
		ru: {
			title: 'Симметричное шифрование (AES)',
			description: `Изучите симметричное шифрование, где один ключ используется для шифрования и дешифрования.

**Что такое симметричное шифрование?**

Один и тот же ключ шифрует и расшифровывает данные. Быстро, но требует безопасной передачи ключа.

**Режимы шифрования:**

| Режим | Описание | Применение |
|-------|----------|------------|
| ECB | Блоки независимы | НИКОГДА! |
| CBC | Блоки связаны через IV | Шифрование файлов |
| GCM | С аутентификацией | Рекомендуется! |

**Ваша задача:**

Реализуйте класс \`SymmetricCrypto\`.`,
			hint1: `Для encrypt с CBC выполните XOR каждого блока открытого текста с предыдущим блоком шифротекста (или IV для первого блока) перед шифрованием.`,
			hint2: `Для режима GCM вычислите тег аутентификации через computeAuthTag и включите его в результат.`,
			whyItMatters: `Симметричное шифрование защищает данные в покое и при передаче.`
		},
		uz: {
			title: 'Simmetrik shifrlash (AES)',
			description: `Simmetrik shifrlashni o'rganing - bir xil kalit shifrlash va shifrni ochish uchun ishlatiladi.

**Simmetrik shifrlash nima?**

Bir xil kalit ma'lumotlarni shifrlaydi va shifrini ochadi.

**Sizning vazifangiz:**

\`SymmetricCrypto\` klassini amalga oshiring.`,
			hint1: `CBC bilan encrypt uchun har bir ochiq matn blokini oldingi shifr bloki (yoki birinchi blok uchun IV) bilan XOR qiling.`,
			hint2: `GCM rejimi uchun computeAuthTag orqali autentifikatsiya tegini hisoblang va natijaga qo'shing.`,
			whyItMatters: `Simmetrik shifrlash ma'lumotlarni saqlashda va uzatishda himoya qiladi.`
		}
	}
};

export default task;
