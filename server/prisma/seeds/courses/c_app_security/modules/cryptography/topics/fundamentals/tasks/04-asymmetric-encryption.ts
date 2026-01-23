import { Task } from '../../../../../../../types';

export const task: Task = {
	slug: 'crypto-asymmetric-encryption',
	title: 'Asymmetric Encryption (RSA/ECC)',
	difficulty: 'hard',
	tags: ['security', 'cryptography', 'rsa', 'public-key', 'typescript'],
	estimatedTime: '40m',
	isPremium: false,
	youtubeUrl: '',
	description: `Learn asymmetric (public-key) cryptography - different keys for encryption and decryption.

**What is Asymmetric Encryption?**

Two mathematically related keys:
- **Public Key** - Share freely, used to encrypt
- **Private Key** - Keep secret, used to decrypt

\`\`\`
Alice encrypts with Bob's PUBLIC key
  ↓
Only Bob can decrypt with his PRIVATE key
\`\`\`

**RSA vs ECC:**

| Property | RSA | ECC |
|----------|-----|-----|
| Key Size (equiv.) | 2048 bits | 256 bits |
| Speed | Slower | Faster |
| Mobile | Heavy | Light |
| Quantum-safe | No | No |

**Common Uses:**

1. **Key Exchange** - Securely share symmetric keys
2. **Digital Signatures** - Sign with private, verify with public
3. **TLS Handshake** - Establish secure connections
4. **Email (PGP/S/MIME)** - End-to-end encryption

**Your Task:**

Implement a \`PublicKeyCrypto\` class that:

1. Generates key pairs
2. Encrypts/decrypts messages
3. Demonstrates hybrid encryption
4. Validates key pairs`,
	initialCode: `interface KeyPair {
  publicKey: string;
  privateKey: string;
  algorithm: string;
  keySize: number;
}

interface AsymmetricEncryption {
  ciphertext: string;
  encryptedKey?: string; // For hybrid encryption
  algorithm: string;
}

interface HybridEncryption {
  encryptedData: string;
  encryptedSymmetricKey: string;
  iv: string;
}

class PublicKeyCrypto {
  // Simulated RSA operations (NOT real RSA - for education)
  private simulateModPow(base: number, exp: number, mod: number): number {
    let result = 1;
    base = base % mod;
    while (exp > 0) {
      if (exp % 2 === 1) {
        result = (result * base) % mod;
      }
      exp = Math.floor(exp / 2);
      base = (base * base) % mod;
    }
    return result;
  }

  generateKeyPair(keySize: number = 2048): KeyPair {
    // TODO: Generate simulated public/private key pair
    // In reality, use proper crypto library
    return { publicKey: '', privateKey: '', algorithm: '', keySize: 0 };
  }

  encrypt(message: string, publicKey: string): AsymmetricEncryption {
    // TODO: Encrypt message with public key
    // Note: RSA can only encrypt small data (< key size)
    return { ciphertext: '', algorithm: '' };
  }

  decrypt(encrypted: AsymmetricEncryption, privateKey: string): string {
    // TODO: Decrypt with private key
    return '';
  }

  hybridEncrypt(plaintext: string, publicKey: string): HybridEncryption {
    // TODO: Hybrid encryption:
    // 1. Generate random symmetric key
    // 2. Encrypt data with symmetric key
    // 3. Encrypt symmetric key with public key
    return { encryptedData: '', encryptedSymmetricKey: '', iv: '' };
  }

  hybridDecrypt(encrypted: HybridEncryption, privateKey: string): string {
    // TODO: Hybrid decryption:
    // 1. Decrypt symmetric key with private key
    // 2. Decrypt data with symmetric key
    return '';
  }

  isValidKeyPair(publicKey: string, privateKey: string): boolean {
    // TODO: Verify that keys are a valid pair
    return false;
  }

  getMaxMessageSize(keySize: number): number {
    // TODO: Calculate max message size for direct RSA encryption
    // RSA can encrypt up to (keySize/8 - 11) bytes with PKCS#1 v1.5
    return 0;
  }

  derivePublicKey(privateKey: string): string {
    // TODO: Derive public key from private key
    return '';
  }
}

export { PublicKeyCrypto, KeyPair, AsymmetricEncryption, HybridEncryption };`,
	solutionCode: `interface KeyPair {
  publicKey: string;
  privateKey: string;
  algorithm: string;
  keySize: number;
}

interface AsymmetricEncryption {
  ciphertext: string;
  encryptedKey?: string;
  algorithm: string;
}

interface HybridEncryption {
  encryptedData: string;
  encryptedSymmetricKey: string;
  iv: string;
}

class PublicKeyCrypto {
  private simulateModPow(base: number, exp: number, mod: number): number {
    let result = 1;
    base = base % mod;
    while (exp > 0) {
      if (exp % 2 === 1) {
        result = (result * base) % mod;
      }
      exp = Math.floor(exp / 2);
      base = (base * base) % mod;
    }
    return result;
  }

  // Simulated key generation (uses small primes for demo)
  private generateSimulatedKeys(): { n: number; e: number; d: number } {
    // Small primes for simulation (real RSA uses 1024+ bit primes)
    const p = 61;
    const q = 53;
    const n = p * q; // 3233
    const phi = (p - 1) * (q - 1); // 3120
    const e = 17; // Common public exponent
    // d * e ≡ 1 (mod phi)
    const d = 2753; // Precomputed modular inverse

    return { n, e, d };
  }

  generateKeyPair(keySize: number = 2048): KeyPair {
    const { n, e, d } = this.generateSimulatedKeys();

    // Encode as simple strings (real keys are much more complex)
    const publicKey = JSON.stringify({ n, e, keySize });
    const privateKey = JSON.stringify({ n, d, keySize });

    return {
      publicKey,
      privateKey,
      algorithm: 'RSA',
      keySize,
    };
  }

  encrypt(message: string, publicKey: string): AsymmetricEncryption {
    const { n, e } = JSON.parse(publicKey);

    // Encrypt each character (simplified - real RSA uses padding)
    let ciphertext = '';
    for (let i = 0; i < message.length; i++) {
      const m = message.charCodeAt(i);
      const c = this.simulateModPow(m, e, n);
      ciphertext += c.toString(16).padStart(4, '0');
    }

    return {
      ciphertext,
      algorithm: 'RSA',
    };
  }

  decrypt(encrypted: AsymmetricEncryption, privateKey: string): string {
    const { n, d } = JSON.parse(privateKey);

    let plaintext = '';
    // Process 4 hex chars at a time
    for (let i = 0; i < encrypted.ciphertext.length; i += 4) {
      const c = parseInt(encrypted.ciphertext.substr(i, 4), 16);
      const m = this.simulateModPow(c, d, n);
      plaintext += String.fromCharCode(m);
    }

    return plaintext;
  }

  hybridEncrypt(plaintext: string, publicKey: string): HybridEncryption {
    // Generate random symmetric key
    const symmetricKey = this.generateRandomHex(32);
    const iv = this.generateRandomHex(32);

    // Encrypt data with symmetric key (XOR for simplicity)
    let encryptedData = '';
    for (let i = 0; i < plaintext.length; i++) {
      const keyByte = parseInt(symmetricKey.substr((i * 2) % 32, 2), 16);
      const plainByte = plaintext.charCodeAt(i);
      encryptedData += (plainByte ^ keyByte).toString(16).padStart(2, '0');
    }

    // Encrypt symmetric key with public key
    const keyEncrypted = this.encrypt(symmetricKey, publicKey);

    return {
      encryptedData,
      encryptedSymmetricKey: keyEncrypted.ciphertext,
      iv,
    };
  }

  hybridDecrypt(encrypted: HybridEncryption, privateKey: string): string {
    // Decrypt symmetric key
    const symmetricKey = this.decrypt(
      { ciphertext: encrypted.encryptedSymmetricKey, algorithm: 'RSA' },
      privateKey
    );

    // Decrypt data with symmetric key
    let plaintext = '';
    for (let i = 0; i < encrypted.encryptedData.length; i += 2) {
      const cipherByte = parseInt(encrypted.encryptedData.substr(i, 2), 16);
      const keyByte = parseInt(symmetricKey.substr(i % 32, 2), 16);
      plaintext += String.fromCharCode(cipherByte ^ keyByte);
    }

    return plaintext;
  }

  isValidKeyPair(publicKey: string, privateKey: string): boolean {
    try {
      const pub = JSON.parse(publicKey);
      const priv = JSON.parse(privateKey);

      // Keys must have same modulus
      if (pub.n !== priv.n) return false;

      // Test encryption/decryption
      const testMessage = 'test';
      const encrypted = this.encrypt(testMessage, publicKey);
      const decrypted = this.decrypt(encrypted, privateKey);

      return decrypted === testMessage;
    } catch {
      return false;
    }
  }

  getMaxMessageSize(keySize: number): number {
    // RSA with PKCS#1 v1.5 padding
    // Max = (keySize in bytes) - 11 bytes padding
    return Math.floor(keySize / 8) - 11;
  }

  derivePublicKey(privateKey: string): string {
    const priv = JSON.parse(privateKey);
    // Public key shares n, uses standard e=17 or e=65537
    return JSON.stringify({ n: priv.n, e: 17, keySize: priv.keySize });
  }

  private generateRandomHex(length: number): string {
    const chars = '0123456789abcdef';
    let result = '';
    for (let i = 0; i < length; i++) {
      result += chars[Math.floor(Math.random() * 16)];
    }
    return result;
  }
}

export { PublicKeyCrypto, KeyPair, AsymmetricEncryption, HybridEncryption };`,
	hint1: `For hybridEncrypt, generate a random symmetric key, encrypt the data with it (using XOR or similar), then encrypt the symmetric key with the public RSA key.`,
	hint2: `For getMaxMessageSize, RSA with PKCS#1 v1.5 can encrypt up to (keySize/8 - 11) bytes. For a 2048-bit key, that's 245 bytes.`,
	testCode: `import { PublicKeyCrypto } from './solution';

// Test1: generateKeyPair returns valid structure
test('Test1', () => {
  const crypto = new PublicKeyCrypto();
  const keyPair = crypto.generateKeyPair();
  expect(keyPair.publicKey).toBeTruthy();
  expect(keyPair.privateKey).toBeTruthy();
  expect(keyPair.algorithm).toBe('RSA');
});

// Test2: encrypt returns ciphertext
test('Test2', () => {
  const crypto = new PublicKeyCrypto();
  const { publicKey } = crypto.generateKeyPair();
  const encrypted = crypto.encrypt('Hi', publicKey);
  expect(encrypted.ciphertext).toBeTruthy();
  expect(encrypted.algorithm).toBe('RSA');
});

// Test3: decrypt recovers original message
test('Test3', () => {
  const crypto = new PublicKeyCrypto();
  const { publicKey, privateKey } = crypto.generateKeyPair();
  const encrypted = crypto.encrypt('Hello', publicKey);
  const decrypted = crypto.decrypt(encrypted, privateKey);
  expect(decrypted).toBe('Hello');
});

// Test4: hybridEncrypt returns all components
test('Test4', () => {
  const crypto = new PublicKeyCrypto();
  const { publicKey } = crypto.generateKeyPair();
  const encrypted = crypto.hybridEncrypt('Large message', publicKey);
  expect(encrypted.encryptedData).toBeTruthy();
  expect(encrypted.encryptedSymmetricKey).toBeTruthy();
  expect(encrypted.iv).toBeTruthy();
});

// Test5: hybridDecrypt recovers original
test('Test5', () => {
  const crypto = new PublicKeyCrypto();
  const { publicKey, privateKey } = crypto.generateKeyPair();
  const message = 'This is a longer message for hybrid encryption';
  const encrypted = crypto.hybridEncrypt(message, publicKey);
  const decrypted = crypto.hybridDecrypt(encrypted, privateKey);
  expect(decrypted).toBe(message);
});

// Test6: isValidKeyPair returns true for matching pair
test('Test6', () => {
  const crypto = new PublicKeyCrypto();
  const { publicKey, privateKey } = crypto.generateKeyPair();
  expect(crypto.isValidKeyPair(publicKey, privateKey)).toBe(true);
});

// Test7: isValidKeyPair returns false for mismatched
test('Test7', () => {
  const crypto = new PublicKeyCrypto();
  const pair1 = crypto.generateKeyPair();
  const pair2 = crypto.generateKeyPair();
  // Same implementation gives same keys, but test the validation
  expect(crypto.isValidKeyPair(pair1.publicKey, 'invalid')).toBe(false);
});

// Test8: getMaxMessageSize calculates correctly
test('Test8', () => {
  const crypto = new PublicKeyCrypto();
  expect(crypto.getMaxMessageSize(2048)).toBe(245);
  expect(crypto.getMaxMessageSize(1024)).toBe(117);
});

// Test9: derivePublicKey works
test('Test9', () => {
  const crypto = new PublicKeyCrypto();
  const { privateKey } = crypto.generateKeyPair();
  const derivedPublic = crypto.derivePublicKey(privateKey);
  expect(derivedPublic).toBeTruthy();
  expect(JSON.parse(derivedPublic).n).toBeTruthy();
});

// Test10: Cannot decrypt with wrong private key
test('Test10', () => {
  const crypto = new PublicKeyCrypto();
  const { publicKey } = crypto.generateKeyPair();
  const encrypted = crypto.encrypt('secret', publicKey);
  const wrongKey = JSON.stringify({ n: 9999, d: 1234, keySize: 2048 });
  const decrypted = crypto.decrypt(encrypted, wrongKey);
  expect(decrypted).not.toBe('secret');
});`,
	whyItMatters: `Asymmetric cryptography enables secure communication without pre-shared secrets.

**How TLS Uses Asymmetric Crypto:**

\`\`\`
1. Client → Server: ClientHello
2. Server → Client: Certificate (contains public key)
3. Client verifies certificate against CA
4. Client generates random pre-master secret
5. Client encrypts pre-master with server's public key
6. Server decrypts with private key
7. Both derive session keys from pre-master
8. Switch to fast symmetric encryption (AES)
\`\`\`

**Why Hybrid Encryption:**

\`\`\`
RSA 2048-bit: ~1,000 ops/second
AES-256:      ~1,000,000,000 ops/second

Solution: Use RSA to encrypt AES key,
then use AES for the actual data!
\`\`\`

**Key Size Comparison:**

| Security | RSA | ECC |
|----------|-----|-----|
| 80 bits | 1024 | 160 |
| 112 bits | 2048 | 224 |
| 128 bits | 3072 | 256 |
| 256 bits | 15360 | 512 |

ECC provides same security with much smaller keys!`,
	order: 3,
	translations: {
		ru: {
			title: 'Асимметричное шифрование (RSA/ECC)',
			description: `Изучите асимметричную криптографию с открытым ключом - разные ключи для шифрования и дешифрования.

**Что такое асимметричное шифрование?**

Два математически связанных ключа:
- **Открытый ключ** - можно распространять свободно, используется для шифрования
- **Закрытый ключ** - хранить в секрете, используется для расшифровки

**Применение:**

1. Обмен ключами
2. Цифровые подписи
3. TLS рукопожатие
4. Email шифрование (PGP)

**Ваша задача:**

Реализуйте класс \`PublicKeyCrypto\`.`,
			hint1: `Для hybridEncrypt сгенерируйте случайный симметричный ключ, зашифруйте данные им, затем зашифруйте симметричный ключ открытым RSA-ключом.`,
			hint2: `Для getMaxMessageSize: RSA с PKCS#1 v1.5 может зашифровать до (keySize/8 - 11) байт.`,
			whyItMatters: `Асимметричная криптография обеспечивает безопасную связь без предварительно разделённых секретов.`
		},
		uz: {
			title: 'Asimmetrik shifrlash (RSA/ECC)',
			description: `Asimmetrik (ochiq kalitli) kriptografiyani o'rganing - shifrlash va shifrni ochish uchun turli kalitlar.

**Asimmetrik shifrlash nima?**

Matematikaviy bog'langan ikkita kalit:
- **Ochiq kalit** - erkin tarqatish mumkin, shifrlash uchun
- **Yopiq kalit** - sir saqlang, shifrni ochish uchun

**Sizning vazifangiz:**

\`PublicKeyCrypto\` klassini amalga oshiring.`,
			hint1: `hybridEncrypt uchun tasodifiy simmetrik kalit yarating, ma'lumotlarni u bilan shifrlang, keyin simmetrik kalitni ochiq RSA kaliti bilan shifrlang.`,
			hint2: `getMaxMessageSize uchun: PKCS#1 v1.5 bilan RSA (keySize/8 - 11) baytgacha shifrlashi mumkin.`,
			whyItMatters: `Asimmetrik kriptografiya oldindan ulashilgan sirlar bo'lmasdan xavfsiz aloqani ta'minlaydi.`
		}
	}
};

export default task;
