import { Task } from '../../../../../../../types';

export const task: Task = {
	slug: 'crypto-digital-signatures',
	title: 'Digital Signatures',
	difficulty: 'medium',
	tags: ['security', 'cryptography', 'signatures', 'authentication', 'typescript'],
	estimatedTime: '30m',
	isPremium: false,
	youtubeUrl: '',
	description: `Learn digital signatures - cryptographic proof of authenticity and integrity.

**What are Digital Signatures?**

Digital signatures use asymmetric cryptography in reverse:
- **Sign** with PRIVATE key
- **Verify** with PUBLIC key

\`\`\`
Message + Private Key → Signature
Message + Signature + Public Key → Valid/Invalid
\`\`\`

**Properties:**

1. **Authenticity** - Proves who signed it
2. **Integrity** - Detects any tampering
3. **Non-repudiation** - Signer cannot deny signing

**Signature Process:**

\`\`\`
1. Hash the message (SHA-256)
2. Encrypt hash with private key → Signature
3. Send message + signature
4. Receiver hashes message
5. Decrypts signature with public key
6. Compare hashes → Verified!
\`\`\`

**Your Task:**

Implement a \`DigitalSignature\` class that:

1. Signs messages with private key
2. Verifies signatures with public key
3. Detects tampered messages
4. Supports timestamped signatures`,
	initialCode: `interface Signature {
  value: string;
  algorithm: string;
  timestamp?: number;
}

interface SignedMessage {
  message: string;
  signature: Signature;
  publicKey: string;
}

interface VerificationResult {
  isValid: boolean;
  reason: string;
  signedAt?: Date;
}

class DigitalSignature {
  // Simulated hash function
  private hash(data: string): string {
    let hash = 0;
    for (let i = 0; i < data.length; i++) {
      const char = data.charCodeAt(i);
      hash = ((hash << 5) - hash) + char;
      hash = hash & hash;
    }
    return Math.abs(hash).toString(16).padStart(16, '0');
  }

  // Simulated RSA operations
  private simulateEncrypt(data: string, key: string): string {
    const keyNum = parseInt(key, 16) || 12345;
    let result = '';
    for (let i = 0; i < data.length; i++) {
      const charCode = data.charCodeAt(i);
      result += ((charCode * keyNum) % 65536).toString(16).padStart(4, '0');
    }
    return result;
  }

  private simulateDecrypt(data: string, key: string): string {
    const keyNum = parseInt(key, 16) || 12345;
    let result = '';
    for (let i = 0; i < data.length; i += 4) {
      const num = parseInt(data.substr(i, 4), 16);
      // Find original char (simplified inverse)
      for (let c = 0; c < 256; c++) {
        if ((c * keyNum) % 65536 === num) {
          result += String.fromCharCode(c);
          break;
        }
      }
    }
    return result;
  }

  generateKeyPair(): { publicKey: string; privateKey: string } {
    // TODO: Generate simulated key pair
    return { publicKey: '', privateKey: '' };
  }

  sign(message: string, privateKey: string, includeTimestamp: boolean = false): Signature {
    // TODO: Create digital signature
    // 1. Hash the message
    // 2. Encrypt hash with private key
    return { value: '', algorithm: '' };
  }

  verify(message: string, signature: Signature, publicKey: string): VerificationResult {
    // TODO: Verify signature
    // 1. Hash the message
    // 2. Decrypt signature with public key
    // 3. Compare hashes
    return { isValid: false, reason: '' };
  }

  createSignedMessage(message: string, privateKey: string, publicKey: string): SignedMessage {
    // TODO: Create complete signed message package
    return { message: '', signature: { value: '', algorithm: '' }, publicKey: '' };
  }

  verifySignedMessage(signedMessage: SignedMessage): VerificationResult {
    // TODO: Verify a complete signed message
    return { isValid: false, reason: '' };
  }

  isTimestampValid(signature: Signature, maxAgeSeconds: number): boolean {
    // TODO: Check if signature timestamp is within acceptable range
    return false;
  }

  detectTampering(original: string, modified: string, signature: Signature, publicKey: string): {
    tampered: boolean;
    originalValid: boolean;
    modifiedValid: boolean;
  } {
    // TODO: Compare verification of original vs modified message
    return { tampered: false, originalValid: false, modifiedValid: false };
  }
}

export { DigitalSignature, Signature, SignedMessage, VerificationResult };`,
	solutionCode: `interface Signature {
  value: string;
  algorithm: string;
  timestamp?: number;
}

interface SignedMessage {
  message: string;
  signature: Signature;
  publicKey: string;
}

interface VerificationResult {
  isValid: boolean;
  reason: string;
  signedAt?: Date;
}

class DigitalSignature {
  private hash(data: string): string {
    let hash = 0;
    for (let i = 0; i < data.length; i++) {
      const char = data.charCodeAt(i);
      hash = ((hash << 5) - hash) + char;
      hash = hash & hash;
    }
    return Math.abs(hash).toString(16).padStart(16, '0');
  }

  private simulateEncrypt(data: string, key: string): string {
    const keyNum = parseInt(key, 16) || 12345;
    let result = '';
    for (let i = 0; i < data.length; i++) {
      const charCode = data.charCodeAt(i);
      result += ((charCode * keyNum) % 65536).toString(16).padStart(4, '0');
    }
    return result;
  }

  private simulateDecrypt(data: string, key: string): string {
    const keyNum = parseInt(key, 16) || 12345;
    let result = '';
    for (let i = 0; i < data.length; i += 4) {
      const num = parseInt(data.substr(i, 4), 16);
      for (let c = 0; c < 256; c++) {
        if ((c * keyNum) % 65536 === num) {
          result += String.fromCharCode(c);
          break;
        }
      }
    }
    return result;
  }

  generateKeyPair(): { publicKey: string; privateKey: string } {
    // Simulated key pair (related mathematically)
    const base = Math.floor(Math.random() * 65535) + 1;
    const privateKey = base.toString(16).padStart(8, '0');
    const publicKey = ((base * 7919) % 65536).toString(16).padStart(8, '0'); // Related

    return { publicKey, privateKey };
  }

  sign(message: string, privateKey: string, includeTimestamp: boolean = false): Signature {
    // 1. Hash the message
    const messageHash = this.hash(message);

    // 2. Encrypt hash with private key (this is the signature)
    const signatureValue = this.simulateEncrypt(messageHash, privateKey);

    const signature: Signature = {
      value: signatureValue,
      algorithm: 'RSA-SHA256',
    };

    if (includeTimestamp) {
      signature.timestamp = Date.now();
    }

    return signature;
  }

  verify(message: string, signature: Signature, publicKey: string): VerificationResult {
    // 1. Hash the message
    const messageHash = this.hash(message);

    // 2. Decrypt signature with public key
    const decryptedHash = this.simulateDecrypt(signature.value, publicKey);

    // 3. Compare hashes
    const isValid = messageHash === decryptedHash;

    const result: VerificationResult = {
      isValid,
      reason: isValid ? 'Signature verified successfully' : 'Signature verification failed - message may be tampered',
    };

    if (signature.timestamp) {
      result.signedAt = new Date(signature.timestamp);
    }

    return result;
  }

  createSignedMessage(message: string, privateKey: string, publicKey: string): SignedMessage {
    const signature = this.sign(message, privateKey, true);

    return {
      message,
      signature,
      publicKey,
    };
  }

  verifySignedMessage(signedMessage: SignedMessage): VerificationResult {
    return this.verify(
      signedMessage.message,
      signedMessage.signature,
      signedMessage.publicKey
    );
  }

  isTimestampValid(signature: Signature, maxAgeSeconds: number): boolean {
    if (!signature.timestamp) {
      return true; // No timestamp to validate
    }

    const age = (Date.now() - signature.timestamp) / 1000;
    return age <= maxAgeSeconds;
  }

  detectTampering(original: string, modified: string, signature: Signature, publicKey: string): {
    tampered: boolean;
    originalValid: boolean;
    modifiedValid: boolean;
  } {
    const originalResult = this.verify(original, signature, publicKey);
    const modifiedResult = this.verify(modified, signature, publicKey);

    return {
      tampered: original !== modified,
      originalValid: originalResult.isValid,
      modifiedValid: modifiedResult.isValid,
    };
  }
}

export { DigitalSignature, Signature, SignedMessage, VerificationResult };`,
	hint1: `For sign, first hash the message using the hash() method, then encrypt the hash with the private key using simulateEncrypt().`,
	hint2: `For verify, hash the received message, decrypt the signature with public key, and compare the hashes. They should match if valid.`,
	testCode: `import { DigitalSignature } from './solution';

// Test1: generateKeyPair returns both keys
test('Test1', () => {
  const ds = new DigitalSignature();
  const { publicKey, privateKey } = ds.generateKeyPair();
  expect(publicKey).toBeTruthy();
  expect(privateKey).toBeTruthy();
  expect(publicKey).not.toBe(privateKey);
});

// Test2: sign returns Signature with value
test('Test2', () => {
  const ds = new DigitalSignature();
  const { privateKey } = ds.generateKeyPair();
  const signature = ds.sign('Hello', privateKey);
  expect(signature.value).toBeTruthy();
  expect(signature.algorithm).toBe('RSA-SHA256');
});

// Test3: verify returns true for valid signature
test('Test3', () => {
  const ds = new DigitalSignature();
  const { publicKey, privateKey } = ds.generateKeyPair();
  const signature = ds.sign('Test message', privateKey);
  const result = ds.verify('Test message', signature, publicKey);
  expect(result.isValid).toBe(true);
});

// Test4: verify returns false for tampered message
test('Test4', () => {
  const ds = new DigitalSignature();
  const { publicKey, privateKey } = ds.generateKeyPair();
  const signature = ds.sign('Original', privateKey);
  const result = ds.verify('Tampered', signature, publicKey);
  expect(result.isValid).toBe(false);
});

// Test5: sign with timestamp includes it
test('Test5', () => {
  const ds = new DigitalSignature();
  const { privateKey } = ds.generateKeyPair();
  const signature = ds.sign('Message', privateKey, true);
  expect(signature.timestamp).toBeTruthy();
  expect(signature.timestamp).toBeLessThanOrEqual(Date.now());
});

// Test6: createSignedMessage packages everything
test('Test6', () => {
  const ds = new DigitalSignature();
  const { publicKey, privateKey } = ds.generateKeyPair();
  const signed = ds.createSignedMessage('Data', privateKey, publicKey);
  expect(signed.message).toBe('Data');
  expect(signed.signature).toBeTruthy();
  expect(signed.publicKey).toBe(publicKey);
});

// Test7: verifySignedMessage validates correctly
test('Test7', () => {
  const ds = new DigitalSignature();
  const { publicKey, privateKey } = ds.generateKeyPair();
  const signed = ds.createSignedMessage('Secure data', privateKey, publicKey);
  const result = ds.verifySignedMessage(signed);
  expect(result.isValid).toBe(true);
});

// Test8: isTimestampValid checks age
test('Test8', () => {
  const ds = new DigitalSignature();
  const recentSig: Signature = { value: 'x', algorithm: 'RSA', timestamp: Date.now() };
  const oldSig: Signature = { value: 'x', algorithm: 'RSA', timestamp: Date.now() - 120000 };
  expect(ds.isTimestampValid(recentSig, 60)).toBe(true);
  expect(ds.isTimestampValid(oldSig, 60)).toBe(false);
});

// Test9: detectTampering identifies changes
test('Test9', () => {
  const ds = new DigitalSignature();
  const { publicKey, privateKey } = ds.generateKeyPair();
  const signature = ds.sign('Original', privateKey);
  const result = ds.detectTampering('Original', 'Modified', signature, publicKey);
  expect(result.tampered).toBe(true);
  expect(result.originalValid).toBe(true);
  expect(result.modifiedValid).toBe(false);
});

// Test10: Different messages produce different signatures
test('Test10', () => {
  const ds = new DigitalSignature();
  const { privateKey } = ds.generateKeyPair();
  const sig1 = ds.sign('Message A', privateKey);
  const sig2 = ds.sign('Message B', privateKey);
  expect(sig1.value).not.toBe(sig2.value);
});`,
	whyItMatters: `Digital signatures are the foundation of trust on the internet.

**Where Signatures Are Used:**

\`\`\`
Software Distribution:
  Windows/Mac apps → Code signing certificates
  npm/PyPI packages → Package signatures
  Docker images → Content trust

Documents:
  PDFs → Adobe Sign, DocuSign
  Legal documents → Digital signature laws
  Financial transactions → Non-repudiation

Internet Security:
  TLS certificates → CA signatures
  JWT tokens → Signed claims
  Email → S/MIME, PGP signatures
  Git commits → GPG signatures
\`\`\`

**Why Non-Repudiation Matters:**

\`\`\`
Without signatures:
  Alice: "I never sent that email!"
  Bob: "But I have a copy..."
  Judge: "Anyone could have faked it."

With digital signatures:
  Alice: "I never sent that email!"
  Bob: "Here's the cryptographic signature."
  Judge: "Only someone with Alice's private key could sign this."
\`\`\`

**Attack: Signature Stripping**

\`\`\`
Original email: "Pay $100" + [VALID SIGNATURE]
Attack: Remove signature, change to "Pay $10000"
Defense: Always verify signatures are present and valid!
\`\`\``,
	order: 4,
	translations: {
		ru: {
			title: 'Цифровые подписи',
			description: `Изучите цифровые подписи - криптографическое доказательство подлинности и целостности.

**Что такое цифровые подписи?**

Цифровые подписи используют асимметричную криптографию в обратном порядке:
- **Подписать** закрытым ключом
- **Проверить** открытым ключом

**Свойства:**

1. **Аутентичность** - доказывает, кто подписал
2. **Целостность** - обнаруживает любые изменения
3. **Неотказуемость** - подписавший не может отрицать подпись

**Ваша задача:**

Реализуйте класс \`DigitalSignature\`.`,
			hint1: `Для sign сначала хешируйте сообщение методом hash(), затем зашифруйте хеш приватным ключом через simulateEncrypt().`,
			hint2: `Для verify хешируйте полученное сообщение, расшифруйте подпись публичным ключом и сравните хеши.`,
			whyItMatters: `Цифровые подписи - основа доверия в интернете.`
		},
		uz: {
			title: 'Raqamli imzolar',
			description: `Raqamli imzolarni o'rganing - haqiqiylik va yaxlitlikning kriptografik isboti.

**Raqamli imzolar nima?**

Raqamli imzolar asimmetrik kriptografiyani teskari tartibda ishlatadi:
- **Imzolash** yopiq kalit bilan
- **Tekshirish** ochiq kalit bilan

**Sizning vazifangiz:**

\`DigitalSignature\` klassini amalga oshiring.`,
			hint1: `sign uchun avval xabarni hash() metodi bilan xeshlang, keyin xeshni yopiq kalit bilan simulateEncrypt() orqali shifrlang.`,
			hint2: `verify uchun qabul qilingan xabarni xeshlang, imzoni ochiq kalit bilan shifrlang va xeshlarni solishtiring.`,
			whyItMatters: `Raqamli imzolar internetda ishonchning asosi.`
		}
	}
};

export default task;
