import { Task } from '../../../../../../../types';

export const task: Task = {
	slug: 'crypto-tls-ssl',
	title: 'TLS/SSL Protocol Fundamentals',
	difficulty: 'hard',
	tags: ['security', 'cryptography', 'tls', 'ssl', 'https', 'typescript'],
	estimatedTime: '40m',
	isPremium: false,
	youtubeUrl: '',
	description: `Learn TLS (Transport Layer Security) - the protocol that secures HTTPS.

**TLS vs SSL:**

| Version | Status |
|---------|--------|
| SSL 2.0 | BROKEN - Never use |
| SSL 3.0 | BROKEN - POODLE attack |
| TLS 1.0 | DEPRECATED |
| TLS 1.1 | DEPRECATED |
| TLS 1.2 | Supported (minimum recommended) |
| TLS 1.3 | Recommended (current) |

**TLS Handshake (simplified):**

\`\`\`
Client                              Server
   │                                   │
   │──── ClientHello ────────────────>│
   │     (supported ciphers, random)  │
   │                                   │
   │<─── ServerHello ─────────────────│
   │     (chosen cipher, random)      │
   │<─── Certificate ─────────────────│
   │<─── ServerHelloDone ─────────────│
   │                                   │
   │──── ClientKeyExchange ──────────>│
   │     (pre-master secret)          │
   │──── ChangeCipherSpec ───────────>│
   │──── Finished ───────────────────>│
   │                                   │
   │<─── ChangeCipherSpec ────────────│
   │<─── Finished ────────────────────│
   │                                   │
   │═══════ Encrypted Traffic ════════│
\`\`\`

**Your Task:**

Implement a \`TLSSimulator\` class that simulates the TLS handshake process.`,
	initialCode: `interface CipherSuite {
  name: string;
  keyExchange: string;
  cipher: string;
  mac: string;
  strength: 'weak' | 'medium' | 'strong';
}

interface Certificate {
  subject: string;
  issuer: string;
  validFrom: Date;
  validTo: Date;
  publicKey: string;
  signature: string;
}

interface HandshakeState {
  step: string;
  clientRandom: string;
  serverRandom: string;
  selectedCipher?: CipherSuite;
  preMasterSecret?: string;
  masterSecret?: string;
  sessionKeys?: { clientKey: string; serverKey: string };
}

interface TLSConnection {
  state: 'handshaking' | 'established' | 'closed' | 'error';
  version: string;
  cipher?: CipherSuite;
  sessionId: string;
}

class TLSSimulator {
  private cipherSuites: CipherSuite[] = [
    { name: 'TLS_AES_256_GCM_SHA384', keyExchange: 'ECDHE', cipher: 'AES-256-GCM', mac: 'SHA384', strength: 'strong' },
    { name: 'TLS_AES_128_GCM_SHA256', keyExchange: 'ECDHE', cipher: 'AES-128-GCM', mac: 'SHA256', strength: 'strong' },
    { name: 'TLS_RSA_WITH_AES_256_CBC_SHA256', keyExchange: 'RSA', cipher: 'AES-256-CBC', mac: 'SHA256', strength: 'medium' },
    { name: 'TLS_RSA_WITH_RC4_128_SHA', keyExchange: 'RSA', cipher: 'RC4', mac: 'SHA1', strength: 'weak' },
  ];

  generateRandom(): string {
    // TODO: Generate 32-byte random value (64 hex chars)
    return '';
  }

  clientHello(supportedCiphers: string[]): { clientRandom: string; ciphers: string[] } {
    // TODO: Initiate client hello with random and supported ciphers
    return { clientRandom: '', ciphers: [] };
  }

  serverHello(clientCiphers: string[]): { serverRandom: string; selectedCipher: CipherSuite | null } {
    // TODO: Select best cipher from client's list
    // Prefer stronger ciphers
    return { serverRandom: '', selectedCipher: null };
  }

  selectBestCipher(clientCiphers: string[]): CipherSuite | null {
    // TODO: Select strongest cipher both sides support
    return null;
  }

  generatePreMasterSecret(): string {
    // TODO: Generate pre-master secret
    return '';
  }

  deriveMasterSecret(preMaster: string, clientRandom: string, serverRandom: string): string {
    // TODO: Derive master secret from pre-master + randoms
    return '';
  }

  deriveSessionKeys(masterSecret: string): { clientKey: string; serverKey: string } {
    // TODO: Derive encryption keys for both directions
    return { clientKey: '', serverKey: '' };
  }

  validateCertificate(cert: Certificate, trustedIssuers: string[]): {
    valid: boolean;
    reason: string;
  } {
    // TODO: Validate certificate (expiry, issuer, etc.)
    return { valid: false, reason: '' };
  }

  performHandshake(clientCiphers: string[]): HandshakeState {
    // TODO: Perform complete TLS handshake simulation
    return {
      step: '',
      clientRandom: '',
      serverRandom: '',
    };
  }

  isCipherSecure(cipherName: string): boolean {
    // TODO: Check if cipher is considered secure
    return false;
  }
}

export { TLSSimulator, CipherSuite, Certificate, HandshakeState, TLSConnection };`,
	solutionCode: `interface CipherSuite {
  name: string;
  keyExchange: string;
  cipher: string;
  mac: string;
  strength: 'weak' | 'medium' | 'strong';
}

interface Certificate {
  subject: string;
  issuer: string;
  validFrom: Date;
  validTo: Date;
  publicKey: string;
  signature: string;
}

interface HandshakeState {
  step: string;
  clientRandom: string;
  serverRandom: string;
  selectedCipher?: CipherSuite;
  preMasterSecret?: string;
  masterSecret?: string;
  sessionKeys?: { clientKey: string; serverKey: string };
}

interface TLSConnection {
  state: 'handshaking' | 'established' | 'closed' | 'error';
  version: string;
  cipher?: CipherSuite;
  sessionId: string;
}

class TLSSimulator {
  private cipherSuites: CipherSuite[] = [
    { name: 'TLS_AES_256_GCM_SHA384', keyExchange: 'ECDHE', cipher: 'AES-256-GCM', mac: 'SHA384', strength: 'strong' },
    { name: 'TLS_AES_128_GCM_SHA256', keyExchange: 'ECDHE', cipher: 'AES-128-GCM', mac: 'SHA256', strength: 'strong' },
    { name: 'TLS_RSA_WITH_AES_256_CBC_SHA256', keyExchange: 'RSA', cipher: 'AES-256-CBC', mac: 'SHA256', strength: 'medium' },
    { name: 'TLS_RSA_WITH_RC4_128_SHA', keyExchange: 'RSA', cipher: 'RC4', mac: 'SHA1', strength: 'weak' },
  ];

  generateRandom(): string {
    const chars = '0123456789abcdef';
    let random = '';
    for (let i = 0; i < 64; i++) {
      random += chars[Math.floor(Math.random() * 16)];
    }
    return random;
  }

  clientHello(supportedCiphers: string[]): { clientRandom: string; ciphers: string[] } {
    return {
      clientRandom: this.generateRandom(),
      ciphers: supportedCiphers,
    };
  }

  serverHello(clientCiphers: string[]): { serverRandom: string; selectedCipher: CipherSuite | null } {
    const selectedCipher = this.selectBestCipher(clientCiphers);

    return {
      serverRandom: this.generateRandom(),
      selectedCipher,
    };
  }

  selectBestCipher(clientCiphers: string[]): CipherSuite | null {
    // Prioritize by strength: strong > medium > weak
    const strengthOrder = ['strong', 'medium', 'weak'];

    for (const strength of strengthOrder) {
      for (const suite of this.cipherSuites) {
        if (suite.strength === strength && clientCiphers.includes(suite.name)) {
          return suite;
        }
      }
    }

    return null;
  }

  generatePreMasterSecret(): string {
    // 48 bytes (96 hex chars) - typically includes TLS version
    const chars = '0123456789abcdef';
    let secret = '0303'; // TLS 1.2 version prefix
    for (let i = 0; i < 92; i++) {
      secret += chars[Math.floor(Math.random() * 16)];
    }
    return secret;
  }

  deriveMasterSecret(preMaster: string, clientRandom: string, serverRandom: string): string {
    // Simplified PRF (Pseudo-Random Function)
    const seed = preMaster + clientRandom + serverRandom;
    let hash = 0;
    for (let i = 0; i < seed.length; i++) {
      hash = ((hash << 5) - hash) + seed.charCodeAt(i);
      hash = hash & hash;
    }

    // Generate 48-byte master secret
    let masterSecret = '';
    for (let i = 0; i < 96; i++) {
      hash = (hash * 1103515245 + 12345) & 0x7fffffff;
      masterSecret += (hash % 16).toString(16);
    }

    return masterSecret;
  }

  deriveSessionKeys(masterSecret: string): { clientKey: string; serverKey: string } {
    // Derive keys from master secret
    let hash1 = 0;
    let hash2 = 0;

    for (let i = 0; i < masterSecret.length; i++) {
      hash1 = ((hash1 << 5) - hash1) + masterSecret.charCodeAt(i);
      hash2 = ((hash2 << 3) + hash2) ^ masterSecret.charCodeAt(i);
    }

    const clientKey = Math.abs(hash1).toString(16).padStart(32, '0');
    const serverKey = Math.abs(hash2).toString(16).padStart(32, '0');

    return { clientKey, serverKey };
  }

  validateCertificate(cert: Certificate, trustedIssuers: string[]): {
    valid: boolean;
    reason: string;
  } {
    const now = new Date();

    // Check expiry
    if (now < cert.validFrom) {
      return { valid: false, reason: 'Certificate not yet valid' };
    }

    if (now > cert.validTo) {
      return { valid: false, reason: 'Certificate has expired' };
    }

    // Check issuer
    if (!trustedIssuers.includes(cert.issuer)) {
      return { valid: false, reason: 'Certificate issuer not trusted' };
    }

    // Check basic fields
    if (!cert.subject || !cert.publicKey) {
      return { valid: false, reason: 'Certificate missing required fields' };
    }

    return { valid: true, reason: 'Certificate validated successfully' };
  }

  performHandshake(clientCiphers: string[]): HandshakeState {
    // Step 1: Client Hello
    const clientHello = this.clientHello(clientCiphers);

    // Step 2: Server Hello
    const serverHello = this.serverHello(clientCiphers);

    if (!serverHello.selectedCipher) {
      return {
        step: 'failed',
        clientRandom: clientHello.clientRandom,
        serverRandom: serverHello.serverRandom,
      };
    }

    // Step 3: Key Exchange
    const preMasterSecret = this.generatePreMasterSecret();

    // Step 4: Derive Master Secret
    const masterSecret = this.deriveMasterSecret(
      preMasterSecret,
      clientHello.clientRandom,
      serverHello.serverRandom
    );

    // Step 5: Derive Session Keys
    const sessionKeys = this.deriveSessionKeys(masterSecret);

    return {
      step: 'completed',
      clientRandom: clientHello.clientRandom,
      serverRandom: serverHello.serverRandom,
      selectedCipher: serverHello.selectedCipher,
      preMasterSecret,
      masterSecret,
      sessionKeys,
    };
  }

  isCipherSecure(cipherName: string): boolean {
    const cipher = this.cipherSuites.find(c => c.name === cipherName);

    if (!cipher) {
      return false;
    }

    // Weak ciphers and those using RC4, MD5, or single DES are insecure
    if (cipher.strength === 'weak') {
      return false;
    }

    if (cipher.cipher.includes('RC4') || cipher.cipher.includes('DES')) {
      return false;
    }

    if (cipher.mac === 'MD5') {
      return false;
    }

    return true;
  }
}

export { TLSSimulator, CipherSuite, Certificate, HandshakeState, TLSConnection };`,
	hint1: `For selectBestCipher, iterate through ciphers by strength (strong first, then medium, then weak) and return the first one that the client supports.`,
	hint2: `For performHandshake, simulate the full flow: clientHello → serverHello → generatePreMasterSecret → deriveMasterSecret → deriveSessionKeys.`,
	testCode: `import { TLSSimulator } from './solution';

// Test1: generateRandom produces 64 hex chars
test('Test1', () => {
  const tls = new TLSSimulator();
  const random = tls.generateRandom();
  expect(random.length).toBe(64);
  expect(/^[0-9a-f]+$/.test(random)).toBe(true);
});

// Test2: clientHello returns random and ciphers
test('Test2', () => {
  const tls = new TLSSimulator();
  const hello = tls.clientHello(['TLS_AES_256_GCM_SHA384']);
  expect(hello.clientRandom.length).toBe(64);
  expect(hello.ciphers).toContain('TLS_AES_256_GCM_SHA384');
});

// Test3: selectBestCipher prefers strong ciphers
test('Test3', () => {
  const tls = new TLSSimulator();
  const cipher = tls.selectBestCipher([
    'TLS_RSA_WITH_RC4_128_SHA',
    'TLS_AES_256_GCM_SHA384',
  ]);
  expect(cipher?.name).toBe('TLS_AES_256_GCM_SHA384');
});

// Test4: selectBestCipher returns null for unsupported
test('Test4', () => {
  const tls = new TLSSimulator();
  const cipher = tls.selectBestCipher(['UNKNOWN_CIPHER']);
  expect(cipher).toBeNull();
});

// Test5: generatePreMasterSecret returns valid format
test('Test5', () => {
  const tls = new TLSSimulator();
  const secret = tls.generatePreMasterSecret();
  expect(secret.length).toBe(96);
  expect(secret.startsWith('0303')).toBe(true); // TLS 1.2
});

// Test6: deriveMasterSecret produces consistent output
test('Test6', () => {
  const tls = new TLSSimulator();
  const master1 = tls.deriveMasterSecret('pre', 'client', 'server');
  const master2 = tls.deriveMasterSecret('pre', 'client', 'server');
  expect(master1).toBe(master2);
  expect(master1.length).toBe(96);
});

// Test7: validateCertificate checks expiry
test('Test7', () => {
  const tls = new TLSSimulator();
  const expiredCert = {
    subject: 'example.com',
    issuer: 'TrustedCA',
    validFrom: new Date('2020-01-01'),
    validTo: new Date('2021-01-01'),
    publicKey: 'key',
    signature: 'sig',
  };
  const result = tls.validateCertificate(expiredCert, ['TrustedCA']);
  expect(result.valid).toBe(false);
  expect(result.reason).toContain('expired');
});

// Test8: performHandshake completes successfully
test('Test8', () => {
  const tls = new TLSSimulator();
  const state = tls.performHandshake(['TLS_AES_256_GCM_SHA384']);
  expect(state.step).toBe('completed');
  expect(state.sessionKeys).toBeTruthy();
  expect(state.masterSecret).toBeTruthy();
});

// Test9: isCipherSecure rejects weak ciphers
test('Test9', () => {
  const tls = new TLSSimulator();
  expect(tls.isCipherSecure('TLS_RSA_WITH_RC4_128_SHA')).toBe(false);
  expect(tls.isCipherSecure('TLS_AES_256_GCM_SHA384')).toBe(true);
});

// Test10: deriveSessionKeys produces different keys
test('Test10', () => {
  const tls = new TLSSimulator();
  const keys = tls.deriveSessionKeys('mastesecret1234567890');
  expect(keys.clientKey).toBeTruthy();
  expect(keys.serverKey).toBeTruthy();
  expect(keys.clientKey).not.toBe(keys.serverKey);
});`,
	whyItMatters: `TLS secures virtually all internet communication - HTTPS, email, messaging, APIs.

**TLS Vulnerabilities Timeline:**

| Year | Attack | Impact |
|------|--------|--------|
| 2011 | BEAST | CBC vulnerability in TLS 1.0 |
| 2013 | BREACH | Compression leak |
| 2014 | Heartbleed | OpenSSL memory leak |
| 2014 | POODLE | SSL 3.0 broken |
| 2015 | Logjam | Weak DH parameters |
| 2016 | DROWN | SSLv2 cross-protocol |

**TLS 1.3 Improvements:**

\`\`\`
TLS 1.2: 2 round trips to establish connection
TLS 1.3: 1 round trip (0-RTT resumption possible)

TLS 1.2: Many cipher options (some weak)
TLS 1.3: Only 5 secure ciphers allowed

TLS 1.2: RSA key exchange allowed
TLS 1.3: Forward secrecy required (ECDHE only)
\`\`\`

**Common TLS Misconfigurations:**

1. Allowing TLS 1.0/1.1 (deprecated)
2. Supporting weak ciphers (RC4, 3DES)
3. Missing HSTS header
4. Expired certificates
5. Wrong certificate hostname
6. Not using certificate pinning for mobile apps`,
	order: 5,
	translations: {
		ru: {
			title: 'Основы протокола TLS/SSL',
			description: `Изучите TLS (Transport Layer Security) - протокол, который защищает HTTPS.

**TLS vs SSL:**

| Версия | Статус |
|--------|--------|
| SSL 2.0/3.0 | ВЗЛОМАНЫ |
| TLS 1.0/1.1 | УСТАРЕЛИ |
| TLS 1.2 | Минимально рекомендуется |
| TLS 1.3 | Рекомендуется (текущий) |

**TLS Handshake:**

1. ClientHello - клиент отправляет поддерживаемые шифры
2. ServerHello - сервер выбирает шифр
3. Certificate - сервер отправляет сертификат
4. KeyExchange - обмен ключами
5. Finished - зашифрованное соединение установлено

**Ваша задача:**

Реализуйте класс \`TLSSimulator\`.`,
			hint1: `Для selectBestCipher итерируйте по шифрам по силе (сначала strong, потом medium, потом weak) и верните первый, который поддерживает клиент.`,
			hint2: `Для performHandshake симулируйте полный flow: clientHello → serverHello → generatePreMasterSecret → deriveMasterSecret → deriveSessionKeys.`,
			whyItMatters: `TLS защищает практически все интернет-коммуникации - HTTPS, email, мессенджеры, API.`
		},
		uz: {
			title: 'TLS/SSL protokoli asoslari',
			description: `TLS (Transport Layer Security) ni o'rganing - HTTPS ni himoya qiluvchi protokol.

**TLS Handshake:**

1. ClientHello - mijoz qo'llab-quvvatlanadigan shifrlarni yuboradi
2. ServerHello - server shifrni tanlaydi
3. Certificate - server sertifikatni yuboradi
4. KeyExchange - kalit almashish
5. Finished - shifrlangan ulanish o'rnatildi

**Sizning vazifangiz:**

\`TLSSimulator\` klassini amalga oshiring.`,
			hint1: `selectBestCipher uchun shifrlarni kuch bo'yicha tekshiring (avval strong, keyin medium, keyin weak) va mijoz qo'llab-quvvatlaydigan birinchisini qaytaring.`,
			hint2: `performHandshake uchun to'liq oqimni simulyatsiya qiling: clientHello → serverHello → generatePreMasterSecret → deriveMasterSecret → deriveSessionKeys.`,
			whyItMatters: `TLS deyarli barcha internet kommunikatsiyalarini himoya qiladi - HTTPS, email, messenjerlar, API.`
		}
	}
};

export default task;
