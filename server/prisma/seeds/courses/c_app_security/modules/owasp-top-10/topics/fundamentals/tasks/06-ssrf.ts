import { Task } from '../../../../../../../types';

export const task: Task = {
	slug: 'sec-ssrf-prevention',
	title: 'SSRF: Server-Side Request Forgery Protection',
	difficulty: 'hard',
	tags: ['security', 'owasp', 'ssrf', 'typescript'],
	estimatedTime: '40m',
	isPremium: true,
	youtubeUrl: '',
	description: `Learn to prevent Server-Side Request Forgery (SSRF) - a critical vulnerability in OWASP Top 10.

**What is SSRF?**

SSRF occurs when an attacker can make a server perform requests to unintended locations. The server essentially becomes a proxy for the attacker, potentially accessing internal services, cloud metadata, or sensitive endpoints.

**Attack Scenarios:**

\`\`\`typescript
// Vulnerable: URL from user input used directly
app.post('/api/fetch-url', async (req, res) => {
  const response = await fetch(req.body.url);  // DANGEROUS!
  res.json(await response.json());
});

// Attacker exploits:
// 1. Access cloud metadata
POST /api/fetch-url
{ "url": "http://169.254.169.254/latest/meta-data/iam/security-credentials/" }

// 2. Scan internal network
POST /api/fetch-url
{ "url": "http://192.168.1.1:8080/admin" }

// 3. Access localhost services
POST /api/fetch-url
{ "url": "http://localhost:6379/KEYS *" }
\`\`\`

**Your Task:**

Implement an \`SsrfProtector\` class that:

1. Validates and sanitizes URLs before making requests
2. Blocks requests to internal/private IP ranges
3. Implements an allowlist of permitted domains
4. Detects and blocks common SSRF bypasses

**Example Usage:**

\`\`\`typescript
const protector = new SsrfProtector({
  allowedDomains: ['api.example.com', 'cdn.example.com']
});

protector.isUrlSafe('https://api.example.com/data'); // true
protector.isUrlSafe('http://localhost/secret'); // false
protector.isUrlSafe('http://169.254.169.254/'); // false (AWS metadata)
protector.isUrlSafe('http://[::1]/'); // false (IPv6 localhost)
\`\`\``,
	initialCode: `interface SsrfConfig {
  allowedDomains?: string[];
  allowedProtocols?: string[];
  blockPrivateIPs?: boolean;
  blockMetadataEndpoints?: boolean;
}

interface UrlValidationResult {
  safe: boolean;
  reason?: string;
  normalizedUrl?: string;
}

class SsrfProtector {
  private config: Required<SsrfConfig>;

  // Private IP ranges to block
  private readonly PRIVATE_IP_PATTERNS = [
    /^127\\./,                    // Loopback
    /^10\\./,                     // Class A private
    /^172\\.(1[6-9]|2[0-9]|3[0-1])\\./,  // Class B private
    /^192\\.168\\./,              // Class C private
    /^169\\.254\\./,              // Link-local
    /^0\\./,                      // Current network
  ];

  // Cloud metadata endpoints
  private readonly METADATA_HOSTS = [
    '169.254.169.254',     // AWS, GCP
    'metadata.google.internal',
    '100.100.100.200',     // Alibaba Cloud
    '169.254.170.2',       // Azure
  ];

  constructor(config?: SsrfConfig) {
    this.config = {
      allowedDomains: config?.allowedDomains || [],
      allowedProtocols: config?.allowedProtocols || ['https'],
      blockPrivateIPs: config?.blockPrivateIPs ?? true,
      blockMetadataEndpoints: config?.blockMetadataEndpoints ?? true,
    };
  }

  validateUrl(urlString: string): UrlValidationResult {
    // TODO: Validate URL comprehensively
    // 1. Parse URL and check protocol
    // 2. Check against allowedDomains if configured
    // 3. Resolve hostname and check for private IPs
    // 4. Check for metadata endpoints
    // 5. Detect bypass attempts (octal IPs, IPv6, etc.)
    return { safe: false, reason: 'Not implemented' };
  }

  isUrlSafe(urlString: string): boolean {
    // TODO: Simple boolean check
    return false;
  }

  isPrivateIP(ip: string): boolean {
    // TODO: Check if IP is in private range
    return false;
  }

  normalizeUrl(urlString: string): string {
    // TODO: Normalize URL to prevent bypasses
    // Handle case sensitivity, encoding, etc.
    return urlString;
  }

  isMetadataEndpoint(hostname: string): boolean {
    // TODO: Check if hostname is cloud metadata endpoint
    return false;
  }
}

export { SsrfProtector, SsrfConfig, UrlValidationResult };`,
	solutionCode: `interface SsrfConfig {
  allowedDomains?: string[];
  allowedProtocols?: string[];
  blockPrivateIPs?: boolean;
  blockMetadataEndpoints?: boolean;
}

interface UrlValidationResult {
  safe: boolean;
  reason?: string;
  normalizedUrl?: string;
}

class SsrfProtector {
  private config: Required<SsrfConfig>;

  // Private IP ranges to block
  private readonly PRIVATE_IP_PATTERNS = [
    /^127\\./,                    // Loopback
    /^10\\./,                     // Class A private
    /^172\\.(1[6-9]|2[0-9]|3[0-1])\\./,  // Class B private
    /^192\\.168\\./,              // Class C private
    /^169\\.254\\./,              // Link-local
    /^0\\./,                      // Current network
  ];

  // Cloud metadata endpoints
  private readonly METADATA_HOSTS = [
    '169.254.169.254',     // AWS, GCP
    'metadata.google.internal',
    '100.100.100.200',     // Alibaba Cloud
    '169.254.170.2',       // Azure
  ];

  constructor(config?: SsrfConfig) {
    this.config = {
      allowedDomains: config?.allowedDomains || [],
      allowedProtocols: config?.allowedProtocols || ['https'],
      blockPrivateIPs: config?.blockPrivateIPs ?? true,
      blockMetadataEndpoints: config?.blockMetadataEndpoints ?? true,
    };
  }

  // Comprehensive URL validation
  validateUrl(urlString: string): UrlValidationResult {
    try {
      // Normalize first to prevent bypasses
      const normalized = this.normalizeUrl(urlString);
      const url = new URL(normalized);

      // Check protocol
      const protocol = url.protocol.replace(':', '');
      if (!this.config.allowedProtocols.includes(protocol)) {
        return { safe: false, reason: \`Protocol '\${protocol}' not allowed\` };
      }

      // Check against allowlist if configured
      if (this.config.allowedDomains.length > 0) {
        const isAllowed = this.config.allowedDomains.some(
          domain => url.hostname === domain || url.hostname.endsWith('.' + domain)
        );
        if (!isAllowed) {
          return { safe: false, reason: \`Domain '\${url.hostname}' not in allowlist\` };
        }
      }

      // Check for metadata endpoints
      if (this.config.blockMetadataEndpoints && this.isMetadataEndpoint(url.hostname)) {
        return { safe: false, reason: 'Cloud metadata endpoint blocked' };
      }

      // Check for private IPs (including localhost variations)
      if (this.config.blockPrivateIPs) {
        // Check if hostname is IP
        if (this.isPrivateIP(url.hostname)) {
          return { safe: false, reason: 'Private IP address blocked' };
        }

        // Check for localhost variations
        if (url.hostname === 'localhost' || url.hostname === '[::1]') {
          return { safe: false, reason: 'Localhost blocked' };
        }
      }

      return { safe: true, normalizedUrl: normalized };
    } catch (error) {
      return { safe: false, reason: 'Invalid URL format' };
    }
  }

  // Simple boolean check
  isUrlSafe(urlString: string): boolean {
    return this.validateUrl(urlString).safe;
  }

  // Check if IP is in private range
  isPrivateIP(ip: string): boolean {
    // Check standard private ranges
    for (const pattern of this.PRIVATE_IP_PATTERNS) {
      if (pattern.test(ip)) {
        return true;
      }
    }

    // Check IPv6 private addresses
    if (ip.startsWith('[')) {
      const ipv6 = ip.slice(1, -1).toLowerCase();
      if (ipv6 === '::1' || ipv6.startsWith('fe80:') || ipv6.startsWith('fc') || ipv6.startsWith('fd')) {
        return true;
      }
    }

    // Check for octal/hex IP bypass attempts
    // e.g., 0177.0.0.1 = 127.0.0.1 in octal
    if (/^0[0-7]+\\./.test(ip) || /^0x[0-9a-f]+/i.test(ip)) {
      return true; // Block octal/hex IPs as suspicious
    }

    return false;
  }

  // Normalize URL to prevent bypasses
  normalizeUrl(urlString: string): string {
    let normalized = urlString.trim();

    // Decode URL encoding that might hide attacks
    try {
      // Decode multiple times to catch double-encoding
      let decoded = normalized;
      for (let i = 0; i < 3; i++) {
        const next = decodeURIComponent(decoded);
        if (next === decoded) break;
        decoded = next;
      }
      normalized = decoded;
    } catch {
      // Keep original if decoding fails
    }

    // Remove any null bytes
    normalized = normalized.replace(/\\x00/g, '');

    // Lowercase protocol and hostname for consistency
    try {
      const url = new URL(normalized);
      normalized = url.href;
    } catch {
      // Return as-is if not valid URL
    }

    return normalized;
  }

  // Check if hostname is cloud metadata endpoint
  isMetadataEndpoint(hostname: string): boolean {
    const normalizedHost = hostname.toLowerCase();

    for (const metadataHost of this.METADATA_HOSTS) {
      if (normalizedHost === metadataHost) {
        return true;
      }
    }

    // Also check for common aliases
    if (normalizedHost.includes('metadata') || normalizedHost.includes('instance-data')) {
      return true;
    }

    return false;
  }
}

export { SsrfProtector, SsrfConfig, UrlValidationResult };`,
	hint1: `For validateUrl, first normalize the URL, then parse it with new URL(). Check protocol, hostname against allowlist, and call helper methods for IP and metadata checks.`,
	hint2: `For isPrivateIP, test against the PRIVATE_IP_PATTERNS regex array. Also check for IPv6 localhost (::1) and octal/hex encoded IPs which are common bypass techniques.`,
	testCode: `import { SsrfProtector } from './solution';

// Test1: Blocks localhost
test('Test1', () => {
  const protector = new SsrfProtector();
  expect(protector.isUrlSafe('http://localhost/admin')).toBe(false);
});

// Test2: Blocks 127.0.0.1
test('Test2', () => {
  const protector = new SsrfProtector();
  expect(protector.isUrlSafe('http://127.0.0.1/secret')).toBe(false);
});

// Test3: Blocks AWS metadata
test('Test3', () => {
  const protector = new SsrfProtector();
  expect(protector.isUrlSafe('http://169.254.169.254/latest/meta-data/')).toBe(false);
});

// Test4: Blocks private IP 10.x.x.x
test('Test4', () => {
  const protector = new SsrfProtector();
  expect(protector.isUrlSafe('http://10.0.0.1/internal')).toBe(false);
});

// Test5: Blocks private IP 192.168.x.x
test('Test5', () => {
  const protector = new SsrfProtector();
  expect(protector.isUrlSafe('http://192.168.1.1/')).toBe(false);
});

// Test6: Allows valid HTTPS URL
test('Test6', () => {
  const protector = new SsrfProtector({ allowedDomains: ['example.com'] });
  expect(protector.isUrlSafe('https://example.com/api')).toBe(true);
});

// Test7: Blocks non-allowlisted domain
test('Test7', () => {
  const protector = new SsrfProtector({ allowedDomains: ['api.example.com'] });
  expect(protector.isUrlSafe('https://evil.com/api')).toBe(false);
});

// Test8: isPrivateIP detects private ranges
test('Test8', () => {
  const protector = new SsrfProtector();
  expect(protector.isPrivateIP('10.0.0.1')).toBe(true);
  expect(protector.isPrivateIP('172.16.0.1')).toBe(true);
  expect(protector.isPrivateIP('8.8.8.8')).toBe(false);
});

// Test9: isMetadataEndpoint detects AWS
test('Test9', () => {
  const protector = new SsrfProtector();
  expect(protector.isMetadataEndpoint('169.254.169.254')).toBe(true);
});

// Test10: Blocks HTTP when only HTTPS allowed
test('Test10', () => {
  const protector = new SsrfProtector({ allowedProtocols: ['https'] });
  expect(protector.isUrlSafe('http://example.com/api')).toBe(false);
});`,
	whyItMatters: `SSRF attacks have led to some of the most severe breaches in cloud environments.

**Major SSRF Incidents:**

**1. Capital One (2019)**
\`\`\`
Impact: 100 million customer records stolen
Method: SSRF to AWS metadata endpoint
Data: SSNs, bank accounts, credit scores
Cost: $80 million fine, $190 million settlement
How: WAF misconfiguration + SSRF to get IAM credentials
\`\`\`

**2. Microsoft Exchange (2021)**
\`\`\`
Impact: ProxyLogon vulnerability chain
Method: SSRF as first step in exploitation
Result: Remote code execution
Affected: Hundreds of thousands of servers
\`\`\`

**3. GitLab (2021)**
\`\`\`
Impact: Internal network scanning
Method: SSRF in project import feature
Result: Kubernetes secrets exposure
Bounty: $33,500 paid
\`\`\`

**Common SSRF Targets:**

| Target | URL | Data Exposed |
|--------|-----|--------------|
| AWS Metadata | http://169.254.169.254/ | IAM credentials, tokens |
| GCP Metadata | http://metadata.google.internal/ | Service account tokens |
| Kubernetes | http://kubernetes.default.svc/ | Cluster secrets |
| Docker | http://unix:/var/run/docker.sock | Container control |
| Internal Services | http://localhost:6379 | Redis, databases |

**SSRF Bypass Techniques:**

\`\`\`typescript
// Attackers try various bypasses:

// 1. Decimal IP
"http://2130706433/"  // = 127.0.0.1

// 2. Octal IP
"http://0177.0.0.1/"  // = 127.0.0.1

// 3. IPv6
"http://[::1]/"       // = localhost

// 4. DNS rebinding
// attacker.com resolves to 127.0.0.1

// 5. URL encoding
"http://%31%32%37.0.0.1/"

// 6. Different ports
"http://localhost:80@evil.com/"
\`\`\`

**Prevention Best Practices:**

\`\`\`typescript
// ✅ Use allowlists, not blocklists
const ALLOWED_DOMAINS = ['api.trusted.com', 'cdn.trusted.com'];

// ✅ Validate after DNS resolution
async function safeFetch(url: string) {
  const { hostname } = new URL(url);
  const ips = await dns.resolve(hostname);

  for (const ip of ips) {
    if (isPrivateIP(ip)) {
      throw new Error('SSRF blocked');
    }
  }

  return fetch(url);
}

// ✅ Use network segmentation
// Isolate servers that make outbound requests

// ✅ Disable unnecessary protocols
// Block file://, gopher://, dict://
\`\`\``,
	order: 5,
	translations: {
		ru: {
			title: 'SSRF: Защита от подделки запросов на стороне сервера',
			description: `Научитесь предотвращать SSRF (Server-Side Request Forgery) - критическую уязвимость в OWASP Top 10.

**Что такое SSRF?**

SSRF возникает, когда атакующий может заставить сервер выполнять запросы к непредусмотренным адресам. Сервер становится прокси для атакующего.

**Ваша задача:**

Реализуйте класс \`SsrfProtector\`:

1. Валидация и санитизация URL перед запросами
2. Блокировка запросов к внутренним/приватным IP
3. Реализация allowlist разрешённых доменов
4. Обнаружение и блокировка SSRF bypass техник`,
			hint1: `Для validateUrl сначала нормализуйте URL, затем распарсите через new URL(). Проверьте протокол, hostname и вызовите вспомогательные методы.`,
			hint2: `Для isPrivateIP проверьте против массива regex PRIVATE_IP_PATTERNS. Также проверьте IPv6 localhost и octal/hex кодированные IP.`,
			whyItMatters: `SSRF-атаки привели к одним из самых серьёзных утечек в облачных средах.`
		},
		uz: {
			title: 'SSRF: Server tomonda so\'rovlarni soxtalashtirish himoyasi',
			description: `Server-Side Request Forgery (SSRF) ni oldini olishni o'rganing - OWASP Top 10 da muhim zaiflik.

**SSRF nima?**

SSRF tajovuzkor serverni kutilmagan joylarga so'rovlar yuborishga majbur qila olganda yuz beradi.

**Sizning vazifangiz:**

\`SsrfProtector\` klassini amalga oshiring:

1. So'rovlardan oldin URL ni tasdiqlash va tozalash
2. Ichki/xususiy IP larga so'rovlarni bloklash
3. Ruxsat berilgan domenlar ro'yxatini amalga oshirish
4. SSRF bypass usullarini aniqlash va bloklash`,
			hint1: `validateUrl uchun avval URL ni normalizatsiya qiling, keyin new URL() orqali parsing qiling.`,
			hint2: `isPrivateIP uchun PRIVATE_IP_PATTERNS regex massiviga qarshi tekshiring.`,
			whyItMatters: `SSRF hujumlari bulut muhitlarida eng jiddiy buzilishlarga olib keldi.`
		}
	}
};

export default task;
