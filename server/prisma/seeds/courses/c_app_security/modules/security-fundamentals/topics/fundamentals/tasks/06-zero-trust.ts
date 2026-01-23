import { Task } from '../../../../../../../types';

export const task: Task = {
	slug: 'sec-zero-trust',
	title: 'Zero Trust Architecture',
	difficulty: 'medium',
	tags: ['security', 'fundamentals', 'zero-trust', 'typescript'],
	estimatedTime: '30m',
	isPremium: false,
	youtubeUrl: '',
	description: `Learn Zero Trust Architecture - "Never trust, always verify."

**What is Zero Trust?**

Zero Trust is a security model that requires strict identity verification for every person and device trying to access resources, regardless of whether they are inside or outside the network perimeter.

**Core Principles:**

1. **Verify Explicitly** - Always authenticate and authorize based on all available data
2. **Use Least Privilege** - Limit access with Just-In-Time and Just-Enough-Access
3. **Assume Breach** - Minimize blast radius and segment access

**Zero Trust vs Traditional Security:**

| Traditional | Zero Trust |
|-------------|------------|
| Trust inside network | Trust no one |
| Perimeter defense | Micro-segmentation |
| VPN = trusted | Every request verified |
| Static policies | Dynamic, context-aware |

**Your Task:**

Implement a \`ZeroTrustValidator\` class that:

1. Validates requests based on multiple signals (user, device, location, time)
2. Calculates trust scores
3. Enforces conditional access policies
4. Implements continuous verification`,
	initialCode: `interface RequestContext {
  userId: string;
  deviceId: string;
  deviceTrust: 'managed' | 'unmanaged' | 'unknown';
  location: string;
  ipAddress: string;
  timestamp: Date;
  resource: string;
  action: string;
}

interface AccessPolicy {
  resource: string;
  requiredTrustScore: number;
  allowedLocations?: string[];
  allowedDeviceTypes?: string[];
  timeRestrictions?: { start: number; end: number }; // Hours 0-23
  requireMfa?: boolean;
}

interface AccessDecision {
  allowed: boolean;
  trustScore: number;
  reasons: string[];
  requiresMfa: boolean;
}

class ZeroTrustValidator {
  private policies: Map<string, AccessPolicy> = new Map();
  private trustedLocations: Set<string> = new Set();
  private trustedDevices: Set<string> = new Set();

  addPolicy(policy: AccessPolicy): void {
    // TODO: Add an access policy for a resource
  }

  addTrustedLocation(location: string): void {
    // TODO: Add a trusted location
  }

  addTrustedDevice(deviceId: string): void {
    // TODO: Add a trusted device
  }

  calculateTrustScore(context: RequestContext): number {
    // TODO: Calculate trust score 0-100 based on context
    // Consider: user auth, device trust, location, time, behavior
    return 0;
  }

  validateAccess(context: RequestContext): AccessDecision {
    // TODO: Validate access request against policies
    // Return decision with trust score and reasons
    return { allowed: false, trustScore: 0, reasons: [], requiresMfa: false };
  }

  isWithinTimeRestriction(policy: AccessPolicy, hour: number): boolean {
    // TODO: Check if current hour is within allowed time window
    return false;
  }

  isTrustedLocation(location: string): boolean {
    // TODO: Check if location is trusted
    return false;
  }

  isTrustedDevice(deviceId: string): boolean {
    // TODO: Check if device is trusted
    return false;
  }

  getAccessDenialReasons(context: RequestContext, policy: AccessPolicy): string[] {
    // TODO: Get list of reasons why access would be denied
    return [];
  }
}

export { ZeroTrustValidator, RequestContext, AccessPolicy, AccessDecision };`,
	solutionCode: `interface RequestContext {
  userId: string;
  deviceId: string;
  deviceTrust: 'managed' | 'unmanaged' | 'unknown';
  location: string;
  ipAddress: string;
  timestamp: Date;
  resource: string;
  action: string;
}

interface AccessPolicy {
  resource: string;
  requiredTrustScore: number;
  allowedLocations?: string[];
  allowedDeviceTypes?: string[];
  timeRestrictions?: { start: number; end: number }; // Hours 0-23
  requireMfa?: boolean;
}

interface AccessDecision {
  allowed: boolean;
  trustScore: number;
  reasons: string[];
  requiresMfa: boolean;
}

class ZeroTrustValidator {
  private policies: Map<string, AccessPolicy> = new Map();
  private trustedLocations: Set<string> = new Set();
  private trustedDevices: Set<string> = new Set();

  addPolicy(policy: AccessPolicy): void {
    this.policies.set(policy.resource, policy);
  }

  addTrustedLocation(location: string): void {
    this.trustedLocations.add(location.toLowerCase());
  }

  addTrustedDevice(deviceId: string): void {
    this.trustedDevices.add(deviceId);
  }

  calculateTrustScore(context: RequestContext): number {
    let score = 0;

    // User authenticated: +20
    if (context.userId) {
      score += 20;
    }

    // Device trust level
    if (context.deviceTrust === 'managed') {
      score += 30;
    } else if (context.deviceTrust === 'unmanaged') {
      score += 10;
    }
    // unknown = 0

    // Trusted device: +15
    if (this.isTrustedDevice(context.deviceId)) {
      score += 15;
    }

    // Trusted location: +20
    if (this.isTrustedLocation(context.location)) {
      score += 20;
    }

    // Business hours (9-17): +10
    const hour = context.timestamp.getHours();
    if (hour >= 9 && hour <= 17) {
      score += 10;
    }

    // Not suspicious IP (simple check): +5
    if (!context.ipAddress.startsWith('10.') && !context.ipAddress.startsWith('192.168.')) {
      score += 5; // Public IP, might be expected
    }

    return Math.min(100, score);
  }

  validateAccess(context: RequestContext): AccessDecision {
    const trustScore = this.calculateTrustScore(context);
    const policy = this.policies.get(context.resource);
    const reasons: string[] = [];

    // No policy = deny by default
    if (!policy) {
      return {
        allowed: false,
        trustScore,
        reasons: ['No policy defined for resource'],
        requiresMfa: false,
      };
    }

    // Check trust score
    if (trustScore < policy.requiredTrustScore) {
      reasons.push(\`Trust score \${trustScore} below required \${policy.requiredTrustScore}\`);
    }

    // Check location
    if (policy.allowedLocations && policy.allowedLocations.length > 0) {
      if (!policy.allowedLocations.map(l => l.toLowerCase()).includes(context.location.toLowerCase())) {
        reasons.push(\`Location '\${context.location}' not allowed\`);
      }
    }

    // Check device type
    if (policy.allowedDeviceTypes && policy.allowedDeviceTypes.length > 0) {
      if (!policy.allowedDeviceTypes.includes(context.deviceTrust)) {
        reasons.push(\`Device type '\${context.deviceTrust}' not allowed\`);
      }
    }

    // Check time restrictions
    if (policy.timeRestrictions) {
      const hour = context.timestamp.getHours();
      if (!this.isWithinTimeRestriction(policy, hour)) {
        reasons.push(\`Access not allowed at hour \${hour}\`);
      }
    }

    const allowed = reasons.length === 0;
    const requiresMfa = policy.requireMfa || trustScore < 70;

    return {
      allowed,
      trustScore,
      reasons,
      requiresMfa,
    };
  }

  isWithinTimeRestriction(policy: AccessPolicy, hour: number): boolean {
    if (!policy.timeRestrictions) return true;

    const { start, end } = policy.timeRestrictions;

    if (start <= end) {
      return hour >= start && hour <= end;
    } else {
      // Handles overnight ranges like 22-6
      return hour >= start || hour <= end;
    }
  }

  isTrustedLocation(location: string): boolean {
    return this.trustedLocations.has(location.toLowerCase());
  }

  isTrustedDevice(deviceId: string): boolean {
    return this.trustedDevices.has(deviceId);
  }

  getAccessDenialReasons(context: RequestContext, policy: AccessPolicy): string[] {
    const reasons: string[] = [];
    const trustScore = this.calculateTrustScore(context);

    if (trustScore < policy.requiredTrustScore) {
      reasons.push('Insufficient trust score');
    }

    if (policy.allowedLocations && !policy.allowedLocations.includes(context.location)) {
      reasons.push('Location not allowed');
    }

    if (policy.allowedDeviceTypes && !policy.allowedDeviceTypes.includes(context.deviceTrust)) {
      reasons.push('Device type not allowed');
    }

    if (policy.timeRestrictions && !this.isWithinTimeRestriction(policy, context.timestamp.getHours())) {
      reasons.push('Outside allowed time window');
    }

    return reasons;
  }
}

export { ZeroTrustValidator, RequestContext, AccessPolicy, AccessDecision };`,
	hint1: `For calculateTrustScore, assign points for each positive signal: authenticated user +20, managed device +30, trusted location +20, business hours +10, etc. Cap at 100.`,
	hint2: `For validateAccess, check trust score against policy's requiredTrustScore, then check location, device type, and time restrictions. Collect all denial reasons.`,
	testCode: `import { ZeroTrustValidator } from './solution';

const createContext = (overrides: Partial<RequestContext> = {}): RequestContext => ({
  userId: 'user1',
  deviceId: 'device1',
  deviceTrust: 'managed',
  location: 'Office',
  ipAddress: '8.8.8.8',
  timestamp: new Date('2024-01-15T10:00:00'),
  resource: '/api/data',
  action: 'read',
  ...overrides,
});

// Test1: calculateTrustScore returns number 0-100
test('Test1', () => {
  const validator = new ZeroTrustValidator();
  const score = validator.calculateTrustScore(createContext());
  expect(score).toBeGreaterThanOrEqual(0);
  expect(score).toBeLessThanOrEqual(100);
});

// Test2: Managed device gets higher score than unknown
test('Test2', () => {
  const validator = new ZeroTrustValidator();
  const managedScore = validator.calculateTrustScore(createContext({ deviceTrust: 'managed' }));
  const unknownScore = validator.calculateTrustScore(createContext({ deviceTrust: 'unknown' }));
  expect(managedScore).toBeGreaterThan(unknownScore);
});

// Test3: Trusted location increases score
test('Test3', () => {
  const validator = new ZeroTrustValidator();
  validator.addTrustedLocation('Office');
  const trustedScore = validator.calculateTrustScore(createContext({ location: 'Office' }));
  const untrustedScore = validator.calculateTrustScore(createContext({ location: 'Unknown' }));
  expect(trustedScore).toBeGreaterThan(untrustedScore);
});

// Test4: validateAccess denies without policy
test('Test4', () => {
  const validator = new ZeroTrustValidator();
  const decision = validator.validateAccess(createContext());
  expect(decision.allowed).toBe(false);
  expect(decision.reasons.some(r => r.includes('policy'))).toBe(true);
});

// Test5: validateAccess allows with matching policy
test('Test5', () => {
  const validator = new ZeroTrustValidator();
  validator.addTrustedLocation('Office');
  validator.addPolicy({ resource: '/api/data', requiredTrustScore: 50 });
  const decision = validator.validateAccess(createContext());
  expect(decision.allowed).toBe(true);
});

// Test6: validateAccess denies with low trust score
test('Test6', () => {
  const validator = new ZeroTrustValidator();
  validator.addPolicy({ resource: '/api/data', requiredTrustScore: 99 });
  const decision = validator.validateAccess(createContext({ deviceTrust: 'unknown' }));
  expect(decision.allowed).toBe(false);
});

// Test7: isWithinTimeRestriction works
test('Test7', () => {
  const validator = new ZeroTrustValidator();
  const policy: AccessPolicy = { resource: 'test', requiredTrustScore: 50, timeRestrictions: { start: 9, end: 17 } };
  expect(validator.isWithinTimeRestriction(policy, 10)).toBe(true);
  expect(validator.isWithinTimeRestriction(policy, 20)).toBe(false);
});

// Test8: Location restriction enforced
test('Test8', () => {
  const validator = new ZeroTrustValidator();
  validator.addPolicy({ resource: '/api/data', requiredTrustScore: 0, allowedLocations: ['Office'] });
  const decision = validator.validateAccess(createContext({ location: 'Home' }));
  expect(decision.allowed).toBe(false);
  expect(decision.reasons.some(r => r.includes('Location'))).toBe(true);
});

// Test9: requiresMfa when trust score low
test('Test9', () => {
  const validator = new ZeroTrustValidator();
  validator.addPolicy({ resource: '/api/data', requiredTrustScore: 30 });
  const decision = validator.validateAccess(createContext({ deviceTrust: 'unknown' }));
  expect(decision.requiresMfa).toBe(true);
});

// Test10: isTrustedDevice works
test('Test10', () => {
  const validator = new ZeroTrustValidator();
  validator.addTrustedDevice('device1');
  expect(validator.isTrustedDevice('device1')).toBe(true);
  expect(validator.isTrustedDevice('device2')).toBe(false);
});`,
	whyItMatters: `Zero Trust has become the standard security model for modern enterprises.

**Why Traditional Security Failed:**

2020 SolarWinds Attack showed why perimeter-based security doesn't work:
- Attackers got "inside" via supply chain compromise
- Once inside, they had free access to move laterally
- Trusted software was actually malicious

**Zero Trust Would Have Helped:**

\`\`\`
Traditional: VPN → Inside Network → Trusted → Access Everything

Zero Trust:
Request for /api/sensitive
├─ User Identity: Valid? ✓
├─ Device: Managed? ✓
├─ Device Health: Compliant? ✓
├─ Location: Expected? ✓
├─ Time: Normal hours? ✓
├─ Behavior: Normal pattern? ✓
├─ Trust Score: 85/100 ✓
├─ Resource sensitivity: High
├─ Decision: Allow with MFA
└─ Logging: Full audit trail
\`\`\`

**Zero Trust Implementation:**

| Component | Traditional | Zero Trust |
|-----------|-------------|------------|
| Network | Flat, trusted | Micro-segmented |
| Identity | Username/password | MFA + contextual |
| Devices | Any on network | Verified, healthy |
| Access | Role-based | Risk-based, dynamic |
| Monitoring | Perimeter logs | All traffic, behavior |`,
	order: 5,
	translations: {
		ru: {
			title: 'Архитектура нулевого доверия',
			description: `Изучите Zero Trust Architecture - "Никогда не доверяй, всегда проверяй."

**Что такое Zero Trust?**

Zero Trust - модель безопасности, требующая строгой проверки личности для каждого человека и устройства, пытающегося получить доступ к ресурсам.

**Основные принципы:**

1. Явная проверка - всегда аутентифицировать и авторизовать
2. Минимальные привилегии - ограничить доступ
3. Предполагать взлом - минимизировать радиус поражения

**Ваша задача:**

Реализуйте класс \`ZeroTrustValidator\`.`,
			hint1: `Для calculateTrustScore начисляйте баллы за каждый положительный сигнал: аутентифицированный пользователь +20, managed device +30, доверенная локация +20.`,
			hint2: `Для validateAccess сравните trust score с requiredTrustScore политики, затем проверьте локацию, тип устройства и временные ограничения.`,
			whyItMatters: `Zero Trust стал стандартной моделью безопасности для современных предприятий.`
		},
		uz: {
			title: 'Nol ishonch arxitekturasi',
			description: `Zero Trust Architecture ni o'rganing - "Hech qachon ishonmang, doimo tekshiring."

**Zero Trust nima?**

Zero Trust - resurslarga kirish uchun har bir shaxs va qurilmani qat'iy tekshirishni talab qiladigan xavfsizlik modeli.

**Sizning vazifangiz:**

\`ZeroTrustValidator\` klassini amalga oshiring.`,
			hint1: `calculateTrustScore uchun har bir ijobiy signal uchun ball bering: autentifikatsiya qilingan foydalanuvchi +20, managed device +30, ishonchli joylashuv +20.`,
			hint2: `validateAccess uchun trust score ni siyosatning requiredTrustScore bilan solishtiring, keyin joylashuv, qurilma turi va vaqt cheklovlarini tekshiring.`,
			whyItMatters: `Zero Trust zamonaviy korxonalar uchun standart xavfsizlik modeliga aylandi.`
		}
	}
};

export default task;
