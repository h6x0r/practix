import { Task } from '../../../../../../../types';

export const task: Task = {
	slug: 'sec-idor-prevention',
	title: 'IDOR: Insecure Direct Object References',
	difficulty: 'medium',
	tags: ['security', 'owasp', 'idor', 'access-control', 'typescript'],
	estimatedTime: '35m',
	isPremium: false,
	youtubeUrl: '',
	description: `Learn to prevent IDOR vulnerabilities - a critical access control flaw in OWASP Top 10.

**What is IDOR?**

Insecure Direct Object Reference occurs when an application uses user-supplied input to directly access objects (files, database records, etc.) without proper authorization checks.

**Attack Example:**

\`\`\`typescript
// Vulnerable endpoint
// GET /api/invoices/123
app.get('/api/invoices/:id', (req, res) => {
  const invoice = db.invoices.find(req.params.id);
  res.json(invoice); // No ownership check!
});

// Attacker changes URL:
// GET /api/invoices/456  -> Gets another user's invoice!
// GET /api/invoices/457  -> Iterates through all invoices!
\`\`\`

**Your Task:**

Implement an \`AccessGuard\` class that:

1. Validates resource ownership before access
2. Uses indirect references (mapping user IDs to safe tokens)
3. Implements proper access control checks
4. Logs unauthorized access attempts

**Example Usage:**

\`\`\`typescript
const guard = new AccessGuard();

// Register user's resources
guard.registerResource('user-1', 'invoice', 'inv-123');
guard.registerResource('user-1', 'invoice', 'inv-124');
guard.registerResource('user-2', 'invoice', 'inv-456');

// Check access
guard.canAccess('user-1', 'invoice', 'inv-123'); // true
guard.canAccess('user-1', 'invoice', 'inv-456'); // false - belongs to user-2

// Use indirect references
const token = guard.createIndirectRef('user-1', 'inv-123');
guard.resolveIndirectRef('user-1', token); // 'inv-123'
guard.resolveIndirectRef('user-2', token); // null - not their resource
\`\`\``,
	initialCode: `interface Resource {
  ownerId: string;
  resourceType: string;
  resourceId: string;
}

interface AccessAttempt {
  userId: string;
  resourceId: string;
  allowed: boolean;
  timestamp: Date;
}

class AccessGuard {
  private resources: Map<string, Resource> = new Map();
  private indirectRefs: Map<string, { userId: string; resourceId: string }> = new Map();
  private accessLog: AccessAttempt[] = [];

  registerResource(ownerId: string, resourceType: string, resourceId: string): void {
    // TODO: Register a resource with its owner
    // Store in resources map with resourceId as key
  }

  canAccess(userId: string, resourceType: string, resourceId: string): boolean {
    // TODO: Check if user can access the resource
    // Log the access attempt
    // Return true only if user owns the resource
    return false;
  }

  createIndirectRef(userId: string, resourceId: string): string {
    // TODO: Create an indirect reference (token) for a resource
    // Only works if user owns the resource
    // Map token to resourceId for later resolution
    return '';
  }

  resolveIndirectRef(userId: string, token: string): string | null {
    // TODO: Resolve indirect reference back to resourceId
    // Only if the token was created for this user
    return null;
  }

  getAccessLog(): AccessAttempt[] {
    // TODO: Return access log for auditing
    return [];
  }

  getUnauthorizedAttempts(): AccessAttempt[] {
    // TODO: Return only failed access attempts
    return [];
  }

  validateBatchAccess(userId: string, resourceIds: string[]): Map<string, boolean> {
    // TODO: Check access for multiple resources at once
    // Return map of resourceId -> canAccess
    return new Map();
  }
}

export { AccessGuard, Resource, AccessAttempt };`,
	solutionCode: `interface Resource {
  ownerId: string;
  resourceType: string;
  resourceId: string;
}

interface AccessAttempt {
  userId: string;
  resourceId: string;
  allowed: boolean;
  timestamp: Date;
}

class AccessGuard {
  private resources: Map<string, Resource> = new Map();
  private indirectRefs: Map<string, { userId: string; resourceId: string }> = new Map();
  private accessLog: AccessAttempt[] = [];

  // Register a resource with its owner
  registerResource(ownerId: string, resourceType: string, resourceId: string): void {
    this.resources.set(resourceId, {
      ownerId,
      resourceType,
      resourceId,
    });
  }

  // Check if user can access the resource
  canAccess(userId: string, resourceType: string, resourceId: string): boolean {
    const resource = this.resources.get(resourceId);

    // Check if resource exists and user is owner
    const allowed = resource !== undefined &&
                    resource.ownerId === userId &&
                    resource.resourceType === resourceType;

    // Log the access attempt
    this.accessLog.push({
      userId,
      resourceId,
      allowed,
      timestamp: new Date(),
    });

    return allowed;
  }

  // Create an indirect reference (token) for a resource
  createIndirectRef(userId: string, resourceId: string): string {
    // Verify ownership first
    const resource = this.resources.get(resourceId);
    if (!resource || resource.ownerId !== userId) {
      return ''; // Cannot create ref for resource user doesn't own
    }

    // Generate random token
    const array = new Uint8Array(16);
    crypto.getRandomValues(array);
    const token = Array.from(array, b => b.toString(16).padStart(2, '0')).join('');

    // Store mapping
    this.indirectRefs.set(token, { userId, resourceId });

    return token;
  }

  // Resolve indirect reference back to resourceId
  resolveIndirectRef(userId: string, token: string): string | null {
    const ref = this.indirectRefs.get(token);

    // Check if token exists and belongs to requesting user
    if (!ref || ref.userId !== userId) {
      return null;
    }

    return ref.resourceId;
  }

  // Return access log for auditing
  getAccessLog(): AccessAttempt[] {
    return [...this.accessLog];
  }

  // Return only failed access attempts
  getUnauthorizedAttempts(): AccessAttempt[] {
    return this.accessLog.filter(attempt => !attempt.allowed);
  }

  // Check access for multiple resources at once
  validateBatchAccess(userId: string, resourceIds: string[]): Map<string, boolean> {
    const results = new Map<string, boolean>();

    for (const resourceId of resourceIds) {
      const resource = this.resources.get(resourceId);
      const allowed = resource !== undefined && resource.ownerId === userId;

      results.set(resourceId, allowed);

      // Log each attempt
      this.accessLog.push({
        userId,
        resourceId,
        allowed,
        timestamp: new Date(),
      });
    }

    return results;
  }
}

export { AccessGuard, Resource, AccessAttempt };`,
	hint1: `For canAccess, get the resource from the Map and check if ownerId matches userId. Log every attempt regardless of result for security auditing.`,
	hint2: `For createIndirectRef, first verify ownership, then generate a random token (use crypto.getRandomValues), and store the token->resourceId mapping in indirectRefs Map.`,
	testCode: `import { AccessGuard } from './solution';

// Test1: registerResource stores resource
test('Test1', () => {
  const guard = new AccessGuard();
  guard.registerResource('user-1', 'invoice', 'inv-123');
  expect(guard.canAccess('user-1', 'invoice', 'inv-123')).toBe(true);
});

// Test2: canAccess returns false for non-owner
test('Test2', () => {
  const guard = new AccessGuard();
  guard.registerResource('user-1', 'invoice', 'inv-123');
  expect(guard.canAccess('user-2', 'invoice', 'inv-123')).toBe(false);
});

// Test3: canAccess returns false for non-existent resource
test('Test3', () => {
  const guard = new AccessGuard();
  expect(guard.canAccess('user-1', 'invoice', 'inv-999')).toBe(false);
});

// Test4: canAccess checks resource type
test('Test4', () => {
  const guard = new AccessGuard();
  guard.registerResource('user-1', 'invoice', 'inv-123');
  expect(guard.canAccess('user-1', 'document', 'inv-123')).toBe(false);
});

// Test5: createIndirectRef returns token for owner
test('Test5', () => {
  const guard = new AccessGuard();
  guard.registerResource('user-1', 'invoice', 'inv-123');
  const token = guard.createIndirectRef('user-1', 'inv-123');
  expect(token.length).toBeGreaterThan(0);
});

// Test6: createIndirectRef returns empty for non-owner
test('Test6', () => {
  const guard = new AccessGuard();
  guard.registerResource('user-1', 'invoice', 'inv-123');
  const token = guard.createIndirectRef('user-2', 'inv-123');
  expect(token).toBe('');
});

// Test7: resolveIndirectRef works for correct user
test('Test7', () => {
  const guard = new AccessGuard();
  guard.registerResource('user-1', 'invoice', 'inv-123');
  const token = guard.createIndirectRef('user-1', 'inv-123');
  expect(guard.resolveIndirectRef('user-1', token)).toBe('inv-123');
});

// Test8: resolveIndirectRef fails for different user
test('Test8', () => {
  const guard = new AccessGuard();
  guard.registerResource('user-1', 'invoice', 'inv-123');
  const token = guard.createIndirectRef('user-1', 'inv-123');
  expect(guard.resolveIndirectRef('user-2', token)).toBeNull();
});

// Test9: getUnauthorizedAttempts returns failed attempts
test('Test9', () => {
  const guard = new AccessGuard();
  guard.registerResource('user-1', 'invoice', 'inv-123');
  guard.canAccess('user-2', 'invoice', 'inv-123'); // Unauthorized
  const attempts = guard.getUnauthorizedAttempts();
  expect(attempts.length).toBe(1);
  expect(attempts[0].userId).toBe('user-2');
});

// Test10: validateBatchAccess returns map of results
test('Test10', () => {
  const guard = new AccessGuard();
  guard.registerResource('user-1', 'invoice', 'inv-1');
  guard.registerResource('user-2', 'invoice', 'inv-2');
  const results = guard.validateBatchAccess('user-1', ['inv-1', 'inv-2']);
  expect(results.get('inv-1')).toBe(true);
  expect(results.get('inv-2')).toBe(false);
});`,
	whyItMatters: `IDOR is one of the most common and exploitable vulnerabilities in web applications.

**Real-World IDOR Incidents:**

**1. First American Financial (2019)**
\`\`\`
Impact: 885 million records exposed
Vulnerability: Sequential document IDs in URL
Method: Changing document ID in URL
Data: SSNs, bank accounts, tax records
Fine: $500,000+ settlement
\`\`\`

**2. Parler Social Network (2021)**
\`\`\`
Impact: All posts and videos scraped
Vulnerability: Sequential post IDs, no auth
Method: Incrementing post IDs
Data: 70TB of posts, including deleted
Result: Platform data leaked
\`\`\`

**3. Facebook (2018)**
\`\`\`
Bug: View As feature IDOR
Impact: 50 million access tokens stolen
Method: Viewing posts as other users
Result: Forced logout of 90M users
\`\`\`

**Prevention Strategies:**

| Method | Description | Example |
|--------|-------------|---------|
| Indirect References | Map user-specific tokens to IDs | UUID per user-resource pair |
| Server-side Checks | Always verify ownership | \`WHERE id = ? AND owner_id = ?\` |
| UUIDs | Use unpredictable IDs | \`550e8400-e29b-41d4-a716-446655440000\` |
| Access Control Layer | Centralized authorization | RBAC/ABAC middleware |

**Secure Code Patterns:**

\`\`\`typescript
// ❌ VULNERABLE: No authorization check
app.get('/api/orders/:id', async (req, res) => {
  const order = await db.order.findUnique({
    where: { id: req.params.id }
  });
  res.json(order);
});

// ✅ SECURE: Check ownership
app.get('/api/orders/:id', async (req, res) => {
  const order = await db.order.findFirst({
    where: {
      id: req.params.id,
      userId: req.user.id  // Ownership check!
    }
  });

  if (!order) {
    return res.status(404).json({ error: 'Not found' });
  }

  res.json(order);
});

// ✅ BETTER: Use Prisma policies or similar
const order = await db.order.findUnique({
  where: {
    id: req.params.id,
    // Policy automatically adds: AND userId = currentUser
  }
});
\`\`\`

**Key Principles:**
1. Never trust client-provided resource IDs alone
2. Always verify authorization on the server
3. Use unpredictable identifiers (UUIDs)
4. Implement centralized access control
5. Log and monitor access patterns`,
	order: 4,
	translations: {
		ru: {
			title: 'IDOR: Небезопасные прямые ссылки на объекты',
			description: `Научитесь предотвращать IDOR-уязвимости - критический недостаток контроля доступа в OWASP Top 10.

**Что такое IDOR?**

Insecure Direct Object Reference возникает, когда приложение использует пользовательский ввод для прямого доступа к объектам без проверки авторизации.

**Ваша задача:**

Реализуйте класс \`AccessGuard\`:

1. Валидация владения ресурсом перед доступом
2. Использование косвенных ссылок (маппинг ID на безопасные токены)
3. Правильные проверки контроля доступа
4. Логирование попыток несанкционированного доступа`,
			hint1: `Для canAccess получите ресурс из Map и проверьте, совпадает ли ownerId с userId. Логируйте каждую попытку.`,
			hint2: `Для createIndirectRef сначала проверьте владение, затем сгенерируйте случайный токен и сохраните маппинг token->resourceId.`,
			whyItMatters: `IDOR - одна из самых распространённых и легко эксплуатируемых уязвимостей в веб-приложениях.`
		},
		uz: {
			title: 'IDOR: Xavfsiz bo\'lmagan to\'g\'ridan-to\'g\'ri ob\'ekt havolalari',
			description: `IDOR zaifliklarini oldini olishni o'rganing - OWASP Top 10 da muhim kirish nazorati kamchiligi.

**IDOR nima?**

Insecure Direct Object Reference ilova foydalanuvchi kiritishini avtorizatsiya tekshirisisiz ob'ektlarga to'g'ridan-to'g'ri kirish uchun ishlatganda yuz beradi.

**Sizning vazifangiz:**

\`AccessGuard\` klassini amalga oshiring:

1. Kirishdan oldin resurs egaligini tasdiqlash
2. Bilvosita havolalardan foydalanish
3. To'g'ri kirish nazorati tekshiruvlari
4. Ruxsatsiz kirish urinishlarini qayd qilish`,
			hint1: `canAccess uchun resursni Map dan oling va ownerId userId ga mos kelishini tekshiring.`,
			hint2: `createIndirectRef uchun avval egalikni tekshiring, keyin tasodifiy token yarating.`,
			whyItMatters: `IDOR veb-ilovalarda eng keng tarqalgan va oson ekspluatatsiya qilinadigan zaifliklardan biri.`
		}
	}
};

export default task;
