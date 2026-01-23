import { Task } from '../../../../../../../types';

export const task: Task = {
	slug: 'auth-rbac-abac',
	title: 'RBAC and ABAC Authorization Models',
	difficulty: 'medium',
	tags: ['security', 'authorization', 'rbac', 'abac', 'typescript'],
	estimatedTime: '35m',
	isPremium: false,
	youtubeUrl: '',
	description: `Learn Role-Based Access Control (RBAC) and Attribute-Based Access Control (ABAC).

**RBAC (Role-Based Access Control):**

Users are assigned roles, roles have permissions.

\`\`\`
User → Role → Permissions

admin → [create, read, update, delete]
editor → [create, read, update]
viewer → [read]
\`\`\`

**ABAC (Attribute-Based Access Control):**

Access decisions based on attributes of user, resource, action, and environment.

\`\`\`
if (user.department === resource.department &&
    user.clearanceLevel >= resource.sensitivityLevel &&
    time.isBusinessHours()) {
  allow();
}
\`\`\`

**RBAC vs ABAC:**

| Feature | RBAC | ABAC |
|---------|------|------|
| Complexity | Simple | Complex |
| Flexibility | Limited | High |
| Scalability | Roles explosion | Handles well |
| Audit | Easy | More complex |

**Your Task:**

Implement both \`RBACManager\` and \`ABACManager\` classes for authorization.`,
	initialCode: `// RBAC Types
interface Role {
  name: string;
  permissions: string[];
  parent?: string; // For role hierarchy
}

interface User {
  id: string;
  roles: string[];
}

// ABAC Types
interface Subject {
  id: string;
  attributes: Record<string, any>;
}

interface Resource {
  id: string;
  type: string;
  attributes: Record<string, any>;
}

interface Environment {
  time: Date;
  ip?: string;
  location?: string;
}

interface PolicyRule {
  id: string;
  effect: 'allow' | 'deny';
  conditions: {
    subject?: Record<string, any>;
    resource?: Record<string, any>;
    action?: string[];
    environment?: Record<string, any>;
  };
}

interface AccessDecision {
  allowed: boolean;
  reason: string;
  matchedRule?: string;
}

class RBACManager {
  private roles: Map<string, Role> = new Map();
  private users: Map<string, User> = new Map();

  addRole(role: Role): void {
    // TODO: Add a role to the system
  }

  assignRole(userId: string, roleName: string): boolean {
    // TODO: Assign a role to a user
    return false;
  }

  removeRole(userId: string, roleName: string): boolean {
    // TODO: Remove a role from a user
    return false;
  }

  hasPermission(userId: string, permission: string): boolean {
    // TODO: Check if user has a specific permission
    // Consider role hierarchy
    return false;
  }

  getUserPermissions(userId: string): string[] {
    // TODO: Get all permissions for a user (including inherited)
    return [];
  }

  getRoleHierarchy(roleName: string): string[] {
    // TODO: Get role and all parent roles
    return [];
  }
}

class ABACManager {
  private policies: PolicyRule[] = [];

  addPolicy(policy: PolicyRule): void {
    // TODO: Add a policy rule
  }

  evaluate(subject: Subject, resource: Resource, action: string, environment: Environment): AccessDecision {
    // TODO: Evaluate access request against all policies
    // Deny takes precedence over allow
    return { allowed: false, reason: '' };
  }

  matchesCondition(actual: Record<string, any>, condition: Record<string, any>): boolean {
    // TODO: Check if actual attributes match condition
    return false;
  }

  getApplicablePolicies(resourceType: string, action: string): PolicyRule[] {
    // TODO: Get policies that apply to this resource/action
    return [];
  }
}

export { RBACManager, ABACManager, Role, User, Subject, Resource, Environment, PolicyRule, AccessDecision };`,
	solutionCode: `// RBAC Types
interface Role {
  name: string;
  permissions: string[];
  parent?: string;
}

interface User {
  id: string;
  roles: string[];
}

// ABAC Types
interface Subject {
  id: string;
  attributes: Record<string, any>;
}

interface Resource {
  id: string;
  type: string;
  attributes: Record<string, any>;
}

interface Environment {
  time: Date;
  ip?: string;
  location?: string;
}

interface PolicyRule {
  id: string;
  effect: 'allow' | 'deny';
  conditions: {
    subject?: Record<string, any>;
    resource?: Record<string, any>;
    action?: string[];
    environment?: Record<string, any>;
  };
}

interface AccessDecision {
  allowed: boolean;
  reason: string;
  matchedRule?: string;
}

class RBACManager {
  private roles: Map<string, Role> = new Map();
  private users: Map<string, User> = new Map();

  addRole(role: Role): void {
    this.roles.set(role.name, role);
  }

  assignRole(userId: string, roleName: string): boolean {
    if (!this.roles.has(roleName)) {
      return false;
    }

    let user = this.users.get(userId);
    if (!user) {
      user = { id: userId, roles: [] };
      this.users.set(userId, user);
    }

    if (!user.roles.includes(roleName)) {
      user.roles.push(roleName);
    }

    return true;
  }

  removeRole(userId: string, roleName: string): boolean {
    const user = this.users.get(userId);
    if (!user) return false;

    const index = user.roles.indexOf(roleName);
    if (index === -1) return false;

    user.roles.splice(index, 1);
    return true;
  }

  hasPermission(userId: string, permission: string): boolean {
    const permissions = this.getUserPermissions(userId);
    return permissions.includes(permission);
  }

  getUserPermissions(userId: string): string[] {
    const user = this.users.get(userId);
    if (!user) return [];

    const allPermissions = new Set<string>();

    for (const roleName of user.roles) {
      const hierarchy = this.getRoleHierarchy(roleName);

      for (const r of hierarchy) {
        const role = this.roles.get(r);
        if (role) {
          role.permissions.forEach(p => allPermissions.add(p));
        }
      }
    }

    return Array.from(allPermissions);
  }

  getRoleHierarchy(roleName: string): string[] {
    const hierarchy: string[] = [];
    let current = roleName;

    while (current) {
      hierarchy.push(current);
      const role = this.roles.get(current);
      current = role?.parent || '';
    }

    return hierarchy;
  }
}

class ABACManager {
  private policies: PolicyRule[] = [];

  addPolicy(policy: PolicyRule): void {
    this.policies.push(policy);
  }

  evaluate(subject: Subject, resource: Resource, action: string, environment: Environment): AccessDecision {
    let allowed = false;
    let denyReason = '';
    let matchedRule: string | undefined;

    for (const policy of this.policies) {
      // Check if policy applies to this action
      if (policy.conditions.action && !policy.conditions.action.includes(action)) {
        continue;
      }

      // Check subject conditions
      if (policy.conditions.subject && !this.matchesCondition(subject.attributes, policy.conditions.subject)) {
        continue;
      }

      // Check resource conditions
      if (policy.conditions.resource && !this.matchesCondition(resource.attributes, policy.conditions.resource)) {
        continue;
      }

      // Check environment conditions
      if (policy.conditions.environment) {
        const envAttrs: Record<string, any> = {
          ip: environment.ip,
          location: environment.location,
          hour: environment.time.getHours(),
        };
        if (!this.matchesCondition(envAttrs, policy.conditions.environment)) {
          continue;
        }
      }

      // Policy matches!
      if (policy.effect === 'deny') {
        return {
          allowed: false,
          reason: 'Denied by policy: ' + policy.id,
          matchedRule: policy.id,
        };
      }

      if (policy.effect === 'allow') {
        allowed = true;
        matchedRule = policy.id;
      }
    }

    return {
      allowed,
      reason: allowed ? 'Allowed by policy' : 'No matching allow policy',
      matchedRule,
    };
  }

  matchesCondition(actual: Record<string, any>, condition: Record<string, any>): boolean {
    for (const [key, value] of Object.entries(condition)) {
      if (typeof value === 'object' && value !== null) {
        // Handle operators like $gt, $lt, $in
        if ('$gt' in value && !(actual[key] > value.$gt)) return false;
        if ('$lt' in value && !(actual[key] < value.$lt)) return false;
        if ('$gte' in value && !(actual[key] >= value.$gte)) return false;
        if ('$lte' in value && !(actual[key] <= value.$lte)) return false;
        if ('$in' in value && !value.$in.includes(actual[key])) return false;
        if ('$eq' in value && actual[key] !== value.$eq) return false;
      } else {
        if (actual[key] !== value) return false;
      }
    }
    return true;
  }

  getApplicablePolicies(resourceType: string, action: string): PolicyRule[] {
    return this.policies.filter(policy => {
      // Check action
      if (policy.conditions.action && !policy.conditions.action.includes(action)) {
        return false;
      }

      // Check resource type if specified
      if (policy.conditions.resource?.type && policy.conditions.resource.type !== resourceType) {
        return false;
      }

      return true;
    });
  }
}

export { RBACManager, ABACManager, Role, User, Subject, Resource, Environment, PolicyRule, AccessDecision };`,
	hint1: `For RBAC hasPermission, get all user roles, traverse role hierarchy for each, collect all permissions, then check if the requested permission is in the set.`,
	hint2: `For ABAC evaluate, iterate through policies, check if conditions match, and remember: deny always takes precedence over allow.`,
	testCode: `import { RBACManager, ABACManager } from './solution';

// Test1: RBAC addRole and assignRole work
test('Test1', () => {
  const rbac = new RBACManager();
  rbac.addRole({ name: 'admin', permissions: ['read', 'write'] });
  expect(rbac.assignRole('user1', 'admin')).toBe(true);
  expect(rbac.assignRole('user1', 'unknown')).toBe(false);
});

// Test2: RBAC hasPermission checks correctly
test('Test2', () => {
  const rbac = new RBACManager();
  rbac.addRole({ name: 'editor', permissions: ['read', 'write'] });
  rbac.assignRole('user1', 'editor');
  expect(rbac.hasPermission('user1', 'read')).toBe(true);
  expect(rbac.hasPermission('user1', 'delete')).toBe(false);
});

// Test3: RBAC role hierarchy works
test('Test3', () => {
  const rbac = new RBACManager();
  rbac.addRole({ name: 'viewer', permissions: ['read'] });
  rbac.addRole({ name: 'editor', permissions: ['write'], parent: 'viewer' });
  rbac.assignRole('user1', 'editor');
  expect(rbac.hasPermission('user1', 'read')).toBe(true);
  expect(rbac.hasPermission('user1', 'write')).toBe(true);
});

// Test4: RBAC removeRole works
test('Test4', () => {
  const rbac = new RBACManager();
  rbac.addRole({ name: 'admin', permissions: ['all'] });
  rbac.assignRole('user1', 'admin');
  expect(rbac.removeRole('user1', 'admin')).toBe(true);
  expect(rbac.hasPermission('user1', 'all')).toBe(false);
});

// Test5: ABAC simple policy evaluation
test('Test5', () => {
  const abac = new ABACManager();
  abac.addPolicy({
    id: 'allow-read',
    effect: 'allow',
    conditions: { action: ['read'] },
  });
  const decision = abac.evaluate(
    { id: 'user1', attributes: {} },
    { id: 'doc1', type: 'document', attributes: {} },
    'read',
    { time: new Date() }
  );
  expect(decision.allowed).toBe(true);
});

// Test6: ABAC deny takes precedence
test('Test6', () => {
  const abac = new ABACManager();
  abac.addPolicy({ id: 'allow-all', effect: 'allow', conditions: { action: ['read'] } });
  abac.addPolicy({ id: 'deny-sensitive', effect: 'deny', conditions: { action: ['read'], resource: { sensitive: true } } });
  const decision = abac.evaluate(
    { id: 'user1', attributes: {} },
    { id: 'doc1', type: 'document', attributes: { sensitive: true } },
    'read',
    { time: new Date() }
  );
  expect(decision.allowed).toBe(false);
});

// Test7: ABAC matchesCondition with operators
test('Test7', () => {
  const abac = new ABACManager();
  expect(abac.matchesCondition({ level: 5 }, { level: { $gte: 3 } })).toBe(true);
  expect(abac.matchesCondition({ level: 2 }, { level: { $gte: 3 } })).toBe(false);
});

// Test8: RBAC getUserPermissions returns all permissions
test('Test8', () => {
  const rbac = new RBACManager();
  rbac.addRole({ name: 'role1', permissions: ['perm1', 'perm2'] });
  rbac.addRole({ name: 'role2', permissions: ['perm3'] });
  rbac.assignRole('user1', 'role1');
  rbac.assignRole('user1', 'role2');
  const perms = rbac.getUserPermissions('user1');
  expect(perms).toContain('perm1');
  expect(perms).toContain('perm3');
});

// Test9: ABAC subject attribute matching
test('Test9', () => {
  const abac = new ABACManager();
  abac.addPolicy({
    id: 'dept-match',
    effect: 'allow',
    conditions: { action: ['read'], subject: { department: 'engineering' } },
  });
  const allowed = abac.evaluate(
    { id: 'user1', attributes: { department: 'engineering' } },
    { id: 'doc1', type: 'doc', attributes: {} },
    'read',
    { time: new Date() }
  );
  expect(allowed.allowed).toBe(true);
});

// Test10: ABAC getApplicablePolicies filters correctly
test('Test10', () => {
  const abac = new ABACManager();
  abac.addPolicy({ id: 'p1', effect: 'allow', conditions: { action: ['read'] } });
  abac.addPolicy({ id: 'p2', effect: 'allow', conditions: { action: ['write'] } });
  const policies = abac.getApplicablePolicies('document', 'read');
  expect(policies.length).toBe(1);
  expect(policies[0].id).toBe('p1');
});`,
	whyItMatters: `Authorization models determine who can do what in your system - get it wrong and you have data breaches.

**Role Explosion Problem:**

\`\`\`
Without hierarchy:
- admin_us_east
- admin_us_west
- admin_eu_east
- admin_eu_west
- editor_us_east
- editor_us_west
... (combinatorial explosion!)

With ABAC:
if (user.role === 'admin' && user.region === resource.region) {
  allow();
}
\`\`\`

**Real-World Example: AWS IAM (ABAC):**

\`\`\`json
{
  "Effect": "Allow",
  "Action": "s3:GetObject",
  "Resource": "arn:aws:s3:::bucket/*",
  "Condition": {
    "StringEquals": {
      "s3:ExistingObjectTag/Department": "\${aws:PrincipalTag/Department}"
    }
  }
}
\`\`\`

**Best Practices:**

1. Start with RBAC, add ABAC when needed
2. Use role hierarchy to reduce duplication
3. Apply least privilege principle
4. Audit authorization decisions
5. Centralize policy management`,
	order: 4,
	translations: {
		ru: {
			title: 'Модели авторизации RBAC и ABAC',
			description: `Изучите Role-Based Access Control (RBAC) и Attribute-Based Access Control (ABAC).

**RBAC (Управление доступом на основе ролей):**

Пользователям назначаются роли, роли имеют разрешения.

**ABAC (Управление доступом на основе атрибутов):**

Решения о доступе на основе атрибутов пользователя, ресурса, действия и окружения.

**RBAC vs ABAC:**

| Свойство | RBAC | ABAC |
|----------|------|------|
| Сложность | Простой | Сложный |
| Гибкость | Ограниченная | Высокая |
| Масштабируемость | Взрыв ролей | Справляется |

**Ваша задача:**

Реализуйте классы \`RBACManager\` и \`ABACManager\`.`,
			hint1: `Для RBAC hasPermission соберите все роли пользователя, пройдите по иерархии каждой, соберите все разрешения и проверьте наличие запрошенного.`,
			hint2: `Для ABAC evaluate итерируйте по политикам, проверяйте условия, и помните: deny всегда имеет приоритет над allow.`,
			whyItMatters: `Модели авторизации определяют, кто что может делать в вашей системе.`
		},
		uz: {
			title: 'RBAC va ABAC avtorizatsiya modellari',
			description: `Role-Based Access Control (RBAC) va Attribute-Based Access Control (ABAC) ni o'rganing.

**RBAC:**
Foydalanuvchilarga rollar tayinlanadi, rollar ruxsatlarga ega.

**ABAC:**
Foydalanuvchi, resurs, harakat va muhit atributlariga asoslangan kirish qarorlari.

**Sizning vazifangiz:**

\`RBACManager\` va \`ABACManager\` klasslarini amalga oshiring.`,
			hint1: `RBAC hasPermission uchun foydalanuvchining barcha rollarini to'plang, har birining iyerarxiyasini o'ting, barcha ruxsatlarni to'plang va so'ralganini tekshiring.`,
			hint2: `ABAC evaluate uchun siyosatlar bo'ylab o'ting, shartlarni tekshiring va esda tuting: deny har doim allow dan ustun.`,
			whyItMatters: `Avtorizatsiya modellari tizimingizda kim nimani qilishi mumkinligini belgilaydi.`
		}
	}
};

export default task;
