import { Task } from '../../../../../../../types';

export const task: Task = {
	slug: 'sec-broken-access-control',
	title: 'Broken Access Control: Implementing RBAC',
	difficulty: 'hard',
	tags: ['security', 'owasp', 'rbac', 'access-control', 'authorization', 'typescript'],
	estimatedTime: '45m',
	isPremium: true,
	youtubeUrl: '',
	description: `Learn to prevent Broken Access Control - #1 in OWASP Top 10 2021 (moved up from #5 in 2017).

**What is Broken Access Control?**

Access control enforces policy such that users cannot act outside of their intended permissions. Broken access control is when these restrictions are not properly enforced, allowing unauthorized access to resources or actions.

**Common Access Control Failures:**

1. **Bypassing access controls** - Modifying URL, state, or HTML to access restricted pages
2. **Missing function-level access control** - API endpoints not checking permissions
3. **Privilege escalation** - Acting as admin when logged in as user
4. **Metadata manipulation** - Tampering with JWT, cookies, or hidden fields
5. **CORS misconfiguration** - Allowing access from unauthorized origins

**Your Task:**

Implement an \`AccessController\` class that provides:

1. Role-Based Access Control (RBAC) system
2. Permission checking for resources and actions
3. Role hierarchy support
4. Audit logging of access decisions

**Example Usage:**

\`\`\`typescript
const ac = new AccessController();

// Define roles with permissions
ac.defineRole('admin', ['read', 'write', 'delete', 'manage_users']);
ac.defineRole('editor', ['read', 'write']);
ac.defineRole('viewer', ['read']);

// Assign roles to users
ac.assignRole('user1', 'admin');
ac.assignRole('user2', 'editor');

// Check permissions
ac.can('user1', 'delete'); // true (admin can delete)
ac.can('user2', 'delete'); // false (editor cannot delete)

// Check resource-specific permissions
ac.canAccess('user1', 'article', '123', 'delete'); // Check if user can delete article 123
\`\`\``,
	initialCode: `interface Role {
  name: string;
  permissions: string[];
  inherits?: string[];  // Roles this role inherits from
}

interface AccessDecision {
  allowed: boolean;
  reason: string;
  timestamp: Date;
  userId: string;
  permission: string;
  resource?: string;
}

class AccessController {
  private roles: Map<string, Role> = new Map();
  private userRoles: Map<string, string[]> = new Map();
  private resourceOwners: Map<string, string> = new Map();  // resourceId -> userId
  private accessLog: AccessDecision[] = [];

  defineRole(name: string, permissions: string[], inherits?: string[]): void {
    // TODO: Define a new role with its permissions
    // Support inheritance from other roles
  }

  assignRole(userId: string, roleName: string): boolean {
    // TODO: Assign a role to a user
    // Return false if role doesn't exist
    return false;
  }

  removeRole(userId: string, roleName: string): void {
    // TODO: Remove a role from a user
  }

  can(userId: string, permission: string): boolean {
    // TODO: Check if user has permission via any of their roles
    // Consider role inheritance
    // Log the access decision
    return false;
  }

  canAccess(userId: string, resourceType: string, resourceId: string, action: string): boolean {
    // TODO: Check if user can perform action on specific resource
    // Consider ownership and role-based permissions
    return false;
  }

  setResourceOwner(resourceId: string, userId: string): void {
    // TODO: Set owner of a resource
  }

  isOwner(userId: string, resourceId: string): boolean {
    // TODO: Check if user owns the resource
    return false;
  }

  getEffectivePermissions(userId: string): string[] {
    // TODO: Get all permissions user has through all their roles
    // Include inherited permissions
    return [];
  }

  getAccessLog(): AccessDecision[] {
    // TODO: Return access decision log for auditing
    return [];
  }
}

export { AccessController, Role, AccessDecision };`,
	solutionCode: `interface Role {
  name: string;
  permissions: string[];
  inherits?: string[];  // Roles this role inherits from
}

interface AccessDecision {
  allowed: boolean;
  reason: string;
  timestamp: Date;
  userId: string;
  permission: string;
  resource?: string;
}

class AccessController {
  private roles: Map<string, Role> = new Map();
  private userRoles: Map<string, string[]> = new Map();
  private resourceOwners: Map<string, string> = new Map();  // resourceId -> userId
  private accessLog: AccessDecision[] = [];

  // Define a new role with its permissions
  defineRole(name: string, permissions: string[], inherits?: string[]): void {
    this.roles.set(name, {
      name,
      permissions,
      inherits: inherits || [],
    });
  }

  // Assign a role to a user
  assignRole(userId: string, roleName: string): boolean {
    // Check if role exists
    if (!this.roles.has(roleName)) {
      return false;
    }

    const currentRoles = this.userRoles.get(userId) || [];
    if (!currentRoles.includes(roleName)) {
      currentRoles.push(roleName);
      this.userRoles.set(userId, currentRoles);
    }
    return true;
  }

  // Remove a role from a user
  removeRole(userId: string, roleName: string): void {
    const currentRoles = this.userRoles.get(userId) || [];
    const index = currentRoles.indexOf(roleName);
    if (index > -1) {
      currentRoles.splice(index, 1);
      this.userRoles.set(userId, currentRoles);
    }
  }

  // Get all permissions for a role, including inherited ones
  private getRolePermissions(roleName: string, visited: Set<string> = new Set()): string[] {
    // Prevent circular inheritance
    if (visited.has(roleName)) {
      return [];
    }
    visited.add(roleName);

    const role = this.roles.get(roleName);
    if (!role) {
      return [];
    }

    let permissions = [...role.permissions];

    // Add inherited permissions
    if (role.inherits) {
      for (const inheritedRole of role.inherits) {
        const inheritedPerms = this.getRolePermissions(inheritedRole, visited);
        permissions = [...permissions, ...inheritedPerms];
      }
    }

    return [...new Set(permissions)];  // Remove duplicates
  }

  // Check if user has permission via any of their roles
  can(userId: string, permission: string): boolean {
    const permissions = this.getEffectivePermissions(userId);
    const allowed = permissions.includes(permission);

    // Log the access decision
    this.accessLog.push({
      allowed,
      reason: allowed ? 'Permission granted via role' : 'Permission not found in user roles',
      timestamp: new Date(),
      userId,
      permission,
    });

    return allowed;
  }

  // Check if user can perform action on specific resource
  canAccess(userId: string, resourceType: string, resourceId: string, action: string): boolean {
    // First check ownership
    const isOwner = this.isOwner(userId, resourceId);

    // Owner gets full access to their resources
    if (isOwner) {
      this.accessLog.push({
        allowed: true,
        reason: 'User is resource owner',
        timestamp: new Date(),
        userId,
        permission: action,
        resource: \`\${resourceType}:\${resourceId}\`,
      });
      return true;
    }

    // Check role-based permission
    const permission = \`\${resourceType}:\${action}\`;
    const hasPermission = this.can(userId, permission) || this.can(userId, action);

    if (!hasPermission) {
      this.accessLog.push({
        allowed: false,
        reason: 'User lacks permission and is not owner',
        timestamp: new Date(),
        userId,
        permission: action,
        resource: \`\${resourceType}:\${resourceId}\`,
      });
    }

    return hasPermission;
  }

  // Set owner of a resource
  setResourceOwner(resourceId: string, userId: string): void {
    this.resourceOwners.set(resourceId, userId);
  }

  // Check if user owns the resource
  isOwner(userId: string, resourceId: string): boolean {
    return this.resourceOwners.get(resourceId) === userId;
  }

  // Get all permissions user has through all their roles
  getEffectivePermissions(userId: string): string[] {
    const userRolesList = this.userRoles.get(userId) || [];
    const allPermissions: string[] = [];

    for (const roleName of userRolesList) {
      const rolePermissions = this.getRolePermissions(roleName);
      allPermissions.push(...rolePermissions);
    }

    return [...new Set(allPermissions)];  // Remove duplicates
  }

  // Return access decision log for auditing
  getAccessLog(): AccessDecision[] {
    return [...this.accessLog];
  }
}

export { AccessController, Role, AccessDecision };`,
	hint1: `For defineRole, store the role in the Map with its permissions and inheritance list. For getRolePermissions (helper), recursively get permissions from inherited roles, using a Set to prevent circular inheritance.`,
	hint2: `For can(), get all effective permissions for user via getEffectivePermissions(), then check if the requested permission is in the list. Always log the access decision for auditing.`,
	testCode: `import { AccessController } from './solution';

// Test1: defineRole creates role
test('Test1', () => {
  const ac = new AccessController();
  ac.defineRole('admin', ['read', 'write', 'delete']);
  ac.assignRole('user1', 'admin');
  expect(ac.can('user1', 'read')).toBe(true);
});

// Test2: User without role has no permissions
test('Test2', () => {
  const ac = new AccessController();
  ac.defineRole('admin', ['read', 'write']);
  expect(ac.can('user1', 'read')).toBe(false);
});

// Test3: assignRole returns false for non-existent role
test('Test3', () => {
  const ac = new AccessController();
  expect(ac.assignRole('user1', 'nonexistent')).toBe(false);
});

// Test4: Role inheritance works
test('Test4', () => {
  const ac = new AccessController();
  ac.defineRole('base', ['read']);
  ac.defineRole('extended', ['write'], ['base']);
  ac.assignRole('user1', 'extended');
  expect(ac.can('user1', 'read')).toBe(true);  // Inherited
  expect(ac.can('user1', 'write')).toBe(true); // Direct
});

// Test5: Resource owner has access
test('Test5', () => {
  const ac = new AccessController();
  ac.defineRole('viewer', ['read']);
  ac.assignRole('user1', 'viewer');
  ac.setResourceOwner('doc-123', 'user1');
  expect(ac.canAccess('user1', 'document', 'doc-123', 'delete')).toBe(true);
});

// Test6: Non-owner without permission denied
test('Test6', () => {
  const ac = new AccessController();
  ac.defineRole('viewer', ['read']);
  ac.assignRole('user2', 'viewer');
  ac.setResourceOwner('doc-123', 'user1');
  expect(ac.canAccess('user2', 'document', 'doc-123', 'delete')).toBe(false);
});

// Test7: getEffectivePermissions returns all permissions
test('Test7', () => {
  const ac = new AccessController();
  ac.defineRole('role1', ['read', 'write']);
  ac.defineRole('role2', ['delete']);
  ac.assignRole('user1', 'role1');
  ac.assignRole('user1', 'role2');
  const perms = ac.getEffectivePermissions('user1');
  expect(perms).toContain('read');
  expect(perms).toContain('write');
  expect(perms).toContain('delete');
});

// Test8: removeRole removes permissions
test('Test8', () => {
  const ac = new AccessController();
  ac.defineRole('admin', ['delete']);
  ac.assignRole('user1', 'admin');
  expect(ac.can('user1', 'delete')).toBe(true);
  ac.removeRole('user1', 'admin');
  expect(ac.can('user1', 'delete')).toBe(false);
});

// Test9: Access log records decisions
test('Test9', () => {
  const ac = new AccessController();
  ac.defineRole('viewer', ['read']);
  ac.assignRole('user1', 'viewer');
  ac.can('user1', 'read');
  ac.can('user1', 'write');
  const log = ac.getAccessLog();
  expect(log.length).toBeGreaterThanOrEqual(2);
  expect(log.some(d => d.allowed)).toBe(true);
  expect(log.some(d => !d.allowed)).toBe(true);
});

// Test10: isOwner returns correct value
test('Test10', () => {
  const ac = new AccessController();
  ac.setResourceOwner('res-1', 'user1');
  expect(ac.isOwner('user1', 'res-1')).toBe(true);
  expect(ac.isOwner('user2', 'res-1')).toBe(false);
});`,
	whyItMatters: `Broken Access Control is now #1 in OWASP Top 10 2021 - the most critical web security risk.

**Real-World Access Control Failures:**

**1. Snapchat (2014)**
\`\`\`
Vulnerability: API returned user data without auth
Impact: 4.6 million usernames + phone numbers leaked
Method: Simple API enumeration
Lesson: Always authenticate API endpoints
\`\`\`

**2. Facebook View As (2018)**
\`\`\`
Vulnerability: "View As" feature leaked access tokens
Impact: 50 million accounts compromised
Method: Combining multiple bugs for privilege escalation
Cost: $5 billion FTC settlement
\`\`\`

**3. First American Financial (2019)**
\`\`\`
Vulnerability: Missing access control on document URLs
Impact: 885 million records exposed
Method: Incrementing document ID in URL
Data: SSNs, bank accounts, mortgage documents
\`\`\`

**Access Control Best Practices:**

| Principle | Implementation |
|-----------|----------------|
| Deny by Default | Start with no access, grant explicitly |
| Least Privilege | Give minimum required permissions |
| Defense in Depth | Check permissions at multiple layers |
| Fail Securely | On error, deny access |
| Audit Everything | Log all access decisions |

**RBAC Implementation Patterns:**

\`\`\`typescript
// ❌ BAD: Check in every endpoint
app.delete('/articles/:id', (req, res) => {
  if (req.user.role !== 'admin' && req.user.role !== 'editor') {
    return res.status(403).json({ error: 'Forbidden' });
  }
  // ...
});

// ✅ GOOD: Centralized middleware
const requirePermission = (permission: string) => {
  return (req: Request, res: Response, next: NextFunction) => {
    if (!accessController.can(req.user.id, permission)) {
      logger.warn('ACCESS_DENIED', {
        userId: req.user.id,
        permission,
        resource: req.path,
      });
      return res.status(403).json({ error: 'Forbidden' });
    }
    next();
  };
};

app.delete('/articles/:id',
  requirePermission('article:delete'),
  articleController.delete
);

// ✅ BETTER: Resource-level checks
const requireResourceAccess = (resourceType: string, action: string) => {
  return async (req: Request, res: Response, next: NextFunction) => {
    const resourceId = req.params.id;

    if (!accessController.canAccess(req.user.id, resourceType, resourceId, action)) {
      return res.status(403).json({ error: 'Forbidden' });
    }
    next();
  };
};

// ✅ Also check at data layer
class ArticleService {
  async delete(userId: string, articleId: string) {
    const article = await this.findById(articleId);

    if (!accessController.canAccess(userId, 'article', articleId, 'delete')) {
      throw new ForbiddenException();
    }

    return this.articleRepo.delete(articleId);
  }
}
\`\`\`

**Key Principles:**
1. Implement access control in a centralized, reusable way
2. Deny by default - whitelist, don't blacklist
3. Check both role AND resource ownership
4. Log all access decisions for audit trail
5. Test access control with different user roles`,
	order: 9,
	translations: {
		ru: {
			title: 'Сломанный контроль доступа: Реализация RBAC',
			description: `Научитесь предотвращать Broken Access Control - #1 в OWASP Top 10 2021.

**Что такое Broken Access Control?**

Контроль доступа обеспечивает политики, запрещающие пользователям действовать за пределами их разрешений. Сломанный контроль доступа - когда эти ограничения не применяются должным образом.

**Ваша задача:**

Реализуйте класс \`AccessController\`:

1. Система ролевого контроля доступа (RBAC)
2. Проверка разрешений для ресурсов и действий
3. Поддержка иерархии ролей
4. Аудит-логирование решений о доступе`,
			hint1: `Для defineRole сохраните роль в Map с её разрешениями и списком наследования. Для getRolePermissions рекурсивно получайте разрешения от наследуемых ролей.`,
			hint2: `Для can() получите все эффективные разрешения пользователя через getEffectivePermissions(), затем проверьте наличие запрошенного разрешения. Всегда логируйте решение о доступе.`,
			whyItMatters: `Broken Access Control теперь #1 в OWASP Top 10 2021 - самый критический риск веб-безопасности.`
		},
		uz: {
			title: 'Buzilgan kirish nazorati: RBAC ni amalga oshirish',
			description: `Broken Access Control ni oldini olishni o'rganing - OWASP Top 10 2021 da #1.

**Broken Access Control nima?**

Kirish nazorati foydalanuvchilarning ruxsat berilgan doirasidan tashqarida harakat qilishiga to'sqinlik qiladigan siyosatlarni amalga oshiradi.

**Sizning vazifangiz:**

\`AccessController\` klassini amalga oshiring:

1. Rolga asoslangan kirish nazorati (RBAC) tizimi
2. Resurslar va harakatlar uchun ruxsat tekshirish
3. Rol ierarxiyasini qo'llab-quvvatlash
4. Kirish qarorlarini audit loglash`,
			hint1: `defineRole uchun rolni Map da ruxsatlari va meros ro'yxati bilan saqlang.`,
			hint2: `can() uchun getEffectivePermissions() orqali foydalanuvchining barcha samarali ruxsatlarini oling.`,
			whyItMatters: `Broken Access Control endi OWASP Top 10 2021 da #1 - eng muhim veb xavfsizlik xavfi.`
		}
	}
};

export default task;
