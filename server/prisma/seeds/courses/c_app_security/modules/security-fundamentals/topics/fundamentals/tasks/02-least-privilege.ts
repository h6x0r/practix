import { Task } from '../../../../../../../types';

export const task: Task = {
	slug: 'sec-least-privilege',
	title: 'Principle of Least Privilege',
	difficulty: 'easy',
	tags: ['security', 'access-control', 'least-privilege', 'typescript'],
	estimatedTime: '35m',
	isPremium: false,
	youtubeUrl: '',
	description: `Implement the Principle of Least Privilege - users and processes should only have the minimum permissions necessary.

**What is Least Privilege?**

The principle states that every user, program, or process should operate using the minimum privileges necessary to complete the job.

**Why It Matters:**

1. **Limits Blast Radius** - If an account is compromised, damage is contained
2. **Reduces Attack Surface** - Fewer permissions = fewer exploit opportunities
3. **Enables Auditing** - Clear who has access to what

**Your Task:**

Implement a \`PermissionManager\` class that enforces least privilege:

1. Define roles with specific permissions
2. Assign users to roles
3. Check if users can perform actions
4. Track permission requests and denials

**Example Usage:**

\`\`\`typescript
const pm = new PermissionManager();

// Define roles with minimal permissions
pm.defineRole('viewer', ['read']);
pm.defineRole('editor', ['read', 'write']);
pm.defineRole('admin', ['read', 'write', 'delete', 'manage-users']);

// Assign users to roles
pm.assignRole('alice', 'viewer');
pm.assignRole('bob', 'editor');

// Check permissions
pm.canPerform('alice', 'read');    // true
pm.canPerform('alice', 'write');   // false
pm.canPerform('bob', 'write');     // true
pm.canPerform('bob', 'delete');    // false
\`\`\`

**Requirements:**

1. \`defineRole(role, permissions)\` - Create a role with specific permissions
2. \`assignRole(user, role)\` - Assign a user to a role
3. \`canPerform(user, action)\` - Check if user can perform action
4. \`getPermissions(user)\` - Get all permissions for a user
5. \`revokeRole(user, role)\` - Remove a role from user`,
	initialCode: `type Permission = string;
type Role = string;
type User = string;

interface RoleDefinition {
  name: Role;
  permissions: Permission[];
}

class PermissionManager {
  private roles: Map<Role, Permission[]> = new Map();
  private userRoles: Map<User, Role[]> = new Map();

  defineRole(role: Role, permissions: Permission[]): void {
    // TODO: Store the role with its permissions
  }

  assignRole(user: User, role: Role): void {
    // TODO: Assign a role to a user
    // Hint: A user can have multiple roles
  }

  revokeRole(user: User, role: Role): void {
    // TODO: Remove a role from a user
  }

  canPerform(user: User, action: Permission): boolean {
    // TODO: Check if user has the permission through any of their roles
    return false;
  }

  getPermissions(user: User): Permission[] {
    // TODO: Return all unique permissions for a user across all roles
    return [];
  }

  getRoles(user: User): Role[] {
    // TODO: Return all roles assigned to a user
    return [];
  }
}

export { PermissionManager, Permission, Role, User };`,
	solutionCode: `type Permission = string;
type Role = string;
type User = string;

interface RoleDefinition {
  name: Role;
  permissions: Permission[];
}

class PermissionManager {
  private roles: Map<Role, Permission[]> = new Map();  // Role -> Permissions mapping
  private userRoles: Map<User, Role[]> = new Map();    // User -> Roles mapping

  // Define a role with specific permissions (least privilege)
  defineRole(role: Role, permissions: Permission[]): void {
    this.roles.set(role, [...permissions]);  // Store copy of permissions array
  }

  // Assign a role to a user (user can have multiple roles)
  assignRole(user: User, role: Role): void {
    if (!this.roles.has(role)) {
      throw new Error(\`Role '\${role}' does not exist\`);  // Fail fast if role undefined
    }

    const currentRoles = this.userRoles.get(user) || [];  // Get existing roles or empty array

    if (!currentRoles.includes(role)) {
      currentRoles.push(role);  // Add role if not already assigned
      this.userRoles.set(user, currentRoles);  // Update user's roles
    }
  }

  // Remove a role from a user
  revokeRole(user: User, role: Role): void {
    const currentRoles = this.userRoles.get(user) || [];  // Get user's current roles
    const index = currentRoles.indexOf(role);  // Find role index

    if (index > -1) {
      currentRoles.splice(index, 1);  // Remove the role
      this.userRoles.set(user, currentRoles);  // Update user's roles
    }
  }

  // Check if user can perform action (through any of their roles)
  canPerform(user: User, action: Permission): boolean {
    const permissions = this.getPermissions(user);  // Get all user permissions
    return permissions.includes(action);  // Check if action is allowed
  }

  // Get all unique permissions for a user across all roles
  getPermissions(user: User): Permission[] {
    const userRoles = this.userRoles.get(user) || [];  // Get user's roles
    const allPermissions = new Set<Permission>();  // Use Set for uniqueness

    for (const role of userRoles) {
      const rolePermissions = this.roles.get(role) || [];  // Get role's permissions
      rolePermissions.forEach(p => allPermissions.add(p));  // Add each permission
    }

    return Array.from(allPermissions);  // Convert Set to array
  }

  // Return all roles assigned to a user
  getRoles(user: User): Role[] {
    return this.userRoles.get(user) || [];  // Return copy of user's roles
  }
}

export { PermissionManager, Permission, Role, User };`,
	hint1: `For defineRole, use this.roles.set(role, [...permissions]). For assignRole, get existing roles with this.userRoles.get(user) || [], push the new role if not exists, then set it back.`,
	hint2: `For getPermissions, iterate through all user roles, collect their permissions into a Set (for uniqueness), then convert to array. canPerform just checks if the action is in getPermissions result.`,
	testCode: `import { PermissionManager } from './solution';

// Test1: PermissionManager can be instantiated
test('Test1', () => {
  const pm = new PermissionManager();
  expect(pm).toBeDefined();
});

// Test2: Can define roles
test('Test2', () => {
  const pm = new PermissionManager();
  pm.defineRole('viewer', ['read']);
  pm.assignRole('user1', 'viewer');
  expect(pm.getRoles('user1')).toContain('viewer');
});

// Test3: User with viewer role can read
test('Test3', () => {
  const pm = new PermissionManager();
  pm.defineRole('viewer', ['read']);
  pm.assignRole('alice', 'viewer');
  expect(pm.canPerform('alice', 'read')).toBe(true);
});

// Test4: User with viewer role cannot write
test('Test4', () => {
  const pm = new PermissionManager();
  pm.defineRole('viewer', ['read']);
  pm.assignRole('alice', 'viewer');
  expect(pm.canPerform('alice', 'write')).toBe(false);
});

// Test5: Editor can read and write
test('Test5', () => {
  const pm = new PermissionManager();
  pm.defineRole('editor', ['read', 'write']);
  pm.assignRole('bob', 'editor');
  expect(pm.canPerform('bob', 'read')).toBe(true);
  expect(pm.canPerform('bob', 'write')).toBe(true);
});

// Test6: Editor cannot delete
test('Test6', () => {
  const pm = new PermissionManager();
  pm.defineRole('editor', ['read', 'write']);
  pm.assignRole('bob', 'editor');
  expect(pm.canPerform('bob', 'delete')).toBe(false);
});

// Test7: User can have multiple roles
test('Test7', () => {
  const pm = new PermissionManager();
  pm.defineRole('viewer', ['read']);
  pm.defineRole('commenter', ['comment']);
  pm.assignRole('charlie', 'viewer');
  pm.assignRole('charlie', 'commenter');
  expect(pm.getRoles('charlie')).toHaveLength(2);
});

// Test8: Permissions are combined from multiple roles
test('Test8', () => {
  const pm = new PermissionManager();
  pm.defineRole('viewer', ['read']);
  pm.defineRole('commenter', ['comment']);
  pm.assignRole('charlie', 'viewer');
  pm.assignRole('charlie', 'commenter');
  expect(pm.canPerform('charlie', 'read')).toBe(true);
  expect(pm.canPerform('charlie', 'comment')).toBe(true);
});

// Test9: Revoking role removes permissions
test('Test9', () => {
  const pm = new PermissionManager();
  pm.defineRole('editor', ['read', 'write']);
  pm.assignRole('dave', 'editor');
  expect(pm.canPerform('dave', 'write')).toBe(true);
  pm.revokeRole('dave', 'editor');
  expect(pm.canPerform('dave', 'write')).toBe(false);
});

// Test10: Unknown user has no permissions
test('Test10', () => {
  const pm = new PermissionManager();
  pm.defineRole('admin', ['all']);
  expect(pm.canPerform('unknown', 'all')).toBe(false);
  expect(pm.getPermissions('unknown')).toHaveLength(0);
});`,
	whyItMatters: `The Principle of Least Privilege is crucial for limiting damage when security incidents occur.

**Real-World Examples:**

**1. Capital One Breach (2019)**
\`\`\`
What happened:
- AWS WAF misconfiguration allowed SSRF attack
- Attacker accessed EC2 metadata endpoint
- IAM role had excessive permissions (access to 100+ S3 buckets)
- 100 million customer records stolen

Least Privilege Fix:
- Role should only access specific buckets needed
- Instance metadata should be restricted (IMDSv2)
- Separate roles for different services
\`\`\`

**2. SolarWinds Attack (2020)**
\`\`\`
What happened:
- Compromised build server had full network access
- Malicious code spread to 18,000+ organizations
- Attackers moved laterally with elevated privileges

Least Privilege Fix:
- Build servers should only access build artifacts
- Network segmentation limits lateral movement
- Service accounts with minimal permissions
\`\`\`

**Implementation Guidelines:**

| Context | Bad Practice | Good Practice |
|---------|--------------|---------------|
| Database | App uses root account | Separate read/write accounts |
| API Keys | Full access key | Scoped to specific endpoints |
| Cloud IAM | Admin role for all | Role per service with minimal perms |
| File System | 777 permissions | 644 for files, 755 for dirs |
| Containers | Root user | Non-root user with capabilities |

**Code Example - AWS IAM:**
\`\`\`json
// BAD: Too permissive
{
  "Effect": "Allow",
  "Action": "s3:*",
  "Resource": "*"
}

// GOOD: Least privilege
{
  "Effect": "Allow",
  "Action": ["s3:GetObject", "s3:PutObject"],
  "Resource": "arn:aws:s3:::my-app-bucket/*"
}
\`\`\`

**Key Takeaways:**
1. Start with zero permissions, add only what's needed
2. Regularly audit and remove unused permissions
3. Use time-limited credentials where possible
4. Separate permissions by environment (dev/staging/prod)`,
	order: 1,
	translations: {
		ru: {
			title: 'Принцип минимальных привилегий',
			description: `Реализуйте принцип минимальных привилегий - пользователи и процессы должны иметь только минимально необходимые разрешения.

**Что такое минимальные привилегии?**

Принцип гласит, что каждый пользователь, программа или процесс должен работать с минимальными привилегиями, необходимыми для выполнения задачи.

**Почему это важно:**

1. **Ограничивает зону поражения** - При компрометации аккаунта ущерб ограничен
2. **Уменьшает поверхность атаки** - Меньше разрешений = меньше возможностей для эксплуатации
3. **Обеспечивает аудит** - Ясно, кто имеет доступ к чему

**Ваша задача:**

Реализуйте класс \`PermissionManager\`, который обеспечивает минимальные привилегии.`,
			hint1: `Для defineRole используйте this.roles.set(role, [...permissions]). Для assignRole получите существующие роли через this.userRoles.get(user) || [], добавьте новую роль если её нет, затем сохраните обратно.`,
			hint2: `Для getPermissions пройдитесь по всем ролям пользователя, соберите их разрешения в Set (для уникальности), затем конвертируйте в массив.`,
			whyItMatters: `Принцип минимальных привилегий критически важен для ограничения ущерба при инцидентах безопасности.`
		},
		uz: {
			title: 'Minimal imtiyozlar printsipi',
			description: `Minimal imtiyozlar prinsipini amalga oshiring - foydalanuvchilar va jarayonlar faqat zarur bo'lgan minimal ruxsatlarga ega bo'lishi kerak.

**Minimal imtiyozlar nima?**

Prinsip shuni bildiradi: har bir foydalanuvchi, dastur yoki jarayon vazifani bajarish uchun zarur bo'lgan minimal imtiyozlar bilan ishlashi kerak.`,
			hint1: `defineRole uchun this.roles.set(role, [...permissions]) dan foydalaning.`,
			hint2: `getPermissions uchun foydalanuvchining barcha rollari bo'yicha aylaning, ularning ruxsatlarini Set ga yig'ing.`,
			whyItMatters: `Minimal imtiyozlar printsipi xavfsizlik hodisalari yuz berganda zararni cheklash uchun juda muhim.`
		}
	}
};

export default task;
