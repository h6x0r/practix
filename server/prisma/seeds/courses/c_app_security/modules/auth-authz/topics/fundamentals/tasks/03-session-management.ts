import { Task } from '../../../../../../../types';

export const task: Task = {
	slug: 'sec-session-management',
	title: 'Secure Session Management',
	difficulty: 'medium',
	tags: ['security', 'sessions', 'authentication', 'cookies', 'typescript'],
	estimatedTime: '35m',
	isPremium: false,
	youtubeUrl: '',
	description: `Learn to implement secure session management - the foundation of web authentication.

**What is Session Management?**

Session management tracks user state between HTTP requests. Poor session management leads to session hijacking, fixation, and unauthorized access.

**Common Session Vulnerabilities:**

1. **Session Fixation** - Attacker sets session ID before user logs in
2. **Session Hijacking** - Stealing session tokens via XSS, network sniffing
3. **Insufficient Session Expiration** - Sessions never expire
4. **Insecure Session Storage** - Predictable IDs, in URL

**Your Task:**

Implement a \`SessionManager\` class that:

1. Generates cryptographically secure session IDs
2. Implements session lifecycle (create, validate, destroy)
3. Enforces security best practices (regeneration, timeout)
4. Provides secure cookie configuration

**Example Usage:**

\`\`\`typescript
const sessionManager = new SessionManager({
  maxAge: 3600000,  // 1 hour
  idleTimeout: 900000,  // 15 min idle
});

// Create session
const sessionId = sessionManager.createSession('user123');

// Validate session
const session = sessionManager.getSession(sessionId);
// { userId: 'user123', createdAt: ..., lastActivity: ... }

// Regenerate on privilege change
const newId = sessionManager.regenerate(sessionId);

// Secure cookie config
sessionManager.getCookieConfig();
// { httpOnly: true, secure: true, sameSite: 'Strict', ... }
\`\`\``,
	initialCode: `interface Session {
  id: string;
  userId: string;
  createdAt: Date;
  lastActivity: Date;
  data: Record<string, any>;
}

interface SessionConfig {
  maxAge?: number;      // Maximum session lifetime (ms)
  idleTimeout?: number; // Inactivity timeout (ms)
  secure?: boolean;     // HTTPS only
}

interface CookieConfig {
  httpOnly: boolean;
  secure: boolean;
  sameSite: 'Strict' | 'Lax' | 'None';
  path: string;
  maxAge: number;
}

class SessionManager {
  private sessions: Map<string, Session> = new Map();
  private config: Required<SessionConfig>;

  constructor(config?: SessionConfig) {
    this.config = {
      maxAge: config?.maxAge || 3600000,     // 1 hour
      idleTimeout: config?.idleTimeout || 900000, // 15 min
      secure: config?.secure ?? true,
    };
  }

  createSession(userId: string, data?: Record<string, any>): string {
    // TODO: Create new session with secure ID
    // Store in sessions map
    return '';
  }

  getSession(sessionId: string): Session | null {
    // TODO: Get session if valid
    // Check expiration and idle timeout
    // Update lastActivity
    return null;
  }

  regenerate(oldSessionId: string): string | null {
    // TODO: Regenerate session ID (keep data)
    // Important for privilege changes
    return null;
  }

  destroySession(sessionId: string): boolean {
    // TODO: Remove session
    return false;
  }

  isValidSession(sessionId: string): boolean {
    // TODO: Check if session exists and not expired
    return false;
  }

  cleanExpiredSessions(): number {
    // TODO: Remove all expired sessions
    return 0;
  }

  getCookieConfig(): CookieConfig {
    // TODO: Return secure cookie configuration
    return {
      httpOnly: false,
      secure: false,
      sameSite: 'None',
      path: '/',
      maxAge: 0,
    };
  }

  private generateSessionId(): string {
    // TODO: Generate cryptographically secure session ID
    return '';
  }
}

export { SessionManager, Session, SessionConfig, CookieConfig };`,
	solutionCode: `interface Session {
  id: string;
  userId: string;
  createdAt: Date;
  lastActivity: Date;
  data: Record<string, any>;
}

interface SessionConfig {
  maxAge?: number;      // Maximum session lifetime (ms)
  idleTimeout?: number; // Inactivity timeout (ms)
  secure?: boolean;     // HTTPS only
}

interface CookieConfig {
  httpOnly: boolean;
  secure: boolean;
  sameSite: 'Strict' | 'Lax' | 'None';
  path: string;
  maxAge: number;
}

class SessionManager {
  private sessions: Map<string, Session> = new Map();
  private config: Required<SessionConfig>;

  constructor(config?: SessionConfig) {
    this.config = {
      maxAge: config?.maxAge || 3600000,     // 1 hour
      idleTimeout: config?.idleTimeout || 900000, // 15 min
      secure: config?.secure ?? true,
    };
  }

  // Create new session with secure ID
  createSession(userId: string, data?: Record<string, any>): string {
    const sessionId = this.generateSessionId();
    const now = new Date();

    const session: Session = {
      id: sessionId,
      userId,
      createdAt: now,
      lastActivity: now,
      data: data || {},
    };

    this.sessions.set(sessionId, session);
    return sessionId;
  }

  // Get session if valid
  getSession(sessionId: string): Session | null {
    const session = this.sessions.get(sessionId);

    if (!session) {
      return null;
    }

    // Check if expired
    if (!this.isValidSession(sessionId)) {
      this.destroySession(sessionId);
      return null;
    }

    // Update last activity
    session.lastActivity = new Date();
    this.sessions.set(sessionId, session);

    return session;
  }

  // Regenerate session ID (keep data) - important for privilege changes
  regenerate(oldSessionId: string): string | null {
    const oldSession = this.sessions.get(oldSessionId);

    if (!oldSession) {
      return null;
    }

    // Create new session with same data
    const newSessionId = this.generateSessionId();
    const now = new Date();

    const newSession: Session = {
      id: newSessionId,
      userId: oldSession.userId,
      createdAt: now,  // Reset creation time
      lastActivity: now,
      data: { ...oldSession.data },
    };

    // Remove old session
    this.sessions.delete(oldSessionId);

    // Add new session
    this.sessions.set(newSessionId, newSession);

    return newSessionId;
  }

  // Remove session
  destroySession(sessionId: string): boolean {
    return this.sessions.delete(sessionId);
  }

  // Check if session exists and not expired
  isValidSession(sessionId: string): boolean {
    const session = this.sessions.get(sessionId);

    if (!session) {
      return false;
    }

    const now = Date.now();
    const createdTime = session.createdAt.getTime();
    const lastActivityTime = session.lastActivity.getTime();

    // Check absolute expiration
    if (now - createdTime > this.config.maxAge) {
      return false;
    }

    // Check idle timeout
    if (now - lastActivityTime > this.config.idleTimeout) {
      return false;
    }

    return true;
  }

  // Remove all expired sessions
  cleanExpiredSessions(): number {
    let cleaned = 0;

    for (const [sessionId] of this.sessions) {
      if (!this.isValidSession(sessionId)) {
        this.sessions.delete(sessionId);
        cleaned++;
      }
    }

    return cleaned;
  }

  // Return secure cookie configuration
  getCookieConfig(): CookieConfig {
    return {
      httpOnly: true,       // Prevent JavaScript access
      secure: this.config.secure,  // HTTPS only in production
      sameSite: 'Strict',   // Prevent CSRF
      path: '/',
      maxAge: this.config.maxAge,
    };
  }

  // Generate cryptographically secure session ID
  private generateSessionId(): string {
    const array = new Uint8Array(32);  // 256 bits
    crypto.getRandomValues(array);
    return Array.from(array, b => b.toString(16).padStart(2, '0')).join('');
  }
}

export { SessionManager, Session, SessionConfig, CookieConfig };`,
	hint1: `For generateSessionId, use crypto.getRandomValues() with 32 bytes (256 bits) and convert to hex string. This ensures cryptographic randomness.`,
	hint2: `For isValidSession, check both absolute expiration (createdAt + maxAge) and idle timeout (lastActivity + idleTimeout). Return false if either is exceeded.`,
	testCode: `import { SessionManager } from './solution';

// Test1: createSession returns session ID
test('Test1', () => {
  const manager = new SessionManager();
  const sessionId = manager.createSession('user123');
  expect(sessionId.length).toBeGreaterThanOrEqual(32);
});

// Test2: getSession returns session data
test('Test2', () => {
  const manager = new SessionManager();
  const sessionId = manager.createSession('user123', { role: 'admin' });
  const session = manager.getSession(sessionId);
  expect(session?.userId).toBe('user123');
  expect(session?.data.role).toBe('admin');
});

// Test3: getSession returns null for invalid ID
test('Test3', () => {
  const manager = new SessionManager();
  const session = manager.getSession('invalid-id');
  expect(session).toBeNull();
});

// Test4: destroySession removes session
test('Test4', () => {
  const manager = new SessionManager();
  const sessionId = manager.createSession('user123');
  manager.destroySession(sessionId);
  expect(manager.getSession(sessionId)).toBeNull();
});

// Test5: regenerate creates new ID
test('Test5', () => {
  const manager = new SessionManager();
  const oldId = manager.createSession('user123', { data: 'test' });
  const newId = manager.regenerate(oldId);
  expect(newId).not.toBe(oldId);
  expect(manager.getSession(oldId)).toBeNull();
  expect(manager.getSession(newId!)?.data.data).toBe('test');
});

// Test6: isValidSession returns true for valid session
test('Test6', () => {
  const manager = new SessionManager();
  const sessionId = manager.createSession('user123');
  expect(manager.isValidSession(sessionId)).toBe(true);
});

// Test7: Expired session is invalid
test('Test7', () => {
  const manager = new SessionManager({ maxAge: 1 }); // 1ms
  const sessionId = manager.createSession('user123');
  // Wait for expiration
  setTimeout(() => {
    expect(manager.isValidSession(sessionId)).toBe(false);
  }, 10);
});

// Test8: getCookieConfig has httpOnly
test('Test8', () => {
  const manager = new SessionManager();
  const config = manager.getCookieConfig();
  expect(config.httpOnly).toBe(true);
  expect(config.sameSite).toBe('Strict');
});

// Test9: Unique session IDs generated
test('Test9', () => {
  const manager = new SessionManager();
  const id1 = manager.createSession('user1');
  const id2 = manager.createSession('user2');
  expect(id1).not.toBe(id2);
});

// Test10: cleanExpiredSessions removes old sessions
test('Test10', () => {
  const manager = new SessionManager({ maxAge: 1, idleTimeout: 1 });
  manager.createSession('user1');
  manager.createSession('user2');
  setTimeout(() => {
    const cleaned = manager.cleanExpiredSessions();
    expect(cleaned).toBe(2);
  }, 10);
});`,
	whyItMatters: `Session management flaws have led to massive account compromises.

**Real-World Session Attacks:**

**1. GitHub (2013)**
\`\`\`
Attack: Session fixation in OAuth flow
Impact: Account takeover possibility
Method: Pre-setting session before OAuth redirect
Fix: Regenerate session on auth state change
\`\`\`

**2. Hotmail/Outlook (Multiple)**
\`\`\`
Attack: XSS → Session theft
Impact: Email account compromise
Method: Steal cookies via malicious email content
Lesson: httpOnly cookies prevent JS access
\`\`\`

**Session Security Checklist:**

| Practice | Implementation |
|----------|----------------|
| Secure IDs | 128+ bits of randomness |
| httpOnly | Cookie flag to prevent XSS theft |
| Secure | Cookie flag for HTTPS only |
| SameSite | Strict/Lax for CSRF protection |
| Regeneration | New ID on login/privilege change |
| Timeout | Absolute max + idle timeout |

**Secure Session Implementation:**

\`\`\`typescript
// Session Creation
const sessionId = crypto.randomBytes(32).toString('hex');

// Cookie Configuration
res.cookie('sessionId', sessionId, {
  httpOnly: true,     // No JavaScript access
  secure: true,       // HTTPS only
  sameSite: 'strict', // CSRF protection
  maxAge: 3600000,    // 1 hour
  path: '/',
  domain: '.myapp.com', // If needed for subdomains
});

// Session Regeneration (CRITICAL on login!)
async function login(user, req, res) {
  // Destroy old session
  await req.session.destroy();

  // Create new session
  req.session.regenerate(() => {
    req.session.userId = user.id;
    req.session.loginTime = Date.now();
  });
}

// Session Validation
function validateSession(req) {
  const session = req.session;

  // Check absolute timeout
  if (Date.now() - session.loginTime > MAX_SESSION_AGE) {
    return false;
  }

  // Check idle timeout
  if (Date.now() - session.lastActivity > IDLE_TIMEOUT) {
    return false;
  }

  // Update activity
  session.lastActivity = Date.now();
  return true;
}
\`\`\``,
	order: 2,
	translations: {
		ru: {
			title: 'Безопасное управление сессиями',
			description: `Научитесь реализовывать безопасное управление сессиями - основу веб-аутентификации.

**Что такое Session Management?**

Управление сессиями отслеживает состояние пользователя между HTTP запросами. Плохое управление сессиями ведёт к перехвату, фиксации сессий и несанкционированному доступу.

**Ваша задача:**

Реализуйте класс \`SessionManager\`:

1. Генерация криптографически безопасных ID сессий
2. Реализация жизненного цикла сессии
3. Соблюдение лучших практик безопасности
4. Предоставление безопасной конфигурации cookies`,
			hint1: `Для generateSessionId используйте crypto.getRandomValues() с 32 байтами (256 бит) и конвертируйте в hex строку.`,
			hint2: `Для isValidSession проверяйте и абсолютное истечение (createdAt + maxAge) и idle timeout (lastActivity + idleTimeout).`,
			whyItMatters: `Ошибки управления сессиями привели к массовым компрометациям аккаунтов.`
		},
		uz: {
			title: 'Xavfsiz sessiya boshqaruvi',
			description: `Xavfsiz sessiya boshqaruvini amalga oshirishni o'rganing - veb autentifikatsiya asosi.

**Session Management nima?**

Sessiya boshqaruvi HTTP so'rovlari orasida foydalanuvchi holatini kuzatib boradi. Yomon sessiya boshqaruvi sessiya o'g'irlash va ruxsatsiz kirishga olib keladi.

**Sizning vazifangiz:**

\`SessionManager\` klassini amalga oshiring:

1. Kriptografik xavfsiz sessiya ID larini yaratish
2. Sessiya hayot siklini amalga oshirish
3. Xavfsizlik eng yaxshi amaliyotlarini ta'minlash
4. Xavfsiz cookie konfiguratsiyasini taqdim etish`,
			hint1: `generateSessionId uchun crypto.getRandomValues() ni 32 bayt (256 bit) bilan ishlating va hex satrga aylantiring.`,
			hint2: `isValidSession uchun absolyut muddati tugash (createdAt + maxAge) va idle timeout (lastActivity + idleTimeout) ni tekshiring.`,
			whyItMatters: `Sessiya boshqaruvi xatolari keng miqyosda akkauntlarni buzilishiga olib keldi.`
		}
	}
};

export default task;
