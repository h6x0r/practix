import { Task } from '../../../../../../../types';

export const task: Task = {
	slug: 'sec-sql-injection',
	title: 'SQL Injection: Detection and Prevention',
	difficulty: 'medium',
	tags: ['security', 'owasp', 'sql-injection', 'typescript'],
	estimatedTime: '45m',
	isPremium: false,
	youtubeUrl: '',
	description: `Learn to detect and prevent SQL Injection - the #1 web vulnerability for decades.

**What is SQL Injection?**

SQL Injection occurs when untrusted data is sent to an interpreter as part of a command or query. The attacker's hostile data can trick the interpreter into executing unintended commands or accessing data without authorization.

**Attack Example:**

\`\`\`typescript
// VULNERABLE CODE
const query = \`SELECT * FROM users WHERE username = '\${username}' AND password = '\${password}'\`;

// Attacker inputs:
// username: admin' --
// password: anything

// Resulting query:
// SELECT * FROM users WHERE username = 'admin' --' AND password = 'anything'
// The -- comments out the password check!
\`\`\`

**Your Task:**

Implement a \`SqlSanitizer\` class that:

1. Detects potential SQL injection patterns in user input
2. Provides safe parameterized query building
3. Escapes special characters when parameterization isn't possible
4. Logs detected injection attempts

**Example Usage:**

\`\`\`typescript
const sanitizer = new SqlSanitizer();

// Detect injection attempts
sanitizer.detectInjection("admin' OR '1'='1"); // true
sanitizer.detectInjection("normal_user");       // false

// Build safe parameterized queries
const query = sanitizer.buildQuery(
  "SELECT * FROM users WHERE id = ? AND status = ?",
  [userId, status]
);

// Escape values (when needed)
sanitizer.escapeValue("O'Brien"); // "O\\'Brien"
\`\`\`

**Requirements:**

1. \`detectInjection(input: string): boolean\` - Detect SQL injection patterns
2. \`buildQuery(template: string, params: any[]): SafeQuery\` - Build parameterized query
3. \`escapeValue(value: string): string\` - Escape special SQL characters
4. \`getDetectedAttempts(): InjectionAttempt[]\` - Get logged attempts`,
	initialCode: `interface SafeQuery {
  sql: string;
  params: any[];
  isSafe: boolean;
}

interface InjectionAttempt {
  input: string;
  pattern: string;
  timestamp: Date;
}

class SqlSanitizer {
  private attempts: InjectionAttempt[] = [];

  // Common SQL injection patterns to detect
  private readonly INJECTION_PATTERNS = [
    /('|"|;|--)/,                    // Basic SQL syntax
    /(\bOR\b|\bAND\b).*?=/i,         // OR/AND with equals
    /\bUNION\b.*?\bSELECT\b/i,       // UNION SELECT
    /\bDROP\b.*?\bTABLE\b/i,         // DROP TABLE
    /\bINSERT\b.*?\bINTO\b/i,        // INSERT INTO
    /\bDELETE\b.*?\bFROM\b/i,        // DELETE FROM
    /\bUPDATE\b.*?\bSET\b/i,         // UPDATE SET
    /\b(EXEC|EXECUTE)\b/i,           // EXEC commands
    /\/\*.*?\*\//,                   // SQL comments
  ];

  detectInjection(input: string): boolean {
    // TODO: Check if input matches any injection patterns
    // If detected, log the attempt and return true
    return false;
  }

  buildQuery(template: string, params: any[]): SafeQuery {
    // TODO: Create a safe parameterized query object
    // Replace ? placeholders with indexed parameters
    // Return SafeQuery with isSafe = true if valid
    return { sql: '', params: [], isSafe: false };
  }

  escapeValue(value: string): string {
    // TODO: Escape special SQL characters
    // Replace ' with \\', \\ with \\\\, etc.
    return value;
  }

  getDetectedAttempts(): InjectionAttempt[] {
    // TODO: Return all logged injection attempts
    return [];
  }

  clearAttempts(): void {
    // TODO: Clear the logged attempts
  }
}

export { SqlSanitizer, SafeQuery, InjectionAttempt };`,
	solutionCode: `interface SafeQuery {
  sql: string;
  params: any[];
  isSafe: boolean;
}

interface InjectionAttempt {
  input: string;
  pattern: string;
  timestamp: Date;
}

class SqlSanitizer {
  private attempts: InjectionAttempt[] = [];

  // Common SQL injection patterns to detect
  private readonly INJECTION_PATTERNS: { regex: RegExp; name: string }[] = [
    { regex: /('|"|;|--)/,                    name: 'SQL syntax characters' },
    { regex: /(\bOR\b|\bAND\b).*?=/i,         name: 'OR/AND injection' },
    { regex: /\bUNION\b.*?\bSELECT\b/i,       name: 'UNION SELECT' },
    { regex: /\bDROP\b.*?\bTABLE\b/i,         name: 'DROP TABLE' },
    { regex: /\bINSERT\b.*?\bINTO\b/i,        name: 'INSERT INTO' },
    { regex: /\bDELETE\b.*?\bFROM\b/i,        name: 'DELETE FROM' },
    { regex: /\bUPDATE\b.*?\bSET\b/i,         name: 'UPDATE SET' },
    { regex: /\b(EXEC|EXECUTE)\b/i,           name: 'EXEC command' },
    { regex: /\/\*.*?\*\//,                   name: 'SQL comment' },
  ];

  // Detect SQL injection patterns in input
  detectInjection(input: string): boolean {
    for (const pattern of this.INJECTION_PATTERNS) {
      if (pattern.regex.test(input)) {
        // Log the detected attempt for auditing
        this.attempts.push({
          input: input.substring(0, 100),  // Limit stored input length
          pattern: pattern.name,
          timestamp: new Date(),
        });
        return true;  // Injection pattern detected
      }
    }
    return false;  // Input appears safe
  }

  // Build a safe parameterized query
  buildQuery(template: string, params: any[]): SafeQuery {
    // Count placeholders in template
    const placeholderCount = (template.match(/\?/g) || []).length;

    // Validate params match placeholders
    if (placeholderCount !== params.length) {
      return {
        sql: template,
        params: params,
        isSafe: false,  // Parameter count mismatch
      };
    }

    // Check for injection in any string parameters
    for (const param of params) {
      if (typeof param === 'string' && this.detectInjection(param)) {
        return {
          sql: template,
          params: params,
          isSafe: false,  // Injection detected in params
        };
      }
    }

    return {
      sql: template,
      params: params,
      isSafe: true,  // Query is safe to execute
    };
  }

  // Escape special SQL characters
  escapeValue(value: string): string {
    return value
      .replace(/\\\\/g, '\\\\\\\\')  // Escape backslashes first
      .replace(/'/g, "\\\\'")         // Escape single quotes
      .replace(/"/g, '\\\\"')         // Escape double quotes
      .replace(/\\x00/g, '\\\\0')     // Escape null bytes
      .replace(/\\n/g, '\\\\n')       // Escape newlines
      .replace(/\\r/g, '\\\\r')       // Escape carriage returns
      .replace(/\\x1a/g, '\\\\Z');    // Escape Ctrl+Z
  }

  // Get all logged injection attempts
  getDetectedAttempts(): InjectionAttempt[] {
    return [...this.attempts];  // Return copy to prevent mutation
  }

  // Clear logged attempts
  clearAttempts(): void {
    this.attempts = [];
  }
}

export { SqlSanitizer, SafeQuery, InjectionAttempt };`,
	hint1: `For detectInjection, loop through INJECTION_PATTERNS and test each regex against the input. If any match, push an attempt object to this.attempts and return true.`,
	hint2: `For buildQuery, count ? in template using match(/\\?/g), compare with params.length. Then check each string param with detectInjection. Return SafeQuery with isSafe based on validation.`,
	testCode: `import { SqlSanitizer } from './solution';

// Test1: Detects basic SQL injection with single quote
test('Test1', () => {
  const sanitizer = new SqlSanitizer();
  expect(sanitizer.detectInjection("admin' OR '1'='1")).toBe(true);
});

// Test2: Safe input passes detection
test('Test2', () => {
  const sanitizer = new SqlSanitizer();
  expect(sanitizer.detectInjection("normal_username")).toBe(false);
});

// Test3: Detects UNION SELECT injection
test('Test3', () => {
  const sanitizer = new SqlSanitizer();
  expect(sanitizer.detectInjection("1 UNION SELECT * FROM users")).toBe(true);
});

// Test4: Detects DROP TABLE injection
test('Test4', () => {
  const sanitizer = new SqlSanitizer();
  expect(sanitizer.detectInjection("1; DROP TABLE users;")).toBe(true);
});

// Test5: buildQuery creates safe query with matching params
test('Test5', () => {
  const sanitizer = new SqlSanitizer();
  const query = sanitizer.buildQuery("SELECT * FROM users WHERE id = ?", [1]);
  expect(query.isSafe).toBe(true);
  expect(query.params).toEqual([1]);
});

// Test6: buildQuery detects param count mismatch
test('Test6', () => {
  const sanitizer = new SqlSanitizer();
  const query = sanitizer.buildQuery("SELECT * FROM users WHERE id = ? AND name = ?", [1]);
  expect(query.isSafe).toBe(false);
});

// Test7: buildQuery detects injection in params
test('Test7', () => {
  const sanitizer = new SqlSanitizer();
  const query = sanitizer.buildQuery("SELECT * FROM users WHERE name = ?", ["admin' --"]);
  expect(query.isSafe).toBe(false);
});

// Test8: escapeValue escapes single quotes
test('Test8', () => {
  const sanitizer = new SqlSanitizer();
  const escaped = sanitizer.escapeValue("O'Brien");
  expect(escaped).toContain("\\\\'");
});

// Test9: getDetectedAttempts returns logged attempts
test('Test9', () => {
  const sanitizer = new SqlSanitizer();
  sanitizer.detectInjection("admin' --");
  const attempts = sanitizer.getDetectedAttempts();
  expect(attempts.length).toBe(1);
  expect(attempts[0].input).toContain("admin");
});

// Test10: clearAttempts clears the log
test('Test10', () => {
  const sanitizer = new SqlSanitizer();
  sanitizer.detectInjection("admin' --");
  sanitizer.clearAttempts();
  expect(sanitizer.getDetectedAttempts().length).toBe(0);
});`,
	whyItMatters: `SQL Injection has been the #1 web vulnerability for over 20 years. Understanding it is critical for any developer.

**Real-World Breaches:**

**1. Heartland Payment Systems (2008)**
\`\`\`
Impact: 130 million credit cards stolen
Method: SQL injection through web form
Cost: $140 million in settlements

The attacker used a simple SQL injection to access
the database containing credit card numbers.
\`\`\`

**2. Sony Pictures (2011)**
\`\`\`
Impact: 77 million PlayStation Network accounts
Method: SQL injection
Cost: $171 million, 23-day outage

SELECT * FROM users WHERE email = '$input'
Attacker input: ' OR '1'='1' --
\`\`\`

**3. Equifax (2017)**
\`\`\`
Impact: 147 million records (SSN, addresses, etc.)
Method: Apache Struts vulnerability + SQL injection
Cost: $700 million settlement

One of the largest data breaches in history.
\`\`\`

**Prevention Strategies:**

| Strategy | Effectiveness | Example |
|----------|---------------|---------|
| Parameterized Queries | ✅ Best | \`db.query("SELECT * FROM users WHERE id = $1", [id])\` |
| ORMs | ✅ Great | Prisma, TypeORM, Sequelize |
| Input Validation | ⚠️ Good | Whitelist allowed characters |
| Escaping | ⚠️ Fallback | \`mysql_real_escape_string()\` |
| WAF | ⚠️ Additional | ModSecurity rules |

**Code Comparison:**

\`\`\`typescript
// ❌ VULNERABLE - String concatenation
const query = \`SELECT * FROM users WHERE id = '\${userId}'\`;

// ✅ SAFE - Parameterized query
const query = await prisma.user.findUnique({ where: { id: userId } });

// ✅ SAFE - Prepared statement
const result = await db.query('SELECT * FROM users WHERE id = $1', [userId]);
\`\`\`

**Key Takeaways:**
1. NEVER concatenate user input into SQL
2. ALWAYS use parameterized queries or ORMs
3. Validate input on the server side
4. Use least privilege database accounts
5. Log and monitor for injection attempts`,
	order: 0,
	translations: {
		ru: {
			title: 'SQL Injection: Обнаружение и предотвращение',
			description: `Научитесь обнаруживать и предотвращать SQL Injection - уязвимость #1 в веб-приложениях на протяжении десятилетий.

**Что такое SQL Injection?**

SQL Injection происходит, когда ненадёжные данные отправляются интерпретатору как часть команды или запроса. Враждебные данные атакующего могут заставить интерпретатор выполнить непредусмотренные команды или получить доступ к данным без авторизации.

**Ваша задача:**

Реализуйте класс \`SqlSanitizer\`, который:

1. Обнаруживает потенциальные паттерны SQL injection во вводе
2. Предоставляет безопасное построение параметризованных запросов
3. Экранирует специальные символы когда параметризация невозможна
4. Логирует обнаруженные попытки инъекции`,
			hint1: `Для detectInjection пройдитесь по INJECTION_PATTERNS и протестируйте каждый regex на вводе. При совпадении добавьте объект попытки в this.attempts и верните true.`,
			hint2: `Для buildQuery подсчитайте ? в шаблоне через match(/\\?/g), сравните с params.length. Затем проверьте каждый строковый параметр через detectInjection.`,
			whyItMatters: `SQL Injection была уязвимостью #1 более 20 лет. Понимание её критически важно для любого разработчика.`
		},
		uz: {
			title: 'SQL Injection: Aniqlash va oldini olish',
			description: `SQL Injection ni aniqlash va oldini olishni o'rganing - o'n yillar davomida #1 veb-zaiflik.

**SQL Injection nima?**

SQL Injection ishonchsiz ma'lumotlar buyruq yoki so'rovning bir qismi sifatida interpretatorga yuborilganda yuz beradi.

**Sizning vazifangiz:**

Quyidagilarni bajaradigan \`SqlSanitizer\` klassini amalga oshiring:

1. Foydalanuvchi kiritishida SQL injection patternlarini aniqlash
2. Xavfsiz parametrlangan so'rovlar yaratish
3. Maxsus belgilarni qochirish
4. Aniqlangan urinishlarni qayd qilish`,
			hint1: `detectInjection uchun INJECTION_PATTERNS bo'ylab yuring va har bir regexni kirishga test qiling.`,
			hint2: `buildQuery uchun shablonda ? ni match(/\\?/g) orqali hisoblang, params.length bilan solishtiring.`,
			whyItMatters: `SQL Injection 20 yildan ortiq davomida #1 zaiflik bo'lib kelgan.`
		}
	}
};

export default task;
