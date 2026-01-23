import { Task } from '../../../../../../../types';

export const task: Task = {
	slug: 'secure-input-validation',
	title: 'Input Validation Strategies',
	difficulty: 'medium',
	tags: ['security', 'secure-coding', 'validation', 'typescript'],
	estimatedTime: '30m',
	isPremium: false,
	youtubeUrl: '',
	description: `Learn comprehensive input validation strategies to prevent injection attacks.

**The Golden Rule:**

> Never trust user input. Validate everything.

**Validation Strategies:**

| Strategy | Description | Example |
|----------|-------------|---------|
| Whitelist | Only allow known good | /^[a-zA-Z]+$/ |
| Blacklist | Block known bad | Reject <script> |
| Sanitize | Clean dangerous chars | escape HTML |
| Type check | Verify data type | parseInt() |
| Length check | Limit size | max 100 chars |

**Validation Layers:**

\`\`\`
Client-side validation → UX feedback (can be bypassed!)
Server-side validation → Security (REQUIRED)
Database constraints → Last defense
\`\`\`

**Your Task:**

Implement an \`InputValidator\` class with comprehensive validation methods.`,
	initialCode: `interface ValidationResult {
  valid: boolean;
  sanitized?: string;
  errors: string[];
}

interface ValidationRules {
  required?: boolean;
  minLength?: number;
  maxLength?: number;
  pattern?: RegExp;
  allowedChars?: string;
  type?: 'string' | 'email' | 'url' | 'number' | 'alphanumeric';
  customValidator?: (value: string) => boolean;
}

class InputValidator {
  // Common dangerous patterns to detect
  private dangerousPatterns = [
    /<script/i,
    /javascript:/i,
    /on\\w+\\s*=/i,  // onclick=, onerror=, etc.
    /\\x00/,         // Null byte
  ];

  validate(input: string, rules: ValidationRules): ValidationResult {
    // TODO: Validate input against all rules
    return { valid: false, errors: [] };
  }

  sanitizeHTML(input: string): string {
    // TODO: Remove/escape HTML special characters
    return '';
  }

  sanitizeSQL(input: string): string {
    // TODO: Escape SQL special characters (for parameterized queries, use prepared statements instead!)
    return '';
  }

  isValidEmail(email: string): boolean {
    // TODO: Validate email format
    return false;
  }

  isValidURL(url: string): boolean {
    // TODO: Validate URL format and protocol
    return false;
  }

  containsDangerousPatterns(input: string): { dangerous: boolean; patterns: string[] } {
    // TODO: Check for XSS/injection patterns
    return { dangerous: false, patterns: [] };
  }

  normalizeInput(input: string): string {
    // TODO: Normalize unicode, trim, collapse whitespace
    return '';
  }

  validateLength(input: string, min: number, max: number): boolean {
    // TODO: Check length bounds
    return false;
  }

  isAlphanumeric(input: string): boolean {
    // TODO: Check if only letters and numbers
    return false;
  }

  validateJSON(input: string): { valid: boolean; parsed?: any; error?: string } {
    // TODO: Safely parse JSON
    return { valid: false };
  }
}

export { InputValidator, ValidationResult, ValidationRules };`,
	solutionCode: `interface ValidationResult {
  valid: boolean;
  sanitized?: string;
  errors: string[];
}

interface ValidationRules {
  required?: boolean;
  minLength?: number;
  maxLength?: number;
  pattern?: RegExp;
  allowedChars?: string;
  type?: 'string' | 'email' | 'url' | 'number' | 'alphanumeric';
  customValidator?: (value: string) => boolean;
}

class InputValidator {
  private dangerousPatterns = [
    /<script/i,
    /javascript:/i,
    /on\\w+\\s*=/i,
    /\\x00/,
  ];

  validate(input: string, rules: ValidationRules): ValidationResult {
    const errors: string[] = [];
    let sanitized = input;

    // Required check
    if (rules.required && (!input || input.trim() === '')) {
      errors.push('Input is required');
      return { valid: false, errors };
    }

    if (!input) {
      return { valid: !rules.required, sanitized: '', errors };
    }

    // Normalize first
    sanitized = this.normalizeInput(input);

    // Length checks
    if (rules.minLength && sanitized.length < rules.minLength) {
      errors.push(\`Minimum length is \${rules.minLength}\`);
    }

    if (rules.maxLength && sanitized.length > rules.maxLength) {
      errors.push(\`Maximum length is \${rules.maxLength}\`);
    }

    // Type checks
    if (rules.type) {
      switch (rules.type) {
        case 'email':
          if (!this.isValidEmail(sanitized)) {
            errors.push('Invalid email format');
          }
          break;
        case 'url':
          if (!this.isValidURL(sanitized)) {
            errors.push('Invalid URL format');
          }
          break;
        case 'number':
          if (isNaN(Number(sanitized))) {
            errors.push('Must be a number');
          }
          break;
        case 'alphanumeric':
          if (!this.isAlphanumeric(sanitized)) {
            errors.push('Only letters and numbers allowed');
          }
          break;
      }
    }

    // Pattern check
    if (rules.pattern && !rules.pattern.test(sanitized)) {
      errors.push('Input does not match required pattern');
    }

    // Allowed chars
    if (rules.allowedChars) {
      const allowedSet = new Set(rules.allowedChars);
      for (const char of sanitized) {
        if (!allowedSet.has(char)) {
          errors.push(\`Character '\${char}' is not allowed\`);
          break;
        }
      }
    }

    // Custom validator
    if (rules.customValidator && !rules.customValidator(sanitized)) {
      errors.push('Custom validation failed');
    }

    // Check for dangerous patterns
    const danger = this.containsDangerousPatterns(sanitized);
    if (danger.dangerous) {
      errors.push('Potentially dangerous content detected');
    }

    return {
      valid: errors.length === 0,
      sanitized: this.sanitizeHTML(sanitized),
      errors,
    };
  }

  sanitizeHTML(input: string): string {
    const htmlEntities: Record<string, string> = {
      '&': '&amp;',
      '<': '&lt;',
      '>': '&gt;',
      '"': '&quot;',
      "'": '&#x27;',
      '/': '&#x2F;',
    };

    return input.replace(/[&<>"'/]/g, char => htmlEntities[char] || char);
  }

  sanitizeSQL(input: string): string {
    // Note: Always use parameterized queries instead!
    return input
      .replace(/'/g, "''")
      .replace(/\\\\/g, '\\\\\\\\')
      .replace(/\\x00/g, '\\\\0')
      .replace(/\\n/g, '\\\\n')
      .replace(/\\r/g, '\\\\r')
      .replace(/\\x1a/g, '\\\\Z');
  }

  isValidEmail(email: string): boolean {
    const emailRegex = /^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$/;
    return emailRegex.test(email) && email.length <= 254;
  }

  isValidURL(url: string): boolean {
    try {
      const parsed = new URL(url);
      return ['http:', 'https:'].includes(parsed.protocol);
    } catch {
      return false;
    }
  }

  containsDangerousPatterns(input: string): { dangerous: boolean; patterns: string[] } {
    const matchedPatterns: string[] = [];

    for (const pattern of this.dangerousPatterns) {
      if (pattern.test(input)) {
        matchedPatterns.push(pattern.toString());
      }
    }

    return {
      dangerous: matchedPatterns.length > 0,
      patterns: matchedPatterns,
    };
  }

  normalizeInput(input: string): string {
    return input
      .normalize('NFC')       // Normalize unicode
      .trim()                 // Remove leading/trailing whitespace
      .replace(/\\s+/g, ' '); // Collapse multiple whitespace
  }

  validateLength(input: string, min: number, max: number): boolean {
    const length = input.length;
    return length >= min && length <= max;
  }

  isAlphanumeric(input: string): boolean {
    return /^[a-zA-Z0-9]+$/.test(input);
  }

  validateJSON(input: string): { valid: boolean; parsed?: any; error?: string } {
    try {
      const parsed = JSON.parse(input);
      return { valid: true, parsed };
    } catch (e) {
      return { valid: false, error: (e as Error).message };
    }
  }
}

export { InputValidator, ValidationResult, ValidationRules };`,
	hint1: `For sanitizeHTML, replace dangerous characters with HTML entities: & → &amp;, < → &lt;, > → &gt;, " → &quot;`,
	hint2: `For validate, check rules in order: required → length → type → pattern → allowedChars → customValidator → dangerous patterns`,
	testCode: `import { InputValidator } from './solution';

// Test1: validate with required field
test('Test1', () => {
  const validator = new InputValidator();
  const result = validator.validate('', { required: true });
  expect(result.valid).toBe(false);
  expect(result.errors.length).toBeGreaterThan(0);
});

// Test2: sanitizeHTML escapes dangerous chars
test('Test2', () => {
  const validator = new InputValidator();
  const result = validator.sanitizeHTML('<script>alert("xss")</script>');
  expect(result).not.toContain('<script>');
  expect(result).toContain('&lt;');
});

// Test3: isValidEmail validates correctly
test('Test3', () => {
  const validator = new InputValidator();
  expect(validator.isValidEmail('test@example.com')).toBe(true);
  expect(validator.isValidEmail('invalid')).toBe(false);
  expect(validator.isValidEmail('test@')).toBe(false);
});

// Test4: isValidURL validates protocols
test('Test4', () => {
  const validator = new InputValidator();
  expect(validator.isValidURL('https://example.com')).toBe(true);
  expect(validator.isValidURL('javascript:alert(1)')).toBe(false);
  expect(validator.isValidURL('ftp://files.com')).toBe(false);
});

// Test5: containsDangerousPatterns detects XSS
test('Test5', () => {
  const validator = new InputValidator();
  const result = validator.containsDangerousPatterns('<script>alert(1)</script>');
  expect(result.dangerous).toBe(true);
});

// Test6: normalizeInput handles whitespace
test('Test6', () => {
  const validator = new InputValidator();
  const result = validator.normalizeInput('  hello   world  ');
  expect(result).toBe('hello world');
});

// Test7: validateLength checks bounds
test('Test7', () => {
  const validator = new InputValidator();
  expect(validator.validateLength('hello', 1, 10)).toBe(true);
  expect(validator.validateLength('hi', 5, 10)).toBe(false);
});

// Test8: isAlphanumeric works
test('Test8', () => {
  const validator = new InputValidator();
  expect(validator.isAlphanumeric('abc123')).toBe(true);
  expect(validator.isAlphanumeric('abc-123')).toBe(false);
});

// Test9: validateJSON parses safely
test('Test9', () => {
  const validator = new InputValidator();
  const valid = validator.validateJSON('{"key": "value"}');
  expect(valid.valid).toBe(true);
  expect(valid.parsed?.key).toBe('value');

  const invalid = validator.validateJSON('not json');
  expect(invalid.valid).toBe(false);
});

// Test10: validate with multiple rules
test('Test10', () => {
  const validator = new InputValidator();
  const result = validator.validate('test@example.com', {
    required: true,
    type: 'email',
    maxLength: 100,
  });
  expect(result.valid).toBe(true);
});`,
	whyItMatters: `Input validation is your first line of defense against injection attacks.

**Attack Examples Prevented by Validation:**

\`\`\`
SQL Injection:
Input: "'; DROP TABLE users; --"
Validation: Reject special chars, use parameterized queries

XSS:
Input: "<script>document.cookie</script>"
Validation: HTML entity encoding

Command Injection:
Input: "; rm -rf /"
Validation: Whitelist allowed characters

Path Traversal:
Input: "../../etc/passwd"
Validation: Normalize path, reject ..
\`\`\`

**Real Breaches from Poor Validation:**

| Company | Attack | Impact |
|---------|--------|--------|
| Equifax 2017 | Command injection | 147M records |
| British Airways | XSS/Magecart | 500K cards |
| Capital One | SSRF | 100M records |

**Validation Best Practices:**

1. Validate on server (never trust client)
2. Whitelist over blacklist
3. Validate type, length, format, range
4. Sanitize before output (context-aware)
5. Use built-in validators when available`,
	order: 0,
	translations: {
		ru: {
			title: 'Стратегии валидации ввода',
			description: `Изучите комплексные стратегии валидации ввода для предотвращения инъекционных атак.

**Золотое правило:**

> Никогда не доверяйте пользовательскому вводу. Валидируйте всё.

**Стратегии валидации:**

| Стратегия | Описание | Пример |
|-----------|----------|--------|
| Whitelist | Разрешить только известное | /^[a-zA-Z]+$/ |
| Blacklist | Блокировать известное плохое | Отклонить <script> |
| Sanitize | Очистить опасные символы | escape HTML |

**Ваша задача:**

Реализуйте класс \`InputValidator\`.`,
			hint1: `Для sanitizeHTML замените опасные символы на HTML-сущности: & → &amp;, < → &lt;, > → &gt;`,
			hint2: `Для validate проверяйте правила по порядку: required → length → type → pattern → allowedChars → customValidator`,
			whyItMatters: `Валидация ввода - первая линия обороны против инъекционных атак.`
		},
		uz: {
			title: 'Kiritishni tekshirish strategiyalari',
			description: `Injection hujumlarini oldini olish uchun kiritishni tekshirish strategiyalarini o'rganing.

**Oltin qoida:**

> Foydalanuvchi kiritishiga hech qachon ishonmang. Hammasini tekshiring.

**Sizning vazifangiz:**

\`InputValidator\` klassini amalga oshiring.`,
			hint1: `sanitizeHTML uchun xavfli belgilarni HTML entity bilan almashtiring: & → &amp;, < → &lt;, > → &gt;`,
			hint2: `validate uchun qoidalarni tartibda tekshiring: required → length → type → pattern → allowedChars → customValidator`,
			whyItMatters: `Kiritishni tekshirish injection hujumlariga qarshi birinchi himoya chizig'i.`
		}
	}
};

export default task;
