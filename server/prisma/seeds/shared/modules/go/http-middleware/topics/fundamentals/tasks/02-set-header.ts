import { Task } from '../../../../types';

export const task: Task = {
	slug: 'go-http-set-header',
	title: 'Set Header Middleware',
	difficulty: 'easy',	tags: ['go', 'http', 'middleware', 'headers'],
	estimatedTime: '15m',	isPremium: false,
	youtubeUrl: '',
	description: `Implement **SetHeader** middleware that ensures every response contains a specified header.

**Requirements:**
1. Create function \`SetHeader(name, value string, next http.Handler) http.Handler\`
2. Trim whitespace from header name
3. Skip if header name is empty (return next unchanged)
4. Set header on every response before calling next
5. Handle nil next handler

**Example:**
\`\`\`go
handler := SetHeader("X-Custom-Header", "my-value", http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
    fmt.Fprintf(w, "Header set")
}))

// Every response will have: X-Custom-Header: my-value
\`\`\`

**Constraints:**
- Must trim whitespace from header name
- Must skip middleware if name is empty`,
	initialCode: `package httpx

import (
	"net/http"
	"strings"
)

// TODO: Implement SetHeader middleware
func SetHeader(name, value string, next http.Handler) http.Handler {
	// TODO: Implement
}`,
	solutionCode: `package httpx

import (
	"net/http"
	"strings"
)

func SetHeader(name, value string, next http.Handler) http.Handler {
	if next == nil {	// Handle nil next handler
		return http.HandlerFunc(func(http.ResponseWriter, *http.Request) {})
	}
	headerName := strings.TrimSpace(name)	// Remove leading/trailing whitespace
	if headerName == "" {	// Empty name check
		return next	// Skip middleware if no header name
	}
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set(headerName, value)	// Set header before calling next
		next.ServeHTTP(w, r)	// Continue to next handler
	})
}`,
			hint1: `Use strings.TrimSpace to remove whitespace from the header name.`,
			hint2: `Set the header using w.Header().Set() before calling next.ServeHTTP().`,
			testCode: `package httpx

import (
	"net/http"
	"net/http/httptest"
	"testing"
)

func Test1(t *testing.T) {
	// Test header is set on response
	h := SetHeader("X-Custom", "value", http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {}))
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest("GET", "/", nil))
	if rec.Header().Get("X-Custom") != "value" {
		t.Error("header should be set")
	}
}

func Test2(t *testing.T) {
	// Test nil handler returns non-nil
	h := SetHeader("X-Test", "value", nil)
	if h == nil {
		t.Error("nil handler should return non-nil")
	}
}

func Test3(t *testing.T) {
	// Test empty name returns next unchanged
	next := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte("ok"))
	})
	h := SetHeader("", "value", next)
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest("GET", "/", nil))
	if rec.Body.String() != "ok" {
		t.Error("empty name should pass through to next")
	}
}

func Test4(t *testing.T) {
	// Test whitespace-only name returns next
	next := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {})
	h := SetHeader("   ", "value", next)
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest("GET", "/", nil))
	if rec.Header().Get("   ") != "" && rec.Header().Get("") != "" {
		t.Error("whitespace name should skip header setting")
	}
}

func Test5(t *testing.T) {
	// Test name is trimmed
	h := SetHeader("  X-Trimmed  ", "value", http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {}))
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest("GET", "/", nil))
	if rec.Header().Get("X-Trimmed") != "value" {
		t.Error("header name should be trimmed")
	}
}

func Test6(t *testing.T) {
	// Test next handler is called
	called := false
	h := SetHeader("X-Test", "value", http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		called = true
	}))
	h.ServeHTTP(httptest.NewRecorder(), httptest.NewRequest("GET", "/", nil))
	if !called {
		t.Error("next handler should be called")
	}
}

func Test7(t *testing.T) {
	// Test empty value is set
	h := SetHeader("X-Empty", "", http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {}))
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest("GET", "/", nil))
	if _, exists := rec.Header()["X-Empty"]; !exists {
		t.Error("empty value should still set header")
	}
}

func Test8(t *testing.T) {
	// Test response status preserved
	h := SetHeader("X-Test", "value", http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusNotFound)
	}))
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest("GET", "/", nil))
	if rec.Code != http.StatusNotFound {
		t.Errorf("status = %d, want 404", rec.Code)
	}
}

func Test9(t *testing.T) {
	// Test request is passed through
	var method string
	h := SetHeader("X-Test", "value", http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		method = r.Method
	}))
	h.ServeHTTP(httptest.NewRecorder(), httptest.NewRequest("POST", "/", nil))
	if method != "POST" {
		t.Errorf("method = %q, want POST", method)
	}
}

func Test10(t *testing.T) {
	// Test multiple middleware chaining
	h := SetHeader("X-First", "1", SetHeader("X-Second", "2", http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {})))
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest("GET", "/", nil))
	if rec.Header().Get("X-First") != "1" || rec.Header().Get("X-Second") != "2" {
		t.Error("both headers should be set")
	}
}`,
			whyItMatters: `SetHeader middleware enables consistent header injection across all responses for security, CORS, and API versioning.

**Common Use Cases:**
- **CORS Headers:** \`SetHeader("Access-Control-Allow-Origin", "*", handler)\`
- **Security Headers:** \`SetHeader("X-Frame-Options", "DENY", handler)\`
- **API Versioning:** \`SetHeader("X-API-Version", "v1", handler)\`
- **Cache Control:** \`SetHeader("Cache-Control", "no-cache", handler)\`

**Production Pattern:**
\`\`\`go
// Security headers middleware
func SecurityHeaders(next http.Handler) http.Handler {
    headers := map[string]string{
        "X-Content-Type-Options": "nosniff",
        "X-Frame-Options":        "DENY",
        "X-XSS-Protection":       "1; mode=block",
        "Strict-Transport-Security": "max-age=31536000",
    }

    return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        for name, value := range headers {
            w.Header().Set(name, value)
        }
        next.ServeHTTP(w, r)
    })
}

// CORS middleware
func CORS(origin string, next http.Handler) http.Handler {
    return Chain(
        SetHeader("Access-Control-Allow-Origin", origin),
        SetHeader("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE"),
        SetHeader("Access-Control-Allow-Headers", "Content-Type, Authorization"),
    )(next)
}
\`\`\`

**Real-World Benefits:**
- **Security:** Prevent clickjacking and XSS attacks with security headers
- **Compliance:** Meet security standards (OWASP, PCI-DSS)
- **Browser Behavior:** Control caching, CORS, and content rendering

**Standard Security Headers:**
- \`X-Content-Type-Options: nosniff\` - Prevent MIME sniffing
- \`X-Frame-Options: DENY\` - Prevent clickjacking
- \`Content-Security-Policy\` - Control resource loading

Simple but powerful—consistent headers across your entire API with one middleware.`,	order: 1,
	translations: {
		ru: {
			title: 'Установка HTTP заголовков в ответе',
			description: `Реализуйте middleware **SetHeader**, который гарантирует, что каждый ответ содержит указанный заголовок.

**Требования:**
1. Создайте функцию \`SetHeader(name, value string, next http.Handler) http.Handler\`
2. Удалите пробелы из имени заголовка
3. Пропустите middleware если имя пустое
4. Установите заголовок перед вызовом next
5. Обработайте nil handler

**Пример:**
\`\`\`go
handler := SetHeader("X-Custom-Header", "my-value", http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
    fmt.Fprintf(w, "Header set")
}))

// Каждый ответ будет содержать: X-Custom-Header: my-value
\`\`\`

**Ограничения:**
- Должен удалять пробелы из имени заголовка
- Должен пропускать middleware если имя пустое`,
			hint1: `Используйте strings.TrimSpace для удаления пробелов из имени заголовка.`,
			hint2: `Установите заголовок через w.Header().Set() перед вызовом next.ServeHTTP().`,
			whyItMatters: `SetHeader middleware обеспечивает консистентное добавление заголовков во все ответы для безопасности, CORS и версионирования API.

**Частые use cases:**
- **CORS заголовки:** \`SetHeader("Access-Control-Allow-Origin", "*")\`
- **Заголовки безопасности:** \`SetHeader("X-Frame-Options", "DENY")\`
- **Версионирование API:** \`SetHeader("X-API-Version", "v1")\`
- **Управление кэшем:** \`SetHeader("Cache-Control", "no-cache")\`

**Продакшен паттерн:**
\`\`\`go
// Middleware для заголовков безопасности
func SecurityHeaders(next http.Handler) http.Handler {
    headers := map[string]string{
        "X-Content-Type-Options": "nosniff",
        "X-Frame-Options":        "DENY",
        "X-XSS-Protection":       "1; mode=block",
        "Strict-Transport-Security": "max-age=31536000",
    }

    return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        for name, value := range headers {
            w.Header().Set(name, value)
        }
        next.ServeHTTP(w, r)
    })
}

// CORS middleware
func CORS(origin string, next http.Handler) http.Handler {
    return Chain(
        SetHeader("Access-Control-Allow-Origin", origin),
        SetHeader("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE"),
        SetHeader("Access-Control-Allow-Headers", "Content-Type, Authorization"),
    )(next)
}
\`\`\`

**Практические преимущества:**
- **Безопасность:** Предотвращение clickjacking и XSS атак через заголовки безопасности
- **Соответствие стандартам:** Соблюдение требований безопасности (OWASP, PCI-DSS)
- **Поведение браузера:** Управление кэшированием, CORS и рендерингом контента

**Стандартные заголовки безопасности:**
- \`X-Content-Type-Options: nosniff\` - Предотвращение MIME sniffing
- \`X-Frame-Options: DENY\` - Защита от clickjacking
- \`Content-Security-Policy\` - Контроль загрузки ресурсов

Простой, но мощный инструмент — консистентные заголовки по всему API с одним middleware.`,
			solutionCode: `package httpx

import (
	"net/http"
	"strings"
)

func SetHeader(name, value string, next http.Handler) http.Handler {
	if next == nil {	// Обработка nil next handler
		return http.HandlerFunc(func(http.ResponseWriter, *http.Request) {})
	}
	headerName := strings.TrimSpace(name)	// Удаление начальных/конечных пробелов
	if headerName == "" {	// Проверка на пустое имя
		return next	// Пропуск middleware если нет имени заголовка
	}
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set(headerName, value)	// Установка заголовка перед вызовом next
		next.ServeHTTP(w, r)	// Продолжение к следующему handler
	})
}`
		},
		uz: {
			title: 'Responseda HTTP headerlarni o\'rnatish',
			description: `Har bir response da belgilangan header mavjudligini ta'minlaydigan **SetHeader** middleware ni amalga oshiring.

**Talablar:**
1. \`SetHeader(name, value string, next http.Handler) http.Handler\` funksiyasini yarating
2. Header nomidan bo'sh joylarni olib tashlang
3. Agar nom bo'sh bo'lsa middleware ni o'tkazing
4. next chaqirishdan oldin headerni o'rnating
5. nil handlerni ishlang

**Misol:**
\`\`\`go
handler := SetHeader("X-Custom-Header", "my-value", http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
    fmt.Fprintf(w, "Header set")
}))

// Har bir response: X-Custom-Header: my-value ga ega bo'ladi
\`\`\`

**Cheklovlar:**
- Header nomidan bo'sh joylarni olib tashlashi kerak
- Agar nom bo'sh bo'lsa middleware ni o'tkazishi kerak`,
			hint1: `Header nomidan bo'sh joylarni olib tashlash uchun strings.TrimSpace dan foydalaning.`,
			hint2: `next.ServeHTTP() chaqirishdan oldin w.Header().Set() orqali headerni o'rnating.`,
			whyItMatters: `SetHeader middleware xavfsizlik, CORS va API versiyalash uchun barcha responsega izchil header qo'shishni ta'minlaydi.

**Keng tarqalgan foydalanish holatlari:**
- **CORS headerlar:** \`SetHeader("Access-Control-Allow-Origin", "*")\`
- **Xavfsizlik headerlar:** \`SetHeader("X-Frame-Options", "DENY")\`
- **API versiyalash:** \`SetHeader("X-API-Version", "v1")\`
- **Kesh boshqaruvi:** \`SetHeader("Cache-Control", "no-cache")\`

**Ishlab chiqarish patterni:**
\`\`\`go
// Xavfsizlik headerlar middlewaresi
func SecurityHeaders(next http.Handler) http.Handler {
    headers := map[string]string{
        "X-Content-Type-Options": "nosniff",
        "X-Frame-Options":        "DENY",
        "X-XSS-Protection":       "1; mode=block",
        "Strict-Transport-Security": "max-age=31536000",
    }

    return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        for name, value := range headers {
            w.Header().Set(name, value)
        }
        next.ServeHTTP(w, r)
    })
}

// CORS middleware
func CORS(origin string, next http.Handler) http.Handler {
    return Chain(
        SetHeader("Access-Control-Allow-Origin", origin),
        SetHeader("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE"),
        SetHeader("Access-Control-Allow-Headers", "Content-Type, Authorization"),
    )(next)
}
\`\`\`

**Amaliy foydalari:**
- **Xavfsizlik:** Xavfsizlik headerlari orqali clickjacking va XSS hujumlarining oldini olish
- **Standartlarga muvofiqlik:** Xavfsizlik talablariga rioya qilish (OWASP, PCI-DSS)
- **Brauzer xatti-harakati:** Keshlash, CORS va kontent renderingni boshqarish

**Standart xavfsizlik headerlari:**
- \`X-Content-Type-Options: nosniff\` - MIME sniffingning oldini olish
- \`X-Frame-Options: DENY\` - Clickjackingdan himoya
- \`Content-Security-Policy\` - Resurslarni yuklashni boshqarish

Oddiy, lekin kuchli — bitta middleware bilan butun API bo'ylab izchil headerlar.`,
			solutionCode: `package httpx

import (
	"net/http"
	"strings"
)

func SetHeader(name, value string, next http.Handler) http.Handler {
	if next == nil {	// nil next handlerni ishlash
		return http.HandlerFunc(func(http.ResponseWriter, *http.Request) {})
	}
	headerName := strings.TrimSpace(name)	// Bosh/oxiridagi bo'sh joylarni olib tashlash
	if headerName == "" {	// Bo'sh nom tekshiruvi
		return next	// Header nomi bo'lmasa middleware ni o'tkazish
	}
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set(headerName, value)	// next chaqirishdan oldin headerni o'rnatish
		next.ServeHTTP(w, r)	// Keyingi handlerga davom etish
	})
}`
		}
	}
};

export default task;
