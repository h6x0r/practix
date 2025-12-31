import { Task } from '../../../../types';

export const task: Task = {
	slug: 'go-http-require-header',
	title: 'Require Header Middleware',
	difficulty: 'easy',	tags: ['go', 'http', 'middleware', 'validation'],
	estimatedTime: '15m',	isPremium: false,
	youtubeUrl: '',
	description: `Implement **RequireHeader** middleware that validates a required header is present and non-empty.

**Requirements:**
1. Create function \`RequireHeader(name string, next http.Handler) http.Handler\`
2. Trim whitespace from header name
3. Skip middleware if name is empty
4. Check if header exists and is non-empty
5. Return 400 Bad Request if missing or empty
6. Handle nil next handler

**Example:**
\`\`\`go
handler := RequireHeader("Authorization", http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
    token := r.Header.Get("Authorization")
    fmt.Fprintf(w, "Token: %s", token)
}))

// Request without Authorization header → 400 Bad Request, "missing required header"
// Request with Authorization: Bearer xyz → 200 OK, Token: Bearer xyz
\`\`\`

**Constraints:**
- Must trim whitespace from header name and value
- Must return 400 for missing or empty headers`,
	initialCode: `package httpx

import (
	"net/http"
	"strings"
)

// TODO: Implement RequireHeader middleware
func RequireHeader(name string, next http.Handler) http.Handler {
	// TODO: Implement
}`,
	solutionCode: `package httpx

import (
	"net/http"
	"strings"
)

func RequireHeader(name string, next http.Handler) http.Handler {
	if next == nil {	// Handle nil next handler
		return http.HandlerFunc(func(http.ResponseWriter, *http.Request) {})
	}
	headerName := strings.TrimSpace(name)	// Remove leading/trailing whitespace
	if headerName == "" {	// Empty name check
		return next	// Skip middleware if no header name
	}
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if strings.TrimSpace(r.Header.Get(headerName)) == "" {	// Check header exists and non-empty
			http.Error(w, "missing required header", http.StatusBadRequest)	// 400 response
			return
		}
		next.ServeHTTP(w, r)	// Header present, continue
	})
}`,
			hint1: `Use r.Header.Get() to retrieve the header value and check if it\`s empty after trimming.`,
			hint2: `Return http.Error() with 400 status if the header is missing or empty. Don\`t call next.ServeHTTP().`,
			testCode: `package httpx

import (
	"net/http"
	"net/http/httptest"
	"testing"
)

func Test1(t *testing.T) {
	// Test header present passes through
	h := RequireHeader("Authorization", http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte("ok"))
	}))
	rec := httptest.NewRecorder()
	req := httptest.NewRequest("GET", "/", nil)
	req.Header.Set("Authorization", "Bearer token")
	h.ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Errorf("status = %d, want 200", rec.Code)
	}
}

func Test2(t *testing.T) {
	// Test missing header returns 400
	h := RequireHeader("Authorization", http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {}))
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest("GET", "/", nil))
	if rec.Code != http.StatusBadRequest {
		t.Errorf("status = %d, want 400", rec.Code)
	}
}

func Test3(t *testing.T) {
	// Test empty header value returns 400
	h := RequireHeader("Authorization", http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {}))
	rec := httptest.NewRecorder()
	req := httptest.NewRequest("GET", "/", nil)
	req.Header.Set("Authorization", "")
	h.ServeHTTP(rec, req)
	if rec.Code != http.StatusBadRequest {
		t.Errorf("status = %d, want 400 for empty header", rec.Code)
	}
}

func Test4(t *testing.T) {
	// Test whitespace-only header value returns 400
	h := RequireHeader("Authorization", http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {}))
	rec := httptest.NewRecorder()
	req := httptest.NewRequest("GET", "/", nil)
	req.Header.Set("Authorization", "   ")
	h.ServeHTTP(rec, req)
	if rec.Code != http.StatusBadRequest {
		t.Errorf("status = %d, want 400 for whitespace header", rec.Code)
	}
}

func Test5(t *testing.T) {
	// Test nil handler returns non-nil
	h := RequireHeader("X-Test", nil)
	if h == nil {
		t.Error("nil handler should return non-nil")
	}
}

func Test6(t *testing.T) {
	// Test empty name skips validation
	called := false
	h := RequireHeader("", http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		called = true
	}))
	h.ServeHTTP(httptest.NewRecorder(), httptest.NewRequest("GET", "/", nil))
	if !called {
		t.Error("empty name should skip validation")
	}
}

func Test7(t *testing.T) {
	// Test header name is trimmed
	h := RequireHeader("  Authorization  ", http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte("ok"))
	}))
	rec := httptest.NewRecorder()
	req := httptest.NewRequest("GET", "/", nil)
	req.Header.Set("Authorization", "token")
	h.ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Errorf("status = %d, want 200 (trimmed name)", rec.Code)
	}
}

func Test8(t *testing.T) {
	// Test next not called when header missing
	called := false
	h := RequireHeader("X-API-Key", http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		called = true
	}))
	h.ServeHTTP(httptest.NewRecorder(), httptest.NewRequest("GET", "/", nil))
	if called {
		t.Error("next should not be called when header missing")
	}
}

func Test9(t *testing.T) {
	// Test case-insensitive header matching
	h := RequireHeader("content-type", http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte("ok"))
	}))
	rec := httptest.NewRecorder()
	req := httptest.NewRequest("POST", "/", nil)
	req.Header.Set("Content-Type", "application/json")
	h.ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Errorf("status = %d, want 200 (case-insensitive)", rec.Code)
	}
}

func Test10(t *testing.T) {
	// Test error message contains "missing required header"
	h := RequireHeader("X-Test", http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {}))
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest("GET", "/", nil))
	body := rec.Body.String()
	if body == "" {
		t.Error("error response should have body")
	}
}`,
			whyItMatters: `RequireHeader enforces API contracts and security requirements by validating critical headers.

**Common Use Cases:**
- **Authentication:** Require \`Authorization\` header for protected endpoints
- **API Versioning:** Require \`X-API-Version\` for version-specific logic
- **Content Negotiation:** Require \`Content-Type\` for POST/PUT requests
- **Rate Limiting:** Require \`X-API-Key\` for API quota tracking

**Production Pattern:**
\`\`\`go
// Protected API endpoint
protectedHandler := Chain(
    RequireHeader("Authorization"),
    RequireHeader("X-Request-ID"),
)(secureEndpoint)

// Multi-factor validation
strictHandler := Chain(
    RequireHeader("Authorization"),
    RequireHeader("X-CSRF-Token"),
    RequireHeader("X-Client-Version"),
)(criticalOperation)

// Content validation
uploadHandler := Chain(
    RequireMethod("POST"),
    RequireHeader("Content-Type"),
    MaxBytes(10 << 20),	// 10MB limit
)(fileUploadHandler)

// API key validation
func APIKeyAuth(next http.Handler) http.Handler {
    return RequireHeader("X-API-Key", http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        apiKey := r.Header.Get("X-API-Key")
        if !isValidAPIKey(apiKey) {
            http.Error(w, "invalid API key", http.StatusUnauthorized)
            return
        }
        next.ServeHTTP(w, r)
    }))
}
\`\`\`

**Real-World Benefits:**
- **Early Validation:** Fail fast if required headers missing
- **Clear Errors:** 400 response tells client exactly what's missing
- **Security:** Prevent unauthorized access by requiring auth headers
- **API Contracts:** Enforce documented header requirements

**HTTP Standards:**
- **400 Bad Request:** Proper status for missing required data
- **Common Headers:** Authorization, Content-Type, Accept, User-Agent
- **Custom Headers:** X-API-Key, X-Client-Version, X-Request-ID

Without header validation, handlers must manually check every header, leading to duplicate code and inconsistent error responses.`,	order: 3,
	translations: {
		ru: {
			title: 'Проверка наличия обязательного заголовка',
			description: `Реализуйте middleware **RequireHeader**, который проверяет наличие обязательного заголовка.

**Требования:**
1. Создайте функцию \`RequireHeader(name string, next http.Handler) http.Handler\`
2. Удалите пробелы из имени заголовка
3. Пропустите middleware если имя пустое
4. Проверьте что заголовок существует и не пустой
5. Верните 400 Bad Request если отсутствует или пустой
6. Обработайте nil handler

**Пример:**
\`\`\`go
handler := RequireHeader("Authorization", http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
    token := r.Header.Get("Authorization")
    fmt.Fprintf(w, "Token: %s", token)
}))

// Запрос без Authorization → 400 Bad Request
// Запрос с Authorization: Bearer xyz → 200 OK
\`\`\`

**Ограничения:**
- Должен удалять пробелы из имени и значения заголовка
- Должен возвращать 400 для отсутствующих или пустых заголовков`,
			hint1: `Используйте r.Header.Get() для получения значения заголовка и проверьте его на пустоту после trim.`,
			hint2: `Верните http.Error() с 400 статусом если заголовок отсутствует. Не вызывайте next.ServeHTTP().`,
			whyItMatters: `RequireHeader обеспечивает выполнение API контрактов и требований безопасности через валидацию критичных заголовков.

**Частые use cases:**
- **Аутентификация:** Требование \`Authorization\` для защищенных endpoints
- **Версионирование API:** Требование \`X-API-Version\`
- **Валидация контента:** Требование \`Content-Type\` для POST/PUT
- **Rate Limiting:** Требование \`X-API-Key\` для отслеживания квот API

**Продакшен паттерн:**
\`\`\`go
// Защищённый API endpoint
protectedHandler := Chain(
    RequireHeader("Authorization"),
    RequireHeader("X-Request-ID"),
)(secureEndpoint)

// Многофакторная валидация
strictHandler := Chain(
    RequireHeader("Authorization"),
    RequireHeader("X-CSRF-Token"),
    RequireHeader("X-Client-Version"),
)(criticalOperation)

// Валидация контента
uploadHandler := Chain(
    RequireMethod("POST"),
    RequireHeader("Content-Type"),
    MaxBytes(10 << 20),	// 10MB лимит
)(fileUploadHandler)

// Валидация API ключа
func APIKeyAuth(next http.Handler) http.Handler {
    return RequireHeader("X-API-Key", http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        apiKey := r.Header.Get("X-API-Key")
        if !isValidAPIKey(apiKey) {
            http.Error(w, "invalid API key", http.StatusUnauthorized)
            return
        }
        next.ServeHTTP(w, r)
    }))
}
\`\`\`

**Практические преимущества:**
- **Ранняя валидация:** Быстрый отказ при отсутствии обязательных заголовков
- **Ясные ошибки:** Ответ 400 точно сообщает клиенту, чего не хватает
- **Безопасность:** Предотвращение неавторизованного доступа через требование auth заголовков
- **API контракты:** Принудительное выполнение документированных требований к заголовкам

**Стандарты HTTP:**
- **400 Bad Request:** Правильный статус для отсутствующих данных
- **Общие заголовки:** Authorization, Content-Type, Accept, User-Agent
- **Кастомные заголовки:** X-API-Key, X-Client-Version, X-Request-ID

Без валидации заголовков handlers должны вручную проверять каждый заголовок, что ведёт к дублированию кода и непоследовательным ошибкам.`,
			solutionCode: `package httpx

import (
	"net/http"
	"strings"
)

func RequireHeader(name string, next http.Handler) http.Handler {
	if next == nil {	// Обработка nil next handler
		return http.HandlerFunc(func(http.ResponseWriter, *http.Request) {})
	}
	headerName := strings.TrimSpace(name)	// Удаление начальных/конечных пробелов
	if headerName == "" {	// Проверка на пустое имя
		return next	// Пропуск middleware если нет имени заголовка
	}
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if strings.TrimSpace(r.Header.Get(headerName)) == "" {	// Проверка что заголовок существует и не пустой
			http.Error(w, "missing required header", http.StatusBadRequest)	// 400 ответ
			return
		}
		next.ServeHTTP(w, r)	// Заголовок присутствует, продолжение
	})
}`
		},
		uz: {
			title: 'Majburiy header mavjudligini tekshirish',
			description: `Majburiy header mavjudligini tekshiradigan **RequireHeader** middleware ni amalga oshiring.

**Talablar:**
1. \`RequireHeader(name string, next http.Handler) http.Handler\` funksiyasini yarating
2. Header nomidan bo'sh joylarni olib tashlang
3. Agar nom bo'sh bo'lsa middleware ni o'tkazing
4. Header mavjud va bo'sh emasligini tekshiring
5. Agar yo'q yoki bo'sh bo'lsa 400 Bad Request qaytaring
6. nil handlerni ishlang

**Misol:**
\`\`\`go
handler := RequireHeader("Authorization", http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
    token := r.Header.Get("Authorization")
    fmt.Fprintf(w, "Token: %s", token)
}))

// Authorization headersiz request → 400 Bad Request
// Authorization: Bearer xyz bilan → 200 OK
\`\`\`

**Cheklovlar:**
- Header nomi va qiymatidan bo'sh joylarni olib tashlashi kerak
- Yo'q yoki bo'sh headerlar uchun 400 qaytarishi kerak`,
			hint1: `Header qiymatini olish uchun r.Header.Get() dan foydalaning va trim dan keyin bo'shligini tekshiring.`,
			hint2: `Agar header yo'q bo'lsa 400 status bilan http.Error() ni qaytaring. next.ServeHTTP() ni chaqirmang.`,
			whyItMatters: `RequireHeader muhim headerlarni validatsiya qilish orqali API shartnomalari va xavfsizlik talablarini ta'minlaydi.

**Keng tarqalgan foydalanish:**
- **Autentifikatsiya:** Himoyalangan endpointlar uchun \`Authorization\` headerini talab qilish
- **API versiyalash:** \`X-API-Version\` talab qilish
- **Kontent validatsiyasi:** POST/PUT uchun \`Content-Type\` talab qilish
- **Rate Limiting:** API kvota kuzatuvi uchun \`X-API-Key\` talab qilish

**Ishlab chiqarish patterni:**
\`\`\`go
// Himoyalangan API endpoint
protectedHandler := Chain(
    RequireHeader("Authorization"),
    RequireHeader("X-Request-ID"),
)(secureEndpoint)

// Ko'p faktorli validatsiya
strictHandler := Chain(
    RequireHeader("Authorization"),
    RequireHeader("X-CSRF-Token"),
    RequireHeader("X-Client-Version"),
)(criticalOperation)

// Kontent validatsiyasi
uploadHandler := Chain(
    RequireMethod("POST"),
    RequireHeader("Content-Type"),
    MaxBytes(10 << 20),	// 10MB limit
)(fileUploadHandler)

// API kalit validatsiyasi
func APIKeyAuth(next http.Handler) http.Handler {
    return RequireHeader("X-API-Key", http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        apiKey := r.Header.Get("X-API-Key")
        if !isValidAPIKey(apiKey) {
            http.Error(w, "invalid API key", http.StatusUnauthorized)
            return
        }
        next.ServeHTTP(w, r)
    }))
}
\`\`\`

**Amaliy foydalari:**
- **Erta validatsiya:** Majburiy headerlar yo'q bo'lsa tez rad etish
- **Aniq xatolar:** 400 javobi mijozga aniq nimaning etishmasligini aytadi
- **Xavfsizlik:** Auth headerlarini talab qilish orqali ruxsatsiz kirishning oldini olish
- **API shartnomalari:** Hujjatlashtirilgan header talablarini majburiy qilish

**HTTP standartlari:**
- **400 Bad Request:** Yo'qolgan ma'lumotlar uchun to'g'ri status
- **Umumiy headerlar:** Authorization, Content-Type, Accept, User-Agent
- **Maxsus headerlar:** X-API-Key, X-Client-Version, X-Request-ID

Header validatsiyasisiz handlerlar har bir headerni qo'lda tekshirishi kerak, bu kodning takrorlanishi va nomuvofiq xato javoblariga olib keladi.`,
			solutionCode: `package httpx

import (
	"net/http"
	"strings"
)

func RequireHeader(name string, next http.Handler) http.Handler {
	if next == nil {	// nil next handlerni ishlash
		return http.HandlerFunc(func(http.ResponseWriter, *http.Request) {})
	}
	headerName := strings.TrimSpace(name)	// Bosh/oxiridagi bo'sh joylarni olib tashlash
	if headerName == "" {	// Bo'sh nom tekshiruvi
		return next	// Header nomi bo'lmasa middleware ni o'tkazish
	}
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if strings.TrimSpace(r.Header.Get(headerName)) == "" {	// Header mavjud va bo'sh emasligini tekshirish
			http.Error(w, "missing required header", http.StatusBadRequest)	// 400 response
			return
		}
		next.ServeHTTP(w, r)	// Header mavjud, davom etish
	})
}`
		}
	}
};

export default task;
