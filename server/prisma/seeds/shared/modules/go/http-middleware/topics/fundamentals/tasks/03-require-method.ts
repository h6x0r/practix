import { Task } from '../../../../types';

export const task: Task = {
	slug: 'go-http-require-method',
	title: 'Require Method Middleware',
	difficulty: 'easy',	tags: ['go', 'http', 'middleware', 'validation'],
	estimatedTime: '20m',	isPremium: false,
	youtubeUrl: '',
	description: `Implement **RequireMethod** middleware that allows only one HTTP method, rejecting others with 405 Method Not Allowed.

**Requirements:**
1. Create function \`RequireMethod(method string, next http.Handler) http.Handler\`
2. Trim whitespace and convert method to uppercase
3. Skip middleware if method is empty
4. Compare request method (case-insensitive)
5. Return 405 with \`Allow\` header if method doesn't match
6. Handle nil next handler

**Example:**
\`\`\`go
handler := RequireMethod("POST", http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
    fmt.Fprintf(w, "POST accepted")
}))

// GET request → 405 Method Not Allowed, Allow: POST
// POST request → 200 OK, POST accepted
\`\`\`

**Constraints:**
- Must be case-insensitive
- Must set Allow header in error response`,
	initialCode: `package httpx

import (
	"net/http"
	"strings"
)

// TODO: Implement RequireMethod middleware
func RequireMethod(method string, next http.Handler) http.Handler {
	// TODO: Implement
}`,
	solutionCode: `package httpx

import (
	"net/http"
	"strings"
)

func RequireMethod(method string, next http.Handler) http.Handler {
	if next == nil {	// Handle nil next handler
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {})
	}
	want := strings.ToUpper(strings.TrimSpace(method))	// Normalize: trim + uppercase
	if want == "" {	// Empty method check
		return next	// Skip middleware if no method specified
	}
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if strings.ToUpper(r.Method) != want {	// Case-insensitive comparison
			w.Header().Set("Allow", want)	// RFC 7231: set Allow header
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)	// 405 response
			return
		}
		next.ServeHTTP(w, r)	// Method matches, continue
	})
}`,
			hint1: `Use strings.ToUpper() to normalize both the expected method and request method for comparison.`,
			hint2: `Set the Allow header and return 405 status if methods don't match. Don't call next.ServeHTTP().`,
			testCode: `package httpx

import (
	"net/http"
	"net/http/httptest"
	"testing"
)

func Test1(t *testing.T) {
	// Test matching method passes through
	h := RequireMethod("GET", http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte("ok"))
	}))
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest("GET", "/", nil))
	if rec.Code != http.StatusOK {
		t.Errorf("status = %d, want 200", rec.Code)
	}
}

func Test2(t *testing.T) {
	// Test non-matching method returns 405
	h := RequireMethod("POST", http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {}))
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest("GET", "/", nil))
	if rec.Code != http.StatusMethodNotAllowed {
		t.Errorf("status = %d, want 405", rec.Code)
	}
}

func Test3(t *testing.T) {
	// Test Allow header is set on 405
	h := RequireMethod("POST", http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {}))
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest("GET", "/", nil))
	if rec.Header().Get("Allow") != "POST" {
		t.Errorf("Allow = %q, want POST", rec.Header().Get("Allow"))
	}
}

func Test4(t *testing.T) {
	// Test case-insensitive matching
	h := RequireMethod("post", http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte("ok"))
	}))
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest("POST", "/", nil))
	if rec.Code != http.StatusOK {
		t.Errorf("status = %d, want 200 (case-insensitive)", rec.Code)
	}
}

func Test5(t *testing.T) {
	// Test nil handler returns non-nil
	h := RequireMethod("GET", nil)
	if h == nil {
		t.Error("nil handler should return non-nil")
	}
}

func Test6(t *testing.T) {
	// Test empty method returns next unchanged
	called := false
	next := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		called = true
	})
	h := RequireMethod("", next)
	h.ServeHTTP(httptest.NewRecorder(), httptest.NewRequest("GET", "/", nil))
	if !called {
		t.Error("empty method should pass through")
	}
}

func Test7(t *testing.T) {
	// Test whitespace-only method skips validation
	called := false
	h := RequireMethod("   ", http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		called = true
	}))
	h.ServeHTTP(httptest.NewRecorder(), httptest.NewRequest("GET", "/", nil))
	if !called {
		t.Error("whitespace method should skip validation")
	}
}

func Test8(t *testing.T) {
	// Test method is trimmed
	h := RequireMethod("  GET  ", http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte("ok"))
	}))
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest("GET", "/", nil))
	if rec.Code != http.StatusOK {
		t.Errorf("status = %d, want 200 (trimmed method)", rec.Code)
	}
}

func Test9(t *testing.T) {
	// Test next not called on method mismatch
	called := false
	h := RequireMethod("POST", http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		called = true
	}))
	h.ServeHTTP(httptest.NewRecorder(), httptest.NewRequest("GET", "/", nil))
	if called {
		t.Error("next should not be called on method mismatch")
	}
}

func Test10(t *testing.T) {
	// Test Allow header uppercased
	h := RequireMethod("post", http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {}))
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest("GET", "/", nil))
	if rec.Header().Get("Allow") != "POST" {
		t.Errorf("Allow = %q, want POST (uppercased)", rec.Header().Get("Allow"))
	}
}`,
			whyItMatters: `RequireMethod enforces HTTP method restrictions, preventing incorrect API usage and improving security.

**Why Method Validation:**
- **API Contracts:** Enforce documented API behavior (e.g., POST for creation)
- **Security:** Prevent CSRF attacks by blocking unexpected methods
- **REST Compliance:** Follow REST conventions (GET=read, POST=create, PUT=update, DELETE=delete)
- **Error Clarity:** Clear 405 response tells clients they're using wrong method

**Production Pattern:**
\`\`\`go
// RESTful API endpoints
mux.Handle("/users", RequireMethod("GET", listUsersHandler))
mux.Handle("/users/create", RequireMethod("POST", createUserHandler))
mux.Handle("/users/{id}", RequireMethod("PUT", updateUserHandler))
mux.Handle("/users/{id}/delete", RequireMethod("DELETE", deleteUserHandler))

// CSRF protection
sensitiveHandler := Chain(
    RequireMethod("POST"),	// Only POST allowed
    RequireHeader("X-CSRF-Token"),	// Must have CSRF token
)(dangerousActionHandler)

// Method routing without framework
func Router() http.Handler {
    return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        switch r.URL.Path {
        case "/api/data":
            RequireMethod("GET", getData).ServeHTTP(w, r)
        case "/api/submit":
            RequireMethod("POST", submitData).ServeHTTP(w, r)
        default:
            http.NotFound(w, r)
        }
    })
}
\`\`\`

**Real-World Benefits:**
- **Browser Safety:** Prevent accidental DELETE via link clicks (browsers use GET for links)
- **CORS Pre-flight:** Proper Allow header helps with CORS preflight requests
- **API Documentation:** Method restrictions are self-documenting API contracts

**HTTP Standards:**
- **405 Method Not Allowed:** Required by RFC 7231
- **Allow Header:** Must list allowed methods
- **Common Methods:** GET, POST, PUT, DELETE, PATCH, OPTIONS, HEAD

Without method validation, APIs accept any method leading to confusing errors and potential security issues.`,	order: 2,
	translations: {
		ru: {
			title: 'Валидация HTTP метода запроса',
			description: `Реализуйте middleware **RequireMethod**, который разрешает только один HTTP метод, отклоняя остальные с 405 Method Not Allowed.

**Требования:**
1. Создайте функцию \`RequireMethod(method string, next http.Handler) http.Handler\`
2. Удалите пробелы и приведите метод к верхнему регистру
3. Пропустите middleware если метод пустой
4. Сравнивайте метод запроса (без учета регистра)
5. Верните 405 с заголовком \`Allow\` если метод не совпадает
6. Обработайте nil handler

**Пример:**
\`\`\`go
handler := RequireMethod("POST", http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
    fmt.Fprintf(w, "POST accepted")
}))

// GET запрос → 405 Method Not Allowed, Allow: POST
// POST запрос → 200 OK, POST accepted
\`\`\`

**Ограничения:**
- Должен быть case-insensitive
- Должен устанавливать Allow header в ответе с ошибкой`,
			hint1: `Используйте strings.ToUpper() для нормализации ожидаемого метода и метода запроса.`,
			hint2: `Установите Allow header и верните 405 статус если методы не совпадают. Не вызывайте next.ServeHTTP().`,
			whyItMatters: `RequireMethod обеспечивает ограничения HTTP методов, предотвращая некорректное использование API и улучшая безопасность.

**Почему важна валидация методов:**
- **API контракты:** Обеспечение документированного поведения API
- **Безопасность:** Предотвращение CSRF атак блокировкой неожиданных методов
- **REST соответствие:** Следование REST конвенциям (GET=чтение, POST=создание)
- **Ясность ошибок:** Чёткий ответ 405 говорит клиентам, что они используют неправильный метод

**Продакшен паттерн:**
\`\`\`go
// RESTful API endpoints
mux.Handle("/users", RequireMethod("GET", listUsersHandler))
mux.Handle("/users/create", RequireMethod("POST", createUserHandler))
mux.Handle("/users/{id}", RequireMethod("PUT", updateUserHandler))
mux.Handle("/users/{id}/delete", RequireMethod("DELETE", deleteUserHandler))

// Защита от CSRF
sensitiveHandler := Chain(
    RequireMethod("POST"),	// Только POST разрешён
    RequireHeader("X-CSRF-Token"),	// Должен быть CSRF токен
)(dangerousActionHandler)

// Роутинг по методам без фреймворка
func Router() http.Handler {
    return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        switch r.URL.Path {
        case "/api/data":
            RequireMethod("GET", getData).ServeHTTP(w, r)
        case "/api/submit":
            RequireMethod("POST", submitData).ServeHTTP(w, r)
        default:
            http.NotFound(w, r)
        }
    })
}
\`\`\`

**Практические преимущества:**
- **Безопасность браузера:** Предотвращение случайного DELETE через клики по ссылкам
- **CORS Pre-flight:** Правильный Allow header помогает с preflight запросами
- **Документация API:** Ограничения методов — самодокументируемые контракты

**Стандарты HTTP:**
- **405 Method Not Allowed:** Требуется по RFC 7231
- **Allow Header:** Должен перечислять разрешённые методы
- **Частые методы:** GET, POST, PUT, DELETE, PATCH, OPTIONS, HEAD

Без валидации методов API принимает любой метод, что ведёт к непонятным ошибкам и проблемам безопасности.`,
			solutionCode: `package httpx

import (
	"net/http"
	"strings"
)

func RequireMethod(method string, next http.Handler) http.Handler {
	if next == nil {	// Обработка nil next handler
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {})
	}
	want := strings.ToUpper(strings.TrimSpace(method))	// Нормализация: trim + верхний регистр
	if want == "" {	// Проверка на пустой метод
		return next	// Пропуск middleware если метод не указан
	}
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if strings.ToUpper(r.Method) != want {	// Сравнение без учета регистра
			w.Header().Set("Allow", want)	// RFC 7231: установка Allow заголовка
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)	// 405 ответ
			return
		}
		next.ServeHTTP(w, r)	// Метод совпадает, продолжение
	})
}`
		},
		uz: {
			title: 'Request HTTP metodini validatsiya qilish',
			description: `Faqat bitta HTTP metodini ruxsat beradigan, qolganlarini 405 Method Not Allowed bilan rad etadigan **RequireMethod** middleware ni amalga oshiring.

**Talablar:**
1. \`RequireMethod(method string, next http.Handler) http.Handler\` funksiyasini yarating
2. Bo'sh joylarni olib tashlang va metodini katta harflarga aylantiring
3. Agar metod bo'sh bo'lsa middleware ni o'tkazing
4. Request metodini solishtiring (katta-kichik harflar e'tiborga olinmaydi)
5. Agar metod mos kelmasa 'Allow' header bilan 405 qaytaring
6. nil handlerni ishlang

**Misol:**
\`\`\`go
handler := RequireMethod("POST", http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
    fmt.Fprintf(w, "POST accepted")
}))

// GET request → 405 Method Not Allowed, Allow: POST
// POST request → 200 OK, POST accepted
\`\`\`

**Cheklovlar:**
- Case-insensitive bo'lishi kerak
- Xato responseda Allow headerni o'rnatishi kerak`,
			hint1: `Kutilayotgan metod va request metodini solishtirish uchun strings.ToUpper() dan foydalaning.`,
			hint2: `Metodlar mos kelmasa Allow headerni o'rnating va 405 statusni qaytaring. next.ServeHTTP() ni chaqirmang.`,
			whyItMatters: `RequireMethod HTTP metod cheklovlarini majburiy qiladi, noto'g'ri API foydalanishning oldini oladi va xavfsizlikni yaxshilaydi.

**Nima uchun metod validatsiyasi:**
- **API shartnomalar:** Hujjatlashtirilgan API xatti-harakatini ta'minlash
- **Xavfsizlik:** Kutilmagan metodlarni bloklash orqali CSRF hujumlarining oldini olish
- **REST muvofiqlik:** REST konventsiyalariga rioya qilish (GET=o'qish, POST=yaratish)
- **Xato aniqligi:** Aniq 405 javobi mijozlarga noto'g'ri metod ishlatayotganini bildiradi

**Ishlab chiqarish patterni:**
\`\`\`go
// RESTful API endpointlari
mux.Handle("/users", RequireMethod("GET", listUsersHandler))
mux.Handle("/users/create", RequireMethod("POST", createUserHandler))
mux.Handle("/users/{id}", RequireMethod("PUT", updateUserHandler))
mux.Handle("/users/{id}/delete", RequireMethod("DELETE", deleteUserHandler))

// CSRF himoyasi
sensitiveHandler := Chain(
    RequireMethod("POST"),	// Faqat POST ruxsat etilgan
    RequireHeader("X-CSRF-Token"),	// CSRF tokeni bo'lishi kerak
)(dangerousActionHandler)

// Framework siz metod routing
func Router() http.Handler {
    return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        switch r.URL.Path {
        case "/api/data":
            RequireMethod("GET", getData).ServeHTTP(w, r)
        case "/api/submit":
            RequireMethod("POST", submitData).ServeHTTP(w, r)
        default:
            http.NotFound(w, r)
        }
    })
}
\`\`\`

**Amaliy foydalari:**
- **Brauzer xavfsizligi:** Havolalar orqali tasodifiy DELETE ning oldini olish
- **CORS Pre-flight:** To'g'ri Allow headeri preflight so'rovlariga yordam beradi
- **API hujjatlari:** Metod cheklovlari o'z-o'zini hujjatlashtiruvchi shartnomalar

**HTTP standartlari:**
- **405 Method Not Allowed:** RFC 7231 tomonidan talab qilinadi
- **Allow Header:** Ruxsat etilgan metodlarni ko'rsatishi kerak
- **Keng tarqalgan metodlar:** GET, POST, PUT, DELETE, PATCH, OPTIONS, HEAD

Metod validatsiyasisiz API har qanday metodni qabul qiladi, bu noaniq xatolar va xavfsizlik muammolariga olib keladi.`,
			solutionCode: `package httpx

import (
	"net/http"
	"strings"
)

func RequireMethod(method string, next http.Handler) http.Handler {
	if next == nil {	// nil next handlerni ishlash
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {})
	}
	want := strings.ToUpper(strings.TrimSpace(method))	// Normalizatsiya: trim + katta harf
	if want == "" {	// Bo'sh metod tekshiruvi
		return next	// Metod ko'rsatilmagan bo'lsa middleware ni o'tkazish
	}
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if strings.ToUpper(r.Method) != want {	// Katta-kichik harflarsiz solishtirish
			w.Header().Set("Allow", want)	// RFC 7231: Allow headerini o'rnatish
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)	// 405 response
			return
		}
		next.ServeHTTP(w, r)	// Metod mos keladi, davom etish
	})
}`
		}
	}
};

export default task;
