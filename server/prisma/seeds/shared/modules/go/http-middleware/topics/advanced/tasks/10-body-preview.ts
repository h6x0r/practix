import { Task } from '../../../../types';

export const task: Task = {
	slug: 'go-http-body-preview',
	title: 'Body Preview Middleware',
	difficulty: 'medium',	tags: ['go', 'http', 'middleware', 'body-handling'],
	estimatedTime: '30m',	isPremium: false,
	youtubeUrl: '',
	description: `Implement **BodyPreview** middleware that reads first N bytes of request body and stores in context without consuming the body.

**Requirements:**
1. Create function \`BodyPreview(limit int, key ctxKey, next http.Handler) http.Handler\`
2. Skip middleware if limit <= 0 or key is empty
3. Read first \`limit\` bytes using buffered reader
4. Store preview bytes in context
5. Pass request with wrapped body to next handler
6. Handle nil next handler and read errors

**Example:**
\`\`\`go
const BodyKey ctxKey = "body_preview"

handler := BodyPreview(100, BodyKey, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
    preview := r.Context().Value(BodyKey).([]byte)
    fmt.Fprintf(w, "Preview: %s", preview)

	// Body can still be read fully
    body, _ := io.ReadAll(r.Body)
    fmt.Fprintf(w, "Full: %s", body)
}))

// Request with body "Hello World" → Preview: Hello World, Full: Hello World
\`\`\`

**Constraints:**
- Must use bufio.Reader with Peek() to preview without consuming
- Must create copy of peeked bytes for context storage
- Must wrap body to allow full reading later`,
	initialCode: `package httpx

import (
	"bufio"
	"context"
	"errors"
	"io"
	"net/http"
)

type ctxKey string

// TODO: Implement BodyPreview middleware
func BodyPreview(limit int, key ctxKey, next http.Handler) http.Handler {
	// TODO: Implement
}`,
	solutionCode: `package httpx

import (
	"bufio"
	"context"
	"errors"
	"io"
	"net/http"
)

type ctxKey string

func BodyPreview(limit int, key ctxKey, next http.Handler) http.Handler {
	if next == nil {	// Handle nil next handler
		return http.HandlerFunc(func(http.ResponseWriter, *http.Request) {})
	}
	if limit <= 0 || key == "" {	// Validate parameters
		return next	// Skip middleware if invalid params
	}
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		reader := bufio.NewReaderSize(r.Body, limit)	// Create buffered reader with limit size
		peek, err := reader.Peek(limit)	// Peek first limit bytes without consuming
		if err != nil && !errors.Is(err, io.EOF) && !errors.Is(err, bufio.ErrBufferFull) {
			http.Error(w, "failed to preview body", http.StatusBadRequest)	// Handle peek errors
			return
		}
		copyBuf := append([]byte(nil), peek...)	// Create copy of peeked bytes
		ctx := context.WithValue(r.Context(), key, copyBuf)	// Store copy in context
		clone := r.Clone(ctx)	// Clone request with new context
		clone.Body = &bodyReadCloser{Reader: reader, closer: r.Body}	// Wrap reader to restore full reading
		next.ServeHTTP(w, clone)	// Pass modified request
	})
}

type bodyReadCloser struct {	// Custom ReadCloser wrapper
	io.Reader	// Embedded Reader for reading
	closer io.Closer	// Original closer for cleanup
}

func (b *bodyReadCloser) Close() error {	// Implement Close method
	if b.closer == nil {
		return nil
	}
	return b.closer.Close()	// Close original body
}`,
			hint1: `Use bufio.NewReaderSize() with Peek() to read without consuming. Copy peeked bytes before storing in context.`,
			hint2: `Create custom type with io.Reader and original closer to restore full reading capability.`,
			testCode: `package httpx

import (
	"bytes"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func Test1(t *testing.T) {
	// Test middleware returns non-nil handler
	const key ctxKey = "preview"
	h := BodyPreview(100, key, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {}))
	if h == nil {
		t.Error("handler should not be nil")
	}
}

func Test2(t *testing.T) {
	// Test nil handler returns non-nil
	h := BodyPreview(100, "key", nil)
	if h == nil {
		t.Error("nil handler should return non-nil")
	}
}

func Test3(t *testing.T) {
	// Test limit <= 0 skips middleware
	called := false
	const key ctxKey = "preview"
	h := BodyPreview(0, key, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		called = true
	}))
	h.ServeHTTP(httptest.NewRecorder(), httptest.NewRequest("POST", "/", strings.NewReader("body")))
	if !called {
		t.Error("limit 0 should skip middleware")
	}
}

func Test4(t *testing.T) {
	// Test empty key skips middleware
	called := false
	h := BodyPreview(100, "", http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		called = true
	}))
	h.ServeHTTP(httptest.NewRecorder(), httptest.NewRequest("POST", "/", strings.NewReader("body")))
	if !called {
		t.Error("empty key should skip middleware")
	}
}

func Test5(t *testing.T) {
	// Test preview bytes are in context
	const key ctxKey = "preview"
	var preview []byte
	h := BodyPreview(100, key, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		preview = r.Context().Value(key).([]byte)
	}))
	h.ServeHTTP(httptest.NewRecorder(), httptest.NewRequest("POST", "/", strings.NewReader("Hello World")))
	if string(preview) != "Hello World" {
		t.Errorf("preview = %q, want 'Hello World'", string(preview))
	}
}

func Test6(t *testing.T) {
	// Test body can still be read after preview
	const key ctxKey = "preview"
	var body []byte
	h := BodyPreview(5, key, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ = io.ReadAll(r.Body)
	}))
	h.ServeHTTP(httptest.NewRecorder(), httptest.NewRequest("POST", "/", strings.NewReader("Hello World")))
	if string(body) != "Hello World" {
		t.Errorf("body = %q, want 'Hello World'", string(body))
	}
}

func Test7(t *testing.T) {
	// Test preview limited to specified bytes
	const key ctxKey = "preview"
	var preview []byte
	h := BodyPreview(5, key, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		preview = r.Context().Value(key).([]byte)
	}))
	h.ServeHTTP(httptest.NewRecorder(), httptest.NewRequest("POST", "/", strings.NewReader("Hello World")))
	if string(preview) != "Hello" {
		t.Errorf("preview = %q, want 'Hello' (5 bytes)", string(preview))
	}
}

func Test8(t *testing.T) {
	// Test empty body
	const key ctxKey = "preview"
	var preview []byte
	h := BodyPreview(100, key, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		preview = r.Context().Value(key).([]byte)
	}))
	h.ServeHTTP(httptest.NewRecorder(), httptest.NewRequest("POST", "/", bytes.NewReader(nil)))
	if len(preview) != 0 {
		t.Errorf("preview len = %d, want 0 for empty body", len(preview))
	}
}

func Test9(t *testing.T) {
	// Test method preserved
	const key ctxKey = "preview"
	var method string
	h := BodyPreview(100, key, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		method = r.Method
	}))
	h.ServeHTTP(httptest.NewRecorder(), httptest.NewRequest("PUT", "/", strings.NewReader("data")))
	if method != "PUT" {
		t.Errorf("method = %q, want PUT", method)
	}
}

func Test10(t *testing.T) {
	// Test response writing works
	const key ctxKey = "preview"
	h := BodyPreview(100, key, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusCreated)
		w.Write([]byte("created"))
	}))
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest("POST", "/", strings.NewReader("data")))
	if rec.Code != http.StatusCreated || rec.Body.String() != "created" {
		t.Errorf("got %d %q, want 201 'created'", rec.Code, rec.Body.String())
	}
}`,
			whyItMatters: `BodyPreview enables logging, validation, and introspection of request bodies without preventing handlers from reading the full body.

**Why Body Preview:**
- **Request Logging:** Log request payloads for debugging without consuming body
- **Validation:** Preview body format (JSON/XML) before parsing
- **Routing:** Route based on body content without full deserialization
- **Security:** Inspect payloads for suspicious patterns

**Production Pattern:**
\`\`\`go
const BodyPreviewKey ctxKey = "body_preview"

// Logging middleware with body preview
func RequestLogger(next http.Handler) http.Handler {
    return BodyPreview(1024, BodyPreviewKey, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        preview := r.Context().Value(BodyPreviewKey).([]byte)

	// Log request with body preview
        log.Printf(
            "method=%s path=%s content_type=%s preview=%s",
            r.Method,
            r.URL.Path,
            r.Header.Get("Content-Type"),
            string(preview),
        )

        next.ServeHTTP(w, r)
    }))
}

// Content-based routing
func ContentTypeRouter(next http.Handler) http.Handler {
    return BodyPreview(10, BodyPreviewKey, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        preview := r.Context().Value(BodyPreviewKey).([]byte)

	// Route based on body content
        if bytes.HasPrefix(preview, []byte("{")) {
            jsonHandler.ServeHTTP(w, r)
            return
        }
        if bytes.HasPrefix(preview, []byte("<")) {
            xmlHandler.ServeHTTP(w, r)
            return
        }

        next.ServeHTTP(w, r)
    }))
}

// Security inspection
func SecurityScanner(next http.Handler) http.Handler {
    return BodyPreview(4096, BodyPreviewKey, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        preview := r.Context().Value(BodyPreviewKey).([]byte)

	// Check for suspicious patterns
        suspicious := []string{"<script>", "DROP TABLE", "'; --"}
        previewStr := string(preview)
        for _, pattern := range suspicious {
            if strings.Contains(previewStr, pattern) {
                log.Printf("SECURITY: suspicious pattern detected: %s", pattern)
                http.Error(w, "forbidden", http.StatusForbidden)
                return
            }
        }

        next.ServeHTTP(w, r)
    }))
}

// Payload size estimation
func PayloadInspector(next http.Handler) http.Handler {
    return BodyPreview(512, BodyPreviewKey, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        preview := r.Context().Value(BodyPreviewKey).([]byte)

	// Estimate JSON object size
        if bytes.HasPrefix(preview, []byte("{")) {
            depth := 0
            for _, ch := range preview {
                if ch == '{' {
                    depth++
                }
            }

            if depth > 10 {
                log.Printf("WARNING: deeply nested JSON detected (depth > 10)")
            }
        }

        next.ServeHTTP(w, r)
    }))
}
\`\`\`

**Real-World Benefits:**
- **Debugging:** See request payloads in logs without complex instrumentation
- **Monitoring:** Track payload patterns and sizes
- **Compliance:** Log sensitive operations with request data
- **Performance:** Route efficiently based on content type

**Body Handling Best Practices:**
- **Copy Peek Results:** Peeked bytes are only valid until next read
- **Preserve Full Reading:** Wrap body to allow handlers to read completely
- **Error Handling:** Handle EOF and buffer full errors gracefully
- **Memory Safety:** Limit preview size to prevent excessive memory usage

**Common Pitfalls:**
- **Not Copying Peek:** Peeked slice is reused by bufio.Reader
- **Breaking Body Reading:** Must preserve ability to read full body
- **Memory Leaks:** Store unlimited preview data in long-lived contexts
- **Race Conditions:** Access preview from multiple goroutines

Without BodyPreview, logging request bodies requires either reading the entire body upfront (memory intensive) or complex custom middleware that breaks body reading.`,	order: 9,
	translations: {
		ru: {
			title: 'Чтение начала тела запроса без потребления',
			description: `Реализуйте middleware **BodyPreview**, который читает первые N байт тела запроса и сохраняет в контексте без потребления тела.

**Требования:**
1. Создайте функцию \`BodyPreview(limit int, key ctxKey, next http.Handler) http.Handler\`
2. Пропустите middleware если limit <= 0 или key пустой
3. Прочитайте первые \`limit\` байт используя буферизированный reader
4. Сохраните preview байты в контексте
5. Передайте запрос с обёрнутым body следующему handler
6. Обработайте nil handler и ошибки чтения

**Пример:**
\`\`\`go
const BodyKey ctxKey = "body_preview"

handler := BodyPreview(100, BodyKey, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
    preview := r.Context().Value(BodyKey).([]byte)
    fmt.Fprintf(w, "Preview: %s", preview)

	// Body всё ещё можно прочитать полностью
    body, _ := io.ReadAll(r.Body)
    fmt.Fprintf(w, "Full: %s", body)
}))

// Запрос с телом "Hello World" → Preview: Hello World, Full: Hello World
\`\`\`

**Ограничения:**
- Должен использовать bufio.Reader с Peek() для preview без потребления
- Должен создавать копию прочитанных байт для хранения в контексте
- Должен оборачивать body чтобы позволить полное чтение позже`,
			hint1: `Используйте bufio.NewReaderSize() с Peek() для чтения без потребления. Копируйте прочитанные байты перед сохранением в контексте.`,
			hint2: `Создайте кастомный тип с io.Reader и оригинальным closer для восстановления полной возможности чтения.`,
			whyItMatters: `BodyPreview позволяет логирование, валидацию и инспекцию тел запросов без препятствования handlers читать полное тело.

**Почему Body Preview:**
- **Логирование запросов:** Логирование request payloads для отладки без потребления body
- **Валидация:** Preview формата body (JSON/XML) перед парсингом
- **Роутинг:** Роутинг на основе содержимого body без полной десериализации
- **Безопасность:** Инспекция payloads на подозрительные паттерны

**Продакшен паттерн:**
\`\`\`go
const BodyPreviewKey ctxKey = "body_preview"

// Middleware логирования с body preview
func RequestLogger(next http.Handler) http.Handler {
    return BodyPreview(1024, BodyPreviewKey, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        preview := r.Context().Value(BodyPreviewKey).([]byte)

        log.Printf(
            "method=%s path=%s content_type=%s preview=%s",
            r.Method,
            r.URL.Path,
            r.Header.Get("Content-Type"),
            string(preview),
        )

        next.ServeHTTP(w, r)
    }))
}

// Роутинг на основе содержимого
func ContentTypeRouter(next http.Handler) http.Handler {
    return BodyPreview(10, BodyPreviewKey, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        preview := r.Context().Value(BodyPreviewKey).([]byte)

        if bytes.HasPrefix(preview, []byte("{")) {
            jsonHandler.ServeHTTP(w, r)
            return
        }
        if bytes.HasPrefix(preview, []byte("<")) {
            xmlHandler.ServeHTTP(w, r)
            return
        }

        next.ServeHTTP(w, r)
    }))
}

// Сканер безопасности
func SecurityScanner(next http.Handler) http.Handler {
    return BodyPreview(4096, BodyPreviewKey, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        preview := r.Context().Value(BodyPreviewKey).([]byte)

        suspicious := []string{"<script>", "DROP TABLE", "'; --"}
        previewStr := string(preview)
        for _, pattern := range suspicious {
            if strings.Contains(previewStr, pattern) {
                log.Printf("SECURITY: suspicious pattern: %s", pattern)
                http.Error(w, "forbidden", http.StatusForbidden)
                return
            }
        }

        next.ServeHTTP(w, r)
    }))
}
\`\`\`

**Практические преимущества:**
- **Отладка:** Видеть request payloads в логах без сложной инструментации
- **Мониторинг:** Отслеживание паттернов и размеров payloads
- **Compliance:** Логирование sensitive операций с данными запроса
- **Производительность:** Эффективный роутинг на основе типа контента

**Best practices работы с Body:**
- **Копирование Peek результатов:** Прочитанные байты валидны только до следующего чтения
- **Сохранение полного чтения:** Оберните body чтобы handlers могли читать полностью
- **Обработка ошибок:** Graceful handling EOF и buffer full ошибок
- **Memory Safety:** Ограничьте размер preview для предотвращения excessive memory usage

Без BodyPreview логирование тел запросов требует либо чтения всего тела заранее (memory intensive), либо сложного custom middleware.`,
			solutionCode: `package httpx

import (
	"bufio"
	"context"
	"errors"
	"io"
	"net/http"
)

type ctxKey string

func BodyPreview(limit int, key ctxKey, next http.Handler) http.Handler {
	if next == nil {	// Обработка nil next handler
		return http.HandlerFunc(func(http.ResponseWriter, *http.Request) {})
	}
	if limit <= 0 || key == "" {	// Валидация параметров
		return next	// Пропуск middleware если невалидные параметры
	}
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		reader := bufio.NewReaderSize(r.Body, limit)	// Создание буферизированного reader с размером limit
		peek, err := reader.Peek(limit)	// Предпросмотр первых limit байт без потребления
		if err != nil && !errors.Is(err, io.EOF) && !errors.Is(err, bufio.ErrBufferFull) {
			http.Error(w, "failed to preview body", http.StatusBadRequest)	// Обработка ошибок предпросмотра
			return
		}
		copyBuf := append([]byte(nil), peek...)	// Создание копии предпросмотренных байт
		ctx := context.WithValue(r.Context(), key, copyBuf)	// Сохранение копии в контексте
		clone := r.Clone(ctx)	// Клонирование запроса с новым контекстом
		clone.Body = &bodyReadCloser{Reader: reader, closer: r.Body}	// Обёртывание reader для восстановления полного чтения
		next.ServeHTTP(w, clone)	// Передача модифицированного запроса
	})
}

type bodyReadCloser struct {	// Кастомная обёртка ReadCloser
	io.Reader	// Встроенный Reader для чтения
	closer io.Closer	// Оригинальный closer для очистки
}

func (b *bodyReadCloser) Close() error {	// Реализация метода Close
	if b.closer == nil {
		return nil
	}
	return b.closer.Close()	// Закрытие оригинального body
}`
		},
		uz: {
			title: 'Request body boshini iste\'mol qilmasdan o\'qish',
			description: `Request body ning birinchi N baytini o'qiydigan va kontekstda bodyni iste'mol qilmasdan saqlaydigan **BodyPreview** middleware ni amalga oshiring.

**Talablar:**
1. \`BodyPreview(limit int, key ctxKey, next http.Handler) http.Handler\` funksiyasini yarating
2. Agar limit <= 0 yoki key bo'sh bo'lsa middleware ni o'tkazing
3. Buferlangan readerni ishlatib birinchi 'limit' baytni o'qing
4. Preview baytlarni kontekstda saqlang
5. O'ralgan body bilan requestni keyingi handlerga o'tkazing
6. nil handler va o'qish xatolarini ishlang

**Misol:**
\`\`\`go
const BodyKey ctxKey = "body_preview"

handler := BodyPreview(100, BodyKey, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
    preview := r.Context().Value(BodyKey).([]byte)
    fmt.Fprintf(w, "Preview: %s", preview)

	// Body hali ham to'liq o'qilishi mumkin
    body, _ := io.ReadAll(r.Body)
    fmt.Fprintf(w, "Full: %s", body)
}))

// "Hello World" bodysi bilan request → Preview: Hello World, Full: Hello World
\`\`\`

**Cheklovlar:**
- Iste'mol qilmasdan preview uchun bufio.Reader bilan Peek() dan foydalanishi kerak
- Kontekstda saqlash uchun o'qilgan baytlarning nusxasini yaratishi kerak
- Keyinroq to'liq o'qish imkonini berish uchun bodyni o'rashi kerak`,
			hint1: `Iste'mol qilmasdan o'qish uchun bufio.NewReaderSize() bilan Peek() dan foydalaning. Kontekstda saqlashdan oldin o'qilgan baytlarni nusxalang.`,
			hint2: `To'liq o'qish imkoniyatini tiklash uchun io.Reader va asl closer bilan custom tur yarating.`,
			whyItMatters: `BodyPreview handlerlarning to'liq bodyni o'qishiga to'sqinlik qilmasdan request bodylarini loglash, validatsiya va tekshirish imkonini beradi.

**Nima uchun Body Preview:**
- **Request loglash:** Bodyni iste'mol qilmasdan debug uchun request payloadlarini loglash
- **Validatsiya:** Parsing qilishdan oldin body formatini (JSON/XML) preview qilish
- **Routing:** To'liq deserializatsiya qilmasdan body tarkibiga asoslangan routing
- **Xavfsizlik:** Payloadlarni shubhali patternlar uchun tekshirish

**Ishlab chiqarish patterni:**
\`\`\`go
const BodyPreviewKey ctxKey = "body_preview"

// Body preview bilan loglash middlewaresi
func RequestLogger(next http.Handler) http.Handler {
    return BodyPreview(1024, BodyPreviewKey, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        preview := r.Context().Value(BodyPreviewKey).([]byte)

        log.Printf(
            "method=%s path=%s content_type=%s preview=%s",
            r.Method,
            r.URL.Path,
            r.Header.Get("Content-Type"),
            string(preview),
        )

        next.ServeHTTP(w, r)
    }))
}

// Kontent asosida routing
func ContentTypeRouter(next http.Handler) http.Handler {
    return BodyPreview(10, BodyPreviewKey, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        preview := r.Context().Value(BodyPreviewKey).([]byte)

        if bytes.HasPrefix(preview, []byte("{")) {
            jsonHandler.ServeHTTP(w, r)
            return
        }
        if bytes.HasPrefix(preview, []byte("<")) {
            xmlHandler.ServeHTTP(w, r)
            return
        }

        next.ServeHTTP(w, r)
    }))
}

// Xavfsizlik skaneri
func SecurityScanner(next http.Handler) http.Handler {
    return BodyPreview(4096, BodyPreviewKey, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        preview := r.Context().Value(BodyPreviewKey).([]byte)

        suspicious := []string{"<script>", "DROP TABLE", "'; --"}
        previewStr := string(preview)
        for _, pattern := range suspicious {
            if strings.Contains(previewStr, pattern) {
                log.Printf("SECURITY: shubhali pattern: %s", pattern)
                http.Error(w, "forbidden", http.StatusForbidden)
                return
            }
        }

        next.ServeHTTP(w, r)
    }))
}
\`\`\`

**Amaliy foydalari:**
- **Debugging:** Murakkab instrumentatsiyasiz loglarda request payloadlarini ko'rish
- **Monitoring:** Payload patternlari va o'lchamlarini kuzatish
- **Compliance:** Request ma'lumotlari bilan sensitive operatsiyalarni loglash
- **Performance:** Kontent turiga asoslangan samarali routing

**Body ishlash uchun best practices:**
- **Peek natijalarini nusxalash:** O'qilgan baytlar faqat keyingi o'qishgacha yaroqli
- **To'liq o'qishni saqlash:** Handlerlar to'liq o'qishi uchun bodyni o'rang
- **Xato ishlash:** EOF va buffer full xatolarini graceful ishlash
- **Memory Safety:** Ortiqcha xotira ishlatishning oldini olish uchun preview o'lchamini cheklang

BodyPreview siz request bodylarini loglash butun bodyni oldindan o'qishni (memory intensive) yoki murakkab custom middleware ni talab qiladi.`,
			solutionCode: `package httpx

import (
	"bufio"
	"context"
	"errors"
	"io"
	"net/http"
)

type ctxKey string

func BodyPreview(limit int, key ctxKey, next http.Handler) http.Handler {
	if next == nil {	// nil next handlerni ishlash
		return http.HandlerFunc(func(http.ResponseWriter, *http.Request) {})
	}
	if limit <= 0 || key == "" {	// Parametrlarni validatsiya qilish
		return next	// Agar parametrlar noto'g'ri bo'lsa middleware ni o'tkazish
	}
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		reader := bufio.NewReaderSize(r.Body, limit)	// limit o'lchamli buferlangan reader yaratish
		peek, err := reader.Peek(limit)	// Iste'mol qilmasdan birinchi limit baytni ko'rish
		if err != nil && !errors.Is(err, io.EOF) && !errors.Is(err, bufio.ErrBufferFull) {
			http.Error(w, "failed to preview body", http.StatusBadRequest)	// Ko'rish xatolarini ishlash
			return
		}
		copyBuf := append([]byte(nil), peek...)	// Ko'rilgan baytlarning nusxasini yaratish
		ctx := context.WithValue(r.Context(), key, copyBuf)	// Nusxani kontekstda saqlash
		clone := r.Clone(ctx)	// Yangi kontekst bilan requestni klonlash
		clone.Body = &bodyReadCloser{Reader: reader, closer: r.Body}	// To'liq o'qishni tiklash uchun readerni o'rash
		next.ServeHTTP(w, clone)	// O'zgartirilgan requestni o'tkazish
	})
}

type bodyReadCloser struct {	// Custom ReadCloser wrapper
	io.Reader	// O'qish uchun ichki Reader
	closer io.Closer	// Tozalash uchun asl closer
}

func (b *bodyReadCloser) Close() error {	// Close metodini amalga oshirish
	if b.closer == nil {
		return nil
	}
	return b.closer.Close()	// Asl bodyni yopish
}`
		}
	}
};

export default task;
