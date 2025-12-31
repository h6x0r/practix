import { Task } from '../../../../types';

export const task: Task = {
	slug: 'go-http-max-bytes',
	title: 'Max Bytes Middleware',
	difficulty: 'medium',	tags: ['go', 'http', 'middleware', 'validation'],
	estimatedTime: '30m',	isPremium: false,
	youtubeUrl: '',
	description: `Implement **MaxBytes** middleware that limits request body size and returns 413 Payload Too Large if exceeded.

**Requirements:**
1. Create function \`MaxBytes(limit int64, next http.Handler) http.Handler\`
2. Skip middleware if limit <= 0
3. Read body with size limit using io.LimitReader
4. Return 413 if body exceeds limit
5. Return 400 on read errors
6. Replace body with read data for next handler
7. Handle nil next handler

**Example:**
\`\`\`go
handler := MaxBytes(100, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
    body, _ := io.ReadAll(r.Body)
    fmt.Fprintf(w, "Body: %s", body)
}))

// Request with 50 bytes → OK
// Request with 150 bytes → 413 Payload Too Large
\`\`\`

**Constraints:**
- Must use io.LimitReader to read with size limit
- Must return 413 status for oversized payloads
- Must close original body and replace with new one`,
	initialCode: `package httpx

import (
	"bytes"
	"io"
	"net/http"
)

// TODO: Implement MaxBytes middleware
func MaxBytes(limit int64, next http.Handler) http.Handler {
	// TODO: Implement
}`,
	solutionCode: `package httpx

import (
	"bytes"
	"io"
	"net/http"
)

func MaxBytes(limit int64, next http.Handler) http.Handler {
	if next == nil {	// Handle nil next handler
		return http.HandlerFunc(func(http.ResponseWriter, *http.Request) {})
	}

	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if limit <= 0 {	// Check if limit is valid
			next.ServeHTTP(w, r)	// Skip middleware if no limit
			return
		}
		payload, err := io.ReadAll(io.LimitReader(r.Body, limit+1))	// Read up to limit+1 bytes
		if err != nil {
			http.Error(w, "failed to read body", http.StatusBadRequest)	// Handle read errors
			return
		}
		if int64(len(payload)) > limit {	// Check if size exceeded
			http.Error(w, "request entity too large", http.StatusRequestEntityTooLarge)	// Return 413
			return
		}
		if err := r.Body.Close(); err != nil {	// Close original body
			http.Error(w, "failed to close body", http.StatusBadRequest)
			return
		}
		r.Body = io.NopCloser(bytes.NewReader(payload))	// Replace body with read data
		next.ServeHTTP(w, r)	// Pass modified request
	})
}`,
			hint1: `Use io.LimitReader(r.Body, limit+1) to read up to limit+1 bytes. If you read more than limit, the body is too large.`,
			hint2: `After reading, close original body and create new one with io.NopCloser(bytes.NewReader(payload)).`,
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
	handler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		w.Write(body)
	})
	h := MaxBytes(100, handler)
	req := httptest.NewRequest("POST", "/", strings.NewReader("hello"))
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Errorf("expected 200, got %d", rec.Code)
	}
	if rec.Body.String() != "hello" {
		t.Errorf("expected body 'hello', got %q", rec.Body.String())
	}
}

func Test2(t *testing.T) {
	handler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
	})
	h := MaxBytes(10, handler)
	req := httptest.NewRequest("POST", "/", strings.NewReader("this is way too long"))
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	if rec.Code != http.StatusRequestEntityTooLarge {
		t.Errorf("expected 413, got %d", rec.Code)
	}
}

func Test3(t *testing.T) {
	handler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		w.Write(body)
	})
	h := MaxBytes(5, handler)
	req := httptest.NewRequest("POST", "/", strings.NewReader("12345"))
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Errorf("expected 200 for exact limit, got %d", rec.Code)
	}
	if rec.Body.String() != "12345" {
		t.Errorf("expected '12345', got %q", rec.Body.String())
	}
}

func Test4(t *testing.T) {
	handler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
	})
	h := MaxBytes(5, handler)
	req := httptest.NewRequest("POST", "/", strings.NewReader("123456"))
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	if rec.Code != http.StatusRequestEntityTooLarge {
		t.Errorf("expected 413 for limit+1, got %d", rec.Code)
	}
}

func Test5(t *testing.T) {
	handler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		w.Write(body)
	})
	h := MaxBytes(0, handler)
	req := httptest.NewRequest("POST", "/", strings.NewReader("any data"))
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Errorf("expected 200 when limit<=0, got %d", rec.Code)
	}
}

func Test6(t *testing.T) {
	handler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		w.Write(body)
	})
	h := MaxBytes(-1, handler)
	req := httptest.NewRequest("POST", "/", strings.NewReader("test"))
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Errorf("expected 200 when limit<0, got %d", rec.Code)
	}
}

func Test7(t *testing.T) {
	h := MaxBytes(100, nil)
	req := httptest.NewRequest("POST", "/", strings.NewReader("test"))
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Errorf("expected 200 for nil handler, got %d", rec.Code)
	}
}

func Test8(t *testing.T) {
	handler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		w.Write(body)
	})
	h := MaxBytes(100, handler)
	req := httptest.NewRequest("POST", "/", strings.NewReader(""))
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Errorf("expected 200 for empty body, got %d", rec.Code)
	}
	if rec.Body.String() != "" {
		t.Errorf("expected empty body, got %q", rec.Body.String())
	}
}

func Test9(t *testing.T) {
	largeData := bytes.Repeat([]byte("x"), 1000)
	handler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
	})
	h := MaxBytes(500, handler)
	req := httptest.NewRequest("POST", "/", bytes.NewReader(largeData))
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	if rec.Code != http.StatusRequestEntityTooLarge {
		t.Errorf("expected 413 for 1000 bytes with 500 limit, got %d", rec.Code)
	}
}

func Test10(t *testing.T) {
	var bodyRead string
	handler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		first, _ := io.ReadAll(r.Body)
		bodyRead = string(first)
		w.WriteHeader(http.StatusOK)
	})
	h := MaxBytes(100, handler)
	req := httptest.NewRequest("POST", "/", strings.NewReader("reread test"))
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	if bodyRead != "reread test" {
		t.Errorf("expected body to be readable by handler, got %q", bodyRead)
	}
}
`,
			whyItMatters: `MaxBytes prevents denial-of-service attacks and resource exhaustion by rejecting oversized payloads before processing.

**Why Limit Request Size:**
- **DoS Protection:** Prevent attackers from sending huge payloads
- **Memory Safety:** Avoid out-of-memory crashes
- **Fair Usage:** Prevent users from monopolizing resources
- **API Design:** Enforce payload size constraints

**Production Pattern:**
\`\`\`go
// Different limits for different endpoints
func Router() *http.ServeMux {
    mux := http.NewServeMux()

	// Small payloads: auth, health
    mux.Handle("/login", MaxBytes(1*KB, loginHandler))
    mux.Handle("/health", MaxBytes(512, healthHandler))

	// Medium payloads: JSON APIs
    mux.Handle("/api/users", MaxBytes(10*KB, usersHandler))
    mux.Handle("/api/orders", MaxBytes(50*KB, ordersHandler))

	// Large payloads: file uploads
    mux.Handle("/upload/avatar", MaxBytes(2*MB, avatarHandler))
    mux.Handle("/upload/document", MaxBytes(10*MB, documentHandler))

	// Very large: video uploads (use streaming, not MaxBytes)
    mux.Handle("/upload/video", streamingVideoHandler)

    return mux
}

// Content-type aware limits
func SmartMaxBytes(next http.Handler) http.Handler {
    return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        contentType := r.Header.Get("Content-Type")

        var limit int64
        switch {
        case strings.Contains(contentType, "application/json"):
            limit = 100 * KB // JSON APIs
        case strings.Contains(contentType, "multipart/form-data"):
            limit = 10 * MB	// File uploads
        case strings.Contains(contentType, "text/plain"):
            limit = 50 * KB	// Text data
        default:
            limit = 1 * MB	// Default limit
        }

        MaxBytes(limit, next).ServeHTTP(w, r)
    })
}

// Progressive limits with warnings
func ProgressiveMaxBytes(warningLimit, hardLimit int64) func(http.Handler) http.Handler {
    return func(next http.Handler) http.Handler {
        return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
            payload, err := io.ReadAll(io.LimitReader(r.Body, hardLimit+1))
            if err != nil {
                http.Error(w, "read failed", http.StatusBadRequest)
                return
            }

            size := int64(len(payload))

            if size > hardLimit {
                http.Error(w, "payload too large", http.StatusRequestEntityTooLarge)
                return
            }

            if size > warningLimit {
                log.Printf("WARNING: Large payload %d bytes (warning threshold: %d)", size, warningLimit)
                w.Header().Set("X-Payload-Warning", "approaching size limit")
            }

            r.Body.Close()
            r.Body = io.NopCloser(bytes.NewReader(payload))
            next.ServeHTTP(w, r)
        })
    }
}

// Usage: Warn at 8MB, reject at 10MB
handler := ProgressiveMaxBytes(8*MB, 10*MB)(uploadHandler)

// Per-user limits based on tier
func TieredMaxBytes(next http.Handler) http.Handler {
    return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        tier := r.Context().Value("user_tier").(string)

        var limit int64
        switch tier {
        case "premium":
            limit = 100 * MB
        case "standard":
            limit = 10 * MB
        default:
            limit = 1 * MB
        }

        MaxBytes(limit, next).ServeHTTP(w, r)
    })
}

// Metrics tracking
func MaxBytesWithMetrics(limit int64, next http.Handler) http.Handler {
    return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        payload, err := io.ReadAll(io.LimitReader(r.Body, limit+1))
        if err != nil {
            http.Error(w, "read failed", http.StatusBadRequest)
            return
        }

        size := int64(len(payload))

	// Record payload size distribution
        metrics.RecordHistogram("http_request_body_bytes", float64(size), map[string]string{
            "path": r.URL.Path,
        })

        if size > limit {
            metrics.IncrementCounter("http_payload_size_rejections", map[string]string{
                "path": r.URL.Path,
            })
            http.Error(w, "payload too large", http.StatusRequestEntityTooLarge)
            return
        }

        r.Body.Close()
        r.Body = io.NopCloser(bytes.NewReader(payload))
        next.ServeHTTP(w, r)
    })
}

// Efficient streaming for large files (alternative to MaxBytes)
func StreamingUpload(maxSize int64) http.Handler {
    return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
	// Use http.MaxBytesReader for streaming validation
        r.Body = http.MaxBytesReader(w, r.Body, maxSize)

	// Process multipart form
        err := r.ParseMultipartForm(maxSize)
        if err != nil {
            if err.Error() == "http: request body too large" {
                http.Error(w, "file too large", http.StatusRequestEntityTooLarge)
                return
            }
            http.Error(w, "failed to parse form", http.StatusBadRequest)
            return
        }

	// Stream file to storage
        file, _, err := r.FormFile("upload")
        if err != nil {
            http.Error(w, "no file", http.StatusBadRequest)
            return
        }
        defer file.Close()

	// Copy to destination (with size tracking)
        written, err := io.Copy(storage, file)
        if err != nil {
            http.Error(w, "upload failed", http.StatusInternalServerError)
            return
        }

        fmt.Fprintf(w, "Uploaded %d bytes", written)
    })
}
\`\`\`

**Real-World Benefits:**
- **Security:** Block DoS attacks via large payloads
- **Stability:** Prevent OOM crashes from huge uploads
- **Cost Control:** Limit storage/bandwidth usage
- **User Experience:** Fast rejection vs slow upload then error

**Size Limits by Use Case:**
- **Authentication:** 512 bytes - 1 KB
- **JSON APIs:** 10 KB - 100 KB
- **Form Data:** 50 KB - 500 KB
- **Image Uploads:** 1 MB - 10 MB
- **Document Uploads:** 10 MB - 50 MB
- **Video Uploads:** Use streaming, not MaxBytes

**Important Notes:**
- **Read limit+1:** Detect oversized payloads by reading one extra byte
- **Memory Usage:** MaxBytes buffers entire body in memory
- **Large Files:** Use http.MaxBytesReader for streaming validation
- **Error Messages:** Return 413 (not 400) for size violations

**Common Sizes:**
\`\`\`go
const (
    B  = 1
    KB = 1024 * B
    MB = 1024 * KB
    GB = 1024 * MB
)
\`\`\`

Without MaxBytes, attackers can upload gigabyte payloads, exhausting memory and crashing the server.`,	order: 16,
	translations: {
		ru: {
			title: 'Ограничение размера тела запроса',
			description: `Реализуйте middleware **MaxBytes**, который ограничивает размер тела запроса и возвращает 413 Payload Too Large если превышен.

**Требования:**
1. Создайте функцию \`MaxBytes(limit int64, next http.Handler) http.Handler\`
2. Пропустите middleware если limit <= 0
3. Прочитайте body с ограничением размера используя io.LimitReader
4. Верните 413 если body превышает лимит
5. Верните 400 при ошибках чтения
6. Замените body прочитанными данными для следующего handler
7. Обработайте nil handler

**Пример:**
\`\`\`go
handler := MaxBytes(100, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
    body, _ := io.ReadAll(r.Body)
    fmt.Fprintf(w, "Body: %s", body)
}))

// Запрос с 50 байтами → OK
// Запрос с 150 байтами → 413 Payload Too Large
\`\`\`

**Ограничения:**
- Должен использовать io.LimitReader для чтения с ограничением размера
- Должен возвращать статус 413 для слишком больших payloads
- Должен закрывать оригинальный body и заменять новым`,
			hint1: `Используйте io.LimitReader(r.Body, limit+1) для чтения до limit+1 байт. Если прочитали больше limit, body слишком большой.`,
			hint2: `После чтения, закройте оригинальный body и создайте новый с io.NopCloser(bytes.NewReader(payload)).`,
			whyItMatters: `MaxBytes предотвращает denial-of-service атаки и исчерпание ресурсов отклоняя слишком большие payloads до обработки.

**Почему ограничивать размер запроса:**
- **DoS защита:** Предотвращение отправки огромных payloads атакующими
- **Безопасность памяти:** Избежание out-of-memory крашей
- **Справедливое использование:** Предотвращение монополизации ресурсов
- **API дизайн:** Принудительные ограничения размера payload

**Продакшен паттерн:**
\`\`\`go
// Разные лимиты для разных endpoints
func Router() *http.ServeMux {
    mux := http.NewServeMux()

	// Маленькие payloads: auth, health
    mux.Handle("/login", MaxBytes(1*KB, loginHandler))
    mux.Handle("/health", MaxBytes(512, healthHandler))

	// Средние payloads: JSON APIs
    mux.Handle("/api/users", MaxBytes(10*KB, usersHandler))
    mux.Handle("/api/orders", MaxBytes(50*KB, ordersHandler))

	// Большие payloads: загрузка файлов
    mux.Handle("/upload/avatar", MaxBytes(2*MB, avatarHandler))
    mux.Handle("/upload/document", MaxBytes(10*MB, documentHandler))

	// Очень большие: видео (используйте streaming, не MaxBytes)
    mux.Handle("/upload/video", streamingVideoHandler)

    return mux
}

// Content-type aware лимиты
func SmartMaxBytes(next http.Handler) http.Handler {
    return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        contentType := r.Header.Get("Content-Type")

        var limit int64
        switch {
        case strings.Contains(contentType, "application/json"):
            limit = 100 * KB // JSON APIs
        case strings.Contains(contentType, "multipart/form-data"):
            limit = 10 * MB	// Загрузка файлов
        case strings.Contains(contentType, "text/plain"):
            limit = 50 * KB	// Текстовые данные
        default:
            limit = 1 * MB	// Лимит по умолчанию
        }

        MaxBytes(limit, next).ServeHTTP(w, r)
    })
}

// Прогрессивные лимиты с предупреждениями
func ProgressiveMaxBytes(warningLimit, hardLimit int64) func(http.Handler) http.Handler {
    return func(next http.Handler) http.Handler {
        return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
            payload, err := io.ReadAll(io.LimitReader(r.Body, hardLimit+1))
            if err != nil {
                http.Error(w, "read failed", http.StatusBadRequest)
                return
            }

            size := int64(len(payload))

            if size > hardLimit {
                http.Error(w, "payload too large", http.StatusRequestEntityTooLarge)
                return
            }

            if size > warningLimit {
                log.Printf("ПРЕДУПРЕЖДЕНИЕ: Большой payload %d байт (порог предупреждения: %d)", size, warningLimit)
                w.Header().Set("X-Payload-Warning", "approaching size limit")
            }

            r.Body.Close()
            r.Body = io.NopCloser(bytes.NewReader(payload))
            next.ServeHTTP(w, r)
        })
    }
}

// Использование: Предупреждение при 8MB, отказ при 10MB
handler := ProgressiveMaxBytes(8*MB, 10*MB)(uploadHandler)

// Лимиты по уровню пользователя
func TieredMaxBytes(next http.Handler) http.Handler {
    return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        tier := r.Context().Value("user_tier").(string)

        var limit int64
        switch tier {
        case "premium":
            limit = 100 * MB
        case "standard":
            limit = 10 * MB
        default:
            limit = 1 * MB
        }

        MaxBytes(limit, next).ServeHTTP(w, r)
    })
}

// Отслеживание метрик
func MaxBytesWithMetrics(limit int64, next http.Handler) http.Handler {
    return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        payload, err := io.ReadAll(io.LimitReader(r.Body, limit+1))
        if err != nil {
            http.Error(w, "read failed", http.StatusBadRequest)
            return
        }

        size := int64(len(payload))

	// Запись распределения размеров payload
        metrics.RecordHistogram("http_request_body_bytes", float64(size), map[string]string{
            "path": r.URL.Path,
        })

        if size > limit {
            metrics.IncrementCounter("http_payload_size_rejections", map[string]string{
                "path": r.URL.Path,
            })
            http.Error(w, "payload too large", http.StatusRequestEntityTooLarge)
            return
        }

        r.Body.Close()
        r.Body = io.NopCloser(bytes.NewReader(payload))
        next.ServeHTTP(w, r)
    })
}

// Эффективный streaming для больших файлов (альтернатива MaxBytes)
func StreamingUpload(maxSize int64) http.Handler {
    return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
	// Использование http.MaxBytesReader для streaming валидации
        r.Body = http.MaxBytesReader(w, r.Body, maxSize)

	// Обработка multipart form
        err := r.ParseMultipartForm(maxSize)
        if err != nil {
            if err.Error() == "http: request body too large" {
                http.Error(w, "file too large", http.StatusRequestEntityTooLarge)
                return
            }
            http.Error(w, "failed to parse form", http.StatusBadRequest)
            return
        }

	// Streaming файла в хранилище
        file, _, err := r.FormFile("upload")
        if err != nil {
            http.Error(w, "no file", http.StatusBadRequest)
            return
        }
        defer file.Close()

	// Копирование в назначение (с отслеживанием размера)
        written, err := io.Copy(storage, file)
        if err != nil {
            http.Error(w, "upload failed", http.StatusInternalServerError)
            return
        }

        fmt.Fprintf(w, "Uploaded %d bytes", written)
    })
}
\`\`\`

**Практические преимущества:**
- **Безопасность:** Блокировка DoS атак через большие payloads
- **Стабильность:** Предотвращение OOM крашей от огромных загрузок
- **Контроль расходов:** Ограничение использования хранилища/трафика
- **UX:** Быстрый отказ vs медленная загрузка и затем ошибка

**Рекомендуемые размеры по Use Case:**
- **Аутентификация:** 512 bytes - 1 KB
- **JSON APIs:** 10 KB - 100 KB
- **Form данные:** 50 KB - 500 KB
- **Загрузка изображений:** 1 MB - 10 MB
- **Загрузка документов:** 10 MB - 50 MB
- **Видео:** Использовать streaming, не MaxBytes

**Важные замечания:**
- **Чтение limit+1:** Обнаружение слишком больших payloads чтением одного дополнительного байта
- **Использование памяти:** MaxBytes буферизует весь body в памяти
- **Большие файлы:** Используйте http.MaxBytesReader для streaming валидации
- **Сообщения об ошибках:** Возвращайте 413 (не 400) для нарушений размера

**Общие размеры:**
\`\`\`go
const (
    B  = 1
    KB = 1024 * B
    MB = 1024 * KB
    GB = 1024 * MB
)
\`\`\`

Без MaxBytes атакующие могут загружать гигабайтные payloads, исчерпывая память и вызывая краш сервера.`,
			solutionCode: `package httpx

import (
	"bytes"
	"io"
	"net/http"
)

func MaxBytes(limit int64, next http.Handler) http.Handler {
	if next == nil {	// Обработка nil next handler
		return http.HandlerFunc(func(http.ResponseWriter, *http.Request) {})
	}

	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if limit <= 0 {	// Проверка валидности лимита
			next.ServeHTTP(w, r)	// Пропуск middleware если нет лимита
			return
		}
		payload, err := io.ReadAll(io.LimitReader(r.Body, limit+1))	// Чтение до limit+1 байт
		if err != nil {
			http.Error(w, "failed to read body", http.StatusBadRequest)	// Обработка ошибок чтения
			return
		}
		if int64(len(payload)) > limit {	// Проверка превышен ли размер
			http.Error(w, "request entity too large", http.StatusRequestEntityTooLarge)	// Возврат 413
			return
		}
		if err := r.Body.Close(); err != nil {	// Закрытие оригинального body
			http.Error(w, "failed to close body", http.StatusBadRequest)
			return
		}
		r.Body = io.NopCloser(bytes.NewReader(payload))	// Замена body прочитанными данными
		next.ServeHTTP(w, r)	// Передача модифицированного запроса
	})
}`
		},
		uz: {
			title: 'Request body hajmini cheklash',
			description: `Request body hajmini cheklovchi va agar oshib ketsa 413 Payload Too Large qaytaruvchi **MaxBytes** middleware ni amalga oshiring.

**Talablar:**
1. \`MaxBytes(limit int64, next http.Handler) http.Handler\` funksiyasini yarating
2. Agar limit <= 0 bo'lsa middleware ni o'tkazing
3. io.LimitReader dan foydalanib hajm chegarasi bilan bodyni o'qing
4. Agar body limitdan oshsa 413 qaytaring
5. O'qish xatolarida 400 qaytaring
6. Keyingi handler uchun bodyni o'qilgan ma'lumotlar bilan almashtiring
7. nil handlerni ishlang

**Misol:**
\`\`\`go
handler := MaxBytes(100, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
    body, _ := io.ReadAll(r.Body)
    fmt.Fprintf(w, "Body: %s", body)
}))

// 50 bayt bilan request → OK
// 150 bayt bilan request → 413 Payload Too Large
\`\`\`

**Cheklovlar:**
- Hajm chegarasi bilan o'qish uchun io.LimitReader dan foydalanishi kerak
- Juda katta payloadlar uchun 413 statusni qaytarishi kerak
- Asl bodyni yopib, yangi bilan almashtirishi kerak`,
			hint1: `limit+1 baytgacha o'qish uchun io.LimitReader(r.Body, limit+1) dan foydalaning. Agar limitdan ko'p o'qisa, body juda katta.`,
			hint2: `O'qishdan keyin, asl bodyni yoping va io.NopCloser(bytes.NewReader(payload)) bilan yangisini yarating.`,
			whyItMatters: `MaxBytes qayta ishlashdan oldin juda katta payloadlarni rad etish orqali denial-of-service hujumlar va resurslar tugashining oldini oladi.

**Nima uchun request hajmini cheklash:**
- **DoS himoyasi:** Tajovuzkorlar tomonidan ulkan payloadlar yuborishning oldini olish
- **Xotira xavfsizligi:** Out-of-memory crashlardan qochish
- **Adolatli foydalanish:** Resurslarni monopolizatsiya qilishning oldini olish
- **API dizayni:** Payload hajmi cheklovlarini majburlash

**Ishlab chiqarish patterni:**
\`\`\`go
// Turli endpointlar uchun turli limitlar
func Router() *http.ServeMux {
    mux := http.NewServeMux()

	// Kichik payloadlar: auth, health
    mux.Handle("/login", MaxBytes(1*KB, loginHandler))
    mux.Handle("/health", MaxBytes(512, healthHandler))

	// O'rtacha payloadlar: JSON APIlar
    mux.Handle("/api/users", MaxBytes(10*KB, usersHandler))
    mux.Handle("/api/orders", MaxBytes(50*KB, ordersHandler))

	// Katta payloadlar: fayl yuklash
    mux.Handle("/upload/avatar", MaxBytes(2*MB, avatarHandler))
    mux.Handle("/upload/document", MaxBytes(10*MB, documentHandler))

	// Juda katta: video (streaming ishlatish, MaxBytes emas)
    mux.Handle("/upload/video", streamingVideoHandler)

    return mux
}

// Content-type aware limitlar
func SmartMaxBytes(next http.Handler) http.Handler {
    return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        contentType := r.Header.Get("Content-Type")

        var limit int64
        switch {
        case strings.Contains(contentType, "application/json"):
            limit = 100 * KB // JSON APIlar
        case strings.Contains(contentType, "multipart/form-data"):
            limit = 10 * MB	// Fayl yuklash
        case strings.Contains(contentType, "text/plain"):
            limit = 50 * KB	// Matnli ma'lumotlar
        default:
            limit = 1 * MB	// Default limit
        }

        MaxBytes(limit, next).ServeHTTP(w, r)
    })
}

// Ogohlantirish bilan progressiv limitlar
func ProgressiveMaxBytes(warningLimit, hardLimit int64) func(http.Handler) http.Handler {
    return func(next http.Handler) http.Handler {
        return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
            payload, err := io.ReadAll(io.LimitReader(r.Body, hardLimit+1))
            if err != nil {
                http.Error(w, "read failed", http.StatusBadRequest)
                return
            }

            size := int64(len(payload))

            if size > hardLimit {
                http.Error(w, "payload too large", http.StatusRequestEntityTooLarge)
                return
            }

            if size > warningLimit {
                log.Printf("OGOHLANTIRISH: Katta payload %d bayt (ogohlantirish chegarasi: %d)", size, warningLimit)
                w.Header().Set("X-Payload-Warning", "approaching size limit")
            }

            r.Body.Close()
            r.Body = io.NopCloser(bytes.NewReader(payload))
            next.ServeHTTP(w, r)
        })
    }
}

// Foydalanish: 8MB da ogohlantirish, 10MB da rad etish
handler := ProgressiveMaxBytes(8*MB, 10*MB)(uploadHandler)

// Foydalanuvchi darajasi bo'yicha limitlar
func TieredMaxBytes(next http.Handler) http.Handler {
    return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        tier := r.Context().Value("user_tier").(string)

        var limit int64
        switch tier {
        case "premium":
            limit = 100 * MB
        case "standard":
            limit = 10 * MB
        default:
            limit = 1 * MB
        }

        MaxBytes(limit, next).ServeHTTP(w, r)
    })
}

// Metrikalarni kuzatish
func MaxBytesWithMetrics(limit int64, next http.Handler) http.Handler {
    return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        payload, err := io.ReadAll(io.LimitReader(r.Body, limit+1))
        if err != nil {
            http.Error(w, "read failed", http.StatusBadRequest)
            return
        }

        size := int64(len(payload))

	// Payload hajm taqsimotini yozish
        metrics.RecordHistogram("http_request_body_bytes", float64(size), map[string]string{
            "path": r.URL.Path,
        })

        if size > limit {
            metrics.IncrementCounter("http_payload_size_rejections", map[string]string{
                "path": r.URL.Path,
            })
            http.Error(w, "payload too large", http.StatusRequestEntityTooLarge)
            return
        }

        r.Body.Close()
        r.Body = io.NopCloser(bytes.NewReader(payload))
        next.ServeHTTP(w, r)
    })
}

// Katta fayllar uchun samarali streaming (MaxBytes alternativasi)
func StreamingUpload(maxSize int64) http.Handler {
    return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
	// Streaming validatsiya uchun http.MaxBytesReader ishlatish
        r.Body = http.MaxBytesReader(w, r.Body, maxSize)

	// Multipart formni qayta ishlash
        err := r.ParseMultipartForm(maxSize)
        if err != nil {
            if err.Error() == "http: request body too large" {
                http.Error(w, "file too large", http.StatusRequestEntityTooLarge)
                return
            }
            http.Error(w, "failed to parse form", http.StatusBadRequest)
            return
        }

	// Faylni saqlashga streaming
        file, _, err := r.FormFile("upload")
        if err != nil {
            http.Error(w, "no file", http.StatusBadRequest)
            return
        }
        defer file.Close()

	// Manzilga nusxalash (hajmni kuzatish bilan)
        written, err := io.Copy(storage, file)
        if err != nil {
            http.Error(w, "upload failed", http.StatusInternalServerError)
            return
        }

        fmt.Fprintf(w, "Uploaded %d bytes", written)
    })
}
\`\`\`

**Amaliy foydalari:**
- **Xavfsizlik:** Katta payloadlar orqali DoS hujumlarini bloklash
- **Barqarorlik:** Ulkan yuklashlardan OOM crashlarning oldini olish
- **Xarajatlarni nazorat qilish:** Saqlash/trafikdan foydalanishni cheklash
- **UX:** Sekin yuklash keyin xato o'rniga tez rad etish

**Use Case bo'yicha tavsiya etilgan hajmlar:**
- **Autentifikatsiya:** 512 bayt - 1 KB
- **JSON APIlar:** 10 KB - 100 KB
- **Form ma'lumotlari:** 50 KB - 500 KB
- **Rasm yuklash:** 1 MB - 10 MB
- **Hujjat yuklash:** 10 MB - 50 MB
- **Video:** Streaming ishlatish, MaxBytes emas

**Muhim eslatmalar:**
- **limit+1 o'qish:** Bitta qo'shimcha bayt o'qish orqali juda katta payloadlarni aniqlash
- **Xotira foydalanishi:** MaxBytes butun bodyni xotirada buferlaydi
- **Katta fayllar:** Streaming validatsiya uchun http.MaxBytesReader ishlatish
- **Xato xabarlari:** Hajm buzilishlari uchun 413 (400 emas) qaytarish

**Umumiy hajmlar:**
\`\`\`go
const (
    B  = 1
    KB = 1024 * B
    MB = 1024 * KB
    GB = 1024 * MB
)
\`\`\`

MaxBytes bo'lmasa, tajovuzkorlar gigabaytlik payloadlar yuklashi mumkin, bu xotirani tugatib serverning crashiga olib keladi.`,
			solutionCode: `package httpx

import (
	"bytes"
	"io"
	"net/http"
)

func MaxBytes(limit int64, next http.Handler) http.Handler {
	if next == nil {	// nil next handlerni ishlash
		return http.HandlerFunc(func(http.ResponseWriter, *http.Request) {})
	}

	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if limit <= 0 {	// Limit haqiqiyligini tekshirish
			next.ServeHTTP(w, r)	// Agar limit bo'lmasa middleware ni o'tkazish
			return
		}
		payload, err := io.ReadAll(io.LimitReader(r.Body, limit+1))	// limit+1 baytgacha o'qish
		if err != nil {
			http.Error(w, "failed to read body", http.StatusBadRequest)	// O'qish xatolarini ishlash
			return
		}
		if int64(len(payload)) > limit {	// Hajm oshib ketganligini tekshirish
			http.Error(w, "request entity too large", http.StatusRequestEntityTooLarge)	// 413 qaytarish
			return
		}
		if err := r.Body.Close(); err != nil {	// Asl bodyni yopish
			http.Error(w, "failed to close body", http.StatusBadRequest)
			return
		}
		r.Body = io.NopCloser(bytes.NewReader(payload))	// Bodyni o'qilgan ma'lumotlar bilan almashtirish
		next.ServeHTTP(w, r)	// O'zgartirilgan requestni o'tkazish
	})
}`
		}
	}
};

export default task;
