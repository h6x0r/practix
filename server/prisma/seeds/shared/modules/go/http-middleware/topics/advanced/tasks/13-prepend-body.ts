import { Task } from '../../../../types';

export const task: Task = {
	slug: 'go-http-prepend-body',
	title: 'Prepend Body Middleware',
	difficulty: 'medium',	tags: ['go', 'http', 'middleware', 'body-handling'],
	estimatedTime: '25m',	isPremium: false,
	youtubeUrl: '',
	description: `Implement **PrependBody** middleware that adds a prefix to the beginning of request body.

**Requirements:**
1. Create function \`PrependBody(prefix []byte, next http.Handler) http.Handler\`
2. Skip middleware if prefix is empty
3. Use io.MultiReader to combine prefix and body
4. Wrap result as ReadCloser preserving original closer
5. Update Content-Length to include prefix length
6. Handle nil next handler

**Example:**
\`\`\`go
prefix := []byte("PREFIX:")

handler := PrependBody(prefix, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
    body, _ := io.ReadAll(r.Body)
    fmt.Fprintf(w, "Body: %s", body)
}))

// Request with body "Hello" → Handler reads: "PREFIX:Hello"
\`\`\`

**Constraints:**
- Must use io.MultiReader to combine streams
- Must update Content-Length header
- Must preserve original body closer`,
	initialCode: `package httpx

import (
	"bytes"
	"io"
	"net/http"
)

// TODO: Implement PrependBody middleware
func PrependBody(prefix []byte, next http.Handler) http.Handler {
	// TODO: Implement
}`,
	solutionCode: `package httpx

import (
	"bytes"
	"io"
	"net/http"
)

func PrependBody(prefix []byte, next http.Handler) http.Handler {
	if next == nil {	// Handle nil next handler
		return http.HandlerFunc(func(http.ResponseWriter, *http.Request) {})
	}
	if len(prefix) == 0 {	// Check if prefix is empty
		return next	// Skip middleware if no prefix
	}
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		reader := io.MultiReader(bytes.NewReader(prefix), r.Body)	// Combine prefix and body
		clone := r.Clone(r.Context())	// Clone request to avoid mutation
		clone.Body = &bodyReadCloser{Reader: reader, closer: r.Body}	// Wrap as ReadCloser
		if clone.ContentLength >= 0 {	// Check if Content-Length is known
			clone.ContentLength += int64(len(prefix))	// Add prefix length to Content-Length
		}
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
			hint1: `Use io.MultiReader(bytes.NewReader(prefix), r.Body) to create a reader that reads prefix first, then body.`,
			hint2: `Update r.ContentLength by adding len(prefix) to account for the prepended data.`,
			testCode: `package httpx

import (
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func Test1(t *testing.T) {
	// Test middleware returns non-nil handler
	h := PrependBody([]byte("prefix"), http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {}))
	if h == nil {
		t.Error("handler should not be nil")
	}
}

func Test2(t *testing.T) {
	// Test nil handler returns non-nil
	h := PrependBody([]byte("prefix"), nil)
	if h == nil {
		t.Error("nil handler should return non-nil")
	}
}

func Test3(t *testing.T) {
	// Test nil prefix skips middleware
	var body []byte
	h := PrependBody(nil, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ = io.ReadAll(r.Body)
	}))
	h.ServeHTTP(httptest.NewRecorder(), httptest.NewRequest("POST", "/", strings.NewReader("data")))
	if string(body) != "data" {
		t.Errorf("body = %q, want 'data' (unchanged)", string(body))
	}
}

func Test4(t *testing.T) {
	// Test empty prefix skips middleware
	var body []byte
	h := PrependBody([]byte{}, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ = io.ReadAll(r.Body)
	}))
	h.ServeHTTP(httptest.NewRecorder(), httptest.NewRequest("POST", "/", strings.NewReader("data")))
	if string(body) != "data" {
		t.Errorf("body = %q, want 'data' (unchanged)", string(body))
	}
}

func Test5(t *testing.T) {
	// Test prefix is prepended
	var body []byte
	h := PrependBody([]byte("PREFIX:"), http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ = io.ReadAll(r.Body)
	}))
	h.ServeHTTP(httptest.NewRecorder(), httptest.NewRequest("POST", "/", strings.NewReader("data")))
	if string(body) != "PREFIX:data" {
		t.Errorf("body = %q, want 'PREFIX:data'", string(body))
	}
}

func Test6(t *testing.T) {
	// Test empty body with prefix
	var body []byte
	h := PrependBody([]byte("PREFIX"), http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ = io.ReadAll(r.Body)
	}))
	h.ServeHTTP(httptest.NewRecorder(), httptest.NewRequest("POST", "/", strings.NewReader("")))
	if string(body) != "PREFIX" {
		t.Errorf("body = %q, want 'PREFIX'", string(body))
	}
}

func Test7(t *testing.T) {
	// Test method preserved
	var method string
	h := PrependBody([]byte("p"), http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		method = r.Method
	}))
	h.ServeHTTP(httptest.NewRecorder(), httptest.NewRequest("PUT", "/", strings.NewReader("data")))
	if method != "PUT" {
		t.Errorf("method = %q, want PUT", method)
	}
}

func Test8(t *testing.T) {
	// Test headers preserved
	var header string
	h := PrependBody([]byte("p"), http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		header = r.Header.Get("X-Custom")
	}))
	req := httptest.NewRequest("POST", "/", strings.NewReader("data"))
	req.Header.Set("X-Custom", "value")
	h.ServeHTTP(httptest.NewRecorder(), req)
	if header != "value" {
		t.Errorf("header = %q, want 'value'", header)
	}
}

func Test9(t *testing.T) {
	// Test response writing works
	h := PrependBody([]byte("p"), http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusCreated)
		w.Write([]byte("ok"))
	}))
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest("POST", "/", strings.NewReader("data")))
	if rec.Code != http.StatusCreated {
		t.Errorf("status = %d, want 201", rec.Code)
	}
}

func Test10(t *testing.T) {
	// Test content length updated
	var contentLen int64
	h := PrependBody([]byte("12345"), http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		contentLen = r.ContentLength
	}))
	req := httptest.NewRequest("POST", "/", strings.NewReader("data"))
	req.ContentLength = 4
	h.ServeHTTP(httptest.NewRecorder(), req)
	if contentLen != 9 { // 5 + 4
		t.Errorf("ContentLength = %d, want 9", contentLen)
	}
}`,
			whyItMatters: `PrependBody enables protocol wrapping, authentication token injection, and message framing without modifying application code.

**Why Prepend Body:**
- **Protocol Wrapping:** Add headers/metadata to payloads
- **Authentication:** Inject auth tokens into requests
- **Message Framing:** Add length prefixes or magic bytes
- **Compatibility:** Adapt payloads to legacy systems

**Production Pattern:**
\`\`\`go
// Add authentication token to all requests
func AuthTokenInjector(token string) func(http.Handler) http.Handler {
    prefix := []byte(fmt.Sprintf("Token: %s\\n", token))
    return func(next http.Handler) http.Handler {
        return PrependBody(prefix, next)
    }
}

// Add JSON array wrapper for batch processing
func JSONArrayWrapper(next http.Handler) http.Handler {
    return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
	// Prepend "["
        handler := PrependBody([]byte("["), http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
	// Append "]" (need custom append middleware)
	// Read body, append "]", then process
            body, _ := io.ReadAll(r.Body)
            fullJSON := append(body, ']')
            r.Body = io.NopCloser(bytes.NewReader(fullJSON))
            next.ServeHTTP(w, r)
        }))
        handler.ServeHTTP(w, r)
    })
}

// Protocol versioning (add version header)
func ProtocolVersioner(version string) func(http.Handler) http.Handler {
    prefix := []byte(fmt.Sprintf("v%s\\n", version))
    return func(next http.Handler) http.Handler {
        return PrependBody(prefix, next)
    }
}

// CSV header injection
func CSVHeaderInjector(headers []string) func(http.Handler) http.Handler {
    headerLine := []byte(strings.Join(headers, ",") + "\\n")
    return func(next http.Handler) http.Handler {
        return PrependBody(headerLine, next)
    }
}

// Usage: Add CSV headers to uploaded data
handler := CSVHeaderInjector([]string{"id", "name", "email"})(uploadHandler)

// Length-prefixed messages (binary protocol)
func LengthPrefixer(next http.Handler) http.Handler {
    return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
	// Read body to calculate length
        body, err := io.ReadAll(r.Body)
        if err != nil {
            http.Error(w, "failed to read body", http.StatusBadRequest)
            return
        }
        r.Body.Close()

	// Prepend 4-byte length prefix (big-endian)
        prefix := make([]byte, 4)
        binary.BigEndian.PutUint32(prefix, uint32(len(body)))

	// Create new body with length prefix
        fullBody := append(prefix, body...)
        clone := r.Clone(r.Context())
        clone.Body = io.NopCloser(bytes.NewReader(fullBody))
        clone.ContentLength = int64(len(fullBody))

        next.ServeHTTP(w, clone)
    })
}

// Magic byte injection (file format identification)
func MagicByteInjector(magic []byte) func(http.Handler) http.Handler {
    return func(next http.Handler) http.Handler {
        return PrependBody(magic, next)
    }
}

// PNG magic bytes
pngHandler := MagicByteInjector([]byte{0x89, 'P', 'N', 'G'})(imageHandler)

// Multipart boundary injection
func MultipartBoundaryWrapper(boundary string) func(http.Handler) http.Handler {
    prefix := []byte(fmt.Sprintf("---%s\\r\\n", boundary))
    return func(next http.Handler) http.Handler {
        return PrependBody(prefix, next)
    }
}
\`\`\`

**Real-World Benefits:**
- **Clean Code:** Modify payloads without handler awareness
- **Reusability:** Single middleware for all endpoints
- **Protocol Flexibility:** Adapt to different protocols
- **Testing:** Inject test data easily

**MultiReader Advantages:**
- **Memory Efficient:** No copying or buffering
- **Streaming:** Works with large bodies
- **Simple:** Single function call
- **Composable:** Chain multiple readers

**Common Use Cases:**
- **Legacy Integration:** Add required headers to modern payloads
- **Batch Processing:** Wrap single requests as arrays
- **Binary Protocols:** Add length prefixes, magic bytes
- **CSV Processing:** Inject headers for headerless CSV uploads

**Important Notes:**
- **Content-Length:** Must update to reflect new size
- **Binary Safety:** Works with both text and binary data
- **Close Original:** MultiReader doesn't own the closer
- **Streaming:** No buffering, efficient for large payloads

Without PrependBody, adding prefixes requires reading the entire body, concatenating, and creating a new body—inefficient and memory-intensive.`,	order: 12,
	translations: {
		ru: {
			title: 'Добавление префикса к телу ответа',
			description: `Реализуйте middleware **PrependBody**, который добавляет префикс в начало тела запроса.

**Требования:**
1. Создайте функцию \`PrependBody(prefix []byte, next http.Handler) http.Handler\`
2. Пропустите middleware если префикс пустой
3. Используйте io.MultiReader для объединения префикса и body
4. Оберните результат как ReadCloser сохраняя оригинальный closer
5. Обновите Content-Length включая длину префикса
6. Обработайте nil handler

**Пример:**
\`\`\`go
prefix := []byte("PREFIX:")

handler := PrependBody(prefix, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
    body, _ := io.ReadAll(r.Body)
    fmt.Fprintf(w, "Body: %s", body)
}))

// Запрос с телом "Hello" → Handler читает: "PREFIX:Hello"
\`\`\`

**Ограничения:**
- Должен использовать io.MultiReader для объединения потоков
- Должен обновлять заголовок Content-Length
- Должен сохранять оригинальный body closer`,
			hint1: `Используйте io.MultiReader(bytes.NewReader(prefix), r.Body) для создания reader который читает сначала префикс, затем body.`,
			hint2: `Обновите r.ContentLength добавив len(prefix) чтобы учесть добавленные данные.`,
			whyItMatters: `PrependBody позволяет обёртывание протоколов, инъекцию токенов аутентификации и framing сообщений без изменения кода приложения.

**Почему Prepend Body:**
- **Protocol Wrapping:** Добавление headers/metadata к payloads
- **Аутентификация:** Инъекция auth токенов в запросы
- **Message Framing:** Добавление префиксов длины или magic bytes
- **Совместимость:** Адаптация payloads к legacy системам

**Продакшен паттерн:**
\`\`\`go
// Добавление auth токена ко всем запросам
func AuthTokenInjector(token string) func(http.Handler) http.Handler {
    prefix := []byte(fmt.Sprintf("Token: %s\\n", token))
    return func(next http.Handler) http.Handler {
        return PrependBody(prefix, next)
    }
}

// Добавление JSON array wrapper для batch processing
func JSONArrayWrapper(next http.Handler) http.Handler {
    return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        handler := PrependBody([]byte("["), http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
            body, _ := io.ReadAll(r.Body)
            fullJSON := append(body, ']')
            r.Body = io.NopCloser(bytes.NewReader(fullJSON))
            next.ServeHTTP(w, r)
        }))
        handler.ServeHTTP(w, r)
    })
}

// Protocol versioning (добавление version header)
func ProtocolVersioner(version string) func(http.Handler) http.Handler {
    prefix := []byte(fmt.Sprintf("v%s\\n", version))
    return func(next http.Handler) http.Handler {
        return PrependBody(prefix, next)
    }
}

// CSV header injection
func CSVHeaderInjector(headers []string) func(http.Handler) http.Handler {
    headerLine := []byte(strings.Join(headers, ",") + "\\n")
    return func(next http.Handler) http.Handler {
        return PrependBody(headerLine, next)
    }
}

// Usage: Добавление CSV headers к загружаемым данным
handler := CSVHeaderInjector([]string{"id", "name", "email"})(uploadHandler)

// Length-prefixed messages (бинарный протокол)
func LengthPrefixer(next http.Handler) http.Handler {
    return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        body, err := io.ReadAll(r.Body)
        if err != nil {
            http.Error(w, "failed to read body", http.StatusBadRequest)
            return
        }
        r.Body.Close()

        // Добавляем 4-байтный length prefix (big-endian)
        prefix := make([]byte, 4)
        binary.BigEndian.PutUint32(prefix, uint32(len(body)))

        fullBody := append(prefix, body...)
        clone := r.Clone(r.Context())
        clone.Body = io.NopCloser(bytes.NewReader(fullBody))
        clone.ContentLength = int64(len(fullBody))

        next.ServeHTTP(w, clone)
    })
}

// Magic byte injection (идентификация формата файла)
func MagicByteInjector(magic []byte) func(http.Handler) http.Handler {
    return func(next http.Handler) http.Handler {
        return PrependBody(magic, next)
    }
}

// PNG magic bytes
pngHandler := MagicByteInjector([]byte{0x89, 'P', 'N', 'G'})(imageHandler)
\`\`\`

**Практические преимущества:**
- **Clean Code:** Модификация payloads без осведомленности handler
- **Reusability:** Один middleware для всех endpoints
- **Protocol Flexibility:** Адаптация к разным протоколам
- **Testing:** Легкая инъекция тестовых данных

**Преимущества MultiReader:**
- **Memory Efficient:** Нет копирования или буферизации
- **Streaming:** Работает с большими bodies
- **Simple:** Один вызов функции
- **Composable:** Chain нескольких readers

**Общие сценарии использования:**
- **Legacy Integration:** Добавление необходимых headers к современным payloads
- **Batch Processing:** Обёртывание одиночных запросов как массивы
- **Binary Protocols:** Добавление length prefixes, magic bytes
- **CSV Processing:** Инъекция headers для CSV загрузок без заголовков

**Важные заметки:**
- **Content-Length:** Должен обновляться для отражения нового размера
- **Binary Safety:** Работает с текстом и бинарными данными
- **Close Original:** MultiReader не владеет closer
- **Streaming:** Нет буферизации, эффективно для больших payloads

**Общие сценарии использования:**
- **Legacy Integration:** Добавление необходимых headers к современным payloads
- **Batch Processing:** Обёртывание одиночных запросов как массивы
- **Binary Protocols:** Добавление length prefixes, magic bytes
- **CSV Processing:** Инъекция headers для CSV загрузок без заголовков

Без PrependBody добавление префиксов требует чтения всего body, конкатенации и создания нового body — неэффективно и требует много памяти.`,
			solutionCode: `package httpx

import (
	"bytes"
	"io"
	"net/http"
)

func PrependBody(prefix []byte, next http.Handler) http.Handler {
	if next == nil {	// Обработка nil next handler
		return http.HandlerFunc(func(http.ResponseWriter, *http.Request) {})
	}
	if len(prefix) == 0 {	// Проверка пустой ли префикс
		return next	// Пропуск middleware если нет префикса
	}
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		reader := io.MultiReader(bytes.NewReader(prefix), r.Body)	// Объединение префикса и body
		clone := r.Clone(r.Context())	// Клонирование запроса во избежание мутации
		clone.Body = &bodyReadCloser{Reader: reader, closer: r.Body}	// Обёртывание как ReadCloser
		if clone.ContentLength >= 0 {	// Проверка известен ли Content-Length
			clone.ContentLength += int64(len(prefix))	// Добавление длины префикса к Content-Length
		}
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
			title: 'Response bodyga prefiks qo\'shish',
			description: `Request body boshiga prefiksni qo'shuvchi **PrependBody** middleware ni amalga oshiring.

**Talablar:**
1. \`PrependBody(prefix []byte, next http.Handler) http.Handler\` funksiyasini yarating
2. Agar prefiks bo'sh bo'lsa middleware ni o'tkazing
3. Prefiks va bodyni birlashtirish uchun io.MultiReader dan foydalaning
4. Asl closerni saqlab, natijani ReadCloser sifatida o'rang
5. Prefiks uzunligini o'z ichiga olgan holda Content-Length ni yangilang
6. nil handlerni ishlang

**Misol:**
\`\`\`go
prefix := []byte("PREFIX:")

handler := PrependBody(prefix, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
    body, _ := io.ReadAll(r.Body)
    fmt.Fprintf(w, "Body: %s", body)
}))

// "Hello" bodysi bilan request → Handler o'qiydi: "PREFIX:Hello"
\`\`\`

**Cheklovlar:**
- Oqimlarni birlashtirish uchun io.MultiReader dan foydalanishi kerak
- Content-Length headerini yangilashi kerak
- Asl body closerni saqlashi kerak`,
			hint1: `Avval prefiksni, keyin bodyni o'quvchi reader yaratish uchun io.MultiReader(bytes.NewReader(prefix), r.Body) dan foydalaning.`,
			hint2: `Qo'shilgan ma'lumotlarni hisobga olish uchun r.ContentLength ga len(prefix) ni qo'shib yangilang.`,
			whyItMatters: `PrependBody ilova kodini o'zgartirmasdan protokol wrapping, autentifikatsiya token in'ektsiyasi va xabar framingini ta'minlaydi.

**Nima uchun Prepend Body:**
- **Protocol Wrapping:** Payloadlarga headers/metadata qo'shish
- **Autentifikatsiya:** Requestlarga auth tokenlarni in'ektsiya qilish
- **Message Framing:** Uzunlik prefikslari yoki magic baytlarni qo'shish
- **Muvofiqlik:** Payloadlarni legacy tizimlarga moslash

**Ishlab chiqarish patterni:**
\`\`\`go
// Barcha requestlarga auth token qo'shish
func AuthTokenInjector(token string) func(http.Handler) http.Handler {
    prefix := []byte(fmt.Sprintf("Token: %s\\n", token))
    return func(next http.Handler) http.Handler {
        return PrependBody(prefix, next)
    }
}

// Batch processing uchun JSON array wrapper qo'shish
func JSONArrayWrapper(next http.Handler) http.Handler {
    return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        handler := PrependBody([]byte("["), http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
            body, _ := io.ReadAll(r.Body)
            fullJSON := append(body, ']')
            r.Body = io.NopCloser(bytes.NewReader(fullJSON))
            next.ServeHTTP(w, r)
        }))
        handler.ServeHTTP(w, r)
    })
}

// Protocol versiyalash (version header qo'shish)
func ProtocolVersioner(version string) func(http.Handler) http.Handler {
    prefix := []byte(fmt.Sprintf("v%s\\n", version))
    return func(next http.Handler) http.Handler {
        return PrependBody(prefix, next)
    }
}

// CSV header in'ektsiyasi
func CSVHeaderInjector(headers []string) func(http.Handler) http.Handler {
    headerLine := []byte(strings.Join(headers, ",") + "\\n")
    return func(next http.Handler) http.Handler {
        return PrependBody(headerLine, next)
    }
}

// Usage: Yuklangan ma'lumotlarga CSV headerlarini qo'shish
handler := CSVHeaderInjector([]string{"id", "name", "email"})(uploadHandler)

// Length-prefixed messages (binary protokol)
func LengthPrefixer(next http.Handler) http.Handler {
    return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        body, err := io.ReadAll(r.Body)
        if err != nil {
            http.Error(w, "failed to read body", http.StatusBadRequest)
            return
        }
        r.Body.Close()

        // 4-baytli length prefix qo'shamiz (big-endian)
        prefix := make([]byte, 4)
        binary.BigEndian.PutUint32(prefix, uint32(len(body)))

        fullBody := append(prefix, body...)
        clone := r.Clone(r.Context())
        clone.Body = io.NopCloser(bytes.NewReader(fullBody))
        clone.ContentLength = int64(len(fullBody))

        next.ServeHTTP(w, clone)
    })
}

// Magic byte in'ektsiyasi (fayl formatini identifikatsiya qilish)
func MagicByteInjector(magic []byte) func(http.Handler) http.Handler {
    return func(next http.Handler) http.Handler {
        return PrependBody(magic, next)
    }
}

// PNG magic bytes
pngHandler := MagicByteInjector([]byte{0x89, 'P', 'N', 'G'})(imageHandler)
\`\`\`

**Amaliy foydalari:**
- **Toza kod:** Handler xabardorsizidan payloadlarni o'zgartirish
- **Qayta foydalanish:** Barcha endpointlar uchun bitta middleware
- **Protokol moslashuvchanligi:** Turli protokollarga moslash
- **Testing:** Test ma'lumotlarini oson in'ektsiya qilish

**MultiReader afzalliklari:**
- **Xotira samaradorligi:** Nusxalash yoki buferlash yo'q
- **Streaming:** Katta bodylar bilan ishlaydi
- **Soddalik:** Bitta funksiya chaqiruvi
- **Composable:** Bir nechta readerlarni chain qilish

**Umumiy foydalanish stsenariylari:**
- **Legacy Integration:** Zamonaviy payloadlarga zarur headerlarni qo'shish
- **Batch Processing:** Yagona requestlarni massivlar sifatida o'rash
- **Binary Protocols:** Length prefixlar, magic baytlarni qo'shish
- **CSV Processing:** Headersiz CSV yuklashlarga headerlarni in'ektsiya qilish

**Muhim eslatmalar:**
- **Content-Length:** Yangi o'lchamni aks ettirish uchun yangilanishi kerak
- **Binary Safety:** Matn va binary ma'lumotlar bilan ishlaydi
- **Close Original:** MultiReader closerga egalik qilmaydi
- **Streaming:** Buferlash yo'q, katta payloadlar uchun samarali

**Umumiy foydalanish stsenariylari:**
- **Legacy Integration:** Zamonaviy payloadlarga zarur headerlarni qo'shish
- **Batch Processing:** Yagona requestlarni massivlar sifatida o'rash
- **Binary Protocols:** Length prefixlar, magic baytlarni qo'shish
- **CSV Processing:** Headersiz CSV yuklashlarga headerlarni in'ektsiya qilish

PrependBody bo'lmasa, prefikslar qo'shish butun bodyni o'qish, birlashtirish va yangi body yaratishni talab qiladi — samarasiz va xotira intensiv.`,
			solutionCode: `package httpx

import (
	"bytes"
	"io"
	"net/http"
)

func PrependBody(prefix []byte, next http.Handler) http.Handler {
	if next == nil {	// nil next handlerni ishlash
		return http.HandlerFunc(func(http.ResponseWriter, *http.Request) {})
	}
	if len(prefix) == 0 {	// Prefiks bo'sh ekanligini tekshirish
		return next	// Agar prefiks bo'lmasa middleware ni o'tkazish
	}
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		reader := io.MultiReader(bytes.NewReader(prefix), r.Body)	// Prefiks va bodyni birlashtirish
		clone := r.Clone(r.Context())	// Mutatsiyadan qochish uchun requestni klonlash
		clone.Body = &bodyReadCloser{Reader: reader, closer: r.Body}	// ReadCloser sifatida o'rash
		if clone.ContentLength >= 0 {	// Content-Length ma'lumligini tekshirish
			clone.ContentLength += int64(len(prefix))	// Content-Length ga prefiks uzunligini qo'shish
		}
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
