import { Task } from '../../../../types';

export const task: Task = {
	slug: 'go-http-tee-body',
	title: 'Tee Body Middleware',
	difficulty: 'medium',	tags: ['go', 'http', 'middleware', 'body-handling'],
	estimatedTime: '25m',	isPremium: false,
	youtubeUrl: '',
	description: `Implement **TeeBody** middleware that duplicates request body stream to a writer while passing it through to the handler.

**Requirements:**
1. Create function \`TeeBody(dst io.Writer, next http.Handler) http.Handler\`
2. Use io.TeeReader to duplicate body stream
3. Write to dst while allowing next to read from body
4. Wrap result as ReadCloser to preserve body interface
5. Handle nil next handler and dst

**Example:**
\`\`\`go
var buf bytes.Buffer

handler := TeeBody(&buf, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
    body, _ := io.ReadAll(r.Body)
    fmt.Fprintf(w, "Read: %s", body)
}))

// Request with body "Hello World"
// buf contains: "Hello World"
// Handler reads: "Hello World"
\`\`\`

**Constraints:**
- Must use io.TeeReader for stream duplication
- Must preserve ReadCloser interface for body
- Must not buffer entire body in memory`,
	initialCode: `package httpx

import (
	"io"
	"net/http"
)

// TODO: Implement TeeBody middleware
func TeeBody(dst io.Writer, next http.Handler) http.Handler {
	// TODO: Implement
}`,
	solutionCode: `package httpx

import (
	"io"
	"net/http"
)

func TeeBody(dst io.Writer, next http.Handler) http.Handler {
	if next == nil {	// Handle nil next handler
		return http.HandlerFunc(func(http.ResponseWriter, *http.Request) {})
	}
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		tee := io.TeeReader(r.Body, dst)	// Create TeeReader that writes to dst
		clone := r.Clone(r.Context())	// Clone request to avoid mutation
		clone.Body = &bodyReadCloser{Reader: tee, closer: r.Body}	// Wrap TeeReader as ReadCloser
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
			hint1: `Use io.TeeReader(r.Body, dst) to create a reader that writes to dst as it\`s read.`,
			hint2: `Wrap the TeeReader with a custom type that implements Close() using the original body closer.`,
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
	var buf bytes.Buffer
	h := TeeBody(&buf, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {}))
	if h == nil {
		t.Error("handler should not be nil")
	}
}

func Test2(t *testing.T) {
	// Test nil handler returns non-nil
	var buf bytes.Buffer
	h := TeeBody(&buf, nil)
	if h == nil {
		t.Error("nil handler should return non-nil")
	}
}

func Test3(t *testing.T) {
	// Test nil writer skips middleware
	called := false
	h := TeeBody(nil, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		called = true
	}))
	h.ServeHTTP(httptest.NewRecorder(), httptest.NewRequest("POST", "/", strings.NewReader("body")))
	if !called {
		t.Error("nil writer should skip middleware")
	}
}

func Test4(t *testing.T) {
	// Test body is copied to writer
	var buf bytes.Buffer
	h := TeeBody(&buf, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		io.ReadAll(r.Body)
	}))
	h.ServeHTTP(httptest.NewRecorder(), httptest.NewRequest("POST", "/", strings.NewReader("Hello World")))
	if buf.String() != "Hello World" {
		t.Errorf("tee buf = %q, want 'Hello World'", buf.String())
	}
}

func Test5(t *testing.T) {
	// Test body can still be read by handler
	var body []byte
	var buf bytes.Buffer
	h := TeeBody(&buf, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ = io.ReadAll(r.Body)
	}))
	h.ServeHTTP(httptest.NewRecorder(), httptest.NewRequest("POST", "/", strings.NewReader("Hello World")))
	if string(body) != "Hello World" {
		t.Errorf("body = %q, want 'Hello World'", string(body))
	}
}

func Test6(t *testing.T) {
	// Test empty body
	var buf bytes.Buffer
	h := TeeBody(&buf, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		io.ReadAll(r.Body)
	}))
	h.ServeHTTP(httptest.NewRecorder(), httptest.NewRequest("POST", "/", bytes.NewReader(nil)))
	if buf.Len() != 0 {
		t.Errorf("buf len = %d, want 0", buf.Len())
	}
}

func Test7(t *testing.T) {
	// Test method preserved
	var buf bytes.Buffer
	var method string
	h := TeeBody(&buf, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		method = r.Method
	}))
	h.ServeHTTP(httptest.NewRecorder(), httptest.NewRequest("PUT", "/", strings.NewReader("data")))
	if method != "PUT" {
		t.Errorf("method = %q, want PUT", method)
	}
}

func Test8(t *testing.T) {
	// Test headers preserved
	var buf bytes.Buffer
	var header string
	h := TeeBody(&buf, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
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
	var buf bytes.Buffer
	h := TeeBody(&buf, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		io.ReadAll(r.Body)
		w.WriteHeader(http.StatusCreated)
		w.Write([]byte("created"))
	}))
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest("POST", "/", strings.NewReader("data")))
	if rec.Code != http.StatusCreated || rec.Body.String() != "created" {
		t.Error("response should pass through")
	}
}

func Test10(t *testing.T) {
	// Test large body
	var buf bytes.Buffer
	data := strings.Repeat("x", 10000)
	h := TeeBody(&buf, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		io.ReadAll(r.Body)
	}))
	h.ServeHTTP(httptest.NewRecorder(), httptest.NewRequest("POST", "/", strings.NewReader(data)))
	if buf.Len() != 10000 {
		t.Errorf("buf len = %d, want 10000", buf.Len())
	}
}`,
			whyItMatters: `TeeBody enables simultaneous logging, archiving, and processing of request bodies without double-reading or buffering.

**Why Tee Body:**
- **Request Logging:** Write request bodies to logs while processing
- **Archiving:** Save request payloads to files/database
- **Debugging:** Capture all requests for replay/analysis
- **Compliance:** Record requests for audit trails

**Production Pattern:**
\`\`\`go
// Log all requests to file
func RequestArchiver(logFile *os.File) func(http.Handler) http.Handler {
    return func(next http.Handler) http.Handler {
        return TeeBody(logFile, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
	// Write timestamp and metadata
            fmt.Fprintf(logFile, "\\n--- %s %s %s ---\\n", time.Now().Format(time.RFC3339), r.Method, r.URL.Path)

	// Body is being written to logFile as handler reads it
            next.ServeHTTP(w, r)
        }))
    }
}

// Debug middleware that captures requests
func DebugCapture(next http.Handler) http.Handler {
    return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        var buf bytes.Buffer

	// Capture body while allowing handler to read
        handler := TeeBody(&buf, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
            next.ServeHTTP(w, r)
        }))

        handler.ServeHTTP(w, r)

	// After request completes, save captured body
        if buf.Len() > 0 {
            debugLog.Printf("Request body: %s", buf.String())
        }
    })
}

// Multi-target tee (log to multiple destinations)
func MultiTeeBody(destinations ...io.Writer) func(http.Handler) http.Handler {
    return func(next http.Handler) http.Handler {
        multiWriter := io.MultiWriter(destinations...)
        return TeeBody(multiWriter, next)
    }
}

// Usage: Tee to file, stdout, and buffer
handler := MultiTeeBody(logFile, os.Stdout, &captureBuffer)(finalHandler)

// Conditional tee based on content type
func ConditionalTee(next http.Handler) http.Handler {
    return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        contentType := r.Header.Get("Content-Type")

	// Only tee JSON requests
        if strings.Contains(contentType, "application/json") {
            var jsonLog bytes.Buffer
            TeeBody(&jsonLog, next).ServeHTTP(w, r)

	// Process captured JSON
            log.Printf("JSON request: %s", jsonLog.String())
        } else {
            next.ServeHTTP(w, r)
        }
    })
}

// Async archiving (non-blocking)
func AsyncArchiver(archive chan<- []byte) func(http.Handler) http.Handler {
    return func(next http.Handler) http.Handler {
        return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
            pr, pw := io.Pipe()

	// Async write to archive channel
            go func() {
                defer pw.Close()
                data, _ := io.ReadAll(pr)
                select {
                case archive <- data:
                case <-time.After(time.Second):
                    log.Printf("archive timeout")
                }
            }()

            TeeBody(pw, next).ServeHTTP(w, r)
        })
    }
}
\`\`\`

**Real-World Benefits:**
- **Zero Copy:** Stream data without buffering entire body
- **Low Memory:** Process large bodies without loading into RAM
- **Simultaneous Processing:** Log while handler processes
- **Flexible Destinations:** Tee to files, networks, buffers

**TeeReader Advantages:**
- **Efficient:** No double-reading or buffering
- **Streaming:** Works with large bodies
- **Simple:** Single line of code for duplication
- **Composable:** Chain multiple tees with io.MultiWriter

**Common Use Cases:**
- **Request Replay:** Capture requests for testing/debugging
- **Compliance:** Save financial transactions to audit logs
- **Analytics:** Stream request data to analytics pipeline
- **Debugging:** Log request bodies in development

**Important Notes:**
- **Read Once:** TeeReader only works as data is read
- **Order Matters:** Data written to dst as it's consumed
- **Error Handling:** Read errors affect both streams
- **Close Original:** Always close original body, not TeeReader

Without TeeBody, logging request bodies requires either reading twice (inefficient) or buffering the entire body (memory intensive).`,	order: 11,
	translations: {
		ru: {
			title: 'Клонирование тела запроса для повторного чтения',
			description: `Реализуйте middleware **TeeBody**, который дублирует поток тела запроса в writer пропуская его через handler.

**Требования:**
1. Создайте функцию \`TeeBody(dst io.Writer, next http.Handler) http.Handler\`
2. Используйте io.TeeReader для дублирования потока body
3. Записывайте в dst позволяя next читать из body
4. Оберните результат как ReadCloser для сохранения интерфейса body
5. Обработайте nil handler и dst

**Пример:**
\`\`\`go
var buf bytes.Buffer

handler := TeeBody(&buf, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
    body, _ := io.ReadAll(r.Body)
    fmt.Fprintf(w, "Read: %s", body)
}))

// Запрос с телом "Hello World"
// buf содержит: "Hello World"
// Handler читает: "Hello World"
\`\`\`

**Ограничения:**
- Должен использовать io.TeeReader для дублирования потока
- Должен сохранять интерфейс ReadCloser для body
- Не должен буферизировать всё тело в памяти`,
			hint1: `Используйте io.TeeReader(r.Body, dst) для создания reader который записывает в dst при чтении.`,
			hint2: `Оберните TeeReader кастомным типом который реализует Close() используя оригинальный body closer.`,
			whyItMatters: `TeeBody позволяет одновременное логирование, архивирование и обработку тел запросов без двойного чтения или буферизации.

**Почему Tee Body:**
- **Логирование запросов:** Запись тел запросов в логи во время обработки
- **Архивирование:** Сохранение request payloads в файлы/базу данных
- **Отладка:** Захват всех запросов для replay/анализа
- **Compliance:** Запись запросов для audit trails

**Продакшен паттерн:**
\`\`\`go
// Логирование всех запросов в файл
func RequestArchiver(logFile *os.File) func(http.Handler) http.Handler {
    return func(next http.Handler) http.Handler {
        return TeeBody(logFile, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
            fmt.Fprintf(logFile, "\\n--- %s %s %s ---\\n", time.Now().Format(time.RFC3339), r.Method, r.URL.Path)
            next.ServeHTTP(w, r)
        }))
    }
}

// Debug middleware для захвата запросов
func DebugCapture(next http.Handler) http.Handler {
    return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        var buf bytes.Buffer

        handler := TeeBody(&buf, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
            next.ServeHTTP(w, r)
        }))

        handler.ServeHTTP(w, r)

        if buf.Len() > 0 {
            debugLog.Printf("Request body: %s", buf.String())
        }
    })
}

// Multi-target tee (лог в несколько destinations)
func MultiTeeBody(destinations ...io.Writer) func(http.Handler) http.Handler {
    return func(next http.Handler) http.Handler {
        multiWriter := io.MultiWriter(destinations...)
        return TeeBody(multiWriter, next)
    }
}

// Usage: Tee в файл, stdout и buffer
handler := MultiTeeBody(logFile, os.Stdout, &captureBuffer)(finalHandler)

// Условное Tee на основе content type
func ConditionalTee(next http.Handler) http.Handler {
    return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        contentType := r.Header.Get("Content-Type")

        // Tee только JSON запросы
        if strings.Contains(contentType, "application/json") {
            var jsonLog bytes.Buffer
            TeeBody(&jsonLog, next).ServeHTTP(w, r)
            log.Printf("JSON request: %s", jsonLog.String())
        } else {
            next.ServeHTTP(w, r)
        }
    })
}

// Async архивирование (неблокирующее)
func AsyncArchiver(archive chan<- []byte) func(http.Handler) http.Handler {
    return func(next http.Handler) http.Handler {
        return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
            pr, pw := io.Pipe()

            go func() {
                defer pw.Close()
                data, _ := io.ReadAll(pr)
                select {
                case archive <- data:
                case <-time.After(time.Second):
                    log.Printf("archive timeout")
                }
            }()

            TeeBody(pw, next).ServeHTTP(w, r)
        })
    }
}
\`\`\`

**Практические преимущества:**
- **Zero Copy:** Streaming данных без буферизации всего body
- **Low Memory:** Обработка больших bodies без загрузки в RAM
- **Simultaneous Processing:** Логирование пока handler обрабатывает
- **Flexible Destinations:** Tee в файлы, сети, буферы

**Преимущества TeeReader:**
- **Эффективность:** Нет double-reading или буферизации
- **Streaming:** Работает с большими bodies
- **Простота:** Одна строка кода для дублирования
- **Composable:** Chaining нескольких tees с io.MultiWriter

**Общие сценарии использования:**
- **Request Replay:** Захват запросов для тестирования/отладки
- **Compliance:** Сохранение финансовых транзакций в audit логи
- **Analytics:** Потоковая передача данных запросов в analytics pipeline
- **Debugging:** Логирование request bodies в разработке

**Важные заметки:**
- **Read Once:** TeeReader работает только когда данные читаются
- **Order Matters:** Данные пишутся в dst по мере потребления
- **Error Handling:** Ошибки чтения влияют на оба потока
- **Close Original:** Всегда закрывайте оригинальный body, не TeeReader

**Общие сценарии использования:**
- **Request Replay:** Захват запросов для тестирования/отладки
- **Compliance:** Сохранение финансовых транзакций в audit логи
- **Analytics:** Потоковая передача данных запросов в analytics pipeline
- **Debugging:** Логирование request bodies в разработке

Без TeeBody логирование тел запросов требует либо двойного чтения (неэффективно), либо буферизации всего body (требует много памяти).`,
			solutionCode: `package httpx

import (
	"io"
	"net/http"
)

func TeeBody(dst io.Writer, next http.Handler) http.Handler {
	if next == nil {	// Обработка nil next handler
		return http.HandlerFunc(func(http.ResponseWriter, *http.Request) {})
	}
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		tee := io.TeeReader(r.Body, dst)	// Создание TeeReader который пишет в dst
		clone := r.Clone(r.Context())	// Клонирование запроса во избежание мутации
		clone.Body = &bodyReadCloser{Reader: tee, closer: r.Body}	// Обёртывание TeeReader как ReadCloser
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
			title: 'Qayta o\'qish uchun request bodyni klonlash',
			description: `Request body oqimini writerga dublikaqi qiluvchi va handlerga o'tkazuvchi **TeeBody** middleware ni amalga oshiring.

**Talablar:**
1. \`TeeBody(dst io.Writer, next http.Handler) http.Handler\` funksiyasini yarating
2. Body oqimini nusxalash uchun io.TeeReader dan foydalaning
3. dst ga yozib, next ga bodydan o'qish imkonini bering
4. Body interfeysini saqlash uchun natijani ReadCloser sifatida o'rang
5. nil handler va dst ni ishlang

**Misol:**
\`\`\`go
var buf bytes.Buffer

handler := TeeBody(&buf, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
    body, _ := io.ReadAll(r.Body)
    fmt.Fprintf(w, "Read: %s", body)
}))

// "Hello World" bodysi bilan request
// buf saqlaydi: "Hello World"
// Handler o'qiydi: "Hello World"
\`\`\`

**Cheklovlar:**
- Oqimni nusxalash uchun io.TeeReader dan foydalanishi kerak
- Body uchun ReadCloser interfeysini saqlashi kerak
- Butun bodyni xotirada buferlamasligi kerak`,
			hint1: `O'qishda dst ga yozuvchi reader yaratish uchun io.TeeReader(r.Body, dst) dan foydalaning.`,
			hint2: `TeeReader ni asl body closer dan foydalanib Close() ni amalga oshiruvchi custom tur bilan o'rang.`,
			whyItMatters: `TeeBody ikki marta o'qish yoki buferlashsiz request bodylarini bir vaqtning o'zida loglash, arxivlash va qayta ishlash imkonini beradi.

**Nima uchun Tee Body:**
- **Request loglash:** Qayta ishlash vaqtida request bodylarini loglarga yozish
- **Arxivlash:** Request payloadlarini fayllar/bazaga saqlash
- **Debug:** Replay/tahlil uchun barcha requestlarni ushlab qolish
- **Compliance:** Audit trails uchun requestlarni yozib qolish

**Ishlab chiqarish patterni:**
\`\`\`go
// Barcha requestlarni faylga loglash
func RequestArchiver(logFile *os.File) func(http.Handler) http.Handler {
    return func(next http.Handler) http.Handler {
        return TeeBody(logFile, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
            fmt.Fprintf(logFile, "\\n--- %s %s %s ---\\n", time.Now().Format(time.RFC3339), r.Method, r.URL.Path)
            next.ServeHTTP(w, r)
        }))
    }
}

// Requestlarni ushlab qoluvchi Debug middleware
func DebugCapture(next http.Handler) http.Handler {
    return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        var buf bytes.Buffer

        handler := TeeBody(&buf, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
            next.ServeHTTP(w, r)
        }))

        handler.ServeHTTP(w, r)

        if buf.Len() > 0 {
            debugLog.Printf("Request body: %s", buf.String())
        }
    })
}

// Multi-target tee (bir nechta destinationlarga log)
func MultiTeeBody(destinations ...io.Writer) func(http.Handler) http.Handler {
    return func(next http.Handler) http.Handler {
        multiWriter := io.MultiWriter(destinations...)
        return TeeBody(multiWriter, next)
    }
}

// Usage: Fayl, stdout va bufferga Tee
handler := MultiTeeBody(logFile, os.Stdout, &captureBuffer)(finalHandler)

// Content typega asoslanib shartli Tee
func ConditionalTee(next http.Handler) http.Handler {
    return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        contentType := r.Header.Get("Content-Type")

        // Faqat JSON requestlarni tee qilish
        if strings.Contains(contentType, "application/json") {
            var jsonLog bytes.Buffer
            TeeBody(&jsonLog, next).ServeHTTP(w, r)
            log.Printf("JSON request: %s", jsonLog.String())
        } else {
            next.ServeHTTP(w, r)
        }
    })
}

// Asinxron arxivlash (blokirovka qilmaydigan)
func AsyncArchiver(archive chan<- []byte) func(http.Handler) http.Handler {
    return func(next http.Handler) http.Handler {
        return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
            pr, pw := io.Pipe()

            go func() {
                defer pw.Close()
                data, _ := io.ReadAll(pr)
                select {
                case archive <- data:
                case <-time.After(time.Second):
                    log.Printf("archive timeout")
                }
            }()

            TeeBody(pw, next).ServeHTTP(w, r)
        })
    }
}
\`\`\`

**Amaliy foydalari:**
- **Zero Copy:** Butun bodyni buferlashsiz ma'lumotlarni streaming
- **Kam xotira:** Katta bodylarni RAMga yuklamasdan qayta ishlash
- **Bir vaqtda qayta ishlash:** Handler qayta ishlayotganda loglash
- **Moslashuvchan destinationlar:** Fayllar, tarmoqlar, buferlarga tee

**TeeReader afzalliklari:**
- **Samaradorlik:** Double-reading yoki buferlash yo'q
- **Streaming:** Katta bodylar bilan ishlaydi
- **Soddalik:** Nusxalash uchun bitta kod qatori
- **Composable:** io.MultiWriter bilan bir nechta telarni chaining

**Umumiy foydalanish stsenariylari:**
- **Request Replay:** Test qilish/debug uchun requestlarni ushlab qolish
- **Compliance:** Moliyaviy tranzaksiyalarni audit loglariga saqlash
- **Analytics:** Request ma'lumotlarini analytics pipeline ga oqim qilish
- **Debugging:** Dasturni ishlab chiqishda request bodylarini loglash

**Muhim eslatmalar:**
- **Bir marta o'qish:** TeeReader faqat ma'lumotlar o'qilganda ishlaydi
- **Tartib muhim:** Ma'lumotlar iste'mol qilinganida dst ga yoziladi
- **Xato ishlash:** O'qish xatolari ikkala oqimga ta'sir qiladi
- **Close Original:** Har doim asl bodyni yoping, TeeReader ni emas

**Umumiy foydalanish stsenariylari:**
- **Request Replay:** Test qilish/debug uchun requestlarni ushlab qolish
- **Compliance:** Moliyaviy tranzaksiyalarni audit loglariga saqlash
- **Analytics:** Request ma'lumotlarini analytics pipeline ga oqim qilish
- **Debugging:** Dasturni ishlab chiqishda request bodylarini loglash

TeeBody bo'lmasa, request bodylarini loglash ikki marta o'qishni (samarasiz) yoki butun bodyni buferlashni (xotira intensiv) talab qiladi.`,
			solutionCode: `package httpx

import (
	"io"
	"net/http"
)

func TeeBody(dst io.Writer, next http.Handler) http.Handler {
	if next == nil {	// nil next handlerni ishlash
		return http.HandlerFunc(func(http.ResponseWriter, *http.Request) {})
	}
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		tee := io.TeeReader(r.Body, dst)	// dst ga yozuvchi TeeReader yaratish
		clone := r.Clone(r.Context())	// Mutatsiyadan qochish uchun requestni klonlash
		clone.Body = &bodyReadCloser{Reader: tee, closer: r.Body}	// TeeReader ni ReadCloser sifatida o'rash
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
