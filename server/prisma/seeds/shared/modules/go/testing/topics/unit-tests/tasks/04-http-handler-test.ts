import { Task } from '../../../../types';

export const task: Task = {
	slug: 'go-testing-http-handler',
	title: 'HTTP Handler Testing',
	difficulty: 'medium',	tags: ['go', 'testing', 'http', 'mocking'],
	estimatedTime: '25m',	isPremium: false,
	youtubeUrl: '',
	description: `Test HTTP handlers using **mock ResponseWriter** - understand how httptest works under the hood!

**Requirements:**
1. Implement \`HealthHandler\` that writes JSON response
2. Use \`MockResponseWriter\` to capture handler output
3. Assert status code is 200
4. Assert Content-Type is "application/json"
5. Assert response body matches expected JSON

**Example:**
\`\`\`go
// Create mock request and response
req := &Request{Method: "GET", URL: "/health"}
rec := &MockResponseWriter{Headers: make(map[string]string)}

// Call handler
HealthHandler(rec, req)

// Assert results
if rec.StatusCode != 200 {
    t.Errorf("status = %d; want 200", rec.StatusCode)
}
\`\`\`

**Why This Matters:**
This is exactly how \`httptest.NewRecorder()\` works internally - it implements
ResponseWriter interface to capture what handler writes!

**Constraints:**
- Use the provided mock types (no external packages)
- Test status code, headers, and body
- Handler must set Content-Type header`,
	initialCode: `package httphandler_test

// Mock HTTP types (simplified version of net/http)
const StatusOK = 200

// Request represents HTTP request
type Request struct {
	Method string
	URL    string
	Body   string
}

// ResponseWriter interface (like http.ResponseWriter)
type ResponseWriter interface {
	Header() map[string]string
	Write([]byte) (int, error)
	WriteHeader(statusCode int)
}

// MockResponseWriter captures handler output (like httptest.Recorder)
type MockResponseWriter struct {
	StatusCode int
	Headers    map[string]string
	Body       []byte
}

func (w *MockResponseWriter) Header() map[string]string {
	if w.Headers == nil {
		w.Headers = make(map[string]string)
	}
	return w.Headers
}

func (w *MockResponseWriter) Write(b []byte) (int, error) {
	if w.StatusCode == 0 {
		w.StatusCode = StatusOK // Default status
	}
	w.Body = append(w.Body, b...)
	return len(b), nil
}

func (w *MockResponseWriter) WriteHeader(statusCode int) {
	w.StatusCode = statusCode
}

func (w *MockResponseWriter) BodyString() string {
	return string(w.Body)
}

// TODO: Implement HealthHandler that returns {"status": "ok"}
func HealthHandler(w ResponseWriter, r *Request) {
	// TODO: Set Content-Type header
	// TODO: Write status code
	// TODO: Write JSON response body
}

// TODO: Test HealthHandler using mock types
func TestHealthHandler(t *T) {
	// TODO: Create mock request and response
	// TODO: Call handler
	// TODO: Assert status code, headers, and body
}`,
	solutionCode: `package httphandler_test

// Mock HTTP types (simplified version of net/http)
const StatusOK = 200

// Request represents HTTP request
type Request struct {
	Method string
	URL    string
	Body   string
}

// ResponseWriter interface (like http.ResponseWriter)
type ResponseWriter interface {
	Header() map[string]string
	Write([]byte) (int, error)
	WriteHeader(statusCode int)
}

// MockResponseWriter captures handler output (like httptest.Recorder)
type MockResponseWriter struct {
	StatusCode int
	Headers    map[string]string
	Body       []byte
}

func (w *MockResponseWriter) Header() map[string]string {
	if w.Headers == nil {
		w.Headers = make(map[string]string)
	}
	return w.Headers
}

func (w *MockResponseWriter) Write(b []byte) (int, error) {
	if w.StatusCode == 0 {
		w.StatusCode = StatusOK // Default status
	}
	w.Body = append(w.Body, b...)
	return len(b), nil
}

func (w *MockResponseWriter) WriteHeader(statusCode int) {
	w.StatusCode = statusCode
}

func (w *MockResponseWriter) BodyString() string {
	return string(w.Body)
}

// HealthHandler returns {"status": "ok"} with proper headers
func HealthHandler(w ResponseWriter, r *Request) {
	w.Header()["Content-Type"] = "application/json"  // Set content type
	w.WriteHeader(StatusOK)                          // Set status code
	w.Write([]byte("{\\"status\\": \\"ok\\"}"))          // Write JSON response
}

func TestHealthHandler(t *T) {
	// Create mock request
	req := &Request{Method: "GET", URL: "/health"}

	// Create mock response recorder
	rec := &MockResponseWriter{Headers: make(map[string]string)}

	// Call handler
	HealthHandler(rec, req)

	// Assert status code
	if rec.StatusCode != StatusOK {
		t.Errorf("status code = %d; want %d", rec.StatusCode, StatusOK)
	}

	// Assert Content-Type header
	contentType := rec.Headers["Content-Type"]
	if contentType != "application/json" {
		t.Errorf("Content-Type = %q; want %q", contentType, "application/json")
	}

	// Assert response body
	expected := "{\\"status\\": \\"ok\\"}"
	if rec.BodyString() != expected {
		t.Errorf("body = %q; want %q", rec.BodyString(), expected)
	}
}`,
		hint1: `Set headers BEFORE calling WriteHeader or Write. Use: w.Header()["Content-Type"] = "application/json"`,
		hint2: `MockResponseWriter captures everything the handler writes. Access rec.StatusCode, rec.Headers, rec.BodyString()`,
		testCode: `package httphandler_test

// Test1: Handler returns 200 OK
func Test1(t *T) {
	req := &Request{Method: "GET", URL: "/health"}
	rec := &MockResponseWriter{Headers: make(map[string]string)}
	HealthHandler(rec, req)
	if rec.StatusCode != StatusOK {
		t.Errorf("status code = %d; want %d", rec.StatusCode, StatusOK)
	}
}

// Test2: Handler returns JSON content type
func Test2(t *T) {
	req := &Request{Method: "GET", URL: "/health"}
	rec := &MockResponseWriter{Headers: make(map[string]string)}
	HealthHandler(rec, req)
	contentType := rec.Headers["Content-Type"]
	if contentType != "application/json" {
		t.Errorf("Content-Type = %q; want %q", contentType, "application/json")
	}
}

// Test3: Handler returns correct body
func Test3(t *T) {
	req := &Request{Method: "GET", URL: "/health"}
	rec := &MockResponseWriter{Headers: make(map[string]string)}
	HealthHandler(rec, req)
	expected := "{\\"status\\": \\"ok\\"}"
	if rec.BodyString() != expected {
		t.Errorf("body = %q; want %q", rec.BodyString(), expected)
	}
}

// Test4: Handler works with POST method
func Test4(t *T) {
	req := &Request{Method: "POST", URL: "/health"}
	rec := &MockResponseWriter{Headers: make(map[string]string)}
	HealthHandler(rec, req)
	if rec.StatusCode != StatusOK {
		t.Errorf("POST request: status code = %d; want %d", rec.StatusCode, StatusOK)
	}
}

// Test5: Body is not empty
func Test5(t *T) {
	req := &Request{Method: "GET", URL: "/health"}
	rec := &MockResponseWriter{Headers: make(map[string]string)}
	HealthHandler(rec, req)
	if len(rec.Body) == 0 {
		t.Error("response body should not be empty")
	}
}

// Test6: Response contains status field
func Test6(t *T) {
	req := &Request{Method: "GET", URL: "/health"}
	rec := &MockResponseWriter{Headers: make(map[string]string)}
	HealthHandler(rec, req)
	body := rec.BodyString()
	if body == "" {
		t.Error("body should not be empty")
	}
}

// Test7: Multiple requests work correctly
func Test7(t *T) {
	for i := 0; i < 3; i++ {
		req := &Request{Method: "GET", URL: "/health"}
		rec := &MockResponseWriter{Headers: make(map[string]string)}
		HealthHandler(rec, req)
		if rec.StatusCode != StatusOK {
			t.Errorf("request %d: status code = %d; want %d", i, rec.StatusCode, StatusOK)
		}
	}
}

// Test8: Handler with query parameters in URL
func Test8(t *T) {
	req := &Request{Method: "GET", URL: "/health?debug=true"}
	rec := &MockResponseWriter{Headers: make(map[string]string)}
	HealthHandler(rec, req)
	if rec.StatusCode != StatusOK {
		t.Errorf("status code = %d; want %d", rec.StatusCode, StatusOK)
	}
}

// Test9: Response headers are set
func Test9(t *T) {
	req := &Request{Method: "GET", URL: "/health"}
	rec := &MockResponseWriter{Headers: make(map[string]string)}
	HealthHandler(rec, req)
	if rec.Headers["Content-Type"] == "" {
		t.Error("Content-Type header should be set")
	}
}

// Test10: Complete health check validation
func Test10(t *T) {
	req := &Request{Method: "GET", URL: "/health"}
	rec := &MockResponseWriter{Headers: make(map[string]string)}
	HealthHandler(rec, req)

	if rec.StatusCode != StatusOK {
		t.Errorf("status = %d; want 200", rec.StatusCode)
	}
	if rec.Headers["Content-Type"] != "application/json" {
		t.Error("wrong content type")
	}
	if rec.BodyString() != "{\\"status\\": \\"ok\\"}" {
		t.Error("wrong body")
	}
}
`,
		whyItMatters: `Understanding HTTP handler testing is crucial for building reliable web APIs.

**What You're Learning:**
This task teaches the EXACT pattern that \`net/http/httptest\` uses internally:
- \`MockResponseWriter\` is a simplified \`httptest.ResponseRecorder\`
- \`ResponseWriter\` interface is identical to \`http.ResponseWriter\`
- Same testing approach, but now you understand the mechanics!

**Why Mock Types Matter:**
- **Insight:** See how frameworks implement testing utilities
- **Portability:** Pattern works in any environment (sandboxes, CI)
- **Debugging:** Easier to understand test failures

**Real-World httptest Equivalent:**
\`\`\`go
// Using net/http/httptest (production code):
req := httptest.NewRequest("GET", "/health", nil)
rec := httptest.NewRecorder()
handler.ServeHTTP(rec, req)

// Our mock version (same pattern):
req := &Request{Method: "GET", URL: "/health"}
rec := &MockResponseWriter{Headers: make(map[string]string)}
HealthHandler(rec, req)
\`\`\`

**Production Benefits:**
Testing handlers catches:
- **Status Codes:** Return 404 instead of 200 for missing resources
- **Headers:** Missing CORS, wrong Content-Type
- **JSON Schema:** Field typos, wrong types
- **Error Cases:** Panics on invalid input

At Google, every HTTP handler has tests using this exact pattern - understanding
it makes you a better engineer.`,
	order: 3,
	translations: {
		ru: {
			title: 'Тестирование HTTP handler',
			description: `Тестируйте HTTP обработчики используя **mock ResponseWriter** - поймите как httptest работает под капотом!

**Требования:**
1. Реализуйте \`HealthHandler\` который пишет JSON ответ
2. Используйте \`MockResponseWriter\` для захвата вывода handler'а
3. Проверьте что код статуса 200
4. Проверьте что Content-Type "application/json"
5. Проверьте что тело ответа соответствует ожидаемому JSON

**Пример:**
\`\`\`go
// Создать mock запрос и ответ
req := &Request{Method: "GET", URL: "/health"}
rec := &MockResponseWriter{Headers: make(map[string]string)}

// Вызвать handler
HealthHandler(rec, req)

// Проверить результаты
if rec.StatusCode != 200 {
    t.Errorf("status = %d; want 200", rec.StatusCode)
}
\`\`\`

**Почему это важно:**
Это именно то, как \`httptest.NewRecorder()\` работает внутри - он реализует
интерфейс ResponseWriter для захвата того, что пишет handler!`,
			hint1: `Устанавливайте заголовки ДО вызова WriteHeader или Write. Используйте: w.Header()["Content-Type"] = "application/json"`,
			hint2: `MockResponseWriter захватывает всё что пишет handler. Используйте rec.StatusCode, rec.Headers, rec.BodyString()`,
			whyItMatters: `Понимание тестирования HTTP handler'ов критически важно для создания надёжных веб API.

**Что вы изучаете:**
Эта задача учит ТОЧНЫЙ паттерн который \`net/http/httptest\` использует внутри:
- \`MockResponseWriter\` - упрощённый \`httptest.ResponseRecorder\`
- Интерфейс \`ResponseWriter\` идентичен \`http.ResponseWriter\`
- Тот же подход к тестированию, но теперь вы понимаете механику!

**Почему mock типы важны:**
- **Понимание:** Видите как фреймворки реализуют утилиты для тестирования
- **Портативность:** Паттерн работает в любом окружении
- **Отладка:** Легче понять причины провала тестов

В Google каждый HTTP handler имеет тесты использующие этот паттерн.`,
			solutionCode: `package httphandler_test

// Mock HTTP типы (упрощённая версия net/http)
const StatusOK = 200

// Request представляет HTTP запрос
type Request struct {
	Method string
	URL    string
	Body   string
}

// ResponseWriter интерфейс (как http.ResponseWriter)
type ResponseWriter interface {
	Header() map[string]string
	Write([]byte) (int, error)
	WriteHeader(statusCode int)
}

// MockResponseWriter захватывает вывод handler'а
type MockResponseWriter struct {
	StatusCode int
	Headers    map[string]string
	Body       []byte
}

func (w *MockResponseWriter) Header() map[string]string {
	if w.Headers == nil {
		w.Headers = make(map[string]string)
	}
	return w.Headers
}

func (w *MockResponseWriter) Write(b []byte) (int, error) {
	if w.StatusCode == 0 {
		w.StatusCode = StatusOK
	}
	w.Body = append(w.Body, b...)
	return len(b), nil
}

func (w *MockResponseWriter) WriteHeader(statusCode int) {
	w.StatusCode = statusCode
}

func (w *MockResponseWriter) BodyString() string {
	return string(w.Body)
}

// HealthHandler возвращает {"status": "ok"} с правильными заголовками
func HealthHandler(w ResponseWriter, r *Request) {
	w.Header()["Content-Type"] = "application/json"  // Установить тип контента
	w.WriteHeader(StatusOK)                          // Установить код статуса
	w.Write([]byte("{\\"status\\": \\"ok\\"}"))          // Записать JSON ответ
}

func TestHealthHandler(t *T) {
	// Создать mock запрос
	req := &Request{Method: "GET", URL: "/health"}

	// Создать mock рекордер ответа
	rec := &MockResponseWriter{Headers: make(map[string]string)}

	// Вызвать handler
	HealthHandler(rec, req)

	// Проверить код статуса
	if rec.StatusCode != StatusOK {
		t.Errorf("код статуса = %d; ожидается %d", rec.StatusCode, StatusOK)
	}

	// Проверить заголовок Content-Type
	contentType := rec.Headers["Content-Type"]
	if contentType != "application/json" {
		t.Errorf("Content-Type = %q; ожидается %q", contentType, "application/json")
	}

	// Проверить тело ответа
	expected := "{\\"status\\": \\"ok\\"}"
	if rec.BodyString() != expected {
		t.Errorf("body = %q; ожидается %q", rec.BodyString(), expected)
	}
}`
		},
		uz: {
			title: `HTTP handler testlash`,
			description: `**mock ResponseWriter** dan foydalanib HTTP handlerlarni test qiling - httptest ichkaridan qanday ishlashini tushuning!

**Talablar:**
1. JSON javobini yozadigan \`HealthHandler\` ni amalga oshiring
2. Handler chiqishini olish uchun \`MockResponseWriter\` dan foydalaning
3. Status kod 200 ekanligini tekshiring
4. Content-Type "application/json" ekanligini tekshiring
5. Javob tanasi kutilgan JSON ga mos kelishini tekshiring`,
			hint1: `Headerlarni WriteHeader yoki Write chaqirishdan OLDIN o'rnating. Foydalaning: w.Header()["Content-Type"] = "application/json"`,
			hint2: `MockResponseWriter handler yozgan hamma narsani yozib oladi. rec.StatusCode, rec.Headers, rec.BodyString() dan foydalaning`,
			whyItMatters: `HTTP handler testlarini tushunish ishonchli veb API yaratish uchun juda muhim.

Bu vazifa \`net/http/httptest\` ichkaridan foydalanadigan ANIQ patternni o'rgatadi.`,
			solutionCode: `package httphandler_test

const StatusOK = 200

type Request struct {
	Method string
	URL    string
	Body   string
}

type ResponseWriter interface {
	Header() map[string]string
	Write([]byte) (int, error)
	WriteHeader(statusCode int)
}

type MockResponseWriter struct {
	StatusCode int
	Headers    map[string]string
	Body       []byte
}

func (w *MockResponseWriter) Header() map[string]string {
	if w.Headers == nil {
		w.Headers = make(map[string]string)
	}
	return w.Headers
}

func (w *MockResponseWriter) Write(b []byte) (int, error) {
	if w.StatusCode == 0 {
		w.StatusCode = StatusOK
	}
	w.Body = append(w.Body, b...)
	return len(b), nil
}

func (w *MockResponseWriter) WriteHeader(statusCode int) {
	w.StatusCode = statusCode
}

func (w *MockResponseWriter) BodyString() string {
	return string(w.Body)
}

func HealthHandler(w ResponseWriter, r *Request) {
	w.Header()["Content-Type"] = "application/json"
	w.WriteHeader(StatusOK)
	w.Write([]byte("{\\"status\\": \\"ok\\"}"))
}

func TestHealthHandler(t *T) {
	req := &Request{Method: "GET", URL: "/health"}
	rec := &MockResponseWriter{Headers: make(map[string]string)}
	HealthHandler(rec, req)

	if rec.StatusCode != StatusOK {
		t.Errorf("status kod = %d; kutilgan %d", rec.StatusCode, StatusOK)
	}
	contentType := rec.Headers["Content-Type"]
	if contentType != "application/json" {
		t.Errorf("Content-Type = %q; kutilgan %q", contentType, "application/json")
	}
	expected := "{\\"status\\": \\"ok\\"}"
	if rec.BodyString() != expected {
		t.Errorf("body = %q; kutilgan %q", rec.BodyString(), expected)
	}
}`
		}
	}
};

export default task;
