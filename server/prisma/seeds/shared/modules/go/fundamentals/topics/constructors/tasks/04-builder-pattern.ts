import { Task } from '../../../../types';

export const task: Task = {
	slug: 'go-fundamentals-builder-pattern',
	title: 'Builder Pattern for Complex Objects',
	difficulty: 'medium',
	tags: ['go', 'patterns', 'constructors', 'builder'],
	estimatedTime: '30m',
	isPremium: false,
	youtubeUrl: '',
	description: `Implement the **Builder Pattern** to construct complex HTTP request objects with a fluent, chainable API.

**Pattern Overview:**
The Builder Pattern separates the construction of a complex object from its representation. It's particularly useful when an object needs many optional parameters or when construction requires multiple steps. The pattern provides a fluent interface where method calls can be chained together, making the code more readable and maintainable.

**Key Components:**

1. **HTTPRequest** - The final immutable object:
   \`\`\`go
   type HTTPRequest struct {
       method  string
       url     string
       headers map[string]string
       body    string
       timeout int  // in seconds
   }
   \`\`\`

2. **RequestBuilder** - Mutable builder for constructing requests:
   \`\`\`go
   type RequestBuilder struct {
       method  string
       url     string
       headers map[string]string
       body    string
       timeout int
   }
   \`\`\`

3. **NewRequestBuilder** - Factory for creating builder:
   \`\`\`go
   func NewRequestBuilder() *RequestBuilder
   \`\`\`
   3.1. Returns pointer to new RequestBuilder
   3.2. Initializes headers map (not nil)
   3.3. Sets default timeout to 30 seconds

4. **Builder Methods** - Chainable configuration methods:

   **Method(method string) *RequestBuilder**
   4.1. Sets HTTP method (GET, POST, etc.)
   4.2. Returns self for chaining

   **URL(url string) *RequestBuilder**
   4.3. Sets request URL
   4.4. Returns self for chaining

   **Header(key, value string) *RequestBuilder**
   4.5. Adds a header key-value pair
   4.6. Returns self for chaining

   **Body(body string) *RequestBuilder**
   4.7. Sets request body
   4.8. Returns self for chaining

   **Timeout(seconds int) *RequestBuilder**
   4.9. Sets request timeout in seconds
   4.10. Returns self for chaining

5. **Build() Method** - Finalizes and validates:
   \`\`\`go
   func (b *RequestBuilder) Build() (HTTPRequest, error)
   \`\`\`
   5.1. Validates method is not empty
   5.2. Validates URL is not empty
   5.3. Validates timeout is positive (> 0)
   5.4. Creates and returns immutable HTTPRequest
   5.5. Returns error if validation fails

6. **HTTPRequest Methods** - Read-only accessors:
   6.1. **Method() string** - Returns HTTP method
   6.2. **URL() string** - Returns URL
   6.3. **Headers() map[string]string** - Returns copy of headers (not the original map)
   6.4. **Body() string** - Returns body
   6.5. **Timeout() int** - Returns timeout in seconds

**Implementation Requirements:**

- All builder methods must return \`*RequestBuilder\` for chaining
- Build() must validate all required fields before creating HTTPRequest
- HTTPRequest should be immutable (all fields private)
- Headers() must return a copy to prevent external modification
- Default timeout is 30 seconds if not specified

**Usage Example:**
\`\`\`go
// Build a complex request with fluent API
req, err := NewRequestBuilder().
    Method("POST").
    URL("https://api.example.com/users").
    Header("Content-Type", "application/json").
    Header("Authorization", "Bearer token123").
    Body(\`{"name": "John Doe", "email": "john@example.com"}\`).
    Timeout(60).
    Build()

if err != nil {
    log.Fatal(err)
}

fmt.Println(req.Method())   // "POST"
fmt.Println(req.URL())      // "https://api.example.com/users"
fmt.Println(req.Timeout())  // 60

// Simple GET request with defaults
req, err = NewRequestBuilder().
    Method("GET").
    URL("https://api.example.com/status").
    Build()

fmt.Println(req.Timeout())  // 30 (default)

// Invalid request returns error
req, err = NewRequestBuilder().
    Method("GET").
    Build()  // Error: "url is required"
\`\`\`

**Why This Pattern:**
- **Readability:** Fluent interface is self-documenting
- **Flexibility:** Easy to add optional parameters
- **Immutability:** Final object cannot be modified after creation
- **Validation:** Centralized validation in Build() method
- **Defaults:** Easy to provide sensible default values`,
	initialCode: `package structinit

import (
	"fmt"
)

// TODO: Define HTTPRequest struct with private fields:
// method, url, headers (map[string]string), body, timeout (int)
type HTTPRequest struct {
	// TODO: Add fields
}

// TODO: Define RequestBuilder struct with same fields as HTTPRequest
// but all fields should be mutable for building
type RequestBuilder struct {
	// TODO: Add fields
}

// TODO: Implement NewRequestBuilder factory
// Initialize headers map and set default timeout to 30
func NewRequestBuilder() *RequestBuilder {
	// TODO: Implement
}

// TODO: Implement Method builder method
// Set method and return self for chaining
func (b *RequestBuilder) Method(method string) *RequestBuilder {
	// TODO: Implement
}

// TODO: Implement URL builder method
// Set url and return self for chaining
func (b *RequestBuilder) URL(url string) *RequestBuilder {
	// TODO: Implement
}

// TODO: Implement Header builder method
// Add header key-value pair and return self for chaining
func (b *RequestBuilder) Header(key, value string) *RequestBuilder {
	// TODO: Implement
}

// TODO: Implement Body builder method
// Set body and return self for chaining
func (b *RequestBuilder) Body(body string) *RequestBuilder {
	// TODO: Implement
}

// TODO: Implement Timeout builder method
// Set timeout and return self for chaining
func (b *RequestBuilder) Timeout(seconds int) *RequestBuilder {
	// TODO: Implement
}

// TODO: Implement Build method
// Validate required fields (method, url not empty; timeout > 0)
// Return HTTPRequest and nil on success, or error on validation failure
func (b *RequestBuilder) Build() (HTTPRequest, error) {
	// TODO: Implement
}

// TODO: Implement getter methods for HTTPRequest
func (r *HTTPRequest) Method() string {
	return "" // TODO: Implement
}

func (r *HTTPRequest) URL() string {
	return "" // TODO: Implement
}

// Return a COPY of the headers map to maintain immutability
func (r HTTPRequest) Headers() map[string]string {
	// TODO: Implement
}

func (r *HTTPRequest) Body() string {
	return "" // TODO: Implement
}

func (r *HTTPRequest) Timeout() int {
	return 0 // TODO: Implement
}`,
	solutionCode: `package structinit

import (
	"fmt"
	"strings"
)

type HTTPRequest struct {
	method  string                               // HTTP method (GET, POST, etc.)
	url     string                               // Request URL
	headers map[string]string                    // HTTP headers
	body    string                               // Request body
	timeout int                                  // Timeout in seconds
}

type RequestBuilder struct {
	method  string                               // Mutable HTTP method
	url     string                               // Mutable URL
	headers map[string]string                    // Mutable headers
	body    string                               // Mutable body
	timeout int                                  // Mutable timeout
}

func NewRequestBuilder() *RequestBuilder {
	return &RequestBuilder{
		headers: make(map[string]string),        // Initialize headers map
		timeout: 30,                             // Set default timeout
	}
}

func (b *RequestBuilder) Method(method string) *RequestBuilder {
	b.method = method                            // Set method
	return b                                     // Return self for chaining
}

func (b *RequestBuilder) URL(url string) *RequestBuilder {
	b.url = url                                  // Set URL
	return b                                     // Return self for chaining
}

func (b *RequestBuilder) Header(key, value string) *RequestBuilder {
	b.headers[key] = value                       // Add header
	return b                                     // Return self for chaining
}

func (b *RequestBuilder) Body(body string) *RequestBuilder {
	b.body = body                                // Set body
	return b                                     // Return self for chaining
}

func (b *RequestBuilder) Timeout(seconds int) *RequestBuilder {
	b.timeout = seconds                          // Set timeout
	return b                                     // Return self for chaining
}

func (b *RequestBuilder) Build() (HTTPRequest, error) {
	if strings.TrimSpace(b.method) == "" {       // Validate method is not empty
		return HTTPRequest{}, fmt.Errorf("method is required")
	}

	if strings.TrimSpace(b.url) == "" {          // Validate URL is not empty
		return HTTPRequest{}, fmt.Errorf("url is required")
	}

	if b.timeout <= 0 {                          // Validate timeout is positive
		return HTTPRequest{}, fmt.Errorf("timeout must be positive")
	}

	return HTTPRequest{                          // Create immutable request
		method:  b.method,
		url:     b.url,
		headers: b.headers,
		body:    b.body,
		timeout: b.timeout,
	}, nil
}

func (r HTTPRequest) Method() string {
	return r.method                              // Return HTTP method
}

func (r HTTPRequest) URL() string {
	return r.url                                 // Return URL
}

func (r HTTPRequest) Headers() map[string]string {
	copy := make(map[string]string)              // Create copy of headers
	for k, v := range r.headers {
		copy[k] = v                              // Copy each header
	}
	return copy                                  // Return copy (maintain immutability)
}

func (r HTTPRequest) Body() string {
	return r.body                                // Return body
}

func (r HTTPRequest) Timeout() int {
	return r.timeout                             // Return timeout
}`,
	testCode: `package structinit

import (
	"reflect"
	"testing"
)

func Test1(t *testing.T) {
	// Basic request build
	req, err := NewRequestBuilder().
		Method("GET").
		URL("https://example.com").
		Build()
	if err != nil || req.Method() != "GET" || req.URL() != "https://example.com" {
		t.Errorf("expected GET https://example.com, got err=%v, method=%s, url=%s", err, req.Method(), req.URL())
	}
}

func Test2(t *testing.T) {
	// Default timeout
	req, _ := NewRequestBuilder().
		Method("GET").
		URL("https://example.com").
		Build()
	if req.Timeout() != 30 {
		t.Errorf("expected default timeout 30, got %d", req.Timeout())
	}
}

func Test3(t *testing.T) {
	// Custom timeout
	req, _ := NewRequestBuilder().
		Method("POST").
		URL("https://api.com").
		Timeout(60).
		Build()
	if req.Timeout() != 60 {
		t.Errorf("expected timeout 60, got %d", req.Timeout())
	}
}

func Test4(t *testing.T) {
	// Headers
	req, _ := NewRequestBuilder().
		Method("POST").
		URL("https://api.com").
		Header("Content-Type", "application/json").
		Header("Authorization", "Bearer token").
		Build()
	headers := req.Headers()
	if headers["Content-Type"] != "application/json" || headers["Authorization"] != "Bearer token" {
		t.Errorf("expected headers, got %v", headers)
	}
}

func Test5(t *testing.T) {
	// Body
	req, _ := NewRequestBuilder().
		Method("POST").
		URL("https://api.com").
		Body(\`{"name":"test"}\`).
		Build()
	if req.Body() != \`{"name":"test"}\` {
		t.Errorf("expected body, got %s", req.Body())
	}
}

func Test6(t *testing.T) {
	// Missing method
	_, err := NewRequestBuilder().
		URL("https://example.com").
		Build()
	if err == nil {
		t.Error("expected error for missing method")
	}
}

func Test7(t *testing.T) {
	// Missing URL
	_, err := NewRequestBuilder().
		Method("GET").
		Build()
	if err == nil {
		t.Error("expected error for missing URL")
	}
}

func Test8(t *testing.T) {
	// Invalid timeout
	_, err := NewRequestBuilder().
		Method("GET").
		URL("https://example.com").
		Timeout(0).
		Build()
	if err == nil {
		t.Error("expected error for zero timeout")
	}
}

func Test9(t *testing.T) {
	// Headers returns copy
	req, _ := NewRequestBuilder().
		Method("GET").
		URL("https://example.com").
		Header("X-Test", "value").
		Build()
	h1 := req.Headers()
	h2 := req.Headers()
	h1["X-Test"] = "modified"
	if reflect.DeepEqual(h1, h2) {
		t.Error("expected headers to return copy")
	}
}

func Test10(t *testing.T) {
	// Full request with chaining
	req, err := NewRequestBuilder().
		Method("PUT").
		URL("https://api.com/users/1").
		Header("Content-Type", "application/json").
		Body(\`{"name":"updated"}\`).
		Timeout(45).
		Build()
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if req.Method() != "PUT" || req.Timeout() != 45 {
		t.Errorf("expected PUT with timeout 45, got %s %d", req.Method(), req.Timeout())
	}
}`,
	hint1: `Start by defining both structs with the same fields - HTTPRequest for the final immutable object and RequestBuilder for construction. In NewRequestBuilder, initialize the headers map and set timeout to 30.`,
	hint2: `Each builder method (Method, URL, Header, Body, Timeout) should modify the builder's field and return the builder pointer (*RequestBuilder) to enable chaining. In Build(), validate required fields and create the final HTTPRequest. Remember to return a copy of headers in the Headers() method.`,
	whyItMatters: `The Builder Pattern is essential for creating complex objects with many optional parameters while maintaining clean, readable, and maintainable code.

**Why Builder Pattern:**
- **Fluent API:** Method chaining creates readable, self-documenting code
- **Immutability:** Final object is immutable and thread-safe
- **Flexibility:** Easy to add new optional parameters without breaking existing code
- **Validation:** Centralized validation logic in one place (Build method)
- **Defaults:** Natural place to specify default values
- **No Telescoping:** Avoids telescoping constructor anti-pattern

**Production Examples:**

\`\`\`go
// Example 1: Database Query Builder
type Query struct {
	table      string
	columns    []string
	conditions []string
	orderBy    string
	limit      int
}

type QueryBuilder struct {
	table      string
	columns    []string
	conditions []string
	orderBy    string
	limit      int
}

func NewQueryBuilder() *QueryBuilder {
	return &QueryBuilder{
		columns: []string{"*"},  // Default: select all
		limit:   100,            // Default limit
	}
}

func (b *QueryBuilder) From(table string) *QueryBuilder {
	b.table = table
	return b
}

func (b *QueryBuilder) Select(columns ...string) *QueryBuilder {
	b.columns = columns
	return b
}

func (b *QueryBuilder) Where(condition string) *QueryBuilder {
	b.conditions = append(b.conditions, condition)
	return b
}

func (b *QueryBuilder) OrderBy(column string) *QueryBuilder {
	b.orderBy = column
	return b
}

func (b *QueryBuilder) Limit(n int) *QueryBuilder {
	b.limit = n
	return b
}

func (b *QueryBuilder) Build() (Query, error) {
	if b.table == "" {
		return Query{}, fmt.Errorf("table is required")
	}
	return Query{
		table:      b.table,
		columns:    b.columns,
		conditions: b.conditions,
		orderBy:    b.orderBy,
		limit:      b.limit,
	}, nil
}

// Usage
query, err := NewQueryBuilder().
	From("users").
	Select("id", "name", "email").
	Where("age > 18").
	Where("active = true").
	OrderBy("created_at DESC").
	Limit(50).
	Build()
\`\`\`

\`\`\`go
// Example 2: Email Builder
type Email struct {
	from        string
	to          []string
	cc          []string
	subject     string
	body        string
	attachments []string
	priority    int
}

type EmailBuilder struct {
	from        string
	to          []string
	cc          []string
	subject     string
	body        string
	attachments []string
	priority    int
}

func NewEmailBuilder() *EmailBuilder {
	return &EmailBuilder{
		to:          []string{},
		cc:          []string{},
		attachments: []string{},
		priority:    3,  // Normal priority
	}
}

func (b *EmailBuilder) From(email string) *EmailBuilder {
	b.from = email
	return b
}

func (b *EmailBuilder) To(emails ...string) *EmailBuilder {
	b.to = append(b.to, emails...)
	return b
}

func (b *EmailBuilder) CC(emails ...string) *EmailBuilder {
	b.cc = append(b.cc, emails...)
	return b
}

func (b *EmailBuilder) Subject(subject string) *EmailBuilder {
	b.subject = subject
	return b
}

func (b *EmailBuilder) Body(body string) *EmailBuilder {
	b.body = body
	return b
}

func (b *EmailBuilder) Attach(filepath string) *EmailBuilder {
	b.attachments = append(b.attachments, filepath)
	return b
}

func (b *EmailBuilder) Priority(level int) *EmailBuilder {
	b.priority = level
	return b
}

func (b *EmailBuilder) Build() (Email, error) {
	if b.from == "" {
		return Email{}, fmt.Errorf("from address is required")
	}
	if len(b.to) == 0 {
		return Email{}, fmt.Errorf("at least one recipient is required")
	}
	return Email{
		from:        b.from,
		to:          b.to,
		cc:          b.cc,
		subject:     b.subject,
		body:        b.body,
		attachments: b.attachments,
		priority:    b.priority,
	}, nil
}

// Usage
email, err := NewEmailBuilder().
	From("sender@example.com").
	To("recipient1@example.com", "recipient2@example.com").
	CC("boss@example.com").
	Subject("Monthly Report").
	Body("Please find the report attached.").
	Attach("report.pdf").
	Priority(1).  // High priority
	Build()
\`\`\`

\`\`\`go
// Example 3: Server Configuration Builder
type ServerConfig struct {
	host         string
	port         int
	readTimeout  time.Duration
	writeTimeout time.Duration
	maxConns     int
	enableTLS    bool
	certFile     string
	keyFile      string
}

type ServerConfigBuilder struct {
	host         string
	port         int
	readTimeout  time.Duration
	writeTimeout time.Duration
	maxConns     int
	enableTLS    bool
	certFile     string
	keyFile      string
}

func NewServerConfigBuilder() *ServerConfigBuilder {
	return &ServerConfigBuilder{
		host:         "localhost",
		port:         8080,
		readTimeout:  30 * time.Second,
		writeTimeout: 30 * time.Second,
		maxConns:     1000,
	}
}

func (b *ServerConfigBuilder) Host(host string) *ServerConfigBuilder {
	b.host = host
	return b
}

func (b *ServerConfigBuilder) Port(port int) *ServerConfigBuilder {
	b.port = port
	return b
}

func (b *ServerConfigBuilder) ReadTimeout(d time.Duration) *ServerConfigBuilder {
	b.readTimeout = d
	return b
}

func (b *ServerConfigBuilder) WriteTimeout(d time.Duration) *ServerConfigBuilder {
	b.writeTimeout = d
	return b
}

func (b *ServerConfigBuilder) MaxConnections(n int) *ServerConfigBuilder {
	b.maxConns = n
	return b
}

func (b *ServerConfigBuilder) WithTLS(certFile, keyFile string) *ServerConfigBuilder {
	b.enableTLS = true
	b.certFile = certFile
	b.keyFile = keyFile
	return b
}

func (b *ServerConfigBuilder) Build() (ServerConfig, error) {
	if b.port < 1 || b.port > 65535 {
		return ServerConfig{}, fmt.Errorf("invalid port: %d", b.port)
	}
	if b.enableTLS && (b.certFile == "" || b.keyFile == "") {
		return ServerConfig{}, fmt.Errorf("TLS enabled but cert/key files not specified")
	}
	return ServerConfig{
		host:         b.host,
		port:         b.port,
		readTimeout:  b.readTimeout,
		writeTimeout: b.writeTimeout,
		maxConns:     b.maxConns,
		enableTLS:    b.enableTLS,
		certFile:     b.certFile,
		keyFile:      b.keyFile,
	}, nil
}

// Usage
config, err := NewServerConfigBuilder().
	Host("0.0.0.0").
	Port(443).
	ReadTimeout(60 * time.Second).
	MaxConnections(5000).
	WithTLS("cert.pem", "key.pem").
	Build()
\`\`\`

**Comparison: Builder vs Other Patterns**

Without Builder (Telescoping Constructor - problematic):
\`\`\`go
// Multiple constructors needed, hard to maintain
NewRequest(method, url string)
NewRequestWithTimeout(method, url string, timeout int)
NewRequestWithHeaders(method, url string, headers map[string]string)
NewRequestFull(method, url string, headers map[string]string, body string, timeout int)
\`\`\`

With Builder (clean and scalable):
\`\`\`go
req, err := NewRequestBuilder().
	Method("POST").
	URL("https://api.example.com").
	Header("Content-Type", "application/json").
	Body(jsonData).
	Timeout(60).
	Build()
\`\`\`

**Real-World Benefits:**
- **gRPC:** Request and response builders for complex messages
- **Kubernetes:** Client builders for configuring API clients
- **AWS SDK:** Builders for creating complex service requests
- **HTTP Libraries:** Request builders (e.g., Go's httptest package)

**Key Design Principles:**
- Builder is mutable, built object is immutable
- Each builder method returns the builder for chaining
- Validation happens in Build() method, not in setters
- Provide sensible defaults in the factory function
- Use pointer receiver for builder methods (*RequestBuilder)
- Final object should have only getters, no setters

The Builder Pattern transforms complex object creation from error-prone and verbose to elegant and maintainable. It's especially valuable when objects have many optional parameters or when construction involves validation and defaults.`,
	order: 3,
	translations: {
		ru: {
			title: 'Паттерн строитель',
			description: `Реализуйте **Паттерн Строитель** для построения сложных объектов HTTP-запросов с гибким цепочечным API.

**Обзор паттерна:**
Паттерн Строитель разделяет конструирование сложного объекта от его представления. Он особенно полезен, когда объекту нужны многие опциональные параметры или когда конструирование требует нескольких шагов. Паттерн предоставляет гибкий интерфейс, где вызовы методов могут быть связаны цепочкой, делая код более читаемым и поддерживаемым.

**Ключевые компоненты:**

1. **HTTPRequest** - финальный неизменяемый объект с приватными полями:
   1.1. method, url, headers, body, timeout

2. **RequestBuilder** - изменяемый строитель для конструирования запросов
   2.1. Те же поля что и HTTPRequest, но изменяемые

3. **NewRequestBuilder** - фабрика для создания строителя:
   3.1. Инициализирует карту заголовков
   3.2. Устанавливает таймаут по умолчанию 30 секунд

4. **Методы строителя** - цепочечные методы конфигурации:
   4.1. Method(method string) - устанавливает HTTP метод
   4.2. URL(url string) - устанавливает URL
   4.3. Header(key, value string) - добавляет заголовок
   4.4. Body(body string) - устанавливает тело запроса
   4.5. Timeout(seconds int) - устанавливает таймаут
   4.6. Все возвращают *RequestBuilder для цепочки

5. **Build()** - финализирует и валидирует:
   5.1. Проверяет что method не пуст
   5.2. Проверяет что URL не пуст
   5.3. Проверяет что timeout положительный
   5.4. Создаёт неизменяемый HTTPRequest
   5.5. Возвращает ошибку при неудаче валидации

6. **Методы HTTPRequest** - геттеры только для чтения

**Пример использования:**
\`\`\`go
req, err := NewRequestBuilder().
    Method("POST").
    URL("https://api.example.com/users").
    Header("Content-Type", "application/json").
    Body(\`{"name": "John Doe"}\`).
    Timeout(60).
    Build()

fmt.Println(req.Method())   // "POST"
fmt.Println(req.Timeout())  // 60
\`\`\`

**Почему этот паттерн:**
- **Читаемость:** Гибкий интерфейс самодокументируется
- **Гибкость:** Легко добавлять опциональные параметры
- **Неизменяемость:** Финальный объект нельзя изменить после создания
- **Валидация:** Централизованная валидация в методе Build()
- **Умолчания:** Легко предоставить разумные значения по умолчанию`,
			hint1: `Начните с определения обеих структур с одинаковыми полями - HTTPRequest для финального неизменяемого объекта и RequestBuilder для конструирования. В NewRequestBuilder инициализируйте карту заголовков и установите таймаут в 30.`,
			hint2: `Каждый метод строителя (Method, URL, Header, Body, Timeout) должен модифицировать поле строителя и возвращать указатель на строитель (*RequestBuilder) для включения цепочки. В Build() валидируйте обязательные поля и создайте финальный HTTPRequest. Не забудьте вернуть копию заголовков в методе Headers().`,
			whyItMatters: `Паттерн Строитель необходим для создания сложных объектов с многими опциональными параметрами при сохранении чистого, читаемого и поддерживаемого кода.

**Почему Паттерн Строитель:**
- **Гибкий API:** Цепочка методов создаёт читаемый самодокументируемый код
- **Неизменяемость:** Финальный объект неизменяемый и потокобезопасный
- **Гибкость:** Легко добавлять новые опциональные параметры без нарушения существующего кода
- **Валидация:** Централизованная логика валидации в одном месте (метод Build)
- **Умолчания:** Естественное место для указания значений по умолчанию
- **Без телескопирования:** Избегает анти-паттерна телескопического конструктора

**Production примеры:**

\`\`\`go
// Пример 1: Строитель запросов к БД
type Query struct {
	table      string
	columns    []string
	conditions []string
	orderBy    string
	limit      int
}

type QueryBuilder struct {
	table      string
	columns    []string
	conditions []string
	orderBy    string
	limit      int
}

func NewQueryBuilder() *QueryBuilder {
	return &QueryBuilder{
		columns: []string{"*"},  // По умолчанию: выбрать все
		limit:   100,            // Лимит по умолчанию
	}
}

func (b *QueryBuilder) From(table string) *QueryBuilder {
	b.table = table
	return b
}

func (b *QueryBuilder) Select(columns ...string) *QueryBuilder {
	b.columns = columns
	return b
}

func (b *QueryBuilder) Where(condition string) *QueryBuilder {
	b.conditions = append(b.conditions, condition)
	return b
}

func (b *QueryBuilder) OrderBy(column string) *QueryBuilder {
	b.orderBy = column
	return b
}

func (b *QueryBuilder) Limit(n int) *QueryBuilder {
	b.limit = n
	return b
}

func (b *QueryBuilder) Build() (Query, error) {
	if b.table == "" {
		return Query{}, fmt.Errorf("table is required")
	}
	return Query{
		table:      b.table,
		columns:    b.columns,
		conditions: b.conditions,
		orderBy:    b.orderBy,
		limit:      b.limit,
	}, nil
}

// Использование
query, err := NewQueryBuilder().
	From("users").
	Select("id", "name", "email").
	Where("age > 18").
	Where("active = true").
	OrderBy("created_at DESC").
	Limit(50).
	Build()
\`\`\`

\`\`\`go
// Пример 2: Строитель Email
type Email struct {
	from        string
	to          []string
	cc          []string
	subject     string
	body        string
	attachments []string
	priority    int
}

type EmailBuilder struct {
	from        string
	to          []string
	cc          []string
	subject     string
	body        string
	attachments []string
	priority    int
}

func NewEmailBuilder() *EmailBuilder {
	return &EmailBuilder{
		to:          []string{},
		cc:          []string{},
		attachments: []string{},
		priority:    3,  // Нормальный приоритет
	}
}

func (b *EmailBuilder) From(email string) *EmailBuilder {
	b.from = email
	return b
}

func (b *EmailBuilder) To(emails ...string) *EmailBuilder {
	b.to = append(b.to, emails...)
	return b
}

func (b *EmailBuilder) CC(emails ...string) *EmailBuilder {
	b.cc = append(b.cc, emails...)
	return b
}

func (b *EmailBuilder) Subject(subject string) *EmailBuilder {
	b.subject = subject
	return b
}

func (b *EmailBuilder) Body(body string) *EmailBuilder {
	b.body = body
	return b
}

func (b *EmailBuilder) Attach(filepath string) *EmailBuilder {
	b.attachments = append(b.attachments, filepath)
	return b
}

func (b *EmailBuilder) Priority(level int) *EmailBuilder {
	b.priority = level
	return b
}

func (b *EmailBuilder) Build() (Email, error) {
	if b.from == "" {
		return Email{}, fmt.Errorf("адрес отправителя обязателен")
	}
	if len(b.to) == 0 {
		return Email{}, fmt.Errorf("требуется хотя бы один получатель")
	}
	return Email{
		from:        b.from,
		to:          b.to,
		cc:          b.cc,
		subject:     b.subject,
		body:        b.body,
		attachments: b.attachments,
		priority:    b.priority,
	}, nil
}

// Использование
email, err := NewEmailBuilder().
	From("sender@example.com").
	To("recipient1@example.com", "recipient2@example.com").
	CC("boss@example.com").
	Subject("Месячный отчёт").
	Body("Пожалуйста, смотрите прикреплённый отчёт.").
	Attach("report.pdf").
	Priority(1).  // Высокий приоритет
	Build()
\`\`\`

\`\`\`go
// Пример 3: Строитель конфигурации сервера
type ServerConfig struct {
	host         string
	port         int
	readTimeout  time.Duration
	writeTimeout time.Duration
	maxConns     int
	enableTLS    bool
	certFile     string
	keyFile      string
}

type ServerConfigBuilder struct {
	host         string
	port         int
	readTimeout  time.Duration
	writeTimeout time.Duration
	maxConns     int
	enableTLS    bool
	certFile     string
	keyFile      string
}

func NewServerConfigBuilder() *ServerConfigBuilder {
	return &ServerConfigBuilder{
		host:         "localhost",
		port:         8080,
		readTimeout:  30 * time.Second,
		writeTimeout: 30 * time.Second,
		maxConns:     1000,
	}
}

func (b *ServerConfigBuilder) Host(host string) *ServerConfigBuilder {
	b.host = host
	return b
}

func (b *ServerConfigBuilder) Port(port int) *ServerConfigBuilder {
	b.port = port
	return b
}

func (b *ServerConfigBuilder) ReadTimeout(d time.Duration) *ServerConfigBuilder {
	b.readTimeout = d
	return b
}

func (b *ServerConfigBuilder) WriteTimeout(d time.Duration) *ServerConfigBuilder {
	b.writeTimeout = d
	return b
}

func (b *ServerConfigBuilder) MaxConnections(n int) *ServerConfigBuilder {
	b.maxConns = n
	return b
}

func (b *ServerConfigBuilder) WithTLS(certFile, keyFile string) *ServerConfigBuilder {
	b.enableTLS = true
	b.certFile = certFile
	b.keyFile = keyFile
	return b
}

func (b *ServerConfigBuilder) Build() (ServerConfig, error) {
	if b.port < 1 || b.port > 65535 {
		return ServerConfig{}, fmt.Errorf("неверный порт: %d", b.port)
	}
	if b.enableTLS && (b.certFile == "" || b.keyFile == "") {
		return ServerConfig{}, fmt.Errorf("TLS включён, но файлы сертификата/ключа не указаны")
	}
	return ServerConfig{
		host:         b.host,
		port:         b.port,
		readTimeout:  b.readTimeout,
		writeTimeout: b.writeTimeout,
		maxConns:     b.maxConns,
		enableTLS:    b.enableTLS,
		certFile:     b.certFile,
		keyFile:      b.keyFile,
	}, nil
}

// Использование
config, err := NewServerConfigBuilder().
	Host("0.0.0.0").
	Port(443).
	ReadTimeout(60 * time.Second).
	MaxConnections(5000).
	WithTLS("cert.pem", "key.pem").
	Build()
\`\`\`

**Сравнение: Строитель против других паттернов**

Без Строителя (телескопический конструктор - проблематично):
\`\`\`go
// Нужны множественные конструкторы, сложно поддерживать
NewRequest(method, url string)
NewRequestWithTimeout(method, url string, timeout int)
NewRequestWithHeaders(method, url string, headers map[string]string)
NewRequestFull(method, url string, headers map[string]string, body string, timeout int)
\`\`\`

Со Строителем (чисто и масштабируемо):
\`\`\`go
req, err := NewRequestBuilder().
	Method("POST").
	URL("https://api.example.com").
	Header("Content-Type", "application/json").
	Body(jsonData).
	Timeout(60).
	Build()
\`\`\`

**Real-World преимущества:**
- **gRPC:** Строители запросов и ответов для сложных сообщений
- **Kubernetes:** Клиентские строители для конфигурации API клиентов
- **AWS SDK:** Строители для создания сложных сервисных запросов
- **HTTP библиотеки:** Строители запросов (например, пакет httptest Go)

**Ключевые принципы дизайна:**
- Строитель изменяемый, построенный объект неизменяемый
- Каждый метод строителя возвращает строитель для цепочки
- Валидация происходит в методе Build(), не в сеттерах
- Предоставляйте разумные умолчания в функции фабрике
- Используйте указатель receiver для методов строителя (*RequestBuilder)
- Финальный объект должен иметь только геттеры, без сеттеров

Паттерн Строитель трансформирует создание сложных объектов из подверженного ошибкам и многословного в элегантное и поддерживаемое. Он особенно ценен когда объекты имеют много опциональных параметров или когда конструирование включает валидацию и умолчания.`,
			solutionCode: `package structinit

import (
	"fmt"
	"strings"
)

type HTTPRequest struct {
	method  string                               // HTTP метод (GET, POST, и т.д.)
	url     string                               // URL запроса
	headers map[string]string                    // HTTP заголовки
	body    string                               // Тело запроса
	timeout int                                  // Таймаут в секундах
}

type RequestBuilder struct {
	method  string                               // Изменяемый HTTP метод
	url     string                               // Изменяемый URL
	headers map[string]string                    // Изменяемые заголовки
	body    string                               // Изменяемое тело
	timeout int                                  // Изменяемый таймаут
}

func NewRequestBuilder() *RequestBuilder {
	return &RequestBuilder{
		headers: make(map[string]string),        // Инициализировать карту заголовков
		timeout: 30,                             // Установить таймаут по умолчанию
	}
}

func (b *RequestBuilder) Method(method string) *RequestBuilder {
	b.method = method                            // Установить метод
	return b                                     // Вернуть self для цепочки
}

func (b *RequestBuilder) URL(url string) *RequestBuilder {
	b.url = url                                  // Установить URL
	return b                                     // Вернуть self для цепочки
}

func (b *RequestBuilder) Header(key, value string) *RequestBuilder {
	b.headers[key] = value                       // Добавить заголовок
	return b                                     // Вернуть self для цепочки
}

func (b *RequestBuilder) Body(body string) *RequestBuilder {
	b.body = body                                // Установить тело
	return b                                     // Вернуть self для цепочки
}

func (b *RequestBuilder) Timeout(seconds int) *RequestBuilder {
	b.timeout = seconds                          // Установить таймаут
	return b                                     // Вернуть self для цепочки
}

func (b *RequestBuilder) Build() (HTTPRequest, error) {
	if strings.TrimSpace(b.method) == "" {       // Проверить метод не пустой
		return HTTPRequest{}, fmt.Errorf("method is required")
	}

	if strings.TrimSpace(b.url) == "" {          // Проверить URL не пустой
		return HTTPRequest{}, fmt.Errorf("url is required")
	}

	if b.timeout <= 0 {                          // Проверить таймаут положительный
		return HTTPRequest{}, fmt.Errorf("timeout must be positive")
	}

	return HTTPRequest{                          // Создать неизменяемый запрос
		method:  b.method,
		url:     b.url,
		headers: b.headers,
		body:    b.body,
		timeout: b.timeout,
	}, nil
}

func (r HTTPRequest) Method() string {
	return r.method                              // Вернуть HTTP метод
}

func (r HTTPRequest) URL() string {
	return r.url                                 // Вернуть URL
}

func (r HTTPRequest) Headers() map[string]string {
	copy := make(map[string]string)              // Создать копию заголовков
	for k, v := range r.headers {
		copy[k] = v                              // Скопировать каждый заголовок
	}
	return copy                                  // Вернуть копию (сохранить неизменяемость)
}

func (r HTTPRequest) Body() string {
	return r.body                                // Вернуть тело
}

func (r HTTPRequest) Timeout() int {
	return r.timeout                             // Вернуть таймаут
}`
		},
		uz: {
			title: 'Builder patterni',
			description: `Suyuq, zanjirli API bilan murakkab HTTP so'rov ob'ektlarini qurish uchun **Quruvchi Paradigmasini** amalga oshiring.

**Paradigma ta'rifi:**
Quruvchi Paradigmasi murakkab ob'ektning qurilishini uning ko'rinishidan ajratadi. Bu ayniqsa ob'ektga ko'plab ixtiyoriy parametrlar kerak bo'lganda yoki qurilish bir nechta qadamlarni talab qilganda foydalidir. Paradigma metod chaqiruvlari zanjirlanishi mumkin bo'lgan suyuq interfeys taqdim etadi, bu esa kodni yanada o'qilishi va saqlanishi oson qiladi.

**Asosiy komponentlar:**

1. **HTTPRequest** - yakuniy o'zgarmas ob'ekt xususiy maydonlar bilan:
   1.1. method, url, headers, body, timeout

2. **RequestBuilder** - so'rovlarni qurish uchun o'zgaruvchan quruvchi
   2.1. HTTPRequest bilan bir xil maydonlar, lekin o'zgaruvchan

3. **NewRequestBuilder** - quruvchi yaratish uchun zavod:
   3.1. Headerlar kartasini initsializatsiya qiladi
   3.2. Standart taymautni 30 soniya o'rnatadi

4. **Quruvchi metodlari** - zanjirli konfiguratsiya metodlari:
   4.1. Method(method string) - HTTP metodini o'rnatadi
   4.2. URL(url string) - URL ni o'rnatadi
   4.3. Header(key, value string) - sarlavha qo'shadi
   4.4. Body(body string) - so'rov tanasini o'rnatadi
   4.5. Timeout(seconds int) - taymautni o'rnatadi
   4.6. Hammasi zanjir uchun *RequestBuilder qaytaradi

5. **Build()** - yakunlaydi va validatsiya qiladi:
   5.1. Method bo'sh emasligini tekshiradi
   5.2. URL bo'sh emasligini tekshiradi
   5.3. Timeout musbat ekanligini tekshiradi
   5.4. O'zgarmas HTTPRequest yaratadi
   5.5. Validatsiya muvaffaqiyatsiz bo'lsa xato qaytaradi

6. **HTTPRequest metodlari** - faqat o'qish uchun getterlar

**Foydalanish misoli:**
\`\`\`go
req, err := NewRequestBuilder().
    Method("POST").
    URL("https://api.example.com/users").
    Header("Content-Type", "application/json").
    Body(\`{"name": "John Doe"}\`).
    Timeout(60).
    Build()

fmt.Println(req.Method())   // "POST"
fmt.Println(req.Timeout())  // 60
\`\`\`

**Nima uchun bu paradigma:**
- **O'qilishi:** Suyuq interfeys o'z-o'zini hujjatlashtiradi
- **Moslashuvchanlik:** Ixtiyoriy parametrlarni qo'shish oson
- **O'zgarmaslik:** Yakuniy ob'ektni yaratilgandan keyin o'zgartirib bo'lmaydi
- **Validatsiya:** Build() metodida markazlashtirilgan validatsiya
- **Standartlar:** Oqilona standart qiymatlarni taqdim etish oson`,
			hint1: `Bir xil maydonlar bilan ikkala strukturani aniqlashdan boshlang - yakuniy o'zgarmas ob'ekt uchun HTTPRequest va qurilish uchun RequestBuilder. NewRequestBuilder da headerlar kartasini initsializatsiya qiling va taymautni 30 ga o'rnating.`,
			hint2: `Har bir quruvchi metodi (Method, URL, Header, Body, Timeout) quruvchi maydonini o'zgartirishi va zanjirni yoqish uchun quruvchi ko'rsatkichini (*RequestBuilder) qaytarishi kerak. Build() da majburiy maydonlarni validatsiya qiling va yakuniy HTTPRequest yarating. Headers() metodida headerlar nusxasini qaytarishni unutmang.`,
			whyItMatters: `Quruvchi Paradigmasi toza, o'qilishi va saqlanishi oson kodini saqlagan holda ko'plab ixtiyoriy parametrlar bilan murakkab ob'ektlarni yaratish uchun zarur.

**Nima uchun Quruvchi Paradigmasi:**
- **Suyuq API:** Metodlar zanjiri o'qilishi oson o'z-o'zini hujjatlashtiruvchi kod yaratadi
- **O'zgarmaslik:** Yakuniy ob'ekt o'zgarmas va thread-safe
- **Moslashuvchanlik:** Mavjud kodni buzmasdan yangi ixtiyoriy parametrlarni qo'shish oson
- **Validatsiya:** Bir joyda markazlashtirilgan validatsiya mantiq (Build metodi)
- **Standartlar:** Standart qiymatlarni ko'rsatish uchun tabiiy joy
- **Teleskoplashsiz:** Teleskopik konstruktor anti-paradigmasidan qochadi

**Production misollari:**

\`\`\`go
// Misol 1: Ma'lumotlar bazasi so'rovlari quruvchisi
type Query struct {
	table      string
	columns    []string
	conditions []string
	orderBy    string
	limit      int
}

type QueryBuilder struct {
	table      string
	columns    []string
	conditions []string
	orderBy    string
	limit      int
}

func NewQueryBuilder() *QueryBuilder {
	return &QueryBuilder{
		columns: []string{"*"},  // Standart: hammasini tanlash
		limit:   100,            // Standart limit
	}
}

func (b *QueryBuilder) From(table string) *QueryBuilder {
	b.table = table
	return b
}

func (b *QueryBuilder) Select(columns ...string) *QueryBuilder {
	b.columns = columns
	return b
}

func (b *QueryBuilder) Where(condition string) *QueryBuilder {
	b.conditions = append(b.conditions, condition)
	return b
}

func (b *QueryBuilder) OrderBy(column string) *QueryBuilder {
	b.orderBy = column
	return b
}

func (b *QueryBuilder) Limit(n int) *QueryBuilder {
	b.limit = n
	return b
}

func (b *QueryBuilder) Build() (Query, error) {
	if b.table == "" {
		return Query{}, fmt.Errorf("table is required")
	}
	return Query{
		table:      b.table,
		columns:    b.columns,
		conditions: b.conditions,
		orderBy:    b.orderBy,
		limit:      b.limit,
	}, nil
}

// Foydalanish
query, err := NewQueryBuilder().
	From("users").
	Select("id", "name", "email").
	Where("age > 18").
	Where("active = true").
	OrderBy("created_at DESC").
	Limit(50).
	Build()
\`\`\`

\`\`\`go
// Misol 2: Email quruvchisi
type Email struct {
	from        string
	to          []string
	cc          []string
	subject     string
	body        string
	attachments []string
	priority    int
}

type EmailBuilder struct {
	from        string
	to          []string
	cc          []string
	subject     string
	body        string
	attachments []string
	priority    int
}

func NewEmailBuilder() *EmailBuilder {
	return &EmailBuilder{
		to:          []string{},
		cc:          []string{},
		attachments: []string{},
		priority:    3,  // Oddiy prioritet
	}
}

func (b *EmailBuilder) From(email string) *EmailBuilder {
	b.from = email
	return b
}

func (b *EmailBuilder) To(emails ...string) *EmailBuilder {
	b.to = append(b.to, emails...)
	return b
}

func (b *EmailBuilder) CC(emails ...string) *EmailBuilder {
	b.cc = append(b.cc, emails...)
	return b
}

func (b *EmailBuilder) Subject(subject string) *EmailBuilder {
	b.subject = subject
	return b
}

func (b *EmailBuilder) Body(body string) *EmailBuilder {
	b.body = body
	return b
}

func (b *EmailBuilder) Attach(filepath string) *EmailBuilder {
	b.attachments = append(b.attachments, filepath)
	return b
}

func (b *EmailBuilder) Priority(level int) *EmailBuilder {
	b.priority = level
	return b
}

func (b *EmailBuilder) Build() (Email, error) {
	if b.from == "" {
		return Email{}, fmt.Errorf("jo'natuvchi manzili talab qilinadi")
	}
	if len(b.to) == 0 {
		return Email{}, fmt.Errorf("kamida bitta qabul qiluvchi talab qilinadi")
	}
	return Email{
		from:        b.from,
		to:          b.to,
		cc:          b.cc,
		subject:     b.subject,
		body:        b.body,
		attachments: b.attachments,
		priority:    b.priority,
	}, nil
}

// Foydalanish
email, err := NewEmailBuilder().
	From("sender@example.com").
	To("recipient1@example.com", "recipient2@example.com").
	CC("boss@example.com").
	Subject("Oylik hisobot").
	Body("Iltimos, biriktirilgan hisobotni ko'ring.").
	Attach("report.pdf").
	Priority(1).  // Yuqori prioritet
	Build()
\`\`\`

\`\`\`go
// Misol 3: Server konfiguratsiya quruvchisi
type ServerConfig struct {
	host         string
	port         int
	readTimeout  time.Duration
	writeTimeout time.Duration
	maxConns     int
	enableTLS    bool
	certFile     string
	keyFile      string
}

type ServerConfigBuilder struct {
	host         string
	port         int
	readTimeout  time.Duration
	writeTimeout time.Duration
	maxConns     int
	enableTLS    bool
	certFile     string
	keyFile      string
}

func NewServerConfigBuilder() *ServerConfigBuilder {
	return &ServerConfigBuilder{
		host:         "localhost",
		port:         8080,
		readTimeout:  30 * time.Second,
		writeTimeout: 30 * time.Second,
		maxConns:     1000,
	}
}

func (b *ServerConfigBuilder) Host(host string) *ServerConfigBuilder {
	b.host = host
	return b
}

func (b *ServerConfigBuilder) Port(port int) *ServerConfigBuilder {
	b.port = port
	return b
}

func (b *ServerConfigBuilder) ReadTimeout(d time.Duration) *ServerConfigBuilder {
	b.readTimeout = d
	return b
}

func (b *ServerConfigBuilder) WriteTimeout(d time.Duration) *ServerConfigBuilder {
	b.writeTimeout = d
	return b
}

func (b *ServerConfigBuilder) MaxConnections(n int) *ServerConfigBuilder {
	b.maxConns = n
	return b
}

func (b *ServerConfigBuilder) WithTLS(certFile, keyFile string) *ServerConfigBuilder {
	b.enableTLS = true
	b.certFile = certFile
	b.keyFile = keyFile
	return b
}

func (b *ServerConfigBuilder) Build() (ServerConfig, error) {
	if b.port < 1 || b.port > 65535 {
		return ServerConfig{}, fmt.Errorf("noto'g'ri port: %d", b.port)
	}
	if b.enableTLS && (b.certFile == "" || b.keyFile == "") {
		return ServerConfig{}, fmt.Errorf("TLS yoqilgan, lekin sertifikat/kalit fayllari ko'rsatilmagan")
	}
	return ServerConfig{
		host:         b.host,
		port:         b.port,
		readTimeout:  b.readTimeout,
		writeTimeout: b.writeTimeout,
		maxConns:     b.maxConns,
		enableTLS:    b.enableTLS,
		certFile:     b.certFile,
		keyFile:      b.keyFile,
	}, nil
}

// Foydalanish
config, err := NewServerConfigBuilder().
	Host("0.0.0.0").
	Port(443).
	ReadTimeout(60 * time.Second).
	MaxConnections(5000).
	WithTLS("cert.pem", "key.pem").
	Build()
\`\`\`

**Taqqoslash: Quruvchi vs boshqa paradigmalar**

Quruvchisiz (teleskopik konstruktor - muammoli):
\`\`\`go
// Ko'plab konstruktorlar kerak, saqlanishi qiyin
NewRequest(method, url string)
NewRequestWithTimeout(method, url string, timeout int)
NewRequestWithHeaders(method, url string, headers map[string]string)
NewRequestFull(method, url string, headers map[string]string, body string, timeout int)
\`\`\`

Quruvchi bilan (toza va kengaytiriladigan):
\`\`\`go
req, err := NewRequestBuilder().
	Method("POST").
	URL("https://api.example.com").
	Header("Content-Type", "application/json").
	Body(jsonData).
	Timeout(60).
	Build()
\`\`\`

**Haqiqiy dunyo foydalari:**
- **gRPC:** Murakkab xabarlar uchun so'rov va javob quruvchilari
- **Kubernetes:** API kliyentlarini konfiguratsiya qilish uchun kliyent quruvchilari
- **AWS SDK:** Murakkab xizmat so'rovlarini yaratish uchun quruvchilar
- **HTTP kutubxonalar:** So'rov quruvchilari (masalan, Go ning httptest paketi)

**Asosiy dizayn prinsiplari:**
- Quruvchi o'zgaruvchan, qurilgan ob'ekt o'zgarmas
- Har bir quruvchi metodi zanjir uchun quruvchini qaytaradi
- Validatsiya Build() metodida sodir bo'ladi, setterlar da emas
- Zavod funksiyasida oqilona standartlarni taqdim eting
- Quruvchi metodlari uchun ko'rsatkich receiverni ishlating (*RequestBuilder)
- Yakuniy ob'ekt faqat getterlar ga ega bo'lishi kerak, setterlar yo'q

Quruvchi Paradigmasi murakkab ob'ektlarni yaratishni xatolarga moyil va ko'p so'zli holatdan elegant va saqlanishi mumkin holatga o'zgartiradi. Bu ayniqsa ob'ektlar ko'plab ixtiyoriy parametrlarga ega bo'lganda yoki qurilish validatsiya va standartlarni o'z ichiga olganda qimmatlidir.`,
			solutionCode: `package structinit

import (
	"fmt"
	"strings"
)

type HTTPRequest struct {
	method  string                               // HTTP metodi (GET, POST, va h.k.)
	url     string                               // So'rov URL manzili
	headers map[string]string                    // HTTP sarlavhalari
	body    string                               // So'rov tanasi
	timeout int                                  // Taymaut soniyalarda
}

type RequestBuilder struct {
	method  string                               // O'zgaruvchan HTTP metodi
	url     string                               // O'zgaruvchan URL
	headers map[string]string                    // O'zgaruvchan sarlavhalar
	body    string                               // O'zgaruvchan tana
	timeout int                                  // O'zgaruvchan taymaut
}

func NewRequestBuilder() *RequestBuilder {
	return &RequestBuilder{
		headers: make(map[string]string),        // Sarlavhalar kartasini initsializatsiya qilish
		timeout: 30,                             // Standart taymautni o'rnatish
	}
}

func (b *RequestBuilder) Method(method string) *RequestBuilder {
	b.method = method                            // Metodini o'rnatish
	return b                                     // Zanjir uchun o'zini qaytarish
}

func (b *RequestBuilder) URL(url string) *RequestBuilder {
	b.url = url                                  // URL ni o'rnatish
	return b                                     // Zanjir uchun o'zini qaytarish
}

func (b *RequestBuilder) Header(key, value string) *RequestBuilder {
	b.headers[key] = value                       // Sarlavha qo'shish
	return b                                     // Zanjir uchun o'zini qaytarish
}

func (b *RequestBuilder) Body(body string) *RequestBuilder {
	b.body = body                                // Tanani o'rnatish
	return b                                     // Zanjir uchun o'zini qaytarish
}

func (b *RequestBuilder) Timeout(seconds int) *RequestBuilder {
	b.timeout = seconds                          // Taymautni o'rnatish
	return b                                     // Zanjir uchun o'zini qaytarish
}

func (b *RequestBuilder) Build() (HTTPRequest, error) {
	if strings.TrimSpace(b.method) == "" {       // Metod bo'sh emasligini tekshirish
		return HTTPRequest{}, fmt.Errorf("method is required")
	}

	if strings.TrimSpace(b.url) == "" {          // URL bo'sh emasligini tekshirish
		return HTTPRequest{}, fmt.Errorf("url is required")
	}

	if b.timeout <= 0 {                          // Taymaut musbat ekanligini tekshirish
		return HTTPRequest{}, fmt.Errorf("timeout must be positive")
	}

	return HTTPRequest{                          // O'zgarmas so'rovni yaratish
		method:  b.method,
		url:     b.url,
		headers: b.headers,
		body:    b.body,
		timeout: b.timeout,
	}, nil
}

func (r HTTPRequest) Method() string {
	return r.method                              // HTTP metodini qaytarish
}

func (r HTTPRequest) URL() string {
	return r.url                                 // URL ni qaytarish
}

func (r HTTPRequest) Headers() map[string]string {
	copy := make(map[string]string)              // Sarlavhalar nusxasini yaratish
	for k, v := range r.headers {
		copy[k] = v                              // Har bir sarlavhani nusxalash
	}
	return copy                                  // Nusxani qaytarish (o'zgarmaslikni saqlash)
}

func (r HTTPRequest) Body() string {
	return r.body                                // Tanani qaytarish
}

func (r HTTPRequest) Timeout() int {
	return r.timeout                             // Taymautni qaytarish
}`
		}
	}
};

export default task;
