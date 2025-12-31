import { Task } from '../../../../../../../types';

export const task: Task = {
	slug: 'go-dp-proxy',
	title: 'Proxy Pattern',
	difficulty: 'medium',
	tags: ['go', 'design-patterns', 'structural', 'proxy'],
	estimatedTime: '30m',
	isPremium: false,
	youtubeUrl: '',
	description: `Implement the Proxy pattern in Go - provide a surrogate or placeholder for another object to control access to it.

The Proxy pattern creates a substitute object that controls access to the original object. This allows you to perform operations before or after the request reaches the original object - like access control, caching, lazy loading, or logging.

**You will implement:**

1. **Server interface** - Common interface for RealServer and ProxyServer
2. **RealServer struct** - The actual server that handles requests
3. **ProxyServer struct** - Controls access with caching and access control

**Example Usage:**

\`\`\`go
proxy := NewProxyServer()	// create proxy with real server inside
proxy.AddBlocked("/admin")	// block sensitive URL

result1 := proxy.HandleRequest("/home")	// forwards to real server
// Output: "RealServer: Handling request for /home"

result2 := proxy.HandleRequest("/home")	// returns cached response
// Output: "Cached: RealServer: Handling request for /home"

result3 := proxy.HandleRequest("/admin")	// blocked URL
// Output: "Access denied: /admin"

fmt.Println(proxy.GetCacheSize())	// prints: 1
\`\`\``,
	initialCode: `package patterns

import "fmt"

type Server interface {
}

type RealServer struct{}

func (s *RealServer) HandleRequest(url string) string {
}

type ProxyServer struct {
	realServer *RealServer
	cache      map[string]string
	blocked    map[string]bool
}

func NewProxyServer() *ProxyServer {
	}
}

func (p *ProxyServer) AddBlocked(url string) {
}

func (p *ProxyServer) HandleRequest(url string) string {
}

func (p *ProxyServer) GetCacheSize() int {
}`,
	solutionCode: `package patterns

import "fmt"	// import for string formatting

// Server interface for handling requests
type Server interface {	// common interface for real and proxy
	HandleRequest(url string) string	// method signature for handling requests
}

// RealServer is the actual server
type RealServer struct{}	// the real subject that does actual work

// HandleRequest processes the request on the real server
func (s *RealServer) HandleRequest(url string) string {	// implements Server interface
	return fmt.Sprintf("RealServer: Handling request for %s", url)	// actual request processing
}

// ProxyServer controls access to RealServer
type ProxyServer struct {	// proxy that wraps real server
	realServer *RealServer	// reference to real server (lazy or eager)
	cache      map[string]string	// caching proxy: stores responses
	blocked    map[string]bool	// protection proxy: access control list
}

// NewProxyServer creates a new proxy with initialized maps
func NewProxyServer() *ProxyServer {	// factory function for proxy
	return &ProxyServer{	// create and initialize proxy
		realServer: &RealServer{},	// create real server (could be lazy)
		cache:      make(map[string]string),	// initialize cache map
		blocked:    make(map[string]bool),	// initialize blocked list
	}
}

// AddBlocked adds URL to blocked list for access control
func (p *ProxyServer) AddBlocked(url string) {	// protection proxy functionality
	p.blocked[url] = true	// mark URL as blocked
}

// HandleRequest controls access and caches responses
func (p *ProxyServer) HandleRequest(url string) string {	// implements Server interface
	// Step 1: Protection proxy - check access control
	if p.blocked[url] {	// check if URL is in blocked list
		return fmt.Sprintf("Access denied: %s", url)	// deny access to blocked URLs
	}

	// Step 2: Caching proxy - check cache first
	if cached, exists := p.cache[url]; exists {	// look up in cache
		return fmt.Sprintf("Cached: %s", cached)	// return cached response
	}

	// Step 3: Forward to real server and cache result
	response := p.realServer.HandleRequest(url)	// delegate to real server
	p.cache[url] = response	// store response in cache for future
	return response	// return fresh response
}

// GetCacheSize returns number of cached responses
func (p *ProxyServer) GetCacheSize() int {	// utility method for monitoring
	return len(p.cache)	// return cache entry count
}`,
	hint1: `RealServer.HandleRequest is straightforward - use fmt.Sprintf to format the URL into the response string. AddBlocked simply sets blocked[url] = true to mark the URL as inaccessible.`,
	hint2: `HandleRequest in ProxyServer checks three things in order: (1) If URL is blocked, return "Access denied: {url}". (2) If URL is in cache, return "Cached: {cached_response}". (3) Otherwise, call realServer.HandleRequest, store result in cache, and return it.`,
	whyItMatters: `**Why the Proxy Pattern Exists**

Without Proxy, you might add access control and caching directly to your server:

\`\`\`go
// Problem: Server has too many responsibilities
type Server struct {
    cache   map[string]string
    blocked map[string]bool
}

func (s *Server) Handle(url string) string {
    if s.blocked[url] { return "denied" }  // access control mixed in
    if c, ok := s.cache[url]; ok { return c }  // caching mixed in
    result := s.actualHandle(url)  // actual logic
    s.cache[url] = result
    return result
}
// Server is now doing 3 jobs: actual handling, caching, access control
\`\`\`

With Proxy, each concern is separate:

\`\`\`go
// Solution: Proxy handles cross-cutting concerns
type RealServer struct{}  // only does actual work

type ProxyServer struct {
    real    *RealServer
    cache   map[string]string
    blocked map[string]bool
}
// Proxy handles caching and access control
// RealServer stays focused on its core responsibility
\`\`\`

**Real-World Proxy Examples in Go**

1. **Virtual Proxy - Lazy Loading Database Connection**:
\`\`\`go
type Database interface {
    Query(sql string) []Row
}

type RealDatabase struct {
    connection *sql.DB  // expensive to create
}

type LazyDatabaseProxy struct {
    realDB *RealDatabase  // nil until first use
    dsn    string
    mu     sync.Mutex
}

func (p *LazyDatabaseProxy) Query(sql string) []Row {
    p.mu.Lock()
    defer p.mu.Unlock()
    if p.realDB == nil {  // lazy initialization
        p.realDB = &RealDatabase{connection: connect(p.dsn)}
    }
    return p.realDB.Query(sql)
}
\`\`\`

2. **Protection Proxy - Role-Based Access Control**:
\`\`\`go
type Document interface {
    Read() string
    Write(content string)
}

type SecureDocumentProxy struct {
    doc  *RealDocument
    user *User
}

func (p *SecureDocumentProxy) Write(content string) {
    if !p.user.HasPermission("write") {
        log.Printf("Access denied for user %s", p.user.ID)
        return
    }
    p.doc.Write(content)
}
\`\`\`

**Production Pattern: HTTP Client with Retry and Circuit Breaker**

\`\`\`go
package main

import (
    "fmt"
    "net/http"
    "time"
    "sync"
)

// HTTPClient interface
type HTTPClient interface {
    Do(req *http.Request) (*http.Response, error)
}

// RealHTTPClient wraps standard http.Client
type RealHTTPClient struct {
    client *http.Client
}

func (c *RealHTTPClient) Do(req *http.Request) (*http.Response, error) {
    return c.client.Do(req)
}

// CircuitState represents circuit breaker states
type CircuitState int

const (
    StateClosed CircuitState = iota  // normal operation
    StateOpen                        // failing, reject requests
    StateHalfOpen                    // testing if service recovered
)

// ResilientClientProxy adds retry and circuit breaker
type ResilientClientProxy struct {
    client       HTTPClient
    maxRetries   int
    retryDelay   time.Duration
    state        CircuitState
    failures     int
    threshold    int
    lastFailure  time.Time
    resetTimeout time.Duration
    mu           sync.RWMutex
}

func NewResilientClient(client HTTPClient) *ResilientClientProxy {
    return &ResilientClientProxy{
        client:       client,
        maxRetries:   3,
        retryDelay:   time.Second,
        state:        StateClosed,
        threshold:    5,
        resetTimeout: 30 * time.Second,
    }
}

func (p *ResilientClientProxy) Do(req *http.Request) (*http.Response, error) {
    // Check circuit breaker state
    if !p.allowRequest() {
        return nil, fmt.Errorf("circuit breaker open")
    }

    // Retry logic
    var lastErr error
    for attempt := 0; attempt <= p.maxRetries; attempt++ {
        resp, err := p.client.Do(req)
        if err == nil && resp.StatusCode < 500 {
            p.recordSuccess()
            return resp, nil
        }
        lastErr = err
        if attempt < p.maxRetries {
            time.Sleep(p.retryDelay * time.Duration(attempt+1))
        }
    }

    p.recordFailure()
    return nil, lastErr
}

func (p *ResilientClientProxy) allowRequest() bool {
    p.mu.RLock()
    defer p.mu.RUnlock()

    switch p.state {
    case StateClosed:
        return true
    case StateOpen:
        if time.Since(p.lastFailure) > p.resetTimeout {
            p.mu.RUnlock()
            p.mu.Lock()
            p.state = StateHalfOpen
            p.mu.Unlock()
            p.mu.RLock()
            return true
        }
        return false
    case StateHalfOpen:
        return true
    }
    return false
}

func (p *ResilientClientProxy) recordSuccess() {
    p.mu.Lock()
    defer p.mu.Unlock()
    p.failures = 0
    p.state = StateClosed
}

func (p *ResilientClientProxy) recordFailure() {
    p.mu.Lock()
    defer p.mu.Unlock()
    p.failures++
    p.lastFailure = time.Now()
    if p.failures >= p.threshold {
        p.state = StateOpen
    }
}
\`\`\`

**Common Mistakes to Avoid**

1. **Not implementing the same interface** - Proxy must be interchangeable with real subject
2. **Forgetting thread safety** - Caching proxies need synchronization in concurrent environments
3. **Creating proxy without real subject** - Lazy proxies must handle initialization properly
4. **Proxy doing too much** - Each proxy should have one responsibility (caching OR access control, not both)
5. **Not considering proxy chains** - Multiple proxies can wrap each other for layered functionality`,
	order: 6,
	testCode: `package patterns

import (
	"testing"
)

// Test1: RealServer.HandleRequest returns formatted string
func Test1(t *testing.T) {
	s := &RealServer{}
	result := s.HandleRequest("/home")
	if result != "RealServer: Handling request for /home" {
		t.Errorf("Unexpected result: %s", result)
	}
}

// Test2: NewProxyServer returns non-nil
func Test2(t *testing.T) {
	p := NewProxyServer()
	if p == nil {
		t.Error("NewProxyServer should return non-nil")
	}
}

// Test3: ProxyServer forwards to RealServer
func Test3(t *testing.T) {
	p := NewProxyServer()
	result := p.HandleRequest("/page")
	if result != "RealServer: Handling request for /page" {
		t.Errorf("Should forward to RealServer: %s", result)
	}
}

// Test4: ProxyServer caches responses
func Test4(t *testing.T) {
	p := NewProxyServer()
	p.HandleRequest("/cached")
	result := p.HandleRequest("/cached")
	if result != "Cached: RealServer: Handling request for /cached" {
		t.Errorf("Should return cached response: %s", result)
	}
}

// Test5: ProxyServer blocks URLs
func Test5(t *testing.T) {
	p := NewProxyServer()
	p.AddBlocked("/admin")
	result := p.HandleRequest("/admin")
	if result != "Access denied: /admin" {
		t.Errorf("Should deny access: %s", result)
	}
}

// Test6: GetCacheSize returns correct count
func Test6(t *testing.T) {
	p := NewProxyServer()
	p.HandleRequest("/a")
	p.HandleRequest("/b")
	if p.GetCacheSize() != 2 {
		t.Errorf("Expected cache size 2, got %d", p.GetCacheSize())
	}
}

// Test7: Blocked URL is not cached
func Test7(t *testing.T) {
	p := NewProxyServer()
	p.AddBlocked("/secret")
	p.HandleRequest("/secret")
	if p.GetCacheSize() != 0 {
		t.Error("Blocked URLs should not be cached")
	}
}

// Test8: RealServer implements Server interface
func Test8(t *testing.T) {
	var s Server = &RealServer{}
	if s == nil {
		t.Error("RealServer should implement Server")
	}
}

// Test9: ProxyServer implements Server interface
func Test9(t *testing.T) {
	var s Server = NewProxyServer()
	if s == nil {
		t.Error("ProxyServer should implement Server")
	}
}

// Test10: Multiple AddBlocked calls work
func Test10(t *testing.T) {
	p := NewProxyServer()
	p.AddBlocked("/a")
	p.AddBlocked("/b")
	r1 := p.HandleRequest("/a")
	r2 := p.HandleRequest("/b")
	if r1 != "Access denied: /a" || r2 != "Access denied: /b" {
		t.Error("Multiple blocked URLs should work")
	}
}
`,
	translations: {
		ru: {
			title: 'Паттерн Proxy (Заместитель)',
			description: `Реализуйте паттерн Proxy на Go — предоставьте суррогат или заместитель для другого объекта для контроля доступа к нему.

Паттерн Proxy создаёт объект-заместитель, который контролирует доступ к оригинальному объекту. Это позволяет выполнять операции до или после того, как запрос достигнет оригинального объекта — например, контроль доступа, кэширование, ленивую загрузку или логирование.

**Вы реализуете:**

1. **Интерфейс Server** — Общий интерфейс для RealServer и ProxyServer
2. **Структура RealServer** — Настоящий сервер, обрабатывающий запросы
3. **Структура ProxyServer** — Контролирует доступ с кэшированием и контролем доступа

**Пример использования:**

\`\`\`go
proxy := NewProxyServer()	// создаём прокси с реальным сервером внутри
proxy.AddBlocked("/admin")	// блокируем чувствительный URL

result1 := proxy.HandleRequest("/home")	// перенаправляет на реальный сервер
// Вывод: "RealServer: Handling request for /home"

result2 := proxy.HandleRequest("/home")	// возвращает кэшированный ответ
// Вывод: "Cached: RealServer: Handling request for /home"

result3 := proxy.HandleRequest("/admin")	// заблокированный URL
// Вывод: "Access denied: /admin"

fmt.Println(proxy.GetCacheSize())	// выводит: 1
\`\`\``,
			hint1: `RealServer.HandleRequest прост — используйте fmt.Sprintf для форматирования URL в строку ответа. AddBlocked просто устанавливает blocked[url] = true, чтобы пометить URL как недоступный.`,
			hint2: `HandleRequest в ProxyServer проверяет три вещи по порядку: (1) Если URL заблокирован, вернуть "Access denied: {url}". (2) Если URL в кэше, вернуть "Cached: {cached_response}". (3) Иначе вызвать realServer.HandleRequest, сохранить результат в кэш и вернуть его.`,
			whyItMatters: `**Зачем нужен паттерн Proxy**

Без Proxy вы можете добавить контроль доступа и кэширование прямо в сервер:

\`\`\`go
// Проблема: Сервер имеет слишком много обязанностей
type Server struct {
    cache   map[string]string
    blocked map[string]bool
}

func (s *Server) Handle(url string) string {
    if s.blocked[url] { return "denied" }  // контроль доступа смешан
    if c, ok := s.cache[url]; ok { return c }  // кэширование смешано
    result := s.actualHandle(url)  // реальная логика
    s.cache[url] = result
    return result
}
// Сервер теперь выполняет 3 работы: обработку, кэширование, контроль доступа
\`\`\`

С Proxy каждая ответственность отделена:

\`\`\`go
// Решение: Proxy обрабатывает сквозные задачи
type RealServer struct{}  // только выполняет реальную работу

type ProxyServer struct {
    real    *RealServer
    cache   map[string]string
    blocked map[string]bool
}
// Proxy обрабатывает кэширование и контроль доступа
// RealServer сосредоточен на своей основной ответственности
\`\`\`

**Реальные примеры Proxy в Go**

1. **Virtual Proxy — Ленивая загрузка подключения к БД**:
\`\`\`go
type Database interface {
    Query(sql string) []Row
}

type RealDatabase struct {
    connection *sql.DB  // дорого создавать
}

type LazyDatabaseProxy struct {
    realDB *RealDatabase  // nil до первого использования
    dsn    string
    mu     sync.Mutex
}

func (p *LazyDatabaseProxy) Query(sql string) []Row {
    p.mu.Lock()
    defer p.mu.Unlock()
    if p.realDB == nil {  // ленивая инициализация
        p.realDB = &RealDatabase{connection: connect(p.dsn)}
    }
    return p.realDB.Query(sql)
}
\`\`\`

2. **Protection Proxy — Ролевой контроль доступа**:
\`\`\`go
type Document interface {
    Read() string
    Write(content string)
}

type SecureDocumentProxy struct {
    doc  *RealDocument
    user *User
}

func (p *SecureDocumentProxy) Write(content string) {
    if !p.user.HasPermission("write") {
        log.Printf("Access denied for user %s", p.user.ID)
        return
    }
    p.doc.Write(content)
}
\`\`\`

**Продакшен паттерн: HTTP клиент с повторами и Circuit Breaker**

\`\`\`go
package main

import (
    "fmt"
    "net/http"
    "time"
    "sync"
)

// HTTPClient интерфейс
type HTTPClient interface {
    Do(req *http.Request) (*http.Response, error)
}

// RealHTTPClient оборачивает стандартный http.Client
type RealHTTPClient struct {
    client *http.Client
}

func (c *RealHTTPClient) Do(req *http.Request) (*http.Response, error) {
    return c.client.Do(req)
}

// CircuitState представляет состояния circuit breaker
type CircuitState int

const (
    StateClosed CircuitState = iota  // нормальная работа
    StateOpen                        // сбои, отклоняем запросы
    StateHalfOpen                    // тестируем восстановление сервиса
)

// ResilientClientProxy добавляет повторы и circuit breaker
type ResilientClientProxy struct {
    client       HTTPClient
    maxRetries   int
    retryDelay   time.Duration
    state        CircuitState
    failures     int
    threshold    int
    lastFailure  time.Time
    resetTimeout time.Duration
    mu           sync.RWMutex
}

func NewResilientClient(client HTTPClient) *ResilientClientProxy {
    return &ResilientClientProxy{
        client:       client,
        maxRetries:   3,
        retryDelay:   time.Second,
        state:        StateClosed,
        threshold:    5,
        resetTimeout: 30 * time.Second,
    }
}

func (p *ResilientClientProxy) Do(req *http.Request) (*http.Response, error) {
    // Проверяем состояние circuit breaker
    if !p.allowRequest() {
        return nil, fmt.Errorf("circuit breaker open")
    }

    // Логика повторов
    var lastErr error
    for attempt := 0; attempt <= p.maxRetries; attempt++ {
        resp, err := p.client.Do(req)
        if err == nil && resp.StatusCode < 500 {
            p.recordSuccess()
            return resp, nil
        }
        lastErr = err
        if attempt < p.maxRetries {
            time.Sleep(p.retryDelay * time.Duration(attempt+1))
        }
    }

    p.recordFailure()
    return nil, lastErr
}

func (p *ResilientClientProxy) allowRequest() bool {
    p.mu.RLock()
    defer p.mu.RUnlock()

    switch p.state {
    case StateClosed:
        return true
    case StateOpen:
        if time.Since(p.lastFailure) > p.resetTimeout {
            p.mu.RUnlock()
            p.mu.Lock()
            p.state = StateHalfOpen
            p.mu.Unlock()
            p.mu.RLock()
            return true
        }
        return false
    case StateHalfOpen:
        return true
    }
    return false
}

func (p *ResilientClientProxy) recordSuccess() {
    p.mu.Lock()
    defer p.mu.Unlock()
    p.failures = 0
    p.state = StateClosed
}

func (p *ResilientClientProxy) recordFailure() {
    p.mu.Lock()
    defer p.mu.Unlock()
    p.failures++
    p.lastFailure = time.Now()
    if p.failures >= p.threshold {
        p.state = StateOpen
    }
}
\`\`\`

**Распространённые ошибки**

1. **Не реализуют тот же интерфейс** — Proxy должен быть взаимозаменяем с реальным субъектом
2. **Забывают о потокобезопасности** — Кэширующие прокси нуждаются в синхронизации в конкурентной среде
3. **Создают прокси без реального субъекта** — Ленивые прокси должны правильно обрабатывать инициализацию
4. **Proxy делает слишком много** — Каждый прокси должен иметь одну ответственность (кэширование ИЛИ контроль доступа, не оба)
5. **Не учитывают цепочки прокси** — Несколько прокси могут оборачивать друг друга для многослойной функциональности`,
			solutionCode: `package patterns

import "fmt"	// импорт для форматирования строк

// Server интерфейс для обработки запросов
type Server interface {	// общий интерфейс для реального и прокси
	HandleRequest(url string) string	// сигнатура метода для обработки запросов
}

// RealServer — настоящий сервер
type RealServer struct{}	// реальный субъект, выполняющий настоящую работу

// HandleRequest обрабатывает запрос на реальном сервере
func (s *RealServer) HandleRequest(url string) string {	// реализует интерфейс Server
	return fmt.Sprintf("RealServer: Handling request for %s", url)	// реальная обработка запроса
}

// ProxyServer контролирует доступ к RealServer
type ProxyServer struct {	// прокси, оборачивающий реальный сервер
	realServer *RealServer	// ссылка на реальный сервер (ленивая или немедленная)
	cache      map[string]string	// кэширующий прокси: хранит ответы
	blocked    map[string]bool	// защитный прокси: список контроля доступа
}

// NewProxyServer создаёт новый прокси с инициализированными картами
func NewProxyServer() *ProxyServer {	// фабричная функция для прокси
	return &ProxyServer{	// создаём и инициализируем прокси
		realServer: &RealServer{},	// создаём реальный сервер (может быть ленивым)
		cache:      make(map[string]string),	// инициализируем карту кэша
		blocked:    make(map[string]bool),	// инициализируем список блокировки
	}
}

// AddBlocked добавляет URL в список заблокированных для контроля доступа
func (p *ProxyServer) AddBlocked(url string) {	// функциональность защитного прокси
	p.blocked[url] = true	// помечаем URL как заблокированный
}

// HandleRequest контролирует доступ и кэширует ответы
func (p *ProxyServer) HandleRequest(url string) string {	// реализует интерфейс Server
	// Шаг 1: Защитный прокси — проверяем контроль доступа
	if p.blocked[url] {	// проверяем, есть ли URL в списке заблокированных
		return fmt.Sprintf("Access denied: %s", url)	// запрещаем доступ к заблокированным URL
	}

	// Шаг 2: Кэширующий прокси — сначала проверяем кэш
	if cached, exists := p.cache[url]; exists {	// ищем в кэше
		return fmt.Sprintf("Cached: %s", cached)	// возвращаем кэшированный ответ
	}

	// Шаг 3: Перенаправляем на реальный сервер и кэшируем результат
	response := p.realServer.HandleRequest(url)	// делегируем реальному серверу
	p.cache[url] = response	// сохраняем ответ в кэш для будущего
	return response	// возвращаем свежий ответ
}

// GetCacheSize возвращает количество кэшированных ответов
func (p *ProxyServer) GetCacheSize() int {	// утилитарный метод для мониторинга
	return len(p.cache)	// возвращаем количество записей в кэше
}`
		},
		uz: {
			title: 'Proxy (Proksi) Pattern',
			description: `Go tilida Proxy patternini amalga oshiring — boshqa ob'ektga kirishni nazorat qilish uchun o'rinbosar yoki joy egallagan taqdim eting.

Proxy patterni asl ob'ektga kirishni nazorat qiluvchi o'rinbosar ob'ekt yaratadi. Bu so'rov asl ob'ektga yetib borgunga qadar yoki keyin operatsiyalarni bajarish imkonini beradi — masalan, kirishni nazorat qilish, keshlash, dangasa yuklash yoki loglash.

**Siz amalga oshirasiz:**

1. **Server interfeysi** — RealServer va ProxyServer uchun umumiy interfeys
2. **RealServer struct** — So'rovlarni qayta ishlovchi haqiqiy server
3. **ProxyServer struct** — Keshlash va kirishni nazorat qilish bilan kirishni nazorat qiladi

**Foydalanish namunasi:**

\`\`\`go
proxy := NewProxyServer()	// ichida haqiqiy server bilan proksi yaratamiz
proxy.AddBlocked("/admin")	// sezgir URL ni bloklaymiz

result1 := proxy.HandleRequest("/home")	// haqiqiy serverga yo'naltiradi
// Chiqish: "RealServer: Handling request for /home"

result2 := proxy.HandleRequest("/home")	// keshlangan javobni qaytaradi
// Chiqish: "Cached: RealServer: Handling request for /home"

result3 := proxy.HandleRequest("/admin")	// bloklangan URL
// Chiqish: "Access denied: /admin"

fmt.Println(proxy.GetCacheSize())	// chiqaradi: 1
\`\`\``,
			hint1: `RealServer.HandleRequest oddiy — URL ni javob satriga formatlash uchun fmt.Sprintf dan foydalaning. AddBlocked shunchaki blocked[url] = true ni o'rnatadi, URL ni kirish mumkin emas deb belgilaydi.`,
			hint2: `ProxyServer dagi HandleRequest uch narsani tartib bilan tekshiradi: (1) Agar URL bloklangan bo'lsa, "Access denied: {url}" qaytaring. (2) Agar URL keshda bo'lsa, "Cached: {cached_response}" qaytaring. (3) Aks holda, realServer.HandleRequest ni chaqiring, natijani keshga saqlang va qaytaring.`,
			whyItMatters: `**Proxy Pattern nima uchun kerak**

Proxy siz kirishni nazorat qilish va keshlashni to'g'ridan-to'g'ri serverga qo'shishingiz mumkin:

\`\`\`go
// Muammo: Server juda ko'p mas'uliyatlarga ega
type Server struct {
    cache   map[string]string
    blocked map[string]bool
}

func (s *Server) Handle(url string) string {
    if s.blocked[url] { return "denied" }  // kirishni nazorat qilish aralashgan
    if c, ok := s.cache[url]; ok { return c }  // keshlash aralashgan
    result := s.actualHandle(url)  // haqiqiy mantiq
    s.cache[url] = result
    return result
}
// Server endi 3 ta ish qilmoqda: qayta ishlash, keshlash, kirishni nazorat qilish
\`\`\`

Proxy bilan har bir mas'uliyat ajratilgan:

\`\`\`go
// Yechim: Proxy kesishuvchi vazifalarni bajaradi
type RealServer struct{}  // faqat haqiqiy ishni bajaradi

type ProxyServer struct {
    real    *RealServer
    cache   map[string]string
    blocked map[string]bool
}
// Proxy keshlash va kirishni nazorat qilishni boshqaradi
// RealServer o'z asosiy mas'uliyatiga e'tibor qaratadi
\`\`\`

**Go da Proxy ning real dunyo misollari**

1. **Virtual Proxy — Ma'lumotlar bazasi ulanishini dangasa yuklash**:
\`\`\`go
type Database interface {
    Query(sql string) []Row
}

type RealDatabase struct {
    connection *sql.DB  // yaratish qimmat
}

type LazyDatabaseProxy struct {
    realDB *RealDatabase  // birinchi foydalanishgacha nil
    dsn    string
    mu     sync.Mutex
}

func (p *LazyDatabaseProxy) Query(sql string) []Row {
    p.mu.Lock()
    defer p.mu.Unlock()
    if p.realDB == nil {  // dangasa initsializatsiya
        p.realDB = &RealDatabase{connection: connect(p.dsn)}
    }
    return p.realDB.Query(sql)
}
\`\`\`

2. **Protection Proxy — Rolga asoslangan kirishni nazorat qilish**:
\`\`\`go
type Document interface {
    Read() string
    Write(content string)
}

type SecureDocumentProxy struct {
    doc  *RealDocument
    user *User
}

func (p *SecureDocumentProxy) Write(content string) {
    if !p.user.HasPermission("write") {
        log.Printf("Access denied for user %s", p.user.ID)
        return
    }
    p.doc.Write(content)
}
\`\`\`

**Prodakshen pattern: Qayta urinish va Circuit Breaker bilan HTTP mijoz**

\`\`\`go
package main

import (
    "fmt"
    "net/http"
    "time"
    "sync"
)

// HTTPClient interfeysi
type HTTPClient interface {
    Do(req *http.Request) (*http.Response, error)
}

// RealHTTPClient standart http.Client ni o'raydi
type RealHTTPClient struct {
    client *http.Client
}

func (c *RealHTTPClient) Do(req *http.Request) (*http.Response, error) {
    return c.client.Do(req)
}

// CircuitState circuit breaker holatlarini ifodalaydi
type CircuitState int

const (
    StateClosed CircuitState = iota  // normal ishlash
    StateOpen                        // xatolar, so'rovlarni rad etamiz
    StateHalfOpen                    // xizmat tiklanganini tekshiramiz
)

// ResilientClientProxy qayta urinish va circuit breaker qo'shadi
type ResilientClientProxy struct {
    client       HTTPClient
    maxRetries   int
    retryDelay   time.Duration
    state        CircuitState
    failures     int
    threshold    int
    lastFailure  time.Time
    resetTimeout time.Duration
    mu           sync.RWMutex
}

func NewResilientClient(client HTTPClient) *ResilientClientProxy {
    return &ResilientClientProxy{
        client:       client,
        maxRetries:   3,
        retryDelay:   time.Second,
        state:        StateClosed,
        threshold:    5,
        resetTimeout: 30 * time.Second,
    }
}

func (p *ResilientClientProxy) Do(req *http.Request) (*http.Response, error) {
    // Circuit breaker holatini tekshiramiz
    if !p.allowRequest() {
        return nil, fmt.Errorf("circuit breaker open")
    }

    // Qayta urinish mantig'i
    var lastErr error
    for attempt := 0; attempt <= p.maxRetries; attempt++ {
        resp, err := p.client.Do(req)
        if err == nil && resp.StatusCode < 500 {
            p.recordSuccess()
            return resp, nil
        }
        lastErr = err
        if attempt < p.maxRetries {
            time.Sleep(p.retryDelay * time.Duration(attempt+1))
        }
    }

    p.recordFailure()
    return nil, lastErr
}

func (p *ResilientClientProxy) allowRequest() bool {
    p.mu.RLock()
    defer p.mu.RUnlock()

    switch p.state {
    case StateClosed:
        return true
    case StateOpen:
        if time.Since(p.lastFailure) > p.resetTimeout {
            p.mu.RUnlock()
            p.mu.Lock()
            p.state = StateHalfOpen
            p.mu.Unlock()
            p.mu.RLock()
            return true
        }
        return false
    case StateHalfOpen:
        return true
    }
    return false
}

func (p *ResilientClientProxy) recordSuccess() {
    p.mu.Lock()
    defer p.mu.Unlock()
    p.failures = 0
    p.state = StateClosed
}

func (p *ResilientClientProxy) recordFailure() {
    p.mu.Lock()
    defer p.mu.Unlock()
    p.failures++
    p.lastFailure = time.Now()
    if p.failures >= p.threshold {
        p.state = StateOpen
    }
}
\`\`\`

**Oldini olish kerak bo'lgan keng tarqalgan xatolar**

1. **Bir xil interfeysni amalga oshirmaslik** — Proxy haqiqiy sub'ekt bilan almashtiriladigan bo'lishi kerak
2. **Thread xavfsizligini unutish** — Keshlash proksilari parallel muhitda sinxronlashni talab qiladi
3. **Haqiqiy sub'ektsiz proksi yaratish** — Dangasa proksilar initsializatsiyani to'g'ri boshqarishi kerak
4. **Proxy juda ko'p ish qilmoqda** — Har bir proksi bitta mas'uliyatga ega bo'lishi kerak (keshlash YOKI kirishni nazorat qilish, ikkisi emas)
5. **Proksi zanjirlarini hisobga olmaslik** — Bir nechta proksilar qatlamli funksionallik uchun bir-birini o'rashi mumkin`,
			solutionCode: `package patterns

import "fmt"	// satrlarni formatlash uchun import

// Server so'rovlarni qayta ishlash uchun interfeys
type Server interface {	// haqiqiy va proksi uchun umumiy interfeys
	HandleRequest(url string) string	// so'rovlarni qayta ishlash uchun metod signaturasi
}

// RealServer — haqiqiy server
type RealServer struct{}	// haqiqiy ishni bajaradigan real sub'ekt

// HandleRequest haqiqiy serverda so'rovni qayta ishlaydi
func (s *RealServer) HandleRequest(url string) string {	// Server interfeysini amalga oshiradi
	return fmt.Sprintf("RealServer: Handling request for %s", url)	// haqiqiy so'rov qayta ishlash
}

// ProxyServer RealServer ga kirishni nazorat qiladi
type ProxyServer struct {	// haqiqiy serverni o'raydigan proksi
	realServer *RealServer	// haqiqiy serverga havola (dangasa yoki darhol)
	cache      map[string]string	// keshlash proksisi: javoblarni saqlaydi
	blocked    map[string]bool	// himoya proksisi: kirishni nazorat qilish ro'yxati
}

// NewProxyServer initsializatsiya qilingan maplar bilan yangi proksi yaratadi
func NewProxyServer() *ProxyServer {	// proksi uchun fabrika funksiyasi
	return &ProxyServer{	// proksini yaratamiz va initsializatsiya qilamiz
		realServer: &RealServer{},	// haqiqiy serverni yaratamiz (dangasa bo'lishi mumkin)
		cache:      make(map[string]string),	// kesh mapini initsializatsiya qilamiz
		blocked:    make(map[string]bool),	// bloklash ro'yxatini initsializatsiya qilamiz
	}
}

// AddBlocked kirishni nazorat qilish uchun URL ni bloklangan ro'yxatga qo'shadi
func (p *ProxyServer) AddBlocked(url string) {	// himoya proksisi funksionalligi
	p.blocked[url] = true	// URL ni bloklangan deb belgilaymiz
}

// HandleRequest kirishni nazorat qiladi va javoblarni keshlaydi
func (p *ProxyServer) HandleRequest(url string) string {	// Server interfeysini amalga oshiradi
	// Qadam 1: Himoya proksisi — kirishni nazorat qilishni tekshiramiz
	if p.blocked[url] {	// URL bloklangan ro'yxatda borligini tekshiramiz
		return fmt.Sprintf("Access denied: %s", url)	// bloklangan URL larga kirishni rad etamiz
	}

	// Qadam 2: Keshlash proksisi — avval keshni tekshiramiz
	if cached, exists := p.cache[url]; exists {	// keshda qidiramiz
		return fmt.Sprintf("Cached: %s", cached)	// keshlangan javobni qaytaramiz
	}

	// Qadam 3: Haqiqiy serverga yo'naltiramiz va natijani keshlaymiz
	response := p.realServer.HandleRequest(url)	// haqiqiy serverga delegatsiya qilamiz
	p.cache[url] = response	// javobni kelajak uchun keshga saqlaymiz
	return response	// yangi javobni qaytaramiz
}

// GetCacheSize keshlangan javoblar sonini qaytaradi
func (p *ProxyServer) GetCacheSize() int {	// monitoring uchun yordamchi metod
	return len(p.cache)	// kesh yozuvlari sonini qaytaramiz
}`
		}
	}
};

export default task;
