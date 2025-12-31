import { Task } from '../../../../../../../types';

export const task: Task = {
	slug: 'go-dp-decorator',
	title: 'Decorator Pattern',
	difficulty: 'medium',
	tags: ['go', 'design-patterns', 'structural', 'decorator'],
	estimatedTime: '35m',
	isPremium: false,
	youtubeUrl: '',
	description: `Implement the Decorator pattern in Go - attach additional responsibilities to an object dynamically.

The Decorator pattern provides a flexible alternative to subclassing for extending functionality. It allows behavior to be added to individual objects, either statically or dynamically, without affecting the behavior of other objects from the same class.

**You will implement:**

1. **Notifier interface** - Send(message) method returning notification results
2. **BasicNotifier struct** - Simple console notification (concrete component)
3. **SMSDecorator** - Adds SMS notification capability
4. **SlackDecorator** - Adds Slack notification capability
5. **EmailDecorator** - Adds Email notification capability

**Example Usage:**

\`\`\`go
// Start with basic console notifier
notifier := &BasicNotifier{}	// concrete component

// Wrap with SMS decorator
notifier = &SMSDecorator{	// first decorator layer
    Notifier: notifier,
    Phone: "+1234567890",
}

// Wrap with Slack decorator
notifier = &SlackDecorator{	// second decorator layer
    Notifier: notifier,
    Channel: "#alerts",
}

// Send notification through all layers
results := notifier.Send("Server down!")
// results contains:
// ["Console: Server down!", "SMS to +1234567890: Server down!", "Slack to #alerts: Server down!"]

// Decorators can be composed in any order
emailFirst := &EmailDecorator{
    Notifier: &BasicNotifier{},
    Email: "admin@example.com",
}
// Different composition, same interface
\`\`\``,
	initialCode: `package patterns

type Notifier interface {
}

type BasicNotifier struct{}

func (n *BasicNotifier) Send(message string) []string {
}

type SMSDecorator struct {
	Notifier Notifier
	Phone    string
}

func (d *SMSDecorator) Send(message string) []string {
}

type SlackDecorator struct {
	Notifier Notifier
	Channel  string
}

func (d *SlackDecorator) Send(message string) []string {
}

type EmailDecorator struct {
	Notifier Notifier
	Email    string
}

func (d *EmailDecorator) Send(message string) []string {
}`,
	solutionCode: `package patterns

import "fmt"

// Notifier interface for sending notifications
type Notifier interface {	// component interface that decorators implement
	Send(message string) []string	// returns all notification results
}

// BasicNotifier sends to console
type BasicNotifier struct{}	// concrete component - the base object being decorated

// Send outputs to console
func (n *BasicNotifier) Send(message string) []string {	// concrete component implementation
	return []string{fmt.Sprintf("Console: %s", message)}	// returns single-element slice with console message
}

// SMSDecorator adds SMS notification
type SMSDecorator struct {	// concrete decorator
	Notifier Notifier	// reference to wrapped component (can be another decorator)
	Phone    string	// decorator-specific data
}

// Send adds SMS notification to chain
func (d *SMSDecorator) Send(message string) []string {	// implements same interface as component
	results := d.Notifier.Send(message)	// delegate to wrapped component first
	return append(results, fmt.Sprintf("SMS to %s: %s", d.Phone, message))	// add own behavior
}

// SlackDecorator adds Slack notification
type SlackDecorator struct {	// concrete decorator
	Notifier Notifier	// reference to wrapped component
	Channel  string	// Slack channel for notification
}

// Send adds Slack notification to chain
func (d *SlackDecorator) Send(message string) []string {	// implements same interface as component
	results := d.Notifier.Send(message)	// delegate to wrapped component first
	return append(results, fmt.Sprintf("Slack to %s: %s", d.Channel, message))	// add own behavior
}

// EmailDecorator adds Email notification
type EmailDecorator struct {	// concrete decorator
	Notifier Notifier	// reference to wrapped component
	Email    string	// email address for notification
}

// Send adds Email notification to chain
func (d *EmailDecorator) Send(message string) []string {	// implements same interface as component
	results := d.Notifier.Send(message)	// delegate to wrapped component first
	return append(results, fmt.Sprintf("Email to %s: %s", d.Email, message))	// add own behavior
}`,
	hint1: `BasicNotifier is the concrete component - it returns a single-element slice:
\`\`\`go
return []string{fmt.Sprintf("Console: %s", message)}
\`\`\`

Each decorator struct has a Notifier field that holds the wrapped component. This field can reference either the BasicNotifier or another decorator, enabling chaining.`,
	hint2: `Each decorator follows the same pattern:
1. Call the wrapped component: \`results := d.Notifier.Send(message)\`
2. Add own notification: \`return append(results, fmt.Sprintf("SMS to %s: %s", d.Phone, message))\`

The key insight is that each decorator delegates to its wrapped component FIRST, then adds its own behavior. This creates a chain where all notifications are accumulated in order.`,
	whyItMatters: `## Why Decorator Exists

The Decorator pattern solves the problem of adding functionality to objects without creating an explosion of subclasses. Without it, combining features requires creating a class for every possible combination.

**Problem - Without Decorator (Subclass Explosion):**
\`\`\`go
// To combine features, you need every combination
type ConsoleNotifier struct{}
type SMSNotifier struct{}
type SlackNotifier struct{}
type ConsoleSMSNotifier struct{}      // Console + SMS
type ConsoleSlackNotifier struct{}    // Console + Slack
type SMSSlackNotifier struct{}        // SMS + Slack
type ConsoleSMSSlackNotifier struct{} // All three
// Adding Email requires 8 more classes!
\`\`\`

**Solution - With Decorator (Linear Growth):**
\`\`\`go
// Just one class per feature
notifier := &BasicNotifier{}
notifier = &SMSDecorator{Notifier: notifier}
notifier = &SlackDecorator{Notifier: notifier}
// Any combination at runtime!
\`\`\`

## Real-World Go Examples

**1. HTTP Middleware Chain:**
\`\`\`go
type Handler interface {
    ServeHTTP(w http.ResponseWriter, r *http.Request)
}

type LoggingMiddleware struct {
    Handler Handler
    Logger  *log.Logger
}

func (m *LoggingMiddleware) ServeHTTP(w http.ResponseWriter, r *http.Request) {
    start := time.Now()
    m.Handler.ServeHTTP(w, r)  // delegate first
    m.Logger.Printf("%s %s %v", r.Method, r.URL, time.Since(start))
}

type AuthMiddleware struct {
    Handler   Handler
    Validator TokenValidator
}

func (m *AuthMiddleware) ServeHTTP(w http.ResponseWriter, r *http.Request) {
    if !m.Validator.IsValid(r.Header.Get("Authorization")) {
        http.Error(w, "Unauthorized", 401)
        return
    }
    m.Handler.ServeHTTP(w, r)  // delegate if authorized
}

// Usage - compose middleware
handler := &MyHandler{}
handler = &AuthMiddleware{Handler: handler}
handler = &LoggingMiddleware{Handler: handler}
\`\`\`

**2. I/O Stream Decorators:**
\`\`\`go
type Reader interface {
    Read(p []byte) (n int, err error)
}

type BufferedReader struct {
    Reader Reader
    buffer []byte
    pos    int
}

func (r *BufferedReader) Read(p []byte) (int, error) {
    if r.pos >= len(r.buffer) {
        // Fill buffer from underlying reader
        n, err := r.Reader.Read(r.buffer)
        if err != nil {
            return 0, err
        }
        r.buffer = r.buffer[:n]
        r.pos = 0
    }
    n := copy(p, r.buffer[r.pos:])
    r.pos += n
    return n, nil
}

type GzipReader struct {
    Reader Reader
    gzReader *gzip.Reader
}

func (r *GzipReader) Read(p []byte) (int, error) {
    return r.gzReader.Read(p)  // decompresses on read
}

// Usage - stack readers
file := openFile("data.gz")
reader := &GzipReader{Reader: file}
reader := &BufferedReader{Reader: reader}
\`\`\`

## Production Pattern: Resilient HTTP Client

\`\`\`go
type HTTPClient interface {
    Do(req *http.Request) (*http.Response, error)
}

type BaseClient struct {
    client *http.Client
}

func (c *BaseClient) Do(req *http.Request) (*http.Response, error) {
    return c.client.Do(req)
}

// Retry decorator
type RetryClient struct {
    Client     HTTPClient
    MaxRetries int
    Backoff    time.Duration
}

func (c *RetryClient) Do(req *http.Request) (*http.Response, error) {
    var resp *http.Response
    var err error
    for i := 0; i <= c.MaxRetries; i++ {
        resp, err = c.Client.Do(req)
        if err == nil && resp.StatusCode < 500 {
            return resp, nil
        }
        time.Sleep(c.Backoff * time.Duration(i+1))
    }
    return resp, err
}

// Timeout decorator
type TimeoutClient struct {
    Client  HTTPClient
    Timeout time.Duration
}

func (c *TimeoutClient) Do(req *http.Request) (*http.Response, error) {
    ctx, cancel := context.WithTimeout(req.Context(), c.Timeout)
    defer cancel()
    return c.Client.Do(req.WithContext(ctx))
}

// Logging decorator
type LoggingClient struct {
    Client HTTPClient
    Logger *log.Logger
}

func (c *LoggingClient) Do(req *http.Request) (*http.Response, error) {
    start := time.Now()
    resp, err := c.Client.Do(req)
    c.Logger.Printf("%s %s -> %d (%v)", req.Method, req.URL, resp.StatusCode, time.Since(start))
    return resp, err
}

// Build resilient client
client := &BaseClient{client: http.DefaultClient}
client = &TimeoutClient{Client: client, Timeout: 10 * time.Second}
client = &RetryClient{Client: client, MaxRetries: 3, Backoff: time.Second}
client = &LoggingClient{Client: client, Logger: logger}
\`\`\`

## Common Mistakes

**1. Forgetting to Delegate:**
\`\`\`go
// Bad - doesn't call wrapped component
func (d *SMSDecorator) Send(message string) []string {
    return []string{fmt.Sprintf("SMS: %s", message)}  // loses chain!
}

// Good - always delegate first
func (d *SMSDecorator) Send(message string) []string {
    results := d.Notifier.Send(message)  // delegate
    return append(results, fmt.Sprintf("SMS: %s", message))
}
\`\`\`

**2. Modifying the Wrapped Component:**
\`\`\`go
// Bad - modifies wrapped component's state
func (d *CachingDecorator) Get(key string) string {
    d.Wrapped.cache[key] = value  // don't modify internals!
    return d.Wrapped.Get(key)
}

// Good - maintain own state
type CachingDecorator struct {
    Wrapped DataSource
    cache   map[string]string  // own cache
}

func (d *CachingDecorator) Get(key string) string {
    if val, ok := d.cache[key]; ok {
        return val
    }
    val := d.Wrapped.Get(key)
    d.cache[key] = val
    return val
}
\`\`\`

**3. Interface Mismatch:**
\`\`\`go
// Bad - decorator adds methods not in interface
type EnhancedNotifier struct {
    Notifier Notifier
}

func (n *EnhancedNotifier) Send(msg string) []string { ... }
func (n *EnhancedNotifier) SendPriority(msg string, priority int) []string { ... }  // not in Notifier!

// Client code can't use SendPriority through Notifier interface
var n Notifier = &EnhancedNotifier{}
n.SendPriority("msg", 1)  // compile error!

// Good - stick to the interface or define new one
type PriorityNotifier interface {
    Notifier
    SendPriority(msg string, priority int) []string
}
\`\`\`

**Key Principles:**
- Decorator implements the same interface as the component it wraps
- Always delegate to the wrapped component (before or after adding behavior)
- Decorators should be transparent to clients - they work with the interface
- Order of decoration matters - outer decorators see results from inner ones first`,
	order: 3,
	testCode: `package patterns

import (
	"testing"
)

// Test1: BasicNotifier.Send returns console message
func Test1(t *testing.T) {
	n := &BasicNotifier{}
	results := n.Send("test")
	if len(results) != 1 || results[0] != "Console: test" {
		t.Error("BasicNotifier should return console message")
	}
}

// Test2: SMSDecorator adds SMS to results
func Test2(t *testing.T) {
	n := &SMSDecorator{Notifier: &BasicNotifier{}, Phone: "+123"}
	results := n.Send("test")
	if len(results) != 2 {
		t.Error("SMSDecorator should add SMS to results")
	}
}

// Test3: SlackDecorator adds Slack to results
func Test3(t *testing.T) {
	n := &SlackDecorator{Notifier: &BasicNotifier{}, Channel: "#test"}
	results := n.Send("test")
	if results[1] != "Slack to #test: test" {
		t.Error("SlackDecorator should add Slack message")
	}
}

// Test4: EmailDecorator adds Email to results
func Test4(t *testing.T) {
	n := &EmailDecorator{Notifier: &BasicNotifier{}, Email: "a@b.com"}
	results := n.Send("test")
	if results[1] != "Email to a@b.com: test" {
		t.Error("EmailDecorator should add Email message")
	}
}

// Test5: Multiple decorators chain correctly
func Test5(t *testing.T) {
	n := &SMSDecorator{Notifier: &BasicNotifier{}, Phone: "+1"}
	n2 := &SlackDecorator{Notifier: n, Channel: "#c"}
	results := n2.Send("msg")
	if len(results) != 3 {
		t.Error("Chained decorators should produce 3 results")
	}
}

// Test6: Order of decorators matters
func Test6(t *testing.T) {
	n := &SlackDecorator{Notifier: &BasicNotifier{}, Channel: "#c"}
	results := n.Send("test")
	if results[0] != "Console: test" {
		t.Error("First result should be from BasicNotifier")
	}
}

// Test7: BasicNotifier implements Notifier
func Test7(t *testing.T) {
	var n Notifier = &BasicNotifier{}
	if n == nil {
		t.Error("BasicNotifier should implement Notifier")
	}
}

// Test8: SMSDecorator implements Notifier
func Test8(t *testing.T) {
	var n Notifier = &SMSDecorator{Notifier: &BasicNotifier{}}
	if n == nil {
		t.Error("SMSDecorator should implement Notifier")
	}
}

// Test9: Triple decorator chain
func Test9(t *testing.T) {
	n := &EmailDecorator{
		Notifier: &SlackDecorator{
			Notifier: &SMSDecorator{
				Notifier: &BasicNotifier{},
				Phone: "+1",
			},
			Channel: "#c",
		},
		Email: "a@b.com",
	}
	results := n.Send("test")
	if len(results) != 4 {
		t.Error("Triple decorator should produce 4 results")
	}
}

// Test10: SMS message format correct
func Test10(t *testing.T) {
	n := &SMSDecorator{Notifier: &BasicNotifier{}, Phone: "+1234567890"}
	results := n.Send("hello")
	if results[1] != "SMS to +1234567890: hello" {
		t.Error("SMS format should be 'SMS to phone: message'")
	}
}
`,
	translations: {
		ru: {
			title: 'Паттерн Decorator (Декоратор)',
			description: `Реализуйте паттерн Decorator на Go — динамически добавляйте объектам дополнительные обязанности.

Паттерн Decorator предоставляет гибкую альтернативу наследованию для расширения функциональности. Он позволяет добавлять поведение отдельным объектам статически или динамически, не влияя на поведение других объектов того же класса.

**Вы реализуете:**

1. **Интерфейс Notifier** - Метод Send(message) возвращающий результаты уведомлений
2. **Структура BasicNotifier** - Простое консольное уведомление (конкретный компонент)
3. **SMSDecorator** - Добавляет возможность SMS-уведомлений
4. **SlackDecorator** - Добавляет возможность Slack-уведомлений
5. **EmailDecorator** - Добавляет возможность Email-уведомлений

**Пример использования:**

\`\`\`go
// Начинаем с базового консольного уведомителя
notifier := &BasicNotifier{}	// конкретный компонент

// Оборачиваем в SMS декоратор
notifier = &SMSDecorator{	// первый слой декоратора
    Notifier: notifier,
    Phone: "+1234567890",
}

// Оборачиваем в Slack декоратор
notifier = &SlackDecorator{	// второй слой декоратора
    Notifier: notifier,
    Channel: "#alerts",
}

// Отправляем уведомление через все слои
results := notifier.Send("Server down!")
// results содержит:
// ["Console: Server down!", "SMS to +1234567890: Server down!", "Slack to #alerts: Server down!"]

// Декораторы можно комбинировать в любом порядке
emailFirst := &EmailDecorator{
    Notifier: &BasicNotifier{},
    Email: "admin@example.com",
}
// Другая композиция, тот же интерфейс
\`\`\``,
			hint1: `BasicNotifier — это конкретный компонент — он возвращает срез из одного элемента:
\`\`\`go
return []string{fmt.Sprintf("Console: %s", message)}
\`\`\`

Каждая структура декоратора имеет поле Notifier, которое хранит обёрнутый компонент. Это поле может ссылаться как на BasicNotifier, так и на другой декоратор, позволяя создавать цепочки.`,
			hint2: `Каждый декоратор следует одному паттерну:
1. Вызвать обёрнутый компонент: \`results := d.Notifier.Send(message)\`
2. Добавить собственное уведомление: \`return append(results, fmt.Sprintf("SMS to %s: %s", d.Phone, message))\`

Ключевая идея в том, что каждый декоратор СНАЧАЛА делегирует обёрнутому компоненту, затем добавляет своё поведение. Это создаёт цепочку, где все уведомления накапливаются по порядку.`,
			whyItMatters: `## Зачем нужен Decorator

Паттерн Decorator решает проблему добавления функциональности объектам без создания взрыва подклассов. Без него комбинирование функций требует создания класса для каждой возможной комбинации.

**Проблема — без Decorator (взрыв подклассов):**
\`\`\`go
// Для комбинации функций нужна каждая комбинация
type ConsoleNotifier struct{}
type SMSNotifier struct{}
type SlackNotifier struct{}
type ConsoleSMSNotifier struct{}      // Console + SMS
type ConsoleSlackNotifier struct{}    // Console + Slack
type SMSSlackNotifier struct{}        // SMS + Slack
type ConsoleSMSSlackNotifier struct{} // Все три
// Добавление Email требует ещё 8 классов!
\`\`\`

**Решение — с Decorator (линейный рост):**
\`\`\`go
// Только один класс на функцию
notifier := &BasicNotifier{}
notifier = &SMSDecorator{Notifier: notifier}
notifier = &SlackDecorator{Notifier: notifier}
// Любая комбинация во время выполнения!
\`\`\`

## Реальные примеры на Go

**1. Цепочка HTTP Middleware:**
\`\`\`go
type Handler interface {
    ServeHTTP(w http.ResponseWriter, r *http.Request)
}

type LoggingMiddleware struct {
    Handler Handler
    Logger  *log.Logger
}

func (m *LoggingMiddleware) ServeHTTP(w http.ResponseWriter, r *http.Request) {
    start := time.Now()
    m.Handler.ServeHTTP(w, r)  // сначала делегируем
    m.Logger.Printf("%s %s %v", r.Method, r.URL, time.Since(start))
}

type AuthMiddleware struct {
    Handler   Handler
    Validator TokenValidator
}

func (m *AuthMiddleware) ServeHTTP(w http.ResponseWriter, r *http.Request) {
    if !m.Validator.IsValid(r.Header.Get("Authorization")) {
        http.Error(w, "Unauthorized", 401)
        return
    }
    m.Handler.ServeHTTP(w, r)  // делегируем если авторизован
}

// Использование — композиция middleware
handler := &MyHandler{}
handler = &AuthMiddleware{Handler: handler}
handler = &LoggingMiddleware{Handler: handler}
\`\`\`

**2. Декораторы потоков ввода-вывода:**
\`\`\`go
type Reader interface {
    Read(p []byte) (n int, err error)
}

type BufferedReader struct {
    Reader Reader
    buffer []byte
    pos    int
}

func (r *BufferedReader) Read(p []byte) (int, error) {
    if r.pos >= len(r.buffer) {
        // Заполнить буфер из нижележащего reader
        n, err := r.Reader.Read(r.buffer)
        if err != nil {
            return 0, err
        }
        r.buffer = r.buffer[:n]
        r.pos = 0
    }
    n := copy(p, r.buffer[r.pos:])
    r.pos += n
    return n, nil
}

type GzipReader struct {
    Reader Reader
    gzReader *gzip.Reader
}

func (r *GzipReader) Read(p []byte) (int, error) {
    return r.gzReader.Read(p)  // распаковывает при чтении
}

// Использование — стек readers
file := openFile("data.gz")
reader := &GzipReader{Reader: file}
reader := &BufferedReader{Reader: reader}
\`\`\`

## Продакшен паттерн: Отказоустойчивый HTTP клиент

\`\`\`go
type HTTPClient interface {
    Do(req *http.Request) (*http.Response, error)
}

type BaseClient struct {
    client *http.Client
}

func (c *BaseClient) Do(req *http.Request) (*http.Response, error) {
    return c.client.Do(req)
}

// Декоратор повторных попыток
type RetryClient struct {
    Client     HTTPClient
    MaxRetries int
    Backoff    time.Duration
}

func (c *RetryClient) Do(req *http.Request) (*http.Response, error) {
    var resp *http.Response
    var err error
    for i := 0; i <= c.MaxRetries; i++ {
        resp, err = c.Client.Do(req)
        if err == nil && resp.StatusCode < 500 {
            return resp, nil
        }
        time.Sleep(c.Backoff * time.Duration(i+1))
    }
    return resp, err
}

// Декоратор таймаута
type TimeoutClient struct {
    Client  HTTPClient
    Timeout time.Duration
}

func (c *TimeoutClient) Do(req *http.Request) (*http.Response, error) {
    ctx, cancel := context.WithTimeout(req.Context(), c.Timeout)
    defer cancel()
    return c.Client.Do(req.WithContext(ctx))
}

// Декоратор логирования
type LoggingClient struct {
    Client HTTPClient
    Logger *log.Logger
}

func (c *LoggingClient) Do(req *http.Request) (*http.Response, error) {
    start := time.Now()
    resp, err := c.Client.Do(req)
    c.Logger.Printf("%s %s -> %d (%v)", req.Method, req.URL, resp.StatusCode, time.Since(start))
    return resp, err
}

// Построение отказоустойчивого клиента
client := &BaseClient{client: http.DefaultClient}
client = &TimeoutClient{Client: client, Timeout: 10 * time.Second}
client = &RetryClient{Client: client, MaxRetries: 3, Backoff: time.Second}
client = &LoggingClient{Client: client, Logger: logger}
\`\`\`

## Частые ошибки

**1. Забыли делегировать:**
\`\`\`go
// Плохо — не вызывает обёрнутый компонент
func (d *SMSDecorator) Send(message string) []string {
    return []string{fmt.Sprintf("SMS: %s", message)}  // теряем цепочку!
}

// Хорошо — всегда сначала делегируем
func (d *SMSDecorator) Send(message string) []string {
    results := d.Notifier.Send(message)  // делегируем
    return append(results, fmt.Sprintf("SMS: %s", message))
}
\`\`\`

**2. Модификация обёрнутого компонента:**
\`\`\`go
// Плохо — модифицирует состояние обёрнутого компонента
func (d *CachingDecorator) Get(key string) string {
    d.Wrapped.cache[key] = value  // не модифицируйте внутренности!
    return d.Wrapped.Get(key)
}

// Хорошо — храним собственное состояние
type CachingDecorator struct {
    Wrapped DataSource
    cache   map[string]string  // собственный кэш
}

func (d *CachingDecorator) Get(key string) string {
    if val, ok := d.cache[key]; ok {
        return val
    }
    val := d.Wrapped.Get(key)
    d.cache[key] = val
    return val
}
\`\`\`

**3. Несоответствие интерфейса:**
\`\`\`go
// Плохо — декоратор добавляет методы не из интерфейса
type EnhancedNotifier struct {
    Notifier Notifier
}

func (n *EnhancedNotifier) Send(msg string) []string { ... }
func (n *EnhancedNotifier) SendPriority(msg string, priority int) []string { ... }  // нет в Notifier!

// Клиентский код не может использовать SendPriority через интерфейс Notifier
var n Notifier = &EnhancedNotifier{}
n.SendPriority("msg", 1)  // ошибка компиляции!

// Хорошо — придерживайтесь интерфейса или определите новый
type PriorityNotifier interface {
    Notifier
    SendPriority(msg string, priority int) []string
}
\`\`\`

**Ключевые принципы:**
- Декоратор реализует тот же интерфейс, что и оборачиваемый компонент
- Всегда делегируйте обёрнутому компоненту (до или после добавления поведения)
- Декораторы должны быть прозрачны для клиентов — они работают с интерфейсом
- Порядок декорирования важен — внешние декораторы первыми видят результаты внутренних`,
			solutionCode: `package patterns

import "fmt"

// Notifier интерфейс для отправки уведомлений
type Notifier interface {	// интерфейс компонента, который реализуют декораторы
	Send(message string) []string	// возвращает все результаты уведомлений
}

// BasicNotifier отправляет в консоль
type BasicNotifier struct{}	// конкретный компонент — базовый объект для декорирования

// Send выводит в консоль
func (n *BasicNotifier) Send(message string) []string {	// реализация конкретного компонента
	return []string{fmt.Sprintf("Console: %s", message)}	// возвращает срез с одним элементом — консольным сообщением
}

// SMSDecorator добавляет SMS уведомление
type SMSDecorator struct {	// конкретный декоратор
	Notifier Notifier	// ссылка на обёрнутый компонент (может быть другим декоратором)
	Phone    string	// данные специфичные для декоратора
}

// Send добавляет SMS уведомление в цепочку
func (d *SMSDecorator) Send(message string) []string {	// реализует тот же интерфейс что и компонент
	results := d.Notifier.Send(message)	// сначала делегируем обёрнутому компоненту
	return append(results, fmt.Sprintf("SMS to %s: %s", d.Phone, message))	// добавляем своё поведение
}

// SlackDecorator добавляет Slack уведомление
type SlackDecorator struct {	// конкретный декоратор
	Notifier Notifier	// ссылка на обёрнутый компонент
	Channel  string	// Slack канал для уведомления
}

// Send добавляет Slack уведомление в цепочку
func (d *SlackDecorator) Send(message string) []string {	// реализует тот же интерфейс что и компонент
	results := d.Notifier.Send(message)	// сначала делегируем обёрнутому компоненту
	return append(results, fmt.Sprintf("Slack to %s: %s", d.Channel, message))	// добавляем своё поведение
}

// EmailDecorator добавляет Email уведомление
type EmailDecorator struct {	// конкретный декоратор
	Notifier Notifier	// ссылка на обёрнутый компонент
	Email    string	// email адрес для уведомления
}

// Send добавляет Email уведомление в цепочку
func (d *EmailDecorator) Send(message string) []string {	// реализует тот же интерфейс что и компонент
	results := d.Notifier.Send(message)	// сначала делегируем обёрнутому компоненту
	return append(results, fmt.Sprintf("Email to %s: %s", d.Email, message))	// добавляем своё поведение
}`
		},
		uz: {
			title: 'Decorator (Dekorator) Pattern',
			description: `Go tilida Decorator patternini amalga oshiring — ob'ektlarga qo'shimcha mas'uliyatlarni dinamik ravishda biriktiring.

Decorator patterni funksionallikni kengaytirish uchun vorislikka moslashuvchan alternativa taqdim etadi. U xatti-harakatni alohida ob'ektlarga statik yoki dinamik ravishda qo'shishga imkon beradi, bir xil klassdagi boshqa ob'ektlar xatti-harakatiga ta'sir qilmaydi.

**Siz amalga oshirasiz:**

1. **Notifier interfeysi** - Bildirishnoma natijalarini qaytaruvchi Send(message) metodi
2. **BasicNotifier strukturasi** - Oddiy konsol bildirishnomasi (konkret komponent)
3. **SMSDecorator** - SMS bildirishnoma imkoniyatini qo'shadi
4. **SlackDecorator** - Slack bildirishnoma imkoniyatini qo'shadi
5. **EmailDecorator** - Email bildirishnoma imkoniyatini qo'shadi

**Foydalanish namunasi:**

\`\`\`go
// Asosiy konsol notifier dan boshlaymiz
notifier := &BasicNotifier{}	// konkret komponent

// SMS dekorator bilan o'raymiz
notifier = &SMSDecorator{	// birinchi dekorator qatlami
    Notifier: notifier,
    Phone: "+1234567890",
}

// Slack dekorator bilan o'raymiz
notifier = &SlackDecorator{	// ikkinchi dekorator qatlami
    Notifier: notifier,
    Channel: "#alerts",
}

// Barcha qatlamlar orqali bildirishnoma yuboramiz
results := notifier.Send("Server down!")
// results quyidagilarni o'z ichiga oladi:
// ["Console: Server down!", "SMS to +1234567890: Server down!", "Slack to #alerts: Server down!"]

// Dekoratorlarni istalgan tartibda birlashtirish mumkin
emailFirst := &EmailDecorator{
    Notifier: &BasicNotifier{},
    Email: "admin@example.com",
}
// Boshqa kompozitsiya, bir xil interfeys
\`\`\``,
			hint1: `BasicNotifier konkret komponent — u bir elementli slice qaytaradi:
\`\`\`go
return []string{fmt.Sprintf("Console: %s", message)}
\`\`\`

Har bir dekorator strukturasi o'ralgan komponentni saqlaydigan Notifier maydoniga ega. Bu maydon BasicNotifier ga yoki boshqa dekoratorga murojaat qilishi mumkin, bu zanjir yaratishga imkon beradi.`,
			hint2: `Har bir dekorator bir xil patternni bajaradi:
1. O'ralgan komponentni chaqirish: \`results := d.Notifier.Send(message)\`
2. O'z bildirishnomasini qo'shish: \`return append(results, fmt.Sprintf("SMS to %s: %s", d.Phone, message))\`

Asosiy tushuncha shundaki, har bir dekorator AVVAL o'ralgan komponentga delegatsiya qiladi, keyin o'z xatti-harakatini qo'shadi. Bu barcha bildirishnomalar tartibda to'planadigan zanjir yaratadi.`,
			whyItMatters: `## Nega Decorator kerak

Decorator patterni subklasslar portlashisiz ob'ektlarga funksionallik qo'shish muammosini hal qiladi. Busiz xususiyatlarni birlashtirish har bir mumkin bo'lgan kombinatsiya uchun klass yaratishni talab qiladi.

**Muammo — Decorator siz (subklass portlashi):**
\`\`\`go
// Xususiyatlarni birlashtirish uchun har bir kombinatsiya kerak
type ConsoleNotifier struct{}
type SMSNotifier struct{}
type SlackNotifier struct{}
type ConsoleSMSNotifier struct{}      // Console + SMS
type ConsoleSlackNotifier struct{}    // Console + Slack
type SMSSlackNotifier struct{}        // SMS + Slack
type ConsoleSMSSlackNotifier struct{} // Hammasi
// Email qo'shish yana 8 ta klass talab qiladi!
\`\`\`

**Yechim — Decorator bilan (chiziqli o'sish):**
\`\`\`go
// Har bir xususiyat uchun faqat bitta klass
notifier := &BasicNotifier{}
notifier = &SMSDecorator{Notifier: notifier}
notifier = &SlackDecorator{Notifier: notifier}
// Runtime da istalgan kombinatsiya!
\`\`\`

## Go dagi real dunyo misollar

**1. HTTP Middleware zanjiri:**
\`\`\`go
type Handler interface {
    ServeHTTP(w http.ResponseWriter, r *http.Request)
}

type LoggingMiddleware struct {
    Handler Handler
    Logger  *log.Logger
}

func (m *LoggingMiddleware) ServeHTTP(w http.ResponseWriter, r *http.Request) {
    start := time.Now()
    m.Handler.ServeHTTP(w, r)  // avval delegatsiya
    m.Logger.Printf("%s %s %v", r.Method, r.URL, time.Since(start))
}

type AuthMiddleware struct {
    Handler   Handler
    Validator TokenValidator
}

func (m *AuthMiddleware) ServeHTTP(w http.ResponseWriter, r *http.Request) {
    if !m.Validator.IsValid(r.Header.Get("Authorization")) {
        http.Error(w, "Unauthorized", 401)
        return
    }
    m.Handler.ServeHTTP(w, r)  // avtorizatsiya bo'lsa delegatsiya
}

// Foydalanish — middleware kompozitsiyasi
handler := &MyHandler{}
handler = &AuthMiddleware{Handler: handler}
handler = &LoggingMiddleware{Handler: handler}
\`\`\`

**2. I/O Stream dekoratorlari:**
\`\`\`go
type Reader interface {
    Read(p []byte) (n int, err error)
}

type BufferedReader struct {
    Reader Reader
    buffer []byte
    pos    int
}

func (r *BufferedReader) Read(p []byte) (int, error) {
    if r.pos >= len(r.buffer) {
        // Asosiy reader dan buferni to'ldirish
        n, err := r.Reader.Read(r.buffer)
        if err != nil {
            return 0, err
        }
        r.buffer = r.buffer[:n]
        r.pos = 0
    }
    n := copy(p, r.buffer[r.pos:])
    r.pos += n
    return n, nil
}

type GzipReader struct {
    Reader Reader
    gzReader *gzip.Reader
}

func (r *GzipReader) Read(p []byte) (int, error) {
    return r.gzReader.Read(p)  // o'qishda dekompressiya
}

// Foydalanish — reader stack
file := openFile("data.gz")
reader := &GzipReader{Reader: file}
reader := &BufferedReader{Reader: reader}
\`\`\`

## Prodakshen pattern: Chidamli HTTP mijoz

\`\`\`go
type HTTPClient interface {
    Do(req *http.Request) (*http.Response, error)
}

type BaseClient struct {
    client *http.Client
}

func (c *BaseClient) Do(req *http.Request) (*http.Response, error) {
    return c.client.Do(req)
}

// Qayta urinish dekoratori
type RetryClient struct {
    Client     HTTPClient
    MaxRetries int
    Backoff    time.Duration
}

func (c *RetryClient) Do(req *http.Request) (*http.Response, error) {
    var resp *http.Response
    var err error
    for i := 0; i <= c.MaxRetries; i++ {
        resp, err = c.Client.Do(req)
        if err == nil && resp.StatusCode < 500 {
            return resp, nil
        }
        time.Sleep(c.Backoff * time.Duration(i+1))
    }
    return resp, err
}

// Timeout dekoratori
type TimeoutClient struct {
    Client  HTTPClient
    Timeout time.Duration
}

func (c *TimeoutClient) Do(req *http.Request) (*http.Response, error) {
    ctx, cancel := context.WithTimeout(req.Context(), c.Timeout)
    defer cancel()
    return c.Client.Do(req.WithContext(ctx))
}

// Logging dekoratori
type LoggingClient struct {
    Client HTTPClient
    Logger *log.Logger
}

func (c *LoggingClient) Do(req *http.Request) (*http.Response, error) {
    start := time.Now()
    resp, err := c.Client.Do(req)
    c.Logger.Printf("%s %s -> %d (%v)", req.Method, req.URL, resp.StatusCode, time.Since(start))
    return resp, err
}

// Chidamli mijoz qurish
client := &BaseClient{client: http.DefaultClient}
client = &TimeoutClient{Client: client, Timeout: 10 * time.Second}
client = &RetryClient{Client: client, MaxRetries: 3, Backoff: time.Second}
client = &LoggingClient{Client: client, Logger: logger}
\`\`\`

## Keng tarqalgan xatolar

**1. Delegatsiyani unutish:**
\`\`\`go
// Yomon — o'ralgan komponentni chaqirmaydi
func (d *SMSDecorator) Send(message string) []string {
    return []string{fmt.Sprintf("SMS: %s", message)}  // zanjir yo'qoladi!
}

// Yaxshi — har doim avval delegatsiya
func (d *SMSDecorator) Send(message string) []string {
    results := d.Notifier.Send(message)  // delegatsiya
    return append(results, fmt.Sprintf("SMS: %s", message))
}
\`\`\`

**2. O'ralgan komponentni o'zgartirish:**
\`\`\`go
// Yomon — o'ralgan komponent holatini o'zgartiradi
func (d *CachingDecorator) Get(key string) string {
    d.Wrapped.cache[key] = value  // ichki qismlarni o'zgartirmang!
    return d.Wrapped.Get(key)
}

// Yaxshi — o'z holatini saqlash
type CachingDecorator struct {
    Wrapped DataSource
    cache   map[string]string  // o'z keshi
}

func (d *CachingDecorator) Get(key string) string {
    if val, ok := d.cache[key]; ok {
        return val
    }
    val := d.Wrapped.Get(key)
    d.cache[key] = val
    return val
}
\`\`\`

**3. Interfeys nomuvofiqiligi:**
\`\`\`go
// Yomon — dekorator interfeysda bo'lmagan metodlar qo'shadi
type EnhancedNotifier struct {
    Notifier Notifier
}

func (n *EnhancedNotifier) Send(msg string) []string { ... }
func (n *EnhancedNotifier) SendPriority(msg string, priority int) []string { ... }  // Notifier da yo'q!

// Mijoz kodi Notifier interfeysi orqali SendPriority ni ishlata olmaydi
var n Notifier = &EnhancedNotifier{}
n.SendPriority("msg", 1)  // kompilyatsiya xatosi!

// Yaxshi — interfeysga amal qiling yoki yangi aniqlang
type PriorityNotifier interface {
    Notifier
    SendPriority(msg string, priority int) []string
}
\`\`\`

**Asosiy tamoyillar:**
- Dekorator o'raydigan komponent bilan bir xil interfeysni amalga oshiradi
- Har doim o'ralgan komponentga delegatsiya qiling (xatti-harakat qo'shishdan oldin yoki keyin)
- Dekoratorlar mijozlar uchun shaffof bo'lishi kerak — ular interfeys bilan ishlaydi
- Dekoratsiya tartibi muhim — tashqi dekoratorlar ichki natijalarni birinchi ko'radi`,
			solutionCode: `package patterns

import "fmt"

// Notifier bildirishnomalar yuborish uchun interfeys
type Notifier interface {	// dekoratorlar amalga oshiradigan komponent interfeysi
	Send(message string) []string	// barcha bildirishnoma natijalarini qaytaradi
}

// BasicNotifier konsolga yuboradi
type BasicNotifier struct{}	// konkret komponent — dekoratsiya qilinadigan asosiy ob'ekt

// Send konsolga chiqaradi
func (n *BasicNotifier) Send(message string) []string {	// konkret komponent amalga oshirishi
	return []string{fmt.Sprintf("Console: %s", message)}	// konsol xabari bilan bir elementli slice qaytaradi
}

// SMSDecorator SMS bildirishnomasini qo'shadi
type SMSDecorator struct {	// konkret dekorator
	Notifier Notifier	// o'ralgan komponentga havola (boshqa dekorator bo'lishi mumkin)
	Phone    string	// dekoratorga xos ma'lumot
}

// Send zanjirga SMS bildirishnomasini qo'shadi
func (d *SMSDecorator) Send(message string) []string {	// komponent bilan bir xil interfeysni amalga oshiradi
	results := d.Notifier.Send(message)	// avval o'ralgan komponentga delegatsiya
	return append(results, fmt.Sprintf("SMS to %s: %s", d.Phone, message))	// o'z xatti-harakatini qo'shish
}

// SlackDecorator Slack bildirishnomasini qo'shadi
type SlackDecorator struct {	// konkret dekorator
	Notifier Notifier	// o'ralgan komponentga havola
	Channel  string	// bildirishnoma uchun Slack kanali
}

// Send zanjirga Slack bildirishnomasini qo'shadi
func (d *SlackDecorator) Send(message string) []string {	// komponent bilan bir xil interfeysni amalga oshiradi
	results := d.Notifier.Send(message)	// avval o'ralgan komponentga delegatsiya
	return append(results, fmt.Sprintf("Slack to %s: %s", d.Channel, message))	// o'z xatti-harakatini qo'shish
}

// EmailDecorator Email bildirishnomasini qo'shadi
type EmailDecorator struct {	// konkret dekorator
	Notifier Notifier	// o'ralgan komponentga havola
	Email    string	// bildirishnoma uchun email manzili
}

// Send zanjirga Email bildirishnomasini qo'shadi
func (d *EmailDecorator) Send(message string) []string {	// komponent bilan bir xil interfeysni amalga oshiradi
	results := d.Notifier.Send(message)	// avval o'ralgan komponentga delegatsiya
	return append(results, fmt.Sprintf("Email to %s: %s", d.Email, message))	// o'z xatti-harakatini qo'shish
}`
		}
	}
};

export default task;
