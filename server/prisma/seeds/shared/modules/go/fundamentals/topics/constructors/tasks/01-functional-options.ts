import { Task } from '../../../../types';

export const task: Task = {
	slug: 'go-fundamentals-functional-options',
	title: 'Functional Options Pattern',
	difficulty: 'medium',	tags: ['go', 'patterns', 'constructors', 'options'],
	estimatedTime: '25m',	isPremium: false,
	youtubeUrl: '',
	description: `Implement the **Functional Options Pattern** for a flexible User struct constructor that allows optional configuration via functions.

**Pattern Overview:**
The Functional Options Pattern is an elegant way to handle optional configuration in Go constructors. Instead of having multiple constructor variants (NewUser, NewUserWithEmail, NewUserWithAge, etc.) or using a configuration struct, you pass functions that modify the object being constructed.

**Key Components:**

1. **User Struct** - Encapsulates user data with:
   1.1. \`id\` (private field, accessible via ID() method)
   1.2. \`Name\` (exported, required)
   1.3. \`Email\` (exported, optional)
   1.4. \`Age\` (exported, optional)

2. **Option Type** - A function type that takes *User and returns error:
   \`\`\`go
   type Option func(*User) error
   \`\`\`

3. **WithEmail Option** - Factory function that returns an Option:
   3.1. Validates email contains "@" and is not empty
   3.2. Returns error for invalid email
   3.3. Sets User.Email if valid

4. **WithAge Option** - Factory function that returns an Option:
   4.1. Validates age is between 0 and 130
   4.2. Returns error for invalid age
   4.3. Sets User.Age if valid

5. **NewUser Constructor** - Core function with signature:
   \`\`\`go
   func NewUser(id int, name string, opts ...Option) (*User, error)
   \`\`\`
   5.1. Validates id > 0 (required)
   5.2. Validates name is not empty (required)
   5.3. Creates User struct with id and name
   5.4. Applies all options sequentially
   5.5. Returns error on first option failure
   5.6. Returns configured User pointer or nil on error

6. **ID() Method** - Getter for private id field:
   6.1. Returns the private id value
   6.2. Safely handles nil receiver (returns 0)

**Why This Pattern:**
- Scales well with many optional fields
- No constructor overloading needed
- Extendable without modifying existing code
- Type-safe compared to config structs
- Clear intent and discoverability

**Usage Examples:**
\`\`\`go
// With no options
user, err := NewUser(1, "Alice")

// With email
user, err := NewUser(2, "Bob", WithEmail("bob@example.com"))

// With email and age
user, err := NewUser(3, "Charlie",
    WithEmail("charlie@example.com"),
    WithAge(25))

// Access data
fmt.Println(user.Name)        // "Charlie"
fmt.Println(user.Email)       // "charlie@example.com"
fmt.Println(user.Age)         // 25
fmt.Println(user.ID())        // 3
\``,	initialCode: `package structinit

import (
	"fmt"
	"strings"
)

// TODO: Define User struct with id (private), Name, Email, and Age fields
type User struct {
	// Add fields here: id (lowercase = private), Name, Email, Age
}

// Option is a function type that takes *User and returns error
type Option func(*User) error

// TODO: Implement WithEmail option that validates and sets email
// Validation: email must not be empty and must contain "@"
func WithEmail(email string) Option {
	return func(u *User) error {
		// TODO: Implement validation and set email
		return nil
	}
}

// TODO: Implement WithAge option that validates age is between 0-130
func WithAge(age int) Option {
	return func(u *User) error {
		// TODO: Implement validation and set age
		return nil
	}
}

// TODO: Implement NewUser constructor
// - id must be positive (> 0)
// - name must not be empty
// - Apply all options sequentially, stop on first error
func NewUser(id int, name string, opts ...Option) (*User, error) {
	// TODO: Validate id and name, create User, apply options
	return nil, fmt.Errorf("not implemented")
}

// TODO: Implement ID() method that returns the private id field
// Handle nil receiver by returning 0
func (u *User) ID() int {
	// TODO: Return id, handle nil receiver
	return 0
}

// Suppress unused import warning
var _ = strings.TrimSpace
var _ = fmt.Errorf`,
	solutionCode: `package structinit

import (
	"fmt"
	"strings"
)

type User struct {
	id    int                                          // Private field, accessible via ID() method
	Name  string                                       // Exported field for user name
	Email string                                       // Exported field for user email
	Age   int                                          // Exported field for user age
}

type Option func(*User) error                         // Function type for optional configuration

func WithEmail(email string) Option {
	return func(u *User) error {                       // Return a function that configures the user
		if strings.TrimSpace(email) == "" || !strings.Contains(email, "@") { // Validate email format
			return fmt.Errorf("invalid format")        // Return error if validation fails
		}
		u.Email = email                                // Set email if valid
		return nil                                     // Return nil on success
	}
}

func WithAge(age int) Option {
	return func(u *User) error {                       // Return a function that configures the user
		if age < 0 || age > 130 {                      // Validate age range
			return fmt.Errorf("invalid age")           // Return error if validation fails
		}
		u.Age = age                                    // Set age if valid
		return nil                                     // Return nil on success
	}
}

func NewUser(id int, name string, opts ...Option) (*User, error) {
	if id <= 0 {                                       // Validate id is positive
		return nil, fmt.Errorf("id must be positive")  // Return error if id is invalid
	}
	if strings.TrimSpace(name) == "" {                 // Validate name is not empty
		return nil, fmt.Errorf("name is required")     // Return error if name is invalid
	}
	u := &User{id: id, Name: name}                     // Create user with required fields
	for _, opt := range opts {                         // Iterate over all provided options
		if opt == nil {                                // Skip nil options safely
			continue
		}
		if err := opt(u); err != nil {                 // Apply option and check for errors
			return nil, err                            // Return error on first failure
		}
	}
	return u, nil                                      // Return configured user pointer
}

func (u *User) ID() int {
	if u == nil {                                      // Handle nil receiver safely
		return 0                                       // Return 0 for nil receiver
	}
	return u.id                                        // Return the private id field
}`,
	testCode: `package structinit

import "testing"

func Test1(t *testing.T) {
	// Basic user without options
	user, err := NewUser(1, "Alice")
	if err != nil || user == nil || user.Name != "Alice" || user.ID() != 1 {
		t.Errorf("expected valid user, got err=%v, user=%+v", err, user)
	}
}

func Test2(t *testing.T) {
	// User with email
	user, err := NewUser(2, "Bob", WithEmail("bob@example.com"))
	if err != nil || user.Email != "bob@example.com" {
		t.Errorf("expected email bob@example.com, got err=%v, email=%s", err, user.Email)
	}
}

func Test3(t *testing.T) {
	// User with age
	user, err := NewUser(3, "Charlie", WithAge(25))
	if err != nil || user.Age != 25 {
		t.Errorf("expected age 25, got err=%v, age=%d", err, user.Age)
	}
}

func Test4(t *testing.T) {
	// User with email and age
	user, err := NewUser(4, "Dave", WithEmail("dave@test.com"), WithAge(30))
	if err != nil || user.Email != "dave@test.com" || user.Age != 30 {
		t.Errorf("expected email and age, got err=%v, user=%+v", err, user)
	}
}

func Test5(t *testing.T) {
	// Invalid email - no @
	_, err := NewUser(5, "Eve", WithEmail("invalid"))
	if err == nil {
		t.Error("expected error for invalid email")
	}
}

func Test6(t *testing.T) {
	// Invalid age - negative
	_, err := NewUser(6, "Frank", WithAge(-1))
	if err == nil {
		t.Error("expected error for negative age")
	}
}

func Test7(t *testing.T) {
	// Invalid age - too old
	_, err := NewUser(7, "Grace", WithAge(131))
	if err == nil {
		t.Error("expected error for age > 130")
	}
}

func Test8(t *testing.T) {
	// Empty name
	_, err := NewUser(8, "")
	if err == nil {
		t.Error("expected error for empty name")
	}
}

func Test9(t *testing.T) {
	// Invalid id
	_, err := NewUser(0, "Henry")
	if err == nil {
		t.Error("expected error for id <= 0")
	}
}

func Test10(t *testing.T) {
	// ID() on nil
	var u *User = nil
	if u.ID() != 0 {
		t.Errorf("expected 0 for nil user, got %d", u.ID())
	}
}`,
	hint1: `The Option type is a function signature. WithEmail and WithAge should return functions that take *User and return error. Remember to validate data before modifying the User.`,
			hint2: `In NewUser, after creating the User struct with required fields (id and name), iterate through the options slice and call each one with the user pointer. Stop and return the error if any option fails.`,
			whyItMatters: `The Functional Options Pattern is a Go idiom that provides elegant, extensible constructor design without the drawbacks of multiple constructor variants or configuration structs.

**Why Functional Options:**
- **Scalability:** Handle many optional parameters elegantly
- **Extensibility:** Add new options without modifying existing code
- **Type Safety:** Options are statically typed functions
- **Clarity:** Code is self-documenting through function names
- **Flexibility:** Users choose which options to apply

**Production Examples:**

\`\`\`go
// Example 1: HTTP Server Configuration
type Server struct {
	host           string
	port           int
	maxConnections int
	timeout        time.Duration
	tlsConfig      *tls.Config
}

type ServerOption func(*Server) error

func WithTimeout(d time.Duration) ServerOption {
	return func(s *Server) error {
		if d <= 0 {
			return fmt.Errorf("timeout must be positive")
		}
		s.timeout = d
		return nil
	}
}

func WithTLS(certFile, keyFile string) ServerOption {
	return func(s *Server) error {
		cert, err := tls.LoadX509KeyPair(certFile, keyFile)
		if err != nil {
			return err
		}
		s.tlsConfig = &tls.Config{Certificates: []tls.Certificate{cert}}
		return nil
	}
}

server, err := NewServer("0.0.0.0", 8080,
	WithTimeout(30*time.Second),
	WithTLS("cert.pem", "key.pem"))
\`\`\`

\`\`\`go
// Example 2: Database Connection Pool
type Pool struct {
	maxConns     int
	maxIdleTime  time.Duration
	maxLifetime  time.Duration
	sslMode      string
}

type PoolOption func(*Pool) error

func WithMaxConnections(n int) PoolOption {
	return func(p *Pool) error {
		if n < 1 {
			return fmt.Errorf("connections must be at least 1")
		}
		p.maxConns = n
		return nil
	}
}

pool, err := NewPool(dsn,
	WithMaxConnections(25),
	WithMaxIdleTime(5*time.Minute))
\`\`\`

\`\`\`go
// Example 3: Logger Configuration
type Logger struct {
	level   LogLevel
	output  io.Writer
	format  string
	hooks   []Hook
}

type LoggerOption func(*Logger) error

func WithLevel(l LogLevel) LoggerOption {
	return func(log *Logger) error {
		if !isValidLevel(l) {
			return fmt.Errorf("invalid log level")
		}
		log.level = l
		return nil
	}
}

func WithHook(h Hook) LoggerOption {
	return func(log *Logger) error {
		if h == nil {
			return fmt.Errorf("hook cannot be nil")
		}
		log.hooks = append(log.hooks, h)
		return nil
	}
}

logger, err := NewLogger(
	WithLevel(DEBUG),
	WithHook(sentryHook),
	WithHook(fileHook))
\`\`\`

\`\`\`go
// Example 4: gRPC Client Configuration
type Client struct {
	conn    *grpc.ClientConn
	timeout time.Duration
	retries int
	auth    credentials.TransportCredentials
}

type ClientOption func(*Client) error

func WithRetries(n int) ClientOption {
	return func(c *Client) error {
		if n < 0 {
			return fmt.Errorf("retries cannot be negative")
		}
		c.retries = n
		return nil
	}
}

client, err := NewClient(target,
	WithTimeout(10*time.Second),
	WithRetries(3),
	WithSecureTransport(tlsCreds))
\`\`\`

**Comparison with Alternatives:**

Before Functional Options (constructor overloading - not possible in Go):
\`\`\`go
// Can't do this in Go!
NewUser(id, name)
NewUserWithEmail(id, name, email)
NewUserWithEmailAndAge(id, name, email, age)
\`\`\`

Configuration struct approach (less scalable):
\`\`\`go
type UserConfig struct {
	Email *string
	Age   *int
}
user := NewUser(id, name, &UserConfig{
	Email: &email,
	Age: &age,
})
\`\`\`

Functional Options (what we're implementing - cleaner!):
\`\`\`go
user, err := NewUser(id, name,
	WithEmail(email),
	WithAge(age))
\`\`\`

**Real-World Benefits:**
- **Kubernetes:** Official Go client uses functional options extensively
- **Docker:** Docker SDK uses this pattern for creating containers
- **AWS SDK:** Configuration uses functional options
- **Prometheus:** Client library uses options for metrics setup

**Key Patterns to Master:**
- Option functions are closures capturing configuration data
- Options are applied after object creation (not during)
- Validation happens inside option functions
- Options can depend on previous options
- nil options can be safely skipped

Without functional options, APIs become rigid and hard to extend. With them, APIs remain clean and backward compatible as requirements grow.`,	order: 0,
	translations: {
		ru: {
			title: 'Паттерн функциональных опций',
			description: `Реализуйте **Паттерн функциональных опций** для гибкого конструктора структуры User, который позволяет дополнительную конфигурацию через функции.

**Обзор паттерна:**
Паттерн функциональных опций - это элегантный способ обработки дополнительной конфигурации в конструкторах Go. Вместо наличия нескольких вариантов конструктора (NewUser, NewUserWithEmail, NewUserWithAge и т.д.) или использования структуры конфигурации, вы передаёте функции, которые модифицируют создаваемый объект.

**Ключевые компоненты:**

1. **Структура User** - инкапсулирует данные пользователя с полями:
   1.1. \`id\` (приватное поле, доступно через метод ID())
   1.2. \`Name\` (экспортированное, обязательное)
   1.3. \`Email\` (экспортированное, опциональное)
   1.4. \`Age\` (экспортированное, опциональное)

2. **Тип Option** - тип функции которая принимает *User и возвращает error:
   \`\`\`go
   type Option func(*User) error
   \`\`\`

3. **Опция WithEmail** - функция-фабрика которая возвращает Option:
   3.1. Проверяет что email содержит "@" и не пуст
   3.2. Возвращает ошибку при неверном email
   3.3. Устанавливает User.Email если валиден

4. **Опция WithAge** - функция-фабрика которая возвращает Option:
   4.1. Проверяет что возраст между 0 и 130
   4.2. Возвращает ошибку при неверном возрасте
   4.3. Устанавливает User.Age если валиден

5. **Конструктор NewUser** - основная функция с сигнатурой:
   \`\`\`go
   func NewUser(id int, name string, opts ...Option) (*User, error)
   \`\`\`
   5.1. Проверяет id > 0 (обязательное)
   5.2. Проверяет name не пуст (обязательное)
   5.3. Создаёт структуру User с id и name
   5.4. Применяет все опции последовательно
   5.5. Возвращает ошибку при первой ошибке опции
   5.6. Возвращает указатель на User или nil при ошибке

6. **Метод ID()** - getter для приватного поля id:
   6.1. Возвращает значение приватного id
   6.2. Безопасно обрабатывает nil receiver (возвращает 0)

**Почему этот паттерн:**
- Хорошо масштабируется с большим количеством опциональных полей
- Не требует перегрузки конструктора
- Расширяемо без модификации существующего кода
- Type-safe по сравнению со структурами конфигурации
- Ясное намерение и обнаруживаемость

**Примеры использования:**
\`\`\`go
// Без опций
user, err := NewUser(1, "Alice")

// С email
user, err := NewUser(2, "Bob", WithEmail("bob@example.com"))

// С email и возрастом
user, err := NewUser(3, "Charlie",
    WithEmail("charlie@example.com"),
    WithAge(25))

// Доступ к данным
fmt.Println(user.Name)        // "Charlie"
fmt.Println(user.Email)       // "charlie@example.com"
fmt.Println(user.Age)         // 25
fmt.Println(user.ID())        // 3
`,
			hint1: `Тип Option - это сигнатура функции. WithEmail и WithAge должны возвращать функции которые принимают *User и возвращают error. Помните о валидации данных перед модификацией User.`,
			hint2: `В NewUser, после создания структуры User с обязательными полями (id и name), пройдите по срезу опций и вызовите каждую с указателем на user. Остановитесь и верните ошибку если какая-то опция сбойит.`,
			whyItMatters: `Паттерн функциональных опций - это идиома Go которая обеспечивает элегантное расширяемое проектирование конструкторов без недостатков нескольких вариантов конструктора или структур конфигурации.

**Почему функциональные опции:**
- **Масштабируемость:** Обработка многих опциональных параметров элегантно
- **Расширяемость:** Добавление новых опций без модификации существующего кода
- **Type Safety:** Опции - это статически типизированные функции
- **Ясность:** Код самодокументируется через имена функций
- **Гибкость:** Пользователи выбирают какие опции применять

**Продакшен примеры:**

\`\`\`go
// Пример 1: Конфигурация HTTP сервера
type Server struct {
	host           string
	port           int
	maxConnections int
	timeout        time.Duration
	tlsConfig      *tls.Config
}

type ServerOption func(*Server) error

func WithTimeout(d time.Duration) ServerOption {
	return func(s *Server) error {
		if d <= 0 {
			return fmt.Errorf("timeout must be positive")
		}
		s.timeout = d
		return nil
	}
}

func WithTLS(certFile, keyFile string) ServerOption {
	return func(s *Server) error {
		cert, err := tls.LoadX509KeyPair(certFile, keyFile)
		if err != nil {
			return err
		}
		s.tlsConfig = &tls.Config{Certificates: []tls.Certificate{cert}}
		return nil
	}
}

server, err := NewServer("0.0.0.0", 8080,
	WithTimeout(30*time.Second),
	WithTLS("cert.pem", "key.pem"))
\`\`\`

\`\`\`go
// Пример 2: Пул соединений с БД
type Pool struct {
	maxConns     int
	maxIdleTime  time.Duration
	maxLifetime  time.Duration
	sslMode      string
}

type PoolOption func(*Pool) error

func WithMaxConnections(n int) PoolOption {
	return func(p *Pool) error {
		if n < 1 {
			return fmt.Errorf("connections must be at least 1")
		}
		p.maxConns = n
		return nil
	}
}

pool, err := NewPool(dsn,
	WithMaxConnections(25),
	WithMaxIdleTime(5*time.Minute))
\`\`\`

\`\`\`go
// Пример 3: Конфигурация логгера
type Logger struct {
	level   LogLevel
	output  io.Writer
	format  string
	hooks   []Hook
}

type LoggerOption func(*Logger) error

func WithLevel(l LogLevel) LoggerOption {
	return func(log *Logger) error {
		if !isValidLevel(l) {
			return fmt.Errorf("invalid log level")
		}
		log.level = l
		return nil
	}
}

func WithHook(h Hook) LoggerOption {
	return func(log *Logger) error {
		if h == nil {
			return fmt.Errorf("hook cannot be nil")
		}
		log.hooks = append(log.hooks, h)
		return nil
	}
}

logger, err := NewLogger(
	WithLevel(DEBUG),
	WithHook(sentryHook),
	WithHook(fileHook))
\`\`\`

\`\`\`go
// Пример 4: Конфигурация gRPC клиента
type Client struct {
	conn    *grpc.ClientConn
	timeout time.Duration
	retries int
	auth    credentials.TransportCredentials
}

type ClientOption func(*Client) error

func WithRetries(n int) ClientOption {
	return func(c *Client) error {
		if n < 0 {
			return fmt.Errorf("retries cannot be negative")
		}
		c.retries = n
		return nil
	}
}

client, err := NewClient(target,
	WithTimeout(10*time.Second),
	WithRetries(3),
	WithSecureTransport(tlsCreds))
\`\`\`

**Сравнение с альтернативами:**

До функциональных опций (перегрузка конструктора - невозможна в Go):
\`\`\`go
// Нельзя сделать в Go!
NewUser(id, name)
NewUserWithEmail(id, name, email)
NewUserWithEmailAndAge(id, name, email, age)
\`\`\`

Подход со структурой конфигурации (менее масштабируемо):
\`\`\`go
type UserConfig struct {
	Email *string
	Age   *int
}
user := NewUser(id, name, &UserConfig{
	Email: &email,
	Age: &age,
})
\`\`\`

Функциональные опции (что мы реализуем - чище!):
\`\`\`go
user, err := NewUser(id, name,
	WithEmail(email),
	WithAge(age))
\`\`\`

**Практические преимущества:**
- **Kubernetes:** Официальный Go клиент использует функциональные опции
- **Docker:** Docker SDK использует этот паттерн для создания контейнеров
- **AWS SDK:** Конфигурация использует функциональные опции
- **Prometheus:** Клиентская библиотека использует опции для настройки метрик

**Ключевые паттерны для освоения:**
- Функции-опции это замыкания захватывающие данные конфигурации
- Опции применяются после создания объекта (не во время)
- Валидация происходит внутри функций-опций
- Опции могут зависеть от предыдущих опций
- nil опции могут быть безопасно пропущены

Без функциональных опций APIs становятся жёсткими и сложными в расширении. С ними APIs остаются чистыми и обратно совместимыми при росте требований.`,
			solutionCode: `package structinit

import (
	"fmt"
	"strings"
)

type User struct {
	id    int                                          // Приватное поле, доступно через метод ID()
	Name  string                                       // Экспортированное поле для имени пользователя
	Email string                                       // Экспортированное поле для email пользователя
	Age   int                                          // Экспортированное поле для возраста пользователя
}

type Option func(*User) error                         // Тип функции для опциональной конфигурации

func WithEmail(email string) Option {
	return func(u *User) error {                       // Вернуть функцию которая конфигурирует пользователя
		if strings.TrimSpace(email) == "" || !strings.Contains(email, "@") { // Валидировать формат email
			return fmt.Errorf("invalid format")        // Вернуть ошибку если валидация провалена
		}
		u.Email = email                                // Установить email если валиден
		return nil                                     // Вернуть nil при успехе
	}
}

func WithAge(age int) Option {
	return func(u *User) error {                       // Вернуть функцию которая конфигурирует пользователя
		if age < 0 || age > 130 {                      // Валидировать диапазон возраста
			return fmt.Errorf("invalid age")           // Вернуть ошибку если валидация провалена
		}
		u.Age = age                                    // Установить возраст если валиден
		return nil                                     // Вернуть nil при успехе
	}
}

func NewUser(id int, name string, opts ...Option) (*User, error) {
	if id <= 0 {                                       // Валидировать id положительный
		return nil, fmt.Errorf("id must be positive")  // Вернуть ошибку если id неверный
	}
	if strings.TrimSpace(name) == "" {                 // Валидировать name не пустой
		return nil, fmt.Errorf("name is required")     // Вернуть ошибку если name неверный
	}
	u := &User{id: id, Name: name}                     // Создать пользователя с обязательными полями
	for _, opt := range opts {                         // Итерация по всем предоставленным опциям
		if opt == nil {                                // Безопасно пропустить nil опции
			continue
		}
		if err := opt(u); err != nil {                 // Применить опцию и проверить на ошибки
			return nil, err                            // Вернуть ошибку при первой неудаче
		}
	}
	return u, nil                                      // Вернуть сконфигурированный указатель на пользователя
}

func (u *User) ID() int {
	if u == nil {                                      // Безопасно обработать nil receiver
		return 0                                       // Вернуть 0 для nil receiver
	}
	return u.id                                        // Вернуть приватное поле id
}`
		},
		uz: {
			title: 'Funksional opsiyalar patterni',
			description: `**Funksional Parametrlar Paradigmasini** User strukturasining moslashtirilgan konstruktori uchun amalga oshiring, bu funksiyalar orqali ixtiyoriy konfiguratsiyaga imkon beradi.

**Paradigma ta'rifi:**
Funksional parametrlar paradigmasi Go konstruktorlarida ixtiyoriy konfiguratsiyani qayta ishlashning elegant usuli. Bir nechta konstruktor variantlariga (NewUser, NewUserWithEmail, NewUserWithAge va h.k.) yoki konfiguratsiya strukturasiga ega bo'lish o'rniga, siz yaratilyotgan ob'ektni o'zgartiruvchi funksiyalarni o'tasiz.

**Asosiy komponentlar:**

1. **User Strukturasi** - foydalanuvchi ma'lumotlarini shunday yigindilaydi:
   1.1. 'id' (xususiy maydon, ID() metodi orqali mavjud)
   1.2. 'Name' (eksportlangan, majburiy)
   1.3. 'Email' (eksportlangan, ixtiyoriy)
   1.4. 'Age' (eksportlangan, ixtiyoriy)

2. **Parametr turi** - *User ni qabul qiluvchi va error qaytaruvchi funksiya turi:
   \`\`\`go
   type Option func(*User) error
   \`\`\`

3. **WithEmail Parametri** - Option qaytaruvchi zavod funksiyasi:
   3.1. Email "@" ni o'z ichiga olganligini va bo'sh emasligini tekshiradi
   3.2. Noto'g'ri email uchun xato qaytaradi
   3.3. Agar to'g'ri bo'lsa User.Email ni o'rnatadi

4. **WithAge Parametri** - Option qaytaruvchi zavod funksiyasi:
   4.1. Yosh 0 dan 130 gacha ekanligini tekshiradi
   4.2. Noto'g'ri yosh uchun xato qaytaradi
   4.3. Agar to'g'ri bo'lsa User.Age ni o'rnatadi

5. **NewUser Konstruktori** - quyidagi imzosi bilan asosiy funksiya:
   \`\`\`go
   func NewUser(id int, name string, opts ...Option) (*User, error)
   \`\`\`
   5.1. id > 0 ekanligini tekshiradi (majburiy)
   5.2. name bo'sh emasligini tekshiradi (majburiy)
   5.3. id va name bilan User strukturasini yaratadi
   5.4. Barcha parametrlarni ketma-ket qo'llaydi
   5.5. Birinchi parametr xatosida xato qaytaradi
   5.6. Configured User ko'rsatkichini yoki xato bo'lsa nil qaytaradi

6. **ID() Metodi** - xususiy id maydonining getter:
   6.1. Xususiy id qiymatini qaytaradi
   6.2. Nil receiverini xavfsiz qayta ishlaydi (0 qaytaradi)

**Nima uchun bu paradigma:**
- Ko'p sonli ixtiyoriy maydonlari bilan yaxshi masshtablanadi
- Konstruktor overloading talab etmaydi
- Mavjud kodni o'zgartirmasdan kengaytiriladi
- Konfiguratsiya strukturalariga qaraganda turli xavfli
- Aniq niyat va topilishni osonlashtiradi

**Foydalanish misollari:**
\`\`\`go
// Parametrlarsiz
user, err := NewUser(1, "Alice")

// Email bilan
user, err := NewUser(2, "Bob", WithEmail("bob@example.com"))

// Email va yosh bilan
user, err := NewUser(3, "Charlie",
    WithEmail("charlie@example.com"),
    WithAge(25))

// Ma'lumotlarga kirish
fmt.Println(user.Name)        // "Charlie"
fmt.Println(user.Email)       // "charlie@example.com"
fmt.Println(user.Age)         // 25
fmt.Println(user.ID())        // 3
`,
			hint1: `Option turi funksiya imzosi. WithEmail va WithAge *User ni qabul qiluvchi va error qaytaruvchi funksiyalarni qaytarishi kerak. User ni modifikatsiya qilishdan oldin ma'lumotlarni tekshirishni unutmang.`,
			hint2: `NewUser da, majburiy maydonlar bilan User strukturasini yaratgandan keyin (id va name), parametrlar bo'limini aylanib chiqing va har birini foydalanuvchi ko'rsatkichi bilan chaqiring. Agar biron bir parametr muvaffaqiyatsiz bo'lsa, to'xtang va xato qaytaring.`,
			whyItMatters: `Funksional parametrlar paradigmasi Go idiomasi bo'lib, bir nechta konstruktor variantlari yoki konfiguratsiya strukturalarining kamchiliklarisiz elegant kengaytirilgan konstruktor dizaynini ta'minlaydi.

**Nima uchun funksional parametrlar:**
- **Masshtablanuvchilik:** Ko'p sonli ixtiyoriy parametrlarni elegant qayta ishlash
- **Kengaytiriluvchilik:** Mavjud kodni o'zgartirmasdan yangi parametrlarni qo'shish
- **Tur xavfsizligi:** Parametrlar statik ravishda tipizlangan funksiyalar
- **Aniqlik:** Kod funksiya nomlari orqali o'zini dokumentlashtiradi
- **Moslashtiriluvchilik:** Foydalanuvchilar qaysi parametrlarni qo'llashni tanlaydilar

**Ishlab chiqarish misollari:**

\`\`\`go
// Misol 1: HTTP server konfiguratsiyasi
type Server struct {
	host           string
	port           int
	maxConnections int
	timeout        time.Duration
	tlsConfig      *tls.Config
}

type ServerOption func(*Server) error

func WithTimeout(d time.Duration) ServerOption {
	return func(s *Server) error {
		if d <= 0 {
			return fmt.Errorf("timeout must be positive")
		}
		s.timeout = d
		return nil
	}
}

func WithTLS(certFile, keyFile string) ServerOption {
	return func(s *Server) error {
		cert, err := tls.LoadX509KeyPair(certFile, keyFile)
		if err != nil {
			return err
		}
		s.tlsConfig = &tls.Config{Certificates: []tls.Certificate{cert}}
		return nil
	}
}

server, err := NewServer("0.0.0.0", 8080,
	WithTimeout(30*time.Second),
	WithTLS("cert.pem", "key.pem"))
\`\`\`

\`\`\`go
// Misol 2: DB ulanish to'plami
type Pool struct {
	maxConns     int
	maxIdleTime  time.Duration
	maxLifetime  time.Duration
	sslMode      string
}

type PoolOption func(*Pool) error

func WithMaxConnections(n int) PoolOption {
	return func(p *Pool) error {
		if n < 1 {
			return fmt.Errorf("connections must be at least 1")
		}
		p.maxConns = n
		return nil
	}
}

pool, err := NewPool(dsn,
	WithMaxConnections(25),
	WithMaxIdleTime(5*time.Minute))
\`\`\`

\`\`\`go
// Misol 3: Logger konfiguratsiyasi
type Logger struct {
	level   LogLevel
	output  io.Writer
	format  string
	hooks   []Hook
}

type LoggerOption func(*Logger) error

func WithLevel(l LogLevel) LoggerOption {
	return func(log *Logger) error {
		if !isValidLevel(l) {
			return fmt.Errorf("invalid log level")
		}
		log.level = l
		return nil
	}
}

func WithHook(h Hook) LoggerOption {
	return func(log *Logger) error {
		if h == nil {
			return fmt.Errorf("hook cannot be nil")
		}
		log.hooks = append(log.hooks, h)
		return nil
	}
}

logger, err := NewLogger(
	WithLevel(DEBUG),
	WithHook(sentryHook),
	WithHook(fileHook))
\`\`\`

\`\`\`go
// Misol 4: gRPC kliyent konfiguratsiyasi
type Client struct {
	conn    *grpc.ClientConn
	timeout time.Duration
	retries int
	auth    credentials.TransportCredentials
}

type ClientOption func(*Client) error

func WithRetries(n int) ClientOption {
	return func(c *Client) error {
		if n < 0 {
			return fmt.Errorf("retries cannot be negative")
		}
		c.retries = n
		return nil
	}
}

client, err := NewClient(target,
	WithTimeout(10*time.Second),
	WithRetries(3),
	WithSecureTransport(tlsCreds))
\`\`\`

**Alternativalar bilan taqqoslash:**

Funksional parametrlardan oldin (konstruktor ortiqcha yuklash - Go da mumkin emas):
\`\`\`go
// Go da buni qilish mumkin emas!
NewUser(id, name)
NewUserWithEmail(id, name, email)
NewUserWithEmailAndAge(id, name, email, age)
\`\`\`

Konfiguratsiya strukturasi yondashuvi (kamroq masshtablanadi):
\`\`\`go
type UserConfig struct {
	Email *string
	Age   *int
}
user := NewUser(id, name, &UserConfig{
	Email: &email,
	Age: &age,
})
\`\`\`

Funksional parametrlar (biz amalga oshirayotgan narsa - tozaroq!):
\`\`\`go
user, err := NewUser(id, name,
	WithEmail(email),
	WithAge(age))
\`\`\`

**Amaliy foydalari:**
- **Kubernetes:** Rasmiy Go kliyenti funksional parametrlarni keng qo'llaydi
- **Docker:** Docker SDK konteynerlarni yaratish uchun bu paradigmadan foydalanadi
- **AWS SDK:** Konfiguratsiya funksional parametrlardan foydalanadi
- **Prometheus:** Kliyent kutubxonasi metrikal o'rnatish uchun parametrlardan foydalanadi

**O'zlashtirish uchun asosiy patternlar:**
- Parametr funksiyalari konfiguratsiya ma'lumotlarini qamrab oluvchi yopilmalardir
- Parametrlar ob'ekt yaratilgandan keyin qo'llaniladi (yaratish vaqtida emas)
- Validatsiya parametr funksiyalari ichida sodir bo'ladi
- Parametrlar oldingi parametrlarga bog'liq bo'lishi mumkin
- nil parametrlar xavfsiz ravishda o'tkazib yuborilishi mumkin

Funksional parametrlarsiz APIlar qat'iy va kengaytirilishi qiyin bo'ladi. Ular bilan APIlar toza va talablar o'zgarishi bilan orqaga mutlaqo mos bo'lib qoladi.`,
			solutionCode: `package structinit

import (
	"fmt"
	"strings"
)

type User struct {
	id    int                                          // Xususiy maydon, ID() metodi orqali mavjud
	Name  string                                       // Foydalanuvchi nomi uchun eksport qilingan maydon
	Email string                                       // Foydalanuvchi emaili uchun eksport qilingan maydon
	Age   int                                          // Foydalanuvchi yoshi uchun eksport qilingan maydon
}

type Option func(*User) error                         // Ixtiyoriy konfiguratsiya uchun funksiya turi

func WithEmail(email string) Option {
	return func(u *User) error {                       // Foydalanuvchini konfiguratsiya qiluvchi funksiyani qaytarish
		if strings.TrimSpace(email) == "" || !strings.Contains(email, "@") { // Email formatini tekshirish
			return fmt.Errorf("invalid format")        // Tekshiruv muvaffaqiyatsiz bo'lsa xato qaytarish
		}
		u.Email = email                                // Agar to'g'ri bo'lsa emailni o'rnatish
		return nil                                     // Muvaffaqiyatda nil qaytarish
	}
}

func WithAge(age int) Option {
	return func(u *User) error {                       // Foydalanuvchini konfiguratsiya qiluvchi funksiyani qaytarish
		if age < 0 || age > 130 {                      // Yosh diapazonini tekshirish
			return fmt.Errorf("invalid age")           // Tekshiruv muvaffaqiyatsiz bo'lsa xato qaytarish
		}
		u.Age = age                                    // Agar to'g'ri bo'lsa yoshni o'rnatish
		return nil                                     // Muvaffaqiyatda nil qaytarish
	}
}

func NewUser(id int, name string, opts ...Option) (*User, error) {
	if id <= 0 {                                       // id musbat ekanligini tekshirish
		return nil, fmt.Errorf("id must be positive")  // id noto'g'ri bo'lsa xato qaytarish
	}
	if strings.TrimSpace(name) == "" {                 // name bo'sh emasligini tekshirish
		return nil, fmt.Errorf("name is required")     // name noto'g'ri bo'lsa xato qaytarish
	}
	u := &User{id: id, Name: name}                     // Majburiy maydonlar bilan foydalanuvchi yaratish
	for _, opt := range opts {                         // Barcha taqdim etilgan parametrlar bo'ylab iteratsiya
		if opt == nil {                                // Nil parametrlarni xavfsiz o'tkazib yuborish
			continue
		}
		if err := opt(u); err != nil {                 // Parametrni qo'llash va xatolarni tekshirish
			return nil, err                            // Birinchi muvaffaqiyatsizlikda xato qaytarish
		}
	}
	return u, nil                                      // Konfiguratsiya qilingan foydalanuvchi ko'rsatkichini qaytarish
}

func (u *User) ID() int {
	if u == nil {                                      // Nil receiverni xavfsiz qayta ishlash
		return 0                                       // Nil receiver uchun 0 qaytarish
	}
	return u.id                                        // Xususiy id maydonini qaytarish
}`
		}
	}
};

export default task;
