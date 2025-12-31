import { Task } from '../../../../types';

export const task: Task = {
	slug: 'go-fundamentals-constructor-validation',
	title: 'Constructor with Validation',
	difficulty: 'easy',
	tags: ['go', 'patterns', 'constructors', 'validation'],
	estimatedTime: '15m',
	isPremium: false,
	youtubeUrl: '',
	description: `Implement a constructor with comprehensive validation to ensure data integrity at object creation time.

**Pattern Overview:**
In Go, constructors are responsible for creating valid objects. Unlike some languages where validation might happen in setters, Go encourages validating all data upfront in the constructor and returning an error if the data is invalid. This ensures that once an object is created, it's always in a valid state.

**Key Components:**

1. **Email Struct** - Represents a validated email address:
   \`\`\`go
   type Email struct {
       address string  // Private field - can only be set through constructor
   }
   \`\`\`

2. **String() Method** - Getter for the private address:
   \`\`\`go
   func (e Email) String() string
   \`\`\`
   2.1. Returns the email address as a string
   2.2. Allows read access to the private field

3. **NewEmail Constructor** - Creates Email with validation:
   \`\`\`go
   func NewEmail(address string) (Email, error)
   \`\`\`
   3.1. Validates email is not empty (after trimming whitespace)
   3.2. Validates email contains exactly one "@" symbol
   3.3. Validates email has at least one character before "@"
   3.4. Validates email has at least one character after "@"
   3.5. Returns Email{} and error if validation fails
   3.6. Returns populated Email{address: address} and nil if valid

**Validation Rules:**

1. **Non-empty:** \`strings.TrimSpace(address) != ""\`
2. **Contains @:** \`strings.Count(address, "@") == 1\`
3. **Local part:** Characters before @ must exist (len > 0)
4. **Domain part:** Characters after @ must exist (len > 0)

**Implementation Strategy:**
- Use \`strings.Split(address, "@")\` to separate local and domain parts
- Check that the resulting slice has exactly 2 elements
- Validate both parts are non-empty after trimming
- Return descriptive error messages for each validation failure

**Usage Example:**
\`\`\`go
// Valid emails
email, err := NewEmail("user@example.com")
if err == nil {
    fmt.Println(email.String()) // "user@example.com"
}

email, err = NewEmail("john.doe@company.co.uk")
if err == nil {
    fmt.Println(email.String()) // "john.doe@company.co.uk"
}

// Invalid emails - all return errors
email, err = NewEmail("")                // Error: "email cannot be empty"
email, err = NewEmail("   ")             // Error: "email cannot be empty"
email, err = NewEmail("notanemail")      // Error: "invalid email format"
email, err = NewEmail("@example.com")    // Error: "invalid email format"
email, err = NewEmail("user@")           // Error: "invalid email format"
email, err = NewEmail("user@@example.com") // Error: "invalid email format"
\`\`\`

**Why This Pattern:**
- Guarantees object invariants from creation
- Prevents invalid state from existing
- Makes bugs easier to catch (fail fast at construction)
- Clear contract: if constructor succeeds, object is valid
- No need to validate on every method call`,
	initialCode: `package structinit

import (
	"fmt"
	"strings"
)

// TODO: Define Email struct with private address field
type Email struct {
	// TODO: Add fields
}

// TODO: Implement String method that returns the email address
func (e *Email) String() string {
	return "" // TODO: Implement
}

// TODO: Implement NewEmail constructor with validation
// Validation rules:
// 1. Address must not be empty (after trimming)
// 2. Must contain exactly one "@" symbol
// 3. Must have at least one character before "@"
// 4. Must have at least one character after "@"
// Return Email{} and error if invalid
// Return Email{address: address} and nil if valid
func NewEmail(address string) (Email, error) {
	var zero Email
	return zero, nil // TODO: Implement
}`,
	solutionCode: `package structinit

import (
	"fmt"
	"strings"
)

type Email struct {
	address string                               // Private field storing email address
}

func (e Email) String() string {
	return e.address                             // Return the email address
}

func NewEmail(address string) (Email, error) {
	trimmed := strings.TrimSpace(address)        // Remove leading/trailing whitespace

	if trimmed == "" {                           // Check if email is empty
		return Email{}, fmt.Errorf("email cannot be empty")
	}

	if strings.Count(trimmed, "@") != 1 {        // Check for exactly one @ symbol
		return Email{}, fmt.Errorf("invalid email format")
	}

	parts := strings.Split(trimmed, "@")         // Split into local and domain parts

	if strings.TrimSpace(parts[0]) == "" {       // Validate local part exists
		return Email{}, fmt.Errorf("invalid email format")
	}

	if strings.TrimSpace(parts[1]) == "" {       // Validate domain part exists
		return Email{}, fmt.Errorf("invalid email format")
	}

	return Email{address: trimmed}, nil          // Return valid Email object
}`,
	testCode: `package structinit

import "testing"

func Test1(t *testing.T) {
	// Valid email
	e, err := NewEmail("user@example.com")
	if err != nil || e.String() != "user@example.com" {
		t.Errorf("expected user@example.com, got err=%v, val=%s", err, e.String())
	}
}

func Test2(t *testing.T) {
	// Email with subdomain
	e, err := NewEmail("john.doe@company.co.uk")
	if err != nil || e.String() != "john.doe@company.co.uk" {
		t.Errorf("expected john.doe@company.co.uk, got err=%v", err)
	}
}

func Test3(t *testing.T) {
	// Empty email
	_, err := NewEmail("")
	if err == nil {
		t.Error("expected error for empty email")
	}
}

func Test4(t *testing.T) {
	// Only whitespace
	_, err := NewEmail("   ")
	if err == nil {
		t.Error("expected error for whitespace email")
	}
}

func Test5(t *testing.T) {
	// No @ symbol
	_, err := NewEmail("notanemail")
	if err == nil {
		t.Error("expected error for email without @")
	}
}

func Test6(t *testing.T) {
	// Missing local part
	_, err := NewEmail("@example.com")
	if err == nil {
		t.Error("expected error for missing local part")
	}
}

func Test7(t *testing.T) {
	// Missing domain part
	_, err := NewEmail("user@")
	if err == nil {
		t.Error("expected error for missing domain")
	}
}

func Test8(t *testing.T) {
	// Multiple @ symbols
	_, err := NewEmail("user@@example.com")
	if err == nil {
		t.Error("expected error for multiple @")
	}
}

func Test9(t *testing.T) {
	// Email with leading/trailing spaces
	e, err := NewEmail("  test@test.com  ")
	if err != nil || e.String() != "test@test.com" {
		t.Errorf("expected trimmed email, got err=%v, val=%s", err, e.String())
	}
}

func Test10(t *testing.T) {
	// Complex valid email
	e, err := NewEmail("a+b.c@x-y.z")
	if err != nil {
		t.Errorf("expected valid email, got err=%v", err)
	}
	if e.String() != "a+b.c@x-y.z" {
		t.Errorf("expected a+b.c@x-y.z, got %s", e.String())
	}
}`,
	hint1: `Start by defining the Email struct with a private 'address' field. The String() method should simply return this field. For validation, trim the input first and check if it's empty.`,
	hint2: `Use strings.Count to check for exactly one "@", then strings.Split to separate the local and domain parts. Make sure both parts exist and are non-empty after trimming. Return Email{} with an error if any validation fails.`,
	whyItMatters: `Constructor validation is a fundamental practice for creating robust, maintainable code that prevents bugs before they happen.

**Why Constructor Validation:**
- **Fail Fast:** Catch invalid data immediately at creation time
- **Invariant Enforcement:** Once created, object is always in valid state
- **Clear Contract:** Constructor success guarantees validity
- **No Defensive Programming:** No need to check validity in every method
- **Better Error Messages:** Validation errors at construction are easier to debug

**Production Examples:**

\`\`\`go
// Example 1: URL Validation
type URL struct {
	scheme string
	host   string
	path   string
}

func NewURL(rawURL string) (URL, error) {
	if strings.TrimSpace(rawURL) == "" {
		return URL{}, fmt.Errorf("URL cannot be empty")
	}

	parsed, err := url.Parse(rawURL)
	if err != nil {
		return URL{}, fmt.Errorf("invalid URL: %w", err)
	}

	if parsed.Scheme == "" {
		return URL{}, fmt.Errorf("URL must have a scheme (http/https)")
	}

	if parsed.Host == "" {
		return URL{}, fmt.Errorf("URL must have a host")
	}

	return URL{
		scheme: parsed.Scheme,
		host:   parsed.Host,
		path:   parsed.Path,
	}, nil
}
\`\`\`

\`\`\`go
// Example 2: Port Number Validation
type Port struct {
	number int
}

func NewPort(port int) (Port, error) {
	if port < 1 || port > 65535 {
		return Port{}, fmt.Errorf("port must be between 1 and 65535, got %d", port)
	}

	return Port{number: port}, nil
}

func (p Port) Number() int {
	return p.number
}
\`\`\`

\`\`\`go
// Example 3: Credit Card Validation
type CreditCard struct {
	number     string
	expiryDate time.Time
	cvv        string
}

func NewCreditCard(number, cvv string, expiry time.Time) (CreditCard, error) {
	// Remove spaces and dashes
	number = strings.ReplaceAll(strings.ReplaceAll(number, " ", ""), "-", "")

	if len(number) < 13 || len(number) > 19 {
		return CreditCard{}, fmt.Errorf("invalid card number length")
	}

	if !luhnCheck(number) {
		return CreditCard{}, fmt.Errorf("invalid card number (Luhn check failed)")
	}

	if len(cvv) < 3 || len(cvv) > 4 {
		return CreditCard{}, fmt.Errorf("CVV must be 3 or 4 digits")
	}

	if expiry.Before(time.Now()) {
		return CreditCard{}, fmt.Errorf("card has expired")
	}

	return CreditCard{
		number:     number,
		cvv:        cvv,
		expiryDate: expiry,
	}, nil
}
\`\`\`

\`\`\`go
// Example 4: Username Validation
type Username struct {
	value string
}

func NewUsername(username string) (Username, error) {
	trimmed := strings.TrimSpace(username)

	if len(trimmed) < 3 {
		return Username{}, fmt.Errorf("username must be at least 3 characters")
	}

	if len(trimmed) > 20 {
		return Username{}, fmt.Errorf("username must be at most 20 characters")
	}

	// Only allow alphanumeric and underscore
	if !regexp.MustCompile(\`^[a-zA-Z0-9_]+$\`).MatchString(trimmed) {
		return Username{}, fmt.Errorf("username can only contain letters, numbers, and underscores")
	}

	return Username{value: trimmed}, nil
}

func (u Username) String() string {
	return u.value
}
\`\`\`

\`\`\`go
// Example 5: Money Amount Validation
type Money struct {
	amount   int64  // Store as cents to avoid floating point issues
	currency string
}

func NewMoney(amount float64, currency string) (Money, error) {
	if amount < 0 {
		return Money{}, fmt.Errorf("amount cannot be negative")
	}

	currency = strings.ToUpper(strings.TrimSpace(currency))
	if len(currency) != 3 {
		return Money{}, fmt.Errorf("currency must be 3-letter ISO code")
	}

	// Convert to cents
	cents := int64(amount * 100)

	return Money{
		amount:   cents,
		currency: currency,
	}, nil
}
\`\`\`

**Comparison: Validation in Constructor vs Methods:**

Without constructor validation (problematic):
\`\`\`go
type Email struct {
	Address string // Public - can be modified
}

// Every method must validate!
func (e Email) Send(message string) error {
	if !isValidEmail(e.Address) {
		return fmt.Errorf("invalid email")
	}
	// ... send logic
}

func (e Email) Domain() string {
	if !isValidEmail(e.Address) {
		return ""
	}
	// ... extract domain
}
\`\`\`

With constructor validation (clean):
\`\`\`go
type Email struct {
	address string // Private
}

func NewEmail(address string) (Email, error) {
	if !isValidEmail(address) {
		return Email{}, fmt.Errorf("invalid email")
	}
	return Email{address: address}, nil
}

// Methods can assume validity!
func (e Email) Send(message string) error {
	// No validation needed - email is always valid
	// ... send logic
}

func (e Email) Domain() string {
	// No validation needed - email is always valid
	// ... extract domain
}
\`\`\`

**Real-World Benefits:**
- **gRPC:** Protocol buffer validators ensure message validity
- **Database Models:** ORMs validate models before persistence
- **HTTP Clients:** URL and request validation at construction
- **Configuration:** Validate config objects at startup

**Key Principles:**
- Private fields prevent external modification
- Validation happens once at construction
- Methods assume object is valid
- Clear error messages guide users
- Zero values should be invalid (force using constructor)

Constructor validation is the foundation of defensive programming in Go. It prevents entire classes of bugs by ensuring invalid objects never exist.`,
	order: 2,
	translations: {
		ru: {
			title: 'Валидация в конструкторе',
			description: `Реализуйте конструктор с всесторонней валидацией для обеспечения целостности данных во время создания объекта.

**Обзор паттерна:**
В Go конструкторы отвечают за создание валидных объектов. В отличие от некоторых языков, где валидация может происходить в сеттерах, Go поощряет валидацию всех данных заранее в конструкторе и возврат ошибки, если данные невалидны. Это гарантирует, что после создания объект всегда находится в валидном состоянии.

**Ключевые компоненты:**

1. **Структура Email** - представляет валидированный адрес электронной почты:
   \`\`\`go
   type Email struct {
       address string  // Приватное поле - может быть установлено только через конструктор
   }
   \`\`\`

2. **Метод String()** - геттер для приватного адреса:
   \`\`\`go
   func (e Email) String() string
   \`\`\`
   2.1. Возвращает адрес электронной почты как строку
   2.2. Позволяет доступ на чтение к приватному полю

3. **Конструктор NewEmail** - создает Email с валидацией:
   \`\`\`go
   func NewEmail(address string) (Email, error)
   \`\`\`
   3.1. Проверяет, что email не пуст (после удаления пробелов)
   3.2. Проверяет, что email содержит ровно один символ "@"
   3.3. Проверяет, что email имеет хотя бы один символ до "@"
   3.4. Проверяет, что email имеет хотя бы один символ после "@"
   3.5. Возвращает Email{} и ошибку, если валидация не прошла
   3.6. Возвращает заполненный Email{address: address} и nil, если валидный

**Правила валидации:**

1. **Не пустой:** \`strings.TrimSpace(address) != ""\`
2. **Содержит @:** \`strings.Count(address, "@") == 1\`
3. **Локальная часть:** Символы до @ должны существовать (len > 0)
4. **Доменная часть:** Символы после @ должны существовать (len > 0)

**Стратегия реализации:**
- Используйте \`strings.Split(address, "@")\` для разделения локальной и доменной частей
- Проверьте что результирующий slice имеет ровно 2 элемента
- Валидируйте что обе части не пусты после обрезки
- Возвращайте описательные сообщения об ошибках для каждой неудачной валидации

**Пример использования:**
\`\`\`go
// Валидные email
email, err := NewEmail("user@example.com")
if err == nil {
    fmt.Println(email.String()) // "user@example.com"
}

// Невалидные email - все возвращают ошибки
email, err = NewEmail("")                // Ошибка: "email cannot be empty"
email, err = NewEmail("notanemail")      // Ошибка: "invalid email format"
email, err = NewEmail("@example.com")    // Ошибка: "invalid email format"
\`\`\`

**Почему этот паттерн:**
- Гарантирует инварианты объекта с момента создания
- Предотвращает существование невалидного состояния
- Упрощает отлов ошибок (быстрый отказ при создании)
- Чёткий контракт: если конструктор успешен, объект валиден
- Не нужно валидировать при каждом вызове метода`,
			hint1: `Начните с определения структуры Email с приватным полем 'address'. Метод String() должен просто возвращать это поле. Для валидации сначала обрежьте входные данные и проверьте, пусты ли они.`,
			hint2: `Используйте strings.Count для проверки ровно одного "@", затем strings.Split для разделения локальной и доменной частей. Убедитесь, что обе части существуют и не пусты после обрезки. Верните Email{} с ошибкой, если какая-либо валидация не прошла.`,
			whyItMatters: `Валидация в конструкторе - это фундаментальная практика для создания надёжного, поддерживаемого кода, который предотвращает ошибки до их возникновения.

**Почему валидация в конструкторе:**
- **Быстрый отказ:** Ловите невалидные данные немедленно при создании
- **Соблюдение инвариантов:** После создания объект всегда в валидном состоянии
- **Чёткий контракт:** Успех конструктора гарантирует валидность
- **Без защитного программирования:** Не нужно проверять валидность в каждом методе
- **Лучшие сообщения об ошибках:** Ошибки валидации при создании легче отлаживать

**Production примеры:**

\`\`\`go
// Пример 1: Валидация URL
type URL struct {
	scheme string
	host   string
	path   string
}

func NewURL(rawURL string) (URL, error) {
	if strings.TrimSpace(rawURL) == "" {
		return URL{}, fmt.Errorf("URL не может быть пустым")
	}

	parsed, err := url.Parse(rawURL)
	if err != nil {
		return URL{}, fmt.Errorf("невалидный URL: %w", err)
	}

	if parsed.Scheme == "" {
		return URL{}, fmt.Errorf("URL должен иметь схему (http/https)")
	}

	if parsed.Host == "" {
		return URL{}, fmt.Errorf("URL должен иметь хост")
	}

	return URL{
		scheme: parsed.Scheme,
		host:   parsed.Host,
		path:   parsed.Path,
	}, nil
}
\`\`\`

\`\`\`go
// Пример 2: Валидация номера порта
type Port struct {
	number int
}

func NewPort(port int) (Port, error) {
	if port < 1 || port > 65535 {
		return Port{}, fmt.Errorf("порт должен быть между 1 и 65535, получено %d", port)
	}

	return Port{number: port}, nil
}

func (p Port) Number() int {
	return p.number
}
\`\`\`

\`\`\`go
// Пример 3: Валидация кредитной карты
type CreditCard struct {
	number     string
	expiryDate time.Time
	cvv        string
}

func NewCreditCard(number, cvv string, expiry time.Time) (CreditCard, error) {
	// Удалить пробелы и тире
	number = strings.ReplaceAll(strings.ReplaceAll(number, " ", ""), "-", "")

	if len(number) < 13 || len(number) > 19 {
		return CreditCard{}, fmt.Errorf("неверная длина номера карты")
	}

	if !luhnCheck(number) {
		return CreditCard{}, fmt.Errorf("неверный номер карты (проверка Luhn не прошла)")
	}

	if len(cvv) < 3 || len(cvv) > 4 {
		return CreditCard{}, fmt.Errorf("CVV должен быть 3 или 4 цифры")
	}

	if expiry.Before(time.Now()) {
		return CreditCard{}, fmt.Errorf("срок действия карты истёк")
	}

	return CreditCard{
		number:     number,
		cvv:        cvv,
		expiryDate: expiry,
	}, nil
}
\`\`\`

\`\`\`go
// Пример 4: Валидация имени пользователя
type Username struct {
	value string
}

func NewUsername(username string) (Username, error) {
	trimmed := strings.TrimSpace(username)

	if len(trimmed) < 3 {
		return Username{}, fmt.Errorf("имя пользователя должно быть минимум 3 символа")
	}

	if len(trimmed) > 20 {
		return Username{}, fmt.Errorf("имя пользователя должно быть максимум 20 символов")
	}

	// Разрешить только буквенно-цифровые символы и подчёркивание
	if !regexp.MustCompile(\`^[a-zA-Z0-9_]+$\`).MatchString(trimmed) {
		return Username{}, fmt.Errorf("имя пользователя может содержать только буквы, цифры и подчёркивания")
	}

	return Username{value: trimmed}, nil
}

func (u Username) String() string {
	return u.value
}
\`\`\`

\`\`\`go
// Пример 5: Валидация денежной суммы
type Money struct {
	amount   int64  // Хранить в центах для избежания проблем с плавающей точкой
	currency string
}

func NewMoney(amount float64, currency string) (Money, error) {
	if amount < 0 {
		return Money{}, fmt.Errorf("сумма не может быть отрицательной")
	}

	currency = strings.ToUpper(strings.TrimSpace(currency))
	if len(currency) != 3 {
		return Money{}, fmt.Errorf("валюта должна быть 3-буквенным ISO кодом")
	}

	// Конвертировать в центы
	cents := int64(amount * 100)

	return Money{
		amount:   cents,
		currency: currency,
	}, nil
}
\`\`\`

**Сравнение: Валидация в конструкторе против методов:**

Без валидации в конструкторе (проблематично):
\`\`\`go
type Email struct {
	Address string // Публичное - может быть изменено
}

// Каждый метод должен валидировать!
func (e Email) Send(message string) error {
	if !isValidEmail(e.Address) {
		return fmt.Errorf("невалидный email")
	}
	// ... логика отправки
}

func (e Email) Domain() string {
	if !isValidEmail(e.Address) {
		return ""
	}
	// ... извлечь домен
}
\`\`\`

С валидацией в конструкторе (чисто):
\`\`\`go
type Email struct {
	address string // Приватное
}

func NewEmail(address string) (Email, error) {
	if !isValidEmail(address) {
		return Email{}, fmt.Errorf("невалидный email")
	}
	return Email{address: address}, nil
}

// Методы могут предполагать валидность!
func (e Email) Send(message string) error {
	// Валидация не нужна - email всегда валидный
	// ... логика отправки
}

func (e Email) Domain() string {
	// Валидация не нужна - email всегда валидный
	// ... извлечь домен
}
\`\`\`

**Real-World преимущества:**
- **gRPC:** Валидаторы protocol buffer обеспечивают валидность сообщений
- **Модели БД:** ORM валидируют модели перед персистентностью
- **HTTP клиенты:** Валидация URL и запросов при создании
- **Конфигурация:** Валидация объектов конфигурации при запуске

**Ключевые принципы:**
- Приватные поля предотвращают внешнюю модификацию
- Валидация происходит один раз при создании
- Методы предполагают что объект валидный
- Чёткие сообщения об ошибках направляют пользователей
- Нулевые значения должны быть невалидными (заставлять использовать конструктор)

Валидация в конструкторе - это основа защитного программирования в Go. Она предотвращает целые классы ошибок, гарантируя, что невалидные объекты никогда не существуют.`,
			solutionCode: `package structinit

import (
	"fmt"
	"strings"
)

type Email struct {
	address string                               // Приватное поле для хранения адреса email
}

func (e Email) String() string {
	return e.address                             // Вернуть адрес электронной почты
}

func NewEmail(address string) (Email, error) {
	trimmed := strings.TrimSpace(address)        // Удалить начальные/конечные пробелы

	if trimmed == "" {                           // Проверить пустой ли email
		return Email{}, fmt.Errorf("email cannot be empty")
	}

	if strings.Count(trimmed, "@") != 1 {        // Проверить ровно один символ @
		return Email{}, fmt.Errorf("invalid email format")
	}

	parts := strings.Split(trimmed, "@")         // Разделить на локальную и доменную части

	if strings.TrimSpace(parts[0]) == "" {       // Проверить существование локальной части
		return Email{}, fmt.Errorf("invalid email format")
	}

	if strings.TrimSpace(parts[1]) == "" {       // Проверить существование доменной части
		return Email{}, fmt.Errorf("invalid email format")
	}

	return Email{address: trimmed}, nil          // Вернуть валидный объект Email
}`
		},
		uz: {
			title: 'Konstruktorda validatsiya',
			description: `Ob'ekt yaratish vaqtida ma'lumotlar yaxlitligini ta'minlash uchun to'liq validatsiya bilan konstruktor amalga oshiring.

**Paradigma haqida:**
Go tilida konstruktorlar to'g'ri ob'ektlarni yaratish uchun javobgardir. Ba'zi tillarda validatsiya setterlar ichida sodir bo'lishi mumkin bo'lsa-da, Go barcha ma'lumotlarni konstruktorda oldindan validatsiyadan o'tkazishni va ma'lumotlar noto'g'ri bo'lsa xato qaytarishni rag'batlantiradi. Bu ob'ekt yaratilgandan keyin har doim to'g'ri holatda bo'lishini kafolatlaydi.

**Asosiy komponentlar:**

1. **Email Strukturasi** - validatsiyadan o'tkazilgan email manzilini ifodalaydi:
   \`\`\`go
   type Email struct {
       address string  // Xususiy maydon - faqat konstruktor orqali o'rnatiladi
   }
   \`\`\`

2. **String() Metodi** - xususiy manzil uchun getter:
   \`\`\`go
   func (e Email) String() string
   \`\`\`
   2.1. Email manzilini satr sifatida qaytaradi
   2.2. Xususiy maydonga o'qish ruxsatini beradi

3. **NewEmail Konstruktori** - validatsiya bilan Email yaratadi:
   \`\`\`go
   func NewEmail(address string) (Email, error)
   \`\`\`
   3.1. Email bo'sh emasligini tekshiradi (bo'sh joylarni tozalagandan keyin)
   3.2. Email aniq bitta "@" belgisini o'z ichiga olganligini tekshiradi
   3.3. Email "@" dan oldin kamida bitta belgiga ega ekanligini tekshiradi
   3.4. Email "@" dan keyin kamida bitta belgiga ega ekanligini tekshiradi
   3.5. Validatsiya muvaffaqiyatsiz bo'lsa Email{} va xato qaytaradi
   3.6. To'g'ri bo'lsa to'ldirilgan Email{address: address} va nil qaytaradi

**Validatsiya qoidalari:**

1. **Bo'sh emas:** \`strings.TrimSpace(address) != ""\`
2. **@ ni o'z ichiga oladi:** \`strings.Count(address, "@") == 1\`
3. **Mahalliy qism:** @ dan oldingi belgilar mavjud bo'lishi kerak (len > 0)
4. **Domen qismi:** @ dan keyingi belgilar mavjud bo'lishi kerak (len > 0)

**Amalga oshirish strategiyasi:**
- Mahalliy va domen qismlarini ajratish uchun \`strings.Split(address, "@")\` dan foydalaning
- Natijada hosil bo'lgan slaysda aniq 2 ta element borligini tekshiring
- Ikkala qism ham tozalangandan keyin bo'sh emasligini validatsiya qiling
- Har bir validatsiya muvaffaqiyatsizligi uchun tavsiflovchi xato xabarlarini qaytaring

**Foydalanish misoli:**
\`\`\`go
// To'g'ri emaillar
email, err := NewEmail("user@example.com")
if err == nil {
    fmt.Println(email.String()) // "user@example.com"
}

// Noto'g'ri emaillar - hammasi xato qaytaradi
email, err = NewEmail("")                // Xato: "email cannot be empty"
email, err = NewEmail("notanemail")      // Xato: "invalid email format"
email, err = NewEmail("@example.com")    // Xato: "invalid email format"
\`\`\`

**Nima uchun bu paradigma:**
- Yaratilgandan boshlab ob'ekt invariantlarini kafolatlaydi
- Noto'g'ri holatning mavjud bo'lishini oldini oladi
- Xatolarni topishni osonlashtiradi (yaratishda tez muvaffaqiyatsizlik)
- Aniq shartnoma: konstruktor muvaffaqiyatli bo'lsa, ob'ekt to'g'ri
- Har bir metod chaqiruvida validatsiya qilish shart emas`,
			hint1: `Xususiy 'address' maydonli Email strukturasini aniqlashdan boshlang. String() metodi oddiy ravishda bu maydonni qaytarishi kerak. Validatsiya uchun avval kirishni kesing va bo'sh ekanligini tekshiring.`,
			hint2: `Aniq bitta "@" ni tekshirish uchun strings.Count dan foydalaning, keyin mahalliy va domen qismlarini ajratish uchun strings.Split dan foydalaning. Ikkala qism ham mavjud va kesishdan keyin bo'sh emasligiga ishonch hosil qiling. Agar biron bir validatsiya muvaffaqiyatsiz bo'lsa, xato bilan Email{} qaytaring.`,
			whyItMatters: `Konstruktorda validatsiya bu xatolar sodir bo'lishidan oldin ularni oldini oladigan ishonchli, saqlash mumkin bo'lgan kod yaratish uchun asosiy amaliyotdir.

**Nima uchun konstruktorda validatsiya:**
- **Tez muvaffaqiyatsizlik:** Noto'g'ri ma'lumotlarni yaratish vaqtida darhol ushlash
- **Invariantlarni majburlash:** Yaratilgandan keyin ob'ekt har doim to'g'ri holatda
- **Aniq shartnoma:** Konstruktor muvaffaqiyati to'g'rilikni kafolatlaydi
- **Himoya dasturlashsiz:** Har bir metodda to'g'rilikni tekshirish shart emas
- **Yaxshi xato xabarlari:** Yaratishdagi validatsiya xatolarini debug qilish osonroq

**Production misollari:**

\`\`\`go
// Misol 1: URL validatsiyasi
type URL struct {
	scheme string
	host   string
	path   string
}

func NewURL(rawURL string) (URL, error) {
	if strings.TrimSpace(rawURL) == "" {
		return URL{}, fmt.Errorf("URL bo'sh bo'lishi mumkin emas")
	}

	parsed, err := url.Parse(rawURL)
	if err != nil {
		return URL{}, fmt.Errorf("noto'g'ri URL: %w", err)
	}

	if parsed.Scheme == "" {
		return URL{}, fmt.Errorf("URL sxemaga ega bo'lishi kerak (http/https)")
	}

	if parsed.Host == "" {
		return URL{}, fmt.Errorf("URL hostga ega bo'lishi kerak")
	}

	return URL{
		scheme: parsed.Scheme,
		host:   parsed.Host,
		path:   parsed.Path,
	}, nil
}
\`\`\`

\`\`\`go
// Misol 2: Port raqami validatsiyasi
type Port struct {
	number int
}

func NewPort(port int) (Port, error) {
	if port < 1 || port > 65535 {
		return Port{}, fmt.Errorf("port 1 va 65535 orasida bo'lishi kerak, olindi %d", port)
	}

	return Port{number: port}, nil
}

func (p Port) Number() int {
	return p.number
}
\`\`\`

\`\`\`go
// Misol 3: Kredit karta validatsiyasi
type CreditCard struct {
	number     string
	expiryDate time.Time
	cvv        string
}

func NewCreditCard(number, cvv string, expiry time.Time) (CreditCard, error) {
	// Bo'sh joylar va chiziqlarni olib tashlash
	number = strings.ReplaceAll(strings.ReplaceAll(number, " ", ""), "-", "")

	if len(number) < 13 || len(number) > 19 {
		return CreditCard{}, fmt.Errorf("noto'g'ri karta raqami uzunligi")
	}

	if !luhnCheck(number) {
		return CreditCard{}, fmt.Errorf("noto'g'ri karta raqami (Luhn tekshiruvi muvaffaqiyatsiz)")
	}

	if len(cvv) < 3 || len(cvv) > 4 {
		return CreditCard{}, fmt.Errorf("CVV 3 yoki 4 raqam bo'lishi kerak")
	}

	if expiry.Before(time.Now()) {
		return CreditCard{}, fmt.Errorf("karta muddati tugagan")
	}

	return CreditCard{
		number:     number,
		cvv:        cvv,
		expiryDate: expiry,
	}, nil
}
\`\`\`

\`\`\`go
// Misol 4: Foydalanuvchi nomi validatsiyasi
type Username struct {
	value string
}

func NewUsername(username string) (Username, error) {
	trimmed := strings.TrimSpace(username)

	if len(trimmed) < 3 {
		return Username{}, fmt.Errorf("foydalanuvchi nomi kamida 3 ta belgi bo'lishi kerak")
	}

	if len(trimmed) > 20 {
		return Username{}, fmt.Errorf("foydalanuvchi nomi ko'pi bilan 20 ta belgi bo'lishi kerak")
	}

	// Faqat harf-raqam va pastki chiziqqa ruxsat berish
	if !regexp.MustCompile(\`^[a-zA-Z0-9_]+$\`).MatchString(trimmed) {
		return Username{}, fmt.Errorf("foydalanuvchi nomi faqat harflar, raqamlar va pastki chiziqlarni o'z ichiga olishi mumkin")
	}

	return Username{value: trimmed}, nil
}

func (u Username) String() string {
	return u.value
}
\`\`\`

\`\`\`go
// Misol 5: Pul miqdori validatsiyasi
type Money struct {
	amount   int64  // Suzuvchi nuqta muammolaridan qochish uchun sentlarda saqlash
	currency string
}

func NewMoney(amount float64, currency string) (Money, error) {
	if amount < 0 {
		return Money{}, fmt.Errorf("miqdor manfiy bo'lishi mumkin emas")
	}

	currency = strings.ToUpper(strings.TrimSpace(currency))
	if len(currency) != 3 {
		return Money{}, fmt.Errorf("valyuta 3 harfli ISO kodi bo'lishi kerak")
	}

	// Sentlarga aylantirish
	cents := int64(amount * 100)

	return Money{
		amount:   cents,
		currency: currency,
	}, nil
}
\`\`\`

**Taqqoslash: Konstruktorda validatsiya vs metodlarda:**

Konstruktorda validatsiyasiz (muammoli):
\`\`\`go
type Email struct {
	Address string // Ommaviy - o'zgartirilishi mumkin
}

// Har bir metod validatsiya qilishi kerak!
func (e Email) Send(message string) error {
	if !isValidEmail(e.Address) {
		return fmt.Errorf("noto'g'ri email")
	}
	// ... yuborish mantiqi
}

func (e Email) Domain() string {
	if !isValidEmail(e.Address) {
		return ""
	}
	// ... domenni ajratib olish
}
\`\`\`

Konstruktorda validatsiya bilan (toza):
\`\`\`go
type Email struct {
	address string // Xususiy
}

func NewEmail(address string) (Email, error) {
	if !isValidEmail(address) {
		return Email{}, fmt.Errorf("noto'g'ri email")
	}
	return Email{address: address}, nil
}

// Metodlar to'g'rilikni taxmin qilishi mumkin!
func (e Email) Send(message string) error {
	// Validatsiya kerak emas - email har doim to'g'ri
	// ... yuborish mantiqi
}

func (e Email) Domain() string {
	// Validatsiya kerak emas - email har doim to'g'ri
	// ... domenni ajratib olish
}
\`\`\`

**Haqiqiy dunyo foydalari:**
- **gRPC:** Protokol bufer validatorlari xabar to'g'riligini ta'minlaydi
- **DB modellari:** ORMlar persistentsiya oldidan modellarni validatsiyadan o'tkazadi
- **HTTP kliyentlari:** URL va so'rov validatsiyasi yaratish vaqtida
- **Konfiguratsiya:** Ishga tushirishda konfiguratsiya ob'ektlarini validatsiya qilish

**Asosiy tamoyillar:**
- Xususiy maydonlar tashqi modifikatsiyaning oldini oladi
- Validatsiya yaratilganda bir marta sodir bo'ladi
- Metodlar ob'ekt to'g'ri ekanligini taxmin qiladi
- Aniq xato xabarlari foydalanuvchilarni yo'naltiradi
- Nol qiymatlar noto'g'ri bo'lishi kerak (konstruktordan foydalanishga majburlash)

Konstruktorda validatsiya Go da himoya dasturlashning asosi. U noto'g'ri ob'ektlar hech qachon mavjud bo'lmasligini kafolatlab, butun xatolar sinflarining oldini oladi.`,
			solutionCode: `package structinit

import (
	"fmt"
	"strings"
)

type Email struct {
	address string                               // Email manzilini saqlash uchun xususiy maydon
}

func (e Email) String() string {
	return e.address                             // Email manzilini qaytarish
}

func NewEmail(address string) (Email, error) {
	trimmed := strings.TrimSpace(address)        // Boshidagi/oxiridagi bo'sh joylarni olib tashlash

	if trimmed == "" {                           // Email bo'sh ekanligini tekshirish
		return Email{}, fmt.Errorf("email cannot be empty")
	}

	if strings.Count(trimmed, "@") != 1 {        // Aniq bitta @ belgisini tekshirish
		return Email{}, fmt.Errorf("invalid email format")
	}

	parts := strings.Split(trimmed, "@")         // Mahalliy va domen qismlariga ajratish

	if strings.TrimSpace(parts[0]) == "" {       // Mahalliy qism mavjudligini tekshirish
		return Email{}, fmt.Errorf("invalid email format")
	}

	if strings.TrimSpace(parts[1]) == "" {       // Domen qismi mavjudligini tekshirish
		return Email{}, fmt.Errorf("invalid email format")
	}

	return Email{address: trimmed}, nil          // To'g'ri Email ob'ektini qaytarish
}`
		}
	}
};

export default task;
