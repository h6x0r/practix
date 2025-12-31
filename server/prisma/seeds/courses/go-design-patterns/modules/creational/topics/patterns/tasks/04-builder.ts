import { Task } from '../../../../../../../types';

export const task: Task = {
	slug: 'go-dp-builder',
	title: 'Builder Pattern',
	difficulty: 'medium',
	tags: ['go', 'design-patterns', 'creational', 'builder'],
	estimatedTime: '40m',
	isPremium: false,
	youtubeUrl: '',
	description: `Implement the Builder pattern in Go - separate the construction of a complex object from its representation using a fluent interface.

**You will implement:**

1. **House struct** - Complex object with multiple fields
2. **HouseBuilder struct** - Builds House step by step
3. **Fluent interface** - Each setter returns *HouseBuilder
4. **Build method** - Returns the final House

**Example Usage:**

\`\`\`go
// Create a new builder
builder := NewHouseBuilder()

// Build house step by step using fluent interface
house := builder.
    SetFoundation("concrete").
    SetWalls("brick").
    SetRoof("tile").
    SetGarage(true).
    SetSwimmingPool(true).
    SetGarden(false).
    Build()

// Access the built house
fmt.Println(house.Foundation)      // "concrete"
fmt.Println(house.HasGarage)       // true
fmt.Println(house.HasSwimmingPool) // true

// Build different house with same builder pattern
simpleHouse := NewHouseBuilder().
    SetFoundation("wood").
    SetWalls("wood").
    SetRoof("shingle").
    Build()
\`\`\``,
	initialCode: `package patterns

type House struct {
	Foundation   string
	Walls        string
	Roof         string
	HasGarage    bool
	HasSwimmingPool bool
	HasGarden    bool
}

type HouseBuilder struct {
	house *House
}

func NewHouseBuilder() *HouseBuilder {
}

func (b *HouseBuilder) SetFoundation(foundation string) *HouseBuilder {
}

func (b *HouseBuilder) SetWalls(walls string) *HouseBuilder {
}

func (b *HouseBuilder) SetRoof(roof string) *HouseBuilder {
}

func (b *HouseBuilder) SetGarage(hasGarage bool) *HouseBuilder {
}

func (b *HouseBuilder) SetSwimmingPool(hasPool bool) *HouseBuilder {
}

func (b *HouseBuilder) SetGarden(hasGarden bool) *HouseBuilder {
}

func (b *HouseBuilder) Build() *House {
}`,
	solutionCode: `package patterns

type House struct {	// Product - the complex object being built
	Foundation      string	// required: type of foundation
	Walls           string	// required: type of walls
	Roof            string	// required: type of roof
	HasGarage       bool	// optional: whether house has garage
	HasSwimmingPool bool	// optional: whether house has pool
	HasGarden       bool	// optional: whether house has garden
}

type HouseBuilder struct {	// Builder - constructs Product step by step
	house *House	// holds the Product being built
}

func NewHouseBuilder() *HouseBuilder {	// creates new Builder instance
	return &HouseBuilder{house: &House{}}	// initialize with empty House
}

func (b *HouseBuilder) SetFoundation(foundation string) *HouseBuilder {	// sets foundation type
	b.house.Foundation = foundation	// assign value to product
	return b	// return builder for method chaining (fluent interface)
}

func (b *HouseBuilder) SetWalls(walls string) *HouseBuilder {	// sets wall type
	b.house.Walls = walls	// assign value to product
	return b	// return builder for method chaining
}

func (b *HouseBuilder) SetRoof(roof string) *HouseBuilder {	// sets roof type
	b.house.Roof = roof	// assign value to product
	return b	// return builder for method chaining
}

func (b *HouseBuilder) SetGarage(hasGarage bool) *HouseBuilder {	// sets garage option
	b.house.HasGarage = hasGarage	// assign boolean value
	return b	// return builder for method chaining
}

func (b *HouseBuilder) SetSwimmingPool(hasPool bool) *HouseBuilder {	// sets pool option
	b.house.HasSwimmingPool = hasPool	// assign boolean value
	return b	// return builder for method chaining
}

func (b *HouseBuilder) SetGarden(hasGarden bool) *HouseBuilder {	// sets garden option
	b.house.HasGarden = hasGarden	// assign boolean value
	return b	// return builder for method chaining
}

func (b *HouseBuilder) Build() *House {	// finalizes and returns the Product
	return b.house	// return the constructed House
}`,
	hint1: `Each Set method assigns the value to b.house field and returns b (the builder itself) for method chaining. This pattern is called "fluent interface" and allows writing builder.SetA().SetB().SetC().Build().`,
	hint2: `Build simply returns b.house. The fluent interface is achieved by returning *HouseBuilder from each setter. Each setter modifies the internal house object and returns the same builder instance.`,
	whyItMatters: `**1. Why Builder Exists**

Builder pattern solves the "telescoping constructor" problem - when a class needs many parameters, especially optional ones. Instead of multiple constructors or constructors with many parameters, Builder provides a step-by-step way to construct objects.

**The Problem It Solves:**

\`\`\`go
// WITHOUT Builder - telescoping constructor anti-pattern
func NewHouse(foundation, walls, roof string, hasGarage, hasPool, hasGarden, hasSauna, hasWineCellar bool) *House {
    // Which bool is which? Easy to make mistakes!
    return &House{...}
}

// Calling code is unclear:
house := NewHouse("concrete", "brick", "tile", true, false, true, false, false)
// What do these booleans mean? Impossible to read!

// Or multiple constructors - doesn't scale
func NewSimpleHouse(foundation, walls, roof string) *House
func NewHouseWithGarage(foundation, walls, roof string, hasGarage bool) *House
func NewHouseWithGarageAndPool(foundation, walls, roof string, hasGarage, hasPool bool) *House
// And so on... 2^n combinations!
\`\`\`

**WITH Builder:**

\`\`\`go
// Clear, readable, self-documenting code
house := NewHouseBuilder().
    SetFoundation("concrete").
    SetWalls("brick").
    SetRoof("tile").
    SetGarage(true).
    SetSwimmingPool(false).
    SetGarden(true).
    Build()

// Each method name documents what it sets
// Order doesn't matter for optional fields
// Easy to add new options without breaking existing code
\`\`\`

**2. Real-World Examples in Go**

**HTTP Request Builder (like Go's http package):**

\`\`\`go
// Standard library uses builder-like pattern
req, _ := http.NewRequest("GET", url, nil)
req.Header.Set("Content-Type", "application/json")
req.Header.Set("Authorization", "Bearer token")

// Custom fluent builder
type RequestBuilder struct {
    method  string
    url     string
    headers map[string]string
    body    io.Reader
    timeout time.Duration
}

func NewRequest() *RequestBuilder {
    return &RequestBuilder{
        headers: make(map[string]string),
        timeout: 30 * time.Second,
    }
}

func (rb *RequestBuilder) Method(m string) *RequestBuilder {
    rb.method = m
    return rb
}

func (rb *RequestBuilder) URL(u string) *RequestBuilder {
    rb.url = u
    return rb
}

func (rb *RequestBuilder) Header(key, value string) *RequestBuilder {
    rb.headers[key] = value
    return rb
}

func (rb *RequestBuilder) Build() (*http.Request, error) {
    req, err := http.NewRequest(rb.method, rb.url, rb.body)
    if err != nil {
        return nil, err
    }
    for k, v := range rb.headers {
        req.Header.Set(k, v)
    }
    return req, nil
}
\`\`\`

**SQL Query Builder:**

\`\`\`go
type QueryBuilder struct {
    table      string
    columns    []string
    conditions []string
    orderBy    string
    limit      int
}

func Select(columns ...string) *QueryBuilder {
    return &QueryBuilder{columns: columns}
}

func (qb *QueryBuilder) From(table string) *QueryBuilder {
    qb.table = table
    return qb
}

func (qb *QueryBuilder) Where(condition string) *QueryBuilder {
    qb.conditions = append(qb.conditions, condition)
    return qb
}

func (qb *QueryBuilder) OrderBy(column string) *QueryBuilder {
    qb.orderBy = column
    return qb
}

func (qb *QueryBuilder) Limit(n int) *QueryBuilder {
    qb.limit = n
    return qb
}

// Usage:
query := Select("id", "name", "email").
    From("users").
    Where("age > 18").
    Where("active = true").
    OrderBy("created_at DESC").
    Limit(10).
    Build()
// SELECT id, name, email FROM users WHERE age > 18 AND active = true ORDER BY created_at DESC LIMIT 10
\`\`\`

**3. Production Pattern with Validation**

\`\`\`go
package config

import (
    "errors"
    "time"
)

type ServerConfig struct {
    Host         string
    Port         int
    ReadTimeout  time.Duration
    WriteTimeout time.Duration
    MaxConns     int
    TLS          bool
    CertFile     string
    KeyFile      string
}

type ServerConfigBuilder struct {
    config *ServerConfig
    errors []error
}

func NewServerConfig() *ServerConfigBuilder {
    return &ServerConfigBuilder{
        config: &ServerConfig{
            Host:         "localhost",
            Port:         8080,
            ReadTimeout:  30 * time.Second,
            WriteTimeout: 30 * time.Second,
            MaxConns:     100,
        },
    }
}

func (b *ServerConfigBuilder) Host(host string) *ServerConfigBuilder {
    if host == "" {
        b.errors = append(b.errors, errors.New("host cannot be empty"))
        return b
    }
    b.config.Host = host
    return b
}

func (b *ServerConfigBuilder) Port(port int) *ServerConfigBuilder {
    if port < 1 || port > 65535 {
        b.errors = append(b.errors, errors.New("port must be between 1 and 65535"))
        return b
    }
    b.config.Port = port
    return b
}

func (b *ServerConfigBuilder) WithTLS(certFile, keyFile string) *ServerConfigBuilder {
    if certFile == "" || keyFile == "" {
        b.errors = append(b.errors, errors.New("TLS requires both cert and key files"))
        return b
    }
    b.config.TLS = true
    b.config.CertFile = certFile
    b.config.KeyFile = keyFile
    return b
}

func (b *ServerConfigBuilder) Build() (*ServerConfig, error) {
    if len(b.errors) > 0 {
        return nil, errors.Join(b.errors...)
    }
    return b.config, nil
}

// Usage:
config, err := NewServerConfig().
    Host("api.example.com").
    Port(443).
    WithTLS("/path/to/cert.pem", "/path/to/key.pem").
    Build()
\`\`\`

**4. Common Mistakes to Avoid**

\`\`\`go
// MISTAKE 1: Forgetting to return builder (breaks chaining)
func (b *HouseBuilder) SetWalls(walls string) {  // Wrong! No return
    b.house.Walls = walls
    // Cannot chain: builder.SetWalls("brick").SetRoof("tile")
}

// MISTAKE 2: Returning new builder instead of same instance
func (b *HouseBuilder) SetWalls(walls string) *HouseBuilder {
    return &HouseBuilder{  // Wrong! Creates new builder
        house: &House{Walls: walls},
    }
}

// MISTAKE 3: Not initializing the product
func NewHouseBuilder() *HouseBuilder {
    return &HouseBuilder{}  // Wrong! house is nil
}

// MISTAKE 4: Modifying product after Build (builder should be single-use)
builder := NewHouseBuilder()
house1 := builder.SetWalls("brick").Build()
house2 := builder.SetWalls("wood").Build()  // Oops! Also changes house1!

// CORRECT: Create new builder for each product
house1 := NewHouseBuilder().SetWalls("brick").Build()
house2 := NewHouseBuilder().SetWalls("wood").Build()

// MISTAKE 5: Using Builder when simple struct literal works
// Don't use Builder for:
type Point struct {
    X, Y int
}
// Just use: point := Point{X: 10, Y: 20}
// Builder is overkill for simple structs
\`\`\``,
	order: 3,
	testCode: `package patterns

import (
	"testing"
)

// Test1: NewHouseBuilder returns non-nil builder
func Test1(t *testing.T) {
	builder := NewHouseBuilder()
	if builder == nil {
		t.Error("NewHouseBuilder should return non-nil builder")
	}
}

// Test2: SetFoundation sets foundation
func Test2(t *testing.T) {
	house := NewHouseBuilder().SetFoundation("concrete").Build()
	if house.Foundation != "concrete" {
		t.Error("SetFoundation should set foundation")
	}
}

// Test3: SetWalls sets walls
func Test3(t *testing.T) {
	house := NewHouseBuilder().SetWalls("brick").Build()
	if house.Walls != "brick" {
		t.Error("SetWalls should set walls")
	}
}

// Test4: SetRoof sets roof
func Test4(t *testing.T) {
	house := NewHouseBuilder().SetRoof("tile").Build()
	if house.Roof != "tile" {
		t.Error("SetRoof should set roof")
	}
}

// Test5: SetGarage sets garage
func Test5(t *testing.T) {
	house := NewHouseBuilder().SetGarage(true).Build()
	if !house.HasGarage {
		t.Error("SetGarage should set HasGarage")
	}
}

// Test6: SetSwimmingPool sets pool
func Test6(t *testing.T) {
	house := NewHouseBuilder().SetSwimmingPool(true).Build()
	if !house.HasSwimmingPool {
		t.Error("SetSwimmingPool should set HasSwimmingPool")
	}
}

// Test7: SetGarden sets garden
func Test7(t *testing.T) {
	house := NewHouseBuilder().SetGarden(true).Build()
	if !house.HasGarden {
		t.Error("SetGarden should set HasGarden")
	}
}

// Test8: Fluent interface chaining works
func Test8(t *testing.T) {
	house := NewHouseBuilder().
		SetFoundation("concrete").
		SetWalls("brick").
		SetRoof("tile").
		SetGarage(true).
		SetSwimmingPool(false).
		SetGarden(true).
		Build()
	if house.Foundation != "concrete" || house.Walls != "brick" || house.Roof != "tile" {
		t.Error("Fluent chaining should set all values")
	}
}

// Test9: Build returns House
func Test9(t *testing.T) {
	house := NewHouseBuilder().Build()
	if house == nil {
		t.Error("Build should return non-nil House")
	}
}

// Test10: Builder creates independent houses
func Test10(t *testing.T) {
	house1 := NewHouseBuilder().SetFoundation("wood").Build()
	house2 := NewHouseBuilder().SetFoundation("concrete").Build()
	if house1.Foundation == house2.Foundation {
		t.Error("Each Build should create independent house")
	}
}
`,
	translations: {
		ru: {
			title: 'Паттерн Builder (Строитель)',
			description: `Реализуйте паттерн Builder на Go — отделите конструирование сложного объекта от его представления с помощью fluent-интерфейса.

**Вы реализуете:**

1. **Структура House** - Сложный объект с множеством полей
2. **Структура HouseBuilder** - Строит House пошагово
3. **Fluent-интерфейс** - Каждый сеттер возвращает *HouseBuilder
4. **Метод Build** - Возвращает готовый House

**Пример использования:**

\`\`\`go
// Создаём новый builder
builder := NewHouseBuilder()

// Строим дом пошагово используя fluent-интерфейс
house := builder.
    SetFoundation("concrete").
    SetWalls("brick").
    SetRoof("tile").
    SetGarage(true).
    SetSwimmingPool(true).
    SetGarden(false).
    Build()

// Доступ к построенному дому
fmt.Println(house.Foundation)      // "concrete"
fmt.Println(house.HasGarage)       // true
fmt.Println(house.HasSwimmingPool) // true

// Строим другой дом тем же паттерном
simpleHouse := NewHouseBuilder().
    SetFoundation("wood").
    SetWalls("wood").
    SetRoof("shingle").
    Build()
\`\`\``,
			hint1: `Каждый Set-метод присваивает значение полю b.house и возвращает b (сам builder) для цепочки вызовов. Этот паттерн называется "fluent interface" и позволяет писать builder.SetA().SetB().SetC().Build().`,
			hint2: `Build просто возвращает b.house. Fluent-интерфейс достигается возвратом *HouseBuilder из каждого сеттера. Каждый сеттер модифицирует внутренний объект house и возвращает тот же экземпляр builder.`,
			whyItMatters: `**1. Зачем нужен Builder**

Паттерн Builder решает проблему "телескопического конструктора" — когда классу нужно много параметров, особенно опциональных. Вместо множества конструкторов или конструкторов с множеством параметров, Builder предоставляет пошаговый способ конструирования объектов.

**Проблема, которую он решает:**

\`\`\`go
// БЕЗ Builder - анти-паттерн телескопического конструктора
func NewHouse(foundation, walls, roof string, hasGarage, hasPool, hasGarden, hasSauna, hasWineCellar bool) *House {
    // Какой bool что означает? Легко ошибиться!
    return &House{...}
}

// Вызывающий код непонятен:
house := NewHouse("concrete", "brick", "tile", true, false, true, false, false)
// Что означают эти булевы значения? Невозможно прочитать!

// Или множество конструкторов - не масштабируется
func NewSimpleHouse(foundation, walls, roof string) *House
func NewHouseWithGarage(foundation, walls, roof string, hasGarage bool) *House
func NewHouseWithGarageAndPool(foundation, walls, roof string, hasGarage, hasPool bool) *House
// И так далее... 2^n комбинаций!
\`\`\`

**С Builder:**

\`\`\`go
// Понятный, читаемый, самодокументирующийся код
house := NewHouseBuilder().
    SetFoundation("concrete").
    SetWalls("brick").
    SetRoof("tile").
    SetGarage(true).
    SetSwimmingPool(false).
    SetGarden(true).
    Build()

// Каждое имя метода документирует что оно устанавливает
// Порядок не важен для опциональных полей
// Легко добавлять новые опции без нарушения существующего кода
\`\`\`

**2. Примеры из реального мира в Go**

**HTTP Request Builder (как в пакете http Go):**

\`\`\`go
// Стандартная библиотека использует builder-подобный паттерн
req, _ := http.NewRequest("GET", url, nil)
req.Header.Set("Content-Type", "application/json")
req.Header.Set("Authorization", "Bearer token")

// Собственный fluent builder
type RequestBuilder struct {
    method  string
    url     string
    headers map[string]string
    body    io.Reader
    timeout time.Duration
}

func NewRequest() *RequestBuilder {
    return &RequestBuilder{
        headers: make(map[string]string),
        timeout: 30 * time.Second,
    }
}

func (rb *RequestBuilder) Method(m string) *RequestBuilder {
    rb.method = m
    return rb
}

func (rb *RequestBuilder) URL(u string) *RequestBuilder {
    rb.url = u
    return rb
}

func (rb *RequestBuilder) Header(key, value string) *RequestBuilder {
    rb.headers[key] = value
    return rb
}

func (rb *RequestBuilder) Build() (*http.Request, error) {
    req, err := http.NewRequest(rb.method, rb.url, rb.body)
    if err != nil {
        return nil, err
    }
    for k, v := range rb.headers {
        req.Header.Set(k, v)
    }
    return req, nil
}
\`\`\`

**SQL Query Builder:**

\`\`\`go
type QueryBuilder struct {
    table      string
    columns    []string
    conditions []string
    orderBy    string
    limit      int
}

func Select(columns ...string) *QueryBuilder {
    return &QueryBuilder{columns: columns}
}

func (qb *QueryBuilder) From(table string) *QueryBuilder {
    qb.table = table
    return qb
}

func (qb *QueryBuilder) Where(condition string) *QueryBuilder {
    qb.conditions = append(qb.conditions, condition)
    return qb
}

func (qb *QueryBuilder) OrderBy(column string) *QueryBuilder {
    qb.orderBy = column
    return qb
}

func (qb *QueryBuilder) Limit(n int) *QueryBuilder {
    qb.limit = n
    return qb
}

// Использование:
query := Select("id", "name", "email").
    From("users").
    Where("age > 18").
    Where("active = true").
    OrderBy("created_at DESC").
    Limit(10).
    Build()
// SELECT id, name, email FROM users WHERE age > 18 AND active = true ORDER BY created_at DESC LIMIT 10
\`\`\`

**3. Продакшн паттерн с валидацией**

\`\`\`go
package config

import (
    "errors"
    "time"
)

type ServerConfig struct {
    Host         string
    Port         int
    ReadTimeout  time.Duration
    WriteTimeout time.Duration
    MaxConns     int
    TLS          bool
    CertFile     string
    KeyFile      string
}

type ServerConfigBuilder struct {
    config *ServerConfig
    errors []error
}

func NewServerConfig() *ServerConfigBuilder {
    return &ServerConfigBuilder{
        config: &ServerConfig{
            Host:         "localhost",
            Port:         8080,
            ReadTimeout:  30 * time.Second,
            WriteTimeout: 30 * time.Second,
            MaxConns:     100,
        },
    }
}

func (b *ServerConfigBuilder) Host(host string) *ServerConfigBuilder {
    if host == "" {
        b.errors = append(b.errors, errors.New("host cannot be empty"))
        return b
    }
    b.config.Host = host
    return b
}

func (b *ServerConfigBuilder) Port(port int) *ServerConfigBuilder {
    if port < 1 || port > 65535 {
        b.errors = append(b.errors, errors.New("port must be between 1 and 65535"))
        return b
    }
    b.config.Port = port
    return b
}

func (b *ServerConfigBuilder) WithTLS(certFile, keyFile string) *ServerConfigBuilder {
    if certFile == "" || keyFile == "" {
        b.errors = append(b.errors, errors.New("TLS requires both cert and key files"))
        return b
    }
    b.config.TLS = true
    b.config.CertFile = certFile
    b.config.KeyFile = keyFile
    return b
}

func (b *ServerConfigBuilder) Build() (*ServerConfig, error) {
    if len(b.errors) > 0 {
        return nil, errors.Join(b.errors...)
    }
    return b.config, nil
}

// Использование:
config, err := NewServerConfig().
    Host("api.example.com").
    Port(443).
    WithTLS("/path/to/cert.pem", "/path/to/key.pem").
    Build()
\`\`\`

**4. Типичные ошибки**

\`\`\`go
// ОШИБКА 1: Забыли вернуть builder (нарушает цепочку)
func (b *HouseBuilder) SetWalls(walls string) {  // Неправильно! Нет return
    b.house.Walls = walls
    // Нельзя цепочить: builder.SetWalls("brick").SetRoof("tile")
}

// ОШИБКА 2: Возврат нового builder вместо того же экземпляра
func (b *HouseBuilder) SetWalls(walls string) *HouseBuilder {
    return &HouseBuilder{  // Неправильно! Создаёт новый builder
        house: &House{Walls: walls},
    }
}

// ОШИБКА 3: Не инициализировали продукт
func NewHouseBuilder() *HouseBuilder {
    return &HouseBuilder{}  // Неправильно! house равен nil
}

// ОШИБКА 4: Модификация продукта после Build (builder должен быть одноразовым)
builder := NewHouseBuilder()
house1 := builder.SetWalls("brick").Build()
house2 := builder.SetWalls("wood").Build()  // Ой! Также меняет house1!

// ПРАВИЛЬНО: Создавать новый builder для каждого продукта
house1 := NewHouseBuilder().SetWalls("brick").Build()
house2 := NewHouseBuilder().SetWalls("wood").Build()

// ОШИБКА 5: Использование Builder когда достаточно простого литерала структуры
// Не используйте Builder для:
type Point struct {
    X, Y int
}
// Просто используйте: point := Point{X: 10, Y: 20}
// Builder избыточен для простых структур
\`\`\``,
			solutionCode: `package patterns

type House struct {	// Продукт - сложный объект который строим
	Foundation      string	// обязательное: тип фундамента
	Walls           string	// обязательное: тип стен
	Roof            string	// обязательное: тип крыши
	HasGarage       bool	// опционально: есть ли гараж
	HasSwimmingPool bool	// опционально: есть ли бассейн
	HasGarden       bool	// опционально: есть ли сад
}

type HouseBuilder struct {	// Строитель - конструирует Продукт пошагово
	house *House	// хранит строящийся Продукт
}

func NewHouseBuilder() *HouseBuilder {	// создаёт новый экземпляр Builder
	return &HouseBuilder{house: &House{}}	// инициализируем пустым House
}

func (b *HouseBuilder) SetFoundation(foundation string) *HouseBuilder {	// устанавливает тип фундамента
	b.house.Foundation = foundation	// присваиваем значение продукту
	return b	// возвращаем builder для цепочки методов (fluent interface)
}

func (b *HouseBuilder) SetWalls(walls string) *HouseBuilder {	// устанавливает тип стен
	b.house.Walls = walls	// присваиваем значение продукту
	return b	// возвращаем builder для цепочки методов
}

func (b *HouseBuilder) SetRoof(roof string) *HouseBuilder {	// устанавливает тип крыши
	b.house.Roof = roof	// присваиваем значение продукту
	return b	// возвращаем builder для цепочки методов
}

func (b *HouseBuilder) SetGarage(hasGarage bool) *HouseBuilder {	// устанавливает опцию гаража
	b.house.HasGarage = hasGarage	// присваиваем булево значение
	return b	// возвращаем builder для цепочки методов
}

func (b *HouseBuilder) SetSwimmingPool(hasPool bool) *HouseBuilder {	// устанавливает опцию бассейна
	b.house.HasSwimmingPool = hasPool	// присваиваем булево значение
	return b	// возвращаем builder для цепочки методов
}

func (b *HouseBuilder) SetGarden(hasGarden bool) *HouseBuilder {	// устанавливает опцию сада
	b.house.HasGarden = hasGarden	// присваиваем булево значение
	return b	// возвращаем builder для цепочки методов
}

func (b *HouseBuilder) Build() *House {	// финализирует и возвращает Продукт
	return b.house	// возвращаем построенный House
}`
		},
		uz: {
			title: 'Builder (Quruvchi) Pattern',
			description: `Go tilida Builder patternini amalga oshiring — murakkab ob'ekt konstruksiyasini uning tasviridan fluent-interfeys yordamida ajrating.

**Siz amalga oshirasiz:**

1. **House strukturasi** - Ko'p maydonli murakkab ob'ekt
2. **HouseBuilder strukturasi** - House ni bosqichma-bosqich quradi
3. **Fluent-interfeys** - Har bir setter *HouseBuilder qaytaradi
4. **Build metodi** - Tayyor House qaytaradi

**Foydalanish namunasi:**

\`\`\`go
// Yangi builder yaratish
builder := NewHouseBuilder()

// Fluent-interfeys yordamida uyni bosqichma-bosqich qurish
house := builder.
    SetFoundation("concrete").
    SetWalls("brick").
    SetRoof("tile").
    SetGarage(true).
    SetSwimmingPool(true).
    SetGarden(false).
    Build()

// Qurilgan uyga kirish
fmt.Println(house.Foundation)      // "concrete"
fmt.Println(house.HasGarage)       // true
fmt.Println(house.HasSwimmingPool) // true

// Xuddi shu pattern bilan boshqa uy qurish
simpleHouse := NewHouseBuilder().
    SetFoundation("wood").
    SetWalls("wood").
    SetRoof("shingle").
    Build()
\`\`\``,
			hint1: `Har bir Set metodi qiymatni b.house maydoniga tayinlaydi va chaqiruvlar zanjiri uchun b (builder o'zi) ni qaytaradi. Bu pattern "fluent interface" deb ataladi va builder.SetA().SetB().SetC().Build() yozish imkonini beradi.`,
			hint2: `Build oddiy b.house ni qaytaradi. Fluent-interfeys har bir setterdan *HouseBuilder qaytarish orqali erishiladi. Har bir setter ichki house ob'ektini o'zgartiradi va xuddi shu builder instansiyasini qaytaradi.`,
			whyItMatters: `**1. Builder nima uchun kerak**

Builder pattern "teleskopik konstruktor" muammosini hal qiladi — klassga ko'p parametrlar kerak bo'lganda, ayniqsa ixtiyoriy parametrlar. Ko'p konstruktorlar yoki ko'p parametrli konstruktorlar o'rniga, Builder ob'ektlarni bosqichma-bosqich qurish usulini taqdim etadi.

**U hal qiladigan muammo:**

\`\`\`go
// Builder SIZ - teleskopik konstruktor anti-pattern
func NewHouse(foundation, walls, roof string, hasGarage, hasPool, hasGarden, hasSauna, hasWineCellar bool) *House {
    // Qaysi bool nima? Xato qilish oson!
    return &House{...}
}

// Chaqiruvchi kod noaniq:
house := NewHouse("concrete", "brick", "tile", true, false, true, false, false)
// Bu boollar nimani anglatadi? O'qib bo'lmaydi!

// Yoki ko'p konstruktorlar - masshtablanmaydi
func NewSimpleHouse(foundation, walls, roof string) *House
func NewHouseWithGarage(foundation, walls, roof string, hasGarage bool) *House
func NewHouseWithGarageAndPool(foundation, walls, roof string, hasGarage, hasPool bool) *House
// Va hokazo... 2^n kombinatsiyalar!
\`\`\`

**Builder BILAN:**

\`\`\`go
// Aniq, o'qilishi oson, o'z-o'zini hujjatlashtiruvchi kod
house := NewHouseBuilder().
    SetFoundation("concrete").
    SetWalls("brick").
    SetRoof("tile").
    SetGarage(true).
    SetSwimmingPool(false).
    SetGarden(true).
    Build()

// Har bir metod nomi nimani o'rnatishini hujjatlashtiradi
// Ixtiyoriy maydonlar uchun tartib muhim emas
// Mavjud kodni buzmasdan yangi opsiyalar qo'shish oson
\`\`\`

**2. Go'da real hayotiy misollar**

**HTTP Request Builder (Go http paketi kabi):**

\`\`\`go
// Standart kutubxona builder-ga o'xshash patterndan foydalanadi
req, _ := http.NewRequest("GET", url, nil)
req.Header.Set("Content-Type", "application/json")
req.Header.Set("Authorization", "Bearer token")

// Maxsus fluent builder
type RequestBuilder struct {
    method  string
    url     string
    headers map[string]string
    body    io.Reader
    timeout time.Duration
}

func NewRequest() *RequestBuilder {
    return &RequestBuilder{
        headers: make(map[string]string),
        timeout: 30 * time.Second,
    }
}

func (rb *RequestBuilder) Method(m string) *RequestBuilder {
    rb.method = m
    return rb
}

func (rb *RequestBuilder) URL(u string) *RequestBuilder {
    rb.url = u
    return rb
}

func (rb *RequestBuilder) Header(key, value string) *RequestBuilder {
    rb.headers[key] = value
    return rb
}

func (rb *RequestBuilder) Build() (*http.Request, error) {
    req, err := http.NewRequest(rb.method, rb.url, rb.body)
    if err != nil {
        return nil, err
    }
    for k, v := range rb.headers {
        req.Header.Set(k, v)
    }
    return req, nil
}
\`\`\`

**SQL Query Builder:**

\`\`\`go
type QueryBuilder struct {
    table      string
    columns    []string
    conditions []string
    orderBy    string
    limit      int
}

func Select(columns ...string) *QueryBuilder {
    return &QueryBuilder{columns: columns}
}

func (qb *QueryBuilder) From(table string) *QueryBuilder {
    qb.table = table
    return qb
}

func (qb *QueryBuilder) Where(condition string) *QueryBuilder {
    qb.conditions = append(qb.conditions, condition)
    return qb
}

func (qb *QueryBuilder) OrderBy(column string) *QueryBuilder {
    qb.orderBy = column
    return qb
}

func (qb *QueryBuilder) Limit(n int) *QueryBuilder {
    qb.limit = n
    return qb
}

// Foydalanish:
query := Select("id", "name", "email").
    From("users").
    Where("age > 18").
    Where("active = true").
    OrderBy("created_at DESC").
    Limit(10).
    Build()
// SELECT id, name, email FROM users WHERE age > 18 AND active = true ORDER BY created_at DESC LIMIT 10
\`\`\`

**3. Validatsiya bilan production pattern**

\`\`\`go
package config

import (
    "errors"
    "time"
)

type ServerConfig struct {
    Host         string
    Port         int
    ReadTimeout  time.Duration
    WriteTimeout time.Duration
    MaxConns     int
    TLS          bool
    CertFile     string
    KeyFile      string
}

type ServerConfigBuilder struct {
    config *ServerConfig
    errors []error
}

func NewServerConfig() *ServerConfigBuilder {
    return &ServerConfigBuilder{
        config: &ServerConfig{
            Host:         "localhost",
            Port:         8080,
            ReadTimeout:  30 * time.Second,
            WriteTimeout: 30 * time.Second,
            MaxConns:     100,
        },
    }
}

func (b *ServerConfigBuilder) Host(host string) *ServerConfigBuilder {
    if host == "" {
        b.errors = append(b.errors, errors.New("host cannot be empty"))
        return b
    }
    b.config.Host = host
    return b
}

func (b *ServerConfigBuilder) Port(port int) *ServerConfigBuilder {
    if port < 1 || port > 65535 {
        b.errors = append(b.errors, errors.New("port must be between 1 and 65535"))
        return b
    }
    b.config.Port = port
    return b
}

func (b *ServerConfigBuilder) WithTLS(certFile, keyFile string) *ServerConfigBuilder {
    if certFile == "" || keyFile == "" {
        b.errors = append(b.errors, errors.New("TLS requires both cert and key files"))
        return b
    }
    b.config.TLS = true
    b.config.CertFile = certFile
    b.config.KeyFile = keyFile
    return b
}

func (b *ServerConfigBuilder) Build() (*ServerConfig, error) {
    if len(b.errors) > 0 {
        return nil, errors.Join(b.errors...)
    }
    return b.config, nil
}

// Foydalanish:
config, err := NewServerConfig().
    Host("api.example.com").
    Port(443).
    WithTLS("/path/to/cert.pem", "/path/to/key.pem").
    Build()
\`\`\`

**4. Keng tarqalgan xatolar**

\`\`\`go
// XATO 1: Builder qaytarishni unutish (zanjirni buzadi)
func (b *HouseBuilder) SetWalls(walls string) {  // Noto'g'ri! Return yo'q
    b.house.Walls = walls
    // Zanjirlab bo'lmaydi: builder.SetWalls("brick").SetRoof("tile")
}

// XATO 2: Xuddi shu instansiya o'rniga yangi builder qaytarish
func (b *HouseBuilder) SetWalls(walls string) *HouseBuilder {
    return &HouseBuilder{  // Noto'g'ri! Yangi builder yaratadi
        house: &House{Walls: walls},
    }
}

// XATO 3: Mahsulotni initsializatsiya qilmaslik
func NewHouseBuilder() *HouseBuilder {
    return &HouseBuilder{}  // Noto'g'ri! house nil
}

// XATO 4: Build dan keyin mahsulotni o'zgartirish (builder bir martalik bo'lishi kerak)
builder := NewHouseBuilder()
house1 := builder.SetWalls("brick").Build()
house2 := builder.SetWalls("wood").Build()  // Xato! house1 ni ham o'zgartiradi!

// TO'G'RI: Har bir mahsulot uchun yangi builder yaratish
house1 := NewHouseBuilder().SetWalls("brick").Build()
house2 := NewHouseBuilder().SetWalls("wood").Build()

// XATO 5: Oddiy struct literal yetarli bo'lganda Builder ishlatish
// Builder ishlatmang:
type Point struct {
    X, Y int
}
// Shunchaki ishlating: point := Point{X: 10, Y: 20}
// Oddiy strukturalar uchun Builder ortiqcha
\`\`\``,
			solutionCode: `package patterns

type House struct {	// Mahsulot - qurilayotgan murakkab ob'ekt
	Foundation      string	// majburiy: poydevor turi
	Walls           string	// majburiy: devor turi
	Roof            string	// majburiy: tom turi
	HasGarage       bool	// ixtiyoriy: garaj bormi
	HasSwimmingPool bool	// ixtiyoriy: basseyn bormi
	HasGarden       bool	// ixtiyoriy: bog' bormi
}

type HouseBuilder struct {	// Quruvchi - Mahsulotni bosqichma-bosqich quradi
	house *House	// qurilayotgan Mahsulotni saqlaydi
}

func NewHouseBuilder() *HouseBuilder {	// yangi Builder instansiyasini yaratadi
	return &HouseBuilder{house: &House{}}	// bo'sh House bilan initsializatsiya
}

func (b *HouseBuilder) SetFoundation(foundation string) *HouseBuilder {	// poydevor turini o'rnatadi
	b.house.Foundation = foundation	// mahsulotga qiymat tayinlaydi
	return b	// metod zanjiri uchun builder qaytaradi (fluent interface)
}

func (b *HouseBuilder) SetWalls(walls string) *HouseBuilder {	// devor turini o'rnatadi
	b.house.Walls = walls	// mahsulotga qiymat tayinlaydi
	return b	// metod zanjiri uchun builder qaytaradi
}

func (b *HouseBuilder) SetRoof(roof string) *HouseBuilder {	// tom turini o'rnatadi
	b.house.Roof = roof	// mahsulotga qiymat tayinlaydi
	return b	// metod zanjiri uchun builder qaytaradi
}

func (b *HouseBuilder) SetGarage(hasGarage bool) *HouseBuilder {	// garaj opsiyasini o'rnatadi
	b.house.HasGarage = hasGarage	// bool qiymat tayinlaydi
	return b	// metod zanjiri uchun builder qaytaradi
}

func (b *HouseBuilder) SetSwimmingPool(hasPool bool) *HouseBuilder {	// basseyn opsiyasini o'rnatadi
	b.house.HasSwimmingPool = hasPool	// bool qiymat tayinlaydi
	return b	// metod zanjiri uchun builder qaytaradi
}

func (b *HouseBuilder) SetGarden(hasGarden bool) *HouseBuilder {	// bog' opsiyasini o'rnatadi
	b.house.HasGarden = hasGarden	// bool qiymat tayinlaydi
	return b	// metod zanjiri uchun builder qaytaradi
}

func (b *HouseBuilder) Build() *House {	// Mahsulotni yakunlaydi va qaytaradi
	return b.house	// qurilgan House ni qaytaradi
}`
		}
	}
};

export default task;
