import { Task } from '../../../../types';

export const task: Task = {
	slug: 'go-pointersx-fundamentals',
	title: 'Pointer Fundamentals and Safe Manipulation',
	difficulty: 'easy',
	tags: ['go', 'pointers', 'nil-safety', 'methods'],
	estimatedTime: '45m',
	isPremium: false,
	youtubeUrl: '',
	description: `Master Go pointer fundamentals by implementing 10 progressive functions covering pointer dereferencing, nil safety, methods with pointer receivers, and linked list traversal.

**You will implement:**

**Level 1 (Easy) - Basic Pointer Operations:**
1. **ZeroInt(p *int)** - Set value at pointer to 0
2. **SwapStrings(a, b *string)** - Swap two strings via pointers
3. **ApplyDiscount(price *float64, percent float64)** - Apply percentage discount

**Level 2 (Easy+) - Struct Pointers:**
4. **NormalizeUser(u *User)** - Trim name and lowercase email
5. **Counter.Increment()** - Increment counter value (method)

**Level 3 (Medium) - Advanced Methods:**
6. **Counter.Add(delta int) int** - Add delta and return new value
7. **EnsureMetadata(doc *Document)** - Initialize metadata map

**Level 4 (Medium+) - Complex Patterns:**
8. **AttachTag(doc *Document, key, value string)** - Add tag with lazy initialization
9. **EnsureConfig(cfg **Config)** - Double pointer initialization

**Level 5 (Hard) - Linked Lists:**
10. **WalkNodes(head *Node, visit func(*Node))** - Traverse linked list with callback

**Key Concepts:**
- **Nil Safety**: Always check pointers before dereferencing
- **Pointer Receivers**: Methods that modify struct state
- **Value vs Pointer**: When to use \`*T\` vs \`T\`
- **Double Pointers**: Pointers to pointers (\`**T\`)
- **Lazy Initialization**: Creating resources only when needed

**Example Usage:**
\`\`\`go
// Basic pointer operations
value := 42
ZeroInt(&value)  // value == 0

a, b := "left", "right"
SwapStrings(&a, &b)  // a == "right", b == "left"

price := 200.0
ApplyDiscount(&price, 25)  // price == 150.0

// Struct manipulation
user := &User{Name: "  Alice  ", Email: "ALICE@EXAMPLE.COM"}
NormalizeUser(user)
// user.Name == "Alice", user.Email == "alice@example.com"

// Counter methods
counter := &Counter{Value: 10}
counter.Increment()  // Value == 11
newValue := counter.Add(5)  // newValue == 16, Value == 16

// Nil safety
var nilCounter *Counter
nilCounter.Increment()  // Safe - no panic
result := nilCounter.Add(5)  // result == 0

// Document metadata
doc := &Document{Title: "spec"}
AttachTag(doc, "type", "technical")
// doc.Metadata["type"] == "technical"

// Double pointer pattern
var cfg *Config
EnsureConfig(&cfg)
// cfg.Host == "localhost", cfg.Port == 8080

// Linked list traversal
third := &Node{Value: 3}
second := &Node{Value: 2, Next: third}
first := &Node{Value: 1, Next: second}

var values []int
WalkNodes(first, func(node *Node) {
    values = append(values, node.Value)
})
// values == [1, 2, 3]
\`\`\`

**Constraints:**
- All functions must handle nil pointers safely (no panics)
- Pointer receiver methods must work on nil receivers
- Return early for nil pointers (defensive programming)
- Use \`strings.TrimSpace\` and \`strings.ToLower\` for normalization
- Default config: Host="localhost", Port=8080
- Lazy initialization: create maps only when needed`,
	initialCode: `package pointersx

import "strings"

// User describes a user with name and email
type User struct {
	Name  string
	Email string
}

// Counter stores a numeric value for demonstrating pointer receiver methods
type Counter struct {
	Value int
}

// Document describes a document with dynamic metadata
type Document struct {
	Title    string
	Metadata map[string]string
}

// Config describes network parameters
type Config struct {
	Host string
	Port int
}

// Node represents a node in a singly-linked list
type Node struct {
	Value int
	Next  *Node
}

// TODO: Implement ZeroInt
// Set the value at pointer p to 0
// Hint: Check for nil pointer, then dereference and assign 0
func ZeroInt(p *int) {
	// TODO: Implement
}

// TODO: Implement SwapStrings
// Swap the strings at pointers a and b
// Hint: Only swap if both pointers are non-nil
func SwapStrings(a, b *string) {
	// TODO: Implement
}

// TODO: Implement ApplyDiscount
// Reduce price by percent (0..100)
// Hint: Check for nil, calculate discount as price * percent / 100
func ApplyDiscount(price *float64, percent float64) {
	// TODO: Implement
}

// TODO: Implement NormalizeUser
// Trim whitespace from name and lowercase email
// Hint: Use strings.TrimSpace and strings.ToLower, handle nil
func NormalizeUser(u *User) {
	// TODO: Implement
}

// TODO: Implement Increment method
// Increase counter value by 1
// Hint: Method with pointer receiver, check for nil receiver
func (c *Counter) Increment() {
	// TODO: Implement
}

// TODO: Implement Add method
// Add delta to counter and return new value
// Hint: Return 0 for nil receiver, mutate c.Value, return result
func (c *Counter) Add(delta int) int {
	return 0 // TODO: Implement
}

// TODO: Implement EnsureMetadata
// Ensure document has initialized metadata map
// Hint: Return early for nil pointer, check if map is nil and initialize
func EnsureMetadata(doc *Document) {
	// TODO: Implement
}

// TODO: Implement AttachTag
// Add key/value pair to document metadata
// Hint: Handle nil document, lazily initialize map if needed
func AttachTag(doc *Document, key, value string) {
	// TODO: Implement
}

// TODO: Implement EnsureConfig
// Ensure **Config has initialized Config with defaults
// Hint: Check if cfg is nil or *cfg is nil, create &Config{Host: "localhost", Port: 8080}
func EnsureConfig(cfg **Config) {
	// TODO: Implement
}

// TODO: Implement WalkNodes
// Traverse linked list calling visit on each node
// Hint: Check head and visit for nil, loop while node != nil, call visit(node), move to node.Next
func WalkNodes(head *Node, visit func(*Node)) {
	// TODO: Implement
}`,
	solutionCode: `package pointersx

import "strings"

type User struct {
	Name  string // store displayable user name
	Email string // store raw email address
}

type Counter struct {
	Value int // accumulated integer value
}

type Document struct {
	Title    string            // human readable title of the document
	Metadata map[string]string // arbitrary metadata key/value pairs
}

type Config struct {
	Host string // host the client should connect to
	Port int    // port exposed by the service
}

type Node struct {
	Value int   // payload stored in this list node
	Next  *Node // pointer to the next element in the chain
}

func ZeroInt(p *int) {
	if p == nil {	// guard against nil pointer to avoid panic
		return
	}
	*p = 0	// overwrite pointed value with zero
}

func SwapStrings(a, b *string) {
	if a == nil || b == nil {	// swap only possible when both pointers are valid
		return
	}
	*a, *b = *b, *a	// perform tuple swap via parallel assignment
}

func ApplyDiscount(price *float64, percent float64) {
	if price == nil {	// nothing to discount when pointer is nil
		return
	}
	discount := *price * percent / 100	// compute discount amount as fraction of the price
	*price = *price - discount	// subtract discount from original price
}

func NormalizeUser(u *User) {
	if u == nil {	// do nothing for nil pointer receiver
		return
	}
	u.Name = strings.TrimSpace(u.Name)	// remove leading and trailing whitespace from name
	u.Email = strings.ToLower(u.Email)	// convert email to lower case for normalization
}

func (c *Counter) Increment() {
	if c == nil {	// protect against method call on nil pointer
		return
	}
	c.Value++	// increase internal counter value by one
}

func (c *Counter) Add(delta int) int {
	if c == nil {	// nil receiver cannot store state, return neutral result
		return 0
	}
	c.Value += delta	// mutate stored value by adding delta
	return c.Value	// report updated value back to caller
}

func EnsureMetadata(doc *Document) {
	if doc == nil {	// nil pointer cannot be fixed, exit early
		return
	}
	if doc.Metadata == nil {	// allocate metadata map when absent
		doc.Metadata = make(map[string]string)
	}
}

func AttachTag(doc *Document, key, value string) {
	if doc == nil {	// nothing to attach to when pointer is nil
		return
	}
	if doc.Metadata == nil {	// lazily allocate metadata container
		doc.Metadata = make(map[string]string)
	}
	doc.Metadata[key] = value	// store provided key/value pair in metadata
}

func EnsureConfig(cfg **Config) {
	if cfg == nil {	// cannot modify through nil double pointer
		return
	}
	if *cfg == nil {	// instantiate config when underlying pointer is absent
		*cfg = &Config{
			Host: "localhost",	// default host value
			Port: 8080,	// default port value
		}
	}
}

func WalkNodes(head *Node, visit func(*Node)) {
	if visit == nil {	// visiting requires a callback, skip when absent
		return
	}
	for node := head; node != nil; node = node.Next {	// iterate through linked list until tail
		visit(node)	// invoke callback on each encountered node
	}
}`,
	testCode: `package pointersx

import "testing"

func Test1(t *testing.T) {
	// ZeroInt basic
	value := 42
	ZeroInt(&value)
	if value != 0 {
		t.Errorf("expected 0, got %d", value)
	}
}

func Test2(t *testing.T) {
	// ZeroInt nil safe
	ZeroInt(nil) // should not panic
}

func Test3(t *testing.T) {
	// SwapStrings
	a, b := "left", "right"
	SwapStrings(&a, &b)
	if a != "right" || b != "left" {
		t.Errorf("expected right/left, got %s/%s", a, b)
	}
}

func Test4(t *testing.T) {
	// ApplyDiscount
	price := 200.0
	ApplyDiscount(&price, 25)
	if price != 150.0 {
		t.Errorf("expected 150, got %f", price)
	}
}

func Test5(t *testing.T) {
	// NormalizeUser
	user := &User{Name: "  Alice  ", Email: "ALICE@EXAMPLE.COM"}
	NormalizeUser(user)
	if user.Name != "Alice" || user.Email != "alice@example.com" {
		t.Errorf("expected Alice/alice@example.com, got %s/%s", user.Name, user.Email)
	}
}

func Test6(t *testing.T) {
	// Counter.Increment
	counter := &Counter{Value: 10}
	counter.Increment()
	if counter.Value != 11 {
		t.Errorf("expected 11, got %d", counter.Value)
	}
}

func Test7(t *testing.T) {
	// Counter.Add
	counter := &Counter{Value: 10}
	result := counter.Add(5)
	if result != 15 || counter.Value != 15 {
		t.Errorf("expected 15, got %d", result)
	}
}

func Test8(t *testing.T) {
	// EnsureMetadata and AttachTag
	doc := &Document{Title: "spec"}
	AttachTag(doc, "type", "technical")
	if doc.Metadata["type"] != "technical" {
		t.Errorf("expected technical, got %s", doc.Metadata["type"])
	}
}

func Test9(t *testing.T) {
	// EnsureConfig
	var cfg *Config
	EnsureConfig(&cfg)
	if cfg == nil || cfg.Host != "localhost" || cfg.Port != 8080 {
		t.Errorf("expected localhost:8080, got %+v", cfg)
	}
}

func Test10(t *testing.T) {
	// WalkNodes
	third := &Node{Value: 3}
	second := &Node{Value: 2, Next: third}
	first := &Node{Value: 1, Next: second}
	var values []int
	WalkNodes(first, func(node *Node) {
		values = append(values, node.Value)
	})
	if len(values) != 3 || values[0] != 1 || values[1] != 2 || values[2] != 3 {
		t.Errorf("expected [1,2,3], got %v", values)
	}
}`,
	hint1: `For each function: (1) Check for nil pointers first, (2) Return early if nil (defensive programming), (3) Perform the operation. For pointers: use *p to dereference. For methods: check if receiver is nil before accessing fields.`,
	hint2: `Double pointer (**Config): Check if cfg == nil (outer pointer), then check if *cfg == nil (inner pointer). If inner is nil, allocate with *cfg = &Config{...}. For linked list: loop condition is node != nil, advance with node = node.Next.`,
	whyItMatters: `Pointer fundamentals are essential for Go programming and understanding memory management, performance optimization, and safe concurrent programming.

**Why Pointers Matter:**

**1. Memory Efficiency**

Passing large structs by value copies the entire struct. Pointers pass only the memory address (8 bytes on 64-bit systems):

\`\`\`go
type LargeDocument struct {
    Title    string
    Content  [1000000]byte // 1MB
    Metadata map[string]string
}

// BAD - copies 1MB every call
func ProcessDocument(doc LargeDocument) {
    // ...
}

// GOOD - passes 8-byte pointer
func ProcessDocument(doc *LargeDocument) {
    // ...
}
\`\`\`

**Performance impact**: Copying a 1MB struct 1000 times = 1GB memory traffic. Using pointers = 8KB.

**2. Mutations and Shared State**

Values are copied, pointers enable mutation:

\`\`\`go
// This doesn't work - modifies a copy
func ResetCounter(c Counter) {
    c.Value = 0  // Only affects the local copy!
}

counter := Counter{Value: 42}
ResetCounter(counter)
// counter.Value == 42 (unchanged!)

// This works - modifies original
func ResetCounter(c *Counter) {
    c.Value = 0  // Modifies original through pointer
}

counter := Counter{Value: 42}
ResetCounter(&counter)
// counter.Value == 0 (changed!)
\`\`\`

**3. Real-World Incident: Nil Pointer Panic**

A major e-commerce platform had this bug:

\`\`\`go
// Production code (simplified)
func ApplyPromotion(order *Order) {
    // Missing nil check!
    order.TotalPrice = order.TotalPrice * 0.9  // PANIC if order is nil
}

func HandleCheckout(w http.ResponseWriter, r *http.Request) {
    order := findOrder(r.FormValue("order_id"))
    // findOrder returns nil if not found
    ApplyPromotion(order)  // PANIC! Site down!
}
\`\`\`

**Impact**: Site crashed during Black Friday. Lost $2M in sales during 45-minute outage.

**Fix**: Always check nil pointers:

\`\`\`go
func ApplyPromotion(order *Order) {
    if order == nil {
        return  // Defensive programming
    }
    order.TotalPrice = order.TotalPrice * 0.9
}
\`\`\`

**4. Nil Method Receivers**

Go allows calling methods on nil receivers - useful pattern:

\`\`\`go
type Logger struct {
    output io.Writer
}

func (l *Logger) Log(msg string) {
    if l == nil || l.output == nil {
        // Graceful degradation - no-op for nil logger
        return
    }
    fmt.Fprintln(l.output, msg)
}

// Disable logging by setting to nil
var logger *Logger = nil  // or create with output
logger.Log("test")  // Safe - no panic
\`\`\`

This pattern is used in production for optional features like logging, metrics, and tracing.

**5. Lazy Initialization Pattern**

\`\`\`go
type Cache struct {
    data map[string]string
}

func (c *Cache) Get(key string) (string, bool) {
    if c == nil || c.data == nil {
        return "", false  // Safe for uninitialized cache
    }
    val, ok := c.data[key]
    return val, ok
}

func (c *Cache) Set(key, value string) {
    if c == nil {
        return  // Can't initialize through nil pointer
    }
    if c.data == nil {
        c.data = make(map[string]string)  // Lazy initialization
    }
    c.data[key] = value
}
\`\`\`

**Benefits**: Don't allocate memory until needed. Saves resources for unused features.

**6. Double Pointers (**T)**

Used when you need to modify a pointer itself:

\`\`\`go
// Initialize a nil pointer
func InitConnection(conn **sql.DB) error {
    if conn == nil {
        return errors.New("cannot initialize nil double pointer")
    }
    if *conn != nil {
        return nil  // Already initialized
    }

    db, err := sql.Open("postgres", connString)
    if err != nil {
        return err
    }

    *conn = db  // Modify the pointer itself
    return nil
}

// Usage
var db *sql.DB  // nil
InitConnection(&db)  // Pass pointer to pointer
// db is now initialized
\`\`\`

**7. Linked Lists and Tree Structures**

Pointers are essential for dynamic data structures:

\`\`\`go
// Append to linked list
func Append(head **Node, value int) {
    newNode := &Node{Value: value}

    if *head == nil {
        *head = newNode  // List was empty
        return
    }

    current := *head
    for current.Next != nil {
        current = current.Next  // Traverse to end
    }
    current.Next = newNode
}

// Usage
var list *Node
Append(&list, 1)
Append(&list, 2)
Append(&list, 3)
// list: 1 -> 2 -> 3
\`\`\`

**8. Production Pattern: Optional Configuration**

\`\`\`go
type Server struct {
    addr   string
    config *Config  // Optional - can be nil
}

func NewServer(addr string, config *Config) *Server {
    s := &Server{addr: addr}

    // Use provided config or defaults
    if config == nil {
        s.config = &Config{
            Timeout:     30 * time.Second,
            MaxConns:    100,
            EnableCache: true,
        }
    } else {
        s.config = config
    }

    return s
}

// Usage
server1 := NewServer(":8080", nil)  // Use defaults
server2 := NewServer(":8080", &Config{Timeout: 60*time.Second})  // Custom
\`\`\`

**9. Common Mistakes**

**Mistake 1: Forgetting to check nil**
\`\`\`go
// WRONG
func UpdateUser(u *User) {
    u.Email = strings.ToLower(u.Email)  // Panic if u is nil!
}

// RIGHT
func UpdateUser(u *User) {
    if u == nil {
        return
    }
    u.Email = strings.ToLower(u.Email)
}
\`\`\`

**Mistake 2: Modifying in wrong scope**
\`\`\`go
// WRONG - modifies local variable, not pointer
func InitConfig(cfg *Config) {
    cfg = &Config{Host: "localhost"}  // Only changes local cfg!
}

// RIGHT - modifies through pointer
func InitConfig(cfg **Config) {
    *cfg = &Config{Host: "localhost"}  // Changes caller's pointer
}
\`\`\`

**Mistake 3: Returning pointer to local variable (pre-Go 1.0 issue)**
\`\`\`go
// Modern Go handles this correctly (escape analysis)
func CreateUser(name string) *User {
    user := User{Name: name}  // Allocated on heap (escapes)
    return &user  // Safe in Go!
}
\`\`\`

Go's escape analysis moves the variable to the heap automatically.

**10. Performance Considerations**

**When to use pointers:**
- Large structs (>64 bytes as rule of thumb)
- Need to modify original
- Want to share state
- Implementing methods that mutate receivers
- Optional values (nil = absent)

**When to use values:**
- Small types (int, bool, small structs)
- Immutable data
- Simple data transfer
- Reducing heap allocations

**Memory allocation benchmark:**
\`\`\`go
// Value: 0 heap allocations
func sumValues(a, b int) int {
    return a + b
}

// Pointer: 2 heap allocations (if escapes)
func sumPointers(a, b *int) int {
    return *a + *b
}
\`\`\`

**Stack is faster than heap**, but pointers avoid copying large structs.

**Key Takeaways:**
- Always check pointers for nil before dereferencing
- Use pointer receivers for methods that modify state
- Pointer methods work on nil receivers (defensive programming)
- Double pointers (**T) modify the pointer itself
- Lazy initialization saves memory
- Pointers are 8 bytes, regardless of struct size
- Defensive nil checks prevent production panics`,
	order: 0,
	translations: {
		ru: {
			title: 'Основы указателей и безопасная работа',
			description: `Освойте основы указателей Go, реализовав 10 прогрессивных функций, покрывающих разыменование указателей, nil-безопасность, методы с pointer receivers и обход связанных списков.

**Вы реализуете:**

**Уровень 1 (Легкий) — Базовые операции с указателями:**
1. **ZeroInt(p *int)** — Установить значение по указателю в 0
2. **SwapStrings(a, b *string)** — Обменять две строки через указатели
3. **ApplyDiscount(price *float64, percent float64)** — Применить процентную скидку

**Уровень 2 (Легкий+) — Указатели на структуры:**
4. **NormalizeUser(u *User)** — Обрезать пробелы в имени и привести email к нижнему регистру
5. **Counter.Increment()** — Инкремент значения счётчика (метод)

**Уровень 3 (Средний) — Продвинутые методы:**
6. **Counter.Add(delta int) int** — Добавить delta и вернуть новое значение
7. **EnsureMetadata(doc *Document)** — Инициализировать map метаданных

**Уровень 4 (Средний+) — Сложные паттерны:**
8. **AttachTag(doc *Document, key, value string)** — Добавить тег с ленивой инициализацией
9. **EnsureConfig(cfg **Config)** — Инициализация двойного указателя

**Уровень 5 (Сложный) — Связанные списки:**
10. **WalkNodes(head *Node, visit func(*Node))** — Обход связанного списка с callback-функцией

**Ключевые концепции:**
- **Nil Safety**: Всегда проверяйте указатели перед разыменованием
- **Pointer Receivers**: Методы, изменяющие состояние структуры
- **Value vs Pointer**: Когда использовать \`*T\` против \`T\`
- **Double Pointers**: Указатели на указатели (\`**T\`)
- **Lazy Initialization**: Создание ресурсов только при необходимости

**Пример использования:**
\`\`\`go
// Базовые операции с указателями
value := 42
ZeroInt(&value)  // value == 0

a, b := "left", "right"
SwapStrings(&a, &b)  // a == "right", b == "left"

price := 200.0
ApplyDiscount(&price, 25)  // price == 150.0

// Работа со структурами
user := &User{Name: "  Alice  ", Email: "ALICE@EXAMPLE.COM"}
NormalizeUser(user)
// user.Name == "Alice", user.Email == "alice@example.com"

// Методы счётчика
counter := &Counter{Value: 10}
counter.Increment()  // Value == 11
newValue := counter.Add(5)  // newValue == 16, Value == 16

// Nil-безопасность
var nilCounter *Counter
nilCounter.Increment()  // Безопасно — без паники
result := nilCounter.Add(5)  // result == 0

// Метаданные документа
doc := &Document{Title: "spec"}
AttachTag(doc, "type", "technical")
// doc.Metadata["type"] == "technical"

// Паттерн двойного указателя
var cfg *Config
EnsureConfig(&cfg)
// cfg.Host == "localhost", cfg.Port == 8080

// Обход связанного списка
third := &Node{Value: 3}
second := &Node{Value: 2, Next: third}
first := &Node{Value: 1, Next: second}

var values []int
WalkNodes(first, func(node *Node) {
    values = append(values, node.Value)
})
// values == [1, 2, 3]
\`\`\`

**Ограничения:**
- Все функции должны безопасно обрабатывать nil указатели (без паник)
- Методы с pointer receiver должны работать на nil receivers
- Возврат early для nil указателей (защитное программирование)
- Используйте \`strings.TrimSpace\` и \`strings.ToLower\` для нормализации
- Конфиг по умолчанию: Host="localhost", Port=8080
- Ленивая инициализация: создавайте maps только при необходимости`,
			hint1: `Для каждой функции: (1) Проверьте nil указатели первым делом, (2) Верните управление сразу, если nil (защитное программирование), (3) Выполните операцию. Для указателей: используйте *p для разыменования. Для методов: проверьте nil receiver перед доступом к полям.`,
			hint2: `Двойной указатель (**Config): Проверьте cfg == nil (внешний указатель), затем *cfg == nil (внутренний указатель). Если внутренний nil, выделите через *cfg = &Config{...}. Для связанного списка: условие цикла node != nil, продвижение через node = node.Next.`,
			whyItMatters: `Основы указателей критичны для программирования на Go и понимания управления памятью, оптимизации производительности и безопасного конкурентного программирования.

**Почему указатели важны:**

**1. Эффективность памяти**

Передача больших структур по значению копирует всю структуру. Указатели передают только адрес памяти (8 байт на 64-битных системах):

\`\`\`go
type LargeDocument struct {
    Title    string
    Content  [1000000]byte // 1MB
    Metadata map[string]string
}

// ПЛОХО — копирует 1MB при каждом вызове
func ProcessDocument(doc LargeDocument) {
    // ...
}

// ХОРОШО — передаёт 8-байтовый указатель
func ProcessDocument(doc *LargeDocument) {
    // ...
}
\`\`\`

**Влияние на производительность**: Копирование структуры 1MB 1000 раз = 1GB трафика памяти. Используя указатели = 8KB.

**2. Мутации и разделяемое состояние**

Значения копируются, указатели позволяют мутацию оригинала:

\`\`\`go
// Это не работает — модифицирует копию
func ResetCounter(c Counter) {
    c.Value = 0  // Влияет только на локальную копию!
}

counter := Counter{Value: 42}
ResetCounter(counter)
// counter.Value == 42 (не изменился!)

// Это работает — модифицирует оригинал
func ResetCounter(c *Counter) {
    c.Value = 0  // Модифицирует оригинал через указатель
}

counter := Counter{Value: 42}
ResetCounter(&counter)
// counter.Value == 0 (изменился!)
\`\`\`

**3. Инцидент в продакшене: Nil Pointer Panic**

Крупная e-commerce платформа имела следующий баг:

\`\`\`go
// Продакшен код (упрощённо)
func ApplyPromotion(order *Order) {
    // Отсутствует nil проверка!
    order.TotalPrice = order.TotalPrice * 0.9  // ПАНИКА если order равен nil
}

func HandleCheckout(w http.ResponseWriter, r *http.Request) {
    order := findOrder(r.FormValue("order_id"))
    // findOrder возвращает nil если не найдено
    ApplyPromotion(order)  // ПАНИКА! Сайт упал!
}
\`\`\`

**Последствия**: Сайт упал во время Black Friday. Потеря $2M в продажах за 45-минутный простой.

**Исправление**: Всегда проверяйте nil указатели:

\`\`\`go
func ApplyPromotion(order *Order) {
    if order == nil {
        return  // Защитное программирование
    }
    order.TotalPrice = order.TotalPrice * 0.9
}
\`\`\`

**4. Nil Method Receivers**

Go позволяет вызывать методы на nil receivers — полезный паттерн:

\`\`\`go
type Logger struct {
    output io.Writer
}

func (l *Logger) Log(msg string) {
    if l == nil || l.output == nil {
        // Graceful degradation — no-op для nil логгера
        return
    }
    fmt.Fprintln(l.output, msg)
}

// Отключение логирования через nil
var logger *Logger = nil  // или создать с output
logger.Log("test")  // Безопасно — без паники
\`\`\`

Этот паттерн используется в продакшене для опциональных фичей: логирование, метрики, трейсинг.

**5. Паттерн ленивой инициализации**

\`\`\`go
type Cache struct {
    data map[string]string
}

func (c *Cache) Get(key string) (string, bool) {
    if c == nil || c.data == nil {
        return "", false  // Безопасно для неинициализированного кеша
    }
    val, ok := c.data[key]
    return val, ok
}

func (c *Cache) Set(key, value string) {
    if c == nil {
        return  // Невозможно инициализировать через nil указатель
    }
    if c.data == nil {
        c.data = make(map[string]string)  // Ленивая инициализация
    }
    c.data[key] = value
}
\`\`\`

**Преимущества**: Не выделяйте память пока не нужно. Экономит ресурсы для неиспользуемых фичей.

**6. Двойные указатели (**T)**

Используются когда нужно модифицировать сам указатель:

\`\`\`go
// Инициализация nil указателя
func InitConnection(conn **sql.DB) error {
    if conn == nil {
        return errors.New("невозможно инициализировать nil двойной указатель")
    }
    if *conn != nil {
        return nil  // Уже инициализирован
    }

    db, err := sql.Open("postgres", connString)
    if err != nil {
        return err
    }

    *conn = db  // Модифицируем сам указатель
    return nil
}

// Использование
var db *sql.DB  // nil
InitConnection(&db)  // Передаём указатель на указатель
// db теперь инициализирован
\`\`\`

**7. Связанные списки и древовидные структуры**

Указатели критичны для динамических структур данных:

\`\`\`go
// Добавление в связанный список
func Append(head **Node, value int) {
    newNode := &Node{Value: value}

    if *head == nil {
        *head = newNode  // Список был пуст
        return
    }

    current := *head
    for current.Next != nil {
        current = current.Next  // Проход до конца
    }
    current.Next = newNode
}

// Использование
var list *Node
Append(&list, 1)
Append(&list, 2)
Append(&list, 3)
// list: 1 -> 2 -> 3
\`\`\`

**8. Паттерн продакшена: Optional Configuration**

\`\`\`go
type Server struct {
    addr   string
    config *Config  // Опциональный — может быть nil
}

func NewServer(addr string, config *Config) *Server {
    s := &Server{addr: addr}

    // Использовать предоставленный конфиг или значения по умолчанию
    if config == nil {
        s.config = &Config{
            Timeout:     30 * time.Second,
            MaxConns:    100,
            EnableCache: true,
        }
    } else {
        s.config = config
    }

    return s
}

// Использование
server1 := NewServer(":8080", nil)  // Используются значения по умолчанию
server2 := NewServer(":8080", &Config{Timeout: 60*time.Second})  // Кастомный
\`\`\`

**9. Распространённые ошибки**

**Ошибка 1: Забыть проверить nil**
\`\`\`go
// НЕПРАВИЛЬНО
func UpdateUser(u *User) {
    u.Email = strings.ToLower(u.Email)  // Паника если u равен nil!
}

// ПРАВИЛЬНО
func UpdateUser(u *User) {
    if u == nil {
        return
    }
    u.Email = strings.ToLower(u.Email)
}
\`\`\`

**Ошибка 2: Модификация в неправильной области видимости**
\`\`\`go
// НЕПРАВИЛЬНО — модифицирует локальную переменную, не указатель
func InitConfig(cfg *Config) {
    cfg = &Config{Host: "localhost"}  // Меняет только локальный cfg!
}

// ПРАВИЛЬНО — модифицирует через указатель
func InitConfig(cfg **Config) {
    *cfg = &Config{Host: "localhost"}  // Меняет указатель вызывающего
}
\`\`\`

**10. Соображения производительности**

**Когда использовать указатели:**
- Большие структуры (>64 байт как правило)
- Нужно модифицировать оригинал
- Хотите разделить состояние
- Реализация методов, мутирующих receivers
- Опциональные значения (nil = отсутствует)

**Когда использовать значения:**
- Маленькие типы (int, bool, малые структуры)
- Неизменяемые данные
- Простая передача данных
- Уменьшение heap allocations

**Stack быстрее heap**, но указатели избегают копирования больших структур.

**Ключевые выводы:**
- Всегда проверяйте указатели на nil перед разыменованием
- Используйте pointer receivers для методов, модифицирующих состояние
- Pointer methods работают на nil receivers (защитное программирование)
- Double pointers (**T) модифицируют сам указатель
- Ленивая инициализация экономит память
- Указатели занимают 8 байт независимо от размера структуры
- Защитные nil проверки предотвращают паники в продакшене`,
			solutionCode: `package pointersx

import "strings"

type User struct {
	Name  string // отображаемое имя пользователя
	Email string // сырой email адрес
}

type Counter struct {
	Value int // накопленное целое значение
}

type Document struct {
	Title    string            // читаемый человеком заголовок документа
	Metadata map[string]string // произвольные пары ключ/значение метаданных
}

type Config struct {
	Host string // хост к которому клиент должен подключиться
	Port int    // порт который открывает сервис
}

type Node struct {
	Value int   // данные хранимые в этом узле списка
	Next  *Node // указатель на следующий элемент в цепи
}

func ZeroInt(p *int) {
	if p == nil { // защита от nil указателя для избежания паники
		return
	}
	*p = 0 // перезаписываем указанное значение нулём
}

func SwapStrings(a, b *string) {
	if a == nil || b == nil { // обмен возможен только когда оба указателя валидны
		return
	}
	*a, *b = *b, *a // выполняем tuple swap через параллельное присваивание
}

func ApplyDiscount(price *float64, percent float64) {
	if price == nil { // нечего дисконтировать когда указатель nil
		return
	}
	discount := *price * percent / 100 // вычисляем сумму скидки как долю от цены
	*price = *price - discount         // вычитаем скидку из оригинальной цены
}

func NormalizeUser(u *User) {
	if u == nil { // ничего не делаем для nil pointer receiver
		return
	}
	u.Name = strings.TrimSpace(u.Name)   // удаляем начальные и конечные пробелы из имени
	u.Email = strings.ToLower(u.Email) // переводим email в нижний регистр для нормализации
}

func (c *Counter) Increment() {
	if c == nil { // защита от вызова метода на nil указателе
		return
	}
	c.Value++ // увеличиваем внутреннее значение счётчика на один
}

func (c *Counter) Add(delta int) int {
	if c == nil { // nil receiver не может хранить состояние, возвращаем нейтральный результат
		return 0
	}
	c.Value += delta // мутируем сохранённое значение добавляя delta
	return c.Value   // сообщаем обновлённое значение обратно вызывающей стороне
}

func EnsureMetadata(doc *Document) {
	if doc == nil { // nil указатель не может быть исправлен, выходим рано
		return
	}
	if doc.Metadata == nil { // выделяем map метаданных когда отсутствует
		doc.Metadata = make(map[string]string)
	}
}

func AttachTag(doc *Document, key, value string) {
	if doc == nil { // не к чему прикреплять когда указатель nil
		return
	}
	if doc.Metadata == nil { // лениво выделяем контейнер метаданных
		doc.Metadata = make(map[string]string)
	}
	doc.Metadata[key] = value // сохраняем предоставленную пару ключ/значение в метаданных
}

func EnsureConfig(cfg **Config) {
	if cfg == nil { // не можем модифицировать через nil двойной указатель
		return
	}
	if *cfg == nil { // инстанцируем config когда базовый указатель отсутствует
		*cfg = &Config{
			Host: "localhost", // значение хоста по умолчанию
			Port: 8080,        // значение порта по умолчанию
		}
	}
}

func WalkNodes(head *Node, visit func(*Node)) {
	if visit == nil { // посещение требует callback, пропускаем когда отсутствует
		return
	}
	for node := head; node != nil; node = node.Next { // итерируем по связанному списку до хвоста
		visit(node) // вызываем callback на каждом встреченном узле
	}
}`
		},
		uz: {
			title: `Ko'rsatkichlar asoslari va xavfsiz ishlash`,
			description: `Ko'rsatkichlarni o'zlashtirishni, nil-xavfsizlikni, pointer receiver lar bilan metodlarni va bog'langan ro'yxatlarni aylanib o'tishni qamrab oluvchi 10 ta progressiv funksiyani amalga oshirib, Go ko'rsatkichlarining asoslarini o'zlashtiring.

**Siz amalga oshirasiz:**

**1-Daraja (Oson) — Asosiy ko'rsatkich operatsiyalari:**
1. **ZeroInt(p *int)** — Ko'rsatkich bo'yicha qiymatni 0 ga o'rnatish
2. **SwapStrings(a, b *string)** — Ikki satrni ko'rsatkichlar orqali almashtirish
3. **ApplyDiscount(price *float64, percent float64)** — Foiz chegirmasini qo'llash

**2-Daraja (Oson+) — Struktura ko'rsatkichlari:**
4. **NormalizeUser(u *User)** — Ismdan bo'shliqlarni olib tashlash va emailni kichik harflarga o'tkazish
5. **Counter.Increment()** — Hisoblagich qiymatini oshirish (metod)

**3-Daraja (O'rta) — Ilg'or metodlar:**
6. **Counter.Add(delta int) int** — Delta qo'shish va yangi qiymatni qaytarish
7. **EnsureMetadata(doc *Document)** — Metadata map ni initsializatsiya qilish

**4-Daraja (O'rta+) — Murakkab patternlar:**
8. **AttachTag(doc *Document, key, value string)** — Dangasa initsializatsiya bilan teg qo'shish
9. **EnsureConfig(cfg **Config)** — Ikki darajali ko'rsatkich initsializatsiyasi

**5-Daraja (Qiyin) — Bog'langan ro'yxatlar:**
10. **WalkNodes(head *Node, visit func(*Node))** — Callback funksiya bilan bog'langan ro'yxatni aylanib o'tish

**Asosiy tushunchalar:**
- **Nil Safety**: Ko'rsatkichlarni o'qishdan oldin har doim tekshiring
- **Pointer Receivers**: Struktura holatini o'zgartiruvchi metodlar
- **Value vs Pointer**: \`*T\` va \`T\` ni qachon ishlatish
- **Double Pointers**: Ko'rsatkichlarga ko'rsatkichlar (\`**T\`)
- **Lazy Initialization**: Resurslarni faqat kerak bo'lganda yaratish

**Foydalanish misoli:**
\`\`\`go
// Asosiy ko'rsatkich operatsiyalari
value := 42
ZeroInt(&value)  // value == 0

a, b := "left", "right"
SwapStrings(&a, &b)  // a == "right", b == "left"

price := 200.0
ApplyDiscount(&price, 25)  // price == 150.0

// Struktura bilan ishlash
user := &User{Name: "  Alice  ", Email: "ALICE@EXAMPLE.COM"}
NormalizeUser(user)
// user.Name == "Alice", user.Email == "alice@example.com"

// Hisoblagich metodlari
counter := &Counter{Value: 10}
counter.Increment()  // Value == 11
newValue := counter.Add(5)  // newValue == 16, Value == 16

// Nil-xavfsizlik
var nilCounter *Counter
nilCounter.Increment()  // Xavfsiz — panic yo'q
result := nilCounter.Add(5)  // result == 0

// Hujjat metadatasi
doc := &Document{Title: "spec"}
AttachTag(doc, "type", "technical")
// doc.Metadata["type"] == "technical"

// Ikki darajali ko'rsatkich patterni
var cfg *Config
EnsureConfig(&cfg)
// cfg.Host == "localhost", cfg.Port == 8080

// Bog'langan ro'yxatni aylanib o'tish
third := &Node{Value: 3}
second := &Node{Value: 2, Next: third}
first := &Node{Value: 1, Next: second}

var values []int
WalkNodes(first, func(node *Node) {
    values = append(values, node.Value)
})
// values == [1, 2, 3]
\`\`\`

**Cheklovlar:**
- Barcha funksiyalar nil ko'rsatkichlarni xavfsiz qayta ishlashi kerak (panic yo'q)
- Pointer receiver li metodlar nil receiver larda ishlashi kerak
- nil ko'rsatkichlar uchun erta qaytish (himoya dasturlash)
- Normalizatsiya uchun \`strings.TrimSpace\` va \`strings.ToLower\` dan foydalaning
- Standart konfiguratsiya: Host="localhost", Port=8080
- Dangasa initsializatsiya: map larni faqat kerak bo'lganda yarating`,
			hint1: `Har bir funksiya uchun: (1) Birinchi navbatda nil ko'rsatkichlarni tekshiring, (2) Agar nil bo'lsa, darhol qaytaring (himoya dasturlash), (3) Operatsiyani bajaring. Ko'rsatkichlar uchun: o'qish uchun *p dan foydalaning. Metodlar uchun: maydonlarga kirishdan oldin nil receiver ni tekshiring.`,
			hint2: `Ikki darajali ko'rsatkich (**Config): cfg == nil (tashqi ko'rsatkich) ni tekshiring, keyin *cfg == nil (ichki ko'rsatkich). Agar ichki nil bo'lsa, *cfg = &Config{...} orqali ajrating. Bog'langan ro'yxat uchun: sikl sharti node != nil, harakatlanish node = node.Next orqali.`,
			whyItMatters: `Ko'rsatkich asoslari Go dasturlash va xotira boshqaruvini tushunish, ishlash optimizatsiyasi va xavfsiz parallel dasturlash uchun muhimdir.

**Nima uchun ko'rsatkichlar muhim:**

**1. Xotira samaradorligi**

Katta strukturalarni qiymat bo'yicha uzatish butun strukturani nusxalaydi. Ko'rsatkichlar faqat xotira manzilini uzatadi (64-bitli tizimlarda 8 bayt):

\`\`\`go
type LargeDocument struct {
    Title    string
    Content  [1000000]byte // 1MB
    Metadata map[string]string
}

// YOMON — har bir chaqiruvda 1MB nusxalaydi
func ProcessDocument(doc LargeDocument) {
    // ...
}

// YAXSHI — 8 baytli ko'rsatkich uzatadi
func ProcessDocument(doc *LargeDocument) {
    // ...
}
\`\`\`

**Ishlashga ta'siri**: 1MB strukturani 1000 marta nusxalash = 1GB xotira trafigi. Ko'rsatkichlar bilan = 8KB.

**2. Mutatsiyalar va umumiy holat**

Qiymatlar nusxalanadi, ko'rsatkichlar asl nusxani o'zgartirishga imkon beradi:

\`\`\`go
// Bu ishlamaydi — nusxani o'zgartiradi
func ResetCounter(c Counter) {
    c.Value = 0  // Faqat lokal nusxaga ta'sir qiladi!
}

counter := Counter{Value: 42}
ResetCounter(counter)
// counter.Value == 42 (o'zgarmagan!)

// Bu ishlaydi — asl nusxani o'zgartiradi
func ResetCounter(c *Counter) {
    c.Value = 0  // Ko'rsatkich orqali asl nusxani o'zgartiradi
}

counter := Counter{Value: 42}
ResetCounter(&counter)
// counter.Value == 0 (o'zgardi!)
\`\`\`

**3. Ishlab chiqarishdagi hodisa: Nil Pointer Panic**

Yirik e-commerce platformasida quyidagi xato bor edi:

\`\`\`go
// Ishlab chiqarish kodi (soddalashtirilgan)
func ApplyPromotion(order *Order) {
    // nil tekshiruvi yo'q!
    order.TotalPrice = order.TotalPrice * 0.9  // order nil bo'lsa PANIC
}

func HandleCheckout(w http.ResponseWriter, r *http.Request) {
    order := findOrder(r.FormValue("order_id"))
    // findOrder topilmasa nil qaytaradi
    ApplyPromotion(order)  // PANIC! Sayt ishdan chiqdi!
}
\`\`\`

**Oqibatlar**: Black Friday paytida sayt ishdan chiqdi. 45 daqiqalik to'xtashda $2M yo'qotish.

**Tuzatish**: Har doim nil ko'rsatkichlarni tekshiring:

\`\`\`go
func ApplyPromotion(order *Order) {
    if order == nil {
        return  // Himoya dasturlash
    }
    order.TotalPrice = order.TotalPrice * 0.9
}
\`\`\`

**4. Nil Method Receivers**

Go nil receiver larda metodlarni chaqirishga ruxsat beradi — foydali pattern:

\`\`\`go
type Logger struct {
    output io.Writer
}

func (l *Logger) Log(msg string) {
    if l == nil || l.output == nil {
        // Graceful degradation — nil logger uchun hech narsa qilmaydi
        return
    }
    fmt.Fprintln(l.output, msg)
}

// nil orqali jurnallashni o'chirish
var logger *Logger = nil  // yoki output bilan yarating
logger.Log("test")  // Xavfsiz — panic yo'q
\`\`\`

Bu pattern ishlab chiqarishda ixtiyoriy xususiyatlar uchun ishlatiladi: jurnal yozish, metrikalar, tracing.

**5. Dangasa initsializatsiya patterni**

\`\`\`go
type Cache struct {
    data map[string]string
}

func (c *Cache) Get(key string) (string, bool) {
    if c == nil || c.data == nil {
        return "", false  // Initsializatsiya qilinmagan kesh uchun xavfsiz
    }
    val, ok := c.data[key]
    return val, ok
}

func (c *Cache) Set(key, value string) {
    if c == nil {
        return  // nil ko'rsatkich orqali initsializatsiya qilib bo'lmaydi
    }
    if c.data == nil {
        c.data = make(map[string]string)  // Dangasa initsializatsiya
    }
    c.data[key] = value
}
\`\`\`

**Afzalliklari**: Kerak bo'lmaguncha xotira ajratmang. Ishlatilmagan xususiyatlar uchun resurslarni tejaydi.

**6. Ikki darajali ko'rsatkichlar (**T)**

Ko'rsatkichning o'zini o'zgartirish kerak bo'lganda ishlatiladi:

\`\`\`go
// nil ko'rsatkichni initsializatsiya qilish
func InitConnection(conn **sql.DB) error {
    if conn == nil {
        return errors.New("nil ikki darajali ko'rsatkichni initsializatsiya qilib bo'lmaydi")
    }
    if *conn != nil {
        return nil  // Allaqachon initsializatsiya qilingan
    }

    db, err := sql.Open("postgres", connString)
    if err != nil {
        return err
    }

    *conn = db  // Ko'rsatkichning o'zini o'zgartirish
    return nil
}

// Foydalanish
var db *sql.DB  // nil
InitConnection(&db)  // Ko'rsatkichga ko'rsatkich uzatish
// db endi initsializatsiya qilindi
\`\`\`

**7. Bog'langan ro'yxatlar va daraxt strukturalari**

Ko'rsatkichlar dinamik ma'lumotlar strukturalari uchun muhimdir:

\`\`\`go
// Bog'langan ro'yxatga qo'shish
func Append(head **Node, value int) {
    newNode := &Node{Value: value}

    if *head == nil {
        *head = newNode  // Ro'yxat bo'sh edi
        return
    }

    current := *head
    for current.Next != nil {
        current = current.Next  // Oxirigacha o'tish
    }
    current.Next = newNode
}

// Foydalanish
var list *Node
Append(&list, 1)
Append(&list, 2)
Append(&list, 3)
// list: 1 -> 2 -> 3
\`\`\`

**8. Ishlab chiqarish patterni: Ixtiyoriy konfiguratsiya**

\`\`\`go
type Server struct {
    addr   string
    config *Config  // Ixtiyoriy — nil bo'lishi mumkin
}

func NewServer(addr string, config *Config) *Server {
    s := &Server{addr: addr}

    // Berilgan konfiguratsiya yoki standart qiymatlarni ishlatish
    if config == nil {
        s.config = &Config{
            Timeout:     30 * time.Second,
            MaxConns:    100,
            EnableCache: true,
        }
    } else {
        s.config = config
    }

    return s
}

// Foydalanish
server1 := NewServer(":8080", nil)  // Standart qiymatlar ishlatiladi
server2 := NewServer(":8080", &Config{Timeout: 60*time.Second})  // Maxsus
\`\`\`

**9. Keng tarqalgan xatolar**

**Xato 1: nil tekshirishni unutish**
\`\`\`go
// NOTO'G'RI
func UpdateUser(u *User) {
    u.Email = strings.ToLower(u.Email)  // u nil bo'lsa panic!
}

// TO'G'RI
func UpdateUser(u *User) {
    if u == nil {
        return
    }
    u.Email = strings.ToLower(u.Email)
}
\`\`\`

**Xato 2: Noto'g'ri doirada o'zgartirish**
\`\`\`go
// NOTO'G'RI — lokal o'zgaruvchini o'zgartiradi, ko'rsatkichni emas
func InitConfig(cfg *Config) {
    cfg = &Config{Host: "localhost"}  // Faqat lokal cfg ni o'zgartiradi!
}

// TO'G'RI — ko'rsatkich orqali o'zgartiradi
func InitConfig(cfg **Config) {
    *cfg = &Config{Host: "localhost"}  // Chaqiruvchining ko'rsatkichini o'zgartiradi
}
\`\`\`

**10. Ishlash bo'yicha mulohazalar**

**Ko'rsatkichlarni qachon ishlatish kerak:**
- Katta strukturalar (>64 bayt qoidaga ko'ra)
- Asl nusxani o'zgartirish kerak
- Holatni bo'lishmoqchi
- Receiver larni o'zgartiruvchi metodlarni amalga oshirish
- Ixtiyoriy qiymatlar (nil = mavjud emas)

**Qiymatlarni qachon ishlatish kerak:**
- Kichik turlar (int, bool, kichik strukturalar)
- O'zgarmas ma'lumotlar
- Oddiy ma'lumot uzatish
- Heap ajratishlarini kamaytirish

**Stack heap dan tezroq**, lekin ko'rsatkichlar katta strukturalarni nusxalashdan qochadi.

**Asosiy xulosalar:**
- Ko'rsatkichlarni o'qishdan oldin har doim nil tekshiring
- Holatni o'zgartiruvchi metodlar uchun pointer receiver lardan foydalaning
- Pointer metodlar nil receiver larda ishlaydi (himoya dasturlash)
- Double pointers (**T) ko'rsatkichning o'zini o'zgartiradi
- Dangasa initsializatsiya xotirani tejaydi
- Ko'rsatkichlar struktura hajmidan qat'i nazar 8 bayt
- Himoya nil tekshiruvlari ishlab chiqarishdagi paniclarni oldini oladi`,
			solutionCode: `package pointersx

import "strings"

type User struct {
	Name  string // ko'rsatiladigan foydalanuvchi ismi
	Email string // xom email manzili
}

type Counter struct {
	Value int // to'plangan butun son qiymati
}

type Document struct {
	Title    string            // hujjatning inson tomonidan o'qiladigan sarlavhasi
	Metadata map[string]string // ixtiyoriy metadata kalit/qiymat juftliklari
}

type Config struct {
	Host string // mijoz ulanishi kerak bo'lgan host
	Port int    // xizmat tomonidan ochilgan port
}

type Node struct {
	Value int   // ushbu ro'yxat tugunida saqlangan ma'lumotlar
	Next  *Node // zanjirdagi keyingi elementga ko'rsatkich
}

func ZeroInt(p *int) {
	if p == nil { // panic dan qochish uchun nil ko'rsatkichdan himoyalanish
		return
	}
	*p = 0 // ko'rsatilgan qiymatni nol bilan qayta yozish
}

func SwapStrings(a, b *string) {
	if a == nil || b == nil { // almashtirish faqat ikkala ko'rsatkich yaroqli bo'lganda mumkin
		return
	}
	*a, *b = *b, *a // parallel tayinlash orqali tuple almashishni bajarish
}

func ApplyDiscount(price *float64, percent float64) {
	if price == nil { // ko'rsatkich nil bo'lganda chegirma qo'llash uchun hech narsa yo'q
		return
	}
	discount := *price * percent / 100 // narxning ulushi sifatida chegirma miqdorini hisoblash
	*price = *price - discount         // asl narxdan chegirmani ayirish
}

func NormalizeUser(u *User) {
	if u == nil { // nil pointer receiver uchun hech narsa qilmaslik
		return
	}
	u.Name = strings.TrimSpace(u.Name)   // ismdan bosh va oxirgi bo'shliqlarni olib tashlash
	u.Email = strings.ToLower(u.Email) // emailni normalizatsiya uchun kichik harflarga o'tkazish
}

func (c *Counter) Increment() {
	if c == nil { // nil ko'rsatkichda metod chaqiruvidan himoyalanish
		return
	}
	c.Value++ // ichki hisoblagich qiymatini birga oshirish
}

func (c *Counter) Add(delta int) int {
	if c == nil { // nil receiver holat saqlay olmaydi, neytral natija qaytarish
		return 0
	}
	c.Value += delta // delta ni qo'shib saqlangan qiymatni o'zgartirish
	return c.Value   // yangilangan qiymatni chaqiruvchiga xabar qilish
}

func EnsureMetadata(doc *Document) {
	if doc == nil { // nil ko'rsatkichni tuzatib bo'lmaydi, erta chiqish
		return
	}
	if doc.Metadata == nil { // yo'q bo'lganda metadata map ni ajratish
		doc.Metadata = make(map[string]string)
	}
}

func AttachTag(doc *Document, key, value string) {
	if doc == nil { // ko'rsatkich nil bo'lganda biriktiradigan narsa yo'q
		return
	}
	if doc.Metadata == nil { // metadata konteynerini dangasa ajratish
		doc.Metadata = make(map[string]string)
	}
	doc.Metadata[key] = value // berilgan kalit/qiymat juftligini metadatada saqlash
}

func EnsureConfig(cfg **Config) {
	if cfg == nil { // nil ikki darajali ko'rsatkich orqali o'zgartira olmaymiz
		return
	}
	if *cfg == nil { // asosiy ko'rsatkich yo'q bo'lganda config ni yaratish
		*cfg = &Config{
			Host: "localhost", // standart host qiymati
			Port: 8080,        // standart port qiymati
		}
	}
}

func WalkNodes(head *Node, visit func(*Node)) {
	if visit == nil { // tashrif callback talab qiladi, yo'q bo'lganda o'tkazib yuborish
		return
	}
	for node := head; node != nil; node = node.Next { // oxirigacha bog'langan ro'yxatdan iteratsiya
		visit(node) // har bir uchragan tugunda callback ni chaqirish
	}
}`
		}
	}
};

export default task;
