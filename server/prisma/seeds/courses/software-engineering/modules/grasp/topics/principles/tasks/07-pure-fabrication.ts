import { Task } from '../../../../../../../types';

export const task: Task = {
	slug: 'se-grasp-pure-fabrication',
	title: 'Pure Fabrication',
	difficulty: 'medium',
	tags: ['go', 'software-engineering', 'grasp', 'pure-fabrication'],
	estimatedTime: '40m',
	isPremium: false,
	youtubeUrl: '',
	description: `Implement the Pure Fabrication principle - create artificial classes that don't represent domain concepts to achieve low coupling and high cohesion.

**You will implement:**

1. **Book struct** - Domain object (ID, Title, Author)
2. **BookRepository** - Pure Fabrication for database operations
3. **BookSerializer** - Pure Fabrication for JSON serialization
4. **BookLogger** - Pure Fabrication for logging operations

**Key Concepts:**
- **Pure Fabrication**: Made-up class not from domain model
- **Service Classes**: Helper classes for technical concerns
- **Separation**: Domain logic separate from infrastructure

**Example Usage:**

\`\`\`go
book := &Book{ID: 1, Title: "Clean Code", Author: "Robert Martin"}

// Repository - pure fabrication for database operations
repo := NewBookRepository()
repo.Save(book)
saved := repo.FindByID(1)

// Serializer - pure fabrication for JSON operations
serializer := NewBookSerializer()
json := serializer.ToJSON(book)
// Output: {"id":1,"title":"Clean Code","author":"Robert Martin"}

restored := serializer.FromJSON(json)

// Logger - pure fabrication for logging
logger := NewBookLogger()
logger.LogBookCreated(book)
// Output: [LOG] Book created: Clean Code by Robert Martin
\`\`\`

**Why Pure Fabrication?**
- **Low Coupling**: Domain objects don't depend on infrastructure
- **High Cohesion**: Each fabrication has one clear purpose
- **Testability**: Easy to test domain objects without infrastructure

**Anti-pattern (Don't do this):**
\`\`\`go
// BAD: Domain object handles infrastructure concerns
type Book struct {
    ID     int
    Title  string
    Author string
    db     *sql.DB  // Book shouldn't know about database!
}

func (b *Book) Save() error {
    // Domain object doing database work - BAD!
    _, err := b.db.Exec("INSERT INTO books ...")
    return err
}

func (b *Book) ToJSON() string {
    // Domain object doing serialization - BAD!
    return fmt.Sprintf("{\"id\":%d,\"title\":\"%s\"}", b.ID, b.Title)
}
// High coupling, low cohesion!
\`\`\`

**Constraints:**
- Book struct must be pure domain object (no infrastructure code)
- Repository, Serializer, and Logger are pure fabrications
- Each fabrication has single, focused responsibility`,
	initialCode: `package principles

import (
	"encoding/json"
	"fmt"
)

// Book is a pure domain object - no infrastructure concerns
type Book struct {
	ID     int    \`json:"id"\`
	Title  string \`json:"title"\`
	Author string \`json:"author"\`
}

// BookRepository is a PURE FABRICATION
// It doesn't represent a domain concept - it's a technical solution for persistence
type BookRepository struct {
	books map[int]*Book
}

// NewBookRepository creates a new repository
func NewBookRepository() *BookRepository {
	return &BookRepository{
		books: make(map[int]*Book),
	}
}

// TODO: Implement Save
// Store book in the map (simulating database)
// This is a pure fabrication - Book doesn't need to know how it's stored
func (r *BookRepository) Save(book *Book) error {
	panic("TODO: implement Save")
}

// TODO: Implement FindByID
// Retrieve book from map by ID
// Return nil if not found
func (r *BookRepository) FindByID(id int) *Book {
	panic("TODO: implement FindByID")
}

// BookSerializer is a PURE FABRICATION
// It doesn't represent a domain concept - it's a technical solution for serialization
type BookSerializer struct{}

// NewBookSerializer creates a new serializer
func NewBookSerializer() *BookSerializer {
	return &BookSerializer{}
}

// TODO: Implement ToJSON
// Convert book to JSON string
// Use json.Marshal to convert book to JSON bytes
// Return string(jsonBytes)
// This is a pure fabrication - Book doesn't need to know about JSON
func (s *BookSerializer) ToJSON(book *Book) string {
	panic("TODO: implement ToJSON")
}

// TODO: Implement FromJSON
// Convert JSON string to Book
// Use json.Unmarshal to convert JSON bytes to book
// Return the book (return nil on error for simplicity)
func (s *BookSerializer) FromJSON(jsonStr string) *Book {
	panic("TODO: implement FromJSON")
}

// BookLogger is a PURE FABRICATION
// It doesn't represent a domain concept - it's a technical solution for logging
type BookLogger struct{}

// NewBookLogger creates a new logger
func NewBookLogger() *BookLogger {
	return &BookLogger{}
}

// TODO: Implement LogBookCreated
// Print a log message: "[LOG] Book created: TITLE by AUTHOR"
// Use fmt.Printf
// This is a pure fabrication - Book doesn't need to know about logging
func (l *BookLogger) LogBookCreated(book *Book) {
	panic("TODO: implement LogBookCreated")
}`,
	solutionCode: `package principles

import (
	"encoding/json"
	"fmt"
)

// Book is a pure domain object
// No database, no JSON, no logging concerns - just domain data
type Book struct {
	ID     int    \`json:"id"\`
	Title  string \`json:"title"\`
	Author string \`json:"author"\`
}

// BookRepository is a PURE FABRICATION for persistence
type BookRepository struct {
	books map[int]*Book
}

func NewBookRepository() *BookRepository {
	return &BookRepository{
		books: make(map[int]*Book),
	}
}

func (r *BookRepository) Save(book *Book) error {
	// Pure fabrication handles persistence
	// Book doesn't need to know about storage
	r.books[book.ID] = book
	return nil
}

func (r *BookRepository) FindByID(id int) *Book {
	// Pure fabrication handles retrieval
	return r.books[id]
}

// BookSerializer is a PURE FABRICATION for serialization
type BookSerializer struct{}

func NewBookSerializer() *BookSerializer {
	return &BookSerializer{}
}

func (s *BookSerializer) ToJSON(book *Book) string {
	// Pure fabrication handles JSON conversion
	// Book doesn't need to know about JSON format
	jsonBytes, err := json.Marshal(book)
	if err != nil {
		return ""
	}
	return string(jsonBytes)
}

func (s *BookSerializer) FromJSON(jsonStr string) *Book {
	// Pure fabrication handles JSON parsing
	var book Book
	err := json.Unmarshal([]byte(jsonStr), &book)
	if err != nil {
		return nil
	}
	return &book
}

// BookLogger is a PURE FABRICATION for logging
type BookLogger struct{}

func NewBookLogger() *BookLogger {
	return &BookLogger{}
}

func (l *BookLogger) LogBookCreated(book *Book) {
	// Pure fabrication handles logging
	// Book doesn't need to know about log format
	fmt.Printf("[LOG] Book created: %s by %s\n", book.Title, book.Author)
}`,
	hint1: `Save: r.books[book.ID] = book. FindByID: return r.books[id]. These are simple storage operations.`,
	hint2: `ToJSON: use json.Marshal(book) then convert to string. FromJSON: use json.Unmarshal([]byte(jsonStr), &book). LogBookCreated: use fmt.Printf with format string.`,
	whyItMatters: `Pure Fabrication keeps domain objects clean by moving infrastructure concerns to dedicated service classes.

**Why Pure Fabrication Matters:**

**1. Keeps Domain Objects Pure**
Domain objects focus on business logic, not technical concerns:

\`\`\`go
// WITHOUT PURE FABRICATION - BAD!
type Customer struct {
    ID       int
    Name     string
    Email    string
    db       *sql.DB        // infrastructure leak!
    logger   *log.Logger    // infrastructure leak!
    validator *Validator    // infrastructure leak!
}

func (c *Customer) Save() error {
    // Domain object mixed with database code - BAD!
    _, err := c.db.Exec("INSERT INTO customers ...", c.ID, c.Name, c.Email)
    c.logger.Printf("Customer saved: %s", c.Name)
    return err
}

func (c *Customer) Validate() error {
    // Domain object mixed with validation framework - BAD!
    return c.validator.Validate(c)
}

// WITH PURE FABRICATION - GOOD!
type Customer struct {
    ID    int
    Name  string
    Email string
    // Pure domain object - no infrastructure!
}

// Pure fabrications for infrastructure
type CustomerRepository struct {
    db *sql.DB
}

func (r *CustomerRepository) Save(customer *Customer) error {
    _, err := r.db.Exec("INSERT INTO customers ...", customer.ID, customer.Name, customer.Email)
    return err
}

type CustomerLogger struct {
    logger *log.Logger
}

func (l *CustomerLogger) LogCustomerSaved(customer *Customer) {
    l.logger.Printf("Customer saved: %s", customer.Name)
}

type CustomerValidator struct{}

func (v *CustomerValidator) Validate(customer *Customer) error {
    if customer.Name == "" {
        return errors.New("name required")
    }
    if !strings.Contains(customer.Email, "@") {
        return errors.New("invalid email")
    }
    return nil
}
\`\`\`

**2. Real-World Example: E-commerce Order**
\`\`\`go
// Pure domain object
type Order struct {
    ID         int
    CustomerID int
    Items      []OrderItem
    Total      float64
    Status     string
}

func (o *Order) CalculateTotal() float64 {
    // Domain logic - belongs in Order
    total := 0.0
    for _, item := range o.Items {
        total += item.Price * float64(item.Quantity)
    }
    return total
}

// Pure fabrication for persistence
type OrderRepository struct {
    db *sql.DB
}

func (r *OrderRepository) Save(order *Order) error {
    query := "INSERT INTO orders (id, customer_id, total, status) VALUES (?, ?, ?, ?)"
    _, err := r.db.Exec(query, order.ID, order.CustomerID, order.Total, order.Status)
    return err
}

func (r *OrderRepository) FindByID(id int) (*Order, error) {
    var order Order
    query := "SELECT id, customer_id, total, status FROM orders WHERE id = ?"
    err := r.db.QueryRow(query, id).Scan(&order.ID, &order.CustomerID, &order.Total, &order.Status)
    return &order, err
}

// Pure fabrication for email notifications
type OrderEmailNotifier struct {
    emailSender EmailSender
}

func (n *OrderEmailNotifier) SendOrderConfirmation(order *Order) error {
    subject := "Order Confirmation"
    body := fmt.Sprintf("Your order #%d has been confirmed. Total: $%.2f", order.ID, order.Total)
    return n.emailSender.Send(order.CustomerEmail, subject, body)
}

// Pure fabrication for PDF generation
type OrderPDFGenerator struct{}

func (g *OrderPDFGenerator) GenerateInvoice(order *Order) ([]byte, error) {
    // Generate PDF invoice for order
    pdf := &PDF{}
    pdf.AddText(fmt.Sprintf("Invoice for Order #%d", order.ID))
    pdf.AddText(fmt.Sprintf("Total: $%.2f", order.Total))
    return pdf.ToBytes(), nil
}

// Pure fabrication for caching
type OrderCache struct {
    cache map[int]*Order
}

func (c *OrderCache) Get(id int) *Order {
    return c.cache[id]
}

func (c *OrderCache) Set(order *Order) {
    c.cache[order.ID] = order
}

// Order service coordinates pure fabrications
type OrderService struct {
    repository *OrderRepository
    cache      *OrderCache
    notifier   *OrderEmailNotifier
    pdfGen     *OrderPDFGenerator
}

func (s *OrderService) CreateOrder(order *Order) error {
    // Calculate domain logic
    order.Total = order.CalculateTotal()
    order.Status = "pending"

    // Use pure fabrications for infrastructure
    if err := s.repository.Save(order); err != nil {
        return err
    }

    s.cache.Set(order)
    s.notifier.SendOrderConfirmation(order)

    return nil
}
\`\`\`

**3. When to Create Pure Fabrications**
Create pure fabrications for:
- **Persistence**: Repositories, DAOs
- **Serialization**: JSON, XML, Protocol Buffer converters
- **Logging**: Loggers, audit trails
- **Validation**: Validators (unless core domain logic)
- **Caching**: Cache managers
- **External APIs**: API clients, adapters
- **Security**: Authentication, authorization services

Don't create pure fabrications for:
- **Core business logic**: Belongs in domain objects
- **Simple calculations**: Keep with domain objects
- **Domain rules**: Should be in domain layer

**4. Testing Benefits**
\`\`\`go
// Easy to test domain object without infrastructure
func TestOrder_CalculateTotal(t *testing.T) {
    order := &Order{
        Items: []OrderItem{
            {Price: 10.0, Quantity: 2},
            {Price: 5.0, Quantity: 3},
        },
    }

    total := order.CalculateTotal()
    expected := 35.0

    if total != expected {
        t.Errorf("expected %.2f, got %.2f", expected, total)
    }
}
// No database, no email, no PDF - just pure logic!

// Test infrastructure separately
func TestOrderRepository_Save(t *testing.T) {
    db := setupTestDB()
    repo := &OrderRepository{db: db}

    order := &Order{ID: 1, Total: 100.0, Status: "pending"}
    err := repo.Save(order)

    if err != nil {
        t.Fatalf("save failed: %v", err)
    }

    saved, _ := repo.FindByID(1)
    if saved.Total != 100.0 {
        t.Errorf("expected 100.0, got %.2f", saved.Total)
    }
}
\`\`\`

**Common Mistakes:**
- Putting infrastructure code in domain objects
- Not recognizing when to create pure fabrication
- Creating too many fabrications (over-engineering)
- Mixing domain logic with infrastructure in service layer

**Rule of Thumb:**
If a class doesn't represent a real-world domain concept but solves a technical problem, it's likely a pure fabrication.`,
	order: 6,
	testCode: `package principles

import (
	"strings"
	"testing"
)

// Test1: BookRepository Save and FindByID
func Test1(t *testing.T) {
	repo := NewBookRepository()
	book := &Book{ID: 1, Title: "Test", Author: "Author"}
	repo.Save(book)
	found := repo.FindByID(1)
	if found == nil || found.Title != "Test" {
		t.Error("Should find saved book")
	}
}

// Test2: BookRepository FindByID returns nil for non-existent
func Test2(t *testing.T) {
	repo := NewBookRepository()
	found := repo.FindByID(999)
	if found != nil {
		t.Error("Should return nil for non-existent ID")
	}
}

// Test3: BookSerializer ToJSON returns valid JSON
func Test3(t *testing.T) {
	s := NewBookSerializer()
	book := &Book{ID: 1, Title: "Clean Code", Author: "Robert"}
	json := s.ToJSON(book)
	if !strings.Contains(json, "Clean Code") {
		t.Error("JSON should contain title")
	}
}

// Test4: BookSerializer FromJSON parses correctly
func Test4(t *testing.T) {
	s := NewBookSerializer()
	json := "{\"id\":1,\"title\":\"Test\",\"author\":\"Author\"}"
	book := s.FromJSON(json)
	if book == nil || book.Title != "Test" {
		t.Error("Should parse JSON to book")
	}
}

// Test5: BookLogger LogBookCreated runs without panic
func Test5(t *testing.T) {
	l := NewBookLogger()
	book := &Book{ID: 1, Title: "Test", Author: "Author"}
	l.LogBookCreated(book) // Should not panic
}

// Test6: Book struct fields
func Test6(t *testing.T) {
	book := Book{ID: 1, Title: "Test", Author: "Tester"}
	if book.ID != 1 || book.Title != "Test" || book.Author != "Tester" {
		t.Error("Book fields not set correctly")
	}
}

// Test7: BookSerializer round-trip
func Test7(t *testing.T) {
	s := NewBookSerializer()
	original := &Book{ID: 5, Title: "Round", Author: "Trip"}
	json := s.ToJSON(original)
	restored := s.FromJSON(json)
	if restored == nil || restored.ID != 5 || restored.Title != "Round" {
		t.Error("Round-trip should preserve data")
	}
}

// Test8: BookRepository handles multiple books
func Test8(t *testing.T) {
	repo := NewBookRepository()
	repo.Save(&Book{ID: 1, Title: "A", Author: "A"})
	repo.Save(&Book{ID: 2, Title: "B", Author: "B"})
	if repo.FindByID(1) == nil || repo.FindByID(2) == nil {
		t.Error("Both books should be found")
	}
}

// Test9: BookSerializer handles empty strings
func Test9(t *testing.T) {
	s := NewBookSerializer()
	book := &Book{ID: 1, Title: "", Author: ""}
	json := s.ToJSON(book)
	if json == "" {
		t.Error("Should serialize book with empty fields")
	}
}

// Test10: BookSerializer FromJSON returns nil on invalid JSON
func Test10(t *testing.T) {
	s := NewBookSerializer()
	book := s.FromJSON("invalid json")
	if book != nil {
		t.Error("Should return nil for invalid JSON")
	}
}
`,
	translations: {
		ru: {
			title: 'Чистая выдумка',
			description: `Реализуйте принцип Чистой выдумки — создавайте искусственные классы, не представляющие доменные концепции, для достижения низкой связанности и высокой связности.

**Вы реализуете:**

1. **Book struct** — Доменный объект (ID, Title, Author)
2. **BookRepository** — Чистая выдумка для операций с БД
3. **BookSerializer** — Чистая выдумка для JSON-сериализации
4. **BookLogger** — Чистая выдумка для операций логирования

**Ключевые концепции:**
- **Чистая выдумка**: Придуманный класс не из доменной модели
- **Сервисные классы**: Вспомогательные классы для технических задач
- **Разделение**: Доменная логика отдельна от инфраструктуры

**Зачем нужна Чистая выдумка?**
- **Низкая связанность**: Доменные объекты не зависят от инфраструктуры
- **Высокая связность**: Каждая выдумка имеет одну чёткую цель
- **Тестируемость**: Легко тестировать доменные объекты без инфраструктуры

**Ограничения:**
- Book struct должна быть чистым доменным объектом (без инфраструктурного кода)
- Repository, Serializer и Logger — чистые выдумки
- Каждая выдумка имеет единственную, сфокусированную ответственность`,
			hint1: `Save: r.books[book.ID] = book. FindByID: return r.books[id]. Это простые операции хранения.`,
			hint2: `ToJSON: используйте json.Marshal(book), затем преобразуйте в string. FromJSON: используйте json.Unmarshal([]byte(jsonStr), &book). LogBookCreated: используйте fmt.Printf с форматной строкой.`,
			whyItMatters: `Чистая выдумка сохраняет доменные объекты чистыми, перемещая инфраструктурные задачи в выделенные сервисные классы.

**Почему Чистая выдумка важна:**

**1. Сохраняет доменные объекты чистыми**
Доменные объекты фокусируются на бизнес-логике, а не на технических задачах.

**Распространённые ошибки:**
- Размещение инфраструктурного кода в доменных объектах
- Непонимание когда создавать чистую выдумку
- Создание слишком много выдумок (излишняя инженерия)
- Смешивание доменной логики с инфраструктурой в слое сервисов`,
			solutionCode: `package principles

import (
	"encoding/json"
	"fmt"
)

// Book - чистый доменный объект
// Никаких БД, JSON, логирования - только доменные данные
type Book struct {
	ID     int    \`json:"id"\`
	Title  string \`json:"title"\`
	Author string \`json:"author"\`
}

// BookRepository - ЧИСТАЯ ВЫДУМКА для персистентности
type BookRepository struct {
	books map[int]*Book
}

func NewBookRepository() *BookRepository {
	return &BookRepository{
		books: make(map[int]*Book),
	}
}

func (r *BookRepository) Save(book *Book) error {
	// Чистая выдумка обрабатывает персистентность
	// Book не нужно знать о хранении
	r.books[book.ID] = book
	return nil
}

func (r *BookRepository) FindByID(id int) *Book {
	// Чистая выдумка обрабатывает извлечение
	return r.books[id]
}

// BookSerializer - ЧИСТАЯ ВЫДУМКА для сериализации
type BookSerializer struct{}

func NewBookSerializer() *BookSerializer {
	return &BookSerializer{}
}

func (s *BookSerializer) ToJSON(book *Book) string {
	// Чистая выдумка обрабатывает преобразование в JSON
	// Book не нужно знать о формате JSON
	jsonBytes, err := json.Marshal(book)
	if err != nil {
		return ""
	}
	return string(jsonBytes)
}

func (s *BookSerializer) FromJSON(jsonStr string) *Book {
	// Чистая выдумка обрабатывает парсинг JSON
	var book Book
	err := json.Unmarshal([]byte(jsonStr), &book)
	if err != nil {
		return nil
	}
	return &book
}

// BookLogger - ЧИСТАЯ ВЫДУМКА для логирования
type BookLogger struct{}

func NewBookLogger() *BookLogger {
	return &BookLogger{}
}

func (l *BookLogger) LogBookCreated(book *Book) {
	// Чистая выдумка обрабатывает логирование
	// Book не нужно знать о формате логов
	fmt.Printf("[LOG] Book created: %s by %s\n", book.Title, book.Author)
}`
		},
		uz: {
			title: 'Pure Fabrication (Sof ixtiro)',
			description: `Pure Fabrication prinsipini amalga oshiring — past bog'lanish va yuqori birlikka erishish uchun domen kontseptsiyalarini ifodalamovchi sun'iy klasslar yarating.

**Siz amalga oshirasiz:**

1. **Book struct** — Domen ob'ekti (ID, Title, Author)
2. **BookRepository** — Ma'lumotlar bazasi operatsiyalari uchun sof ixtiro
3. **BookSerializer** — JSON serializatsiyasi uchun sof ixtiro
4. **BookLogger** — Logging operatsiyalari uchun sof ixtiro

**Asosiy tushunchalar:**
- **Pure Fabrication**: Domen modelidan emas, o'ylab topilgan klass
- **Servis klasslari**: Texnik vazifalar uchun yordamchi klasslar
- **Ajratish**: Domen mantiqi infrastrukturadan alohida

**Nima uchun Pure Fabrication?**
- **Past bog'lanish**: Domen ob'ektlari infrastrukturaga bog'liq emas
- **Yuqori birlik**: Har bir ixtiro bitta aniq maqsadga ega
- **Testlanish**: Infrastrukturasiz domen ob'ektlarini osongina test qilish

**Cheklovlar:**
- Book struct sof domen ob'ekti bo'lishi kerak (infrastruktura kodsiz)
- Repository, Serializer va Logger sof ixtirolar
- Har bir ixtiro yagona, fokusli mas'uliyatga ega`,
			hint1: `Save: r.books[book.ID] = book. FindByID: return r.books[id]. Bular oddiy saqlash operatsiyalari.`,
			hint2: `ToJSON: json.Marshal(book) dan foydalaning, keyin stringga o'giring. FromJSON: json.Unmarshal([]byte(jsonStr), &book) dan foydalaning. LogBookCreated: format stringi bilan fmt.Printf dan foydalaning.`,
			whyItMatters: `Pure Fabrication infrastruktura vazifalarini maxsus servis klasslariga ko'chirib, domen ob'ektlarini toza saqlaydi.

**Pure Fabrication nima uchun muhim:**

**1. Domen ob'ektlarini toza saqlaydi**
Domen ob'ektlari texnik vazifalar emas, biznes mantiqiga e'tibor qaratadi.

**Umumiy xatolar:**
- Infrastruktura kodini domen ob'ektlarida joylashtirish
- Qachon sof ixtiro yaratish kerakligini tushunmaslik
- Juda ko'p ixtirolar yaratish (ortiqcha muhandislik)
- Servis qatlamida domen mantiqini infrastruktura bilan aralashtirib yuborish`,
			solutionCode: `package principles

import (
	"encoding/json"
	"fmt"
)

// Book - sof domen ob'ekti
// Ma'lumotlar bazasi, JSON, logging yo'q - faqat domen ma'lumotlari
type Book struct {
	ID     int    \`json:"id"\`
	Title  string \`json:"title"\`
	Author string \`json:"author"\`
}

// BookRepository - persistentlik uchun SOF IXTIRO
type BookRepository struct {
	books map[int]*Book
}

func NewBookRepository() *BookRepository {
	return &BookRepository{
		books: make(map[int]*Book),
	}
}

func (r *BookRepository) Save(book *Book) error {
	// Sof ixtiro persistentlikni boshqaradi
	// Book saqlash haqida bilishi shart emas
	r.books[book.ID] = book
	return nil
}

func (r *BookRepository) FindByID(id int) *Book {
	// Sof ixtiro olishni boshqaradi
	return r.books[id]
}

// BookSerializer - serializatsiya uchun SOF IXTIRO
type BookSerializer struct{}

func NewBookSerializer() *BookSerializer {
	return &BookSerializer{}
}

func (s *BookSerializer) ToJSON(book *Book) string {
	// Sof ixtiro JSON konvertatsiyasini boshqaradi
	// Book JSON formati haqida bilishi shart emas
	jsonBytes, err := json.Marshal(book)
	if err != nil {
		return ""
	}
	return string(jsonBytes)
}

func (s *BookSerializer) FromJSON(jsonStr string) *Book {
	// Sof ixtiro JSON parslashtirish boshqaradi
	var book Book
	err := json.Unmarshal([]byte(jsonStr), &book)
	if err != nil {
		return nil
	}
	return &book
}

// BookLogger - logging uchun SOF IXTIRO
type BookLogger struct{}

func NewBookLogger() *BookLogger {
	return &BookLogger{}
}

func (l *BookLogger) LogBookCreated(book *Book) {
	// Sof ixtiro loggingni boshqaradi
	// Book log formati haqida bilishi shart emas
	fmt.Printf("[LOG] Book created: %s by %s\n", book.Title, book.Author)
}`
		}
	}
};

export default task;
