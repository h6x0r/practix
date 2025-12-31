import { Task } from '../../../../../../../types';

export const task: Task = {
	slug: 'se-grasp-high-cohesion',
	title: 'High Cohesion',
	difficulty: 'medium',
	tags: ['go', 'software-engineering', 'grasp', 'high-cohesion'],
	estimatedTime: '35m',
	isPremium: false,
	youtubeUrl: '',
	description: `Implement the High Cohesion principle - keep related functionality together and unrelated functionality separate.

**You will implement:**

1. **Customer struct** - Customer data (ID, Name, Email)
2. **CustomerRepository struct** - Database operations ONLY (Save, FindByID)
3. **CustomerValidator struct** - Validation logic ONLY (ValidateEmail, ValidateName)
4. Each class has ONE clear responsibility (high cohesion)

**Key Concepts:**
- **High Cohesion**: Each class has focused, related responsibilities
- **Single Responsibility**: One reason to change
- **Separation of Concerns**: Validation separate from persistence

**Example Usage:**

\`\`\`go
validator := NewCustomerValidator()
repository := NewCustomerRepository()

// Validator only validates
if err := validator.ValidateEmail("invalid-email"); err != nil {
    fmt.Println("Invalid email") // validation failed
}

// Repository only handles persistence
customer := &Customer{ID: 1, Name: "John", Email: "john@example.com"}
if err := repository.Save(customer); err != nil {
    log.Fatal(err)
}

found := repository.FindByID(1)
fmt.Println(found.Name) // John
\`\`\`

**Why High Cohesion?**
- **Focused Classes**: Easy to understand and maintain
- **Easy to Test**: Test validation and persistence separately
- **Reusability**: Use CustomerValidator in different contexts

**Anti-pattern (Don't do this):**
\`\`\`go
// LOW COHESION - BAD!
type CustomerService struct {
    // This class does EVERYTHING - low cohesion!
}

func (c *CustomerService) SaveCustomer(customer *Customer) error {
    // Validation logic
    if !strings.Contains(customer.Email, "@") {
        return errors.New("invalid email")
    }

    // Database logic
    db.Save(customer)

    // Email logic
    smtp.SendWelcomeEmail(customer.Email)

    // Logging logic
    log.Printf("Customer saved: %s", customer.Name)

    // Analytics logic
    analytics.Track("customer_created")

    return nil
}
// Too many responsibilities! Hard to test, hard to maintain!
\`\`\`

**Constraints:**
- CustomerRepository handles ONLY database operations
- CustomerValidator handles ONLY validation logic
- No mixing of validation and persistence`,
	initialCode: `package principles

import (
	"errors"
	"strings"
)

type Customer struct {
	ID    int
	Name  string
	Email string
}

type CustomerValidator struct{}

func NewCustomerValidator() *CustomerValidator {
	return &CustomerValidator{}
}

func (v *CustomerValidator) ValidateEmail(email string) error {
}

func (v *CustomerValidator) ValidateName(name string) error {
}

type CustomerRepository struct {
	customers map[int]*Customer
}

func NewCustomerRepository() *CustomerRepository {
	}
}

func (r *CustomerRepository) Save(customer *Customer) error {
}

func (r *CustomerRepository) FindByID(id int) *Customer {
}`,
	solutionCode: `package principles

import (
	"errors"
	"strings"
)

type Customer struct {
	ID    int
	Name  string
	Email string
}

// CustomerValidator - HIGH COHESION: only validation, nothing else
type CustomerValidator struct{}

func NewCustomerValidator() *CustomerValidator {
	return &CustomerValidator{}
}

func (v *CustomerValidator) ValidateEmail(email string) error {
	// Focused responsibility: email validation only
	if !strings.Contains(email, "@") {
		return errors.New("email must contain @")
	}
	return nil
}

func (v *CustomerValidator) ValidateName(name string) error {
	// Focused responsibility: name validation only
	if name == "" {
		return errors.New("name is required")
	}
	if len(name) < 2 {
		return errors.New("name must be at least 2 characters")
	}
	return nil
}

// CustomerRepository - HIGH COHESION: only persistence, nothing else
type CustomerRepository struct {
	customers map[int]*Customer	// simulates database
}

func NewCustomerRepository() *CustomerRepository {
	return &CustomerRepository{
		customers: make(map[int]*Customer),
	}
}

func (r *CustomerRepository) Save(customer *Customer) error {
	// Focused responsibility: just save to database
	// No validation, no logging, no email sending
	// Just pure persistence logic
	r.customers[customer.ID] = customer
	return nil
}

func (r *CustomerRepository) FindByID(id int) *Customer {
	// Focused responsibility: just retrieve from database
	// No business logic, no validation, no transformation
	// Just pure data retrieval
	return r.customers[id]	// returns nil if not found
}`,
	hint1: `ValidateEmail should check if email contains "@" using strings.Contains. ValidateName should check if name is not empty and len(name) >= 2.`,
	hint2: `Save should just do r.customers[customer.ID] = customer. FindByID should return r.customers[id].`,
	whyItMatters: `High Cohesion leads to maintainable, understandable, and testable code by keeping related things together.

**Why High Cohesion Matters:**

**1. Focused, Understandable Classes**
High cohesion classes are easy to understand because they do one thing well:

\`\`\`go
// HIGH COHESION - GOOD! Each class has one focus
type OrderValidator struct{}

func (v *OrderValidator) ValidateOrder(order *Order) error {
    if len(order.Items) == 0 {
        return errors.New("order must have items")
    }
    if order.Total <= 0 {
        return errors.New("order total must be positive")
    }
    return nil
}

type OrderRepository struct {
    db *sql.DB
}

func (r *OrderRepository) Save(order *Order) error {
    _, err := r.db.Exec("INSERT INTO orders (...) VALUES (...)")
    return err
}

type OrderNotifier struct {
    emailSender EmailSender
}

func (n *OrderNotifier) NotifyOrderCreated(order *Order) error {
    return n.emailSender.Send(order.CustomerEmail, "Order Confirmed", ...)
}

// LOW COHESION - BAD! One class does everything
type OrderManager struct {
    db          *sql.DB
    emailSender EmailSender
}

func (m *OrderManager) ProcessOrder(order *Order) error {
    // Validation
    if len(order.Items) == 0 { ... }

    // Persistence
    m.db.Exec("INSERT INTO orders ...")

    // Notification
    m.emailSender.Send(...)

    // Logging
    log.Printf("Order processed")

    // Analytics
    analytics.Track("order_created")

    // Too many unrelated responsibilities!
}
\`\`\`

**2. Easy to Test**
High cohesion makes unit testing simple and focused:

\`\`\`go
// Test validation independently
func TestOrderValidator_ValidateOrder(t *testing.T) {
    validator := NewOrderValidator()

    // Test empty order
    emptyOrder := &Order{Items: []Item{}}
    err := validator.ValidateOrder(emptyOrder)
    if err == nil {
        t.Error("expected error for empty order")
    }

    // Test valid order
    validOrder := &Order{Items: []Item{{ID: 1}}, Total: 100}
    err = validator.ValidateOrder(validOrder)
    if err != nil {
        t.Errorf("unexpected error: %v", err)
    }
}

// Test persistence independently (no need to test validation)
func TestOrderRepository_Save(t *testing.T) {
    repo := NewOrderRepository(testDB)
    order := &Order{ID: 1, Items: []Item{{ID: 1}}, Total: 100}

    err := repo.Save(order)
    if err != nil {
        t.Fatalf("save failed: %v", err)
    }

    saved := repo.FindByID(1)
    if saved.ID != order.ID {
        t.Errorf("expected ID %d, got %d", order.ID, saved.ID)
    }
}
\`\`\`

**3. Reusable Components**
High cohesion components can be reused in different contexts:

\`\`\`go
// EmailValidator is highly cohesive - only validates emails
type EmailValidator struct{}

func (v *EmailValidator) Validate(email string) error {
    if !strings.Contains(email, "@") {
        return errors.New("invalid email")
    }
    if !strings.Contains(email, ".") {
        return errors.New("invalid email")
    }
    return nil
}

// Can be reused anywhere!
func RegisterUser(username, email string) error {
    validator := &EmailValidator{}
    if err := validator.Validate(email); err != nil {
        return err
    }
    // ...
}

func UpdateProfile(userID int, email string) error {
    validator := &EmailValidator{}
    if err := validator.Validate(email); err != nil {
        return err
    }
    // ...
}

func InviteUser(email string) error {
    validator := &EmailValidator{}
    if err := validator.Validate(email); err != nil {
        return err
    }
    // ...
}
\`\`\`

**4. Real-World Example: User Management**
\`\`\`go
// HIGH COHESION design

// UserRepository - only data access
type UserRepository struct {
    db *sql.DB
}

func (r *UserRepository) Save(user *User) error {
    _, err := r.db.Exec("INSERT INTO users ...")
    return err
}

func (r *UserRepository) FindByEmail(email string) (*User, error) {
    var user User
    err := r.db.QueryRow("SELECT * FROM users WHERE email = ?", email).Scan(...)
    return &user, err
}

// PasswordHasher - only password hashing
type PasswordHasher struct{}

func (h *PasswordHasher) Hash(password string) (string, error) {
    return bcrypt.GenerateFromPassword([]byte(password), bcrypt.DefaultCost)
}

func (h *PasswordHasher) Compare(hashed, password string) error {
    return bcrypt.CompareHashAndPassword([]byte(hashed), []byte(password))
}

// UserValidator - only validation
type UserValidator struct{}

func (v *UserValidator) ValidateRegistration(username, email, password string) error {
    if len(username) < 3 {
        return errors.New("username too short")
    }
    if !strings.Contains(email, "@") {
        return errors.New("invalid email")
    }
    if len(password) < 8 {
        return errors.New("password too short")
    }
    return nil
}

// UserService - orchestrates the cohesive components
type UserService struct {
    repo     *UserRepository
    hasher   *PasswordHasher
    validator *UserValidator
}

func (s *UserService) RegisterUser(username, email, password string) error {
    // Each component does its focused job
    if err := s.validator.ValidateRegistration(username, email, password); err != nil {
        return err
    }

    hashed, err := s.hasher.Hash(password)
    if err != nil {
        return err
    }

    user := &User{Username: username, Email: email, PasswordHash: hashed}
    return s.repo.Save(user)
}
\`\`\`

**5. Measuring Cohesion**
Low cohesion signs:
- Class has unrelated methods
- Methods use different sets of fields
- Class name is vague (Manager, Helper, Util)
- Hard to describe what the class does in one sentence

High cohesion signs:
- All methods work with same data
- Clear, focused class name
- Easy to describe class purpose
- Methods are related to each other

**Common Mistakes:**
- Creating "God classes" that do everything
- Mixing presentation, business logic, and data access
- Using vague names like Manager, Handler, Util

**Rule of Thumb:**
If you can't describe a class's purpose in one clear sentence without using "and", it probably has low cohesion.`,
	order: 4,
	testCode: `package principles

import (
	"testing"
)

// Test1: ValidateEmail accepts valid email
func Test1(t *testing.T) {
	v := NewCustomerValidator()
	err := v.ValidateEmail("test@example.com")
	if err != nil {
		t.Errorf("Valid email should pass, got %v", err)
	}
}

// Test2: ValidateEmail rejects email without @
func Test2(t *testing.T) {
	v := NewCustomerValidator()
	err := v.ValidateEmail("invalid-email")
	if err == nil {
		t.Error("Should reject email without @")
	}
}

// Test3: ValidateName accepts valid name
func Test3(t *testing.T) {
	v := NewCustomerValidator()
	err := v.ValidateName("John")
	if err != nil {
		t.Errorf("Valid name should pass, got %v", err)
	}
}

// Test4: ValidateName rejects empty name
func Test4(t *testing.T) {
	v := NewCustomerValidator()
	err := v.ValidateName("")
	if err == nil {
		t.Error("Should reject empty name")
	}
}

// Test5: ValidateName rejects single character
func Test5(t *testing.T) {
	v := NewCustomerValidator()
	err := v.ValidateName("J")
	if err == nil {
		t.Error("Should reject name with less than 2 chars")
	}
}

// Test6: Repository Save and FindByID
func Test6(t *testing.T) {
	r := NewCustomerRepository()
	c := &Customer{ID: 1, Name: "John", Email: "john@test.com"}
	r.Save(c)
	found := r.FindByID(1)
	if found == nil || found.Name != "John" {
		t.Error("Should find saved customer")
	}
}

// Test7: Repository FindByID returns nil for non-existent
func Test7(t *testing.T) {
	r := NewCustomerRepository()
	found := r.FindByID(999)
	if found != nil {
		t.Error("Should return nil for non-existent ID")
	}
}

// Test8: Customer struct fields
func Test8(t *testing.T) {
	c := Customer{ID: 1, Name: "Test", Email: "test@test.com"}
	if c.ID != 1 || c.Name != "Test" || c.Email != "test@test.com" {
		t.Error("Customer fields not set correctly")
	}
}

// Test9: ValidateName accepts exactly 2 chars
func Test9(t *testing.T) {
	v := NewCustomerValidator()
	err := v.ValidateName("Jo")
	if err != nil {
		t.Errorf("Name with 2 chars should pass, got %v", err)
	}
}

// Test10: Multiple customers can be saved
func Test10(t *testing.T) {
	r := NewCustomerRepository()
	r.Save(&Customer{ID: 1, Name: "A", Email: "a@test.com"})
	r.Save(&Customer{ID: 2, Name: "B", Email: "b@test.com"})
	r.Save(&Customer{ID: 3, Name: "C", Email: "c@test.com"})
	if r.FindByID(1) == nil || r.FindByID(2) == nil || r.FindByID(3) == nil {
		t.Error("All customers should be found")
	}
}
`,
	translations: {
		ru: {
			title: 'Высокая связность',
			description: `Реализуйте принцип Высокой связности — держите связанную функциональность вместе, а несвязанную — отдельно.

**Вы реализуете:**

1. **Customer struct** — Данные клиента (ID, Name, Email)
2. **CustomerRepository struct** — ТОЛЬКО операции с БД (Save, FindByID)
3. **CustomerValidator struct** — ТОЛЬКО логика валидации (ValidateEmail, ValidateName)
4. Каждый класс имеет ОДНУ чёткую ответственность (высокая связность)

**Ключевые концепции:**
- **Высокая связность**: Каждый класс имеет сфокусированные, связанные обязанности
- **Единственная ответственность**: Одна причина для изменения
- **Разделение ответственностей**: Валидация отдельно от персистентности

**Зачем нужна Высокая связность?**
- **Сфокусированные классы**: Легко понимать и поддерживать
- **Легко тестировать**: Тестируйте валидацию и персистентность отдельно
- **Переиспользуемость**: Используйте CustomerValidator в разных контекстах

**Ограничения:**
- CustomerRepository обрабатывает ТОЛЬКО операции с БД
- CustomerValidator обрабатывает ТОЛЬКО логику валидации
- Никакого смешивания валидации и персистентности`,
			hint1: `ValidateEmail должен проверить наличие "@" через strings.Contains. ValidateName должен проверить, что имя не пустое и len(name) >= 2.`,
			hint2: `Save должен просто делать r.customers[customer.ID] = customer. FindByID должен возвращать r.customers[id].`,
			whyItMatters: `Высокая связность приводит к поддерживаемому, понятному и тестируемому коду, держа связанное вместе.

**Почему Высокая связность важна:**

**1. Сфокусированные, понятные классы**
Классы с высокой связностью легко понимать, потому что они делают одну вещь хорошо.

**Распространённые ошибки:**
- Создание "божественных классов", которые делают всё
- Смешивание представления, бизнес-логики и доступа к данным
- Использование неопределённых имён вроде Manager, Handler, Util`,
			solutionCode: `package principles

import (
	"errors"
	"strings"
)

type Customer struct {
	ID    int
	Name  string
	Email string
}

// CustomerValidator - ВЫСОКАЯ СВЯЗНОСТЬ: только валидация, ничего больше
type CustomerValidator struct{}

func NewCustomerValidator() *CustomerValidator {
	return &CustomerValidator{}
}

func (v *CustomerValidator) ValidateEmail(email string) error {
	// Сфокусированная ответственность: только валидация email
	if !strings.Contains(email, "@") {
		return errors.New("email must contain @")
	}
	return nil
}

func (v *CustomerValidator) ValidateName(name string) error {
	// Сфокусированная ответственность: только валидация имени
	if name == "" {
		return errors.New("name is required")
	}
	if len(name) < 2 {
		return errors.New("name must be at least 2 characters")
	}
	return nil
}

// CustomerRepository - ВЫСОКАЯ СВЯЗНОСТЬ: только персистентность, ничего больше
type CustomerRepository struct {
	customers map[int]*Customer	// имитирует БД
}

func NewCustomerRepository() *CustomerRepository {
	return &CustomerRepository{
		customers: make(map[int]*Customer),
	}
}

func (r *CustomerRepository) Save(customer *Customer) error {
	// Сфокусированная ответственность: только сохранение в БД
	// Никакой валидации, никакого логирования, никакой отправки email
	// Только чистая логика персистентности
	r.customers[customer.ID] = customer
	return nil
}

func (r *CustomerRepository) FindByID(id int) *Customer {
	// Сфокусированная ответственность: только извлечение из БД
	// Никакой бизнес-логики, никакой валидации, никакого преобразования
	// Только чистое извлечение данных
	return r.customers[id]	// возвращает nil если не найден
}`
		},
		uz: {
			title: 'High Cohesion (Yuqori birlik)',
			description: `High Cohesion prinsipini amalga oshiring — bog'liq funksionallikni birga saqlang va bog'liq bo'lmagan funksionallikni alohida saqlang.

**Siz amalga oshirasiz:**

1. **Customer struct** — Mijoz ma'lumotlari (ID, Name, Email)
2. **CustomerRepository struct** — FAQAT ma'lumotlar bazasi operatsiyalari (Save, FindByID)
3. **CustomerValidator struct** — FAQAT validatsiya mantiqi (ValidateEmail, ValidateName)
4. Har bir klass BITTA aniq mas'uliyatga ega (yuqori birlik)

**Asosiy tushunchalar:**
- **High Cohesion**: Har bir klass fokusli, bog'liq mas'uliyatlarga ega
- **Yagona mas'uliyat**: O'zgartirish uchun bitta sabab
- **Mas'uliyatlarni ajratish**: Validatsiya persistentlikdan alohida

**Nima uchun High Cohesion?**
- **Fokusli klasslar**: Tushunish va parvarish qilish oson
- **Test qilish oson**: Validatsiya va persistentlikni alohida test qilish
- **Qayta ishlatish**: CustomerValidator ni turli kontekstlarda ishlatish

**Cheklovlar:**
- CustomerRepository FAQAT ma'lumotlar bazasi operatsiyalarini boshqaradi
- CustomerValidator FAQAT validatsiya mantiqini boshqaradi
- Validatsiya va persistentlikni aralashtirib yubormaslik`,
			hint1: `ValidateEmail strings.Contains orqali "@" mavjudligini tekshirishi kerak. ValidateName nom bo'sh emasligini va len(name) >= 2 ekanligini tekshirishi kerak.`,
			hint2: `Save faqat r.customers[customer.ID] = customer qilishi kerak. FindByID r.customers[id] ni qaytarishi kerak.`,
			whyItMatters: `High Cohesion bog'liq narsalarni birga saqlash orqali parvarish qilinadigan, tushunarli va testlangan kodga olib keladi.

**High Cohesion nima uchun muhim:**

**1. Fokusli, tushunarli klasslar**
Yuqori birlikka ega klasslar tushunish oson chunki ular bitta narsani yaxshi qiladi.

**Umumiy xatolar:**
- Hamma narsani qiladigan "Xudo klasslari" yaratish
- Taqdimot, biznes mantiqi va ma'lumotlar kirishini aralashtirib yuborish
- Manager, Handler, Util kabi noaniq nomlardan foydalanish`,
			solutionCode: `package principles

import (
	"errors"
	"strings"
)

type Customer struct {
	ID    int
	Name  string
	Email string
}

// CustomerValidator - YUQORI BIRLIK: faqat validatsiya, boshqa hech narsa
type CustomerValidator struct{}

func NewCustomerValidator() *CustomerValidator {
	return &CustomerValidator{}
}

func (v *CustomerValidator) ValidateEmail(email string) error {
	// Fokusli mas'uliyat: faqat email validatsiyasi
	if !strings.Contains(email, "@") {
		return errors.New("email must contain @")
	}
	return nil
}

func (v *CustomerValidator) ValidateName(name string) error {
	// Fokusli mas'uliyat: faqat nom validatsiyasi
	if name == "" {
		return errors.New("name is required")
	}
	if len(name) < 2 {
		return errors.New("name must be at least 2 characters")
	}
	return nil
}

// CustomerRepository - YUQORI BIRLIK: faqat persistentlik, boshqa hech narsa
type CustomerRepository struct {
	customers map[int]*Customer	// ma'lumotlar bazasini simulyatsiya qiladi
}

func NewCustomerRepository() *CustomerRepository {
	return &CustomerRepository{
		customers: make(map[int]*Customer),
	}
}

func (r *CustomerRepository) Save(customer *Customer) error {
	// Fokusli mas'uliyat: faqat ma'lumotlar bazasiga saqlash
	// Validatsiya yo'q, logging yo'q, email yuborish yo'q
	// Faqat sof persistentlik mantiqi
	r.customers[customer.ID] = customer
	return nil
}

func (r *CustomerRepository) FindByID(id int) *Customer {
	// Fokusli mas'uliyat: faqat ma'lumotlar bazasidan olish
	// Biznes mantiqi yo'q, validatsiya yo'q, transformatsiya yo'q
	// Faqat sof ma'lumotlarni olish
	return r.customers[id]	// topilmasa nil qaytaradi
}`
		}
	}
};

export default task;
