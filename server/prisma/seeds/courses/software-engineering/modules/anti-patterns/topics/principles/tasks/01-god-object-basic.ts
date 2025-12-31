import { Task } from '../../../../../../../types';

export const task: Task = {
	slug: 'se-ap-god-object-basic',
	title: 'God Object Anti-pattern - Basic',
	difficulty: 'easy',
	tags: ['go', 'anti-patterns', 'god-object', 'refactoring'],
	estimatedTime: '20m',
	isPremium: false,
	youtubeUrl: '',
	description: `Refactor a God Object that does too many things by splitting it into focused, single-responsibility components.

**The Problem:**

A God Object (also called Blob or Swiss Army Knife) is a class that knows too much or does too much. It violates the Single Responsibility Principle by handling multiple unrelated concerns.

**You will refactor:**

A monolithic \`UserManager\` that handles:
1. User data storage
2. Email sending
3. Password validation
4. Database operations

**Split into:**
1. **User struct** - Just the data
2. **UserRepository** - Database operations
3. **EmailService** - Email functionality
4. **PasswordValidator** - Password validation logic

**Example Usage:**

\`\`\`go
// After refactoring
repo := NewUserRepository()
emailSvc := NewEmailService()
validator := NewPasswordValidator()

user := User{ID: 1, Name: "Alice", Email: "alice@example.com"}
if validator.IsValid("Pass123!") {
    repo.Save(user)
    emailSvc.SendWelcome(user.Email)
}
\`\`\`

**Your Task:**

Complete the refactored implementation with proper separation of concerns.`,
	initialCode: `package antipatterns

type User struct {
	ID       int
	Name     string
	Email    string
	Password string
}

type UserRepository struct {
	users map[int]User
}

func NewUserRepository() *UserRepository {
}

func (r *UserRepository) Save(user User) {
}

func (r *UserRepository) FindByID(id int) User {
}

type EmailService struct{}

func NewEmailService() *EmailService {
}

func (s *EmailService) SendWelcome(email string) string {
}

type PasswordValidator struct {
	minLength int
}

func NewPasswordValidator() *PasswordValidator {
}

func (v *PasswordValidator) IsValid(password string) bool {
}`,
	solutionCode: `package antipatterns

// User represents a user entity - single responsibility: data model
type User struct {
	ID       int
	Name     string
	Email    string
	Password string
}

// UserRepository handles user data persistence - single responsibility: storage
type UserRepository struct {
	users map[int]User	// in-memory storage
}

func NewUserRepository() *UserRepository {
	return &UserRepository{
		users: make(map[int]User),	// initialize empty storage
	}
}

func (r *UserRepository) Save(user User) {
	r.users[user.ID] = user	// store user by ID
}

func (r *UserRepository) FindByID(id int) User {
	return r.users[id]	// returns zero value if not found
}

// EmailService handles email operations - single responsibility: notifications
type EmailService struct{}

func NewEmailService() *EmailService {
	return &EmailService{}
}

func (s *EmailService) SendWelcome(email string) string {
	return "Welcome email sent to: " + email	// simulated email sending
}

// PasswordValidator handles password validation - single responsibility: validation
type PasswordValidator struct {
	minLength int	// validation rule configuration
}

func NewPasswordValidator() *PasswordValidator {
	return &PasswordValidator{
		minLength: 8,	// standard minimum password length
	}
}

func (v *PasswordValidator) IsValid(password string) bool {
	return len(password) >= v.minLength	// simple length validation
}`,
	hint1: `For NewUserRepository, NewEmailService, and NewPasswordValidator, create and return a new instance with initialized fields. UserRepository needs make(map[int]User).`,
	hint2: `Save stores user in map, FindByID returns from map (zero value if missing). SendWelcome returns a formatted string. IsValid compares password length to minLength.`,
	whyItMatters: `God Objects are one of the most common and damaging anti-patterns in software development.

**Why God Objects are Problematic:**

**1. Violates Single Responsibility Principle**
A God Object does everything, making it impossible to maintain:

\`\`\`go
// BAD: God Object - UserManager does EVERYTHING
type UserManager struct {
    users map[int]User
    db *sql.DB
    smtpServer string
}

func (m *UserManager) CreateUser(name, email, password string) error {
    // Validates password
    if len(password) < 8 { return errors.New("weak password") }

    // Creates user
    user := User{Name: name, Email: email}

    // Saves to database
    m.db.Exec("INSERT INTO users...", user)

    // Sends email
    smtp.SendEmail(m.smtpServer, email, "Welcome!")

    // Logs activity
    log.Printf("User created: %s", name)

    return nil
}
// This function does FIVE different things! Any change affects everything.
\`\`\`

**2. Hard to Test**
Testing a God Object requires mocking everything:

\`\`\`go
// Testing the God Object requires:
// - Mock database
// - Mock SMTP server
// - Mock logger
// - Mock file system
// Just to test password validation!

func TestUserManager(t *testing.T) {
    mockDB := setupMockDB()           // complex
    mockSMTP := setupMockSMTP()       // complex
    mockLogger := setupMockLogger()   // complex

    manager := &UserManager{
        db: mockDB,
        smtpServer: mockSMTP.URL,
    }
    // Just trying to test password validation...
}
\`\`\`

**3. Tight Coupling**
Everything is coupled together - can't reuse anything:

\`\`\`go
// BAD: Want to use email functionality elsewhere?
// Too bad! It's buried inside UserManager
manager := &UserManager{}
// Can't use SendWelcome without the entire UserManager!

// GOOD: With separation
emailSvc := NewEmailService()
emailSvc.SendWelcome("anyone@example.com")  // reusable!
\`\`\`

**4. Merge Conflicts in Teams**
When one class does everything, everyone edits it:

\`\`\`go
// Developer A: Adding password reset
func (m *UserManager) ResetPassword(...) { }

// Developer B: Adding email templates
func (m *UserManager) SendTemplateEmail(...) { }

// Developer C: Adding user search
func (m *UserManager) SearchUsers(...) { }

// All editing the same 2000-line file = constant merge conflicts!
\`\`\`

**Real-World Example - Before Refactoring:**

\`\`\`go
// 1500 lines of nightmare
type OrderProcessor struct {
    db *sql.DB
    paymentGateway *PaymentAPI
    inventoryDB *sql.DB
    emailServer string
    taxRules map[string]float64
    shippingRates map[string]float64
    discountCodes map[string]float64
}

func (p *OrderProcessor) ProcessOrder(order Order) error {
    // Validates order (100 lines)
    // Checks inventory (150 lines)
    // Calculates tax (80 lines)
    // Calculates shipping (90 lines)
    // Applies discounts (120 lines)
    // Processes payment (200 lines)
    // Updates inventory (100 lines)
    // Sends confirmation email (50 lines)
    // Logs to database (40 lines)
    // Updates analytics (60 lines)
    // ... 1000+ more lines
}
\`\`\`

**After Refactoring - Clean Architecture:**

\`\`\`go
// Each service has one clear responsibility

type OrderValidator struct { /* validation only */ }
type InventoryService struct { /* inventory only */ }
type TaxCalculator struct { /* tax only */ }
type ShippingCalculator struct { /* shipping only */ }
type DiscountService struct { /* discounts only */ }
type PaymentProcessor struct { /* payment only */ }
type EmailNotifier struct { /* emails only */ }
type OrderRepository struct { /* persistence only */ }

// Orchestrator coordinates (but doesn't implement) everything
type OrderService struct {
    validator   *OrderValidator
    inventory   *InventoryService
    tax         *TaxCalculator
    shipping    *ShippingCalculator
    discount    *DiscountService
    payment     *PaymentProcessor
    email       *EmailNotifier
    repository  *OrderRepository
}

func (s *OrderService) ProcessOrder(order Order) error {
    // Each line delegates to a focused service
    if err := s.validator.Validate(order); err != nil {
        return err
    }

    if err := s.inventory.Reserve(order.Items); err != nil {
        return err
    }

    tax := s.tax.Calculate(order)
    shipping := s.shipping.Calculate(order)
    discount := s.discount.Apply(order)

    if err := s.payment.Process(order, tax, shipping, discount); err != nil {
        s.inventory.Release(order.Items)  // rollback
        return err
    }

    s.repository.Save(order)
    s.email.SendConfirmation(order)

    return nil
}
\`\`\`

**Benefits of Refactoring:**

1. **Easy Testing**: Test each service independently
2. **Team Scalability**: Different developers work on different services
3. **Reusability**: Use EmailNotifier anywhere in the app
4. **Maintainability**: Bug in tax calculation? Only look at TaxCalculator
5. **Flexibility**: Swap PaymentProcessor implementation without touching other code

**Warning Signs of a God Object:**

- Class with 500+ lines of code
- Class name ends with "Manager", "Handler", "Controller", "Service" but does multiple things
- Hard to explain what the class does in one sentence
- Class imports 20+ packages
- Every new feature requires editing this class
- Tests require extensive mocking`,
	order: 0,
	testCode: `package antipatterns

import (
	"strings"
	"testing"
)

// Test1: UserRepository Save and FindByID
func Test1(t *testing.T) {
	repo := NewUserRepository()
	user := User{ID: 1, Name: "John", Email: "john@test.com"}
	repo.Save(user)
	found := repo.FindByID(1)
	if found.Name != "John" {
		t.Error("Should find saved user")
	}
}

// Test2: UserRepository returns zero value for non-existent
func Test2(t *testing.T) {
	repo := NewUserRepository()
	found := repo.FindByID(999)
	if found.ID != 0 {
		t.Error("Should return zero value for non-existent ID")
	}
}

// Test3: EmailService SendWelcome returns correct message
func Test3(t *testing.T) {
	svc := NewEmailService()
	result := svc.SendWelcome("test@example.com")
	if !strings.Contains(result, "test@example.com") {
		t.Error("Should contain email address")
	}
	if !strings.Contains(result, "Welcome") {
		t.Error("Should contain Welcome")
	}
}

// Test4: PasswordValidator accepts valid password
func Test4(t *testing.T) {
	v := NewPasswordValidator()
	if !v.IsValid("password123") {
		t.Error("Should accept password with 8+ chars")
	}
}

// Test5: PasswordValidator rejects short password
func Test5(t *testing.T) {
	v := NewPasswordValidator()
	if v.IsValid("short") {
		t.Error("Should reject password with < 8 chars")
	}
}

// Test6: PasswordValidator accepts exactly 8 chars
func Test6(t *testing.T) {
	v := NewPasswordValidator()
	if !v.IsValid("12345678") {
		t.Error("Should accept exactly 8 chars")
	}
}

// Test7: PasswordValidator rejects 7 chars
func Test7(t *testing.T) {
	v := NewPasswordValidator()
	if v.IsValid("1234567") {
		t.Error("Should reject 7 chars")
	}
}

// Test8: User struct fields
func Test8(t *testing.T) {
	user := User{ID: 1, Name: "Test", Email: "test@test.com", Password: "pass"}
	if user.ID != 1 || user.Name != "Test" || user.Email != "test@test.com" {
		t.Error("User fields not set correctly")
	}
}

// Test9: Multiple users can be saved
func Test9(t *testing.T) {
	repo := NewUserRepository()
	repo.Save(User{ID: 1, Name: "A"})
	repo.Save(User{ID: 2, Name: "B"})
	if repo.FindByID(1).Name != "A" || repo.FindByID(2).Name != "B" {
		t.Error("Both users should be found")
	}
}

// Test10: EmailService returns non-empty string
func Test10(t *testing.T) {
	svc := NewEmailService()
	result := svc.SendWelcome("any@email.com")
	if result == "" {
		t.Error("SendWelcome should return non-empty string")
	}
}
`,
	translations: {
		ru: {
			title: 'Антипаттерн God Object - Базовый',
			description: `Рефакторьте God Object, который делает слишком много, разделив его на сфокусированные компоненты с единственной ответственностью.

**Проблема:**

God Object (также называемый Blob или Swiss Army Knife) — это класс, который знает слишком много или делает слишком много. Он нарушает принцип единственной ответственности, обрабатывая множество несвязанных задач.

**Вы выполните рефакторинг:**

Монолитного \`UserManager\`, который обрабатывает:
1. Хранение данных пользователя
2. Отправку email
3. Валидацию паролей
4. Операции с базой данных

**Разделите на:**
1. **User struct** - Только данные
2. **UserRepository** - Операции с БД
3. **EmailService** - Функциональность email
4. **PasswordValidator** - Логика валидации паролей

**Пример использования:**

\`\`\`go
// После рефакторинга
repo := NewUserRepository()
emailSvc := NewEmailService()
validator := NewPasswordValidator()

user := User{ID: 1, Name: "Alice", Email: "alice@example.com"}
if validator.IsValid("Pass123!") {
    repo.Save(user)
    emailSvc.SendWelcome(user.Email)
}
\`\`\`

**Ваша задача:**

Завершите рефакторинг с правильным разделением обязанностей.`,
			hint1: `Для NewUserRepository, NewEmailService и NewPasswordValidator создайте и верните новый экземпляр с инициализированными полями. UserRepository требует make(map[int]User).`,
			hint2: `Save сохраняет пользователя в map, FindByID возвращает из map (нулевое значение если отсутствует). SendWelcome возвращает форматированную строку. IsValid сравнивает длину пароля с minLength.`,
			whyItMatters: `God Objects — один из самых распространённых и разрушительных антипаттернов в разработке ПО.

**Почему God Objects проблематичны:**

**1. Нарушает принцип единственной ответственности**
God Object делает всё, что делает его невозможным для поддержки:

\`\`\`go
// ПЛОХО: God Object - UserManager делает ВСЁ
type UserManager struct {
    users map[int]User
    db *sql.DB
    smtpServer string
}

func (m *UserManager) CreateUser(name, email, password string) error {
    // Валидирует пароль
    if len(password) < 8 { return errors.New("weak password") }

    // Создаёт пользователя
    user := User{Name: name, Email: email}

    // Сохраняет в БД
    m.db.Exec("INSERT INTO users...", user)

    // Отправляет email
    smtp.SendEmail(m.smtpServer, email, "Welcome!")

    // Логирует активность
    log.Printf("User created: %s", name)

    return nil
}
// Эта функция делает ПЯТЬ разных вещей! Любое изменение влияет на всё.
\`\`\`

**2. Сложно тестировать**
Тестирование God Object требует мокирования всего:

\`\`\`go
// Тестирование God Object требует:
// - Mock базы данных
// - Mock SMTP сервера
// - Mock логгера
// - Mock файловой системы
// Просто чтобы протестировать валидацию пароля!

func TestUserManager(t *testing.T) {
    mockDB := setupMockDB()           // сложно
    mockSMTP := setupMockSMTP()       // сложно
    mockLogger := setupMockLogger()   // сложно

    manager := &UserManager{
        db: mockDB,
        smtpServer: mockSMTP.URL,
    }
    // Просто пытаемся протестировать валидацию пароля...
}
\`\`\`

**3. Жёсткая связанность**
Всё связано вместе - ничего нельзя переиспользовать:

\`\`\`go
// ПЛОХО: Хотите использовать email функциональность в другом месте?
// Не повезло! Она зарыта внутри UserManager
manager := &UserManager{}
// Нельзя использовать SendWelcome без всего UserManager!

// ХОРОШО: С разделением
emailSvc := NewEmailService()
emailSvc.SendWelcome("anyone@example.com")  // переиспользуемо!
\`\`\`

**4. Конфликты слияния в командах**
Когда один класс делает всё, все его редактируют:

\`\`\`go
// Разработчик A: Добавляет сброс пароля
func (m *UserManager) ResetPassword(...) { }

// Разработчик B: Добавляет email шаблоны
func (m *UserManager) SendTemplateEmail(...) { }

// Разработчик C: Добавляет поиск пользователей
func (m *UserManager) SearchUsers(...) { }

// Все редактируют один файл на 2000 строк = постоянные конфликты слияния!
\`\`\`

**Преимущества рефакторинга:**

1. **Простое тестирование**: Тестируйте каждый сервис независимо
2. **Масштабируемость команды**: Разные разработчики работают над разными сервисами
3. **Переиспользуемость**: Используйте EmailNotifier где угодно в приложении
4. **Поддерживаемость**: Баг в расчёте налогов? Смотрите только TaxCalculator
5. **Гибкость**: Замените реализацию PaymentProcessor без изменения другого кода`,
			solutionCode: `package antipatterns

// User представляет сущность пользователя - единственная ответственность: модель данных
type User struct {
	ID       int
	Name     string
	Email    string
	Password string
}

// UserRepository обрабатывает персистентность данных пользователя - единственная ответственность: хранение
type UserRepository struct {
	users map[int]User	// хранилище в памяти
}

func NewUserRepository() *UserRepository {
	return &UserRepository{
		users: make(map[int]User),	// инициализируем пустое хранилище
	}
}

func (r *UserRepository) Save(user User) {
	r.users[user.ID] = user	// сохраняем пользователя по ID
}

func (r *UserRepository) FindByID(id int) User {
	return r.users[id]	// возвращает нулевое значение если не найдено
}

// EmailService обрабатывает email операции - единственная ответственность: уведомления
type EmailService struct{}

func NewEmailService() *EmailService {
	return &EmailService{}
}

func (s *EmailService) SendWelcome(email string) string {
	return "Welcome email sent to: " + email	// симуляция отправки email
}

// PasswordValidator обрабатывает валидацию паролей - единственная ответственность: валидация
type PasswordValidator struct {
	minLength int	// конфигурация правила валидации
}

func NewPasswordValidator() *PasswordValidator {
	return &PasswordValidator{
		minLength: 8,	// стандартная минимальная длина пароля
	}
}

func (v *PasswordValidator) IsValid(password string) bool {
	return len(password) >= v.minLength	// простая валидация длины
}`
		},
		uz: {
			title: 'God Object Anti-pattern - Asosiy',
			description: `Juda ko'p narsalarni qiladigan God Object ni bir mas'uliyatli komponentlarga bo'lib refaktoring qiling.

**Muammo:**

God Object (Blob yoki Swiss Army Knife deb ham ataladi) — bu juda ko'p bilgan yoki juda ko'p ish qiladigan klass. U bir nechta bog'liq bo'lmagan vazifalarni bajarib, Yagona Mas'uliyat Printsipini buzadi.

**Siz refaktoring qilasiz:**

Quyidagilarni boshqaradigan monolitik \`UserManager\`:
1. Foydalanuvchi ma'lumotlarini saqlash
2. Email yuborish
3. Parol validatsiyasi
4. Ma'lumotlar bazasi operatsiyalari

**Bo'ling:**
1. **User struct** - Faqat ma'lumotlar
2. **UserRepository** - Ma'lumotlar bazasi operatsiyalari
3. **EmailService** - Email funksionalligi
4. **PasswordValidator** - Parol validatsiya logikasi

**Foydalanish misoli:**

\`\`\`go
// Refaktoringdan keyin
repo := NewUserRepository()
emailSvc := NewEmailService()
validator := NewPasswordValidator()

user := User{ID: 1, Name: "Alice", Email: "alice@example.com"}
if validator.IsValid("Pass123!") {
    repo.Save(user)
    emailSvc.SendWelcome(user.Email)
}
\`\`\`

**Sizning vazifangiz:**

Mas'uliyatlarni to'g'ri ajratgan holda refaktoringni yakunlang.`,
			hint1: `NewUserRepository, NewEmailService va NewPasswordValidator uchun initsializatsiya qilingan maydonlar bilan yangi nusxa yarating va qaytaring. UserRepository uchun make(map[int]User) kerak.`,
			hint2: `Save foydalanuvchini map ga saqlaydi, FindByID map dan qaytaradi (topilmasa nol qiymat). SendWelcome formatlangan string qaytaradi. IsValid parol uzunligini minLength bilan solishtiradi.`,
			whyItMatters: `God Objects dasturiy ta'minot ishlab chiqishda eng keng tarqalgan va zararli anti-patternlardan biridir.

**God Objects nima uchun muammoli:**

**1. Yagona Mas'uliyat Printsipini buzadi**
God Object hamma narsani qiladi, bu uni qo'llab-quvvatlashni imkonsiz qiladi:

\`\`\`go
// YOMON: God Object - UserManager HAMMA NARSANI qiladi
type UserManager struct {
    users map[int]User
    db *sql.DB
    smtpServer string
}

func (m *UserManager) CreateUser(name, email, password string) error {
    // Parolni validatsiya qiladi
    if len(password) < 8 { return errors.New("weak password") }

    // Foydalanuvchi yaratadi
    user := User{Name: name, Email: email}

    // Ma'lumotlar bazasiga saqlaydi
    m.db.Exec("INSERT INTO users...", user)

    // Email yuboradi
    smtp.SendEmail(m.smtpServer, email, "Welcome!")

    // Faoliyatni log qiladi
    log.Printf("User created: %s", name)

    return nil
}
// Bu funksiya BESH xil ishni qiladi! Har qanday o'zgarish hamma narsaga ta'sir qiladi.
\`\`\`

**2. Test qilish qiyin**
God Object ni test qilish hamma narsani mock qilishni talab qiladi:

\`\`\`go
// God Object ni test qilish talab qiladi:
// - Mock ma'lumotlar bazasi
// - Mock SMTP server
// - Mock logger
// - Mock fayl tizimi
// Faqat parol validatsiyasini test qilish uchun!

func TestUserManager(t *testing.T) {
    mockDB := setupMockDB()           // murakkab
    mockSMTP := setupMockSMTP()       // murakkab
    mockLogger := setupMockLogger()   // murakkab

    manager := &UserManager{
        db: mockDB,
        smtpServer: mockSMTP.URL,
    }
    // Faqat parol validatsiyasini test qilmoqchimiz...
}
\`\`\`

**3. Qattiq bog'lanish**
Hamma narsa birgalikda bog'langan - hech narsani qayta ishlatib bo'lmaydi:

\`\`\`go
// YOMON: Email funksiyasini boshqa joyda ishlatmoqchimisiz?
// Afsuski! U UserManager ichida ko'milgan
manager := &UserManager{}
// SendWelcome ni butun UserManager siz ishlatib bo'lmaydi!

// YAXSHI: Ajratish bilan
emailSvc := NewEmailService()
emailSvc.SendWelcome("anyone@example.com")  // qayta foydalanish mumkin!
\`\`\`

**4. Jamoalarda qo'shilish konfliktlari**
Bitta klass hamma narsani qilganda, hamma uni tahrirlaydi:

\`\`\`go
// Dasturchi A: Parolni qayta o'rnatishni qo'shmoqda
func (m *UserManager) ResetPassword(...) { }

// Dasturchi B: Email shablonlarini qo'shmoqda
func (m *UserManager) SendTemplateEmail(...) { }

// Dasturchi C: Foydalanuvchilarni qidirishni qo'shmoqda
func (m *UserManager) SearchUsers(...) { }

// Hammasi 2000 qatorli bir faylni tahrirlaydi = doimiy qo'shilish konfliktlari!
\`\`\`

**Refaktoringning afzalliklari:**

1. **Oson test qilish**: Har bir xizmatni mustaqil test qiling
2. **Jamoa miqyosliligi**: Turli dasturchilar turli xizmatlarda ishlaydi
3. **Qayta foydalanish**: EmailNotifier ni ilovaning istalgan joyida ishlating
4. **Qo'llab-quvvatlash**: Soliq hisoblashida xato? Faqat TaxCalculator ga qarang
5. **Moslashuvchanlik**: PaymentProcessor implementatsiyasini boshqa kodni o'zgartirmasdan almashtiring`,
			solutionCode: `package antipatterns

// User foydalanuvchi entitetini ifodalaydi - yagona mas'uliyat: ma'lumot modeli
type User struct {
	ID       int
	Name     string
	Email    string
	Password string
}

// UserRepository foydalanuvchi ma'lumotlarini saqlashni boshqaradi - yagona mas'uliyat: saqlash
type UserRepository struct {
	users map[int]User	// xotirada saqlash
}

func NewUserRepository() *UserRepository {
	return &UserRepository{
		users: make(map[int]User),	// bo'sh saqlashni initsializatsiya qilamiz
	}
}

func (r *UserRepository) Save(user User) {
	r.users[user.ID] = user	// foydalanuvchini ID bo'yicha saqlaymiz
}

func (r *UserRepository) FindByID(id int) User {
	return r.users[id]	// topilmasa nol qiymatni qaytaradi
}

// EmailService email operatsiyalarini boshqaradi - yagona mas'uliyat: bildirishnomalar
type EmailService struct{}

func NewEmailService() *EmailService {
	return &EmailService{}
}

func (s *EmailService) SendWelcome(email string) string {
	return "Welcome email sent to: " + email	// email yuborishni simulyatsiya qilish
}

// PasswordValidator parol validatsiyasini boshqaradi - yagona mas'uliyat: validatsiya
type PasswordValidator struct {
	minLength int	// validatsiya qoidasi konfiguratsiyasi
}

func NewPasswordValidator() *PasswordValidator {
	return &PasswordValidator{
		minLength: 8,	// standart minimal parol uzunligi
	}
}

func (v *PasswordValidator) IsValid(password string) bool {
	return len(password) >= v.minLength	// oddiy uzunlik validatsiyasi
}`
		}
	}
};

export default task;
