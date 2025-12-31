import { Task } from '../../../../../../../types';

export const task: Task = {
	slug: 'se-solid-srp',
	title: 'Single Responsibility Principle',
	difficulty: 'easy',
	tags: ['go', 'solid', 'srp', 'clean-code'],
	estimatedTime: '30m',
	isPremium: false,
	youtubeUrl: '',
	description: `Implement the Single Responsibility Principle (SRP) - a class should have only one reason to change.

**You will refactor:**

A User struct that violates SRP by handling multiple responsibilities:
- User data management
- Password validation
- Email sending
- Database persistence

**Your task:**

Refactor the code to follow SRP by creating separate types for each responsibility:

1. **User struct** - Only holds user data (name, email, password)
2. **PasswordValidator** - Validates password strength
3. **EmailService** - Sends emails
4. **UserRepository** - Handles database operations

**Key Concepts:**
- **Single Responsibility**: Each type should have one reason to change
- **Separation of Concerns**: Different functionalities should be in different types
- **Maintainability**: Changes to one feature don't affect others

**Example Usage:**

\`\`\`go
// Create user with valid data
user := NewUser("John Doe", "john@example.com", "SecurePass123!")

// Validate password using dedicated validator
validator := &PasswordValidator{}
if err := validator.Validate(user.Password); err != nil {
    fmt.Println("Invalid password:", err)
}

// Send email using dedicated service
emailService := &EmailService{}
emailService.SendWelcomeEmail(user.Email, user.Name)

// Save user using dedicated repository
repo := &UserRepository{}
repo.Save(user)
\`\`\`

**Why SRP matters:**
- Changes to password rules don't affect email sending
- Database changes don't affect validation
- Each component can be tested independently
- Code is easier to understand and maintain

**Constraints:**
- User struct should only hold data and have a constructor
- PasswordValidator should only validate passwords
- EmailService should only send emails
- UserRepository should only handle database operations`,
	initialCode: `package principles

import (
	"database/sql"
	"fmt"
	"net/smtp"
	"strings"
)

type User struct {
	Name     string
	Email    string
	Password string
}

func (u *User) ValidatePassword() error {
	if len(u.Password) < 8 {
		return fmt.Errorf("password must be at least 8 characters")
	}
		return fmt.Errorf("password must contain at least one digit")
	}
	}
	return nil
}

func (u *User) SendWelcomeEmail() error {
}

func (u *User) SaveToDatabase(db *sql.DB) error {
}

type PasswordValidator struct{}

func (pv *PasswordValidator) Validate(password string) error {
}

type EmailService struct{}

func (es *EmailService) SendWelcomeEmail(email, name string) error {
}

type UserRepository struct {
	db *sql.DB
}

func (ur *UserRepository) Save(user *User) error {
}

func NewUser(name, email, password string) *User {
}`,
	solutionCode: `package principles

import (
	"database/sql"
	"fmt"
	"net/smtp"
	"strings"
)

// User now has single responsibility: hold user data
// No validation, no email, no database logic
type User struct {
	Name     string	// user's full name
	Email    string	// user's email address
	Password string	// user's password (should be hashed in production)
}

// NewUser constructs a user with basic data
// Validation happens separately via PasswordValidator
func NewUser(name, email, password string) *User {
	return &User{	// simple constructor, just creates the struct
		Name:     name,
		Email:    email,
		Password: password,
	}
}

// PasswordValidator has single responsibility: validate passwords
// Changes to password rules only affect this type
type PasswordValidator struct{}

// Validate checks password strength requirements
// Returns error if password doesn't meet security criteria
func (pv *PasswordValidator) Validate(password string) error {
	if len(password) < 8 {	// minimum length requirement
		return fmt.Errorf("password must be at least 8 characters")
	}
	if !strings.ContainsAny(password, "0123456789") {	// must have digits
		return fmt.Errorf("password must contain at least one digit")
	}
	if !strings.ContainsAny(password, "ABCDEFGHIJKLMNOPQRSTUVWXYZ") {	// must have uppercase
		return fmt.Errorf("password must contain at least one uppercase letter")
	}
	return nil	// password meets all requirements
}

// EmailService has single responsibility: send emails
// Changes to email provider only affect this type
type EmailService struct{}

// SendWelcomeEmail sends welcome email to new user
// In production, this would use a proper email service
func (es *EmailService) SendWelcomeEmail(email, name string) error {
	// SMTP configuration (in production, use env vars)
	auth := smtp.PlainAuth("", "from@example.com", "password", "smtp.example.com")

	// Compose email message
	msg := []byte("Subject: Welcome!\\n\\nWelcome " + name)

	// Send via SMTP (isolated from User logic)
	return smtp.SendMail("smtp.example.com:587", auth, "from@example.com", []string{email}, msg)
}

// UserRepository has single responsibility: persist users
// Changes to database only affect this type
type UserRepository struct {
	db *sql.DB	// database connection injected via constructor
}

// Save persists user to database
// Handles SQL operations, completely separate from User struct
func (ur *UserRepository) Save(user *User) error {
	// Execute INSERT query
	_, err := ur.db.Exec(
		"INSERT INTO users (name, email, password) VALUES (?, ?, ?)",
		user.Name, user.Email, user.Password,
	)
	return err	// return any database errors
}`,
	hint1: `Start by making User a simple data holder - just name, email, password fields and a NewUser constructor that returns *User. Remove all methods from User.`,
	hint2: `For PasswordValidator.Validate, copy the validation logic from the old ValidatePassword method. For EmailService.SendWelcomeEmail and UserRepository.Save, copy their respective logic but use the email/name or user passed as parameters instead of u.Email/u.Name.`,
	testCode: `package principles

import (
	"strings"
	"testing"
)

// Test1: NewUser creates user with correct fields
func Test1(t *testing.T) {
	user := NewUser("John", "john@example.com", "Pass123!")
	if user == nil {
		t.Fatal("NewUser returned nil")
	}
	if user.Name != "John" {
		t.Errorf("Name = %q, want John", user.Name)
	}
}

// Test2: PasswordValidator rejects short passwords
func Test2(t *testing.T) {
	pv := &PasswordValidator{}
	err := pv.Validate("Short1")
	if err == nil {
		t.Error("expected error for password < 8 chars")
	}
}

// Test3: PasswordValidator requires digits
func Test3(t *testing.T) {
	pv := &PasswordValidator{}
	err := pv.Validate("NoDigitsHere")
	if err == nil {
		t.Error("expected error for password without digits")
	}
}

// Test4: PasswordValidator requires uppercase
func Test4(t *testing.T) {
	pv := &PasswordValidator{}
	err := pv.Validate("nouppercase123")
	if err == nil {
		t.Error("expected error for password without uppercase")
	}
}

// Test5: PasswordValidator accepts valid password
func Test5(t *testing.T) {
	pv := &PasswordValidator{}
	err := pv.Validate("ValidPass123")
	if err != nil {
		t.Errorf("unexpected error for valid password: %v", err)
	}
}

// Test6: EmailService.SendWelcomeEmail returns nil
func Test6(t *testing.T) {
	es := &EmailService{}
	// Note: In test environment, this simulates success
	_ = es
}

// Test7: User struct holds all fields
func Test7(t *testing.T) {
	user := NewUser("Jane", "jane@test.com", "SecurePass1")
	if user.Email != "jane@test.com" {
		t.Errorf("Email = %q, want jane@test.com", user.Email)
	}
	if user.Password != "SecurePass1" {
		t.Errorf("Password not stored correctly")
	}
}

// Test8: PasswordValidator error messages are descriptive
func Test8(t *testing.T) {
	pv := &PasswordValidator{}
	err := pv.Validate("short")
	if err == nil || !strings.Contains(err.Error(), "8") {
		t.Error("error should mention 8 characters requirement")
	}
}

// Test9: Multiple users can be created independently
func Test9(t *testing.T) {
	user1 := NewUser("User1", "user1@test.com", "Pass1234")
	user2 := NewUser("User2", "user2@test.com", "Pass5678")
	if user1.Name == user2.Name {
		t.Error("users should have different names")
	}
}

// Test10: All components exist and are separate
func Test10(t *testing.T) {
	user := NewUser("Test", "test@test.com", "TestPass1")
	validator := &PasswordValidator{}
	email := &EmailService{}
	repo := &UserRepository{}

	_ = user
	_ = validator
	_ = email
	_ = repo
	// All components should compile and be separate types
}
`,
	whyItMatters: `The Single Responsibility Principle is the foundation of maintainable software.

**Why SRP Matters:**

**1. Real Cost of Violating SRP**

Imagine you need to change password rules (add special character requirement):

\`\`\`go
// WITHOUT SRP - User does everything
type User struct {
	Name, Email, Password string
}

func (u *User) ValidatePassword() error {
	// Change password rules HERE
	// But User also has SaveToDatabase, SendEmail...
	// Risk: might accidentally break unrelated features
	// Testing: must test entire User type for one change
}

// WITH SRP - Isolated change
type PasswordValidator struct{}

func (pv *PasswordValidator) Validate(password string) error {
	// Change password rules HERE
	// Only affects password validation
	// Testing: only test PasswordValidator
	// Other components unaffected
}
\`\`\`

**2. Independence and Testability**

\`\`\`go
// WITHOUT SRP - Can't test validation without database/email
func TestUser(t *testing.T) {
	user := &User{Password: "weak"}

	// To test password validation, we need:
	// - Mock database (db.Exec)
	// - Mock SMTP server (smtp.SendMail)
	// - Test becomes complex and slow
}

// WITH SRP - Test each component independently
func TestPasswordValidator(t *testing.T) {
	validator := &PasswordValidator{}

	// Just test validation - no database, no email needed
	err := validator.Validate("weak")
	if err == nil {
		t.Error("should reject weak password")
	}
	// Fast, simple, focused test
}
\`\`\`

**3. Real-World Scenario: E-commerce Order**

\`\`\`go
// VIOLATES SRP - Order does too much
type Order struct {
	Items []Item
	Total float64
}

// Multiple responsibilities mixed together
func (o *Order) Process(db *sql.DB, paymentGateway PaymentAPI) error {
	// 1. Calculate total (business logic)
	for _, item := range o.Items {
		o.Total += item.Price
	}

	// 2. Apply discount (business logic)
	if o.Total > 100 {
		o.Total *= 0.9
	}

	// 3. Process payment (external service)
	if err := paymentGateway.Charge(o.Total); err != nil {
		return err
	}

	// 4. Save to database (persistence)
	if err := db.Exec("INSERT INTO orders..."); err != nil {
		return err
	}

	// 5. Send confirmation email (notification)
	smtp.SendMail(...)

	return nil
}
// Problem: Changing discount rules requires touching payment/database/email code!

// FOLLOWS SRP - Each responsibility separated
type Order struct {
	Items []Item
	Total float64
}

type OrderCalculator struct{}
func (oc *OrderCalculator) CalculateTotal(items []Item) float64 {
	total := 0.0
	for _, item := range items {
		total += item.Price
	}
	return total	// Only calculates, nothing else
}

type DiscountService struct{}
func (ds *DiscountService) ApplyDiscount(total float64) float64 {
	if total > 100 {
		return total * 0.9	// Only discounts, nothing else
	}
	return total
}

type PaymentService struct {
	gateway PaymentAPI
}
func (ps *PaymentService) Charge(amount float64) error {
	return ps.gateway.Charge(amount)	// Only payment, nothing else
}

type OrderRepository struct {
	db *sql.DB
}
func (or *OrderRepository) Save(order *Order) error {
	return or.db.Exec("INSERT INTO orders...")	// Only persistence
}

type NotificationService struct{}
func (ns *NotificationService) SendOrderConfirmation(order *Order) error {
	return smtp.SendMail(...)	// Only notifications
}

// Now each component can change independently!
\`\`\`

**4. Team Collaboration Benefits**

\`\`\`go
// WITH SRP - Different teams can work in parallel
// Team A: Changes password rules in PasswordValidator
// Team B: Switches from PostgreSQL to MongoDB in UserRepository
// Team C: Migrates from SMTP to SendGrid in EmailService
// No merge conflicts! Each team works on different files

// WITHOUT SRP - All teams modify same User type
// Constant merge conflicts and coordination overhead
\`\`\`

**5. Production Examples from Go Standard Library**

\`\`\`go
// http.Server follows SRP
type Server struct {
	Addr    string
	Handler Handler	// Server only serves, doesn't define routes
}

// Routing is separate (http.ServeMux)
mux := http.NewServeMux()	// Only handles routing
mux.HandleFunc("/", handler)

// Server only manages HTTP protocol
server := &http.Server{Addr: ":8080", Handler: mux}
server.ListenAndServe()

// Not like this (VIOLATES SRP):
// type Server struct {
//     routes map[string]func()	// routing mixed with serving
//     addr string
// }
\`\`\`

**Signs Your Code Violates SRP:**
- Type name has "And" or "Manager" (UserAndEmailManager)
- Long files (>300 lines for one type)
- Frequent changes break unrelated tests
- Hard to name the type (it does too much)
- Import statements include unrelated packages (sql + smtp in same file)

**How to Identify Responsibilities:**
Ask: "What reasons would this type need to change?"
- User type changes if: user fields change, validation changes, email template changes, database schema changes
- That's 4 reasons = 4 responsibilities = VIOLATES SRP
- Solution: Split into 4 types, each with 1 reason to change`,
	order: 0,
	translations: {
		ru: {
			title: 'Принцип единственной ответственности',
			description: `Реализуйте принцип единственной ответственности (SRP) — класс должен иметь только одну причину для изменения.

**Вы будете рефакторить:**

Структуру User, которая нарушает SRP, обрабатывая несколько обязанностей:
- Управление данными пользователя
- Валидация пароля
- Отправка email
- Сохранение в БД

**Ваша задача:**

Рефакторить код для соблюдения SRP, создав отдельные типы для каждой обязанности:

1. **User struct** - Только хранит данные пользователя (имя, email, пароль)
2. **PasswordValidator** - Валидирует надёжность пароля
3. **EmailService** - Отправляет email
4. **UserRepository** - Обрабатывает операции с БД

**Ключевые концепции:**
- **Единственная ответственность**: Каждый тип должен иметь одну причину для изменения
- **Разделение ответственностей**: Различные функциональности должны быть в разных типах
- **Поддерживаемость**: Изменения одной функции не влияют на другие

**Пример использования:**

\`\`\`go
// Создаём пользователя с валидными данными
user := NewUser("John Doe", "john@example.com", "SecurePass123!")

// Валидируем пароль используя специализированный валидатор
validator := &PasswordValidator{}
if err := validator.Validate(user.Password); err != nil {
    fmt.Println("Невалидный пароль:", err)
}

// Отправляем email используя специализированный сервис
emailService := &EmailService{}
emailService.SendWelcomeEmail(user.Email, user.Name)

// Сохраняем пользователя используя специализированный репозиторий
repo := &UserRepository{}
repo.Save(user)
\`\`\`

**Почему важен SRP:**
- Изменения правил паролей не влияют на отправку email
- Изменения БД не влияют на валидацию
- Каждый компонент можно тестировать независимо
- Код проще понимать и поддерживать

**Ограничения:**
- User struct должен только хранить данные и иметь конструктор
- PasswordValidator должен только валидировать пароли
- EmailService должен только отправлять email
- UserRepository должен только обрабатывать операции с БД`,
			hint1: `Начните с превращения User в простой holder данных - только поля name, email, password и конструктор NewUser, который возвращает *User. Удалите все методы из User.`,
			hint2: `Для PasswordValidator.Validate скопируйте логику валидации из старого метода ValidatePassword. Для EmailService.SendWelcomeEmail и UserRepository.Save скопируйте их соответствующую логику, но используйте email/name или user, переданные как параметры вместо u.Email/u.Name.`,
			whyItMatters: `Принцип единственной ответственности — это основа поддерживаемого программного обеспечения.

**Почему важен SRP:**

**1. Реальная стоимость нарушения SRP**

Представьте, что вам нужно изменить правила паролей (добавить требование специального символа):

\`\`\`go
// БЕЗ SRP - User делает всё
type User struct {
	Name, Email, Password string
}

func (u *User) ValidatePassword() error {
	// Изменяем правила паролей ЗДЕСЬ
	// Но User также имеет SaveToDatabase, SendEmail...
	// Риск: можете случайно сломать несвязанные функции
	// Тестирование: нужно тестировать весь тип User для одного изменения
}

// С SRP - Изолированное изменение
type PasswordValidator struct{}

func (pv *PasswordValidator) Validate(password string) error {
	// Изменяем правила паролей ЗДЕСЬ
	// Влияет только на валидацию пароля
	// Тестирование: тестируем только PasswordValidator
	// Другие компоненты не затронуты
}
\`\`\`

**2. Независимость и тестируемость**

\`\`\`go
// БЕЗ SRP - Нельзя тестировать валидацию без БД/email
func TestUser(t *testing.T) {
	user := &User{Password: "weak"}

	// Для теста валидации пароля нужны:
	// - Mock базы данных (db.Exec)
	// - Mock SMTP сервера (smtp.SendMail)
	// - Тест становится сложным и медленным
}

// С SRP - Тестируем каждый компонент независимо
func TestPasswordValidator(t *testing.T) {
	validator := &PasswordValidator{}

	// Только тестируем валидацию - без БД, без email
	err := validator.Validate("weak")
	if err == nil {
		t.Error("должен отклонить слабый пароль")
	}
	// Быстрый, простой, сфокусированный тест
}
\`\`\`

**Признаки нарушения SRP в коде:**
- Имя типа содержит "And" или "Manager" (UserAndEmailManager)
- Длинные файлы (>300 строк для одного типа)
- Частые изменения ломают несвязанные тесты
- Трудно назвать тип (он делает слишком много)
- Import'ы включают несвязанные пакеты (sql + smtp в одном файле)`,
			solutionCode: `package principles

import (
	"database/sql"
	"fmt"
	"net/smtp"
	"strings"
)

// User теперь имеет единственную ответственность: хранить данные пользователя
// Нет валидации, нет email, нет логики БД
type User struct {
	Name     string	// полное имя пользователя
	Email    string	// email адрес пользователя
	Password string	// пароль пользователя (должен быть хеширован в продакшене)
}

// NewUser конструирует пользователя с базовыми данными
// Валидация происходит отдельно через PasswordValidator
func NewUser(name, email, password string) *User {
	return &User{	// простой конструктор, просто создаёт структуру
		Name:     name,
		Email:    email,
		Password: password,
	}
}

// PasswordValidator имеет единственную ответственность: валидировать пароли
// Изменения правил паролей влияют только на этот тип
type PasswordValidator struct{}

// Validate проверяет требования к надёжности пароля
// Возвращает ошибку если пароль не соответствует критериям безопасности
func (pv *PasswordValidator) Validate(password string) error {
	if len(password) < 8 {	// требование минимальной длины
		return fmt.Errorf("password must be at least 8 characters")
	}
	if !strings.ContainsAny(password, "0123456789") {	// должны быть цифры
		return fmt.Errorf("password must contain at least one digit")
	}
	if !strings.ContainsAny(password, "ABCDEFGHIJKLMNOPQRSTUVWXYZ") {	// должны быть заглавные
		return fmt.Errorf("password must contain at least one uppercase letter")
	}
	return nil	// пароль соответствует всем требованиям
}

// EmailService имеет единственную ответственность: отправлять email
// Изменения email провайдера влияют только на этот тип
type EmailService struct{}

// SendWelcomeEmail отправляет приветственный email новому пользователю
// В продакшене использовался бы нормальный email сервис
func (es *EmailService) SendWelcomeEmail(email, name string) error {
	// SMTP конфигурация (в продакшене используйте env переменные)
	auth := smtp.PlainAuth("", "from@example.com", "password", "smtp.example.com")

	// Составляем email сообщение
	msg := []byte("Subject: Welcome!\\n\\nWelcome " + name)

	// Отправляем через SMTP (изолировано от логики User)
	return smtp.SendMail("smtp.example.com:587", auth, "from@example.com", []string{email}, msg)
}

// UserRepository имеет единственную ответственность: сохранять пользователей
// Изменения БД влияют только на этот тип
type UserRepository struct {
	db *sql.DB	// подключение к БД инжектится через конструктор
}

// Save сохраняет пользователя в БД
// Обрабатывает SQL операции, полностью отдельно от структуры User
func (ur *UserRepository) Save(user *User) error {
	// Выполняем INSERT запрос
	_, err := ur.db.Exec(
		"INSERT INTO users (name, email, password) VALUES (?, ?, ?)",
		user.Name, user.Email, user.Password,
	)
	return err	// возвращаем любые ошибки БД
}`
		},
		uz: {
			title: 'Yagona mas\'uliyat printsipi',
			description: `Yagona mas'uliyat prinsipini (SRP) amalga oshiring - klass faqat bitta o'zgarish sababiga ega bo'lishi kerak.

**Siz refaktoring qilasiz:**

Bir nechta mas'uliyatlarni bajarib SRP ni buzuvchi User strukturasini:
- Foydalanuvchi ma'lumotlarini boshqarish
- Parol tekshirish
- Email yuborish
- Ma'lumotlar bazasiga saqlash

**Sizning vazifangiz:**

Har bir mas'uliyat uchun alohida turlar yaratib, SRP ga rioya qilish uchun kodni refaktoring qiling:

1. **User struct** - Faqat foydalanuvchi ma'lumotlarini saqlaydi (ism, email, parol)
2. **PasswordValidator** - Parol kuchliligini tekshiradi
3. **EmailService** - Email yuboradi
4. **UserRepository** - Ma'lumotlar bazasi operatsiyalarini boshqaradi

**Asosiy tushunchalar:**
- **Yagona mas'uliyat**: Har bir tur faqat bitta o'zgarish sababiga ega bo'lishi kerak
- **Mas'uliyatlarni ajratish**: Turli funktsionalliklar turli turlarda bo'lishi kerak
- **Saqlanish**: Bitta funksiyaning o'zgarishi boshqalarga ta'sir qilmasligi kerak

**Foydalanish misoli:**

\`\`\`go
// To'g'ri ma'lumotlar bilan foydalanuvchi yaratamiz
user := NewUser("John Doe", "john@example.com", "SecurePass123!")

// Maxsus validatordan foydalanib parolni tekshiramiz
validator := &PasswordValidator{}
if err := validator.Validate(user.Password); err != nil {
    fmt.Println("Noto'g'ri parol:", err)
}

// Maxsus servisdan foydalanib email yuboramiz
emailService := &EmailService{}
emailService.SendWelcomeEmail(user.Email, user.Name)

// Maxsus repositorydan foydalanib foydalanuvchini saqlaymiz
repo := &UserRepository{}
repo.Save(user)
\`\`\`

**SRP nima uchun muhim:**
- Parol qoidalari o'zgarsa email yuborishga ta'sir qilmaydi
- Ma'lumotlar bazasi o'zgarsa tekshirishga ta'sir qilmaydi
- Har bir komponentni mustaqil test qilish mumkin
- Kodni tushunish va saqlash osonroq

**Cheklovlar:**
- User struct faqat ma'lumotlarni saqlashi va konstruktorga ega bo'lishi kerak
- PasswordValidator faqat parollarni tekshirishi kerak
- EmailService faqat email yuborishi kerak
- UserRepository faqat ma'lumotlar bazasi operatsiyalarini bajarishi kerak`,
			hint1: `User ni oddiy ma'lumot ushlagichga aylantirish bilan boshlang - faqat name, email, password maydonlari va *User qaytaruvchi NewUser konstruktor. User dan barcha metodlarni o'chiring.`,
			hint2: `PasswordValidator.Validate uchun eski ValidatePassword metodidan tekshirish mantiqini nusxalang. EmailService.SendWelcomeEmail va UserRepository.Save uchun ularning tegishli mantiqini nusxalang, lekin u.Email/u.Name o'rniga parametr sifatida berilgan email/name yoki user dan foydalaning.`,
			whyItMatters: `Yagona mas'uliyat printsipi saqlanishi mumkin bo'lgan dasturiy ta'minotning poydevoridir.

**SRP nima uchun muhim:**

**1. SRP buzishning haqiqiy narxi**

Tasavvur qiling, parol qoidalarini o'zgartirishingiz kerak (maxsus belgi talabini qo'shish):

\`\`\`go
// SRP siz - User hamma narsani qiladi
type User struct {
	Name, Email, Password string
}

func (u *User) ValidatePassword() error {
	// Parol qoidalarini BU YERDA o'zgartiring
	// Lekin User da SaveToDatabase, SendEmail ham bor...
	// Xavf: boshqa funktsiyalarni tasodifan buzishingiz mumkin
	// Test: bitta o'zgarish uchun butun User turini test qilish kerak
}

// SRP bilan - Izolyatsiya qilingan o'zgarish
type PasswordValidator struct{}

func (pv *PasswordValidator) Validate(password string) error {
	// Parol qoidalarini BU YERDA o'zgartiring
	// Faqat parol tekshirishga ta'sir qiladi
	// Test: faqat PasswordValidator ni test qilish
	// Boshqa komponentlar ta'sirlanmaydi
}
\`\`\`

**SRP buzilganining belgilari:**
- Tur nomi "And" yoki "Manager" so'zini o'z ichiga oladi (UserAndEmailManager)
- Uzun fayllar (bitta tur uchun >300 qator)
- Tez-tez o'zgarishlar bog'lanmagan testlarni buzadi
- Turni nomlash qiyin (u juda ko'p narsa qiladi)
- Import iboralar bog'lanmagan paketlarni o'z ichiga oladi (bitta faylda sql + smtp)`,
			solutionCode: `package principles

import (
	"database/sql"
	"fmt"
	"net/smtp"
	"strings"
)

// User endi yagona mas'uliyatga ega: foydalanuvchi ma'lumotlarini saqlash
// Tekshirish yo'q, email yo'q, ma'lumotlar bazasi mantiqi yo'q
type User struct {
	Name     string	// foydalanuvchining to'liq ismi
	Email    string	// foydalanuvchining email manzili
	Password string	// foydalanuvchining paroli (ishlab chiqarishda xeshlanishi kerak)
}

// NewUser asosiy ma'lumotlar bilan foydalanuvchi yaratadi
// Tekshirish PasswordValidator orqali alohida amalga oshiriladi
func NewUser(name, email, password string) *User {
	return &User{	// oddiy konstruktor, shunchaki strukturani yaratadi
		Name:     name,
		Email:    email,
		Password: password,
	}
}

// PasswordValidator yagona mas'uliyatga ega: parollarni tekshirish
// Parol qoidalari o'zgarishi faqat bu turga ta'sir qiladi
type PasswordValidator struct{}

// Validate parol kuchliligi talablarini tekshiradi
// Parol xavfsizlik mezonlariga javob bermasa xato qaytaradi
func (pv *PasswordValidator) Validate(password string) error {
	if len(password) < 8 {	// minimal uzunlik talabi
		return fmt.Errorf("password must be at least 8 characters")
	}
	if !strings.ContainsAny(password, "0123456789") {	// raqamlar bo'lishi kerak
		return fmt.Errorf("password must contain at least one digit")
	}
	if !strings.ContainsAny(password, "ABCDEFGHIJKLMNOPQRSTUVWXYZ") {	// katta harflar bo'lishi kerak
		return fmt.Errorf("password must contain at least one uppercase letter")
	}
	return nil	// parol barcha talablarga javob beradi
}

// EmailService yagona mas'uliyatga ega: emaillarni yuborish
// Email provayderining o'zgarishi faqat bu turga ta'sir qiladi
type EmailService struct{}

// SendWelcomeEmail yangi foydalanuvchiga xush kelibsiz emailini yuboradi
// Ishlab chiqarishda to'g'ri email servisi ishlatiladi
func (es *EmailService) SendWelcomeEmail(email, name string) error {
	// SMTP konfiguratsiyasi (ishlab chiqarishda env o'zgaruvchilaridan foydalaning)
	auth := smtp.PlainAuth("", "from@example.com", "password", "smtp.example.com")

	// Email xabarini tuzamiz
	msg := []byte("Subject: Welcome!\\n\\nWelcome " + name)

	// SMTP orqali yuboramiz (User mantiqidan ajratilgan)
	return smtp.SendMail("smtp.example.com:587", auth, "from@example.com", []string{email}, msg)
}

// UserRepository yagona mas'uliyatga ega: foydalanuvchilarni saqlash
// Ma'lumotlar bazasi o'zgarishi faqat bu turga ta'sir qiladi
type UserRepository struct {
	db *sql.DB	// ma'lumotlar bazasi ulanishi konstruktor orqali kiritiladi
}

// Save foydalanuvchini ma'lumotlar bazasiga saqlaydi
// SQL operatsiyalarini bajaradi, User strukturasidan butunlay alohida
func (ur *UserRepository) Save(user *User) error {
	// INSERT so'rovini bajaramiz
	_, err := ur.db.Exec(
		"INSERT INTO users (name, email, password) VALUES (?, ?, ?)",
		user.Name, user.Email, user.Password,
	)
	return err	// har qanday ma'lumotlar bazasi xatolarini qaytaramiz
}`
		}
	}
};

export default task;
