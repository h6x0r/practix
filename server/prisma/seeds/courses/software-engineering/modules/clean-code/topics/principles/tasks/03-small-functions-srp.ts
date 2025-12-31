import { Task } from '../../../../../../../types';

export const task: Task = {
	slug: 'se-clean-code-small-functions-srp',
	title: 'Small Functions - Single Responsibility',
	difficulty: 'medium',
	tags: ['go', 'clean-code', 'functions', 'srp'],
	estimatedTime: '30m',
	isPremium: false,
	youtubeUrl: '',
	description: `Refactor a large function into smaller functions, each with a single responsibility.

**You will refactor:**

1. **ProcessUserRegistration** - A large function doing too many things
2. Break it into smaller functions with single responsibilities:
   - ValidateUserInput
   - CreateUser
   - SendWelcomeEmail
   - LogRegistration

**Key Concepts:**
- **Single Responsibility Principle**: Each function should do one thing and do it well
- **Function Length**: Aim for functions under 20 lines
- **Descriptive Names**: Small functions with clear names are self-documenting
- **Testability**: Small functions are easier to test

**Example - Before:**

\`\`\`go
func ProcessUserRegistration(email, password string) error {
    // Validation
    if !strings.Contains(email, "@") {
        return errors.New("invalid email")
    }
    if len(password) < 8 {
        return errors.New("password too short")
    }

    // Create user
    user := &User{Email: email, Password: hashPassword(password)}
    if err := db.Save(user); err != nil {
        return err
    }

    // Send email
    sendEmail(user.Email, "Welcome!")

    // Log
    log.Printf("User registered: %s", email)
    return nil
}
\`\`\`

**Example - After:**

\`\`\`go
func ProcessUserRegistration(email, password string) error {
    if err := ValidateUserInput(email, password); err != nil {
        return err
    }

    user, err := CreateUser(email, password)
    if err != nil {
        return err
    }

    SendWelcomeEmail(user)
    LogRegistration(user)

    return nil
}

func ValidateUserInput(email, password string) error { ... }
func CreateUser(email, password string) (*User, error) { ... }
func SendWelcomeEmail(user *User) { ... }
func LogRegistration(user *User) { ... }
\`\`\`

**When to extract functions:**
- Function is doing more than one thing
- Function is hard to name without "And"
- Function is hard to test
- Function has multiple levels of abstraction

**Constraints:**
- Extract at least 4 helper functions
- Each helper should have one responsibility
- Main function should be a high-level orchestrator`,
	initialCode: `package principles

import (
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"strings"
)

type User struct {
	ID       int
	Email    string
	Password string
}

func ProcessUserRegistration(email, password string) error {
		return errors.New("invalid email format")
	}

	}

		}
		}
		}
	}
	}

	}

	}

	}

	return nil
}

func generateUserID() int {
}

func saveUserToDB(user *User) error {
	return nil
}

func sendEmailService(to, subject, body string) error {
	return nil
}`,
	solutionCode: `package principles

import (
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"strings"
)

type User struct {
	ID       int
	Email    string
	Password string
}

// ProcessUserRegistration orchestrates the user registration process
// This function is now a high-level coordinator with clear steps
// Each step is delegated to a specialized function
func ProcessUserRegistration(email, password string) error {
	// Step 1: Validate input
	if err := ValidateUserInput(email, password); err != nil {
		return err
	}

	// Step 2: Create and save user
	user, err := CreateUser(email, password)
	if err != nil {
		return err
	}

	// Step 3: Send welcome notification
	SendWelcomeEmail(user)

	// Step 4: Log the event
	LogRegistration(user)

	return nil
}

// ValidateUserInput checks if email and password meet requirements
// Single Responsibility: Only validates input, doesn't create or save anything
func ValidateUserInput(email, password string) error {
	// Validate email format
	if !strings.Contains(email, "@") || !strings.Contains(email, ".") {
		return errors.New("invalid email format")
	}

	// Validate password length
	if len(password) < 8 {
		return errors.New("password must be at least 8 characters")
	}

	// Validate password complexity
	hasUpper := false
	hasLower := false
	hasDigit := false
	for _, char := range password {
		if char >= 'A' && char <= 'Z' {
			hasUpper = true
		}
		if char >= 'a' && char <= 'z' {
			hasLower = true
		}
		if char >= '0' && char <= '9' {
			hasDigit = true
		}
	}
	if !hasUpper || !hasLower || !hasDigit {
		return errors.New("password must contain uppercase, lowercase, and digit")
	}

	return nil
}

// CreateUser creates a new user with hashed password and saves to database
// Single Responsibility: User creation and persistence
func CreateUser(email, password string) (*User, error) {
	// Hash password
	hash := sha256.Sum256([]byte(password))
	hashedPassword := hex.EncodeToString(hash[:])

	// Create user object
	user := &User{
		ID:       generateUserID(),
		Email:    email,
		Password: hashedPassword,
	}

	// Save to database
	if err := saveUserToDB(user); err != nil {
		return nil, fmt.Errorf("failed to save user: %w", err)
	}

	return user, nil
}

// SendWelcomeEmail sends a welcome email to newly registered user
// Single Responsibility: Email notification only
// Errors are logged but don't fail registration (non-critical operation)
func SendWelcomeEmail(user *User) {
	emailBody := fmt.Sprintf("Welcome to our platform, %s!", user.Email)
	if err := sendEmailService(user.Email, "Welcome!", emailBody); err != nil {
		fmt.Printf("Failed to send welcome email: %v\n", err)
	}
}

// LogRegistration records the registration event in logs
// Single Responsibility: Logging only
func LogRegistration(user *User) {
	fmt.Printf("[INFO] User registered successfully: %s (ID: %d)\n", user.Email, user.ID)
}

// Helper functions (simulated)
var userIDCounter = 1000

func generateUserID() int {
	userIDCounter++
	return userIDCounter
}

func saveUserToDB(user *User) error {
	// Simulate database save
	return nil
}

func sendEmailService(to, subject, body string) error {
	// Simulate email service
	return nil
}`,
	hint1: `Extract ValidateUserInput to handle all validation logic (email and password checks). Extract CreateUser to handle user creation and database save.`,
	hint2: `Extract SendWelcomeEmail for email sending logic (notice it doesn't return error, just logs). Extract LogRegistration for logging. The main function should just call these 4 functions in order.`,
	testCode: `package principles

import "testing"

// Test1: Valid email and password passes validation
func Test1(t *testing.T) {
	err := ValidateUserInput("test@example.com", "Password123")
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
}

// Test2: Invalid email without @ fails
func Test2(t *testing.T) {
	err := ValidateUserInput("invalid-email", "Password123")
	if err == nil {
		t.Error("expected error for invalid email")
	}
}

// Test3: Short password fails
func Test3(t *testing.T) {
	err := ValidateUserInput("test@example.com", "Pass1")
	if err == nil {
		t.Error("expected error for short password")
	}
}

// Test4: Password without uppercase fails
func Test4(t *testing.T) {
	err := ValidateUserInput("test@example.com", "password123")
	if err == nil {
		t.Error("expected error for password without uppercase")
	}
}

// Test5: Password without digit fails
func Test5(t *testing.T) {
	err := ValidateUserInput("test@example.com", "PasswordABC")
	if err == nil {
		t.Error("expected error for password without digit")
	}
}

// Test6: CreateUser returns user
func Test6(t *testing.T) {
	user, err := CreateUser("test@example.com", "Password123")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if user == nil {
		t.Error("CreateUser returned nil user")
	}
	if user.Email != "test@example.com" {
		t.Errorf("Email = %q, want test@example.com", user.Email)
	}
}

// Test7: User password is hashed
func Test7(t *testing.T) {
	user, _ := CreateUser("test@example.com", "Password123")
	if user.Password == "Password123" {
		t.Error("Password should be hashed, not plaintext")
	}
	if len(user.Password) != 64 { // SHA256 hex = 64 chars
		t.Error("Password hash has wrong length")
	}
}

// Test8: Full registration succeeds
func Test8(t *testing.T) {
	err := ProcessUserRegistration("test@example.com", "Password123")
	if err != nil {
		t.Errorf("ProcessUserRegistration error: %v", err)
	}
}

// Test9: Registration fails with invalid input
func Test9(t *testing.T) {
	err := ProcessUserRegistration("invalid", "weak")
	if err == nil {
		t.Error("expected error for invalid input")
	}
}

// Test10: User has unique ID
func Test10(t *testing.T) {
	user1, _ := CreateUser("user1@example.com", "Password1")
	user2, _ := CreateUser("user2@example.com", "Password2")
	if user1.ID == user2.ID {
		t.Error("Users should have different IDs")
	}
}
`,
	whyItMatters: `Small functions with single responsibilities are the cornerstone of maintainable code.

**Why Small Functions Matter:**

**1. Cognitive Load Reduction**

\`\`\`go
// BAD: 60-line function - must hold entire context in head
func ProcessUserRegistration(email, password string) error {
    // 10 lines of email validation
    // 15 lines of password validation
    // 10 lines of user creation
    // 10 lines of email sending
    // 5 lines of logging
    // 10 lines of error handling
    // Developer must understand ALL 60 lines to modify ANY part
}

// GOOD: 10-line orchestrator function
func ProcessUserRegistration(email, password string) error {
    if err := ValidateUserInput(email, password); err != nil {
        return err
    }
    user, err := CreateUser(email, password)
    if err != nil {
        return err
    }
    SendWelcomeEmail(user)
    LogRegistration(user)
    return nil
}
// Developer understands flow immediately, can drill into any step
\`\`\`

**2. Testability**

\`\`\`go
// BAD: Hard to test - must mock database, email service, logger all at once
func TestProcessUserRegistration(t *testing.T) {
    // Setup: Mock DB, mock email, mock logger
    // Test: Call one big function
    // Problem: Can't test validation separately from email sending
}

// GOOD: Easy to test each piece independently
func TestValidateUserInput(t *testing.T) {
    // Only test validation - no mocks needed
    err := ValidateUserInput("bad-email", "weak")
    if err == nil {
        t.Error("Expected validation error")
    }
}

func TestCreateUser(t *testing.T) {
    // Only mock database
    user, err := CreateUser("test@example.com", "SecurePass123")
    if err != nil {
        t.Errorf("Expected user creation to succeed")
    }
}
\`\`\`

**3. Reusability**

\`\`\`go
// BAD: Validation is buried in registration function
func ProcessUserRegistration(email, password string) error {
    // validation code here...
    // Can't reuse validation for login, password reset, etc.
}

// GOOD: Validation is extracted and reusable
func ValidateUserInput(email, password string) error { ... }

// Now can reuse in multiple places:
func ProcessLogin(email, password string) error {
    if err := ValidateUserInput(email, password); err != nil {
        return err
    }
    // login logic...
}

func ResetPassword(email, newPassword string) error {
    if err := ValidateUserInput(email, newPassword); err != nil {
        return err
    }
    // reset logic...
}
\`\`\`

**4. Real Bug from Production**

\`\`\`go
// Production code at e-commerce company:
func ProcessCheckout(cart *Cart) error {
    // 100+ lines of code:
    // - Validate cart items
    // - Calculate total
    // - Charge credit card
    // - Update inventory
    // - Send confirmation email
    // - Create shipping label

    // BUG: If email sending fails, inventory was already updated
    // and card was already charged, but error was returned
    // RESULT: Customer charged but no order record!

    // Cost: $250,000 in lost orders and customer service
}

// Fix: Break into transactional steps
func ProcessCheckout(cart *Cart) error {
    // Step 1: Validations (can fail safely)
    if err := ValidateCart(cart); err != nil {
        return err
    }

    // Step 2: Financial transaction (atomic)
    payment, err := ProcessPayment(cart)
    if err != nil {
        return err  // Nothing to rollback yet
    }

    // Step 3: Order creation (atomic with payment)
    order, err := CreateOrder(cart, payment)
    if err != nil {
        RefundPayment(payment)  // Rollback payment
        return err
    }

    // Step 4: Non-critical operations (don't fail order)
    SendConfirmationEmail(order)  // Failures just log
    CreateShippingLabel(order)     // Failures just log

    return nil
}
// Each step is testable, reversible, and has clear boundaries
\`\`\`

**5. Single Level of Abstraction**

\`\`\`go
// BAD: Mixing abstraction levels
func ProcessOrder(order *Order) error {
    // High-level abstraction
    if err := validateOrder(order); err != nil {
        return err
    }

    // Low-level implementation details - breaks abstraction!
    conn, err := net.Dial("tcp", "payment-gateway:443")
    if err != nil {
        return err
    }
    defer conn.Close()

    data := fmt.Sprintf("CHARGE %f TO %s", order.Total, order.CardNumber)
    conn.Write([]byte(data))
    // ... more low-level networking code

    // Back to high-level
    if err := sendConfirmation(order); err != nil {
        return err
    }
}

// GOOD: Consistent abstraction level
func ProcessOrder(order *Order) error {
    // All at same high level
    if err := ValidateOrder(order); err != nil {
        return err
    }
    if err := ChargePayment(order); err != nil {
        return err
    }
    if err := SendConfirmation(order); err != nil {
        return err
    }
    return nil
}
// Low-level details hidden in ChargePayment function
\`\`\`

**6. Function Length Guidelines**

\`\`\`go
// Rule of thumb from Clean Code:
// - Functions should rarely be 20 lines long
// - Functions should hardly ever be 100 lines long
// - If you can't see entire function on screen, it's too long

// Ideal function length by type:
// - Orchestrator functions: 5-10 lines
// - Business logic functions: 10-20 lines
// - Complex algorithms: 20-40 lines (but consider extracting helpers)
// - Data transformation: 10-30 lines

// If function is > 40 lines, ask:
// - Can I extract validation logic?
// - Can I extract data transformation?
// - Can I extract error handling?
// - Can I extract a loop body?
\`\`\`

**7. Benefits Summary**

Small functions with single responsibilities:
- **Reduce bugs**: Easier to understand, less chance of mistakes
- **Accelerate development**: Copy and modify small functions faster
- **Enable refactoring**: Change one function without affecting others
- **Improve collaboration**: Team members understand small functions instantly
- **Facilitate code review**: Reviewer can focus on one responsibility at a time

**8. When NOT to Extract**

\`\`\`go
// Don't extract if function is only called once and extraction doesn't improve clarity

// BAD: Over-extraction makes code harder to follow
func ProcessData(data []int) int {
    return calculateResult(data)
}

func calculateResult(data []int) int {
    return sumElements(data)
}

func sumElements(data []int) int {
    return computeSum(data)
}

func computeSum(data []int) int {
    sum := 0
    for _, v := range data {
        sum += v
    }
    return sum
}

// GOOD: Simple enough to keep inline
func ProcessData(data []int) int {
    sum := 0
    for _, v := range data {
        sum += v
    }
    return sum
}
\`\`\``,
	order: 2,
	translations: {
		ru: {
			title: 'Малые функции - Единственная ответственность',
			description: `Выполните рефакторинг большой функции на меньшие функции, каждая с единственной ответственностью.

**Вы выполните рефакторинг:**

1. **ProcessUserRegistration** - Большая функция делающая слишком много
2. Разбейте её на меньшие функции с единственными ответственностями:
   - ValidateUserInput
   - CreateUser
   - SendWelcomeEmail
   - LogRegistration

**Ключевые концепции:**
- **Принцип единственной ответственности**: Каждая функция должна делать одну вещь и делать её хорошо
- **Длина функции**: Стремитесь к функциям менее 20 строк
- **Описательные имена**: Малые функции с чёткими именами самодокументируются
- **Тестируемость**: Малые функции легче тестировать

**Когда извлекать функции:**
- Функция делает более одной вещи
- Функцию сложно назвать без "И"
- Функцию сложно тестировать
- Функция имеет множественные уровни абстракции

**Ограничения:**
- Извлеките минимум 4 вспомогательные функции
- Каждая вспомогательная должна иметь одну ответственность
- Главная функция должна быть высокоуровневым оркестратором`,
			hint1: `Извлеките ValidateUserInput для обработки всей логики валидации (проверки email и пароля). Извлеките CreateUser для обработки создания пользователя и сохранения в БД.`,
			hint2: `Извлеките SendWelcomeEmail для логики отправки email (заметьте она не возвращает error, только логирует). Извлеките LogRegistration для логирования. Главная функция должна только вызывать эти 4 функции по порядку.`,
			whyItMatters: `Малые функции с единственными ответственностями — краеугольный камень поддерживаемого кода.

**Почему малые функции важны:**

**1. Снижение когнитивной нагрузки**

Разработчик понимает поток мгновенно, может углубиться в любой шаг.

**2. Тестируемость**

Легко тестировать каждую часть независимо. Валидацию можно тестировать отдельно от отправки email.

**3. Переиспользуемость**

Валидация извлечена и переиспользуема в логине, сбросе пароля и т.д.

**4. Единый уровень абстракции**

Все на одном высоком уровне. Низкоуровневые детали скрыты во вспомогательных функциях.`,
			solutionCode: `package principles

import (
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"strings"
)

type User struct {
	ID       int
	Email    string
	Password string
}

// ProcessUserRegistration оркеструет процесс регистрации пользователя
// Эта функция теперь высокоуровневый координатор с чёткими шагами
// Каждый шаг делегируется специализированной функции
func ProcessUserRegistration(email, password string) error {
	// Шаг 1: Валидация ввода
	if err := ValidateUserInput(email, password); err != nil {
		return err
	}

	// Шаг 2: Создание и сохранение пользователя
	user, err := CreateUser(email, password)
	if err != nil {
		return err
	}

	// Шаг 3: Отправка приветственного уведомления
	SendWelcomeEmail(user)

	// Шаг 4: Логирование события
	LogRegistration(user)

	return nil
}

// ValidateUserInput проверяет соответствуют ли email и пароль требованиям
// Единственная ответственность: Только валидирует ввод, не создаёт и не сохраняет ничего
func ValidateUserInput(email, password string) error {
	// Валидация формата email
	if !strings.Contains(email, "@") || !strings.Contains(email, ".") {
		return errors.New("invalid email format")
	}

	// Валидация длины пароля
	if len(password) < 8 {
		return errors.New("password must be at least 8 characters")
	}

	// Валидация сложности пароля
	hasUpper := false
	hasLower := false
	hasDigit := false
	for _, char := range password {
		if char >= 'A' && char <= 'Z' {
			hasUpper = true
		}
		if char >= 'a' && char <= 'z' {
			hasLower = true
		}
		if char >= '0' && char <= '9' {
			hasDigit = true
		}
	}
	if !hasUpper || !hasLower || !hasDigit {
		return errors.New("password must contain uppercase, lowercase, and digit")
	}

	return nil
}

// CreateUser создаёт нового пользователя с хешированным паролем и сохраняет в БД
// Единственная ответственность: Создание пользователя и сохранение
func CreateUser(email, password string) (*User, error) {
	// Хеширование пароля
	hash := sha256.Sum256([]byte(password))
	hashedPassword := hex.EncodeToString(hash[:])

	// Создание объекта пользователя
	user := &User{
		ID:       generateUserID(),
		Email:    email,
		Password: hashedPassword,
	}

	// Сохранение в БД
	if err := saveUserToDB(user); err != nil {
		return nil, fmt.Errorf("failed to save user: %w", err)
	}

	return user, nil
}

// SendWelcomeEmail отправляет приветственный email новому пользователю
// Единственная ответственность: Только отправка email уведомлений
// Ошибки логируются но не прерывают регистрацию (некритическая операция)
func SendWelcomeEmail(user *User) {
	emailBody := fmt.Sprintf("Welcome to our platform, %s!", user.Email)
	if err := sendEmailService(user.Email, "Welcome!", emailBody); err != nil {
		fmt.Printf("Failed to send welcome email: %v\n", err)
	}
}

// LogRegistration записывает событие регистрации в логи
// Единственная ответственность: Только логирование
func LogRegistration(user *User) {
	fmt.Printf("[INFO] User registered successfully: %s (ID: %d)\n", user.Email, user.ID)
}

// Вспомогательные функции (симуляция)
var userIDCounter = 1000

func generateUserID() int {
	userIDCounter++
	return userIDCounter
}

func saveUserToDB(user *User) error {
	// Симуляция сохранения в БД
	return nil
}

func sendEmailService(to, subject, body string) error {
	// Симуляция сервиса email
	return nil
}`
		},
		uz: {
			title: 'Kichik funksiyalar - Yagona mas\'uliyat',
			description: `Katta funksiyani har biri yagona mas'uliyatga ega bo'lgan kichikroq funksiyalarga refaktoring qiling.

**Siz refaktoring qilasiz:**

1. **ProcessUserRegistration** - Juda ko'p ish qiladigan katta funksiya
2. Uni yagona mas'uliyatlarga ega kichikroq funksiyalarga ajrating:
   - ValidateUserInput
   - CreateUser
   - SendWelcomeEmail
   - LogRegistration

**Asosiy tushunchalar:**
- **Yagona mas'uliyat printsipi**: Har bir funksiya bitta narsani yaxshi qilishi kerak
- **Funksiya uzunligi**: 20 qatordan kam funksiyalarga intiling
- **Ta'riflovchi nomlar**: Aniq nomli kichik funksiyalar o'z-o'zini hujjatlaydi
- **Testlanuvchilik**: Kichik funksiyalarni test qilish oson

**Qachon funksiyalarni ajratish:**
- Funksiya bir nechta ish qilayotgan bo'lsa
- Funksiyani "Va" so'zisiz nomlash qiyin bo'lsa
- Funksiyani test qilish qiyin bo'lsa
- Funksiya bir nechta abstraktsiya darajalariga ega bo'lsa

**Cheklovlar:**
- Kamida 4 yordamchi funksiya ajrating
- Har bir yordamchi bitta mas'uliyatga ega bo'lishi kerak
- Asosiy funksiya yuqori darajali orkestrator bo'lishi kerak`,
			hint1: `ValidateUserInput ni barcha validatsiya mantiqini boshqarish uchun ajrating (email va parol tekshiruvlari). CreateUser ni foydalanuvchi yaratish va ma'lumotlar bazasiga saqlash uchun ajrating.`,
			hint2: `SendWelcomeEmail ni email yuborish mantiqi uchun ajrating (e'tibor bering, u xato qaytarmaydi, faqat logga yozadi). LogRegistration ni loglash uchun ajrating. Asosiy funksiya faqat bu 4 funksiyani ketma-ket chaqirishi kerak.`,
			whyItMatters: `Yagona mas'uliyatlarga ega kichik funksiyalar qo'llab-quvvatlanadigan kodning asosi.

**Kichik funksiyalar nima uchun muhim:**

**1. Kognitiv yukni kamaytirish**

Dasturchi oqimni darhol tushunadi, har qanday qadamga chuqurroq kirishi mumkin.

**2. Testlanuvchilik**

Har bir qismni mustaqil ravishda test qilish oson. Validatsiyani email yuborishdan alohida test qilish mumkin.

**3. Qayta ishlatish**

Validatsiya ajratilgan va login, parol tiklashda va h.k. qayta ishlatilishi mumkin.

**4. Yagona abstraktsiya darajasi**

Hammasi bir xil yuqori darajada. Past darajali tafsilotlar yordamchi funksiyalarda yashirilgan.`,
			solutionCode: `package principles

import (
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"strings"
)

type User struct {
	ID       int
	Email    string
	Password string
}

// ProcessUserRegistration foydalanuvchi ro'yxatdan o'tish jarayonini boshqaradi
// Bu funksiya endi aniq qadamlarga ega yuqori darajali koordinator
// Har bir qadam ixtisoslashtirilgan funksiyaga topshiriladi
func ProcessUserRegistration(email, password string) error {
	// 1-qadam: Kirishni tekshirish
	if err := ValidateUserInput(email, password); err != nil {
		return err
	}

	// 2-qadam: Foydalanuvchini yaratish va saqlash
	user, err := CreateUser(email, password)
	if err != nil {
		return err
	}

	// 3-qadam: Xush kelibsiz xabarini yuborish
	SendWelcomeEmail(user)

	// 4-qadam: Hodisani loglash
	LogRegistration(user)

	return nil
}

// ValidateUserInput email va parol talablariga mos kelishini tekshiradi
// Yagona mas'uliyat: Faqat kirishni tekshiradi, hech narsa yaratmaydi yoki saqlamaydi
func ValidateUserInput(email, password string) error {
	// Email formatini tekshirish
	if !strings.Contains(email, "@") || !strings.Contains(email, ".") {
		return errors.New("invalid email format")
	}

	// Parol uzunligini tekshirish
	if len(password) < 8 {
		return errors.New("password must be at least 8 characters")
	}

	// Parol murakkabligini tekshirish
	hasUpper := false
	hasLower := false
	hasDigit := false
	for _, char := range password {
		if char >= 'A' && char <= 'Z' {
			hasUpper = true
		}
		if char >= 'a' && char <= 'z' {
			hasLower = true
		}
		if char >= '0' && char <= '9' {
			hasDigit = true
		}
	}
	if !hasUpper || !hasLower || !hasDigit {
		return errors.New("password must contain uppercase, lowercase, and digit")
	}

	return nil
}

// CreateUser yangi foydalanuvchini xeshlanyan parol bilan yaratadi va ma'lumotlar bazasiga saqlaydi
// Yagona mas'uliyat: Foydalanuvchi yaratish va saqlash
func CreateUser(email, password string) (*User, error) {
	// Parolni xeshlash
	hash := sha256.Sum256([]byte(password))
	hashedPassword := hex.EncodeToString(hash[:])

	// Foydalanuvchi ob'ektini yaratish
	user := &User{
		ID:       generateUserID(),
		Email:    email,
		Password: hashedPassword,
	}

	// Ma'lumotlar bazasiga saqlash
	if err := saveUserToDB(user); err != nil {
		return nil, fmt.Errorf("failed to save user: %w", err)
	}

	return user, nil
}

// SendWelcomeEmail yangi foydalanuvchiga xush kelibsiz emailini yuboradi
// Yagona mas'uliyat: Faqat email xabarnomalarini yuborish
// Xatolar logga yoziladi lekin ro'yxatdan o'tishni to'xtatmaydi (muhim emas operatsiya)
func SendWelcomeEmail(user *User) {
	emailBody := fmt.Sprintf("Welcome to our platform, %s!", user.Email)
	if err := sendEmailService(user.Email, "Welcome!", emailBody); err != nil {
		fmt.Printf("Failed to send welcome email: %v\n", err)
	}
}

// LogRegistration ro'yxatdan o'tish hodisasini loglarda yozadi
// Yagona mas'uliyat: Faqat loglash
func LogRegistration(user *User) {
	fmt.Printf("[INFO] User registered successfully: %s (ID: %d)\n", user.Email, user.ID)
}

// Yordamchi funksiyalar (simulyatsiya)
var userIDCounter = 1000

func generateUserID() int {
	userIDCounter++
	return userIDCounter
}

func saveUserToDB(user *User) error {
	// Ma'lumotlar bazasiga saqlash simulyatsiyasi
	return nil
}

func sendEmailService(to, subject, body string) error {
	// Email servisi simulyatsiyasi
	return nil
}`
		}
	}
};

export default task;
