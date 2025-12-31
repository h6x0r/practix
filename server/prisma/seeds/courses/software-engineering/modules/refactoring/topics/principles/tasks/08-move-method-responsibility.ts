import { Task } from '../../../../../../../types';

export const task: Task = {
	slug: 'se-refactoring-move-method-responsibility',
	title: 'Move Method - Misplaced Responsibility',
	difficulty: 'medium',
	tags: ['refactoring', 'move-method', 'clean-code', 'go'],
	estimatedTime: '25m',
	isPremium: false,
	youtubeUrl: '',
	description: `Move a method to the struct that should be responsible for that behavior, following Single Responsibility Principle.

**You will refactor:**

1. **User.SendNotification()** - User shouldn't know how to send notifications
2. Move to **NotificationService.SendToUser()** - Service handles sending
3. Update **User.NotifyPasswordChange()** to use the service

**Key Concepts:**
- **Single Responsibility**: Each struct has one reason to change
- **Separation of Concerns**: User is about data, NotificationService is about sending
- **Dependency Direction**: High-level modules shouldn't depend on low-level details
- **Cohesion**: Related behaviors stay together

**Before Refactoring:**

\`\`\`go
type User struct { ... }

func (u *User) SendNotification(message string) {
    // User knows SMTP details - wrong responsibility!
    smtp.SendMail(u.Email, message)
}
\`\`\`

**After Refactoring:**

\`\`\`go
type NotificationService struct { ... }

func (ns *NotificationService) SendToUser(user User, message string) {
    // Service handles notification logic
    smtp.SendMail(user.Email, message)
}
\`\`\`

**When to Move for Responsibility:**
- Method doesn't fit struct's core purpose
- Violates Single Responsibility Principle
- Struct has too many responsibilities
- Method deals with external systems (DB, network, files)
- Better separation of concerns by moving

**Constraints:**
- Move SendNotification logic to NotificationService.SendToUser()
- Update NotifyPasswordChange to use the service
- NotificationService must be passed as parameter
- Maintain exact same notification behavior`,
	initialCode: `package refactoring

import "fmt"

type User struct {
	Email string
	Name  string
}

func (u *User) SendNotification(message string) {
}

func (u *User) NotifyPasswordChange() {
}

type NotificationService struct {
	// Notification service configuration
	SMTPHost string
	SMTPPort int
}`,
	solutionCode: `package refactoring

import "fmt"

type User struct {
	Email string
	Name  string
}

// NotifyPasswordChange now delegates to NotificationService
func (u *User) NotifyPasswordChange(service *NotificationService) {
	message := "Your password has been changed"
	service.SendToUser(*u, message)	// Tell service to send, don't do it yourself
}

type NotificationService struct {
	// Notification service configuration
	SMTPHost string
	SMTPPort int
}

// SendToUser is where notification logic belongs - in the service
func (ns *NotificationService) SendToUser(user User, message string) {
	// NotificationService knows how to send notifications
	fmt.Printf("Sending to %s (%s): %s\n", user.Name, user.Email, message)
}`,
	hint1: `Create the SendToUser method on NotificationService that takes (user User, message string) parameters and contains the fmt.Printf logic from the old SendNotification method.`,
	hint2: `Remove the SendNotification method from User. Update NotifyPasswordChange to accept *NotificationService as a parameter and call service.SendToUser(*u, message) instead.`,
	whyItMatters: `Moving methods to their proper responsibility improves maintainability and follows the Single Responsibility Principle.

**Why Proper Responsibility Placement Matters:**

**1. Single Responsibility Principle**
Each struct should have one reason to change:

\`\`\`go
// Before: User has multiple responsibilities
type User struct {
    Email string
    Name  string
}

func (u *User) SaveToDatabase() error {
    // Database logic in User - wrong!
    return db.Exec("INSERT INTO users...")
}

func (u *User) SendWelcomeEmail() error {
    // Email logic in User - wrong!
    return smtp.Send(u.Email, "Welcome!")
}

func (u *User) ValidateEmail() bool {
    // Validation logic - also questionable
    return strings.Contains(u.Email, "@")
}

// Changes to database, email system, or validation rules all require changing User!

// After: Each responsibility in its own struct
type User struct {
    Email string
    Name  string
}

type UserRepository struct {
    db *Database
}

func (ur *UserRepository) Save(user User) error {
    return ur.db.Exec("INSERT INTO users...")
}

type EmailService struct {
    smtp *SMTPClient
}

func (es *EmailService) SendWelcome(user User) error {
    return es.smtp.Send(user.Email, "Welcome!")
}

type EmailValidator struct {}

func (ev *EmailValidator) IsValid(email string) bool {
    return strings.Contains(email, "@")
}

// Now: Database changes affect UserRepository, email changes affect EmailService
\`\`\`

**2. Testability**
Easy to mock dependencies:

\`\`\`go
// Before: Can't test User.ProcessOrder without real database and payment gateway
type User struct {
    ID string
}

func (u *User) ProcessOrder(order Order) error {
    // Hard to test - requires real DB and payment system
    if err := database.Save(order); err != nil {
        return err
    }
    if err := paymentGateway.Charge(order.Total); err != nil {
        return err
    }
    return nil
}

// After: Can test with mocks
type OrderProcessor struct {
    repo     OrderRepository
    payments PaymentService
}

func (op *OrderProcessor) Process(order Order) error {
    if err := op.repo.Save(order); err != nil {
        return err
    }
    if err := op.payments.Charge(order.Total); err != nil {
        return err
    }
    return nil
}

// Testing is easy
func TestProcessOrder(t *testing.T) {
    mockRepo := &MockOrderRepository{}
    mockPayment := &MockPaymentService{}
    processor := OrderProcessor{repo: mockRepo, payments: mockPayment}

    err := processor.Process(testOrder)
    assert.NoError(t, err)
}
\`\`\`

**3. Reusability**
Services can be used by multiple structs:

\`\`\`go
// Before: Duplicate notification logic in every struct
type User struct { Email string }
func (u *User) SendNotification(msg string) { /* SMTP code */ }

type Admin struct { Email string }
func (a *Admin) SendNotification(msg string) { /* SMTP code - duplicated! */ }

type Moderator struct { Email string }
func (m *Moderator) SendNotification(msg string) { /* SMTP code - duplicated! */ }

// After: Single notification service used by all
type NotificationService struct {
    smtp *SMTPClient
}

func (ns *NotificationService) Send(email, message string) error {
    return ns.smtp.Send(email, message)
}

// All structs use the same service
processor.notifications.Send(user.Email, "Welcome")
processor.notifications.Send(admin.Email, "Alert")
processor.notifications.Send(mod.Email, "Report")
\`\`\`

**4. Configuration and Dependencies**
External dependencies isolated in services:

\`\`\`go
// Before: Configuration scattered
type OrderHandler struct {}

func (oh *OrderHandler) Process(order Order) {
    // Hardcoded configuration
    timeout := 30 * time.Second
    maxRetries := 3
    apiKey := "hardcoded-key"

    // Process with these values
}

// After: Configuration in service
type PaymentConfig struct {
    Timeout    time.Duration
    MaxRetries int
    APIKey     string
}

type PaymentService struct {
    config PaymentConfig
}

func NewPaymentService(cfg PaymentConfig) *PaymentService {
    return &PaymentService{config: cfg}
}

func (ps *PaymentService) Charge(amount float64) error {
    // Use ps.config.Timeout, ps.config.MaxRetries, etc.
}

// Easy to change configuration without modifying structs
\`\`\`

**5. Easier Maintenance**
Changes localized to appropriate services:

\`\`\`go
// Before: Changing email provider requires modifying all structs
type User struct {}
func (u *User) Notify() { /* SendGrid code */ }

type Order struct {}
func (o *Order) SendConfirmation() { /* SendGrid code */ }

// Need to switch to Mailgun? Must change User, Order, and 10 other structs!

// After: One change in one place
type EmailService struct {
    provider EmailProvider // interface
}

func (es *EmailService) Send(to, subject, body string) error {
    return es.provider.Send(to, subject, body)
}

// Switch provider: just change EmailService initialization
// No other code needs to change
\`\`\`

**Real-World Example - E-commerce System:**

\`\`\`go
// Before: Order does everything
type Order struct {
    ID     string
    Items  []Item
    Total  float64
    UserID string
}

func (o *Order) Process() error {
    // Database logic
    db.Save(o)

    // Payment logic
    paymentAPI.Charge(o.Total)

    // Email logic
    smtp.Send("Order confirmed")

    // Analytics logic
    analytics.Track("order_placed", o.ID)

    // Inventory logic
    for _, item := range o.Items {
        inventory.Decrease(item.ID, item.Quantity)
    }

    return nil
}

// After: Proper separation
type Order struct {
    ID     string
    Items  []Item
    Total  float64
    UserID string
}

type OrderService struct {
    repo         OrderRepository
    payments     PaymentService
    emails       EmailService
    analytics    AnalyticsService
    inventory    InventoryService
}

func (os *OrderService) Process(order Order) error {
    if err := os.repo.Save(order); err != nil {
        return err
    }

    if err := os.payments.Charge(order.Total); err != nil {
        return err
    }

    os.emails.SendOrderConfirmation(order)
    os.analytics.TrackOrder(order)

    for _, item := range order.Items {
        os.inventory.Decrease(item.ID, item.Quantity)
    }

    return nil
}
\`\`\`

**Benefits:**
- Each struct has clear, single purpose
- Easy to test with mocks/stubs
- Configuration centralized in services
- Easy to swap implementations
- Changes are localized
- Code is more maintainable and extensible`,
	order: 7,
	testCode: `package refactoring

import (
	"testing"
)

// Test1: NotificationService.SendToUser runs without panic
func Test1(t *testing.T) {
	service := &NotificationService{SMTPHost: "smtp.test.com", SMTPPort: 587}
	user := User{Email: "test@example.com", Name: "Test User"}
	service.SendToUser(user, "Hello") // Should not panic
}

// Test2: User.NotifyPasswordChange calls service
func Test2(t *testing.T) {
	service := &NotificationService{SMTPHost: "smtp.test.com", SMTPPort: 587}
	user := &User{Email: "user@example.com", Name: "John"}
	user.NotifyPasswordChange(service) // Should not panic
}

// Test3: NotificationService struct fields
func Test3(t *testing.T) {
	service := NotificationService{SMTPHost: "mail.example.com", SMTPPort: 25}
	if service.SMTPHost != "mail.example.com" || service.SMTPPort != 25 {
		t.Error("NotificationService fields not set correctly")
	}
}

// Test4: User struct fields
func Test4(t *testing.T) {
	user := User{Email: "test@test.com", Name: "Test"}
	if user.Email != "test@test.com" || user.Name != "Test" {
		t.Error("User fields not set correctly")
	}
}

// Test5: Multiple users can notify password change
func Test5(t *testing.T) {
	service := &NotificationService{SMTPHost: "smtp.test.com", SMTPPort: 587}
	user1 := &User{Email: "user1@test.com", Name: "User One"}
	user2 := &User{Email: "user2@test.com", Name: "User Two"}

	user1.NotifyPasswordChange(service)
	user2.NotifyPasswordChange(service)
}

// Test6: Service can send to multiple users
func Test6(t *testing.T) {
	service := &NotificationService{SMTPHost: "smtp.test.com", SMTPPort: 587}
	user1 := User{Email: "a@b.com", Name: "A"}
	user2 := User{Email: "c@d.com", Name: "C"}

	service.SendToUser(user1, "Message 1")
	service.SendToUser(user2, "Message 2")
}

// Test7: SendToUser with empty message
func Test7(t *testing.T) {
	service := &NotificationService{SMTPHost: "smtp.test.com", SMTPPort: 587}
	user := User{Email: "test@test.com", Name: "Test"}
	service.SendToUser(user, "") // Should handle empty message
}

// Test8: SendToUser with long message
func Test8(t *testing.T) {
	service := &NotificationService{SMTPHost: "smtp.test.com", SMTPPort: 587}
	user := User{Email: "test@test.com", Name: "Test"}
	longMessage := "This is a very long message that contains many words"
	service.SendToUser(user, longMessage)
}

// Test9: NotificationService with default port
func Test9(t *testing.T) {
	service := &NotificationService{SMTPHost: "localhost"}
	user := User{Email: "test@test.com", Name: "Test"}
	service.SendToUser(user, "Test message")
}

// Test10: User with special characters in name
func Test10(t *testing.T) {
	service := &NotificationService{SMTPHost: "smtp.test.com", SMTPPort: 587}
	user := &User{Email: "special@test.com", Name: "O'Brien"}
	user.NotifyPasswordChange(service)
}
`,
	translations: {
		ru: {
			title: 'Move Method - Неправильная ответственность',
			description: `Переместите метод в структуру, которая должна отвечать за это поведение, следуя принципу единственной ответственности.

**Вы выполните рефакторинг:**

1. **User.SendNotification()** - User не должен знать, как отправлять уведомления
2. Переместить в **NotificationService.SendToUser()** - Сервис обрабатывает отправку
3. Обновить **User.NotifyPasswordChange()** для использования сервиса`,
			hint1: `Создайте метод SendToUser в NotificationService, который принимает параметры (user User, message string) и содержит логику fmt.Printf из старого метода SendNotification.`,
			hint2: `Удалите метод SendNotification из User. Обновите NotifyPasswordChange, чтобы он принимал *NotificationService как параметр и вызывал service.SendToUser(*u, message) вместо этого.`,
			whyItMatters: `Перемещение методов к их правильной ответственности улучшает поддерживаемость и следует принципу единственной ответственности.`
		},
		uz: {
			title: 'Move Method - Noto\'g\'ri mas\'uliyat',
			description: `Metodini ushbu xatti-harakat uchun javobgar bo'lishi kerak bo'lgan strukturaga ko'chiring, Yagona mas'uliyat printsipiga amal qilgan holda.

**Siz refaktoring qilasiz:**

1. **User.SendNotification()** - User bildirishnomalarni qanday yuborishni bilmasligi kerak
2. Ko'chirish **NotificationService.SendToUser()** ga - Servis yuborishni boshqaradi
3. **User.NotifyPasswordChange()** ni servisdan foydalanish uchun yangilash`,
			hint1: `NotificationService da SendToUser metodini yarating, u (user User, message string) parametrlarini qabul qiladi va eski SendNotification metodidan fmt.Printf logikasini o'z ichiga oladi.`,
			hint2: `User dan SendNotification metodini o'chiring. NotifyPasswordChange ni *NotificationService parametr sifatida qabul qilishi va service.SendToUser(*u, message) ni chaqirishi uchun yangilang.`,
			whyItMatters: `Metodlarni to'g'ri mas'uliyatga ko'chirish qo'llab-quvvatlashni yaxshilaydi va Yagona mas'uliyat printsipiga amal qiladi.`
		}
	}
};

export default task;
