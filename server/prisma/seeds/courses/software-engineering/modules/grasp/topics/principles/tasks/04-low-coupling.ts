import { Task } from '../../../../../../../types';

export const task: Task = {
	slug: 'se-grasp-low-coupling',
	title: 'Low Coupling',
	difficulty: 'medium',
	tags: ['go', 'software-engineering', 'grasp', 'low-coupling'],
	estimatedTime: '40m',
	isPremium: false,
	youtubeUrl: '',
	description: `Implement the Low Coupling principle - minimize dependencies between classes to reduce the impact of changes.

**You will implement:**

1. **EmailSender interface** - Abstraction for sending emails
2. **SMTPEmailSender struct** - Concrete SMTP implementation
3. **NotificationService struct** - Uses EmailSender interface (not concrete type!)
4. **SendWelcomeEmail(email string) error** - Send notification using injected sender

**Key Concepts:**
- **Low Coupling**: Classes depend on abstractions, not concrete implementations
- **Dependency Injection**: Pass dependencies through constructor
- **Interface Segregation**: Small, focused interfaces

**Example Usage:**

\`\`\`go
// Low coupling - NotificationService doesn't know about SMTP
emailSender := NewSMTPEmailSender("smtp.gmail.com", 587)
notificationService := NewNotificationService(emailSender)

err := notificationService.SendWelcomeEmail("user@example.com")
// Easy to swap implementations without changing NotificationService!

// Can use different sender in tests
mockSender := &MockEmailSender{}
testService := NewNotificationService(mockSender)
\`\`\`

**Why Low Coupling?**
- **Flexibility**: Easy to change implementations
- **Testability**: Can inject mocks for testing
- **Maintainability**: Changes in EmailSender don't affect NotificationService

**Anti-pattern (Don't do this):**
\`\`\`go
// HIGH COUPLING - BAD!
type NotificationService struct {
    smtpHost string
    smtpPort int
}

func (n *NotificationService) SendEmail(to, message string) {
    // NotificationService knows SMTP details - tightly coupled!
    smtp.SendMail(n.smtpHost, nil, "from@example.com", []string{to}, []byte(message))
}
// Can't test without real SMTP server!
// Can't switch to SendGrid without rewriting NotificationService!
\`\`\`

**Constraints:**
- NotificationService must depend on EmailSender interface
- NotificationService must NOT import or reference SMTP directly
- EmailSender must be injected via constructor`,
	initialCode: `package principles

type EmailSender interface {
}

type SMTPEmailSender struct {
	host string
	port int
}

func NewSMTPEmailSender(host string, port int) *SMTPEmailSender {
	}
}

func (s *SMTPEmailSender) Send(to, subject, body string) error {
}

type NotificationService struct {
	// This creates LOW COUPLING - depends on interface, not concrete type
}

func NewNotificationService(emailSender EmailSender) *NotificationService {
}

func (n *NotificationService) SendWelcomeEmail(email string) error {
}`,
	solutionCode: `package principles

// EmailSender is an interface for sending emails
// Interface creates abstraction - low coupling
type EmailSender interface {
	Send(to, subject, body string) error
}

// SMTPEmailSender implements EmailSender using SMTP
type SMTPEmailSender struct {
	host string
	port int
}

func NewSMTPEmailSender(host string, port int) *SMTPEmailSender {
	return &SMTPEmailSender{
		host: host,
		port: port,
	}
}

func (s *SMTPEmailSender) Send(to, subject, body string) error {
	// In real implementation: use net/smtp
	// For this exercise, simulate successful send
	// fmt.Printf("Sending via SMTP %s:%d to %s\n", s.host, s.port, to)
	return nil
}

// NotificationService depends on EmailSender interface (LOW COUPLING)
type NotificationService struct {
	emailSender EmailSender	// interface, not concrete type!
}

// Dependency injection - pass interface, not concrete type
func NewNotificationService(emailSender EmailSender) *NotificationService {
	return &NotificationService{
		emailSender: emailSender,	// any EmailSender implementation works
	}
}

func (n *NotificationService) SendWelcomeEmail(email string) error {
	// NotificationService doesn't know HOW emails are sent
	// It just knows THAT they can be sent via the interface
	subject := "Welcome!"
	body := "Thank you for signing up."
	return n.emailSender.Send(email, subject, body)
}`,
	hint1: `Define EmailSender interface with Send(to, subject, body string) error. SMTPEmailSender.Send should return nil (simulating success).`,
	hint2: `NotificationService should have emailSender EmailSender field. NewNotificationService takes EmailSender and returns &NotificationService{emailSender: emailSender}.`,
	whyItMatters: `Low Coupling is critical for maintainable, testable, and flexible software systems.

**Why Low Coupling Matters:**

**1. Easy to Change Implementations**
With low coupling, you can swap implementations without changing dependent code:

\`\`\`go
// LOW COUPLING - GOOD!
type EmailSender interface {
    Send(to, subject, body string) error
}

type NotificationService struct {
    emailSender EmailSender  // depends on interface
}

// Can easily switch email providers
func main() {
    // Use SMTP
    sender := NewSMTPEmailSender("smtp.gmail.com", 587)
    service := NewNotificationService(sender)

    // Later, switch to SendGrid - NO changes to NotificationService!
    sender = NewSendGridEmailSender("api-key")
    service = NewNotificationService(sender)

    // Or use AWS SES - still NO changes to NotificationService!
    sender = NewSESEmailSender("region", "access-key")
    service = NewNotificationService(sender)
}

// HIGH COUPLING - BAD!
type NotificationService struct {
    smtpHost string
    smtpPort int
}

func (n *NotificationService) Send(to, subject, body string) {
    smtp.SendMail(n.smtpHost, ...) // directly coupled to SMTP!
}
// To switch to SendGrid, must rewrite NotificationService!
\`\`\`

**2. Testability**
Low coupling allows easy testing with mocks:

\`\`\`go
// Mock implementation for testing
type MockEmailSender struct {
    SentEmails []Email
}

func (m *MockEmailSender) Send(to, subject, body string) error {
    m.SentEmails = append(m.SentEmails, Email{to, subject, body})
    return nil
}

// Test without real email service
func TestNotificationService_SendWelcomeEmail(t *testing.T) {
    mock := &MockEmailSender{}
    service := NewNotificationService(mock)

    err := service.SendWelcomeEmail("test@example.com")
    if err != nil {
        t.Fatalf("unexpected error: %v", err)
    }

    if len(mock.SentEmails) != 1 {
        t.Errorf("expected 1 email, got %d", len(mock.SentEmails))
    }

    if mock.SentEmails[0].To != "test@example.com" {
        t.Errorf("wrong recipient: %s", mock.SentEmails[0].To)
    }
}
// Easy testing - no real SMTP server needed!
\`\`\`

**3. Real-World Example: Payment Processing**
\`\`\`go
// LOW COUPLING design
type PaymentProcessor interface {
    Charge(amount float64, token string) (*Payment, error)
    Refund(paymentID string, amount float64) error
}

type OrderService struct {
    paymentProcessor PaymentProcessor  // interface!
}

// Stripe implementation
type StripeProcessor struct {
    apiKey string
}

func (s *StripeProcessor) Charge(amount float64, token string) (*Payment, error) {
    // Stripe-specific logic
    return stripe.Charges.New(&stripe.ChargeParams{
        Amount:   int64(amount * 100),
        Currency: "usd",
        Source:   token,
    })
}

// PayPal implementation
type PayPalProcessor struct {
    clientID     string
    clientSecret string
}

func (p *PayPalProcessor) Charge(amount float64, token string) (*Payment, error) {
    // PayPal-specific logic
    return paypal.Charge(amount, token)
}

// OrderService works with ANY payment processor
func (o *OrderService) CheckoutOrder(orderID int, paymentToken string) error {
    order := o.getOrder(orderID)
    // OrderService doesn't know if it's Stripe, PayPal, or Square
    payment, err := o.paymentProcessor.Charge(order.Total, paymentToken)
    if err != nil {
        return err
    }
    order.PaymentID = payment.ID
    return o.updateOrder(order)
}
\`\`\`

**4. Dependency Injection Patterns**
\`\`\`go
// Constructor injection (preferred)
type UserService struct {
    db    Database
    cache Cache
    email EmailSender
}

func NewUserService(db Database, cache Cache, email EmailSender) *UserService {
    return &UserService{
        db:    db,
        cache: cache,
        email: email,
    }
}

// All dependencies are visible and testable
func main() {
    db := NewPostgresDB("connection-string")
    cache := NewRedisCache("redis://localhost")
    email := NewSMTPSender("smtp.gmail.com", 587)

    service := NewUserService(db, cache, email)
}
\`\`\`

**5. Measuring Coupling**
High coupling signs:
- Changes in one class require changes in many other classes
- Can't test a class without instantiating many dependencies
- Can't reuse a class without bringing lots of other classes

Low coupling signs:
- Changes localized to one or few classes
- Easy to write unit tests with mocks
- Classes can be reused independently

**6. When Some Coupling is OK**
Not all coupling is bad:
\`\`\`go
// OK: Coupling to standard library
import "time"

type Event struct {
    OccurredAt time.Time  // coupling to time package is fine
}

// OK: Coupling to stable domain concepts
type Order struct {
    Items []OrderItem  // Order naturally coupled to OrderItem
}

// AVOID: Coupling to volatile implementation details
type OrderService struct {
    postgresDB *sql.DB  // BAD: coupled to specific database
}

// BETTER: Depend on interface
type OrderService struct {
    db Database  // interface allows any database
}
\`\`\`

**Common Mistakes:**
- Creating interfaces with too many methods (prefer small interfaces)
- Not using dependency injection
- Depending on concrete types instead of interfaces
- Creating interfaces you don't need (don't create interface if only one implementation)

**Rule of Thumb:**
Depend on abstractions (interfaces), not concretions. Inject dependencies rather than creating them inside classes.`,
	order: 3,
	testCode: `package principles

import (
	"testing"
)

// Test1: SMTPEmailSender implements EmailSender
func Test1(t *testing.T) {
	sender := NewSMTPEmailSender("smtp.test.com", 587)
	var _ EmailSender = sender // compile-time check
	if sender == nil {
		t.Error("Should create SMTP sender")
	}
}

// Test2: SMTPEmailSender Send returns nil
func Test2(t *testing.T) {
	sender := NewSMTPEmailSender("smtp.test.com", 587)
	err := sender.Send("to@test.com", "Subject", "Body")
	if err != nil {
		t.Errorf("Send should return nil, got %v", err)
	}
}

// Test3: NotificationService creation
func Test3(t *testing.T) {
	sender := NewSMTPEmailSender("smtp.test.com", 587)
	service := NewNotificationService(sender)
	if service == nil {
		t.Error("Should create NotificationService")
	}
}

// Test4: SendWelcomeEmail runs without error
func Test4(t *testing.T) {
	sender := NewSMTPEmailSender("smtp.test.com", 587)
	service := NewNotificationService(sender)
	err := service.SendWelcomeEmail("user@test.com")
	if err != nil {
		t.Errorf("SendWelcomeEmail should succeed, got %v", err)
	}
}

// Test5: SMTPEmailSender stores host and port
func Test5(t *testing.T) {
	sender := NewSMTPEmailSender("mail.example.com", 25)
	// Just verify creation works - fields are private
	if sender == nil {
		t.Error("Sender should not be nil")
	}
}

// Test6: Can use different EmailSender with NotificationService
func Test6(t *testing.T) {
	sender1 := NewSMTPEmailSender("smtp1.test.com", 587)
	sender2 := NewSMTPEmailSender("smtp2.test.com", 465)
	service1 := NewNotificationService(sender1)
	service2 := NewNotificationService(sender2)
	err1 := service1.SendWelcomeEmail("a@test.com")
	err2 := service2.SendWelcomeEmail("b@test.com")
	if err1 != nil || err2 != nil {
		t.Error("Both services should work")
	}
}

// Test7: Send with empty parameters
func Test7(t *testing.T) {
	sender := NewSMTPEmailSender("smtp.test.com", 587)
	err := sender.Send("", "", "")
	if err != nil {
		t.Error("Should handle empty parameters")
	}
}

// Test8: SendWelcomeEmail with different emails
func Test8(t *testing.T) {
	sender := NewSMTPEmailSender("smtp.test.com", 587)
	service := NewNotificationService(sender)
	emails := []string{"a@test.com", "b@test.com", "c@test.com"}
	for _, email := range emails {
		if err := service.SendWelcomeEmail(email); err != nil {
			t.Errorf("Failed for %s: %v", email, err)
		}
	}
}

// Test9: Multiple NotificationService instances are independent
func Test9(t *testing.T) {
	sender := NewSMTPEmailSender("smtp.test.com", 587)
	service1 := NewNotificationService(sender)
	service2 := NewNotificationService(sender)
	if service1 == service2 {
		t.Error("Services should be different instances")
	}
}

// Test10: Send with special characters in email
func Test10(t *testing.T) {
	sender := NewSMTPEmailSender("smtp.test.com", 587)
	err := sender.Send("user+tag@test.com", "Test Subject!", "Body with special chars: <>&")
	if err != nil {
		t.Errorf("Should handle special characters, got %v", err)
	}
}
`,
	translations: {
		ru: {
			title: 'Низкая связанность',
			description: `Реализуйте принцип Низкой связанности — минимизируйте зависимости между классами, чтобы уменьшить влияние изменений.

**Вы реализуете:**

1. **EmailSender interface** — Абстракция для отправки email
2. **SMTPEmailSender struct** — Конкретная SMTP-реализация
3. **NotificationService struct** — Использует интерфейс EmailSender (не конкретный тип!)
4. **SendWelcomeEmail(email string) error** — Отправка уведомления через внедрённый sender

**Ключевые концепции:**
- **Низкая связанность**: Классы зависят от абстракций, а не от конкретных реализаций
- **Внедрение зависимостей**: Передача зависимостей через конструктор
- **Разделение интерфейсов**: Малые, сфокусированные интерфейсы

**Зачем нужна Низкая связанность?**
- **Гибкость**: Легко менять реализации
- **Тестируемость**: Можно внедрять моки для тестирования
- **Поддерживаемость**: Изменения в EmailSender не влияют на NotificationService

**Ограничения:**
- NotificationService должен зависеть от интерфейса EmailSender
- NotificationService НЕ должен импортировать или ссылаться на SMTP напрямую
- EmailSender должен внедряться через конструктор`,
			hint1: `Определите интерфейс EmailSender с Send(to, subject, body string) error. SMTPEmailSender.Send должен вернуть nil (имитация успеха).`,
			hint2: `NotificationService должен иметь поле emailSender EmailSender. NewNotificationService принимает EmailSender и возвращает &NotificationService{emailSender: emailSender}.`,
			whyItMatters: `Низкая связанность критична для поддерживаемых, тестируемых и гибких программных систем.

**Почему Низкая связанность важна:**

**1. Легко менять реализации**
При низкой связанности можно менять реализации без изменения зависимого кода.

**Распространённые ошибки:**
- Создание интерфейсов со слишком многими методами (предпочитайте малые интерфейсы)
- Неиспользование внедрения зависимостей
- Зависимость от конкретных типов вместо интерфейсов`,
			solutionCode: `package principles

// EmailSender - интерфейс для отправки email
// Интерфейс создаёт абстракцию - низкая связанность
type EmailSender interface {
	Send(to, subject, body string) error
}

// SMTPEmailSender реализует EmailSender через SMTP
type SMTPEmailSender struct {
	host string
	port int
}

func NewSMTPEmailSender(host string, port int) *SMTPEmailSender {
	return &SMTPEmailSender{
		host: host,
		port: port,
	}
}

func (s *SMTPEmailSender) Send(to, subject, body string) error {
	// В реальной реализации: использовать net/smtp
	// Для этого упражнения имитируем успешную отправку
	return nil
}

// NotificationService зависит от интерфейса EmailSender (НИЗКАЯ СВЯЗАННОСТЬ)
type NotificationService struct {
	emailSender EmailSender	// интерфейс, не конкретный тип!
}

// Внедрение зависимостей - передаём интерфейс, не конкретный тип
func NewNotificationService(emailSender EmailSender) *NotificationService {
	return &NotificationService{
		emailSender: emailSender,	// работает любая реализация EmailSender
	}
}

func (n *NotificationService) SendWelcomeEmail(email string) error {
	// NotificationService не знает КАК отправляются email
	// Он только знает ЧТО они могут быть отправлены через интерфейс
	subject := "Welcome!"
	body := "Thank you for signing up."
	return n.emailSender.Send(email, subject, body)
}`
		},
		uz: {
			title: 'Low Coupling (Past bog\'lanish)',
			description: `Low Coupling prinsipini amalga oshiring — o'zgarishlar ta'sirini kamaytirish uchun klasslar orasidagi bog'liqliklarni minimallashtiring.

**Siz amalga oshirasiz:**

1. **EmailSender interface** — Email yuborish uchun abstraktsiya
2. **SMTPEmailSender struct** — Konkret SMTP implementatsiyasi
3. **NotificationService struct** — EmailSender interfeysidan foydalanadi (konkret tipdan emas!)
4. **SendWelcomeEmail(email string) error** — Injeksiya qilingan sender orqali bildirishnoma yuborish

**Asosiy tushunchalar:**
- **Low Coupling**: Klasslar konkret implementatsiyalardan emas, abstraktsiyalardan bog'liq
- **Dependency Injection**: Bog'liqliklarni konstruktor orqali o'tkazish
- **Interface Segregation**: Kichik, fokusli interfeyslar

**Nima uchun Low Coupling?**
- **Moslashuvchanlik**: Implementatsiyalarni osongina o'zgartirish
- **Testlanish**: Test uchun mocklar injeksiya qilish mumkin
- **Parvarish qilish**: EmailSender dagi o'zgarishlar NotificationService ga ta'sir qilmaydi

**Cheklovlar:**
- NotificationService EmailSender interfeysiga bog'liq bo'lishi kerak
- NotificationService to'g'ridan-to'g'ri SMTP ni import qilmasligi yoki havola qilmasligi kerak
- EmailSender konstruktor orqali injeksiya qilinishi kerak`,
			hint1: `EmailSender interfeysini Send(to, subject, body string) error bilan aniqlang. SMTPEmailSender.Send nil qaytarishi kerak (muvaffaqiyatni simulyatsiya qilish).`,
			hint2: `NotificationService emailSender EmailSender maydoniga ega bo'lishi kerak. NewNotificationService EmailSender ni qabul qiladi va &NotificationService{emailSender: emailSender} qaytaradi.`,
			whyItMatters: `Low Coupling parvarish qilinadigan, testlangan va moslashuvchan dasturiy ta'minot tizimlari uchun muhimdir.

**Low Coupling nima uchun muhim:**

**1. Implementatsiyalarni osongina o'zgartirish**
Past bog'lanish bilan siz bog'liq kodni o'zgartirmasdan implementatsiyalarni almashtira olasiz.

**Umumiy xatolar:**
- Juda ko'p metodlar bilan interfeyslar yaratish (kichik interfeyslarni afzal ko'ring)
- Dependency injection dan foydalanmaslik
- Interfeyslar o'rniga konkret tiplarga bog'liqlik`,
			solutionCode: `package principles

// EmailSender - email yuborish uchun interfeys
// Interfeys abstraktsiya yaratadi - past bog'lanish
type EmailSender interface {
	Send(to, subject, body string) error
}

// SMTPEmailSender EmailSender ni SMTP orqali amalga oshiradi
type SMTPEmailSender struct {
	host string
	port int
}

func NewSMTPEmailSender(host string, port int) *SMTPEmailSender {
	return &SMTPEmailSender{
		host: host,
		port: port,
	}
}

func (s *SMTPEmailSender) Send(to, subject, body string) error {
	// Haqiqiy implementatsiyada: net/smtp dan foydalaning
	// Bu mashq uchun muvaffaqiyatli yuborishni simulyatsiya qilamiz
	return nil
}

// NotificationService EmailSender interfeysiga bog'liq (PAST BOG'LANISH)
type NotificationService struct {
	emailSender EmailSender	// interfeys, konkret tip emas!
}

// Dependency injection - konkret tip emas, interfeys o'tkazamiz
func NewNotificationService(emailSender EmailSender) *NotificationService {
	return &NotificationService{
		emailSender: emailSender,	// har qanday EmailSender implementatsiyasi ishlaydi
	}
}

func (n *NotificationService) SendWelcomeEmail(email string) error {
	// NotificationService emaillar QANDAY yuborilishini bilmaydi
	// U faqat ular interfeys orqali yuborilishi MUMKINLIGINI biladi
	subject := "Welcome!"
	body := "Thank you for signing up."
	return n.emailSender.Send(email, subject, body)
}`
		}
	}
};

export default task;
