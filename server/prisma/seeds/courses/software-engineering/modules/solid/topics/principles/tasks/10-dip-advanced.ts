import { Task } from '../../../../../../../types';

export const task: Task = {
	slug: 'se-solid-dip-advanced',
	title: 'Dependency Inversion Principle - Advanced',
	difficulty: 'hard',
	tags: ['go', 'solid', 'dip', 'dependency-injection', 'advanced'],
	estimatedTime: '40m',
	isPremium: false,
	youtubeUrl: '',
	description: `Apply DIP to a complex order processing system with multiple dependencies.

**Current Problem:**

An OrderProcessor directly creates and manages all its dependencies (payment, inventory, notification), making it impossible to test or extend.

**Your task:**

Refactor to use dependency injection with multiple abstractions:

1. **PaymentGateway interface** - Payment processing abstraction
2. **InventoryService interface** - Inventory management abstraction
3. **NotificationService interface** - Notification abstraction
4. **OrderProcessor** - High-level orchestrator depending only on interfaces
5. **StripeGateway, InventoryManager, EmailNotifier** - Concrete implementations

**Key Concepts:**
- **Multiple dependencies**: Inject all dependencies via constructor
- **Constructor injection**: Pass dependencies when creating object
- **Testability**: Easy to test with mocks
- **Flexibility**: Swap implementations without changing core logic

**Example Usage:**

\`\`\`go
// Production configuration
payment := &StripeGateway{APIKey: "sk_live_..."}
inventory := &InventoryManager{DB: db}
notifier := &EmailNotifier{SMTPHost: "smtp.gmail.com"}

processor := NewOrderProcessor(payment, inventory, notifier)
processor.Process(order)

// Test configuration
mockPayment := &MockPaymentGateway{}
mockInventory := &MockInventoryService{}
mockNotifier := &MockNotificationService{}

testProcessor := NewOrderProcessor(mockPayment, mockInventory, mockNotifier)
testProcessor.Process(testOrder)  // fully testable!
\`\`\`

**Real-World Impact:**
- Test without external services
- Switch payment providers easily
- Different notification channels per environment
- Clear separation of concerns`,
	initialCode: `package principles

import (
	"fmt"
)

type Order struct {
	ID       int
	Total    float64
	Customer string
}

type OrderProcessor struct {
	// No abstraction - creates dependencies internally!
}

func NewOrderProcessor() *OrderProcessor {
	return &OrderProcessor{}
}

func (op *OrderProcessor) Process(order *Order) error {

	if err := payment.Charge(order.Total); err != nil {
		return err
	}

	if err := inventory.Reserve(order.Items); err != nil {
		return err
	}

	return nil
}

type StripeGateway struct {
	APIKey string
}

func (sg *StripeGateway) Charge(amount float64) error {
	return nil
}

type InventoryManager struct{}

func (im *InventoryManager) Reserve(items []string) error {
	return nil
}

type EmailNotifier struct{}

func (en *EmailNotifier) Send(to, message string) error {
	return nil
}

type PaymentGateway interface {
}

type InventoryService interface {
}

type NotificationService interface {
}

type StripeGatewayRefactored struct {
	APIKey string
}

func (sg *StripeGatewayRefactored) Charge(amount float64) error {
}

type PayPalGateway struct {
	ClientID string
}

func (pg *PayPalGateway) Charge(amount float64) error {
}

type InventoryManagerRefactored struct {
	Database string
}

func (im *InventoryManagerRefactored) Reserve(items []string) error {
}

type EmailNotifierRefactored struct {
	SMTPHost string
}

func (en *EmailNotifierRefactored) Send(recipient, message string) error {
}

type SMSNotifier struct {
	Provider string
}

func (sn *SMSNotifier) Send(recipient, message string) error {
}

type OrderProcessorRefactored struct {
	payment     PaymentGateway      // abstraction, not concrete type
	inventory   InventoryService    // abstraction, not concrete type
	notifier    NotificationService // abstraction, not concrete type
}

func NewOrderProcessorRefactored(
}

func (op *OrderProcessorRefactored) Process(order *Order) error {
}`,
	solutionCode: `package principles

import (
	"fmt"
)

type Order struct {
	ID       int
	Items    []string
	Total    float64
	Customer string
}

// PaymentGateway - abstraction for payment processing
// DIP compliant: high and low-level modules depend on this
type PaymentGateway interface {
	Charge(amount float64) error
}

// InventoryService - abstraction for inventory management
// DIP compliant: defines contract without implementation
type InventoryService interface {
	Reserve(items []string) error
}

// NotificationService - abstraction for notifications
// DIP compliant: allows multiple notification implementations
type NotificationService interface {
	Send(recipient, message string) error
}

// StripeGatewayRefactored - concrete payment implementation
type StripeGatewayRefactored struct {
	APIKey string
}

// Charge processes payment via Stripe
func (sg *StripeGatewayRefactored) Charge(amount float64) error {
	fmt.Printf("Stripe: Charging $%.2f with key %s\\n", amount, sg.APIKey)
	// In production: call Stripe API
	return nil
}

// PayPalGateway - alternative payment implementation
type PayPalGateway struct {
	ClientID string
}

// Charge processes payment via PayPal
func (pg *PayPalGateway) Charge(amount float64) error {
	fmt.Printf("PayPal: Charging $%.2f with client %s\\n", amount, pg.ClientID)
	// In production: call PayPal API
	return nil
}

// InventoryManagerRefactored - concrete inventory implementation
type InventoryManagerRefactored struct {
	Database string
}

// Reserve reserves items in inventory
func (im *InventoryManagerRefactored) Reserve(items []string) error {
	fmt.Printf("Inventory (%s): Reserving items %v\\n", im.Database, items)
	// In production: update database
	return nil
}

// EmailNotifierRefactored - concrete email notification implementation
type EmailNotifierRefactored struct {
	SMTPHost string
}

// Send sends email notification
func (en *EmailNotifierRefactored) Send(recipient, message string) error {
	fmt.Printf("Email (%s): Sending to %s: %s\\n", en.SMTPHost, recipient, message)
	// In production: send via SMTP
	return nil
}

// SMSNotifier - alternative notification implementation
type SMSNotifier struct {
	Provider string
}

// Send sends SMS notification
func (sn *SMSNotifier) Send(recipient, message string) error {
	fmt.Printf("SMS (%s): Sending to %s: %s\\n", sn.Provider, recipient, message)
	// In production: send via SMS provider
	return nil
}

// OrderProcessorRefactored - high-level module depending ONLY on abstractions
// DIP compliant: no concrete dependencies
type OrderProcessorRefactored struct {
	payment   PaymentGateway      // interface, not concrete type
	inventory InventoryService    // interface, not concrete type
	notifier  NotificationService // interface, not concrete type
}

// NewOrderProcessorRefactored - constructor injection pattern
// DIP compliant: receives all dependencies from outside
func NewOrderProcessorRefactored(
	payment PaymentGateway,
	inventory InventoryService,
	notifier NotificationService,
) *OrderProcessorRefactored {
	return &OrderProcessorRefactored{
		payment:   payment,   // any PaymentGateway implementation
		inventory: inventory, // any InventoryService implementation
		notifier:  notifier,  // any NotificationService implementation
	}
}

// Process orchestrates order processing using injected dependencies
// DIP compliant: works with any implementation of the interfaces
func (op *OrderProcessorRefactored) Process(order *Order) error {
	// Step 1: Charge payment using injected gateway
	if err := op.payment.Charge(order.Total); err != nil {
		return fmt.Errorf("payment failed: %w", err)
	}

	// Step 2: Reserve inventory using injected service
	if err := op.inventory.Reserve(order.Items); err != nil {
		return fmt.Errorf("inventory reservation failed: %w", err)
	}

	// Step 3: Send notification using injected notifier
	message := fmt.Sprintf("Order #%d processed successfully", order.ID)
	if err := op.notifier.Send(order.Customer, message); err != nil {
		return fmt.Errorf("notification failed: %w", err)
	}

	fmt.Printf("Order #%d processed successfully\\n", order.ID)
	return nil
}

// Usage demonstrates DIP flexibility:
//
// Production with Stripe + Email:
// processor := NewOrderProcessorRefactored(
//     &StripeGatewayRefactored{APIKey: "sk_live_..."},
//     &InventoryManagerRefactored{Database: "prod_db"},
//     &EmailNotifierRefactored{SMTPHost: "smtp.gmail.com"},
// )
//
// Production with PayPal + SMS:
// processor := NewOrderProcessorRefactored(
//     &PayPalGateway{ClientID: "paypal_client"},
//     &InventoryManagerRefactored{Database: "prod_db"},
//     &SMSNotifier{Provider: "Twilio"},
// )
//
// Testing with mocks:
// processor := NewOrderProcessorRefactored(
//     &MockPaymentGateway{},
//     &MockInventoryService{},
//     &MockNotificationService{},
// )
//
// Same OrderProcessor code works with all configurations!`,
	hint1: `For concrete implementations, print messages showing which service is being used and what action is taken. Each should return nil on success. Include identifying info (APIKey, Database, SMTPHost, Provider) in the messages.`,
	hint2: `For NewOrderProcessorRefactored, return &OrderProcessorRefactored with all three dependencies assigned. For Process, call op.payment.Charge(order.Total), then op.inventory.Reserve(order.Items), then op.notifier.Send with a success message. Check and return errors from each step.`,
	testCode: `package principles

import "testing"

// Mocks for testing
type MockPayment struct {
	ChargeCalled bool
	Amount       float64
}

func (m *MockPayment) Charge(amount float64) error {
	m.ChargeCalled = true
	m.Amount = amount
	return nil
}

type MockInventory struct {
	ReserveCalled bool
	Items         []string
}

func (m *MockInventory) Reserve(items []string) error {
	m.ReserveCalled = true
	m.Items = items
	return nil
}

type MockNotifier struct {
	SendCalled bool
	Recipient  string
	Message    string
}

func (m *MockNotifier) Send(recipient, message string) error {
	m.SendCalled = true
	m.Recipient = recipient
	m.Message = message
	return nil
}

// Test1: NewOrderProcessorRefactored creates processor
func Test1(t *testing.T) {
	processor := NewOrderProcessorRefactored(&MockPayment{}, &MockInventory{}, &MockNotifier{})
	if processor == nil {
		t.Error("NewOrderProcessorRefactored returned nil")
	}
}

// Test2: Process calls payment gateway
func Test2(t *testing.T) {
	payment := &MockPayment{}
	processor := NewOrderProcessorRefactored(payment, &MockInventory{}, &MockNotifier{})
	order := &Order{ID: 1, Total: 100.0}
	processor.Process(order)
	if !payment.ChargeCalled || payment.Amount != 100.0 {
		t.Error("Payment not charged correctly")
	}
}

// Test3: Process calls inventory service
func Test3(t *testing.T) {
	inventory := &MockInventory{}
	processor := NewOrderProcessorRefactored(&MockPayment{}, inventory, &MockNotifier{})
	order := &Order{ID: 1, Items: []string{"item1", "item2"}}
	processor.Process(order)
	if !inventory.ReserveCalled {
		t.Error("Inventory not reserved")
	}
}

// Test4: Process calls notification service
func Test4(t *testing.T) {
	notifier := &MockNotifier{}
	processor := NewOrderProcessorRefactored(&MockPayment{}, &MockInventory{}, notifier)
	order := &Order{ID: 1, Customer: "test@test.com"}
	processor.Process(order)
	if !notifier.SendCalled {
		t.Error("Notification not sent")
	}
}

// Test5: StripeGateway implements PaymentGateway
func Test5(t *testing.T) {
	var gateway PaymentGateway = &StripeGatewayRefactored{APIKey: "test"}
	if err := gateway.Charge(50.0); err != nil {
		t.Errorf("Charge error: %v", err)
	}
}

// Test6: PayPalGateway implements PaymentGateway
func Test6(t *testing.T) {
	var gateway PaymentGateway = &PayPalGateway{ClientID: "test"}
	if err := gateway.Charge(75.0); err != nil {
		t.Errorf("Charge error: %v", err)
	}
}

// Test7: EmailNotifier implements NotificationService
func Test7(t *testing.T) {
	var notifier NotificationService = &EmailNotifierRefactored{SMTPHost: "test"}
	if err := notifier.Send("test@test.com", "Hello"); err != nil {
		t.Errorf("Send error: %v", err)
	}
}

// Test8: SMSNotifier implements NotificationService
func Test8(t *testing.T) {
	var notifier NotificationService = &SMSNotifier{Provider: "test"}
	if err := notifier.Send("+1234567890", "Hello"); err != nil {
		t.Errorf("Send error: %v", err)
	}
}

// Test9: Process returns nil on success
func Test9(t *testing.T) {
	processor := NewOrderProcessorRefactored(&MockPayment{}, &MockInventory{}, &MockNotifier{})
	order := &Order{ID: 1, Total: 50.0, Items: []string{"a"}, Customer: "c@c.com"}
	err := processor.Process(order)
	if err != nil {
		t.Errorf("Process should return nil on success, got: %v", err)
	}
}

// Test10: Different configurations work
func Test10(t *testing.T) {
	stripe := &StripeGatewayRefactored{APIKey: "sk_test"}
	paypal := &PayPalGateway{ClientID: "client_test"}
	inventory := &InventoryManagerRefactored{Database: "test_db"}
	email := &EmailNotifierRefactored{SMTPHost: "smtp.test.com"}
	sms := &SMSNotifier{Provider: "Twilio"}

	p1 := NewOrderProcessorRefactored(stripe, inventory, email)
	p2 := NewOrderProcessorRefactored(paypal, inventory, sms)

	_ = p1
	_ = p2
}
`,
	whyItMatters: `Advanced DIP shows how to build testable, flexible systems with multiple dependencies.

**Why Advanced DIP Matters:**

**1. Complex Testing Made Simple**

\`\`\`go
// WITHOUT DIP - impossible to test
type CheckoutService struct{}

func (cs *CheckoutService) Checkout(cart *Cart) error {
	// Creates dependencies internally
	payment := stripe.New("sk_live_...")      // real Stripe!
	shipping := fedex.New("credentials")      // real FedEx!
	email := sendgrid.New("api_key")          // real SendGrid!

	// Test must hit all external services
	// Slow, expensive, requires credentials
	// Fails if services are down
}

// WITH DIP - easy to test
type PaymentProvider interface { Charge(float64) error }
type ShippingProvider interface { Ship(*Address) error }
type EmailProvider interface { Send(string, string) error }

type CheckoutService struct {
	payment  PaymentProvider
	shipping ShippingProvider
	email    EmailProvider
}

// Test with mocks
func TestCheckout(t *testing.T) {
	mockPayment := &MockPayment{}
	mockShipping := &MockShipping{}
	mockEmail := &MockEmail{}

	service := &CheckoutService{
		payment:  mockPayment,
		shipping: mockShipping,
		email:    mockEmail,
	}

	cart := &Cart{Total: 100}
	err := service.Checkout(cart)

	if err != nil {
		t.Error("checkout failed")
	}
	if !mockPayment.ChargeCalled {
		t.Error("payment not called")
	}
	// Fast, reliable, no external dependencies!
}
\`\`\`

**2. Environment-Specific Configuration**

\`\`\`go
// Different implementations per environment
func NewProductionProcessor() *OrderProcessor {
	return NewOrderProcessor(
		&StripeGateway{APIKey: os.Getenv("STRIPE_KEY")},
		&PostgresInventory{DB: prodDB},
		&SendGridNotifier{APIKey: os.Getenv("SENDGRID_KEY")},
	)
}

func NewStagingProcessor() *OrderProcessor {
	return NewOrderProcessor(
		&StripeGateway{APIKey: os.Getenv("STRIPE_TEST_KEY")},
		&PostgresInventory{DB: stagingDB},
		&LogNotifier{},  // log instead of sending emails
	)
}

func NewTestProcessor() *OrderProcessor {
	return NewOrderProcessor(
		&MockPayment{},
		&InMemoryInventory{},
		&MockNotifier{},
	)
}
\`\`\`

**3. Real Production Example: Order System**

\`\`\`go
// Before DIP - nightmare to test and maintain
type OrderService struct {
	// Creates all dependencies internally
}

func (os *OrderService) ProcessOrder(order *Order) error {
	// Hardcoded Stripe
	stripe := stripe.New("sk_live_...")
	stripe.Charge(order.Total)

	// Hardcoded database
	db, _ := sql.Open("postgres", "hardcoded_connection")
	db.Exec("UPDATE inventory...")

	// Hardcoded email
	smtp.SendMail("smtp.gmail.com:587", auth, ...)

	// Hardcoded analytics
	analytics.Track("order_processed")

	return nil
}
// Problems:
// - Can't test without real Stripe, database, email, analytics
// - Can't switch payment providers
// - Can't use different databases per environment
// - Credentials hardcoded

// After DIP - flexible and testable
type PaymentGateway interface { Charge(float64) error }
type OrderRepository interface { Update(*Order) error }
type Notifier interface { Notify(string) error }
type Analytics interface { Track(string, map[string]interface{}) error }

type OrderService struct {
	payment   PaymentGateway
	repo      OrderRepository
	notifier  Notifier
	analytics Analytics
}

func NewOrderService(
	payment PaymentGateway,
	repo OrderRepository,
	notifier Notifier,
	analytics Analytics,
) *OrderService {
	return &OrderService{payment, repo, notifier, analytics}
}

func (os *OrderService) ProcessOrder(order *Order) error {
	if err := os.payment.Charge(order.Total); err != nil {
		return err
	}
	if err := os.repo.Update(order); err != nil {
		return err
	}
	os.notifier.Notify("Order processed")
	os.analytics.Track("order_processed", map[string]interface{}{
		"order_id": order.ID,
		"total":    order.Total,
	})
	return nil
}

// Benefits:
// ✓ Test with mocks (no external services)
// ✓ Switch Stripe → PayPal (inject different payment)
// ✓ Different DB per environment (inject different repo)
// ✓ Disable analytics in test (inject no-op analytics)
// ✓ All credentials injected (no hardcoding)
\`\`\``,
	order: 9,
	translations: {
		ru: {
			title: 'Принцип инверсии зависимостей - Продвинутый',
			description: `Примените DIP к сложной системе обработки заказов с множественными зависимостями.`,
			hint1: `Для конкретных реализаций выводите сообщения, показывающие какой сервис используется и какое действие выполняется. Каждый должен возвращать nil при успехе.`,
			hint2: `Для NewOrderProcessorRefactored верните &OrderProcessorRefactored со всеми тремя назначенными зависимостями. Для Process вызовите op.payment.Charge, затем op.inventory.Reserve, затем op.notifier.Send.`,
			whyItMatters: `Продвинутый DIP показывает как строить тестируемые, гибкие системы с множественными зависимостями.`,
			solutionCode: `package principles

import "fmt"

type Order struct {
	ID       int
	Items    []string
	Total    float64
	Customer string
}

type PaymentGateway interface {
	Charge(amount float64) error
}

type InventoryService interface {
	Reserve(items []string) error
}

type NotificationService interface {
	Send(recipient, message string) error
}

type StripeGatewayRefactored struct {
	APIKey string
}

func (sg *StripeGatewayRefactored) Charge(amount float64) error {
	fmt.Printf("Stripe: Списание $%.2f\\n", amount)
	return nil
}

type InventoryManagerRefactored struct {
	Database string
}

func (im *InventoryManagerRefactored) Reserve(items []string) error {
	fmt.Printf("Инвентарь: Резервирование %v\\n", items)
	return nil
}

type EmailNotifierRefactored struct {
	SMTPHost string
}

func (en *EmailNotifierRefactored) Send(recipient, message string) error {
	fmt.Printf("Email: Отправка %s: %s\\n", recipient, message)
	return nil
}

type OrderProcessorRefactored struct {
	payment   PaymentGateway
	inventory InventoryService
	notifier  NotificationService
}

func NewOrderProcessorRefactored(
	payment PaymentGateway,
	inventory InventoryService,
	notifier NotificationService,
) *OrderProcessorRefactored {
	return &OrderProcessorRefactored{payment, inventory, notifier}
}

func (op *OrderProcessorRefactored) Process(order *Order) error {
	if err := op.payment.Charge(order.Total); err != nil {
		return err
	}
	if err := op.inventory.Reserve(order.Items); err != nil {
		return err
	}
	message := fmt.Sprintf("Заказ #%d обработан", order.ID)
	return op.notifier.Send(order.Customer, message)
}`
		},
		uz: {
			title: 'Bog\'liqlik inversiyasi printsipi - Kengaytirilgan',
			description: `Ko'plab bog'liqliklarga ega murakkab buyurtmalarni qayta ishlash tizimiga DIP ni qo'llang.`,
			hint1: `Konkret amalga oshirishlar uchun qaysi xizmat ishlatilayotgani va qanday harakat amalga oshirilayotganini ko'rsatuvchi xabarlarni chiqaring. Har biri muvaffaqiyatda nil qaytarishi kerak.`,
			hint2: `NewOrderProcessorRefactored uchun barcha uchta bog'liqlik tayinlangan &OrderProcessorRefactored ni qaytaring. Process uchun op.payment.Charge, keyin op.inventory.Reserve, keyin op.notifier.Send ni chaqiring.`,
			whyItMatters: `Kengaytirilgan DIP ko'plab bog'liqliklarga ega test qilinadigan, moslashuvchan tizimlarni qanday qurishni ko'rsatadi.`,
			solutionCode: `package principles

import "fmt"

type Order struct {
	ID       int
	Items    []string
	Total    float64
	Customer string
}

type PaymentGateway interface {
	Charge(amount float64) error
}

type InventoryService interface {
	Reserve(items []string) error
}

type NotificationService interface {
	Send(recipient, message string) error
}

type StripeGatewayRefactored struct {
	APIKey string
}

func (sg *StripeGatewayRefactored) Charge(amount float64) error {
	fmt.Printf("Stripe: $%.2f to'lov olinmoqda\\n", amount)
	return nil
}

type InventoryManagerRefactored struct {
	Database string
}

func (im *InventoryManagerRefactored) Reserve(items []string) error {
	fmt.Printf("Inventar: %v zahiralanmoqda\\n", items)
	return nil
}

type EmailNotifierRefactored struct {
	SMTPHost string
}

func (en *EmailNotifierRefactored) Send(recipient, message string) error {
	fmt.Printf("Email: %s ga yuborilmoqda: %s\\n", recipient, message)
	return nil
}

type OrderProcessorRefactored struct {
	payment   PaymentGateway
	inventory InventoryService
	notifier  NotificationService
}

func NewOrderProcessorRefactored(
	payment PaymentGateway,
	inventory InventoryService,
	notifier NotificationService,
) *OrderProcessorRefactored {
	return &OrderProcessorRefactored{payment, inventory, notifier}
}

func (op *OrderProcessorRefactored) Process(order *Order) error {
	if err := op.payment.Charge(order.Total); err != nil {
		return err
	}
	if err := op.inventory.Reserve(order.Items); err != nil {
		return err
	}
	message := fmt.Sprintf("Buyurtma #%d qayta ishlandi", order.ID)
	return op.notifier.Send(order.Customer, message)
}`
		}
	}
};

export default task;
