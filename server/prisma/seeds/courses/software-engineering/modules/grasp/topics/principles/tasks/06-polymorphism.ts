import { Task } from '../../../../../../../types';

export const task: Task = {
	slug: 'se-grasp-polymorphism',
	title: 'Polymorphism',
	difficulty: 'medium',
	tags: ['go', 'software-engineering', 'grasp', 'polymorphism'],
	estimatedTime: '40m',
	isPremium: false,
	youtubeUrl: '',
	description: `Implement the Polymorphism principle - use polymorphic operations instead of explicit type checking.

**You will implement:**

1. **PaymentMethod interface** - Common interface for all payment types
2. **CreditCard struct** - Credit card payment implementation
3. **PayPal struct** - PayPal payment implementation
4. **ProcessPayment(amount float64, method PaymentMethod)** - Process using polymorphism

**Key Concepts:**
- **Polymorphism**: Same interface, different implementations
- **No Type Switching**: Avoid if/switch statements for types
- **Open/Closed Principle**: Open for extension, closed for modification

**Example Usage:**

\`\`\`go
// Different payment methods with same interface
creditCard := &CreditCard{CardNumber: "1234-5678"}
paypal := &PayPal{Email: "user@example.com"}

// Process payments polymorphically - no type checking needed!
ProcessPayment(100.0, creditCard)
// Output: Processing $100.00 via Credit Card: 1234-5678

ProcessPayment(50.0, paypal)
// Output: Processing $50.00 via PayPal: user@example.com

// Easy to add new payment methods without changing ProcessPayment
bitcoin := &Bitcoin{WalletAddress: "1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa"}
ProcessPayment(200.0, bitcoin) // Just works!
\`\`\`

**Why Polymorphism?**
- **Eliminates Type Checking**: No if/switch statements
- **Easy to Extend**: Add new types without changing existing code
- **Clean Code**: Same interface for all implementations

**Anti-pattern (Don't do this):**
\`\`\`go
// BAD: Type checking instead of polymorphism
func ProcessPayment(amount float64, paymentType string, details map[string]string) {
    switch paymentType {
    case "credit_card":
        // Credit card logic
        fmt.Printf("Processing via credit card: %s\n", details["card_number"])
    case "paypal":
        // PayPal logic
        fmt.Printf("Processing via PayPal: %s\n", details["email"])
    case "bitcoin":
        // Bitcoin logic
        fmt.Printf("Processing via Bitcoin: %s\n", details["wallet"])
    // Must modify this function for every new payment type!
    }
}
\`\`\`

**Constraints:**
- All payment methods must implement PaymentMethod interface
- ProcessPayment must work with interface, not concrete types
- No type switching or type assertions in ProcessPayment`,
	initialCode: `package principles

import "fmt"

type PaymentMethod interface {
}

type CreditCard struct {
	CardNumber string
}

func (c *CreditCard) Process(amount float64) string {
}

type PayPal struct {
	Email string
}

func (p *PayPal) Process(amount float64) string {
}

func ProcessPayment(amount float64, method PaymentMethod) {
}`,
	solutionCode: `package principles

import "fmt"

// PaymentMethod interface allows polymorphic payment processing
type PaymentMethod interface {
	Process(amount float64) string
}

// CreditCard implements PaymentMethod
type CreditCard struct {
	CardNumber string
}

func (c *CreditCard) Process(amount float64) string {
	// Each type implements Process differently
	return fmt.Sprintf("Processing $%.2f via Credit Card: %s", amount, c.CardNumber)
}

// PayPal implements PaymentMethod
type PayPal struct {
	Email string
}

func (p *PayPal) Process(amount float64) string {
	// Different implementation, same interface
	return fmt.Sprintf("Processing $%.2f via PayPal: %s", amount, p.Email)
}

// ProcessPayment uses polymorphism - works with any PaymentMethod
func ProcessPayment(amount float64, method PaymentMethod) {
	// No type checking needed!
	// The correct Process method is called based on the actual type
	result := method.Process(amount)	// polymorphic call
	fmt.Println(result)
}`,
	hint1: `Define PaymentMethod interface with Process(amount float64) string. Each implementation should return a formatted string with payment details.`,
	hint2: `ProcessPayment should call method.Process(amount) and print the result. No type checking - just use the interface method!`,
	whyItMatters: `Polymorphism eliminates conditional logic and makes code extensible without modification.

**Why Polymorphism Matters:**

**1. Eliminates Type Checking Code**
Polymorphism replaces ugly type checking with clean interface calls:

\`\`\`go
// WITHOUT POLYMORPHISM - BAD!
type Shape struct {
    Type   string  // "circle", "square", "triangle"
    Radius float64
    Side   float64
    Base   float64
    Height float64
}

func CalculateArea(shape Shape) float64 {
    switch shape.Type {
    case "circle":
        return math.Pi * shape.Radius * shape.Radius
    case "square":
        return shape.Side * shape.Side
    case "triangle":
        return 0.5 * shape.Base * shape.Height
    default:
        return 0
    }
}
// Must modify this function for every new shape!

// WITH POLYMORPHISM - GOOD!
type Shape interface {
    Area() float64
}

type Circle struct {
    Radius float64
}

func (c Circle) Area() float64 {
    return math.Pi * c.Radius * c.Radius
}

type Square struct {
    Side float64
}

func (s Square) Area() float64 {
    return s.Side * s.Side
}

type Triangle struct {
    Base, Height float64
}

func (t Triangle) Area() float64 {
    return 0.5 * t.Base * t.Height
}

// Calculate area for ANY shape - no type checking!
func PrintArea(shape Shape) {
    fmt.Printf("Area: %.2f\n", shape.Area())
}

// Add new shapes without changing existing code
type Hexagon struct {
    Side float64
}

func (h Hexagon) Area() float64 {
    return (3 * math.Sqrt(3) / 2) * h.Side * h.Side
}
// PrintArea works with Hexagon automatically!
\`\`\`

**2. Open/Closed Principle**
Polymorphism makes code open for extension, closed for modification:

\`\`\`go
// Notification system with polymorphism
type Notifier interface {
    Send(message string) error
}

type EmailNotifier struct {
    SMTPServer string
}

func (e *EmailNotifier) Send(message string) error {
    // Send via email
    return smtp.SendMail(e.SMTPServer, nil, "from@example.com", []string{"to@example.com"}, []byte(message))
}

type SMSNotifier struct {
    TwilioAPIKey string
}

func (s *SMSNotifier) Send(message string) error {
    // Send via SMS
    return twilio.SendSMS(s.TwilioAPIKey, "+1234567890", message)
}

// NotificationService works with ANY notifier
type NotificationService struct {
    notifiers []Notifier
}

func (n *NotificationService) Notify(message string) error {
    for _, notifier := range n.notifiers {
        if err := notifier.Send(message); err != nil {
            return err
        }
    }
    return nil
}

// Add new notification types without changing NotificationService
type SlackNotifier struct {
    WebhookURL string
}

func (s *SlackNotifier) Send(message string) error {
    // Send via Slack
    return http.Post(s.WebhookURL, "application/json", bytes.NewBuffer([]byte(message)))
}

// Just add to notifiers list - no code changes needed!
func main() {
    service := &NotificationService{
        notifiers: []Notifier{
            &EmailNotifier{SMTPServer: "smtp.gmail.com"},
            &SMSNotifier{TwilioAPIKey: "key"},
            &SlackNotifier{WebhookURL: "https://hooks.slack.com/..."},
        },
    }
    service.Notify("Hello World")
}
\`\`\`

**3. Real-World Example: Storage System**
\`\`\`go
// Storage interface for polymorphic file storage
type Storage interface {
    Save(filename string, data []byte) error
    Load(filename string) ([]byte, error)
    Delete(filename string) error
}

// Local file system storage
type LocalStorage struct {
    basePath string
}

func (l *LocalStorage) Save(filename string, data []byte) error {
    path := filepath.Join(l.basePath, filename)
    return ioutil.WriteFile(path, data, 0644)
}

func (l *LocalStorage) Load(filename string) ([]byte, error) {
    path := filepath.Join(l.basePath, filename)
    return ioutil.ReadFile(path)
}

func (l *LocalStorage) Delete(filename string) error {
    path := filepath.Join(l.basePath, filename)
    return os.Remove(path)
}

// S3 cloud storage
type S3Storage struct {
    bucket string
    client *s3.Client
}

func (s *S3Storage) Save(filename string, data []byte) error {
    _, err := s.client.PutObject(&s3.PutObjectInput{
        Bucket: &s.bucket,
        Key:    &filename,
        Body:   bytes.NewReader(data),
    })
    return err
}

func (s *S3Storage) Load(filename string) ([]byte, error) {
    result, err := s.client.GetObject(&s3.GetObjectInput{
        Bucket: &s.bucket,
        Key:    &filename,
    })
    if err != nil {
        return nil, err
    }
    return ioutil.ReadAll(result.Body)
}

func (s *S3Storage) Delete(filename string) error {
    _, err := s.client.DeleteObject(&s3.DeleteObjectInput{
        Bucket: &s.bucket,
        Key:    &filename,
    })
    return err
}

// FileManager works with ANY storage implementation
type FileManager struct {
    storage Storage
}

func (f *FileManager) UploadFile(filename string, data []byte) error {
    // Polymorphic call - works with LocalStorage, S3Storage, etc.
    return f.storage.Save(filename, data)
}

func (f *FileManager) DownloadFile(filename string) ([]byte, error) {
    return f.storage.Load(filename)
}

// Easy to switch storage backends
func main() {
    // Use local storage
    local := &FileManager{storage: &LocalStorage{basePath: "/tmp"}}
    local.UploadFile("test.txt", []byte("hello"))

    // Switch to S3 - no changes to FileManager!
    s3 := &FileManager{storage: &S3Storage{bucket: "my-bucket", client: s3Client}}
    s3.UploadFile("test.txt", []byte("hello"))
}
\`\`\`

**4. Testing with Polymorphism**
\`\`\`go
// Mock implementation for testing
type MockStorage struct {
    Files map[string][]byte
}

func (m *MockStorage) Save(filename string, data []byte) error {
    if m.Files == nil {
        m.Files = make(map[string][]byte)
    }
    m.Files[filename] = data
    return nil
}

func (m *MockStorage) Load(filename string) ([]byte, error) {
    data, ok := m.Files[filename]
    if !ok {
        return nil, errors.New("file not found")
    }
    return data, nil
}

func (m *MockStorage) Delete(filename string) error {
    delete(m.Files, filename)
    return nil
}

// Test FileManager with mock - no real storage needed!
func TestFileManager_UploadFile(t *testing.T) {
    mock := &MockStorage{}
    manager := &FileManager{storage: mock}

    data := []byte("test data")
    err := manager.UploadFile("test.txt", data)
    if err != nil {
        t.Fatalf("upload failed: %v", err)
    }

    // Verify using mock
    saved, err := mock.Load("test.txt")
    if err != nil {
        t.Fatalf("load failed: %v", err)
    }
    if string(saved) != string(data) {
        t.Errorf("expected %s, got %s", data, saved)
    }
}
\`\`\`

**Common Mistakes:**
- Using type switches instead of polymorphism
- Creating interfaces with too many methods
- Not using interfaces when multiple implementations exist
- Type assertions everywhere (defeats purpose of polymorphism)

**Rule of Thumb:**
If you find yourself using type switch or type assertions, consider using polymorphism instead. Let the type system do the work!`,
	order: 5,
	testCode: `package principles

import (
	"strings"
	"testing"
)

// Test1: CreditCard implements PaymentMethod
func Test1(t *testing.T) {
	cc := &CreditCard{CardNumber: "1234-5678"}
	var _ PaymentMethod = cc // compile-time check
	if cc == nil {
		t.Error("CreditCard should not be nil")
	}
}

// Test2: PayPal implements PaymentMethod
func Test2(t *testing.T) {
	pp := &PayPal{Email: "user@test.com"}
	var _ PaymentMethod = pp // compile-time check
	if pp == nil {
		t.Error("PayPal should not be nil")
	}
}

// Test3: CreditCard.Process returns correct message
func Test3(t *testing.T) {
	cc := &CreditCard{CardNumber: "1234-5678"}
	result := cc.Process(100.0)
	if !strings.Contains(result, "Credit Card") {
		t.Error("Result should mention Credit Card")
	}
	if !strings.Contains(result, "1234-5678") {
		t.Error("Result should contain card number")
	}
}

// Test4: PayPal.Process returns correct message
func Test4(t *testing.T) {
	pp := &PayPal{Email: "user@test.com"}
	result := pp.Process(50.0)
	if !strings.Contains(result, "PayPal") {
		t.Error("Result should mention PayPal")
	}
	if !strings.Contains(result, "user@test.com") {
		t.Error("Result should contain email")
	}
}

// Test5: ProcessPayment with CreditCard runs without panic
func Test5(t *testing.T) {
	cc := &CreditCard{CardNumber: "1111-2222"}
	ProcessPayment(200.0, cc) // Should not panic
}

// Test6: ProcessPayment with PayPal runs without panic
func Test6(t *testing.T) {
	pp := &PayPal{Email: "pay@test.com"}
	ProcessPayment(75.0, pp) // Should not panic
}

// Test7: CreditCard.Process includes amount
func Test7(t *testing.T) {
	cc := &CreditCard{CardNumber: "9999-8888"}
	result := cc.Process(123.45)
	if !strings.Contains(result, "123.45") {
		t.Error("Result should contain formatted amount")
	}
}

// Test8: PayPal.Process includes amount
func Test8(t *testing.T) {
	pp := &PayPal{Email: "amount@test.com"}
	result := pp.Process(99.99)
	if !strings.Contains(result, "99.99") {
		t.Error("Result should contain formatted amount")
	}
}

// Test9: CreditCard struct fields
func Test9(t *testing.T) {
	cc := CreditCard{CardNumber: "4444-3333"}
	if cc.CardNumber != "4444-3333" {
		t.Error("CardNumber field not set correctly")
	}
}

// Test10: PayPal struct fields
func Test10(t *testing.T) {
	pp := PayPal{Email: "field@test.com"}
	if pp.Email != "field@test.com" {
		t.Error("Email field not set correctly")
	}
}
`,
	translations: {
		ru: {
			title: 'Полиморфизм',
			description: `Реализуйте принцип Полиморфизма — используйте полиморфные операции вместо явной проверки типов.

**Вы реализуете:**

1. **PaymentMethod interface** — Общий интерфейс для всех типов платежей
2. **CreditCard struct** — Реализация платежа кредитной картой
3. **PayPal struct** — Реализация платежа через PayPal
4. **ProcessPayment(amount float64, method PaymentMethod)** — Обработка с использованием полиморфизма

**Ключевые концепции:**
- **Полиморфизм**: Один интерфейс, разные реализации
- **Без проверки типов**: Избегайте if/switch для типов
- **Принцип открытости/закрытости**: Открыт для расширения, закрыт для модификации

**Зачем нужен Полиморфизм?**
- **Устраняет проверку типов**: Нет if/switch операторов
- **Легко расширять**: Добавляйте новые типы без изменения существующего кода
- **Чистый код**: Один интерфейс для всех реализаций

**Ограничения:**
- Все методы платежа должны реализовывать интерфейс PaymentMethod
- ProcessPayment должен работать с интерфейсом, а не с конкретными типами
- Никаких проверок типов или type assertions в ProcessPayment`,
			hint1: `Определите интерфейс PaymentMethod с Process(amount float64) string. Каждая реализация должна возвращать отформатированную строку с деталями платежа.`,
			hint2: `ProcessPayment должен вызвать method.Process(amount) и напечатать результат. Никаких проверок типов - просто используйте метод интерфейса!`,
			whyItMatters: `Полиморфизм устраняет условную логику и делает код расширяемым без модификации.

**Почему Полиморфизм важен:**

**1. Устраняет код проверки типов**
Полиморфизм заменяет некрасивую проверку типов на чистые вызовы интерфейса.

**Распространённые ошибки:**
- Использование type switch вместо полиморфизма
- Создание интерфейсов со слишком многими методами
- Неиспользование интерфейсов когда есть множественные реализации
- Type assertions повсюду (нарушает цель полиморфизма)`,
			solutionCode: `package principles

import "fmt"

// PaymentMethod интерфейс позволяет полиморфную обработку платежей
type PaymentMethod interface {
	Process(amount float64) string
}

// CreditCard реализует PaymentMethod
type CreditCard struct {
	CardNumber string
}

func (c *CreditCard) Process(amount float64) string {
	// Каждый тип реализует Process по-своему
	return fmt.Sprintf("Processing $%.2f via Credit Card: %s", amount, c.CardNumber)
}

// PayPal реализует PaymentMethod
type PayPal struct {
	Email string
}

func (p *PayPal) Process(amount float64) string {
	// Другая реализация, тот же интерфейс
	return fmt.Sprintf("Processing $%.2f via PayPal: %s", amount, p.Email)
}

// ProcessPayment использует полиморфизм - работает с любым PaymentMethod
func ProcessPayment(amount float64, method PaymentMethod) {
	// Проверка типов не нужна!
	// Правильный метод Process вызывается на основе реального типа
	result := method.Process(amount)	// полиморфный вызов
	fmt.Println(result)
}`
		},
		uz: {
			title: 'Polymorphism (Polimorfizm)',
			description: `Polymorphism prinsipini amalga oshiring — aniq tip tekshiruvi o'rniga polimorf operatsiyalardan foydalaning.

**Siz amalga oshirasiz:**

1. **PaymentMethod interface** — Barcha to'lov turlari uchun umumiy interfeys
2. **CreditCard struct** — Kredit karta to'lovi implementatsiyasi
3. **PayPal struct** — PayPal to'lovi implementatsiyasi
4. **ProcessPayment(amount float64, method PaymentMethod)** — Polimorfizm yordamida qayta ishlash

**Asosiy tushunchalar:**
- **Polimorfizm**: Bir xil interfeys, turli implementatsiyalar
- **Tip tekshiruvsiz**: if/switch operatorlaridan qoching
- **Ochiq/Yopiq printsipi**: Kengaytirish uchun ochiq, o'zgartirish uchun yopiq

**Nima uchun Polimorfizm?**
- **Tip tekshiruvini yo'q qiladi**: if/switch operatorlari yo'q
- **Kengaytirish oson**: Mavjud kodni o'zgartirmasdan yangi tiplarni qo'shing
- **Toza kod**: Barcha implementatsiyalar uchun bir xil interfeys

**Cheklovlar:**
- Barcha to'lov metodlari PaymentMethod interfeysini amalga oshirishi kerak
- ProcessPayment konkret tiplar bilan emas, interfeys bilan ishlashi kerak
- ProcessPayment da tip tekshiruv yoki tip assertionlar yo'q`,
			hint1: `PaymentMethod interfeysini Process(amount float64) string bilan aniqlang. Har bir implementatsiya to'lov tafsilotlari bilan formatlangan stringni qaytarishi kerak.`,
			hint2: `ProcessPayment method.Process(amount) ni chaqirishi va natijani chop etishi kerak. Tip tekshiruv yo'q - faqat interfeys metodini ishlating!`,
			whyItMatters: `Polimorfizm shartli mantiqni yo'q qiladi va kodni o'zgartirmasdan kengaytirilishi mumkin qiladi.

**Polimorfizm nima uchun muhim:**

**1. Tip tekshiruv kodini yo'q qiladi**
Polimorfizm yomon tip tekshiruvni toza interfeys chaqiruvlari bilan almashtiradi.

**Umumiy xatolar:**
- Polimorfizm o'rniga type switch dan foydalanish
- Juda ko'p metodlar bilan interfeyslar yaratish
- Ko'plab implementatsiyalar mavjud bo'lganda interfeyslardan foydalanmaslik
- Hamma joyda type assertionlar (polimorfizm maqsadini buzadi)`,
			solutionCode: `package principles

import "fmt"

// PaymentMethod interfeysi polimorf to'lov qayta ishlashga imkon beradi
type PaymentMethod interface {
	Process(amount float64) string
}

// CreditCard PaymentMethod ni amalga oshiradi
type CreditCard struct {
	CardNumber string
}

func (c *CreditCard) Process(amount float64) string {
	// Har bir tip Process ni boshqacha amalga oshiradi
	return fmt.Sprintf("Processing $%.2f via Credit Card: %s", amount, c.CardNumber)
}

// PayPal PaymentMethod ni amalga oshiradi
type PayPal struct {
	Email string
}

func (p *PayPal) Process(amount float64) string {
	// Boshqa implementatsiya, bir xil interfeys
	return fmt.Sprintf("Processing $%.2f via PayPal: %s", amount, p.Email)
}

// ProcessPayment polimorfizmdan foydalanadi - har qanday PaymentMethod bilan ishlaydi
func ProcessPayment(amount float64, method PaymentMethod) {
	// Tip tekshiruv kerak emas!
	// To'g'ri Process metodi haqiqiy tipga asoslanib chaqiriladi
	result := method.Process(amount)	// polimorf chaqiruv
	fmt.Println(result)
}`
		}
	}
};

export default task;
