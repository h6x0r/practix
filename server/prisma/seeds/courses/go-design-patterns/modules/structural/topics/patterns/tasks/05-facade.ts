import { Task } from '../../../../../../../types';

export const task: Task = {
	slug: 'go-dp-facade',
	title: 'Facade Pattern',
	difficulty: 'easy',
	tags: ['go', 'design-patterns', 'structural', 'facade'],
	estimatedTime: '25m',
	isPremium: false,
	youtubeUrl: '',
	description: `Implement the Facade pattern in Go - provide a unified interface to a set of interfaces in a subsystem.

The Facade pattern provides a simplified interface to a complex subsystem. It doesn't encapsulate the subsystem but rather provides a convenient high-level interface while still allowing direct access to subsystem components when needed.

**You will implement:**

1. **CPU struct** - Freeze, Execute, Jump methods for processor operations
2. **Memory struct** - Load method for RAM operations
3. **HardDrive struct** - Read method for storage operations
4. **ComputerFacade** - Simple Start() method that orchestrates the boot sequence

**Example Usage:**

\`\`\`go
// Create facade - client doesn't need to know about CPU, Memory, HardDrive
computer := NewComputerFacade()	// facade encapsulates subsystem

// Simple one-call interface hides complex boot sequence
result := computer.Start()	// orchestrates all subsystem operations
// result contains:
// ["CPU: Freezing processor", "HardDrive: Reading sector 0 (1024 bytes)",
//  "Memory: Loading data at 0", "CPU: Jumping to address 0", "CPU: Executing instructions"]

// Direct subsystem access still possible if needed
cpu := &CPU{}
cpu.Freeze()	// can still use subsystem directly
\`\`\``,
	initialCode: `package patterns

import "fmt"

type CPU struct{}

func (c *CPU) Freeze() string {
}

func (c *CPU) Jump(addr int64) string {
}

func (c *CPU) Execute() string {
}

type Memory struct{}

func (m *Memory) Load(position int64, data []byte) string {
}

type HardDrive struct{}

func (h *HardDrive) Read(lba int64, size int) string {
}

type ComputerFacade struct {
	cpu       *CPU
}

func NewComputerFacade() *ComputerFacade {
	}
}

func (c *ComputerFacade) Start() []string {
}`,
	solutionCode: `package patterns

import "fmt"

// CPU represents the processor
type CPU struct{}	// subsystem component

// Freeze stops the processor
func (c *CPU) Freeze() string {	// subsystem operation
	return "CPU: Freezing processor"	// return operation result
}

// Jump moves to memory address
func (c *CPU) Jump(addr int64) string {	// subsystem operation with parameter
	return fmt.Sprintf("CPU: Jumping to address %d", addr)	// format with address
}

// Execute runs instructions
func (c *CPU) Execute() string {	// subsystem operation
	return "CPU: Executing instructions"	// return operation result
}

// Memory represents RAM
type Memory struct{}	// subsystem component

// Load puts data in memory
func (m *Memory) Load(position int64, data []byte) string {	// subsystem operation
	return fmt.Sprintf("Memory: Loading data at %d", position)	// format with position
}

// HardDrive represents storage
type HardDrive struct{}	// subsystem component

// Read gets data from disk
func (h *HardDrive) Read(lba int64, size int) string {	// subsystem operation
	return fmt.Sprintf("HardDrive: Reading sector %d (%d bytes)", lba, size)	// format with sector and size
}

// ComputerFacade provides simple interface to complex subsystem
type ComputerFacade struct {	// facade - unified interface
	cpu       *CPU	// subsystem reference
	memory    *Memory	// subsystem reference
	hardDrive *HardDrive	// subsystem reference
}

// NewComputerFacade creates facade with all components
func NewComputerFacade() *ComputerFacade {	// factory function for facade
	return &ComputerFacade{	// initialize all subsystem components
		cpu:       &CPU{},	// create CPU
		memory:    &Memory{},	// create Memory
		hardDrive: &HardDrive{},	// create HardDrive
	}
}

// Start performs boot sequence
func (c *ComputerFacade) Start() []string {	// facade method - simplified interface
	results := make([]string, 0, 5)	// pre-allocate for expected results
	results = append(results, c.cpu.Freeze())	// step 1: freeze processor
	results = append(results, c.hardDrive.Read(0, 1024))	// step 2: read boot sector
	results = append(results, c.memory.Load(0, nil))	// step 3: load into RAM
	results = append(results, c.cpu.Jump(0))	// step 4: jump to start address
	results = append(results, c.cpu.Execute())	// step 5: begin execution
	return results	// return all operation results
}`,
	hint1: `Each subsystem method is straightforward - just return a formatted string describing the operation:
- Freeze(): return "CPU: Freezing processor"
- Jump(addr): use fmt.Sprintf("CPU: Jumping to address %d", addr)
- Execute(): return "CPU: Executing instructions"
- Load(position, data): use fmt.Sprintf("Memory: Loading data at %d", position)
- Read(lba, size): use fmt.Sprintf("HardDrive: Reading sector %d (%d bytes)", lba, size)`,
	hint2: `Start() orchestrates the boot sequence by calling subsystem methods in order:
\`\`\`go
results := make([]string, 0, 5)  // pre-allocate slice
results = append(results, c.cpu.Freeze())
results = append(results, c.hardDrive.Read(0, 1024))
results = append(results, c.memory.Load(0, nil))
results = append(results, c.cpu.Jump(0))
results = append(results, c.cpu.Execute())
return results
\`\`\`

The facade doesn't add logic - it simply coordinates the correct sequence of subsystem calls.`,
	whyItMatters: `## Why Facade Exists

The Facade pattern solves the problem of complex subsystem APIs being difficult to use correctly. Without it, clients must understand how to coordinate multiple subsystem components, leading to complex client code and tight coupling.

**Problem - Without Facade:**
\`\`\`go
// Client must know correct boot sequence
func bootComputer() {
    cpu := &CPU{}
    memory := &Memory{}
    hd := &HardDrive{}

    cpu.Freeze()  // must call first
    data := hd.Read(0, 1024)  // read boot sector
    memory.Load(0, data)  // load to correct address
    cpu.Jump(0)  // jump to boot address
    cpu.Execute()  // start execution
    // Wrong order = system crash
}
\`\`\`

**Solution - With Facade:**
\`\`\`go
// Client just calls Start()
func bootComputer() {
    computer := NewComputerFacade()
    computer.Start()  // facade handles correct sequence
}
\`\`\`

## Real-World Go Examples

**1. HTTP Server Facade:**
\`\`\`go
// Complex subsystems
type Router struct { /* routing logic */ }
type Middleware struct { /* auth, logging */ }
type Handler struct { /* request handling */ }
type Server struct { /* HTTP server */ }

// Facade simplifies setup
type WebAppFacade struct {
    router     *Router
    middleware *Middleware
    server     *Server
}

func NewWebApp(config Config) *WebAppFacade {
    return &WebAppFacade{
        router:     NewRouter(),
        middleware: NewMiddleware(config.Auth),
        server:     NewServer(config.Port),
    }
}

func (w *WebAppFacade) Start() error {
    w.router.Setup()
    w.middleware.Apply(w.router)
    return w.server.Listen(w.router)
}

// Client code is simple
app := NewWebApp(config)
app.Start()
\`\`\`

**2. Database Migration Facade:**
\`\`\`go
type SchemaReader struct { /* reads schema */ }
type DiffCalculator struct { /* compares schemas */ }
type MigrationGenerator struct { /* creates SQL */ }
type MigrationRunner struct { /* executes migrations */ }

type MigrationFacade struct {
    reader    *SchemaReader
    differ    *DiffCalculator
    generator *MigrationGenerator
    runner    *MigrationRunner
}

func (f *MigrationFacade) Migrate(targetSchema Schema) error {
    current := f.reader.ReadCurrentSchema()
    diff := f.differ.Calculate(current, targetSchema)
    migration := f.generator.Generate(diff)
    return f.runner.Execute(migration)
}

// Client just calls
facade.Migrate(newSchema)
\`\`\`

## Production Pattern: E-commerce Checkout

\`\`\`go
// Complex subsystems
type InventoryService struct{}
func (i *InventoryService) CheckStock(items []Item) (bool, error)
func (i *InventoryService) Reserve(items []Item) (string, error)
func (i *InventoryService) Release(reservationID string) error

type PaymentService struct{}
func (p *PaymentService) Validate(card CardInfo) error
func (p *PaymentService) Charge(card CardInfo, amount float64) (string, error)
func (p *PaymentService) Refund(transactionID string) error

type ShippingService struct{}
func (s *ShippingService) CalculateRates(addr Address, items []Item) []Rate
func (s *ShippingService) CreateLabel(order Order) (string, error)

type NotificationService struct{}
func (n *NotificationService) SendConfirmation(email string, order Order) error

// Checkout Facade - orchestrates complex flow
type CheckoutFacade struct {
    inventory    *InventoryService
    payment      *PaymentService
    shipping     *ShippingService
    notification *NotificationService
}

func (f *CheckoutFacade) ProcessOrder(cart Cart, card CardInfo) (*Order, error) {
    // Step 1: Check inventory
    available, err := f.inventory.CheckStock(cart.Items)
    if err != nil || !available {
        return nil, ErrOutOfStock
    }

    // Step 2: Reserve items
    reservationID, err := f.inventory.Reserve(cart.Items)
    if err != nil {
        return nil, err
    }

    // Step 3: Process payment
    if err := f.payment.Validate(card); err != nil {
        f.inventory.Release(reservationID)
        return nil, err
    }

    transactionID, err := f.payment.Charge(card, cart.Total())
    if err != nil {
        f.inventory.Release(reservationID)
        return nil, err
    }

    // Step 4: Create shipping label
    order := &Order{Items: cart.Items, TransactionID: transactionID}
    trackingNumber, err := f.shipping.CreateLabel(*order)
    if err != nil {
        f.payment.Refund(transactionID)
        f.inventory.Release(reservationID)
        return nil, err
    }
    order.TrackingNumber = trackingNumber

    // Step 5: Send confirmation
    f.notification.SendConfirmation(cart.CustomerEmail, *order)

    return order, nil
}

// Client code is simple
order, err := checkout.ProcessOrder(cart, cardInfo)
\`\`\`

## Common Mistakes

**1. Putting Business Logic in Facade:**
\`\`\`go
// Bad - facade calculates prices
func (f *OrderFacade) CreateOrder(items []Item) *Order {
    total := 0.0
    for _, item := range items {
        total += item.Price * float64(item.Quantity)  // business logic!
        if item.Quantity > 10 {
            total *= 0.9  // discount logic in facade!
        }
    }
    return f.orderService.Create(items, total)
}

// Good - facade just coordinates
func (f *OrderFacade) CreateOrder(items []Item) *Order {
    total := f.pricingService.Calculate(items)  // delegate to service
    return f.orderService.Create(items, total)
}
\`\`\`

**2. Making Facade the Only Access Point:**
\`\`\`go
// Bad - hiding subsystems completely
type BadFacade struct {
    inventory *inventoryService  // private, no direct access
}

// Good - facade is convenience, not restriction
type GoodFacade struct {
    Inventory *InventoryService  // public, direct access if needed
    Payment   *PaymentService
}

// Client can use facade
facade.ProcessOrder(cart)
// Or access subsystems directly for advanced use
facade.Inventory.GetDetailedStock()
\`\`\`

**3. Too Many Methods in Facade:**
\`\`\`go
// Bad - facade becomes a god object
type BadFacade struct { /* ... */ }
func (f *BadFacade) CreateUser() {}
func (f *BadFacade) UpdateUser() {}
func (f *BadFacade) DeleteUser() {}
func (f *BadFacade) CreateOrder() {}
func (f *BadFacade) UpdateOrder() {}
// ... 50 more methods

// Good - focused facades
type UserFacade struct { /* ... */ }
func (f *UserFacade) Register() {}
func (f *UserFacade) UpdateProfile() {}

type OrderFacade struct { /* ... */ }
func (f *OrderFacade) Checkout() {}
func (f *OrderFacade) Cancel() {}
\`\`\`

**Key Principles:**
- Facade provides convenience, not control - subsystems remain accessible
- Facade coordinates but doesn't contain business logic
- Create focused facades for different use cases rather than one mega-facade
- Facade simplifies the common case while allowing flexibility for advanced usage`,
	order: 4,
	testCode: `package patterns

import (
	"strings"
	"testing"
)

// Test1: CPU.Freeze returns correct message
func Test1(t *testing.T) {
	cpu := &CPU{}
	if cpu.Freeze() != "CPU: Freezing processor" {
		t.Error("CPU.Freeze should return correct message")
	}
}

// Test2: CPU.Execute returns correct message
func Test2(t *testing.T) {
	cpu := &CPU{}
	if cpu.Execute() != "CPU: Executing instructions" {
		t.Error("CPU.Execute should return correct message")
	}
}

// Test3: CPU.Jump returns formatted message
func Test3(t *testing.T) {
	cpu := &CPU{}
	result := cpu.Jump(100)
	if result != "CPU: Jumping to address 100" {
		t.Errorf("Expected 'CPU: Jumping to address 100', got '%s'", result)
	}
}

// Test4: Memory.Load returns correct message
func Test4(t *testing.T) {
	mem := &Memory{}
	result := mem.Load(256, nil)
	if result != "Memory: Loading data at 256" {
		t.Errorf("Expected 'Memory: Loading data at 256', got '%s'", result)
	}
}

// Test5: HardDrive.Read returns correct message
func Test5(t *testing.T) {
	hd := &HardDrive{}
	result := hd.Read(0, 1024)
	if result != "HardDrive: Reading sector 0 (1024 bytes)" {
		t.Errorf("Unexpected result: %s", result)
	}
}

// Test6: NewComputerFacade returns non-nil
func Test6(t *testing.T) {
	facade := NewComputerFacade()
	if facade == nil {
		t.Error("NewComputerFacade should return non-nil")
	}
}

// Test7: Start returns 5 results
func Test7(t *testing.T) {
	facade := NewComputerFacade()
	results := facade.Start()
	if len(results) != 5 {
		t.Errorf("Start should return 5 results, got %d", len(results))
	}
}

// Test8: Start first step is Freeze
func Test8(t *testing.T) {
	facade := NewComputerFacade()
	results := facade.Start()
	if !strings.Contains(results[0], "Freezing") {
		t.Error("First step should be Freeze")
	}
}

// Test9: Start last step is Execute
func Test9(t *testing.T) {
	facade := NewComputerFacade()
	results := facade.Start()
	if !strings.Contains(results[4], "Executing") {
		t.Error("Last step should be Execute")
	}
}

// Test10: Start includes HardDrive read
func Test10(t *testing.T) {
	facade := NewComputerFacade()
	results := facade.Start()
	found := false
	for _, r := range results {
		if strings.Contains(r, "HardDrive") {
			found = true
			break
		}
	}
	if !found {
		t.Error("Start should include HardDrive read")
	}
}
`,
	translations: {
		ru: {
			title: 'Паттерн Facade (Фасад)',
			description: `Реализуйте паттерн Facade на Go — предоставьте унифицированный интерфейс к набору интерфейсов подсистемы.

Паттерн Facade предоставляет упрощённый интерфейс к сложной подсистеме. Он не инкапсулирует подсистему, а скорее предоставляет удобный высокоуровневый интерфейс, при этом позволяя прямой доступ к компонентам подсистемы при необходимости.

**Вы реализуете:**

1. **Структура CPU** - Методы Freeze, Execute, Jump для операций процессора
2. **Структура Memory** - Метод Load для операций с ОЗУ
3. **Структура HardDrive** - Метод Read для операций с накопителем
4. **ComputerFacade** - Простой метод Start(), оркестрирующий последовательность загрузки

**Пример использования:**

\`\`\`go
// Создание фасада — клиенту не нужно знать о CPU, Memory, HardDrive
computer := NewComputerFacade()	// фасад инкапсулирует подсистему

// Простой однометодный интерфейс скрывает сложную последовательность загрузки
result := computer.Start()	// оркестрирует все операции подсистемы
// result содержит:
// ["CPU: Freezing processor", "HardDrive: Reading sector 0 (1024 bytes)",
//  "Memory: Loading data at 0", "CPU: Jumping to address 0", "CPU: Executing instructions"]

// Прямой доступ к подсистеме по-прежнему возможен при необходимости
cpu := &CPU{}
cpu.Freeze()	// можно использовать подсистему напрямую
\`\`\``,
			hint1: `Каждый метод подсистемы прост — просто возвращает форматированную строку с описанием операции:
- Freeze(): возвращает "CPU: Freezing processor"
- Jump(addr): использует fmt.Sprintf("CPU: Jumping to address %d", addr)
- Execute(): возвращает "CPU: Executing instructions"
- Load(position, data): использует fmt.Sprintf("Memory: Loading data at %d", position)
- Read(lba, size): использует fmt.Sprintf("HardDrive: Reading sector %d (%d bytes)", lba, size)`,
			hint2: `Start() оркестрирует последовательность загрузки, вызывая методы подсистемы по порядку:
\`\`\`go
results := make([]string, 0, 5)  // предварительно выделяем slice
results = append(results, c.cpu.Freeze())
results = append(results, c.hardDrive.Read(0, 1024))
results = append(results, c.memory.Load(0, nil))
results = append(results, c.cpu.Jump(0))
results = append(results, c.cpu.Execute())
return results
\`\`\`

Фасад не добавляет логику — он просто координирует правильную последовательность вызовов подсистемы.`,
			whyItMatters: `## Зачем нужен Facade

Паттерн Facade решает проблему сложности использования API подсистем. Без него клиенты должны понимать, как координировать множество компонентов подсистемы, что приводит к сложному клиентскому коду и тесной связанности.

**Проблема — без Facade:**
\`\`\`go
// Клиент должен знать правильную последовательность загрузки
func bootComputer() {
    cpu := &CPU{}
    memory := &Memory{}
    hd := &HardDrive{}

    cpu.Freeze()  // должен вызываться первым
    data := hd.Read(0, 1024)  // чтение загрузочного сектора
    memory.Load(0, data)  // загрузка по правильному адресу
    cpu.Jump(0)  // переход к адресу загрузки
    cpu.Execute()  // начало выполнения
    // Неправильный порядок = сбой системы
}
\`\`\`

**Решение — с Facade:**
\`\`\`go
// Клиент просто вызывает Start()
func bootComputer() {
    computer := NewComputerFacade()
    computer.Start()  // фасад обрабатывает правильную последовательность
}
\`\`\`

## Реальные примеры на Go

**1. Фасад HTTP-сервера:**
\`\`\`go
// Сложные подсистемы
type Router struct { /* логика маршрутизации */ }
type Middleware struct { /* аутентификация, логирование */ }
type Handler struct { /* обработка запросов */ }
type Server struct { /* HTTP-сервер */ }

// Фасад упрощает настройку
type WebAppFacade struct {
    router     *Router
    middleware *Middleware
    server     *Server
}

func NewWebApp(config Config) *WebAppFacade {
    return &WebAppFacade{
        router:     NewRouter(),
        middleware: NewMiddleware(config.Auth),
        server:     NewServer(config.Port),
    }
}

func (w *WebAppFacade) Start() error {
    w.router.Setup()
    w.middleware.Apply(w.router)
    return w.server.Listen(w.router)
}

// Клиентский код прост
app := NewWebApp(config)
app.Start()
\`\`\`

**2. Фасад миграции базы данных:**
\`\`\`go
type SchemaReader struct { /* читает схему */ }
type DiffCalculator struct { /* сравнивает схемы */ }
type MigrationGenerator struct { /* создаёт SQL */ }
type MigrationRunner struct { /* выполняет миграции */ }

type MigrationFacade struct {
    reader    *SchemaReader
    differ    *DiffCalculator
    generator *MigrationGenerator
    runner    *MigrationRunner
}

func (f *MigrationFacade) Migrate(targetSchema Schema) error {
    current := f.reader.ReadCurrentSchema()
    diff := f.differ.Calculate(current, targetSchema)
    migration := f.generator.Generate(diff)
    return f.runner.Execute(migration)
}

// Клиент просто вызывает
facade.Migrate(newSchema)
\`\`\`

## Продакшен паттерн: Оформление заказа E-commerce

\`\`\`go
// Сложные подсистемы
type InventoryService struct{}
func (i *InventoryService) CheckStock(items []Item) (bool, error)
func (i *InventoryService) Reserve(items []Item) (string, error)
func (i *InventoryService) Release(reservationID string) error

type PaymentService struct{}
func (p *PaymentService) Validate(card CardInfo) error
func (p *PaymentService) Charge(card CardInfo, amount float64) (string, error)
func (p *PaymentService) Refund(transactionID string) error

type ShippingService struct{}
func (s *ShippingService) CalculateRates(addr Address, items []Item) []Rate
func (s *ShippingService) CreateLabel(order Order) (string, error)

type NotificationService struct{}
func (n *NotificationService) SendConfirmation(email string, order Order) error

// Фасад Checkout — оркестрирует сложный поток
type CheckoutFacade struct {
    inventory    *InventoryService
    payment      *PaymentService
    shipping     *ShippingService
    notification *NotificationService
}

func (f *CheckoutFacade) ProcessOrder(cart Cart, card CardInfo) (*Order, error) {
    // Шаг 1: Проверка наличия
    available, err := f.inventory.CheckStock(cart.Items)
    if err != nil || !available {
        return nil, ErrOutOfStock
    }

    // Шаг 2: Резервирование товаров
    reservationID, err := f.inventory.Reserve(cart.Items)
    if err != nil {
        return nil, err
    }

    // Шаг 3: Обработка платежа
    if err := f.payment.Validate(card); err != nil {
        f.inventory.Release(reservationID)
        return nil, err
    }

    transactionID, err := f.payment.Charge(card, cart.Total())
    if err != nil {
        f.inventory.Release(reservationID)
        return nil, err
    }

    // Шаг 4: Создание этикетки доставки
    order := &Order{Items: cart.Items, TransactionID: transactionID}
    trackingNumber, err := f.shipping.CreateLabel(*order)
    if err != nil {
        f.payment.Refund(transactionID)
        f.inventory.Release(reservationID)
        return nil, err
    }
    order.TrackingNumber = trackingNumber

    // Шаг 5: Отправка подтверждения
    f.notification.SendConfirmation(cart.CustomerEmail, *order)

    return order, nil
}

// Клиентский код прост
order, err := checkout.ProcessOrder(cart, cardInfo)
\`\`\`

## Частые ошибки

**1. Размещение бизнес-логики в фасаде:**
\`\`\`go
// Плохо — фасад вычисляет цены
func (f *OrderFacade) CreateOrder(items []Item) *Order {
    total := 0.0
    for _, item := range items {
        total += item.Price * float64(item.Quantity)  // бизнес-логика!
        if item.Quantity > 10 {
            total *= 0.9  // логика скидок в фасаде!
        }
    }
    return f.orderService.Create(items, total)
}

// Хорошо — фасад только координирует
func (f *OrderFacade) CreateOrder(items []Item) *Order {
    total := f.pricingService.Calculate(items)  // делегирование сервису
    return f.orderService.Create(items, total)
}
\`\`\`

**2. Делать фасад единственной точкой доступа:**
\`\`\`go
// Плохо — полностью скрываем подсистемы
type BadFacade struct {
    inventory *inventoryService  // приватно, нет прямого доступа
}

// Хорошо — фасад для удобства, не для ограничения
type GoodFacade struct {
    Inventory *InventoryService  // публично, прямой доступ при необходимости
    Payment   *PaymentService
}

// Клиент может использовать фасад
facade.ProcessOrder(cart)
// Или обращаться к подсистемам напрямую для продвинутого использования
facade.Inventory.GetDetailedStock()
\`\`\`

**3. Слишком много методов в фасаде:**
\`\`\`go
// Плохо — фасад становится god object
type BadFacade struct { /* ... */ }
func (f *BadFacade) CreateUser() {}
func (f *BadFacade) UpdateUser() {}
func (f *BadFacade) DeleteUser() {}
func (f *BadFacade) CreateOrder() {}
func (f *BadFacade) UpdateOrder() {}
// ... ещё 50 методов

// Хорошо — сфокусированные фасады
type UserFacade struct { /* ... */ }
func (f *UserFacade) Register() {}
func (f *UserFacade) UpdateProfile() {}

type OrderFacade struct { /* ... */ }
func (f *OrderFacade) Checkout() {}
func (f *OrderFacade) Cancel() {}
\`\`\`

**Ключевые принципы:**
- Фасад обеспечивает удобство, а не контроль — подсистемы остаются доступными
- Фасад координирует, но не содержит бизнес-логику
- Создавайте сфокусированные фасады для разных сценариев, а не один мега-фасад
- Фасад упрощает типичный случай, позволяя гибкость для продвинутого использования`,
			solutionCode: `package patterns

import "fmt"

// CPU представляет процессор
type CPU struct{}	// компонент подсистемы

// Freeze останавливает процессор
func (c *CPU) Freeze() string {	// операция подсистемы
	return "CPU: Freezing processor"	// возвращает результат операции
}

// Jump переходит по адресу памяти
func (c *CPU) Jump(addr int64) string {	// операция подсистемы с параметром
	return fmt.Sprintf("CPU: Jumping to address %d", addr)	// форматирование с адресом
}

// Execute выполняет инструкции
func (c *CPU) Execute() string {	// операция подсистемы
	return "CPU: Executing instructions"	// возвращает результат операции
}

// Memory представляет ОЗУ
type Memory struct{}	// компонент подсистемы

// Load загружает данные в память
func (m *Memory) Load(position int64, data []byte) string {	// операция подсистемы
	return fmt.Sprintf("Memory: Loading data at %d", position)	// форматирование с позицией
}

// HardDrive представляет накопитель
type HardDrive struct{}	// компонент подсистемы

// Read читает данные с диска
func (h *HardDrive) Read(lba int64, size int) string {	// операция подсистемы
	return fmt.Sprintf("HardDrive: Reading sector %d (%d bytes)", lba, size)	// форматирование с сектором и размером
}

// ComputerFacade предоставляет простой интерфейс к сложной подсистеме
type ComputerFacade struct {	// фасад — унифицированный интерфейс
	cpu       *CPU	// ссылка на подсистему
	memory    *Memory	// ссылка на подсистему
	hardDrive *HardDrive	// ссылка на подсистему
}

// NewComputerFacade создаёт фасад со всеми компонентами
func NewComputerFacade() *ComputerFacade {	// фабричная функция для фасада
	return &ComputerFacade{	// инициализация всех компонентов подсистемы
		cpu:       &CPU{},	// создание CPU
		memory:    &Memory{},	// создание Memory
		hardDrive: &HardDrive{},	// создание HardDrive
	}
}

// Start выполняет последовательность загрузки
func (c *ComputerFacade) Start() []string {	// метод фасада — упрощённый интерфейс
	results := make([]string, 0, 5)	// предварительное выделение для ожидаемых результатов
	results = append(results, c.cpu.Freeze())	// шаг 1: заморозка процессора
	results = append(results, c.hardDrive.Read(0, 1024))	// шаг 2: чтение загрузочного сектора
	results = append(results, c.memory.Load(0, nil))	// шаг 3: загрузка в ОЗУ
	results = append(results, c.cpu.Jump(0))	// шаг 4: переход к начальному адресу
	results = append(results, c.cpu.Execute())	// шаг 5: начало выполнения
	return results	// возврат всех результатов операций
}`
		},
		uz: {
			title: 'Facade (Fasad) Pattern',
			description: `Go tilida Facade patternini amalga oshiring — quyi tizim interfeyslariga birlashtirilgan interfeys taqdim eting.

Facade patterni murakkab quyi tizimga soddalashtirilgan interfeys taqdim etadi. U quyi tizimni inkapsulyatsiya qilmaydi, balki qulay yuqori darajadagi interfeys taqdim etadi, shu bilan birga kerak bo'lganda quyi tizim komponentlariga to'g'ridan-to'g'ri kirishga ruxsat beradi.

**Siz amalga oshirasiz:**

1. **CPU strukturasi** - Protsessor operatsiyalari uchun Freeze, Execute, Jump metodlari
2. **Memory strukturasi** - RAM operatsiyalari uchun Load metodi
3. **HardDrive strukturasi** - Saqlash operatsiyalari uchun Read metodi
4. **ComputerFacade** - Yuklash ketma-ketligini boshqaradigan oddiy Start() metodi

**Foydalanish namunasi:**

\`\`\`go
// Fasad yaratish — mijozga CPU, Memory, HardDrive haqida bilish kerak emas
computer := NewComputerFacade()	// fasad quyi tizimni inkapsulyatsiya qiladi

// Oddiy bir chaqiruv interfeysi murakkab yuklash ketma-ketligini yashiradi
result := computer.Start()	// barcha quyi tizim operatsiyalarini boshqaradi
// result quyidagilarni o'z ichiga oladi:
// ["CPU: Freezing processor", "HardDrive: Reading sector 0 (1024 bytes)",
//  "Memory: Loading data at 0", "CPU: Jumping to address 0", "CPU: Executing instructions"]

// Kerak bo'lganda quyi tizimga to'g'ridan-to'g'ri kirish hali ham mumkin
cpu := &CPU{}
cpu.Freeze()	// quyi tizimni to'g'ridan-to'g'ri ishlatish mumkin
\`\`\``,
			hint1: `Har bir quyi tizim metodi oddiy — operatsiyani tavsiflovchi formatlangan satr qaytaradi:
- Freeze(): "CPU: Freezing processor" qaytaradi
- Jump(addr): fmt.Sprintf("CPU: Jumping to address %d", addr) ishlatadi
- Execute(): "CPU: Executing instructions" qaytaradi
- Load(position, data): fmt.Sprintf("Memory: Loading data at %d", position) ishlatadi
- Read(lba, size): fmt.Sprintf("HardDrive: Reading sector %d (%d bytes)", lba, size) ishlatadi`,
			hint2: `Start() quyi tizim metodlarini tartib bilan chaqirib yuklash ketma-ketligini boshqaradi:
\`\`\`go
results := make([]string, 0, 5)  // slice ni oldindan ajratish
results = append(results, c.cpu.Freeze())
results = append(results, c.hardDrive.Read(0, 1024))
results = append(results, c.memory.Load(0, nil))
results = append(results, c.cpu.Jump(0))
results = append(results, c.cpu.Execute())
return results
\`\`\`

Fasad mantiq qo'shmaydi — u shunchaki quyi tizim chaqiruvlarining to'g'ri ketma-ketligini muvofiqlashtiradi.`,
			whyItMatters: `## Nega Facade kerak

Facade patterni quyi tizim API laridan foydalanish murakkabligini hal qiladi. Busiz mijozlar bir nechta quyi tizim komponentlarini qanday muvofiqlashtirish kerakligini tushunishi kerak, bu murakkab mijoz kodi va qattiq bog'lanishga olib keladi.

**Muammo — Facade siz:**
\`\`\`go
// Mijoz to'g'ri yuklash ketma-ketligini bilishi kerak
func bootComputer() {
    cpu := &CPU{}
    memory := &Memory{}
    hd := &HardDrive{}

    cpu.Freeze()  // birinchi chaqirilishi kerak
    data := hd.Read(0, 1024)  // yuklash sektorini o'qish
    memory.Load(0, data)  // to'g'ri manzilga yuklash
    cpu.Jump(0)  // yuklash manziliga o'tish
    cpu.Execute()  // bajarishni boshlash
    // Noto'g'ri tartib = tizim buzilishi
}
\`\`\`

**Yechim — Facade bilan:**
\`\`\`go
// Mijoz shunchaki Start() ni chaqiradi
func bootComputer() {
    computer := NewComputerFacade()
    computer.Start()  // fasad to'g'ri ketma-ketlikni boshqaradi
}
\`\`\`

## Go dagi real dunyo misollar

**1. HTTP Server fasadi:**
\`\`\`go
// Murakkab quyi tizimlar
type Router struct { /* marshrutlash mantiqi */ }
type Middleware struct { /* autentifikatsiya, loglash */ }
type Handler struct { /* so'rovlarni qayta ishlash */ }
type Server struct { /* HTTP server */ }

// Fasad sozlashni soddalashtiradi
type WebAppFacade struct {
    router     *Router
    middleware *Middleware
    server     *Server
}

func NewWebApp(config Config) *WebAppFacade {
    return &WebAppFacade{
        router:     NewRouter(),
        middleware: NewMiddleware(config.Auth),
        server:     NewServer(config.Port),
    }
}

func (w *WebAppFacade) Start() error {
    w.router.Setup()
    w.middleware.Apply(w.router)
    return w.server.Listen(w.router)
}

// Mijoz kodi oddiy
app := NewWebApp(config)
app.Start()
\`\`\`

**2. Ma'lumotlar bazasi migratsiyasi fasadi:**
\`\`\`go
type SchemaReader struct { /* sxemani o'qiydi */ }
type DiffCalculator struct { /* sxemalarni solishtiradi */ }
type MigrationGenerator struct { /* SQL yaratadi */ }
type MigrationRunner struct { /* migratsiyalarni bajaradi */ }

type MigrationFacade struct {
    reader    *SchemaReader
    differ    *DiffCalculator
    generator *MigrationGenerator
    runner    *MigrationRunner
}

func (f *MigrationFacade) Migrate(targetSchema Schema) error {
    current := f.reader.ReadCurrentSchema()
    diff := f.differ.Calculate(current, targetSchema)
    migration := f.generator.Generate(diff)
    return f.runner.Execute(migration)
}

// Mijoz shunchaki chaqiradi
facade.Migrate(newSchema)
\`\`\`

## Prodakshen pattern: E-commerce buyurtmani rasmiylashtirish

\`\`\`go
// Murakkab quyi tizimlar
type InventoryService struct{}
func (i *InventoryService) CheckStock(items []Item) (bool, error)
func (i *InventoryService) Reserve(items []Item) (string, error)
func (i *InventoryService) Release(reservationID string) error

type PaymentService struct{}
func (p *PaymentService) Validate(card CardInfo) error
func (p *PaymentService) Charge(card CardInfo, amount float64) (string, error)
func (p *PaymentService) Refund(transactionID string) error

type ShippingService struct{}
func (s *ShippingService) CalculateRates(addr Address, items []Item) []Rate
func (s *ShippingService) CreateLabel(order Order) (string, error)

type NotificationService struct{}
func (n *NotificationService) SendConfirmation(email string, order Order) error

// Checkout fasadi — murakkab oqimni boshqaradi
type CheckoutFacade struct {
    inventory    *InventoryService
    payment      *PaymentService
    shipping     *ShippingService
    notification *NotificationService
}

func (f *CheckoutFacade) ProcessOrder(cart Cart, card CardInfo) (*Order, error) {
    // Qadam 1: Mavjudlikni tekshirish
    available, err := f.inventory.CheckStock(cart.Items)
    if err != nil || !available {
        return nil, ErrOutOfStock
    }

    // Qadam 2: Mahsulotlarni band qilish
    reservationID, err := f.inventory.Reserve(cart.Items)
    if err != nil {
        return nil, err
    }

    // Qadam 3: To'lovni qayta ishlash
    if err := f.payment.Validate(card); err != nil {
        f.inventory.Release(reservationID)
        return nil, err
    }

    transactionID, err := f.payment.Charge(card, cart.Total())
    if err != nil {
        f.inventory.Release(reservationID)
        return nil, err
    }

    // Qadam 4: Yetkazib berish yorlig'ini yaratish
    order := &Order{Items: cart.Items, TransactionID: transactionID}
    trackingNumber, err := f.shipping.CreateLabel(*order)
    if err != nil {
        f.payment.Refund(transactionID)
        f.inventory.Release(reservationID)
        return nil, err
    }
    order.TrackingNumber = trackingNumber

    // Qadam 5: Tasdiqlash yuborish
    f.notification.SendConfirmation(cart.CustomerEmail, *order)

    return order, nil
}

// Mijoz kodi oddiy
order, err := checkout.ProcessOrder(cart, cardInfo)
\`\`\`

## Keng tarqalgan xatolar

**1. Biznes mantiqni fasadda joylashtirish:**
\`\`\`go
// Yomon — fasad narxlarni hisobaydi
func (f *OrderFacade) CreateOrder(items []Item) *Order {
    total := 0.0
    for _, item := range items {
        total += item.Price * float64(item.Quantity)  // biznes mantiq!
        if item.Quantity > 10 {
            total *= 0.9  // chegirma mantiqi fasadda!
        }
    }
    return f.orderService.Create(items, total)
}

// Yaxshi — fasad faqat muvofiqlashtiradi
func (f *OrderFacade) CreateOrder(items []Item) *Order {
    total := f.pricingService.Calculate(items)  // xizmatga delegatsiya
    return f.orderService.Create(items, total)
}
\`\`\`

**2. Fasadni yagona kirish nuqtasi qilish:**
\`\`\`go
// Yomon — quyi tizimlarni to'liq yashirish
type BadFacade struct {
    inventory *inventoryService  // xususiy, to'g'ridan-to'g'ri kirish yo'q
}

// Yaxshi — fasad qulaylik uchun, cheklov uchun emas
type GoodFacade struct {
    Inventory *InventoryService  // ommaviy, kerak bo'lganda to'g'ridan-to'g'ri kirish
    Payment   *PaymentService
}

// Mijoz fasadni ishlata oladi
facade.ProcessOrder(cart)
// Yoki ilg'or foydalanish uchun quyi tizimlarga to'g'ridan-to'g'ri kirish
facade.Inventory.GetDetailedStock()
\`\`\`

**3. Fasadda juda ko'p metodlar:**
\`\`\`go
// Yomon — fasad god object ga aylanadi
type BadFacade struct { /* ... */ }
func (f *BadFacade) CreateUser() {}
func (f *BadFacade) UpdateUser() {}
func (f *BadFacade) DeleteUser() {}
func (f *BadFacade) CreateOrder() {}
func (f *BadFacade) UpdateOrder() {}
// ... yana 50 ta metod

// Yaxshi — yo'naltirilgan fasadlar
type UserFacade struct { /* ... */ }
func (f *UserFacade) Register() {}
func (f *UserFacade) UpdateProfile() {}

type OrderFacade struct { /* ... */ }
func (f *OrderFacade) Checkout() {}
func (f *OrderFacade) Cancel() {}
\`\`\`

**Asosiy tamoyillar:**
- Fasad qulaylik beradi, nazorat emas — quyi tizimlar kirish mumkin bo'lib qoladi
- Fasad muvofiqlashtiradi, lekin biznes mantiqni o'z ichiga olmaydi
- Bitta mega-fasad o'rniga turli stsenariylar uchun yo'naltirilgan fasadlar yarating
- Fasad oddiy holatni soddalashtiradi, ilg'or foydalanish uchun moslashuvchanlikka ruxsat beradi`,
			solutionCode: `package patterns

import "fmt"

// CPU protsessorni ifodalaydi
type CPU struct{}	// quyi tizim komponenti

// Freeze protsessorni to'xtatadi
func (c *CPU) Freeze() string {	// quyi tizim operatsiyasi
	return "CPU: Freezing processor"	// operatsiya natijasini qaytaradi
}

// Jump xotira manziliga o'tadi
func (c *CPU) Jump(addr int64) string {	// parametrli quyi tizim operatsiyasi
	return fmt.Sprintf("CPU: Jumping to address %d", addr)	// manzil bilan formatlash
}

// Execute ko'rsatmalarni bajaradi
func (c *CPU) Execute() string {	// quyi tizim operatsiyasi
	return "CPU: Executing instructions"	// operatsiya natijasini qaytaradi
}

// Memory RAM ni ifodalaydi
type Memory struct{}	// quyi tizim komponenti

// Load ma'lumotlarni xotiraga yuklaydi
func (m *Memory) Load(position int64, data []byte) string {	// quyi tizim operatsiyasi
	return fmt.Sprintf("Memory: Loading data at %d", position)	// pozitsiya bilan formatlash
}

// HardDrive saqlash qurilmasini ifodalaydi
type HardDrive struct{}	// quyi tizim komponenti

// Read diskdan ma'lumotlarni o'qiydi
func (h *HardDrive) Read(lba int64, size int) string {	// quyi tizim operatsiyasi
	return fmt.Sprintf("HardDrive: Reading sector %d (%d bytes)", lba, size)	// sektor va hajm bilan formatlash
}

// ComputerFacade murakkab quyi tizimga oddiy interfeys beradi
type ComputerFacade struct {	// fasad — birlashtirilgan interfeys
	cpu       *CPU	// quyi tizimga havola
	memory    *Memory	// quyi tizimga havola
	hardDrive *HardDrive	// quyi tizimga havola
}

// NewComputerFacade barcha komponentlar bilan fasad yaratadi
func NewComputerFacade() *ComputerFacade {	// fasad uchun fabrika funksiyasi
	return &ComputerFacade{	// barcha quyi tizim komponentlarini ishga tushirish
		cpu:       &CPU{},	// CPU yaratish
		memory:    &Memory{},	// Memory yaratish
		hardDrive: &HardDrive{},	// HardDrive yaratish
	}
}

// Start yuklash ketma-ketligini bajaradi
func (c *ComputerFacade) Start() []string {	// fasad metodi — soddalashtirilgan interfeys
	results := make([]string, 0, 5)	// kutilgan natijalar uchun oldindan ajratish
	results = append(results, c.cpu.Freeze())	// qadam 1: protsessorni muzlatish
	results = append(results, c.hardDrive.Read(0, 1024))	// qadam 2: yuklash sektorini o'qish
	results = append(results, c.memory.Load(0, nil))	// qadam 3: RAM ga yuklash
	results = append(results, c.cpu.Jump(0))	// qadam 4: boshlang'ich manzilga o'tish
	results = append(results, c.cpu.Execute())	// qadam 5: bajarishni boshlash
	return results	// barcha operatsiya natijalarini qaytarish
}`
		}
	}
};

export default task;
