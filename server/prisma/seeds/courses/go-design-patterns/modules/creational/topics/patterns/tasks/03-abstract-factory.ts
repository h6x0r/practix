import { Task } from '../../../../../../../types';

export const task: Task = {
	slug: 'go-dp-abstract-factory',
	title: 'Abstract Factory Pattern',
	difficulty: 'hard',
	tags: ['go', 'design-patterns', 'creational', 'abstract-factory'],
	estimatedTime: '45m',
	isPremium: false,
	youtubeUrl: '',
	description: `Implement the Abstract Factory pattern in Go - provide an interface for creating families of related objects without specifying their concrete classes.

**You will implement:**

1. **GUIFactory interface** - Creates related Button and Checkbox
2. **Button interface** - Render() method
3. **Checkbox interface** - Check() method
4. **WindowsFactory** - Creates Windows-styled components
5. **MacFactory** - Creates Mac-styled components

**Example Usage:**

\`\`\`go
// Get factory based on OS type
factory := GetFactory("windows")

// Create related components from the same family
button := factory.CreateButton()
checkbox := factory.CreateCheckbox()

// Use components - they are guaranteed to be compatible
button.Render()   // "Rendering Windows button"
checkbox.Check()  // "Windows checkbox checked"

// Switch to Mac factory
macFactory := GetFactory("mac")
macButton := macFactory.CreateButton()
macButton.Render() // "Rendering Mac button"
\`\`\``,
	initialCode: `package patterns

type Button interface {
}

type Checkbox interface {
}

type GUIFactory interface {
}

type WindowsButton struct{}

func (b *WindowsButton) Render() string {
}

type WindowsCheckbox struct{}

func (c *WindowsCheckbox) Check() string {
}

type MacButton struct{}

func (b *MacButton) Render() string {
}

type MacCheckbox struct{}

func (c *MacCheckbox) Check() string {
}

type WindowsFactory struct{}

func (f *WindowsFactory) CreateButton() Button {
}

func (f *WindowsFactory) CreateCheckbox() Checkbox {
}

type MacFactory struct{}

func (f *MacFactory) CreateButton() Button {
}

func (f *MacFactory) CreateCheckbox() Checkbox {
}

func GetFactory(osType string) GUIFactory {
}`,
	solutionCode: `package patterns

type Button interface {	// Abstract Product A - common interface for all button types
	Render() string	// all buttons must be able to render themselves
}

type Checkbox interface {	// Abstract Product B - common interface for all checkboxes
	Check() string	// all checkboxes must be able to check themselves
}

type GUIFactory interface {	// Abstract Factory - creates families of related products
	CreateButton() Button	// factory method for creating buttons
	CreateCheckbox() Checkbox	// factory method for creating checkboxes
}

type WindowsButton struct{}	// Concrete Product A1 - Windows-specific button

func (b *WindowsButton) Render() string {	// implements Button interface for Windows
	return "Rendering Windows button"	// Windows-specific rendering logic
}

type WindowsCheckbox struct{}	// Concrete Product B1 - Windows-specific checkbox

func (c *WindowsCheckbox) Check() string {	// implements Checkbox interface for Windows
	return "Windows checkbox checked"	// Windows-specific check behavior
}

type MacButton struct{}	// Concrete Product A2 - Mac-specific button

func (b *MacButton) Render() string {	// implements Button interface for Mac
	return "Rendering Mac button"	// Mac-specific rendering logic
}

type MacCheckbox struct{}	// Concrete Product B2 - Mac-specific checkbox

func (c *MacCheckbox) Check() string {	// implements Checkbox interface for Mac
	return "Mac checkbox checked"	// Mac-specific check behavior
}

type WindowsFactory struct{}	// Concrete Factory 1 - creates Windows product family

func (f *WindowsFactory) CreateButton() Button {	// implements GUIFactory for Windows
	return &WindowsButton{}	// returns Windows-specific button
}

func (f *WindowsFactory) CreateCheckbox() Checkbox {	// implements GUIFactory for Windows
	return &WindowsCheckbox{}	// returns Windows-specific checkbox
}

type MacFactory struct{}	// Concrete Factory 2 - creates Mac product family

func (f *MacFactory) CreateButton() Button {	// implements GUIFactory for Mac
	return &MacButton{}	// returns Mac-specific button
}

func (f *MacFactory) CreateCheckbox() Checkbox {	// implements GUIFactory for Mac
	return &MacCheckbox{}	// returns Mac-specific checkbox
}

func GetFactory(osType string) GUIFactory {	// factory selector based on runtime configuration
	switch osType {	// determine which factory to create
	case "mac":	// if Mac OS requested
		return &MacFactory{}	// return Mac factory that creates Mac components
	default:	// for Windows or any other OS
		return &WindowsFactory{}	// default to Windows factory
	}
}`,
	hint1: `Each concrete product implements its interface with a simple string return. Windows components return "Windows" in their output, Mac components return "Mac". The key is that each factory creates products from the SAME family - Windows factory creates only Windows components.`,
	hint2: `GetFactory uses a switch statement to return the appropriate factory based on OS type. Each factory's CreateButton and CreateCheckbox methods return new instances of their respective concrete products. This ensures that a Windows factory will never accidentally create a Mac button.`,
	whyItMatters: `**1. Why Abstract Factory Exists**

Abstract Factory solves the problem of creating FAMILIES of related objects that must work together. Unlike Factory Method which creates single products, Abstract Factory ensures that all products created by a factory are compatible with each other.

**The Problem It Solves:**

\`\`\`go
// WITHOUT Abstract Factory - mixing incompatible components
func CreateUI(osType string) {
    var button Button
    var checkbox Checkbox

    // Dangerous: could mix Windows button with Mac checkbox!
    if osType == "windows" {
        button = &WindowsButton{}
    } else {
        button = &MacButton{}
    }

    // Bug: developer might use wrong condition
    if osType == "mac" {  // Oops! Different condition
        checkbox = &MacCheckbox{}
    } else {
        checkbox = &WindowsCheckbox{}
    }

    // Now we have incompatible UI components!
}
\`\`\`

**WITH Abstract Factory:**

\`\`\`go
// Safe: factory guarantees all components are from same family
func CreateUI(osType string) {
    factory := GetFactory(osType)  // get the right factory

    // All components are guaranteed to be compatible
    button := factory.CreateButton()      // same family
    checkbox := factory.CreateCheckbox()  // same family

    // Impossible to mix Windows with Mac components
}
\`\`\`

**2. Real-World Examples in Go**

**Database Driver Abstraction:**

\`\`\`go
// Abstract Factory for database operations
type DBFactory interface {
    CreateConnection() Connection
    CreateQueryBuilder() QueryBuilder
    CreateTransaction() Transaction
}

// PostgreSQL family
type PostgresFactory struct{}
func (f *PostgresFactory) CreateConnection() Connection {
    return &PostgresConnection{}
}
func (f *PostgresFactory) CreateQueryBuilder() QueryBuilder {
    return &PostgresQueryBuilder{}  // uses $1, $2 placeholders
}

// MySQL family
type MySQLFactory struct{}
func (f *MySQLFactory) CreateConnection() Connection {
    return &MySQLConnection{}
}
func (f *MySQLFactory) CreateQueryBuilder() QueryBuilder {
    return &MySQLQueryBuilder{}  // uses ? placeholders
}
\`\`\`

**Cloud Provider Abstraction:**

\`\`\`go
// Abstract Factory for cloud services
type CloudFactory interface {
    CreateStorage() ObjectStorage
    CreateCompute() ComputeService
    CreateDatabase() ManagedDB
}

// AWS family - all services work together
type AWSFactory struct{}
func (f *AWSFactory) CreateStorage() ObjectStorage {
    return &S3Storage{}  // uses AWS IAM roles
}

// GCP family - all services work together
type GCPFactory struct{}
func (f *GCPFactory) CreateStorage() ObjectStorage {
    return &GCSStorage{}  // uses GCP service accounts
}
\`\`\`

**3. Production Pattern**

\`\`\`go
package factory

import "os"

// Product interfaces
type Logger interface {
    Log(message string)
    Error(err error)
}

type Metrics interface {
    Count(name string)
    Gauge(name string, value float64)
}

type Tracer interface {
    StartSpan(name string) Span
}

// Observability factory - creates compatible observability tools
type ObservabilityFactory interface {
    CreateLogger() Logger
    CreateMetrics() Metrics
    CreateTracer() Tracer
}

// Production factory - uses real services
type ProductionFactory struct {
    serviceName string
}

func NewProductionFactory(serviceName string) *ProductionFactory {
    return &ProductionFactory{serviceName: serviceName}
}

func (f *ProductionFactory) CreateLogger() Logger {
    return NewDatadogLogger(f.serviceName)
}

func (f *ProductionFactory) CreateMetrics() Metrics {
    return NewDatadogMetrics(f.serviceName)
}

func (f *ProductionFactory) CreateTracer() Tracer {
    return NewDatadogTracer(f.serviceName)
}

// Development factory - uses console/mock implementations
type DevelopmentFactory struct{}

func (f *DevelopmentFactory) CreateLogger() Logger {
    return NewConsoleLogger()
}

func (f *DevelopmentFactory) CreateMetrics() Metrics {
    return NewNoOpMetrics()  // metrics disabled in dev
}

func (f *DevelopmentFactory) CreateTracer() Tracer {
    return NewNoOpTracer()  // tracing disabled in dev
}

// Factory selector
func GetObservabilityFactory(serviceName string) ObservabilityFactory {
    if os.Getenv("ENV") == "production" {
        return NewProductionFactory(serviceName)
    }
    return &DevelopmentFactory{}
}
\`\`\`

**4. Common Mistakes to Avoid**

\`\`\`go
// MISTAKE 1: Creating products outside the factory
factory := GetFactory("windows")
button := factory.CreateButton()
checkbox := &MacCheckbox{}  // Wrong! Bypasses factory, breaks consistency

// MISTAKE 2: Adding non-related products to factory
type GUIFactory interface {
    CreateButton() Button
    CreateCheckbox() Checkbox
    CreateHTTPClient() HTTPClient  // Wrong! Not related to GUI family
}

// MISTAKE 3: Factory that can create mixed families
type BadFactory struct {
    buttonOS   string
    checkboxOS string
}
func (f *BadFactory) CreateButton() Button {
    if f.buttonOS == "mac" { return &MacButton{} }
    return &WindowsButton{}
}
func (f *BadFactory) CreateCheckbox() Checkbox {
    // Can create checkbox from different OS! Defeats the purpose
    if f.checkboxOS == "mac" { return &MacCheckbox{} }
    return &WindowsCheckbox{}
}

// CORRECT: Each factory creates only its own family
type WindowsFactory struct{}  // only creates Windows components
type MacFactory struct{}      // only creates Mac components
\`\`\``,
	order: 2,
	testCode: `package patterns

import (
	"testing"
)

// Test1: WindowsButton.Render returns correct string
func Test1(t *testing.T) {
	btn := &WindowsButton{}
	if btn.Render() != "Rendering Windows button" {
		t.Error("WindowsButton.Render should return 'Rendering Windows button'")
	}
}

// Test2: MacButton.Render returns correct string
func Test2(t *testing.T) {
	btn := &MacButton{}
	if btn.Render() != "Rendering Mac button" {
		t.Error("MacButton.Render should return 'Rendering Mac button'")
	}
}

// Test3: WindowsCheckbox.Check returns correct string
func Test3(t *testing.T) {
	cb := &WindowsCheckbox{}
	if cb.Check() != "Windows checkbox checked" {
		t.Error("WindowsCheckbox.Check should return 'Windows checkbox checked'")
	}
}

// Test4: MacCheckbox.Check returns correct string
func Test4(t *testing.T) {
	cb := &MacCheckbox{}
	if cb.Check() != "Mac checkbox checked" {
		t.Error("MacCheckbox.Check should return 'Mac checkbox checked'")
	}
}

// Test5: WindowsFactory creates Windows components
func Test5(t *testing.T) {
	factory := &WindowsFactory{}
	btn := factory.CreateButton()
	if btn.Render() != "Rendering Windows button" {
		t.Error("WindowsFactory should create WindowsButton")
	}
}

// Test6: MacFactory creates Mac components
func Test6(t *testing.T) {
	factory := &MacFactory{}
	cb := factory.CreateCheckbox()
	if cb.Check() != "Mac checkbox checked" {
		t.Error("MacFactory should create MacCheckbox")
	}
}

// Test7: GetFactory returns WindowsFactory by default
func Test7(t *testing.T) {
	factory := GetFactory("windows")
	btn := factory.CreateButton()
	if btn.Render() != "Rendering Windows button" {
		t.Error("GetFactory('windows') should return WindowsFactory")
	}
}

// Test8: GetFactory returns MacFactory for 'mac'
func Test8(t *testing.T) {
	factory := GetFactory("mac")
	btn := factory.CreateButton()
	if btn.Render() != "Rendering Mac button" {
		t.Error("GetFactory('mac') should return MacFactory")
	}
}

// Test9: GetFactory defaults to Windows for unknown OS
func Test9(t *testing.T) {
	factory := GetFactory("linux")
	btn := factory.CreateButton()
	if btn.Render() != "Rendering Windows button" {
		t.Error("GetFactory should default to WindowsFactory for unknown OS")
	}
}

// Test10: Factory creates consistent family
func Test10(t *testing.T) {
	factory := GetFactory("mac")
	btn := factory.CreateButton()
	cb := factory.CreateCheckbox()
	if btn.Render() != "Rendering Mac button" || cb.Check() != "Mac checkbox checked" {
		t.Error("Factory should create components from same family")
	}
}
`,
	translations: {
		ru: {
			title: 'Паттерн Abstract Factory (Абстрактная Фабрика)',
			description: `Реализуйте паттерн Abstract Factory на Go — предоставьте интерфейс для создания семейств связанных объектов без указания их конкретных классов.

**Вы реализуете:**

1. **Интерфейс GUIFactory** - Создаёт связанные Button и Checkbox
2. **Интерфейс Button** - Метод Render()
3. **Интерфейс Checkbox** - Метод Check()
4. **WindowsFactory** - Создаёт компоненты в стиле Windows
5. **MacFactory** - Создаёт компоненты в стиле Mac

**Пример использования:**

\`\`\`go
// Получаем фабрику по типу ОС
factory := GetFactory("windows")

// Создаём связанные компоненты из одного семейства
button := factory.CreateButton()
checkbox := factory.CreateCheckbox()

// Используем компоненты - они гарантированно совместимы
button.Render()   // "Rendering Windows button"
checkbox.Check()  // "Windows checkbox checked"

// Переключаемся на Mac фабрику
macFactory := GetFactory("mac")
macButton := macFactory.CreateButton()
macButton.Render() // "Rendering Mac button"
\`\`\``,
			hint1: `Каждый конкретный продукт реализует свой интерфейс с простым возвратом строки. Windows-компоненты возвращают "Windows", Mac-компоненты — "Mac". Ключевой момент в том, что каждая фабрика создаёт продукты из ОДНОГО семейства — Windows-фабрика создаёт только Windows-компоненты.`,
			hint2: `GetFactory использует switch для возврата подходящей фабрики по типу ОС. Методы CreateButton и CreateCheckbox каждой фабрики возвращают новые экземпляры соответствующих конкретных продуктов. Это гарантирует, что Windows-фабрика никогда случайно не создаст Mac-кнопку.`,
			whyItMatters: `**1. Зачем нужен Abstract Factory**

Abstract Factory решает проблему создания СЕМЕЙСТВ связанных объектов, которые должны работать вместе. В отличие от Factory Method, который создаёт отдельные продукты, Abstract Factory гарантирует совместимость всех продуктов, созданных одной фабрикой.

**Проблема, которую он решает:**

\`\`\`go
// БЕЗ Abstract Factory - смешивание несовместимых компонентов
func CreateUI(osType string) {
    var button Button
    var checkbox Checkbox

    // Опасно: можно смешать Windows кнопку с Mac чекбоксом!
    if osType == "windows" {
        button = &WindowsButton{}
    } else {
        button = &MacButton{}
    }

    // Баг: разработчик может использовать другое условие
    if osType == "mac" {  // Ой! Другое условие
        checkbox = &MacCheckbox{}
    } else {
        checkbox = &WindowsCheckbox{}
    }

    // Теперь у нас несовместимые UI компоненты!
}
\`\`\`

**С Abstract Factory:**

\`\`\`go
// Безопасно: фабрика гарантирует все компоненты из одного семейства
func CreateUI(osType string) {
    factory := GetFactory(osType)  // получаем нужную фабрику

    // Все компоненты гарантированно совместимы
    button := factory.CreateButton()      // одно семейство
    checkbox := factory.CreateCheckbox()  // одно семейство

    // Невозможно смешать Windows с Mac компонентами
}
\`\`\`

**2. Примеры из реального мира в Go**

**Абстракция драйверов баз данных:**

\`\`\`go
// Abstract Factory для операций с базой данных
type DBFactory interface {
    CreateConnection() Connection
    CreateQueryBuilder() QueryBuilder
    CreateTransaction() Transaction
}

// PostgreSQL семейство
type PostgresFactory struct{}
func (f *PostgresFactory) CreateConnection() Connection {
    return &PostgresConnection{}
}
func (f *PostgresFactory) CreateQueryBuilder() QueryBuilder {
    return &PostgresQueryBuilder{}  // использует $1, $2 плейсхолдеры
}

// MySQL семейство
type MySQLFactory struct{}
func (f *MySQLFactory) CreateConnection() Connection {
    return &MySQLConnection{}
}
func (f *MySQLFactory) CreateQueryBuilder() QueryBuilder {
    return &MySQLQueryBuilder{}  // использует ? плейсхолдеры
}
\`\`\`

**Абстракция облачных провайдеров:**

\`\`\`go
// Abstract Factory для облачных сервисов
type CloudFactory interface {
    CreateStorage() ObjectStorage
    CreateCompute() ComputeService
    CreateDatabase() ManagedDB
}

// AWS семейство - все сервисы работают вместе
type AWSFactory struct{}
func (f *AWSFactory) CreateStorage() ObjectStorage {
    return &S3Storage{}  // использует AWS IAM роли
}

// GCP семейство - все сервисы работают вместе
type GCPFactory struct{}
func (f *GCPFactory) CreateStorage() ObjectStorage {
    return &GCSStorage{}  // использует GCP service accounts
}
\`\`\`

**3. Продакшн паттерн**

\`\`\`go
package factory

import "os"

// Интерфейсы продуктов
type Logger interface {
    Log(message string)
    Error(err error)
}

type Metrics interface {
    Count(name string)
    Gauge(name string, value float64)
}

type Tracer interface {
    StartSpan(name string) Span
}

// Фабрика наблюдаемости - создаёт совместимые инструменты
type ObservabilityFactory interface {
    CreateLogger() Logger
    CreateMetrics() Metrics
    CreateTracer() Tracer
}

// Продакшн фабрика - использует реальные сервисы
type ProductionFactory struct {
    serviceName string
}

func NewProductionFactory(serviceName string) *ProductionFactory {
    return &ProductionFactory{serviceName: serviceName}
}

func (f *ProductionFactory) CreateLogger() Logger {
    return NewDatadogLogger(f.serviceName)
}

func (f *ProductionFactory) CreateMetrics() Metrics {
    return NewDatadogMetrics(f.serviceName)
}

func (f *ProductionFactory) CreateTracer() Tracer {
    return NewDatadogTracer(f.serviceName)
}

// Фабрика для разработки - использует консольные/mock реализации
type DevelopmentFactory struct{}

func (f *DevelopmentFactory) CreateLogger() Logger {
    return NewConsoleLogger()
}

func (f *DevelopmentFactory) CreateMetrics() Metrics {
    return NewNoOpMetrics()  // метрики отключены в dev
}

func (f *DevelopmentFactory) CreateTracer() Tracer {
    return NewNoOpTracer()  // трейсинг отключен в dev
}

// Выбор фабрики
func GetObservabilityFactory(serviceName string) ObservabilityFactory {
    if os.Getenv("ENV") == "production" {
        return NewProductionFactory(serviceName)
    }
    return &DevelopmentFactory{}
}
\`\`\`

**4. Типичные ошибки**

\`\`\`go
// ОШИБКА 1: Создание продуктов вне фабрики
factory := GetFactory("windows")
button := factory.CreateButton()
checkbox := &MacCheckbox{}  // Неправильно! Обход фабрики, нарушение консистентности

// ОШИБКА 2: Добавление несвязанных продуктов в фабрику
type GUIFactory interface {
    CreateButton() Button
    CreateCheckbox() Checkbox
    CreateHTTPClient() HTTPClient  // Неправильно! Не относится к GUI семейству
}

// ОШИБКА 3: Фабрика, создающая смешанные семейства
type BadFactory struct {
    buttonOS   string
    checkboxOS string
}
func (f *BadFactory) CreateButton() Button {
    if f.buttonOS == "mac" { return &MacButton{} }
    return &WindowsButton{}
}
func (f *BadFactory) CreateCheckbox() Checkbox {
    // Может создать чекбокс из другой ОС! Нарушает цель паттерна
    if f.checkboxOS == "mac" { return &MacCheckbox{} }
    return &WindowsCheckbox{}
}

// ПРАВИЛЬНО: Каждая фабрика создаёт только своё семейство
type WindowsFactory struct{}  // создаёт только Windows компоненты
type MacFactory struct{}      // создаёт только Mac компоненты
\`\`\``,
			solutionCode: `package patterns

type Button interface {	// Абстрактный Продукт A - общий интерфейс для всех кнопок
	Render() string	// все кнопки должны уметь отрисовывать себя
}

type Checkbox interface {	// Абстрактный Продукт B - общий интерфейс для всех чекбоксов
	Check() string	// все чекбоксы должны уметь отмечать себя
}

type GUIFactory interface {	// Абстрактная Фабрика - создаёт семейства связанных продуктов
	CreateButton() Button	// фабричный метод для создания кнопок
	CreateCheckbox() Checkbox	// фабричный метод для создания чекбоксов
}

type WindowsButton struct{}	// Конкретный Продукт A1 - Windows-специфичная кнопка

func (b *WindowsButton) Render() string {	// реализует интерфейс Button для Windows
	return "Rendering Windows button"	// Windows-специфичная логика отрисовки
}

type WindowsCheckbox struct{}	// Конкретный Продукт B1 - Windows-специфичный чекбокс

func (c *WindowsCheckbox) Check() string {	// реализует интерфейс Checkbox для Windows
	return "Windows checkbox checked"	// Windows-специфичное поведение
}

type MacButton struct{}	// Конкретный Продукт A2 - Mac-специфичная кнопка

func (b *MacButton) Render() string {	// реализует интерфейс Button для Mac
	return "Rendering Mac button"	// Mac-специфичная логика отрисовки
}

type MacCheckbox struct{}	// Конкретный Продукт B2 - Mac-специфичный чекбокс

func (c *MacCheckbox) Check() string {	// реализует интерфейс Checkbox для Mac
	return "Mac checkbox checked"	// Mac-специфичное поведение
}

type WindowsFactory struct{}	// Конкретная Фабрика 1 - создаёт Windows семейство

func (f *WindowsFactory) CreateButton() Button {	// реализует GUIFactory для Windows
	return &WindowsButton{}	// возвращает Windows-специфичную кнопку
}

func (f *WindowsFactory) CreateCheckbox() Checkbox {	// реализует GUIFactory для Windows
	return &WindowsCheckbox{}	// возвращает Windows-специфичный чекбокс
}

type MacFactory struct{}	// Конкретная Фабрика 2 - создаёт Mac семейство

func (f *MacFactory) CreateButton() Button {	// реализует GUIFactory для Mac
	return &MacButton{}	// возвращает Mac-специфичную кнопку
}

func (f *MacFactory) CreateCheckbox() Checkbox {	// реализует GUIFactory для Mac
	return &MacCheckbox{}	// возвращает Mac-специфичный чекбокс
}

func GetFactory(osType string) GUIFactory {	// селектор фабрики на основе конфигурации
	switch osType {	// определяем какую фабрику создать
	case "mac":	// если запрошена Mac ОС
		return &MacFactory{}	// возвращаем Mac фабрику
	default:	// для Windows или любой другой ОС
		return &WindowsFactory{}	// по умолчанию Windows фабрика
	}
}`
		},
		uz: {
			title: 'Abstract Factory (Abstrakt Fabrika) Pattern',
			description: `Go tilida Abstract Factory patternini amalga oshiring — konkret klasslarni ko'rsatmasdan bog'liq ob'ektlar oilalarini yaratish uchun interfeys taqdim eting.

**Siz amalga oshirasiz:**

1. **GUIFactory interfeysi** - Bog'liq Button va Checkbox yaratadi
2. **Button interfeysi** - Render() metodi
3. **Checkbox interfeysi** - Check() metodi
4. **WindowsFactory** - Windows uslubidagi komponentlarni yaratadi
5. **MacFactory** - Mac uslubidagi komponentlarni yaratadi

**Foydalanish namunasi:**

\`\`\`go
// OS turiga qarab fabrikani olish
factory := GetFactory("windows")

// Bir oiladan bog'liq komponentlarni yaratish
button := factory.CreateButton()
checkbox := factory.CreateCheckbox()

// Komponentlardan foydalanish - ular kafolatlangan moslikka ega
button.Render()   // "Rendering Windows button"
checkbox.Check()  // "Windows checkbox checked"

// Mac fabrikasiga o'tish
macFactory := GetFactory("mac")
macButton := macFactory.CreateButton()
macButton.Render() // "Rendering Mac button"
\`\`\``,
			hint1: `Har bir konkret mahsulot o'z interfeysini oddiy string qaytarish bilan amalga oshiradi. Windows komponentlari "Windows", Mac komponentlari "Mac" qaytaradi. Asosiy nuqta shundaki, har bir fabrika BITTA oiladan mahsulotlar yaratadi — Windows fabrikasi faqat Windows komponentlarini yaratadi.`,
			hint2: `GetFactory OS turiga qarab mos fabrikani qaytarish uchun switch ishlatadi. Har bir fabrikaning CreateButton va CreateCheckbox metodlari tegishli konkret mahsulotlarning yangi nusxalarini qaytaradi. Bu Windows fabrikasi tasodifan Mac tugmasini yaratmasligini ta'minlaydi.`,
			whyItMatters: `**1. Abstract Factory nima uchun kerak**

Abstract Factory birga ishlashi kerak bo'lgan BOG'LIQ ob'ektlar OILALARIni yaratish muammosini hal qiladi. Factory Method alohida mahsulotlar yaratishidan farqli, Abstract Factory bitta fabrika tomonidan yaratilgan barcha mahsulotlarning mosligini ta'minlaydi.

**U hal qiladigan muammo:**

\`\`\`go
// Abstract Factory SIZ - mos kelmaydigan komponentlarni aralashtirish
func CreateUI(osType string) {
    var button Button
    var checkbox Checkbox

    // Xavfli: Windows tugmasini Mac checkbox bilan aralashtirib yuborish mumkin!
    if osType == "windows" {
        button = &WindowsButton{}
    } else {
        button = &MacButton{}
    }

    // Xato: dasturchi boshqa shart ishlatishi mumkin
    if osType == "mac" {  // Xato! Boshqa shart
        checkbox = &MacCheckbox{}
    } else {
        checkbox = &WindowsCheckbox{}
    }

    // Endi bizda mos kelmaydigan UI komponentlar!
}
\`\`\`

**Abstract Factory BILAN:**

\`\`\`go
// Xavfsiz: fabrika barcha komponentlarning bir oiladan ekanligini kafolatlaydi
func CreateUI(osType string) {
    factory := GetFactory(osType)  // kerakli fabrikani olish

    // Barcha komponentlar kafolatlangan moslikka ega
    button := factory.CreateButton()      // bir oila
    checkbox := factory.CreateCheckbox()  // bir oila

    // Windows bilan Mac komponentlarini aralashtirish mumkin emas
}
\`\`\`

**2. Go'da real hayotiy misollar**

**Ma'lumotlar bazasi drayverlarini abstraksiya qilish:**

\`\`\`go
// Ma'lumotlar bazasi operatsiyalari uchun Abstract Factory
type DBFactory interface {
    CreateConnection() Connection
    CreateQueryBuilder() QueryBuilder
    CreateTransaction() Transaction
}

// PostgreSQL oilasi
type PostgresFactory struct{}
func (f *PostgresFactory) CreateConnection() Connection {
    return &PostgresConnection{}
}
func (f *PostgresFactory) CreateQueryBuilder() QueryBuilder {
    return &PostgresQueryBuilder{}  // $1, $2 placeholderlardan foydalanadi
}

// MySQL oilasi
type MySQLFactory struct{}
func (f *MySQLFactory) CreateConnection() Connection {
    return &MySQLConnection{}
}
func (f *MySQLFactory) CreateQueryBuilder() QueryBuilder {
    return &MySQLQueryBuilder{}  // ? placeholderlardan foydalanadi
}
\`\`\`

**Bulut provayderlarini abstraksiya qilish:**

\`\`\`go
// Bulut xizmatlari uchun Abstract Factory
type CloudFactory interface {
    CreateStorage() ObjectStorage
    CreateCompute() ComputeService
    CreateDatabase() ManagedDB
}

// AWS oilasi - barcha xizmatlar birga ishlaydi
type AWSFactory struct{}
func (f *AWSFactory) CreateStorage() ObjectStorage {
    return &S3Storage{}  // AWS IAM rollardan foydalanadi
}

// GCP oilasi - barcha xizmatlar birga ishlaydi
type GCPFactory struct{}
func (f *GCPFactory) CreateStorage() ObjectStorage {
    return &GCSStorage{}  // GCP service accountlardan foydalanadi
}
\`\`\`

**3. Production pattern**

\`\`\`go
package factory

import "os"

// Mahsulot interfeyslari
type Logger interface {
    Log(message string)
    Error(err error)
}

type Metrics interface {
    Count(name string)
    Gauge(name string, value float64)
}

type Tracer interface {
    StartSpan(name string) Span
}

// Kuzatuv fabrikasi - mos kuzatuv vositalarini yaratadi
type ObservabilityFactory interface {
    CreateLogger() Logger
    CreateMetrics() Metrics
    CreateTracer() Tracer
}

// Production fabrikasi - haqiqiy xizmatlardan foydalanadi
type ProductionFactory struct {
    serviceName string
}

func NewProductionFactory(serviceName string) *ProductionFactory {
    return &ProductionFactory{serviceName: serviceName}
}

func (f *ProductionFactory) CreateLogger() Logger {
    return NewDatadogLogger(f.serviceName)
}

func (f *ProductionFactory) CreateMetrics() Metrics {
    return NewDatadogMetrics(f.serviceName)
}

func (f *ProductionFactory) CreateTracer() Tracer {
    return NewDatadogTracer(f.serviceName)
}

// Development fabrikasi - konsol/mock implementatsiyalardan foydalanadi
type DevelopmentFactory struct{}

func (f *DevelopmentFactory) CreateLogger() Logger {
    return NewConsoleLogger()
}

func (f *DevelopmentFactory) CreateMetrics() Metrics {
    return NewNoOpMetrics()  // dev da metrikalar o'chirilgan
}

func (f *DevelopmentFactory) CreateTracer() Tracer {
    return NewNoOpTracer()  // dev da tracing o'chirilgan
}

// Fabrika tanlash
func GetObservabilityFactory(serviceName string) ObservabilityFactory {
    if os.Getenv("ENV") == "production" {
        return NewProductionFactory(serviceName)
    }
    return &DevelopmentFactory{}
}
\`\`\`

**4. Keng tarqalgan xatolar**

\`\`\`go
// XATO 1: Fabrikadan tashqarida mahsulotlar yaratish
factory := GetFactory("windows")
button := factory.CreateButton()
checkbox := &MacCheckbox{}  // Noto'g'ri! Fabrikani chetlab o'tish, izchillikni buzish

// XATO 2: Fabrikaga bog'liq bo'lmagan mahsulotlarni qo'shish
type GUIFactory interface {
    CreateButton() Button
    CreateCheckbox() Checkbox
    CreateHTTPClient() HTTPClient  // Noto'g'ri! GUI oilasiga tegishli emas
}

// XATO 3: Aralash oilalar yaratadigav fabrika
type BadFactory struct {
    buttonOS   string
    checkboxOS string
}
func (f *BadFactory) CreateButton() Button {
    if f.buttonOS == "mac" { return &MacButton{} }
    return &WindowsButton{}
}
func (f *BadFactory) CreateCheckbox() Checkbox {
    // Boshqa OS dan checkbox yaratishi mumkin! Maqsadni buzadi
    if f.checkboxOS == "mac" { return &MacCheckbox{} }
    return &WindowsCheckbox{}
}

// TO'G'RI: Har bir fabrika faqat o'z oilasini yaratadi
type WindowsFactory struct{}  // faqat Windows komponentlarini yaratadi
type MacFactory struct{}      // faqat Mac komponentlarini yaratadi
\`\`\``,
			solutionCode: `package patterns

type Button interface {	// Abstrakt Mahsulot A - barcha tugmalar uchun umumiy interfeys
	Render() string	// barcha tugmalar o'zini chizishni bilishi kerak
}

type Checkbox interface {	// Abstrakt Mahsulot B - barcha checkboxlar uchun umumiy interfeys
	Check() string	// barcha checkboxlar o'zini belgilashni bilishi kerak
}

type GUIFactory interface {	// Abstrakt Fabrika - bog'liq mahsulotlar oilalarini yaratadi
	CreateButton() Button	// tugmalar yaratish uchun fabrika metodi
	CreateCheckbox() Checkbox	// checkboxlar yaratish uchun fabrika metodi
}

type WindowsButton struct{}	// Konkret Mahsulot A1 - Windows-maxsus tugma

func (b *WindowsButton) Render() string {	// Windows uchun Button interfeysini amalga oshiradi
	return "Rendering Windows button"	// Windows-maxsus chizish logikasi
}

type WindowsCheckbox struct{}	// Konkret Mahsulot B1 - Windows-maxsus checkbox

func (c *WindowsCheckbox) Check() string {	// Windows uchun Checkbox interfeysini amalga oshiradi
	return "Windows checkbox checked"	// Windows-maxsus xatti-harakat
}

type MacButton struct{}	// Konkret Mahsulot A2 - Mac-maxsus tugma

func (b *MacButton) Render() string {	// Mac uchun Button interfeysini amalga oshiradi
	return "Rendering Mac button"	// Mac-maxsus chizish logikasi
}

type MacCheckbox struct{}	// Konkret Mahsulot B2 - Mac-maxsus checkbox

func (c *MacCheckbox) Check() string {	// Mac uchun Checkbox interfeysini amalga oshiradi
	return "Mac checkbox checked"	// Mac-maxsus xatti-harakat
}

type WindowsFactory struct{}	// Konkret Fabrika 1 - Windows oilasini yaratadi

func (f *WindowsFactory) CreateButton() Button {	// Windows uchun GUIFactory ni amalga oshiradi
	return &WindowsButton{}	// Windows-maxsus tugmani qaytaradi
}

func (f *WindowsFactory) CreateCheckbox() Checkbox {	// Windows uchun GUIFactory ni amalga oshiradi
	return &WindowsCheckbox{}	// Windows-maxsus checkboxni qaytaradi
}

type MacFactory struct{}	// Konkret Fabrika 2 - Mac oilasini yaratadi

func (f *MacFactory) CreateButton() Button {	// Mac uchun GUIFactory ni amalga oshiradi
	return &MacButton{}	// Mac-maxsus tugmani qaytaradi
}

func (f *MacFactory) CreateCheckbox() Checkbox {	// Mac uchun GUIFactory ni amalga oshiradi
	return &MacCheckbox{}	// Mac-maxsus checkboxni qaytaradi
}

func GetFactory(osType string) GUIFactory {	// konfiguratsiyaga asoslangan fabrika tanlash
	switch osType {	// qaysi fabrikani yaratishni aniqlash
	case "mac":	// agar Mac OS so'ralsa
		return &MacFactory{}	// Mac komponentlarini yaratadigan Mac fabrikasini qaytarish
	default:	// Windows yoki boshqa OS uchun
		return &WindowsFactory{}	// standart Windows fabrikasi
	}
}`
		}
	}
};

export default task;
