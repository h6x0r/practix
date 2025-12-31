import { Task } from '../../../../../../../types';

export const task: Task = {
	slug: 'go-dp-factory-method',
	title: 'Factory Method Pattern',
	difficulty: 'medium',
	tags: ['go', 'design-patterns', 'creational', 'factory'],
	estimatedTime: '30m',
	isPremium: false,
	youtubeUrl: '',
	description: `Implement the Factory Method pattern in Go - define an interface for creating objects, but let subclasses decide which class to instantiate.

**You will implement:**

1. **Document interface** - Common interface with Print() method
2. **PDFDocument struct** - PDF implementation
3. **WordDocument struct** - Word implementation
4. **DocumentFactory interface** - Factory with CreateDocument() method
5. **PDFFactory, WordFactory structs** - Concrete factories

**Key Concepts:**
- **Product Interface**: Defines the common contract for all products
- **Concrete Products**: Implement the product interface
- **Creator Interface**: Declares the factory method
- **Concrete Creators**: Override the factory method to return specific products

**Example Usage:**

\`\`\`go
// Create factories for different document types
pdfFactory := &PDFFactory{}
wordFactory := &WordFactory{}

// Create documents through factories (client doesn't know concrete types)
pdf := pdfFactory.CreateDocument()
fmt.Println(pdf.Print()) // "Printing PDF document"

word := wordFactory.CreateDocument()
fmt.Println(word.Print()) // "Printing Word document"

// Polymorphic usage - work with any factory
func processDocument(factory DocumentFactory) {
    doc := factory.CreateDocument()
    fmt.Println(doc.Print())
}
processDocument(pdfFactory)  // "Printing PDF document"
processDocument(wordFactory) // "Printing Word document"
\`\`\`

**When to use Factory Method:**
- When you don't know beforehand the exact types of objects
- When you want to provide users a way to extend internal components
- When you want to save system resources by reusing existing objects
- When you need to decouple object creation from usage`,
	initialCode: `package patterns

type Document interface {
}

type PDFDocument struct{}

func (p *PDFDocument) Print() string {
}

type WordDocument struct{}

func (w *WordDocument) Print() string {
}

type DocumentFactory interface {
}

type PDFFactory struct{}

func (f *PDFFactory) CreateDocument() Document {
}

type WordFactory struct{}

func (f *WordFactory) CreateDocument() Document {
}`,
	solutionCode: `package patterns

type Document interface {	// Product interface - defines what all documents can do
	Print() string	// common method that all document types must implement
}

type PDFDocument struct{}	// Concrete Product - specific implementation for PDF

func (p *PDFDocument) Print() string {	// implements Document interface
	return "Printing PDF document"	// PDF-specific behavior
}

type WordDocument struct{}	// Concrete Product - specific implementation for Word

func (w *WordDocument) Print() string {	// implements Document interface
	return "Printing Word document"	// Word-specific behavior
}

type DocumentFactory interface {	// Creator interface - declares the factory method
	CreateDocument() Document	// factory method returns Product interface, not concrete type
}

type PDFFactory struct{}	// Concrete Creator - creates PDF documents

func (f *PDFFactory) CreateDocument() Document {	// implements DocumentFactory
	return &PDFDocument{}	// creates and returns specific product type
}

type WordFactory struct{}	// Concrete Creator - creates Word documents

func (f *WordFactory) CreateDocument() Document {	// implements DocumentFactory
	return &WordDocument{}	// creates and returns specific product type
}`,
	hint1: `Each Print() method should return a string describing the document type. PDFDocument.Print() returns "Printing PDF document".`,
	hint2: `Each factory's CreateDocument() should return a pointer to a new instance of its document type. For example, PDFFactory returns &PDFDocument{}.`,
	whyItMatters: `Factory Method is one of the most commonly used design patterns that enables loose coupling and extensibility.

**Why Factory Method Matters:**

**1. Decoupling Creation from Usage**
Client code works with interfaces, not concrete types:

\`\`\`go
// WITHOUT Factory Method - tight coupling
func processReport(reportType string) {
    var doc Document
    if reportType == "pdf" {	// client knows all concrete types
        doc = &PDFDocument{}	// direct instantiation
    } else if reportType == "word" {
        doc = &WordDocument{}
    }
    doc.Print()
}

// WITH Factory Method - loose coupling
func processReport(factory DocumentFactory) {	// client only knows interface
    doc := factory.CreateDocument()	// creation delegated to factory
    doc.Print()	// works with any document type
}
\`\`\`

**2. Open/Closed Principle**
Add new types without modifying existing code:

\`\`\`go
// Adding new document type - NO changes to existing code!
type ExcelDocument struct{}

func (e *ExcelDocument) Print() string {
    return "Printing Excel spreadsheet"
}

type ExcelFactory struct{}

func (f *ExcelFactory) CreateDocument() Document {
    return &ExcelDocument{}	// new factory for new type
}

// Existing code works unchanged
processReport(&ExcelFactory{})	// just pass new factory
\`\`\`

**3. Centralized Object Creation**
Configuration, validation, caching in one place:

\`\`\`go
type CachedPDFFactory struct {
    cache map[string]*PDFDocument	// reuse instances
}

func (f *CachedPDFFactory) CreateDocument() Document {
    if cached, ok := f.cache["default"]; ok {
        return cached	// return cached instance
    }
    doc := &PDFDocument{}
    f.cache["default"] = doc	// cache for reuse
    return doc
}
\`\`\`

**Real-World Examples in Go:**

\`\`\`go
// 1. database/sql - Driver registration is factory pattern
import _ "github.com/lib/pq"	// registers postgres driver
db, _ := sql.Open("postgres", connStr)	// factory creates DB-specific connection

// 2. net/http - Handler creation
http.HandleFunc("/pdf", func(w http.ResponseWriter, r *http.Request) {
    factory := &PDFFactory{}
    doc := factory.CreateDocument()	// create appropriate document
    w.Write([]byte(doc.Print()))
})

// 3. encoding/json vs encoding/xml - same Encoder interface
json.NewEncoder(w)	// JSON encoder factory
xml.NewEncoder(w)	// XML encoder factory
\`\`\`

**Production Pattern:**
\`\`\`go
// Registry-based factory for plugin architecture
type DocumentRegistry struct {
    factories map[string]DocumentFactory
}

func NewDocumentRegistry() *DocumentRegistry {
    return &DocumentRegistry{
        factories: make(map[string]DocumentFactory),
    }
}

func (r *DocumentRegistry) Register(name string, factory DocumentFactory) {
    r.factories[name] = factory	// register factory by name
}

func (r *DocumentRegistry) Create(name string) (Document, error) {
    factory, ok := r.factories[name]
    if !ok {
        return nil, fmt.Errorf("unknown document type: %s", name)
    }
    return factory.CreateDocument(), nil	// delegate to registered factory
}

// Usage
registry := NewDocumentRegistry()
registry.Register("pdf", &PDFFactory{})
registry.Register("word", &WordFactory{})
doc, _ := registry.Create("pdf")	// create by name
\`\`\`

**Common Mistakes to Avoid:**
- Creating factory for single product type (over-engineering)
- Returning concrete types instead of interfaces from factory
- Not using factory when object creation is complex
- Mixing factory logic with business logic`,
	order: 1,
	testCode: `package patterns

import (
	"testing"
)

// Test1: PDFDocument.Print returns correct string
func Test1(t *testing.T) {
	pdf := &PDFDocument{}
	if pdf.Print() != "Printing PDF document" {
		t.Error("PDFDocument.Print should return 'Printing PDF document'")
	}
}

// Test2: WordDocument.Print returns correct string
func Test2(t *testing.T) {
	word := &WordDocument{}
	if word.Print() != "Printing Word document" {
		t.Error("WordDocument.Print should return 'Printing Word document'")
	}
}

// Test3: PDFFactory creates PDFDocument
func Test3(t *testing.T) {
	factory := &PDFFactory{}
	doc := factory.CreateDocument()
	if doc.Print() != "Printing PDF document" {
		t.Error("PDFFactory should create PDFDocument")
	}
}

// Test4: WordFactory creates WordDocument
func Test4(t *testing.T) {
	factory := &WordFactory{}
	doc := factory.CreateDocument()
	if doc.Print() != "Printing Word document" {
		t.Error("WordFactory should create WordDocument")
	}
}

// Test5: PDFDocument implements Document interface
func Test5(t *testing.T) {
	var doc Document = &PDFDocument{}
	if doc == nil {
		t.Error("PDFDocument should implement Document interface")
	}
}

// Test6: WordDocument implements Document interface
func Test6(t *testing.T) {
	var doc Document = &WordDocument{}
	if doc == nil {
		t.Error("WordDocument should implement Document interface")
	}
}

// Test7: PDFFactory implements DocumentFactory interface
func Test7(t *testing.T) {
	var factory DocumentFactory = &PDFFactory{}
	if factory == nil {
		t.Error("PDFFactory should implement DocumentFactory interface")
	}
}

// Test8: WordFactory implements DocumentFactory interface
func Test8(t *testing.T) {
	var factory DocumentFactory = &WordFactory{}
	if factory == nil {
		t.Error("WordFactory should implement DocumentFactory interface")
	}
}

// Test9: Factory creates new instance each time
func Test9(t *testing.T) {
	factory := &PDFFactory{}
	doc1 := factory.CreateDocument()
	doc2 := factory.CreateDocument()
	if doc1 == doc2 {
		t.Error("Factory should create new instance each time")
	}
}

// Test10: Polymorphic factory usage
func Test10(t *testing.T) {
	factories := []DocumentFactory{&PDFFactory{}, &WordFactory{}}
	results := []string{"Printing PDF document", "Printing Word document"}
	for i, factory := range factories {
		doc := factory.CreateDocument()
		if doc.Print() != results[i] {
			t.Errorf("Expected %s", results[i])
		}
	}
}
`,
	translations: {
		ru: {
			title: 'Паттерн Factory Method (Фабричный метод)',
			description: `Реализуйте паттерн Factory Method на Go — определите интерфейс для создания объектов, но позвольте подклассам решать, какой класс инстанцировать.

**Вы реализуете:**

1. **Document interface** — Общий интерфейс с методом Print()
2. **PDFDocument struct** — Реализация PDF
3. **WordDocument struct** — Реализация Word
4. **DocumentFactory interface** — Фабрика с методом CreateDocument()
5. **PDFFactory, WordFactory structs** — Конкретные фабрики

**Ключевые концепции:**
- **Интерфейс продукта**: Определяет общий контракт для всех продуктов
- **Конкретные продукты**: Реализуют интерфейс продукта
- **Интерфейс создателя**: Объявляет фабричный метод
- **Конкретные создатели**: Переопределяют фабричный метод для возврата конкретных продуктов

**Пример использования:**

\`\`\`go
// Создаём фабрики для разных типов документов
pdfFactory := &PDFFactory{}
wordFactory := &WordFactory{}

// Создаём документы через фабрики (клиент не знает конкретных типов)
pdf := pdfFactory.CreateDocument()
fmt.Println(pdf.Print()) // "Printing PDF document"

word := wordFactory.CreateDocument()
fmt.Println(word.Print()) // "Printing Word document"

// Полиморфное использование — работает с любой фабрикой
func processDocument(factory DocumentFactory) {
    doc := factory.CreateDocument()
    fmt.Println(doc.Print())
}
processDocument(pdfFactory)  // "Printing PDF document"
processDocument(wordFactory) // "Printing Word document"
\`\`\`

**Когда использовать Factory Method:**
- Когда заранее неизвестны точные типы объектов
- Когда нужно дать пользователям возможность расширять компоненты
- Когда нужно экономить ресурсы переиспользованием объектов
- Когда нужно отделить создание объектов от их использования`,
			hint1: `Каждый метод Print() должен возвращать строку, описывающую тип документа. PDFDocument.Print() возвращает "Printing PDF document".`,
			hint2: `CreateDocument() каждой фабрики должен возвращать указатель на новый экземпляр своего типа документа. Например, PDFFactory возвращает &PDFDocument{}.`,
			whyItMatters: `Factory Method — один из самых используемых паттернов, обеспечивающий слабую связанность и расширяемость.

**Почему Factory Method важен:**

**1. Отделение создания от использования**
Клиентский код работает с интерфейсами, а не с конкретными типами:

\`\`\`go
// БЕЗ Factory Method — жёсткая связанность
func processReport(reportType string) {
    var doc Document
    if reportType == "pdf" {	// клиент знает все конкретные типы
        doc = &PDFDocument{}	// прямое создание экземпляра
    } else if reportType == "word" {
        doc = &WordDocument{}
    }
    doc.Print()
}

// С Factory Method — слабая связанность
func processReport(factory DocumentFactory) {	// клиент знает только интерфейс
    doc := factory.CreateDocument()	// создание делегировано фабрике
    doc.Print()	// работает с любым типом документа
}
\`\`\`

**2. Принцип открытости/закрытости**
Добавление новых типов без изменения существующего кода:

\`\`\`go
// Добавляем новый тип документа — БЕЗ изменений существующего кода!
type ExcelDocument struct{}

func (e *ExcelDocument) Print() string {
    return "Printing Excel spreadsheet"
}

type ExcelFactory struct{}

func (f *ExcelFactory) CreateDocument() Document {
    return &ExcelDocument{}	// новая фабрика для нового типа
}

// Существующий код работает без изменений
processReport(&ExcelFactory{})	// просто передаём новую фабрику
\`\`\`

**3. Централизованное создание объектов**
Конфигурация, валидация, кэширование в одном месте:

\`\`\`go
type CachedPDFFactory struct {
    cache map[string]*PDFDocument	// переиспользование экземпляров
}

func (f *CachedPDFFactory) CreateDocument() Document {
    if cached, ok := f.cache["default"]; ok {
        return cached	// возвращаем закэшированный экземпляр
    }
    doc := &PDFDocument{}
    f.cache["default"] = doc	// кэшируем для переиспользования
    return doc
}
\`\`\`

**Реальные примеры в Go:**

\`\`\`go
// 1. database/sql — регистрация драйверов это паттерн фабрика
import _ "github.com/lib/pq"	// регистрирует postgres драйвер
db, _ := sql.Open("postgres", connStr)	// фабрика создаёт DB-специфичное подключение

// 2. net/http — создание обработчиков
http.HandleFunc("/pdf", func(w http.ResponseWriter, r *http.Request) {
    factory := &PDFFactory{}
    doc := factory.CreateDocument()	// создаём подходящий документ
    w.Write([]byte(doc.Print()))
})

// 3. encoding/json vs encoding/xml — один интерфейс Encoder
json.NewEncoder(w)	// фабрика JSON энкодера
xml.NewEncoder(w)	// фабрика XML энкодера
\`\`\`

**Продакшен паттерн:**
\`\`\`go
// Фабрика на основе реестра для плагинной архитектуры
type DocumentRegistry struct {
    factories map[string]DocumentFactory
}

func NewDocumentRegistry() *DocumentRegistry {
    return &DocumentRegistry{
        factories: make(map[string]DocumentFactory),
    }
}

func (r *DocumentRegistry) Register(name string, factory DocumentFactory) {
    r.factories[name] = factory	// регистрируем фабрику по имени
}

func (r *DocumentRegistry) Create(name string) (Document, error) {
    factory, ok := r.factories[name]
    if !ok {
        return nil, fmt.Errorf("неизвестный тип документа: %s", name)
    }
    return factory.CreateDocument(), nil	// делегируем зарегистрированной фабрике
}

// Использование
registry := NewDocumentRegistry()
registry.Register("pdf", &PDFFactory{})
registry.Register("word", &WordFactory{})
doc, _ := registry.Create("pdf")	// создаём по имени
\`\`\`

**Распространённые ошибки:**
- Создание фабрики для единственного типа продукта (избыточность)
- Возврат конкретных типов вместо интерфейсов из фабрики
- Неиспользование фабрики когда создание объектов сложное
- Смешивание логики фабрики с бизнес-логикой`,
			solutionCode: `package patterns

type Document interface {	// Интерфейс продукта — определяет что могут делать все документы
	Print() string	// общий метод, который должны реализовать все типы документов
}

type PDFDocument struct{}	// Конкретный продукт — специфичная реализация для PDF

func (p *PDFDocument) Print() string {	// реализует интерфейс Document
	return "Printing PDF document"	// поведение специфичное для PDF
}

type WordDocument struct{}	// Конкретный продукт — специфичная реализация для Word

func (w *WordDocument) Print() string {	// реализует интерфейс Document
	return "Printing Word document"	// поведение специфичное для Word
}

type DocumentFactory interface {	// Интерфейс создателя — объявляет фабричный метод
	CreateDocument() Document	// фабричный метод возвращает интерфейс Product, не конкретный тип
}

type PDFFactory struct{}	// Конкретный создатель — создаёт PDF документы

func (f *PDFFactory) CreateDocument() Document {	// реализует DocumentFactory
	return &PDFDocument{}	// создаёт и возвращает конкретный тип продукта
}

type WordFactory struct{}	// Конкретный создатель — создаёт Word документы

func (f *WordFactory) CreateDocument() Document {	// реализует DocumentFactory
	return &WordDocument{}	// создаёт и возвращает конкретный тип продукта
}`
		},
		uz: {
			title: 'Factory Method (Fabrika Metodi) Pattern',
			description: `Go tilida Factory Method patternini amalga oshiring — ob'ektlar yaratish uchun interfeys aniqlang, lekin qaysi klassni instansiyalashni subklasslar hal qilsin.

**Siz amalga oshirasiz:**

1. **Document interface** — Print() metodi bilan umumiy interfeys
2. **PDFDocument struct** — PDF amalga oshirish
3. **WordDocument struct** — Word amalga oshirish
4. **DocumentFactory interface** — CreateDocument() metodi bilan fabrika
5. **PDFFactory, WordFactory structs** — Konkret fabrikalar

**Asosiy tushunchalar:**
- **Mahsulot interfeysi**: Barcha mahsulotlar uchun umumiy shartnomani belgilaydi
- **Konkret mahsulotlar**: Mahsulot interfeysini amalga oshiradi
- **Yaratuvchi interfeysi**: Fabrika metodini e'lon qiladi
- **Konkret yaratuvchilar**: Muayyan mahsulotlarni qaytarish uchun fabrika metodini qayta aniqlaydi

**Foydalanish misoli:**

\`\`\`go
// Turli dokument turlari uchun fabrikalar yaratamiz
pdfFactory := &PDFFactory{}
wordFactory := &WordFactory{}

// Fabrikalar orqali dokumentlar yaratamiz (klient konkret turlarni bilmaydi)
pdf := pdfFactory.CreateDocument()
fmt.Println(pdf.Print()) // "Printing PDF document"

word := wordFactory.CreateDocument()
fmt.Println(word.Print()) // "Printing Word document"

// Polimorfik foydalanish — istalgan fabrika bilan ishlaydi
func processDocument(factory DocumentFactory) {
    doc := factory.CreateDocument()
    fmt.Println(doc.Print())
}
processDocument(pdfFactory)  // "Printing PDF document"
processDocument(wordFactory) // "Printing Word document"
\`\`\`

**Factory Method qachon ishlatiladi:**
- Ob'ektlarning aniq turlari oldindan noma'lum bo'lganda
- Foydalanuvchilarga ichki komponentlarni kengaytirish imkoniyatini berishni xohlaganingizda
- Mavjud ob'ektlarni qayta ishlatish orqali resurslarni tejashni xohlaganingizda
- Ob'ektlarni yaratishni ulardan foydalanishdan ajratish kerak bo'lganda`,
			hint1: `Har bir Print() metodi dokument turini tavsiflovchi string qaytarishi kerak. PDFDocument.Print() "Printing PDF document" qaytaradi.`,
			hint2: `Har bir fabrikaning CreateDocument() metodi o'z dokument turiga yangi nusxa ko'rsatkichini qaytarishi kerak. Masalan, PDFFactory &PDFDocument{} qaytaradi.`,
			whyItMatters: `Factory Method — zaif bog'lanish va kengayuvchanlikni ta'minlovchi eng ko'p ishlatiladigan patternlardan biri.

**Factory Method nima uchun muhim:**

**1. Yaratishni foydalanishdan ajratish**
Klient kodi interfeyslar bilan ishlaydi, konkret turlar bilan emas:

\`\`\`go
// Factory Method SIZ — qattiq bog'lanish
func processReport(reportType string) {
    var doc Document
    if reportType == "pdf" {	// klient barcha konkret turlarni biladi
        doc = &PDFDocument{}	// to'g'ridan-to'g'ri instansiyalash
    } else if reportType == "word" {
        doc = &WordDocument{}
    }
    doc.Print()
}

// Factory Method BILAN — zaif bog'lanish
func processReport(factory DocumentFactory) {	// klient faqat interfeysni biladi
    doc := factory.CreateDocument()	// yaratish fabrikaga topshiriladi
    doc.Print()	// istalgan dokument turi bilan ishlaydi
}
\`\`\`

**2. Ochiq/Yopiq prinsipi**
Mavjud kodni o'zgartirmasdan yangi turlar qo'shish:

\`\`\`go
// Yangi dokument turi qo'shish — mavjud kodga O'ZGARISHLAR YO'Q!
type ExcelDocument struct{}

func (e *ExcelDocument) Print() string {
    return "Printing Excel spreadsheet"
}

type ExcelFactory struct{}

func (f *ExcelFactory) CreateDocument() Document {
    return &ExcelDocument{}	// yangi tur uchun yangi fabrika
}

// Mavjud kod o'zgarishsiz ishlaydi
processReport(&ExcelFactory{})	// shunchaki yangi fabrikani uzatamiz
\`\`\`

**3. Markazlashtirilgan ob'ekt yaratish**
Konfiguratsiya, validatsiya, keshlash bir joyda:

\`\`\`go
type CachedPDFFactory struct {
    cache map[string]*PDFDocument	// nusxalarni qayta ishlatish
}

func (f *CachedPDFFactory) CreateDocument() Document {
    if cached, ok := f.cache["default"]; ok {
        return cached	// keshlangan nusxani qaytaramiz
    }
    doc := &PDFDocument{}
    f.cache["default"] = doc	// qayta ishlatish uchun keshlaymiz
    return doc
}
\`\`\`

**Go dagi real misollar:**

\`\`\`go
// 1. database/sql — drayverni ro'yxatdan o'tkazish fabrika patterni
import _ "github.com/lib/pq"	// postgres drayverni ro'yxatdan o'tkazadi
db, _ := sql.Open("postgres", connStr)	// fabrika DB-ga xos ulanish yaratadi

// 2. net/http — handlerlar yaratish
http.HandleFunc("/pdf", func(w http.ResponseWriter, r *http.Request) {
    factory := &PDFFactory{}
    doc := factory.CreateDocument()	// mos dokumentni yaratamiz
    w.Write([]byte(doc.Print()))
})

// 3. encoding/json vs encoding/xml — bir xil Encoder interfeysi
json.NewEncoder(w)	// JSON enkoder fabrikasi
xml.NewEncoder(w)	// XML enkoder fabrikasi
\`\`\`

**Ishlab chiqarish patterni:**
\`\`\`go
// Plagin arxitekturasi uchun reestrga asoslangan fabrika
type DocumentRegistry struct {
    factories map[string]DocumentFactory
}

func NewDocumentRegistry() *DocumentRegistry {
    return &DocumentRegistry{
        factories: make(map[string]DocumentFactory),
    }
}

func (r *DocumentRegistry) Register(name string, factory DocumentFactory) {
    r.factories[name] = factory	// fabrikani nom bilan ro'yxatdan o'tkazamiz
}

func (r *DocumentRegistry) Create(name string) (Document, error) {
    factory, ok := r.factories[name]
    if !ok {
        return nil, fmt.Errorf("noma'lum dokument turi: %s", name)
    }
    return factory.CreateDocument(), nil	// ro'yxatdan o'tgan fabrikaga topshiramiz
}

// Foydalanish
registry := NewDocumentRegistry()
registry.Register("pdf", &PDFFactory{})
registry.Register("word", &WordFactory{})
doc, _ := registry.Create("pdf")	// nom bo'yicha yaratamiz
\`\`\`

**Oldini olish kerak bo'lgan xatolar:**
- Yagona mahsulot turi uchun fabrika yaratish (ortiqcha murakkablik)
- Fabrikadan interfeyslar o'rniga konkret turlarni qaytarish
- Ob'ekt yaratish murakkab bo'lganda fabrikadan foydalanmaslik
- Fabrika mantiqini biznes-mantiq bilan aralashtirish`,
			solutionCode: `package patterns

type Document interface {	// Mahsulot interfeysi — barcha dokumentlar nima qila olishini belgilaydi
	Print() string	// barcha dokument turlari amalga oshirishi kerak bo'lgan umumiy metod
}

type PDFDocument struct{}	// Konkret mahsulot — PDF uchun maxsus amalga oshirish

func (p *PDFDocument) Print() string {	// Document interfeysini amalga oshiradi
	return "Printing PDF document"	// PDF ga xos xatti-harakat
}

type WordDocument struct{}	// Konkret mahsulot — Word uchun maxsus amalga oshirish

func (w *WordDocument) Print() string {	// Document interfeysini amalga oshiradi
	return "Printing Word document"	// Word ga xos xatti-harakat
}

type DocumentFactory interface {	// Yaratuvchi interfeysi — fabrika metodini e'lon qiladi
	CreateDocument() Document	// fabrika metodi konkret tur emas, Product interfeysini qaytaradi
}

type PDFFactory struct{}	// Konkret yaratuvchi — PDF dokumentlarni yaratadi

func (f *PDFFactory) CreateDocument() Document {	// DocumentFactory ni amalga oshiradi
	return &PDFDocument{}	// konkret mahsulot turini yaratadi va qaytaradi
}

type WordFactory struct{}	// Konkret yaratuvchi — Word dokumentlarni yaratadi

func (f *WordFactory) CreateDocument() Document {	// DocumentFactory ni amalga oshiradi
	return &WordDocument{}	// konkret mahsulot turini yaratadi va qaytaradi
}`
		}
	}
};

export default task;
