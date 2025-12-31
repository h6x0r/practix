import { Task } from '../../../../../../../types';

export const task: Task = {
	slug: 'go-dp-adapter',
	title: 'Adapter Pattern',
	difficulty: 'medium',
	tags: ['go', 'design-patterns', 'structural', 'adapter'],
	estimatedTime: '30m',
	isPremium: false,
	youtubeUrl: '',
	description: `Implement the Adapter pattern in Go - convert the interface of a class into another interface clients expect.

**You will implement:**

1. **Target interface** - The interface that the client expects
2. **Adaptee struct** - The existing class with incompatible interface
3. **Adapter struct** - Converts Adaptee's interface to Target

**Example Usage:**

\`\`\`go
// Old system uses XML format
oldSystem := &XMLDataSource{data: "<user>John</user>"}

// New system expects JSON through Target interface
adapter := &XMLToJSONAdapter{xmlSource: oldSystem}

// Client uses Target interface - doesn't know about XML
json := adapter.GetJSON() // {"xml": "<user>John</user>"}

// The adapter can be used anywhere Target is expected
func ProcessData(source Target) {
    data := source.GetJSON()
    fmt.Println("Processing:", data)
}

ProcessData(adapter) // Works seamlessly!
\`\`\``,
	initialCode: `package patterns

type Target interface {
}

type XMLDataSource struct {
	data string
}

func (x *XMLDataSource) GetXML() string {
	return x.data
}

type XMLToJSONAdapter struct {
	xmlSource *XMLDataSource
}

func (a *XMLToJSONAdapter) GetJSON() string {
}`,
	solutionCode: `package patterns

import "fmt"

type Target interface {	// Target interface - what the client expects
	GetJSON() string	// client wants JSON data
}

type XMLDataSource struct {	// Adaptee - existing class with incompatible interface
	data string	// stores XML data
}

func (x *XMLDataSource) GetXML() string {	// Adaptee's method - returns XML
	return x.data	// returns data in XML format
}

type XMLToJSONAdapter struct {	// Adapter - converts Adaptee to Target interface
	xmlSource *XMLDataSource	// holds reference to the adaptee
}

func (a *XMLToJSONAdapter) GetJSON() string {	// implements Target interface
	xml := a.xmlSource.GetXML()	// get data from adaptee using its interface
	return fmt.Sprintf("{\\"xml\\": \\"%s\\"}", xml)	// convert/wrap to JSON format
}`,
	hint1: `Create the adapter struct with a field to hold the XMLDataSource (the adaptee). The GetJSON method should call xmlSource.GetXML() to get the data, then convert/wrap it to JSON format.`,
	hint2: `Use fmt.Sprintf to create the JSON string. Remember to escape double quotes in Go strings with backslash: \\" produces a literal quote character in the output.`,
	whyItMatters: `**1. Why Adapter Exists**

Adapter pattern solves the interface incompatibility problem - when you have existing code (legacy systems, third-party libraries) with one interface, but your new code expects a different interface. Instead of modifying existing code (which may be impossible or risky), you create an adapter.

**The Problem It Solves:**

\`\`\`go
// WITHOUT Adapter - forced to change client code or adaptee
// Option 1: Modify client (breaks Open/Closed Principle)
func ProcessData(source interface{}) {
    switch s := source.(type) {
    case *JSONDataSource:
        data := s.GetJSON()
    case *XMLDataSource:
        // Must handle XML conversion inline
        data := convertXMLToJSON(s.GetXML())
    case *CSVDataSource:
        // More cases for each format...
        data := convertCSVToJSON(s.GetCSV())
    }
    // Ugly, violates OCP, hard to maintain
}

// Option 2: Modify adaptee (may be impossible for third-party code)
// Can't modify XMLDataSource from external library!
\`\`\`

**WITH Adapter:**

\`\`\`go
// Clean: client only knows about Target interface
func ProcessData(source Target) {
    data := source.GetJSON()  // works with any adapter
    process(data)
}

// Each data source has its own adapter
xmlAdapter := &XMLToJSONAdapter{xmlSource: xmlSource}
csvAdapter := &CSVToJSONAdapter{csvSource: csvSource}
yamlAdapter := &YAMLToJSONAdapter{yamlSource: yamlSource}

// All work through the same interface
ProcessData(xmlAdapter)   // adapts XML
ProcessData(csvAdapter)   // adapts CSV
ProcessData(yamlAdapter)  // adapts YAML
\`\`\`

**2. Real-World Examples in Go**

**Database Driver Adapter:**

\`\`\`go
// Target interface your app expects
type Database interface {
    Query(sql string) ([]Row, error)
    Execute(sql string) error
}

// Adaptee: Legacy MongoDB client
type MongoClient struct {
    // MongoDB uses different API
}

func (m *MongoClient) Find(collection string, filter map[string]interface{}) []Document {
    // MongoDB-specific implementation
}

// Adapter: Makes MongoDB look like SQL database
type MongoDBAdapter struct {
    client *MongoClient
}

func (a *MongoDBAdapter) Query(sql string) ([]Row, error) {
    // Parse SQL and convert to MongoDB query
    collection, filter := parseSQLToMongo(sql)
    docs := a.client.Find(collection, filter)
    return convertDocsToRows(docs), nil
}

func (a *MongoDBAdapter) Execute(sql string) error {
    // Convert SQL INSERT/UPDATE to MongoDB operations
    return a.client.Execute(convertSQLToMongoOp(sql))
}
\`\`\`

**HTTP Client Adapter:**

\`\`\`go
// Target: Your app's HTTP client interface
type HTTPClient interface {
    Get(url string) (*Response, error)
    Post(url string, body []byte) (*Response, error)
}

// Adaptee: Third-party HTTP library (e.g., resty, fasthttp)
type FastHTTPClient struct {
    // Different API
}

func (f *FastHTTPClient) DoRequest(method, url string, body []byte) (int, []byte, error) {
    // FastHTTP implementation
}

// Adapter
type FastHTTPAdapter struct {
    client *FastHTTPClient
}

func (a *FastHTTPAdapter) Get(url string) (*Response, error) {
    status, body, err := a.client.DoRequest("GET", url, nil)
    return &Response{Status: status, Body: body}, err
}

func (a *FastHTTPAdapter) Post(url string, body []byte) (*Response, error) {
    status, respBody, err := a.client.DoRequest("POST", url, body)
    return &Response{Status: status, Body: respBody}, err
}
\`\`\`

**3. Production Pattern - Bidirectional Adapter**

\`\`\`go
package adapter

import (
    "encoding/json"
    "encoding/xml"
)

// Target interface for modern systems
type JSONProcessor interface {
    ProcessJSON(data []byte) error
    GetJSON() ([]byte, error)
}

// Adaptee - legacy XML system
type XMLSystem struct {
    data []byte
}

func (x *XMLSystem) ProcessXML(data []byte) error {
    x.data = data
    return nil
}

func (x *XMLSystem) GetXML() ([]byte, error) {
    return x.data, nil
}

// Bidirectional adapter - converts both ways
type XMLJSONAdapter struct {
    xmlSystem *XMLSystem
}

func NewXMLJSONAdapter(system *XMLSystem) *XMLJSONAdapter {
    return &XMLJSONAdapter{xmlSystem: system}
}

// Implement JSONProcessor interface
func (a *XMLJSONAdapter) ProcessJSON(jsonData []byte) error {
    // Convert JSON to XML for the legacy system
    var data map[string]interface{}
    if err := json.Unmarshal(jsonData, &data); err != nil {
        return err
    }

    xmlData, err := xml.Marshal(data)
    if err != nil {
        return err
    }

    return a.xmlSystem.ProcessXML(xmlData)
}

func (a *XMLJSONAdapter) GetJSON() ([]byte, error) {
    // Get XML from legacy system and convert to JSON
    xmlData, err := a.xmlSystem.GetXML()
    if err != nil {
        return nil, err
    }

    var data map[string]interface{}
    if err := xml.Unmarshal(xmlData, &data); err != nil {
        return nil, err
    }

    return json.Marshal(data)
}

// Usage:
func ProcessOrder(processor JSONProcessor, orderJSON []byte) error {
    // Works with adapter seamlessly
    if err := processor.ProcessJSON(orderJSON); err != nil {
        return err
    }

    result, err := processor.GetJSON()
    if err != nil {
        return err
    }

    fmt.Println("Processed:", string(result))
    return nil
}
\`\`\`

**4. Common Mistakes to Avoid**

\`\`\`go
// MISTAKE 1: Adapter that modifies the adaptee
type BadAdapter struct {
    xmlSource *XMLDataSource
}

func (a *BadAdapter) GetJSON() string {
    a.xmlSource.data = "modified"  // Wrong! Don't modify adaptee
    return a.xmlSource.GetXML()
}

// CORRECT: Adapter only converts, doesn't modify
func (a *GoodAdapter) GetJSON() string {
    xml := a.xmlSource.GetXML()  // Read only
    return convertToJSON(xml)    // Convert without side effects
}

// MISTAKE 2: Exposing adaptee's interface
type BadAdapter struct {
    XMLSource *XMLDataSource  // Wrong! Public field exposes adaptee
}

// Clients can bypass adapter:
adapter.XMLSource.GetXML()  // Defeats purpose of adapter

// CORRECT: Keep adaptee private
type GoodAdapter struct {
    xmlSource *XMLDataSource  // Private, only adapter can access
}

// MISTAKE 3: Creating adapter for compatible interfaces
type JSONSource struct{}
func (j *JSONSource) GetJSON() string { return "{}" }

// Wrong! JSONSource already implements Target - no adapter needed
type UnnecessaryAdapter struct {
    source *JSONSource
}
func (a *UnnecessaryAdapter) GetJSON() string {
    return a.source.GetJSON()  // Just delegating, no conversion
}

// CORRECT: Use directly when interfaces match
var target Target = &JSONSource{}

// MISTAKE 4: Single adapter doing too much conversion
type OverloadedAdapter struct {
    xmlSource  *XMLDataSource
    csvSource  *CSVDataSource
    yamlSource *YAMLDataSource
}

// Wrong! One adapter should adapt ONE adaptee
func (a *OverloadedAdapter) GetJSON() string {
    if a.xmlSource != nil { /* convert */ }
    if a.csvSource != nil { /* convert */ }
    if a.yamlSource != nil { /* convert */ }
}

// CORRECT: Separate adapter for each adaptee
type XMLAdapter struct{ source *XMLDataSource }
type CSVAdapter struct{ source *CSVDataSource }
type YAMLAdapter struct{ source *YAMLDataSource }
\`\`\``,
	order: 0,
	testCode: `package patterns

import (
	"strings"
	"testing"
)

// Test1: XMLDataSource.GetXML returns data
func Test1(t *testing.T) {
	xml := &XMLDataSource{data: "<test>data</test>"}
	if xml.GetXML() != "<test>data</test>" {
		t.Error("GetXML should return stored data")
	}
}

// Test2: XMLToJSONAdapter implements Target
func Test2(t *testing.T) {
	xml := &XMLDataSource{data: "<user>John</user>"}
	var target Target = &XMLToJSONAdapter{xmlSource: xml}
	if target == nil {
		t.Error("Adapter should implement Target")
	}
}

// Test3: GetJSON wraps XML in JSON format
func Test3(t *testing.T) {
	xml := &XMLDataSource{data: "<user>John</user>"}
	adapter := &XMLToJSONAdapter{xmlSource: xml}
	result := adapter.GetJSON()
	if !strings.Contains(result, "xml") {
		t.Error("JSON should contain xml key")
	}
}

// Test4: GetJSON contains original XML data
func Test4(t *testing.T) {
	xml := &XMLDataSource{data: "<test>value</test>"}
	adapter := &XMLToJSONAdapter{xmlSource: xml}
	result := adapter.GetJSON()
	if !strings.Contains(result, "<test>value</test>") {
		t.Error("JSON should contain original XML")
	}
}

// Test5: GetJSON returns valid JSON-like structure
func Test5(t *testing.T) {
	xml := &XMLDataSource{data: "<data>test</data>"}
	adapter := &XMLToJSONAdapter{xmlSource: xml}
	result := adapter.GetJSON()
	if !strings.HasPrefix(result, "{") || !strings.HasSuffix(result, "}") {
		t.Error("Result should be JSON-like with braces")
	}
}

// Test6: Empty XML data works
func Test6(t *testing.T) {
	xml := &XMLDataSource{data: ""}
	adapter := &XMLToJSONAdapter{xmlSource: xml}
	result := adapter.GetJSON()
	if result == "" {
		t.Error("Should handle empty XML")
	}
}

// Test7: XMLDataSource stores data correctly
func Test7(t *testing.T) {
	xml := XMLDataSource{data: "test"}
	if xml.data != "test" {
		t.Error("XMLDataSource should store data")
	}
}

// Test8: Adapter has xmlSource field
func Test8(t *testing.T) {
	xml := &XMLDataSource{data: "data"}
	adapter := XMLToJSONAdapter{xmlSource: xml}
	if adapter.xmlSource == nil {
		t.Error("Adapter should have xmlSource field")
	}
}

// Test9: Adapter wraps with quotes
func Test9(t *testing.T) {
	xml := &XMLDataSource{data: "<a>b</a>"}
	adapter := &XMLToJSONAdapter{xmlSource: xml}
	result := adapter.GetJSON()
	if !strings.Contains(result, "\"") {
		t.Error("JSON should contain quotes")
	}
}

// Test10: Multiple adapters work independently
func Test10(t *testing.T) {
	xml1 := &XMLDataSource{data: "<a>1</a>"}
	xml2 := &XMLDataSource{data: "<b>2</b>"}
	a1 := &XMLToJSONAdapter{xmlSource: xml1}
	a2 := &XMLToJSONAdapter{xmlSource: xml2}
	if a1.GetJSON() == a2.GetJSON() {
		t.Error("Different adapters should produce different results")
	}
}
`,
	translations: {
		ru: {
			title: 'Паттерн Adapter (Адаптер)',
			description: `Реализуйте паттерн Adapter на Go — преобразуйте интерфейс класса в другой интерфейс, который ожидают клиенты.

**Вы реализуете:**

1. **Target интерфейс** - Интерфейс, который ожидает клиент
2. **Adaptee структура** - Существующий класс с несовместимым интерфейсом
3. **Adapter структура** - Преобразует интерфейс Adaptee в Target

**Пример использования:**

\`\`\`go
// Старая система использует формат XML
oldSystem := &XMLDataSource{data: "<user>John</user>"}

// Новая система ожидает JSON через интерфейс Target
adapter := &XMLToJSONAdapter{xmlSource: oldSystem}

// Клиент использует интерфейс Target - не знает об XML
json := adapter.GetJSON() // {"xml": "<user>John</user>"}

// Адаптер можно использовать везде, где ожидается Target
func ProcessData(source Target) {
    data := source.GetJSON()
    fmt.Println("Processing:", data)
}

ProcessData(adapter) // Работает бесшовно!
\`\`\``,
			hint1: `Создайте структуру адаптера с полем для хранения XMLDataSource (adaptee). Метод GetJSON должен вызывать xmlSource.GetXML() для получения данных, затем преобразовать/обернуть их в JSON формат.`,
			hint2: `Используйте fmt.Sprintf для создания JSON строки. Помните об экранировании двойных кавычек в Go строках с помощью обратного слеша: \\" создаёт литеральный символ кавычки в выводе.`,
			whyItMatters: `**1. Зачем нужен Adapter**

Паттерн Adapter решает проблему несовместимости интерфейсов — когда у вас есть существующий код (legacy системы, сторонние библиотеки) с одним интерфейсом, но ваш новый код ожидает другой интерфейс. Вместо модификации существующего кода (что может быть невозможно или рискованно), вы создаёте адаптер.

**Проблема, которую он решает:**

\`\`\`go
// БЕЗ Adapter - вынуждены менять клиентский код или adaptee
// Вариант 1: Модифицировать клиент (нарушает принцип Open/Closed)
func ProcessData(source interface{}) {
    switch s := source.(type) {
    case *JSONDataSource:
        data := s.GetJSON()
    case *XMLDataSource:
        // Нужно обрабатывать XML конвертацию inline
        data := convertXMLToJSON(s.GetXML())
    case *CSVDataSource:
        // Больше case для каждого формата...
        data := convertCSVToJSON(s.GetCSV())
    }
    // Некрасиво, нарушает OCP, сложно поддерживать
}

// Вариант 2: Модифицировать adaptee (может быть невозможно для стороннего кода)
// Нельзя изменить XMLDataSource из внешней библиотеки!
\`\`\`

**С Adapter:**

\`\`\`go
// Чисто: клиент знает только про Target интерфейс
func ProcessData(source Target) {
    data := source.GetJSON()  // работает с любым адаптером
    process(data)
}

// Каждый источник данных имеет свой адаптер
xmlAdapter := &XMLToJSONAdapter{xmlSource: xmlSource}
csvAdapter := &CSVToJSONAdapter{csvSource: csvSource}
yamlAdapter := &YAMLToJSONAdapter{yamlSource: yamlSource}

// Все работают через один интерфейс
ProcessData(xmlAdapter)   // адаптирует XML
ProcessData(csvAdapter)   // адаптирует CSV
ProcessData(yamlAdapter)  // адаптирует YAML
\`\`\`

**2. Примеры из реального мира в Go**

**Адаптер драйвера базы данных:**

\`\`\`go
// Target интерфейс, который ожидает ваше приложение
type Database interface {
    Query(sql string) ([]Row, error)
    Execute(sql string) error
}

// Adaptee: Legacy MongoDB клиент
type MongoClient struct {
    // MongoDB использует другой API
}

func (m *MongoClient) Find(collection string, filter map[string]interface{}) []Document {
    // MongoDB-специфичная реализация
}

// Adapter: Делает MongoDB похожей на SQL базу данных
type MongoDBAdapter struct {
    client *MongoClient
}

func (a *MongoDBAdapter) Query(sql string) ([]Row, error) {
    // Парсим SQL и конвертируем в MongoDB запрос
    collection, filter := parseSQLToMongo(sql)
    docs := a.client.Find(collection, filter)
    return convertDocsToRows(docs), nil
}

func (a *MongoDBAdapter) Execute(sql string) error {
    // Конвертируем SQL INSERT/UPDATE в MongoDB операции
    return a.client.Execute(convertSQLToMongoOp(sql))
}
\`\`\`

**Адаптер HTTP клиента:**

\`\`\`go
// Target: HTTP клиент интерфейс вашего приложения
type HTTPClient interface {
    Get(url string) (*Response, error)
    Post(url string, body []byte) (*Response, error)
}

// Adaptee: Сторонняя HTTP библиотека (например, resty, fasthttp)
type FastHTTPClient struct {
    // Другой API
}

func (f *FastHTTPClient) DoRequest(method, url string, body []byte) (int, []byte, error) {
    // FastHTTP реализация
}

// Adapter
type FastHTTPAdapter struct {
    client *FastHTTPClient
}

func (a *FastHTTPAdapter) Get(url string) (*Response, error) {
    status, body, err := a.client.DoRequest("GET", url, nil)
    return &Response{Status: status, Body: body}, err
}

func (a *FastHTTPAdapter) Post(url string, body []byte) (*Response, error) {
    status, respBody, err := a.client.DoRequest("POST", url, body)
    return &Response{Status: status, Body: respBody}, err
}
\`\`\`

**3. Продакшн паттерн - Двунаправленный адаптер**

\`\`\`go
package adapter

import (
    "encoding/json"
    "encoding/xml"
)

// Target интерфейс для современных систем
type JSONProcessor interface {
    ProcessJSON(data []byte) error
    GetJSON() ([]byte, error)
}

// Adaptee - legacy XML система
type XMLSystem struct {
    data []byte
}

func (x *XMLSystem) ProcessXML(data []byte) error {
    x.data = data
    return nil
}

func (x *XMLSystem) GetXML() ([]byte, error) {
    return x.data, nil
}

// Двунаправленный адаптер - конвертирует в обе стороны
type XMLJSONAdapter struct {
    xmlSystem *XMLSystem
}

func NewXMLJSONAdapter(system *XMLSystem) *XMLJSONAdapter {
    return &XMLJSONAdapter{xmlSystem: system}
}

// Реализуем JSONProcessor интерфейс
func (a *XMLJSONAdapter) ProcessJSON(jsonData []byte) error {
    // Конвертируем JSON в XML для legacy системы
    var data map[string]interface{}
    if err := json.Unmarshal(jsonData, &data); err != nil {
        return err
    }

    xmlData, err := xml.Marshal(data)
    if err != nil {
        return err
    }

    return a.xmlSystem.ProcessXML(xmlData)
}

func (a *XMLJSONAdapter) GetJSON() ([]byte, error) {
    // Получаем XML из legacy системы и конвертируем в JSON
    xmlData, err := a.xmlSystem.GetXML()
    if err != nil {
        return nil, err
    }

    var data map[string]interface{}
    if err := xml.Unmarshal(xmlData, &data); err != nil {
        return nil, err
    }

    return json.Marshal(data)
}

// Использование:
func ProcessOrder(processor JSONProcessor, orderJSON []byte) error {
    // Работает с адаптером бесшовно
    if err := processor.ProcessJSON(orderJSON); err != nil {
        return err
    }

    result, err := processor.GetJSON()
    if err != nil {
        return err
    }

    fmt.Println("Processed:", string(result))
    return nil
}
\`\`\`

**4. Типичные ошибки**

\`\`\`go
// ОШИБКА 1: Адаптер, модифицирующий adaptee
type BadAdapter struct {
    xmlSource *XMLDataSource
}

func (a *BadAdapter) GetJSON() string {
    a.xmlSource.data = "modified"  // Неправильно! Не модифицируйте adaptee
    return a.xmlSource.GetXML()
}

// ПРАВИЛЬНО: Адаптер только конвертирует, не модифицирует
func (a *GoodAdapter) GetJSON() string {
    xml := a.xmlSource.GetXML()  // Только чтение
    return convertToJSON(xml)    // Конвертация без побочных эффектов
}

// ОШИБКА 2: Раскрытие интерфейса adaptee
type BadAdapter struct {
    XMLSource *XMLDataSource  // Неправильно! Публичное поле раскрывает adaptee
}

// Клиенты могут обходить адаптер:
adapter.XMLSource.GetXML()  // Нарушает цель адаптера

// ПРАВИЛЬНО: Держите adaptee приватным
type GoodAdapter struct {
    xmlSource *XMLDataSource  // Приватное, доступ только через адаптер
}

// ОШИБКА 3: Создание адаптера для совместимых интерфейсов
type JSONSource struct{}
func (j *JSONSource) GetJSON() string { return "{}" }

// Неправильно! JSONSource уже реализует Target - адаптер не нужен
type UnnecessaryAdapter struct {
    source *JSONSource
}
func (a *UnnecessaryAdapter) GetJSON() string {
    return a.source.GetJSON()  // Просто делегирование, без конверсии
}

// ПРАВИЛЬНО: Используйте напрямую когда интерфейсы совпадают
var target Target = &JSONSource{}

// ОШИБКА 4: Один адаптер делает слишком много конверсий
type OverloadedAdapter struct {
    xmlSource  *XMLDataSource
    csvSource  *CSVDataSource
    yamlSource *YAMLDataSource
}

// Неправильно! Один адаптер должен адаптировать ОДИН adaptee
func (a *OverloadedAdapter) GetJSON() string {
    if a.xmlSource != nil { /* конвертируем */ }
    if a.csvSource != nil { /* конвертируем */ }
    if a.yamlSource != nil { /* конвертируем */ }
}

// ПРАВИЛЬНО: Отдельный адаптер для каждого adaptee
type XMLAdapter struct{ source *XMLDataSource }
type CSVAdapter struct{ source *CSVDataSource }
type YAMLAdapter struct{ source *YAMLDataSource }
\`\`\``,
			solutionCode: `package patterns

import "fmt"

type Target interface {	// Target интерфейс - что ожидает клиент
	GetJSON() string	// клиент хочет JSON данные
}

type XMLDataSource struct {	// Adaptee - существующий класс с несовместимым интерфейсом
	data string	// хранит XML данные
}

func (x *XMLDataSource) GetXML() string {	// метод Adaptee - возвращает XML
	return x.data	// возвращает данные в формате XML
}

type XMLToJSONAdapter struct {	// Adapter - преобразует Adaptee в Target интерфейс
	xmlSource *XMLDataSource	// хранит ссылку на adaptee
}

func (a *XMLToJSONAdapter) GetJSON() string {	// реализует Target интерфейс
	xml := a.xmlSource.GetXML()	// получаем данные от adaptee используя его интерфейс
	return fmt.Sprintf("{\\"xml\\": \\"%s\\"}", xml)	// преобразуем/оборачиваем в JSON формат
}`
		},
		uz: {
			title: 'Adapter Pattern',
			description: `Go tilida Adapter patternini amalga oshiring — klassning interfeysini mijozlar kutayotgan boshqa interfeysga aylantiring.

**Siz amalga oshirasiz:**

1. **Target interfeys** - Mijoz kutayotgan interfeys
2. **Adaptee strukturasi** - Mos kelmaydigan interfeysga ega mavjud klass
3. **Adapter strukturasi** - Adaptee interfeysini Target ga o'zgartiradi

**Foydalanish namunasi:**

\`\`\`go
// Eski tizim XML formatidan foydalanadi
oldSystem := &XMLDataSource{data: "<user>John</user>"}

// Yangi tizim Target interfeysi orqali JSON kutadi
adapter := &XMLToJSONAdapter{xmlSource: oldSystem}

// Mijoz Target interfeysidan foydalanadi - XML haqida bilmaydi
json := adapter.GetJSON() // {"xml": "<user>John</user>"}

// Adapter Target kutilgan har qanday joyda ishlatilishi mumkin
func ProcessData(source Target) {
    data := source.GetJSON()
    fmt.Println("Processing:", data)
}

ProcessData(adapter) // Muammosiz ishlaydi!
\`\`\``,
			hint1: `XMLDataSource (adaptee) ni saqlash uchun maydonli adapter strukturasini yarating. GetJSON metodi ma'lumot olish uchun xmlSource.GetXML() ni chaqirishi, so'ng uni JSON formatiga o'zgartirishi/o'rashi kerak.`,
			hint2: `JSON string yaratish uchun fmt.Sprintf dan foydalaning. Go stringlarida qo'sh tirnoqlarni backslash bilan ekranlashni unutmang: \\" chiqishda literal tirnoq belgisini yaratadi.`,
			whyItMatters: `**1. Adapter nima uchun kerak**

Adapter pattern interfeys mos kelmasligi muammosini hal qiladi — sizda bitta interfeysga ega mavjud kod (legacy tizimlar, uchinchi tomon kutubxonalari) bor, lekin yangi kodingiz boshqa interfeysni kutadi. Mavjud kodni o'zgartirish o'rniga (bu imkonsiz yoki xavfli bo'lishi mumkin), adapter yaratasiz.

**U hal qiladigan muammo:**

\`\`\`go
// Adapter SIZ - mijoz kodini yoki adaptee ni o'zgartirishga majburmiz
// 1-variant: Mijozni o'zgartirish (Open/Closed printsipini buzadi)
func ProcessData(source interface{}) {
    switch s := source.(type) {
    case *JSONDataSource:
        data := s.GetJSON()
    case *XMLDataSource:
        // XML konversiyasini inline ishlash kerak
        data := convertXMLToJSON(s.GetXML())
    case *CSVDataSource:
        // Har bir format uchun ko'proq case...
        data := convertCSVToJSON(s.GetCSV())
    }
    // Yomon, OCP ni buzadi, qo'llab-quvvatlash qiyin
}

// 2-variant: Adaptee ni o'zgartirish (uchinchi tomon kodi uchun imkonsiz bo'lishi mumkin)
// Tashqi kutubxonadan XMLDataSource ni o'zgartirib bo'lmaydi!
\`\`\`

**Adapter BILAN:**

\`\`\`go
// Toza: mijoz faqat Target interfeysini biladi
func ProcessData(source Target) {
    data := source.GetJSON()  // har qanday adapter bilan ishlaydi
    process(data)
}

// Har bir ma'lumot manbai o'z adapteriga ega
xmlAdapter := &XMLToJSONAdapter{xmlSource: xmlSource}
csvAdapter := &CSVToJSONAdapter{csvSource: csvSource}
yamlAdapter := &YAMLToJSONAdapter{yamlSource: yamlSource}

// Hammasi bitta interfeys orqali ishlaydi
ProcessData(xmlAdapter)   // XML ni moslashtiradi
ProcessData(csvAdapter)   // CSV ni moslashtiradi
ProcessData(yamlAdapter)  // YAML ni moslashtiradi
\`\`\`

**2. Go'da real hayotiy misollar**

**Ma'lumotlar bazasi drayveri adapteri:**

\`\`\`go
// Target interfeys - ilovangiz kutayotgan
type Database interface {
    Query(sql string) ([]Row, error)
    Execute(sql string) error
}

// Adaptee: Legacy MongoDB mijozi
type MongoClient struct {
    // MongoDB boshqa API ishlatadi
}

func (m *MongoClient) Find(collection string, filter map[string]interface{}) []Document {
    // MongoDB-maxsus implementatsiya
}

// Adapter: MongoDB ni SQL bazasiga o'xshatadi
type MongoDBAdapter struct {
    client *MongoClient
}

func (a *MongoDBAdapter) Query(sql string) ([]Row, error) {
    // SQL ni tahlil qilamiz va MongoDB so'roviga aylantiramiz
    collection, filter := parseSQLToMongo(sql)
    docs := a.client.Find(collection, filter)
    return convertDocsToRows(docs), nil
}

func (a *MongoDBAdapter) Execute(sql string) error {
    // SQL INSERT/UPDATE ni MongoDB operatsiyalariga aylantiramiz
    return a.client.Execute(convertSQLToMongoOp(sql))
}
\`\`\`

**HTTP mijoz adapteri:**

\`\`\`go
// Target: Ilovangizning HTTP mijoz interfeysi
type HTTPClient interface {
    Get(url string) (*Response, error)
    Post(url string, body []byte) (*Response, error)
}

// Adaptee: Uchinchi tomon HTTP kutubxonasi (masalan, resty, fasthttp)
type FastHTTPClient struct {
    // Boshqa API
}

func (f *FastHTTPClient) DoRequest(method, url string, body []byte) (int, []byte, error) {
    // FastHTTP implementatsiyasi
}

// Adapter
type FastHTTPAdapter struct {
    client *FastHTTPClient
}

func (a *FastHTTPAdapter) Get(url string) (*Response, error) {
    status, body, err := a.client.DoRequest("GET", url, nil)
    return &Response{Status: status, Body: body}, err
}

func (a *FastHTTPAdapter) Post(url string, body []byte) (*Response, error) {
    status, respBody, err := a.client.DoRequest("POST", url, body)
    return &Response{Status: status, Body: respBody}, err
}
\`\`\`

**3. Production pattern - Ikki tomonlama adapter**

\`\`\`go
package adapter

import (
    "encoding/json"
    "encoding/xml"
)

// Zamonaviy tizimlar uchun Target interfeys
type JSONProcessor interface {
    ProcessJSON(data []byte) error
    GetJSON() ([]byte, error)
}

// Adaptee - legacy XML tizim
type XMLSystem struct {
    data []byte
}

func (x *XMLSystem) ProcessXML(data []byte) error {
    x.data = data
    return nil
}

func (x *XMLSystem) GetXML() ([]byte, error) {
    return x.data, nil
}

// Ikki tomonlama adapter - ikki yo'nalishda ham o'zgartiradi
type XMLJSONAdapter struct {
    xmlSystem *XMLSystem
}

func NewXMLJSONAdapter(system *XMLSystem) *XMLJSONAdapter {
    return &XMLJSONAdapter{xmlSystem: system}
}

// JSONProcessor interfeysini amalga oshiramiz
func (a *XMLJSONAdapter) ProcessJSON(jsonData []byte) error {
    // Legacy tizim uchun JSON ni XML ga aylantiramiz
    var data map[string]interface{}
    if err := json.Unmarshal(jsonData, &data); err != nil {
        return err
    }

    xmlData, err := xml.Marshal(data)
    if err != nil {
        return err
    }

    return a.xmlSystem.ProcessXML(xmlData)
}

func (a *XMLJSONAdapter) GetJSON() ([]byte, error) {
    // Legacy tizimdan XML olamiz va JSON ga aylantiramiz
    xmlData, err := a.xmlSystem.GetXML()
    if err != nil {
        return nil, err
    }

    var data map[string]interface{}
    if err := xml.Unmarshal(xmlData, &data); err != nil {
        return nil, err
    }

    return json.Marshal(data)
}

// Foydalanish:
func ProcessOrder(processor JSONProcessor, orderJSON []byte) error {
    // Adapter bilan muammosiz ishlaydi
    if err := processor.ProcessJSON(orderJSON); err != nil {
        return err
    }

    result, err := processor.GetJSON()
    if err != nil {
        return err
    }

    fmt.Println("Processed:", string(result))
    return nil
}
\`\`\`

**4. Keng tarqalgan xatolar**

\`\`\`go
// XATO 1: Adaptee ni o'zgartiradigan adapter
type BadAdapter struct {
    xmlSource *XMLDataSource
}

func (a *BadAdapter) GetJSON() string {
    a.xmlSource.data = "modified"  // Noto'g'ri! Adaptee ni o'zgartirmang
    return a.xmlSource.GetXML()
}

// TO'G'RI: Adapter faqat o'zgartiradi, modifikatsiya qilmaydi
func (a *GoodAdapter) GetJSON() string {
    xml := a.xmlSource.GetXML()  // Faqat o'qish
    return convertToJSON(xml)    // Yon ta'sirlarsiz konversiya
}

// XATO 2: Adaptee interfeysini ochish
type BadAdapter struct {
    XMLSource *XMLDataSource  // Noto'g'ri! Ochiq maydon adaptee ni ochib qo'yadi
}

// Mijozlar adapterni chetlab o'tishi mumkin:
adapter.XMLSource.GetXML()  // Adapter maqsadini buzadi

// TO'G'RI: Adaptee ni maxfiy saqlang
type GoodAdapter struct {
    xmlSource *XMLDataSource  // Maxfiy, faqat adapter kirishi mumkin
}

// XATO 3: Mos interfeyslar uchun adapter yaratish
type JSONSource struct{}
func (j *JSONSource) GetJSON() string { return "{}" }

// Noto'g'ri! JSONSource allaqachon Target ni amalga oshiradi - adapter kerak emas
type UnnecessaryAdapter struct {
    source *JSONSource
}
func (a *UnnecessaryAdapter) GetJSON() string {
    return a.source.GetJSON()  // Shunchaki delegatsiya, konversiyasiz
}

// TO'G'RI: Interfeyslar mos kelganda to'g'ridan-to'g'ri foydalaning
var target Target = &JSONSource{}

// XATO 4: Bitta adapter juda ko'p konversiya qiladi
type OverloadedAdapter struct {
    xmlSource  *XMLDataSource
    csvSource  *CSVDataSource
    yamlSource *YAMLDataSource
}

// Noto'g'ri! Bitta adapter BITTA adaptee ni moslashtirishi kerak
func (a *OverloadedAdapter) GetJSON() string {
    if a.xmlSource != nil { /* o'zgartiramiz */ }
    if a.csvSource != nil { /* o'zgartiramiz */ }
    if a.yamlSource != nil { /* o'zgartiramiz */ }
}

// TO'G'RI: Har bir adaptee uchun alohida adapter
type XMLAdapter struct{ source *XMLDataSource }
type CSVAdapter struct{ source *CSVDataSource }
type YAMLAdapter struct{ source *YAMLDataSource }
\`\`\``,
			solutionCode: `package patterns

import "fmt"

type Target interface {	// Target interfeys - mijoz nimani kutayotgani
	GetJSON() string	// mijoz JSON ma'lumot xohlaydi
}

type XMLDataSource struct {	// Adaptee - mos kelmaydigan interfeysga ega mavjud klass
	data string	// XML ma'lumotlarni saqlaydi
}

func (x *XMLDataSource) GetXML() string {	// Adaptee metodi - XML qaytaradi
	return x.data	// XML formatida ma'lumotlarni qaytaradi
}

type XMLToJSONAdapter struct {	// Adapter - Adaptee ni Target interfeysiga o'zgartiradi
	xmlSource *XMLDataSource	// adaptee ga havolani saqlaydi
}

func (a *XMLToJSONAdapter) GetJSON() string {	// Target interfeysini amalga oshiradi
	xml := a.xmlSource.GetXML()	// adaptee dan uning interfeysi orqali ma'lumot olish
	return fmt.Sprintf("{\\"xml\\": \\"%s\\"}", xml)	// JSON formatiga o'zgartirish/o'rash
}`
		}
	}
};

export default task;
