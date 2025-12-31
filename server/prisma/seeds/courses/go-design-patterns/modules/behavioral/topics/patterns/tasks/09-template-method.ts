import { Task } from '../../../../../../../types';

export const task: Task = {
	slug: 'go-dp-template-method',
	title: 'Template Method Pattern',
	difficulty: 'easy',
	tags: ['go', 'design-patterns', 'behavioral', 'template-method'],
	estimatedTime: '25m',
	isPremium: false,
	youtubeUrl: '',
	description: `Implement the Template Method pattern in Go - define the skeleton of an algorithm, deferring some steps to subclasses.

**You will implement:**

1. **DataMiner interface** - Steps for data mining
2. **DataMinerTemplate** - Template with Execute method
3. **CSVMiner, JSONMiner** - Concrete implementations

**Example Usage:**

\`\`\`go
csvMiner := NewCSVMiner()	// create CSV miner with template
result := csvMiner.Execute("data.csv")	// execute all steps in order

jsonMiner := NewJSONMiner()	// create JSON miner with template
result2 := jsonMiner.Execute("config.json")	// same algorithm, different steps

fmt.Println(result[0])	// "Opening CSV file: data.csv"
fmt.Println(result[1])	// "Extracting CSV rows"
fmt.Println(result[2])	// "Parsing CSV data"
fmt.Println(result[3])	// "Closing CSV file"
\`\`\``,
	initialCode: `package patterns

type DataMiner interface {
}

type DataMinerTemplate struct {
	miner DataMiner
}

func (t *DataMinerTemplate) Execute(path string) []string {
}

type CSVMiner struct{}

func (m *CSVMiner) OpenFile(path string) string {
}

func (m *CSVMiner) ExtractData() string {
}

func (m *CSVMiner) ParseData() string {
}

func (m *CSVMiner) CloseFile() string {
}

type JSONMiner struct{}

func (m *JSONMiner) OpenFile(path string) string {
}

func (m *JSONMiner) ExtractData() string {
}

func (m *JSONMiner) ParseData() string {
}

func (m *JSONMiner) CloseFile() string {
}

func NewCSVMiner() *DataMinerTemplate {
}

func NewJSONMiner() *DataMinerTemplate {
}`,
	solutionCode: `package patterns

import "fmt"	// format package for string formatting

// DataMiner defines the steps of data mining algorithm
type DataMiner interface {	// abstract steps interface
	OpenFile(path string) string	// step 1: open the file
	ExtractData() string	// step 2: extract raw data
	ParseData() string	// step 3: parse extracted data
	CloseFile() string	// step 4: close the file
}

// DataMinerTemplate provides the template method
type DataMinerTemplate struct {	// template struct holds miner
	miner DataMiner	// concrete miner implementation
}

// Execute is the template method that defines algorithm skeleton
func (t *DataMinerTemplate) Execute(path string) []string {	// invariant algorithm structure
	results := make([]string, 0, 4)	// preallocate slice for 4 steps
	results = append(results, t.miner.OpenFile(path))	// step 1: delegate to miner
	results = append(results, t.miner.ExtractData())	// step 2: delegate to miner
	results = append(results, t.miner.ParseData())	// step 3: delegate to miner
	results = append(results, t.miner.CloseFile())	// step 4: delegate to miner
	return results	// return all step results
}

// CSVMiner implements data mining for CSV files
type CSVMiner struct{}	// concrete implementation for CSV

// OpenFile opens CSV file at given path
func (m *CSVMiner) OpenFile(path string) string {	// CSV-specific file opening
	return fmt.Sprintf("Opening CSV file: %s", path)	// format with path
}

// ExtractData extracts rows from CSV
func (m *CSVMiner) ExtractData() string {	// CSV-specific extraction
	return "Extracting CSV rows"	// CSV extracts rows
}

// ParseData parses CSV data into records
func (m *CSVMiner) ParseData() string {	// CSV-specific parsing
	return "Parsing CSV data"	// parse comma-separated values
}

// CloseFile closes the CSV file
func (m *CSVMiner) CloseFile() string {	// CSV-specific cleanup
	return "Closing CSV file"	// close file handle
}

// JSONMiner implements data mining for JSON files
type JSONMiner struct{}	// concrete implementation for JSON

// OpenFile opens JSON file at given path
func (m *JSONMiner) OpenFile(path string) string {	// JSON-specific file opening
	return fmt.Sprintf("Opening JSON file: %s", path)	// format with path
}

// ExtractData extracts objects from JSON
func (m *JSONMiner) ExtractData() string {	// JSON-specific extraction
	return "Extracting JSON objects"	// JSON extracts objects
}

// ParseData parses JSON data into structures
func (m *JSONMiner) ParseData() string {	// JSON-specific parsing
	return "Parsing JSON data"	// parse JSON notation
}

// CloseFile closes the JSON file
func (m *JSONMiner) CloseFile() string {	// JSON-specific cleanup
	return "Closing JSON file"	// close file handle
}

// NewCSVMiner creates template with CSV miner
func NewCSVMiner() *DataMinerTemplate {	// factory for CSV mining
	return &DataMinerTemplate{miner: &CSVMiner{}}	// inject CSV miner
}

// NewJSONMiner creates template with JSON miner
func NewJSONMiner() *DataMinerTemplate {	// factory for JSON mining
	return &DataMinerTemplate{miner: &JSONMiner{}}	// inject JSON miner
}`,
	hint1: `**Understanding Template Method Structure:**

The Template Method pattern has two key components:

\`\`\`go
// 1. Abstract interface - defines customizable steps
type DataMiner interface {
	OpenFile(path string) string	// step varies by file type
	ExtractData() string	// extraction varies by format
	ParseData() string	// parsing varies by format
	CloseFile() string	// cleanup varies by file type
}

// 2. Template - defines invariant algorithm
type DataMinerTemplate struct {
	miner DataMiner	// holds concrete step implementations
}

// Template method - fixed algorithm structure
func (t *DataMinerTemplate) Execute(path string) []string {
	// Steps always execute in this order
	// Only the HOW of each step varies
}
\`\`\`

Each miner implements steps differently but algorithm order never changes.`,
	hint2: `**Complete Execute Implementation:**

\`\`\`go
func (t *DataMinerTemplate) Execute(path string) []string {
	results := make([]string, 0, 4)	// preallocate for efficiency

	// Algorithm skeleton - order is fixed
	results = append(results, t.miner.OpenFile(path))	// always first
	results = append(results, t.miner.ExtractData())	// always second
	results = append(results, t.miner.ParseData())	// always third
	results = append(results, t.miner.CloseFile())	// always last

	return results	// return all results in order
}

// CSVMiner step implementation
func (m *CSVMiner) OpenFile(path string) string {
	return fmt.Sprintf("Opening CSV file: %s", path)
}

func (m *CSVMiner) ExtractData() string {
	return "Extracting CSV rows"	// CSV-specific wording
}
\`\`\`

The template guarantees algorithm structure while miners customize steps.`,
	whyItMatters: `## Why Template Method Exists

**The Problem: Duplicated Algorithm Structure**

Without Template Method, you duplicate invariant algorithm logic:

\`\`\`go
// ❌ WITHOUT TEMPLATE METHOD - duplicated structure
func ProcessCSV(path string) []string {
	results := []string{}
	results = append(results, openCSV(path))	// duplicated order
	results = append(results, extractCSV())	// duplicated order
	results = append(results, parseCSV())	// duplicated order
	results = append(results, closeCSV())	// duplicated order
	return results
}

func ProcessJSON(path string) []string {
	results := []string{}
	results = append(results, openJSON(path))	// same structure repeated
	results = append(results, extractJSON())	// same structure repeated
	results = append(results, parseJSON())	// same structure repeated
	results = append(results, closeJSON())	// same structure repeated
	return results
}

// ✅ WITH TEMPLATE METHOD - single algorithm definition
func (t *DataMinerTemplate) Execute(path string) []string {
	results := make([]string, 0, 4)	// algorithm defined once
	results = append(results, t.miner.OpenFile(path))
	results = append(results, t.miner.ExtractData())
	results = append(results, t.miner.ParseData())
	results = append(results, t.miner.CloseFile())
	return results	// only steps vary, not structure
}
\`\`\`

---

## Real-World Examples in Go

**1. HTTP Handler Middleware (net/http):**
\`\`\`go
// Template: ServeHTTP structure
// Steps: Authentication, Authorization, Business Logic, Logging
\`\`\`

**2. Database/SQL (database/sql):**
\`\`\`go
// Template: Query execution flow
// Steps: Connect, Prepare, Execute, Scan, Close
\`\`\`

**3. Testing (testing package):**
\`\`\`go
// Template: Test execution
// Steps: Setup, Run, Teardown, Report
\`\`\`

**4. Build Tools (go build):**
\`\`\`go
// Template: Build process
// Steps: Parse, Type-check, Generate, Link
\`\`\`

---

## Production Pattern: HTTP Request Handler

\`\`\`go
package main

import (
	"fmt"
	"time"
)

// RequestHandler defines customizable request handling steps
type RequestHandler interface {
	ValidateRequest(req *Request) error	// step 1: validate input
	Authenticate(req *Request) error	// step 2: check auth
	Process(req *Request) *Response	// step 3: business logic
	FormatResponse(resp *Response) string	// step 4: format output
}

// Request represents incoming HTTP request
type Request struct {
	Method  string	// HTTP method
	Path    string	// request path
	Token   string	// auth token
	Body    string	// request body
}

// Response represents outgoing HTTP response
type Response struct {
	Status  int	// HTTP status code
	Data    interface{}	// response data
	Error   string	// error message if any
}

// RequestTemplate provides template method for request handling
type RequestTemplate struct {
	handler RequestHandler	// concrete handler
	logger  func(string)	// logging function
}

// Handle is the template method - invariant algorithm structure
func (t *RequestTemplate) Handle(req *Request) string {
	start := time.Now()	// track request duration

	// Step 1: Validate (hook point)
	if err := t.handler.ValidateRequest(req); err != nil {
		t.logger(fmt.Sprintf("Validation failed: %v", err))
		return t.handler.FormatResponse(&Response{Status: 400, Error: err.Error()})
	}

	// Step 2: Authenticate (hook point)
	if err := t.handler.Authenticate(req); err != nil {
		t.logger(fmt.Sprintf("Auth failed: %v", err))
		return t.handler.FormatResponse(&Response{Status: 401, Error: err.Error()})
	}

	// Step 3: Process (hook point)
	resp := t.handler.Process(req)

	// Step 4: Format and return (hook point)
	result := t.handler.FormatResponse(resp)

	t.logger(fmt.Sprintf("Request completed in %v", time.Since(start)))
	return result
}

// APIHandler handles REST API requests
type APIHandler struct{}

func (h *APIHandler) ValidateRequest(req *Request) error {
	if req.Method == "" || req.Path == "" {
		return fmt.Errorf("invalid request: missing method or path")
	}
	return nil
}

func (h *APIHandler) Authenticate(req *Request) error {
	if req.Token != "valid-token" {
		return fmt.Errorf("invalid or missing token")
	}
	return nil
}

func (h *APIHandler) Process(req *Request) *Response {
	return &Response{
		Status: 200,
		Data:   map[string]string{"message": "API request processed", "path": req.Path},
	}
}

func (h *APIHandler) FormatResponse(resp *Response) string {
	if resp.Error != "" {
		return fmt.Sprintf("{\\"status\\": %d, \\"error\\": \\"%s\\"}", resp.Status, resp.Error)
	}
	return fmt.Sprintf("{\\"status\\": %d, \\"data\\": %v}", resp.Status, resp.Data)
}

// WebhookHandler handles webhook requests
type WebhookHandler struct{}

func (h *WebhookHandler) ValidateRequest(req *Request) error {
	if req.Body == "" {
		return fmt.Errorf("webhook body cannot be empty")
	}
	return nil
}

func (h *WebhookHandler) Authenticate(req *Request) error {
	// Webhooks might use signature verification instead
	return nil	// simplified for example
}

func (h *WebhookHandler) Process(req *Request) *Response {
	return &Response{
		Status: 202,	// accepted
		Data:   "Webhook queued for processing",
	}
}

func (h *WebhookHandler) FormatResponse(resp *Response) string {
	return fmt.Sprintf("Status: %d - %v", resp.Status, resp.Data)
}

// NewAPITemplate creates template for API requests
func NewAPITemplate(logger func(string)) *RequestTemplate {
	return &RequestTemplate{handler: &APIHandler{}, logger: logger}
}

// NewWebhookTemplate creates template for webhook requests
func NewWebhookTemplate(logger func(string)) *RequestTemplate {
	return &RequestTemplate{handler: &WebhookHandler{}, logger: logger}
}

// Usage
func main() {
	logger := func(msg string) { fmt.Println("[LOG]", msg) }

	apiTemplate := NewAPITemplate(logger)
	result := apiTemplate.Handle(&Request{
		Method: "GET",
		Path:   "/users",
		Token:  "valid-token",
	})
	fmt.Println(result)

	webhookTemplate := NewWebhookTemplate(logger)
	result2 := webhookTemplate.Handle(&Request{
		Method: "POST",
		Path:   "/webhook",
		Body:   "{\\"event\\": \\"user.created\\"}",
	})
	fmt.Println(result2)
}
\`\`\`

---

## Common Mistakes to Avoid

**1. Template Method vs Strategy:**
\`\`\`go
// ❌ WRONG - Template Method when Strategy fits better
type Algorithm interface {
	Step1()
	Step2()
}
// If steps don't need fixed order, use Strategy

// ✅ RIGHT - Template Method for fixed algorithm structure
func (t *Template) Execute() {
	t.hook.Step1()	// order matters
	t.hook.Step2()	// must follow step1
	t.hook.Step3()	// must follow step2
}
\`\`\`

**2. Too Many Hook Points:**
\`\`\`go
// ❌ WRONG - every line is a hook
type Handler interface {
	BeforeValidation()
	Validate()
	AfterValidation()
	BeforeProcess()
	Process()
	AfterProcess()
	// Too granular!
}

// ✅ RIGHT - meaningful hook points only
type Handler interface {
	Validate() error	// clear responsibility
	Process() *Result	// clear responsibility
	Format(*Result) string	// clear responsibility
}
\`\`\`

**3. Forgetting to Call Hooks:**
\`\`\`go
// ❌ WRONG - hook not called
func (t *Template) Execute() {
	t.hook.Step1()
	// Forgot Step2!
	t.hook.Step3()
}

// ✅ RIGHT - all hooks called in order
func (t *Template) Execute() {
	t.hook.Step1()
	t.hook.Step2()	// don't forget any step
	t.hook.Step3()
}
\`\`\``,
	order: 8,
	testCode: `package patterns

import (
	"strings"
	"testing"
)

// Test1: NewCSVMiner returns DataMinerTemplate
func Test1(t *testing.T) {
	miner := NewCSVMiner()
	if miner == nil {
		t.Error("NewCSVMiner should return non-nil template")
	}
}

// Test2: NewJSONMiner returns DataMinerTemplate
func Test2(t *testing.T) {
	miner := NewJSONMiner()
	if miner == nil {
		t.Error("NewJSONMiner should return non-nil template")
	}
}

// Test3: CSVMiner.Execute returns 4 results
func Test3(t *testing.T) {
	miner := NewCSVMiner()
	results := miner.Execute("test.csv")
	if len(results) != 4 {
		t.Errorf("Expected 4 results, got %d", len(results))
	}
}

// Test4: CSVMiner.OpenFile includes path
func Test4(t *testing.T) {
	miner := NewCSVMiner()
	results := miner.Execute("data.csv")
	if !strings.Contains(results[0], "data.csv") {
		t.Error("OpenFile should include file path")
	}
}

// Test5: CSVMiner steps are CSV-specific
func Test5(t *testing.T) {
	miner := NewCSVMiner()
	results := miner.Execute("test.csv")
	if !strings.Contains(results[1], "CSV") {
		t.Error("ExtractData should mention CSV")
	}
}

// Test6: JSONMiner.Execute returns 4 results
func Test6(t *testing.T) {
	miner := NewJSONMiner()
	results := miner.Execute("config.json")
	if len(results) != 4 {
		t.Errorf("Expected 4 results, got %d", len(results))
	}
}

// Test7: JSONMiner.OpenFile includes path
func Test7(t *testing.T) {
	miner := NewJSONMiner()
	results := miner.Execute("config.json")
	if !strings.Contains(results[0], "config.json") {
		t.Error("OpenFile should include file path")
	}
}

// Test8: JSONMiner steps are JSON-specific
func Test8(t *testing.T) {
	miner := NewJSONMiner()
	results := miner.Execute("test.json")
	if !strings.Contains(results[1], "JSON") {
		t.Error("ExtractData should mention JSON")
	}
}

// Test9: Execute calls all 4 steps in order
func Test9(t *testing.T) {
	miner := NewCSVMiner()
	results := miner.Execute("test.csv")
	if !strings.Contains(results[0], "Opening") || !strings.Contains(results[3], "Closing") {
		t.Error("Execute should call steps in order: Open, Extract, Parse, Close")
	}
}

// Test10: Different miners produce different output
func Test10(t *testing.T) {
	csv := NewCSVMiner()
	json := NewJSONMiner()
	csvResult := csv.Execute("file.csv")
	jsonResult := json.Execute("file.json")
	if csvResult[1] == jsonResult[1] {
		t.Error("Different miners should produce different extract output")
	}
}
`,
	translations: {
		ru: {
			title: 'Паттерн Template Method (Шаблонный метод)',
			description: `Реализуйте паттерн Template Method на Go — определите скелет алгоритма, откладывая некоторые шаги на подклассы.

**Вы реализуете:**

1. **Интерфейс DataMiner** - Шаги для обработки данных
2. **DataMinerTemplate** - Шаблон с методом Execute
3. **CSVMiner, JSONMiner** - Конкретные реализации

**Пример использования:**

\`\`\`go
csvMiner := NewCSVMiner()	// создаём CSV miner с шаблоном
result := csvMiner.Execute("data.csv")	// выполняем все шаги по порядку

jsonMiner := NewJSONMiner()	// создаём JSON miner с шаблоном
result2 := jsonMiner.Execute("config.json")	// тот же алгоритм, другие шаги

fmt.Println(result[0])	// "Opening CSV file: data.csv"
fmt.Println(result[1])	// "Extracting CSV rows"
fmt.Println(result[2])	// "Parsing CSV data"
fmt.Println(result[3])	// "Closing CSV file"
\`\`\``,
			hint1: `**Понимание структуры Template Method:**

Паттерн Template Method имеет два ключевых компонента:

\`\`\`go
// 1. Абстрактный интерфейс - определяет настраиваемые шаги
type DataMiner interface {
	OpenFile(path string) string	// шаг зависит от типа файла
	ExtractData() string	// извлечение зависит от формата
	ParseData() string	// парсинг зависит от формата
	CloseFile() string	// очистка зависит от типа файла
}

// 2. Шаблон - определяет неизменный алгоритм
type DataMinerTemplate struct {
	miner DataMiner	// хранит конкретную реализацию шагов
}

// Шаблонный метод - фиксированная структура алгоритма
func (t *DataMinerTemplate) Execute(path string) []string {
	// Шаги всегда выполняются в этом порядке
	// Меняется только КАК выполняется каждый шаг
}
\`\`\`

Каждый miner реализует шаги по-разному, но порядок алгоритма никогда не меняется.`,
			hint2: `**Полная реализация Execute:**

\`\`\`go
func (t *DataMinerTemplate) Execute(path string) []string {
	results := make([]string, 0, 4)	// предвыделяем для эффективности

	// Скелет алгоритма - порядок фиксирован
	results = append(results, t.miner.OpenFile(path))	// всегда первый
	results = append(results, t.miner.ExtractData())	// всегда второй
	results = append(results, t.miner.ParseData())	// всегда третий
	results = append(results, t.miner.CloseFile())	// всегда последний

	return results	// возвращаем все результаты по порядку
}

// Реализация шага CSVMiner
func (m *CSVMiner) OpenFile(path string) string {
	return fmt.Sprintf("Opening CSV file: %s", path)
}

func (m *CSVMiner) ExtractData() string {
	return "Extracting CSV rows"	// формулировка для CSV
}
\`\`\`

Шаблон гарантирует структуру алгоритма, а miner'ы настраивают шаги.`,
			whyItMatters: `## Почему существует Template Method

**Проблема: Дублирование структуры алгоритма**

Без Template Method вы дублируете неизменную логику алгоритма:

\`\`\`go
// ❌ БЕЗ TEMPLATE METHOD - дублированная структура
func ProcessCSV(path string) []string {
	results := []string{}
	results = append(results, openCSV(path))	// дублированный порядок
	results = append(results, extractCSV())	// дублированный порядок
	results = append(results, parseCSV())	// дублированный порядок
	results = append(results, closeCSV())	// дублированный порядок
	return results
}

func ProcessJSON(path string) []string {
	results := []string{}
	results = append(results, openJSON(path))	// та же структура повторяется
	results = append(results, extractJSON())	// та же структура повторяется
	results = append(results, parseJSON())	// та же структура повторяется
	results = append(results, closeJSON())	// та же структура повторяется
	return results
}

// ✅ С TEMPLATE METHOD - одно определение алгоритма
func (t *DataMinerTemplate) Execute(path string) []string {
	results := make([]string, 0, 4)	// алгоритм определён один раз
	results = append(results, t.miner.OpenFile(path))
	results = append(results, t.miner.ExtractData())
	results = append(results, t.miner.ParseData())
	results = append(results, t.miner.CloseFile())
	return results	// меняются только шаги, не структура
}
\`\`\`

---

## Примеры из реального мира в Go

**1. HTTP Handler Middleware (net/http):**
\`\`\`go
// Шаблон: структура ServeHTTP
// Шаги: Аутентификация, Авторизация, Бизнес-логика, Логирование
\`\`\`

**2. Database/SQL (database/sql):**
\`\`\`go
// Шаблон: поток выполнения запроса
// Шаги: Connect, Prepare, Execute, Scan, Close
\`\`\`

**3. Тестирование (пакет testing):**
\`\`\`go
// Шаблон: выполнение теста
// Шаги: Setup, Run, Teardown, Report
\`\`\`

**4. Инструменты сборки (go build):**
\`\`\`go
// Шаблон: процесс сборки
// Шаги: Parse, Type-check, Generate, Link
\`\`\`

---

## Продакшн паттерн: HTTP Request Handler

\`\`\`go
package main

import (
	"fmt"
	"time"
)

// RequestHandler определяет настраиваемые шаги обработки запроса
type RequestHandler interface {
	ValidateRequest(req *Request) error	// шаг 1: валидация входных данных
	Authenticate(req *Request) error	// шаг 2: проверка авторизации
	Process(req *Request) *Response	// шаг 3: бизнес-логика
	FormatResponse(resp *Response) string	// шаг 4: форматирование вывода
}

// Request представляет входящий HTTP запрос
type Request struct {
	Method  string	// HTTP метод
	Path    string	// путь запроса
	Token   string	// токен авторизации
	Body    string	// тело запроса
}

// Response представляет исходящий HTTP ответ
type Response struct {
	Status  int	// HTTP код статуса
	Data    interface{}	// данные ответа
	Error   string	// сообщение об ошибке если есть
}

// RequestTemplate предоставляет шаблонный метод для обработки запросов
type RequestTemplate struct {
	handler RequestHandler	// конкретный обработчик
	logger  func(string)	// функция логирования
}

// Handle это шаблонный метод - неизменная структура алгоритма
func (t *RequestTemplate) Handle(req *Request) string {
	start := time.Now()	// отслеживаем длительность запроса

	// Шаг 1: Валидация (точка расширения)
	if err := t.handler.ValidateRequest(req); err != nil {
		t.logger(fmt.Sprintf("Validation failed: %v", err))
		return t.handler.FormatResponse(&Response{Status: 400, Error: err.Error()})
	}

	// Шаг 2: Аутентификация (точка расширения)
	if err := t.handler.Authenticate(req); err != nil {
		t.logger(fmt.Sprintf("Auth failed: %v", err))
		return t.handler.FormatResponse(&Response{Status: 401, Error: err.Error()})
	}

	// Шаг 3: Обработка (точка расширения)
	resp := t.handler.Process(req)

	// Шаг 4: Форматирование и возврат (точка расширения)
	result := t.handler.FormatResponse(resp)

	t.logger(fmt.Sprintf("Request completed in %v", time.Since(start)))
	return result
}

// APIHandler обрабатывает REST API запросы
type APIHandler struct{}

func (h *APIHandler) ValidateRequest(req *Request) error {
	if req.Method == "" || req.Path == "" {
		return fmt.Errorf("invalid request: missing method or path")
	}
	return nil
}

func (h *APIHandler) Authenticate(req *Request) error {
	if req.Token != "valid-token" {
		return fmt.Errorf("invalid or missing token")
	}
	return nil
}

func (h *APIHandler) Process(req *Request) *Response {
	return &Response{
		Status: 200,
		Data:   map[string]string{"message": "API request processed", "path": req.Path},
	}
}

func (h *APIHandler) FormatResponse(resp *Response) string {
	if resp.Error != "" {
		return fmt.Sprintf("{\\"status\\": %d, \\"error\\": \\"%s\\"}", resp.Status, resp.Error)
	}
	return fmt.Sprintf("{\\"status\\": %d, \\"data\\": %v}", resp.Status, resp.Data)
}

// WebhookHandler обрабатывает webhook запросы
type WebhookHandler struct{}

func (h *WebhookHandler) ValidateRequest(req *Request) error {
	if req.Body == "" {
		return fmt.Errorf("webhook body cannot be empty")
	}
	return nil
}

func (h *WebhookHandler) Authenticate(req *Request) error {
	// Webhooks могут использовать проверку подписи вместо токена
	return nil	// упрощено для примера
}

func (h *WebhookHandler) Process(req *Request) *Response {
	return &Response{
		Status: 202,	// accepted
		Data:   "Webhook queued for processing",
	}
}

func (h *WebhookHandler) FormatResponse(resp *Response) string {
	return fmt.Sprintf("Status: %d - %v", resp.Status, resp.Data)
}

// NewAPITemplate создаёт шаблон для API запросов
func NewAPITemplate(logger func(string)) *RequestTemplate {
	return &RequestTemplate{handler: &APIHandler{}, logger: logger}
}

// NewWebhookTemplate создаёт шаблон для webhook запросов
func NewWebhookTemplate(logger func(string)) *RequestTemplate {
	return &RequestTemplate{handler: &WebhookHandler{}, logger: logger}
}

// Использование
func main() {
	logger := func(msg string) { fmt.Println("[LOG]", msg) }

	apiTemplate := NewAPITemplate(logger)
	result := apiTemplate.Handle(&Request{
		Method: "GET",
		Path:   "/users",
		Token:  "valid-token",
	})
	fmt.Println(result)

	webhookTemplate := NewWebhookTemplate(logger)
	result2 := webhookTemplate.Handle(&Request{
		Method: "POST",
		Path:   "/webhook",
		Body:   "{\\"event\\": \\"user.created\\"}",
	})
	fmt.Println(result2)
}
\`\`\`

---

## Распространённые ошибки

**1. Template Method vs Strategy:**
\`\`\`go
// ❌ НЕПРАВИЛЬНО - Template Method когда лучше подходит Strategy
type Algorithm interface {
	Step1()
	Step2()
}
// Если порядок шагов не важен, используйте Strategy

// ✅ ПРАВИЛЬНО - Template Method для фиксированной структуры алгоритма
func (t *Template) Execute() {
	t.hook.Step1()	// порядок важен
	t.hook.Step2()	// должен следовать за step1
	t.hook.Step3()	// должен следовать за step2
}
\`\`\`

**2. Слишком много точек расширения:**
\`\`\`go
// ❌ НЕПРАВИЛЬНО - каждая строка это хук
type Handler interface {
	BeforeValidation()
	Validate()
	AfterValidation()
	BeforeProcess()
	Process()
	AfterProcess()
	// Слишком детализировано!
}

// ✅ ПРАВИЛЬНО - только значимые точки расширения
type Handler interface {
	Validate() error	// чёткая ответственность
	Process() *Result	// чёткая ответственность
	Format(*Result) string	// чёткая ответственность
}
\`\`\`

**3. Забытые вызовы хуков:**
\`\`\`go
// ❌ НЕПРАВИЛЬНО - хук не вызван
func (t *Template) Execute() {
	t.hook.Step1()
	// Забыли Step2!
	t.hook.Step3()
}

// ✅ ПРАВИЛЬНО - все хуки вызываются по порядку
func (t *Template) Execute() {
	t.hook.Step1()
	t.hook.Step2()	// не забывайте ни один шаг
	t.hook.Step3()
}
\`\`\``
		},
		uz: {
			title: 'Template Method (Shablon Metod) Pattern',
			description: `Go tilida Template Method patternini amalga oshiring — algoritm skeletini aniqlang, ba'zi qadamlarni pastki sinflarga qoldiring.

**Siz amalga oshirasiz:**

1. **DataMiner interfeysi** - Ma'lumotlarni qayta ishlash qadamlari
2. **DataMinerTemplate** - Execute metodi bilan shablon
3. **CSVMiner, JSONMiner** - Aniq realizatsiyalar

**Foydalanish namunasi:**

\`\`\`go
csvMiner := NewCSVMiner()	// shablon bilan CSV miner yaratamiz
result := csvMiner.Execute("data.csv")	// barcha qadamlarni tartib bilan bajaramiz

jsonMiner := NewJSONMiner()	// shablon bilan JSON miner yaratamiz
result2 := jsonMiner.Execute("config.json")	// xuddi shu algoritm, boshqa qadamlar

fmt.Println(result[0])	// "Opening CSV file: data.csv"
fmt.Println(result[1])	// "Extracting CSV rows"
fmt.Println(result[2])	// "Parsing CSV data"
fmt.Println(result[3])	// "Closing CSV file"
\`\`\``,
			hint1: `**Template Method tuzilmasini tushunish:**

Template Method patterni ikki asosiy komponentga ega:

\`\`\`go
// 1. Abstrakt interfeys - sozlanadigan qadamlarni aniqlaydi
type DataMiner interface {
	OpenFile(path string) string	// qadam fayl turiga bog'liq
	ExtractData() string	// ajratib olish formatga bog'liq
	ParseData() string	// tahlil qilish formatga bog'liq
	CloseFile() string	// tozalash fayl turiga bog'liq
}

// 2. Shablon - o'zgarmas algoritmni aniqlaydi
type DataMinerTemplate struct {
	miner DataMiner	// aniq qadam realizatsiyasini saqlaydi
}

// Shablon metod - belgilangan algoritm tuzilmasi
func (t *DataMinerTemplate) Execute(path string) []string {
	// Qadamlar doimo shu tartibda bajariladi
	// Faqat har bir qadam QANDAY bajarilishi o'zgaradi
}
\`\`\`

Har bir miner qadamlarni turlicha bajaradi, lekin algoritm tartibi hech qachon o'zgarmaydi.`,
			hint2: `**To'liq Execute realizatsiyasi:**

\`\`\`go
func (t *DataMinerTemplate) Execute(path string) []string {
	results := make([]string, 0, 4)	// samaradorlik uchun oldindan ajratamiz

	// Algoritm skeleti - tartib belgilangan
	results = append(results, t.miner.OpenFile(path))	// doimo birinchi
	results = append(results, t.miner.ExtractData())	// doimo ikkinchi
	results = append(results, t.miner.ParseData())	// doimo uchinchi
	results = append(results, t.miner.CloseFile())	// doimo oxirgi

	return results	// barcha natijalarni tartibda qaytaramiz
}

// CSVMiner qadam realizatsiyasi
func (m *CSVMiner) OpenFile(path string) string {
	return fmt.Sprintf("Opening CSV file: %s", path)
}

func (m *CSVMiner) ExtractData() string {
	return "Extracting CSV rows"	// CSV uchun so'z
}
\`\`\`

Shablon algoritm tuzilmasini kafolatlaydi, miner'lar esa qadamlarni sozlaydi.`,
			whyItMatters: `## Nima uchun Template Method mavjud

**Muammo: Algoritm tuzilmasining takrorlanishi**

Template Method'siz siz o'zgarmas algoritm mantiqini takrorlaysiz:

\`\`\`go
// ❌ TEMPLATE METHOD'SIZ - takrorlangan tuzilma
func ProcessCSV(path string) []string {
	results := []string{}
	results = append(results, openCSV(path))	// takrorlangan tartib
	results = append(results, extractCSV())	// takrorlangan tartib
	results = append(results, parseCSV())	// takrorlangan tartib
	results = append(results, closeCSV())	// takrorlangan tartib
	return results
}

func ProcessJSON(path string) []string {
	results := []string{}
	results = append(results, openJSON(path))	// xuddi shu tuzilma takrorlanadi
	results = append(results, extractJSON())	// xuddi shu tuzilma takrorlanadi
	results = append(results, parseJSON())	// xuddi shu tuzilma takrorlanadi
	results = append(results, closeJSON())	// xuddi shu tuzilma takrorlanadi
	return results
}

// ✅ TEMPLATE METHOD BILAN - bitta algoritm ta'rifi
func (t *DataMinerTemplate) Execute(path string) []string {
	results := make([]string, 0, 4)	// algoritm bir marta aniqlangan
	results = append(results, t.miner.OpenFile(path))
	results = append(results, t.miner.ExtractData())
	results = append(results, t.miner.ParseData())
	results = append(results, t.miner.CloseFile())
	return results	// faqat qadamlar o'zgaradi, tuzilma emas
}
\`\`\`

---

## Go'da haqiqiy dunyo misollari

**1. HTTP Handler Middleware (net/http):**
\`\`\`go
// Shablon: ServeHTTP tuzilmasi
// Qadamlar: Autentifikatsiya, Avtorizatsiya, Biznes-mantiq, Loglash
\`\`\`

**2. Database/SQL (database/sql):**
\`\`\`go
// Shablon: So'rov bajarish oqimi
// Qadamlar: Connect, Prepare, Execute, Scan, Close
\`\`\`

**3. Testlash (testing paketi):**
\`\`\`go
// Shablon: Test bajarish
// Qadamlar: Setup, Run, Teardown, Report
\`\`\`

**4. Build vositalari (go build):**
\`\`\`go
// Shablon: Build jarayoni
// Qadamlar: Parse, Type-check, Generate, Link
\`\`\`

---

## Production Pattern: HTTP Request Handler

\`\`\`go
package main

import (
	"fmt"
	"time"
)

// RequestHandler so'rovni qayta ishlashning sozlanadigan qadamlarini aniqlaydi
type RequestHandler interface {
	ValidateRequest(req *Request) error	// qadam 1: kirishni tekshirish
	Authenticate(req *Request) error	// qadam 2: autentifikatsiyani tekshirish
	Process(req *Request) *Response	// qadam 3: biznes-mantiq
	FormatResponse(resp *Response) string	// qadam 4: chiqishni formatlash
}

// Request kiruvchi HTTP so'rovini ifodalaydi
type Request struct {
	Method  string	// HTTP metod
	Path    string	// so'rov yo'li
	Token   string	// autentifikatsiya tokeni
	Body    string	// so'rov tanasi
}

// Response chiquvchi HTTP javobini ifodalaydi
type Response struct {
	Status  int	// HTTP status kodi
	Data    interface{}	// javob ma'lumotlari
	Error   string	// xato xabari agar bo'lsa
}

// RequestTemplate so'rovlarni qayta ishlash uchun shablon metodini taqdim etadi
type RequestTemplate struct {
	handler RequestHandler	// aniq handler
	logger  func(string)	// loglash funksiyasi
}

// Handle shablon metodi - o'zgarmas algoritm tuzilmasi
func (t *RequestTemplate) Handle(req *Request) string {
	start := time.Now()	// so'rov davomiyligini kuzatamiz

	// Qadam 1: Validatsiya (kengaytirish nuqtasi)
	if err := t.handler.ValidateRequest(req); err != nil {
		t.logger(fmt.Sprintf("Validation failed: %v", err))
		return t.handler.FormatResponse(&Response{Status: 400, Error: err.Error()})
	}

	// Qadam 2: Autentifikatsiya (kengaytirish nuqtasi)
	if err := t.handler.Authenticate(req); err != nil {
		t.logger(fmt.Sprintf("Auth failed: %v", err))
		return t.handler.FormatResponse(&Response{Status: 401, Error: err.Error()})
	}

	// Qadam 3: Qayta ishlash (kengaytirish nuqtasi)
	resp := t.handler.Process(req)

	// Qadam 4: Formatlash va qaytarish (kengaytirish nuqtasi)
	result := t.handler.FormatResponse(resp)

	t.logger(fmt.Sprintf("Request completed in %v", time.Since(start)))
	return result
}

// APIHandler REST API so'rovlarini qayta ishlaydi
type APIHandler struct{}

func (h *APIHandler) ValidateRequest(req *Request) error {
	if req.Method == "" || req.Path == "" {
		return fmt.Errorf("invalid request: missing method or path")
	}
	return nil
}

func (h *APIHandler) Authenticate(req *Request) error {
	if req.Token != "valid-token" {
		return fmt.Errorf("invalid or missing token")
	}
	return nil
}

func (h *APIHandler) Process(req *Request) *Response {
	return &Response{
		Status: 200,
		Data:   map[string]string{"message": "API request processed", "path": req.Path},
	}
}

func (h *APIHandler) FormatResponse(resp *Response) string {
	if resp.Error != "" {
		return fmt.Sprintf("{\\"status\\": %d, \\"error\\": \\"%s\\"}", resp.Status, resp.Error)
	}
	return fmt.Sprintf("{\\"status\\": %d, \\"data\\": %v}", resp.Status, resp.Data)
}

// WebhookHandler webhook so'rovlarini qayta ishlaydi
type WebhookHandler struct{}

func (h *WebhookHandler) ValidateRequest(req *Request) error {
	if req.Body == "" {
		return fmt.Errorf("webhook body cannot be empty")
	}
	return nil
}

func (h *WebhookHandler) Authenticate(req *Request) error {
	// Webhooks token o'rniga imzo tekshirishdan foydalanishi mumkin
	return nil	// misol uchun soddalashtirilgan
}

func (h *WebhookHandler) Process(req *Request) *Response {
	return &Response{
		Status: 202,	// qabul qilindi
		Data:   "Webhook queued for processing",
	}
}

func (h *WebhookHandler) FormatResponse(resp *Response) string {
	return fmt.Sprintf("Status: %d - %v", resp.Status, resp.Data)
}

// NewAPITemplate API so'rovlari uchun shablon yaratadi
func NewAPITemplate(logger func(string)) *RequestTemplate {
	return &RequestTemplate{handler: &APIHandler{}, logger: logger}
}

// NewWebhookTemplate webhook so'rovlari uchun shablon yaratadi
func NewWebhookTemplate(logger func(string)) *RequestTemplate {
	return &RequestTemplate{handler: &WebhookHandler{}, logger: logger}
}

// Foydalanish
func main() {
	logger := func(msg string) { fmt.Println("[LOG]", msg) }

	apiTemplate := NewAPITemplate(logger)
	result := apiTemplate.Handle(&Request{
		Method: "GET",
		Path:   "/users",
		Token:  "valid-token",
	})
	fmt.Println(result)

	webhookTemplate := NewWebhookTemplate(logger)
	result2 := webhookTemplate.Handle(&Request{
		Method: "POST",
		Path:   "/webhook",
		Body:   "{\\"event\\": \\"user.created\\"}",
	})
	fmt.Println(result2)
}
\`\`\`

---

## Keng tarqalgan xatolar

**1. Template Method vs Strategy:**
\`\`\`go
// ❌ NOTO'G'RI - Strategy yaxshiroq mos kelganda Template Method
type Algorithm interface {
	Step1()
	Step2()
}
// Agar qadamlar tartibi muhim bo'lmasa, Strategy ishlating

// ✅ TO'G'RI - Belgilangan algoritm tuzilmasi uchun Template Method
func (t *Template) Execute() {
	t.hook.Step1()	// tartib muhim
	t.hook.Step2()	// step1 dan keyin bo'lishi kerak
	t.hook.Step3()	// step2 dan keyin bo'lishi kerak
}
\`\`\`

**2. Juda ko'p kengaytirish nuqtalari:**
\`\`\`go
// ❌ NOTO'G'RI - har bir qator hook
type Handler interface {
	BeforeValidation()
	Validate()
	AfterValidation()
	BeforeProcess()
	Process()
	AfterProcess()
	// Juda batafsil!
}

// ✅ TO'G'RI - faqat mazmunli kengaytirish nuqtalari
type Handler interface {
	Validate() error	// aniq mas'uliyat
	Process() *Result	// aniq mas'uliyat
	Format(*Result) string	// aniq mas'uliyat
}
\`\`\`

**3. Unutilgan hook chaqiruvlari:**
\`\`\`go
// ❌ NOTO'G'RI - hook chaqirilmagan
func (t *Template) Execute() {
	t.hook.Step1()
	// Step2 ni unutdik!
	t.hook.Step3()
}

// ✅ TO'G'RI - barcha hooklar tartibda chaqiriladi
func (t *Template) Execute() {
	t.hook.Step1()
	t.hook.Step2()	// hech bir qadamni unutmang
	t.hook.Step3()
}
\`\`\``
		}
	}
};

export default task;
