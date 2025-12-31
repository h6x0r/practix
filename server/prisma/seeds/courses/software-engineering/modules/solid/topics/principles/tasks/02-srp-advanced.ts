import { Task } from '../../../../../../../types';

export const task: Task = {
	slug: 'se-solid-srp-advanced',
	title: 'Single Responsibility Principle - Advanced',
	difficulty: 'medium',
	tags: ['go', 'solid', 'srp', 'clean-code', 'advanced'],
	estimatedTime: '35m',
	isPremium: false,
	youtubeUrl: '',
	description: `Apply SRP to a complex Report system with multiple cross-cutting concerns.

**Current Problem:**

A Report struct handles:
- Data aggregation from database
- Business logic (calculations, filtering)
- Multiple export formats (PDF, Excel, CSV)
- Email delivery
- Audit logging

**Your task:**

Refactor into focused components following SRP:

1. **Report struct** - Only data container (Title, Data, CreatedAt)
2. **ReportGenerator** - Aggregates data from database
3. **ReportCalculator** - Performs business calculations
4. **PDFExporter** - Exports to PDF format
5. **ExcelExporter** - Exports to Excel format
6. **ReportDelivery** - Handles email delivery
7. **AuditLogger** - Logs report operations

**Key Concepts:**
- **Cohesion**: Each component has highly related functionality
- **Coupling**: Components communicate through interfaces
- **Composition**: Build complex behavior from simple parts

**Example Usage:**

\`\`\`go
// Generate report data
generator := &ReportGenerator{db: db}
reportData := generator.Generate("sales", startDate, endDate)

// Create report
calculator := &ReportCalculator{}
report := calculator.Calculate(reportData)

// Export to PDF
pdfExporter := &PDFExporter{}
pdfData := pdfExporter.Export(report)

// Deliver via email
delivery := &ReportDelivery{}
delivery.Send("user@example.com", pdfData)

// Log the operation
logger := &AuditLogger{}
logger.Log("report_generated", report.Title)
\`\`\`

**Real-World Impact:**
- Need to add JSON export? Create JsonExporter without touching PDF/Excel
- Change email provider? Modify only ReportDelivery
- Add caching to data? Change only ReportGenerator
- Switch logging system? Update only AuditLogger

**Constraints:**
- Each component must have a single, well-defined purpose
- Components should be independently testable
- Use composition to build complex workflows`,
	initialCode: `package principles

import (
	"database/sql"
	"fmt"
	"time"
)

type Report struct {
	Title     string
	CreatedAt time.Time
}

func (r *Report) GenerateFromDB(db *sql.DB, reportType string, start, end time.Time) error {
	if err != nil {
		return err
	}
	defer rows.Close()

	}
	return nil
}

func (r *Report) CalculateTotals() map[string]float64 {
		}
		}
	}
}

func (r *Report) ExportToPDF() ([]byte, error) {
}

func (r *Report) ExportToExcel() ([]byte, error) {
}

func (r *Report) SendEmail(to string, attachment []byte) error {
	return nil
}

func (r *Report) LogAudit(action string) error {
	return nil
}

type ReportGenerator struct {
	db *sql.DB
}

func (rg *ReportGenerator) Generate(reportType string, start, end time.Time) []map[string]interface{} {
}

type ReportCalculator struct{}

func (rc *ReportCalculator) Calculate(data []map[string]interface{}) *Report {
}

type PDFExporter struct{}

func (pe *PDFExporter) Export(report *Report) ([]byte, error) {
}

type ExcelExporter struct{}

func (ee *ExcelExporter) Export(report *Report) ([]byte, error) {
}

type ReportDelivery struct{}

func (rd *ReportDelivery) Send(to string, attachment []byte) error {
}

type AuditLogger struct{}

func (al *AuditLogger) Log(action, details string) error {
}`,
	solutionCode: `package principles

import (
	"database/sql"
	"fmt"
	"time"
)

// Report is now just a data container
// Single responsibility: hold report data
type Report struct {
	Title     string				// report title
	Data      []map[string]interface{}	// raw report data
	Totals    map[string]float64		// calculated totals
	CreatedAt time.Time			// creation timestamp
}

// ReportGenerator has single responsibility: fetch and aggregate data
// Changes to data source only affect this component
type ReportGenerator struct {
	db *sql.DB	// database connection for data fetching
}

// Generate fetches data from database based on report type and date range
func (rg *ReportGenerator) Generate(reportType string, start, end time.Time) []map[string]interface{} {
	// Query database for report data
	query := fmt.Sprintf("SELECT * FROM %s WHERE date BETWEEN ? AND ?", reportType)
	rows, err := rg.db.Query(query, start, end)
	if err != nil {
		return nil	// in production, handle error properly
	}
	defer rows.Close()

	// Aggregate results (simplified for example)
	data := []map[string]interface{}{
		{"revenue": 50000.0, "costs": 30000.0},	// sample data
		{"revenue": 45000.0, "costs": 28000.0},
	}
	return data	// return aggregated data
}

// ReportCalculator has single responsibility: perform business calculations
// Changes to calculation logic only affect this component
type ReportCalculator struct{}

// Calculate creates report with computed totals from raw data
func (rc *ReportCalculator) Calculate(data []map[string]interface{}) *Report {
	report := &Report{
		Title:     "Sales Report",
		Data:      data,
		Totals:    make(map[string]float64),
		CreatedAt: time.Now(),
	}

	// Perform business calculations
	for _, row := range data {
		if revenue, ok := row["revenue"].(float64); ok {
			report.Totals["total_revenue"] += revenue	// sum revenues
		}
		if costs, ok := row["costs"].(float64); ok {
			report.Totals["total_costs"] += costs	// sum costs
		}
	}
	report.Totals["profit"] = report.Totals["total_revenue"] - report.Totals["total_costs"]

	return report	// return calculated report
}

// PDFExporter has single responsibility: export reports to PDF format
// Changes to PDF library or formatting only affect this component
type PDFExporter struct{}

// Export converts report to PDF format
func (pe *PDFExporter) Export(report *Report) ([]byte, error) {
	// In production, use actual PDF library (e.g., gofpdf)
	pdfContent := fmt.Sprintf(
		"PDF Report: %s\\nCreated: %v\\nData: %v\\nTotals: %v",
		report.Title, report.CreatedAt, report.Data, report.Totals,
	)
	return []byte(pdfContent), nil	// return PDF bytes
}

// ExcelExporter has single responsibility: export reports to Excel format
// Changes to Excel library or formatting only affect this component
type ExcelExporter struct{}

// Export converts report to Excel format
func (ee *ExcelExporter) Export(report *Report) ([]byte, error) {
	// In production, use actual Excel library (e.g., excelize)
	excelContent := fmt.Sprintf(
		"Excel Report: %s\\nCreated: %v\\nData: %v\\nTotals: %v",
		report.Title, report.CreatedAt, report.Data, report.Totals,
	)
	return []byte(excelContent), nil	// return Excel bytes
}

// ReportDelivery has single responsibility: deliver reports via email
// Changes to email provider only affect this component
type ReportDelivery struct{}

// Send delivers report attachment to recipient via email
func (rd *ReportDelivery) Send(to string, attachment []byte) error {
	// In production, use email service (e.g., SendGrid, AWS SES)
	fmt.Printf("Sending email to %s\\n", to)
	fmt.Printf("Attachment size: %d bytes\\n", len(attachment))
	return nil	// return nil on success
}

// AuditLogger has single responsibility: record audit trail
// Changes to logging system only affect this component
type AuditLogger struct{}

// Log records audit event with action and details
func (al *AuditLogger) Log(action, details string) error {
	// In production, write to audit log system (database, file, etc.)
	timestamp := time.Now().Format(time.RFC3339)
	fmt.Printf("[AUDIT] %s | %s | %s\\n", timestamp, action, details)
	return nil	// return nil on success
}`,
	hint1: `For ReportGenerator.Generate, copy the database query logic from GenerateFromDB but return just the data slice. For ReportCalculator.Calculate, create a new Report, copy the calculation logic from CalculateTotals, and store results in report.Totals.`,
	hint2: `For exporters (PDF/Excel), copy their respective export logic but take *Report as parameter. For ReportDelivery.Send and AuditLogger.Log, copy their logic but make them standalone without accessing report fields directly.`,
	testCode: `package principles

import (
	"strings"
	"testing"
	"time"
)

// Test1: ReportCalculator creates report with totals
func Test1(t *testing.T) {
	rc := &ReportCalculator{}
	data := []map[string]interface{}{
		{"revenue": 1000.0, "costs": 500.0},
	}
	report := rc.Calculate(data)
	if report == nil {
		t.Fatal("Calculate returned nil")
	}
	if report.Totals["profit"] != 500 {
		t.Errorf("profit = %v, want 500", report.Totals["profit"])
	}
}

// Test2: PDFExporter returns non-empty bytes
func Test2(t *testing.T) {
	pe := &PDFExporter{}
	report := &Report{Title: "Test", CreatedAt: time.Now()}
	data, err := pe.Export(report)
	if err != nil {
		t.Fatalf("Export error: %v", err)
	}
	if len(data) == 0 {
		t.Error("PDF data should not be empty")
	}
}

// Test3: ExcelExporter returns non-empty bytes
func Test3(t *testing.T) {
	ee := &ExcelExporter{}
	report := &Report{Title: "Test", CreatedAt: time.Now()}
	data, err := ee.Export(report)
	if err != nil {
		t.Fatalf("Export error: %v", err)
	}
	if len(data) == 0 {
		t.Error("Excel data should not be empty")
	}
}

// Test4: ReportDelivery.Send returns nil on success
func Test4(t *testing.T) {
	rd := &ReportDelivery{}
	err := rd.Send("test@example.com", []byte("test data"))
	if err != nil {
		t.Errorf("Send error: %v", err)
	}
}

// Test5: AuditLogger.Log returns nil on success
func Test5(t *testing.T) {
	al := &AuditLogger{}
	err := al.Log("test_action", "test details")
	if err != nil {
		t.Errorf("Log error: %v", err)
	}
}

// Test6: Report struct holds data correctly
func Test6(t *testing.T) {
	report := &Report{
		Title:     "Sales Report",
		CreatedAt: time.Now(),
	}
	if report.Title != "Sales Report" {
		t.Errorf("Title = %q, want Sales Report", report.Title)
	}
}

// Test7: ReportCalculator handles multiple rows
func Test7(t *testing.T) {
	rc := &ReportCalculator{}
	data := []map[string]interface{}{
		{"revenue": 1000.0, "costs": 400.0},
		{"revenue": 2000.0, "costs": 600.0},
	}
	report := rc.Calculate(data)
	if report.Totals["total_revenue"] != 3000 {
		t.Errorf("total_revenue = %v, want 3000", report.Totals["total_revenue"])
	}
}

// Test8: PDFExporter includes report title
func Test8(t *testing.T) {
	pe := &PDFExporter{}
	report := &Report{Title: "MyReport", CreatedAt: time.Now()}
	data, _ := pe.Export(report)
	if !strings.Contains(string(data), "MyReport") {
		t.Error("PDF should contain report title")
	}
}

// Test9: Empty data produces zero totals
func Test9(t *testing.T) {
	rc := &ReportCalculator{}
	report := rc.Calculate([]map[string]interface{}{})
	if report.Totals["profit"] != 0 {
		t.Errorf("empty data should have 0 profit, got %v", report.Totals["profit"])
	}
}

// Test10: All components are independently usable
func Test10(t *testing.T) {
	_ = &ReportGenerator{}
	_ = &ReportCalculator{}
	_ = &PDFExporter{}
	_ = &ExcelExporter{}
	_ = &ReportDelivery{}
	_ = &AuditLogger{}
	// All should be separate, independent types
}
`,
	whyItMatters: `Advanced SRP application shows how to manage complex systems with multiple cross-cutting concerns.

**Why Advanced SRP Matters:**

**1. Real-World Complexity: E-commerce Order Processing**

\`\`\`go
// VIOLATES SRP - Order does everything
type Order struct {
	Items []Item
}

func (o *Order) Process() error {
	// 1. Validate inventory
	for _, item := range o.Items {
		if !checkStock(item) {
			return errors.New("out of stock")
		}
	}

	// 2. Calculate pricing
	total := 0.0
	for _, item := range o.Items {
		total += item.Price
		if item.OnSale { total *= 0.9 }
	}

	// 3. Process payment
	if err := chargeCard(total); err != nil {
		return err
	}

	// 4. Update inventory
	for _, item := range o.Items {
		decrementStock(item)
	}

	// 5. Send confirmation email
	sendEmail("Order confirmed")

	// 6. Create shipment
	createShipment(o.Items)

	// 7. Log analytics
	logEvent("order_completed", total)

	return nil
}
// Problem: To change email template, you must touch code that handles payments and inventory!

// FOLLOWS SRP - Separated concerns
type InventoryChecker struct {
	repo InventoryRepository
}
func (ic *InventoryChecker) CheckAvailability(items []Item) error {
	// Only checks inventory - nothing else
}

type PricingCalculator struct {
	discounts DiscountService
}
func (pc *PricingCalculator) Calculate(items []Item) float64 {
	// Only calculates prices - nothing else
}

type PaymentProcessor struct {
	gateway PaymentGateway
}
func (pp *PaymentProcessor) Charge(amount float64) error {
	// Only processes payments - nothing else
}

type InventoryUpdater struct {
	repo InventoryRepository
}
func (iu *InventoryUpdater) Decrement(items []Item) error {
	// Only updates inventory - nothing else
}

type OrderNotifier struct {
	emailService EmailService
}
func (on *OrderNotifier) SendConfirmation(order *Order) error {
	// Only sends notifications - nothing else
}

type ShipmentCreator struct {
	shipper ShippingService
}
func (sc *ShipmentCreator) Create(items []Item) error {
	// Only creates shipments - nothing else
}

type AnalyticsLogger struct {
	tracker Analytics
}
func (al *AnalyticsLogger) LogOrder(total float64) error {
	// Only logs analytics - nothing else
}

// Orchestrate with a coordinator (not shown here)
// Each component can evolve independently!
\`\`\`

**2. Testing Benefits: Isolated Unit Tests**

\`\`\`go
// WITHOUT SRP - Must mock everything to test pricing
func TestOrderProcess(t *testing.T) {
	// To test pricing calculation, must mock:
	// - Database for inventory check
	// - Payment gateway
	// - Email service
	// - Shipping API
	// - Analytics service
	// Complex, slow, brittle tests
}

// WITH SRP - Test only what matters
func TestPricingCalculator(t *testing.T) {
	calc := &PricingCalculator{}

	items := []Item{
		{Price: 100, OnSale: true},   // 90 after discount
		{Price: 200, OnSale: false},  // 200 no discount
	}

	total := calc.Calculate(items)
	if total != 290 {
		t.Errorf("expected 290, got %f", total)
	}
	// Fast, simple, focused test - no mocks needed!
}
\`\`\`

**3. Maintenance: Independent Evolution**

\`\`\`go
// Scenario: Switch from SMTP to SendGrid for emails

// WITH SRP - Change only EmailService
type EmailService struct {
	// client smtp.Client	// OLD
	client *sendgrid.Client	// NEW - only this file changes
}

// WITHOUT SRP - Change Order type
type Order struct {
	// ... 500 lines of code ...
	// smtpClient smtp.Client	// OLD - buried in massive struct
	// sendgridClient *sendgrid.Client	// NEW - affects all order code
}
// Risk: Break unrelated features while changing email
\`\`\`

**4. Real Production Example: Report System**

\`\`\`go
// Initial requirement: PDF reports via email

// WITHOUT SRP
type Report struct{}
func (r *Report) Generate() {
	data := fetchData()
	pdf := createPDF(data)
	sendEmail(pdf)
}

// New requirement: Add Excel export
func (r *Report) Generate() {
	data := fetchData()
	pdf := createPDF(data)
	excel := createExcel(data)  // NEW - added to existing method
	sendEmail(pdf)
	sendEmail(excel)  // NEW
}

// New requirement: Add Slack delivery
func (r *Report) Generate() {
	data := fetchData()
	pdf := createPDF(data)
	excel := createExcel(data)
	sendEmail(pdf)
	sendEmail(excel)
	sendSlack(pdf)  // NEW - Generate() keeps growing!
}
// Method becomes unmaintainable monster

// WITH SRP - Easy to extend
type ReportGenerator struct{}		// Fetches data
type PDFExporter struct{}		// Exports PDF
type ExcelExporter struct{}		// Exports Excel  <- just add this
type EmailDelivery struct{}		// Delivers via email
type SlackDelivery struct{}		// Delivers via Slack  <- just add this

// Workflow orchestration (separate concern)
func CreateAndDeliverReport() {
	data := generator.Generate()

	pdf := pdfExporter.Export(data)
	excel := excelExporter.Export(data)  // NEW - separate component

	emailDelivery.Send(pdf)
	emailDelivery.Send(excel)
	slackDelivery.Send(pdf)  // NEW - separate component
}
// Each component added independently, no changes to existing code!
\`\`\`

**5. Recognizing SRP Violations in Complex Systems**

Signs your code violates SRP at scale:
- Files over 500 lines for single type
- Type name is generic ("Manager", "Service", "Handler", "Processor")
- Needs many imports (database, email, payment, logging, analytics)
- Method names have "And" (ProcessAndEmail, SaveAndNotify)
- Tests need >3 mocks
- Changes to one feature break unrelated features
- Multiple teams can't work on same file simultaneously

**When to Apply Advanced SRP:**
- Building complex business workflows
- Integrating multiple external services
- High-change-frequency features
- Systems with multiple delivery formats
- When testing becomes painful`,
	order: 1,
	translations: {
		ru: {
			title: 'Принцип единственной ответственности - Продвинутый',
			description: `Примените SRP к сложной системе отчётов с множественными сквозными задачами.

**Текущая проблема:**

Структура Report обрабатывает:
- Агрегацию данных из БД
- Бизнес-логику (вычисления, фильтрацию)
- Множественные форматы экспорта (PDF, Excel, CSV)
- Доставку email
- Аудит-логирование

**Ваша задача:**

Рефакторить в сфокусированные компоненты следуя SRP:

1. **Report struct** - Только контейнер данных (Title, Data, CreatedAt)
2. **ReportGenerator** - Агрегирует данные из БД
3. **ReportCalculator** - Выполняет бизнес-вычисления
4. **PDFExporter** - Экспортирует в формат PDF
5. **ExcelExporter** - Экспортирует в формат Excel
6. **ReportDelivery** - Обрабатывает доставку email
7. **AuditLogger** - Логирует операции с отчётами`,
			hint1: `Для ReportGenerator.Generate скопируйте логику запроса к БД из GenerateFromDB, но возвращайте только slice данных. Для ReportCalculator.Calculate создайте новый Report, скопируйте логику вычислений из CalculateTotals и сохраните результаты в report.Totals.`,
			hint2: `Для экспортеров (PDF/Excel) скопируйте их соответствующую логику экспорта, но принимайте *Report как параметр. Для ReportDelivery.Send и AuditLogger.Log скопируйте их логику, но сделайте их автономными без прямого доступа к полям отчёта.`,
			whyItMatters: `Продвинутое применение SRP показывает, как управлять сложными системами с множественными сквозными задачами.

**Почему важен продвинутый SRP:**

**1. Реальная сложность: Обработка заказов в e-commerce**

Без SRP тип Order делает всё: проверяет инвентарь, рассчитывает цены, обрабатывает платежи, обновляет склад, отправляет email, создаёт отправку, логирует аналитику.

С SRP каждая ответственность — отдельный компонент:
- InventoryChecker - только проверяет наличие
- PricingCalculator - только рассчитывает цены
- PaymentProcessor - только обрабатывает платежи
- И так далее...

Каждый компонент может развиваться независимо!`,
			solutionCode: `package principles

import (
	"database/sql"
	"fmt"
	"time"
)

// Report теперь просто контейнер данных
// Единственная ответственность: хранить данные отчёта
type Report struct {
	Title     string				// название отчёта
	Data      []map[string]interface{}	// сырые данные отчёта
	Totals    map[string]float64		// вычисленные итоги
	CreatedAt time.Time			// временная метка создания
}

// ReportGenerator имеет единственную ответственность: получать и агрегировать данные
// Изменения источника данных влияют только на этот компонент
type ReportGenerator struct {
	db *sql.DB	// подключение к БД для получения данных
}

// Generate получает данные из БД на основе типа отчёта и диапазона дат
func (rg *ReportGenerator) Generate(reportType string, start, end time.Time) []map[string]interface{} {
	// Запрашиваем БД для данных отчёта
	query := fmt.Sprintf("SELECT * FROM %s WHERE date BETWEEN ? AND ?", reportType)
	rows, err := rg.db.Query(query, start, end)
	if err != nil {
		return nil	// в продакшене обрабатывайте ошибку правильно
	}
	defer rows.Close()

	// Агрегируем результаты (упрощено для примера)
	data := []map[string]interface{}{
		{"revenue": 50000.0, "costs": 30000.0},	// пример данных
		{"revenue": 45000.0, "costs": 28000.0},
	}
	return data	// возвращаем агрегированные данные
}

// ReportCalculator имеет единственную ответственность: выполнять бизнес-вычисления
// Изменения логики вычислений влияют только на этот компонент
type ReportCalculator struct{}

// Calculate создаёт отчёт с вычисленными итогами из сырых данных
func (rc *ReportCalculator) Calculate(data []map[string]interface{}) *Report {
	report := &Report{
		Title:     "Отчёт по продажам",
		Data:      data,
		Totals:    make(map[string]float64),
		CreatedAt: time.Now(),
	}

	// Выполняем бизнес-вычисления
	for _, row := range data {
		if revenue, ok := row["revenue"].(float64); ok {
			report.Totals["total_revenue"] += revenue	// суммируем доходы
		}
		if costs, ok := row["costs"].(float64); ok {
			report.Totals["total_costs"] += costs	// суммируем расходы
		}
	}
	report.Totals["profit"] = report.Totals["total_revenue"] - report.Totals["total_costs"]

	return report	// возвращаем вычисленный отчёт
}

// PDFExporter имеет единственную ответственность: экспортировать отчёты в формат PDF
type PDFExporter struct{}

func (pe *PDFExporter) Export(report *Report) ([]byte, error) {
	pdfContent := fmt.Sprintf(
		"PDF Отчёт: %s\\nСоздан: %v\\nДанные: %v\\nИтоги: %v",
		report.Title, report.CreatedAt, report.Data, report.Totals,
	)
	return []byte(pdfContent), nil
}

// ExcelExporter имеет единственную ответственность: экспортировать отчёты в формат Excel
type ExcelExporter struct{}

func (ee *ExcelExporter) Export(report *Report) ([]byte, error) {
	excelContent := fmt.Sprintf(
		"Excel Отчёт: %s\\nСоздан: %v\\nДанные: %v\\nИтоги: %v",
		report.Title, report.CreatedAt, report.Data, report.Totals,
	)
	return []byte(excelContent), nil
}

// ReportDelivery имеет единственную ответственность: доставлять отчёты через email
type ReportDelivery struct{}

func (rd *ReportDelivery) Send(to string, attachment []byte) error {
	fmt.Printf("Отправка email на %s\\n", to)
	fmt.Printf("Размер вложения: %d байт\\n", len(attachment))
	return nil
}

// AuditLogger имеет единственную ответственность: записывать аудит-логи
type AuditLogger struct{}

func (al *AuditLogger) Log(action, details string) error {
	timestamp := time.Now().Format(time.RFC3339)
	fmt.Printf("[AUDIT] %s | %s | %s\\n", timestamp, action, details)
	return nil
}`
		},
		uz: {
			title: 'Yagona mas\'uliyat printsipi - Kengaytirilgan',
			description: `Ko'plab kesishuvchi muammolar bilan murakkab Hisobot tizimiga SRP ni qo'llang.

**Hozirgi muammo:**

Report strukturasi quyidagilarni bajaradi:
- Ma'lumotlar bazasidan ma'lumotlarni agregatsiya qilish
- Biznes mantiqi (hisoblashlar, filtrlash)
- Ko'plab eksport formatlari (PDF, Excel, CSV)
- Email yetkazib berish
- Audit loglash

**Sizning vazifangiz:**

SRP ga rioya qilgan holda e'tiborli komponentlarga refaktoring qiling:

1. **Report struct** - Faqat ma'lumotlar konteyineri (Title, Data, CreatedAt)
2. **ReportGenerator** - Ma'lumotlar bazasidan ma'lumotlarni agregatsiya qiladi
3. **ReportCalculator** - Biznes hisoblashlarini bajaradi
4. **PDFExporter** - PDF formatiga eksport qiladi
5. **ExcelExporter** - Excel formatiga eksport qiladi
6. **ReportDelivery** - Email yetkazib berishni boshqaradi
7. **AuditLogger** - Hisobot operatsiyalarini loglaydi`,
			hint1: `ReportGenerator.Generate uchun GenerateFromDB dan ma'lumotlar bazasi so'rovi mantiqini nusxalang, lekin faqat ma'lumotlar slice ini qaytaring. ReportCalculator.Calculate uchun yangi Report yarating, CalculateTotals dan hisoblash mantiqini nusxalang va natijalarni report.Totals ga saqlang.`,
			hint2: `Eksportchilar (PDF/Excel) uchun ularning tegishli eksport mantiqini nusxalang, lekin parametr sifatida *Report qabul qiling. ReportDelivery.Send va AuditLogger.Log uchun ularning mantiqini nusxalang, lekin hisobot maydonlariga to'g'ridan-to'g'ri kirmasdan mustaqil qiling.`,
			whyItMatters: `Kengaytirilgan SRP qo'llanilishi ko'plab kesishuvchi muammolar bilan murakkab tizimlarni qanday boshqarishni ko'rsatadi.

**Kengaytirilgan SRP nima uchun muhim:**

**1. Haqiqiy murakkablik: E-commerce buyurtmalarni qayta ishlash**

SRP siz Order turi hamma narsani qiladi: inventarni tekshiradi, narxlarni hisobla ydi, to'lovlarni qayta ishlaydi, omborni yangilaydi, email yuboradi, jo'natma yaratadi, analitikani loglaydi.

SRP bilan har bir mas'uliyat alohida komponent:
- InventoryChecker - faqat mavjudlikni tekshiradi
- PricingCalculator - faqat narxlarni hisobla ydi
- PaymentProcessor - faqat to'lovlarni qayta ishlaydi
- Va hokazo...

Har bir komponent mustaqil rivojlanishi mumkin!`,
			solutionCode: `package principles

import (
	"database/sql"
	"fmt"
	"time"
)

// Report endi shunchaki ma'lumotlar konteyineri
// Yagona mas'uliyat: hisobot ma'lumotlarini saqlash
type Report struct {
	Title     string				// hisobot nomi
	Data      []map[string]interface{}	// xom hisobot ma'lumotlari
	Totals    map[string]float64		// hisoblangan jami
	CreatedAt time.Time			// yaratilish vaqt belgisi
}

// ReportGenerator yagona mas'uliyatga ega: ma'lumotlarni olish va agregatsiya qilish
type ReportGenerator struct {
	db *sql.DB	// ma'lumotlarni olish uchun ma'lumotlar bazasi ulanishi
}

func (rg *ReportGenerator) Generate(reportType string, start, end time.Time) []map[string]interface{} {
	query := fmt.Sprintf("SELECT * FROM %s WHERE date BETWEEN ? AND ?", reportType)
	rows, err := rg.db.Query(query, start, end)
	if err != nil {
		return nil
	}
	defer rows.Close()

	data := []map[string]interface{}{
		{"revenue": 50000.0, "costs": 30000.0},
		{"revenue": 45000.0, "costs": 28000.0},
	}
	return data
}

// ReportCalculator yagona mas'uliyatga ega: biznes hisoblashlarini bajarish
type ReportCalculator struct{}

func (rc *ReportCalculator) Calculate(data []map[string]interface{}) *Report {
	report := &Report{
		Title:     "Sotish hisoboti",
		Data:      data,
		Totals:    make(map[string]float64),
		CreatedAt: time.Now(),
	}

	for _, row := range data {
		if revenue, ok := row["revenue"].(float64); ok {
			report.Totals["total_revenue"] += revenue
		}
		if costs, ok := row["costs"].(float64); ok {
			report.Totals["total_costs"] += costs
		}
	}
	report.Totals["profit"] = report.Totals["total_revenue"] - report.Totals["total_costs"]

	return report
}

// PDFExporter yagona mas'uliyatga ega: hisobotlarni PDF formatiga eksport qilish
type PDFExporter struct{}

func (pe *PDFExporter) Export(report *Report) ([]byte, error) {
	pdfContent := fmt.Sprintf(
		"PDF Hisobot: %s\\nYaratildi: %v\\nMa'lumotlar: %v\\nJami: %v",
		report.Title, report.CreatedAt, report.Data, report.Totals,
	)
	return []byte(pdfContent), nil
}

// ExcelExporter yagona mas'uliyatga ega: hisobotlarni Excel formatiga eksport qilish
type ExcelExporter struct{}

func (ee *ExcelExporter) Export(report *Report) ([]byte, error) {
	excelContent := fmt.Sprintf(
		"Excel Hisobot: %s\\nYaratildi: %v\\nMa'lumotlar: %v\\nJami: %v",
		report.Title, report.CreatedAt, report.Data, report.Totals,
	)
	return []byte(excelContent), nil
}

// ReportDelivery yagona mas'uliyatga ega: hisobotlarni email orqali yetkazib berish
type ReportDelivery struct{}

func (rd *ReportDelivery) Send(to string, attachment []byte) error {
	fmt.Printf("%s ga email yuborilmoqda\\n", to)
	fmt.Printf("Biriktirma hajmi: %d bayt\\n", len(attachment))
	return nil
}

// AuditLogger yagona mas'uliyatga ega: audit loglarini yozish
type AuditLogger struct{}

func (al *AuditLogger) Log(action, details string) error {
	timestamp := time.Now().Format(time.RFC3339)
	fmt.Printf("[AUDIT] %s | %s | %s\\n", timestamp, action, details)
	return nil
}`
		}
	}
};

export default task;
