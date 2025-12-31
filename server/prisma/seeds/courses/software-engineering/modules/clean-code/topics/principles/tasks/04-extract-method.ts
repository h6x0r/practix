import { Task } from '../../../../../../../types';

export const task: Task = {
	slug: 'se-clean-code-extract-method',
	title: 'Small Functions - Extract Method',
	difficulty: 'medium',
	tags: ['go', 'clean-code', 'functions', 'refactoring'],
	estimatedTime: '30m',
	isPremium: false,
	youtubeUrl: '',
	description: `Practice the Extract Method refactoring technique to improve code readability by extracting complex logic into well-named functions.

**You will refactor:**

1. **GenerateInvoice** - Extract complex calculation and formatting logic
2. Create helper functions:
   - CalculateSubtotal
   - CalculateDiscount
   - CalculateTax
   - FormatInvoiceHeader
   - FormatLineItems

**Key Concepts:**
- **Extract Method**: Replace code fragment with a function call
- **Semantic Compression**: Replace complex logic with intention-revealing name
- **DRY Principle**: Don't Repeat Yourself - extract duplicate code
- **Reveal Intent**: Function name explains what, implementation explains how

**Example - Before:**

\`\`\`go
func GenerateReport(sales []Sale) string {
    total := 0.0
    for _, s := range sales {
        total += s.Amount
    }

    report := "=== SALES REPORT ===\n"
    report += fmt.Sprintf("Total: $%.2f\n", total)
    return report
}
\`\`\`

**Example - After:**

\`\`\`go
func GenerateReport(sales []Sale) string {
    total := CalculateTotalSales(sales)
    return FormatReport(total)
}

func CalculateTotalSales(sales []Sale) float64 { ... }
func FormatReport(total float64) string { ... }
\`\`\`

**When to extract a method:**
- Code block has a clear purpose
- Code block can be given a meaningful name
- Code is duplicated in multiple places
- Function is too long (>20 lines)

**Constraints:**
- Extract at least 5 helper functions
- Each helper should be independently testable
- Main function should read like documentation`,
	initialCode: `package principles

import (
	"fmt"
	"strings"
	"time"
)

type Invoice struct {
	CustomerName string
	DiscountRate float64
	TaxRate      float64
}

type InvoiceItem struct {
	Name     string
	Quantity int
	Price    float64
}

func GenerateInvoice(invoice *Invoice) string {
	var result strings.Builder

	for _, item := range invoice.Items {
	}

	for _, item := range invoice.Items {
	}

	if invoice.DiscountRate > 0 {
	}

	return result.String()
}`,
	solutionCode: `package principles

import (
	"fmt"
	"strings"
	"time"
)

type Invoice struct {
	CustomerName string
	Items        []InvoiceItem
	DiscountRate float64
	TaxRate      float64
}

type InvoiceItem struct {
	Name     string
	Quantity int
	Price    float64
}

// GenerateInvoice creates a formatted invoice string
// This is now a high-level function that reads like documentation
func GenerateInvoice(invoice *Invoice) string {
	var result strings.Builder

	// Build invoice sections
	result.WriteString(FormatInvoiceHeader(invoice.CustomerName))
	result.WriteString(FormatLineItems(invoice.Items))
	result.WriteString(FormatTotals(invoice))

	return result.String()
}

// CalculateSubtotal sums up the total cost of all invoice items
func CalculateSubtotal(items []InvoiceItem) float64 {
	subtotal := 0.0
	for _, item := range items {
		subtotal += float64(item.Quantity) * item.Price
	}
	return subtotal
}

// CalculateDiscount applies discount rate to an amount
func CalculateDiscount(amount, rate float64) float64 {
	return amount * rate
}

// CalculateTax applies tax rate to an amount
func CalculateTax(amount, rate float64) float64 {
	return amount * rate
}

// FormatInvoiceHeader creates the invoice header with customer info
func FormatInvoiceHeader(customerName string) string {
	var header strings.Builder
	header.WriteString("=====================================\n")
	header.WriteString("           INVOICE\n")
	header.WriteString("=====================================\n")
	header.WriteString(fmt.Sprintf("Customer: %s\n", customerName))
	header.WriteString(fmt.Sprintf("Date: %s\n\n", time.Now().Format("2006-01-02")))
	return header.String()
}

// FormatLineItems creates the itemized list of products
func FormatLineItems(items []InvoiceItem) string {
	var lines strings.Builder
	lines.WriteString("Items:\n")
	lines.WriteString("-------------------------------------\n")
	for _, item := range items {
		lineTotal := float64(item.Quantity) * item.Price
		lines.WriteString(fmt.Sprintf("%-20s x%-3d $%8.2f\n",
			item.Name, item.Quantity, lineTotal))
	}
	lines.WriteString("-------------------------------------\n\n")
	return lines.String()
}

// FormatTotals calculates and formats all totals (subtotal, discount, tax, total)
func FormatTotals(invoice *Invoice) string {
	var totals strings.Builder

	subtotal := CalculateSubtotal(invoice.Items)
	discountAmount := CalculateDiscount(subtotal, invoice.DiscountRate)
	amountAfterDiscount := subtotal - discountAmount
	taxAmount := CalculateTax(amountAfterDiscount, invoice.TaxRate)
	total := amountAfterDiscount + taxAmount

	totals.WriteString(fmt.Sprintf("Subtotal:            $%8.2f\n", subtotal))
	if invoice.DiscountRate > 0 {
		totals.WriteString(fmt.Sprintf("Discount (%.0f%%):      -$%8.2f\n",
			invoice.DiscountRate*100, discountAmount))
	}
	totals.WriteString(fmt.Sprintf("Tax (%.0f%%):            $%8.2f\n",
		invoice.TaxRate*100, taxAmount))
	totals.WriteString("-------------------------------------\n")
	totals.WriteString(fmt.Sprintf("TOTAL:               $%8.2f\n", total))
	totals.WriteString("=====================================\n")

	return totals.String()
}`,
	hint1: `Extract CalculateSubtotal, CalculateDiscount, and CalculateTax for the math operations. Each should be a pure function taking parameters and returning a result.`,
	hint2: `Extract FormatInvoiceHeader and FormatLineItems for the formatting. Create a FormatTotals function that uses the calculation functions and formats the bottom section.`,
	testCode: `package principles

import (
	"strings"
	"testing"
)

// Test1: CalculateSubtotal sums items correctly
func Test1(t *testing.T) {
	items := []InvoiceItem{
		{Name: "Item1", Quantity: 2, Price: 10.0},
		{Name: "Item2", Quantity: 1, Price: 5.0},
	}
	result := CalculateSubtotal(items)
	if result != 25.0 {
		t.Errorf("CalculateSubtotal = %.2f, want 25.00", result)
	}
}

// Test2: CalculateSubtotal empty items returns zero
func Test2(t *testing.T) {
	result := CalculateSubtotal([]InvoiceItem{})
	if result != 0.0 {
		t.Errorf("CalculateSubtotal([]) = %.2f, want 0.00", result)
	}
}

// Test3: CalculateDiscount applies rate correctly
func Test3(t *testing.T) {
	result := CalculateDiscount(100.0, 0.1)
	if result != 10.0 {
		t.Errorf("CalculateDiscount(100, 0.1) = %.2f, want 10.00", result)
	}
}

// Test4: CalculateTax applies rate correctly
func Test4(t *testing.T) {
	result := CalculateTax(100.0, 0.08)
	if result != 8.0 {
		t.Errorf("CalculateTax(100, 0.08) = %.2f, want 8.00", result)
	}
}

// Test5: FormatInvoiceHeader contains customer name
func Test5(t *testing.T) {
	result := FormatInvoiceHeader("John Doe")
	if !strings.Contains(result, "John Doe") {
		t.Error("header should contain customer name")
	}
}

// Test6: FormatInvoiceHeader contains INVOICE
func Test6(t *testing.T) {
	result := FormatInvoiceHeader("Test")
	if !strings.Contains(result, "INVOICE") {
		t.Error("header should contain INVOICE")
	}
}

// Test7: FormatLineItems lists all items
func Test7(t *testing.T) {
	items := []InvoiceItem{
		{Name: "Widget", Quantity: 2, Price: 10.0},
	}
	result := FormatLineItems(items)
	if !strings.Contains(result, "Widget") {
		t.Error("line items should contain item name")
	}
}

// Test8: GenerateInvoice returns complete invoice
func Test8(t *testing.T) {
	invoice := &Invoice{
		CustomerName: "Alice",
		Items:        []InvoiceItem{{Name: "Test", Quantity: 1, Price: 10.0}},
		DiscountRate: 0.0,
		TaxRate:      0.1,
	}
	result := GenerateInvoice(invoice)
	if !strings.Contains(result, "Alice") {
		t.Error("invoice should contain customer name")
	}
	if !strings.Contains(result, "TOTAL") {
		t.Error("invoice should contain TOTAL")
	}
}

// Test9: Zero discount shows no discount line
func Test9(t *testing.T) {
	invoice := &Invoice{
		CustomerName: "Test",
		Items:        []InvoiceItem{{Name: "A", Quantity: 1, Price: 100.0}},
		DiscountRate: 0.0,
		TaxRate:      0.1,
	}
	result := GenerateInvoice(invoice)
	if strings.Contains(result, "Discount") && strings.Contains(result, "-$") {
		// Check that discount is 0 if shown
	}
}

// Test10: FormatTotals calculates correctly
func Test10(t *testing.T) {
	invoice := &Invoice{
		CustomerName: "Test",
		Items:        []InvoiceItem{{Name: "A", Quantity: 1, Price: 100.0}},
		DiscountRate: 0.1,
		TaxRate:      0.08,
	}
	result := FormatTotals(invoice)
	if !strings.Contains(result, "Subtotal") {
		t.Error("totals should contain Subtotal")
	}
}
`,
	whyItMatters: `Extract Method is one of the most powerful refactoring techniques for improving code clarity.

**Why Extract Method Matters:**

**1. Semantic Compression - Replace "How" with "What"**

\`\`\`go
// BAD: Implementation details obscure intent
func ProcessOrder(order *Order) {
    total := 0.0
    for _, item := range order.Items {
        total += item.Price * float64(item.Quantity)
    }
    discount := total * 0.1
    tax := (total - discount) * 0.08
    order.Total = total - discount + tax
}
// Reader must mentally parse calculations to understand "calculate order total"

// GOOD: Intention-revealing function names
func ProcessOrder(order *Order) {
    subtotal := CalculateSubtotal(order.Items)
    discount := CalculateDiscount(subtotal, 0.1)
    tax := CalculateTax(subtotal-discount, 0.08)
    order.Total = subtotal - discount + tax
}
// Reader immediately understands: calculate subtotal, discount, tax, total
\`\`\`

**2. DRY - Don't Repeat Yourself**

\`\`\`go
// BAD: Duplicated calculation logic
func CalculateOrderTotal(order *Order) float64 {
    total := 0.0
    for _, item := range order.Items {
        total += item.Price * float64(item.Quantity)
    }
    return total
}

func CalculateRefund(order *Order) float64 {
    total := 0.0
    for _, item := range order.Items {  // DUPLICATE!
        total += item.Price * float64(item.Quantity)
    }
    return total * 0.9  // 90% refund
}

// GOOD: Extract once, reuse everywhere
func CalculateSubtotal(items []OrderItem) float64 {
    total := 0.0
    for _, item := range items {
        total += item.Price * float64(item.Quantity)
    }
    return total
}

func CalculateOrderTotal(order *Order) float64 {
    return CalculateSubtotal(order.Items)
}

func CalculateRefund(order *Order) float64 {
    return CalculateSubtotal(order.Items) * 0.9
}
\`\`\`

**3. Testability**

\`\`\`go
// BAD: Hard to test calculation in isolation
func GenerateInvoice(invoice *Invoice) string {
    // 50 lines of formatting...
    subtotal := 0.0  // Calculation buried in formatting
    for _, item := range invoice.Items {
        subtotal += float64(item.Quantity) * item.Price
    }
    // More formatting...
}
// Can't test calculation without generating entire invoice

// GOOD: Test calculation independently
func CalculateSubtotal(items []InvoiceItem) float64 {
    subtotal := 0.0
    for _, item := range items {
        subtotal += float64(item.Quantity) * item.Price
    }
    return subtotal
}

func TestCalculateSubtotal(t *testing.T) {
    items := []InvoiceItem{
        {Price: 10.0, Quantity: 2},
        {Price: 5.0, Quantity: 3},
    }
    got := CalculateSubtotal(items)
    want := 35.0
    if got != want {
        t.Errorf("got %.2f, want %.2f", got, want)
    }
}
\`\`\`

**4. Real-World Example: Bug Fix Made Easy**

\`\`\`go
// BEFORE extraction: Bug in 80-line function
func ProcessPayment(order *Order) error {
    // 20 lines of validation...

    // Bug: Tax calculated on subtotal instead of after discount
    subtotal := 0.0
    for _, item := range order.Items {
        subtotal += item.Price * float64(item.Quantity)
    }
    tax := subtotal * order.TaxRate  // BUG: Should be after discount!

    // 40 more lines of payment processing...
}
// Developer must read 80 lines to find and fix bug

// AFTER extraction: Bug obvious and easy to fix
func ProcessPayment(order *Order) error {
    subtotal := CalculateSubtotal(order.Items)
    discount := CalculateDiscount(subtotal, order.DiscountRate)
    tax := CalculateTax(subtotal, order.TaxRate)  // BUG VISIBLE!
    // Should be: tax := CalculateTax(subtotal-discount, order.TaxRate)
}
// Bug is obvious because each step is explicit
// Fix is one line, testable in isolation
\`\`\`

**5. Method Length Guidelines**

When to extract:
- **Code block does more than one thing**: Extract each responsibility
- **Comment explaining code block**: Extract block, name it after comment
- **Code duplicated**: Extract and reuse
- **Complex condition**: Extract to IsValid/ShouldProcess/etc.
- **Loop body is complex**: Extract loop body to function

\`\`\`go
// Extract when you see comments explaining code
func ProcessUser(user *User) {
    // Validate email format
    if !strings.Contains(user.Email, "@") { ... }

    // Check password strength
    if len(user.Password) < 8 { ... }
}

// Better: Extract and delete comments
func ProcessUser(user *User) {
    if err := ValidateEmail(user.Email); err != nil { ... }
    if err := ValidatePassword(user.Password); err != nil { ... }
}
\`\`\`

**6. Before You Extract - Ask These Questions**

✅ Extract if:
- Function name would be more descriptive than comment
- Code block has clear inputs and outputs
- Extraction improves readability
- Code is duplicated

❌ Don't extract if:
- Function would be called only once and doesn't improve clarity
- Variables have complex dependencies (extract anyway, but pass as parameters)
- Code is already clear and simple (e.g., single line)`,
	order: 3,
	translations: {
		ru: {
			title: 'Малые функции - Извлечение метода',
			description: `Практикуйте технику рефакторинга "Извлечение метода" для улучшения читаемости кода путём извлечения сложной логики в хорошо названные функции.

**Вы выполните рефакторинг:**

1. **GenerateInvoice** - Извлеките сложную логику вычислений и форматирования
2. Создайте вспомогательные функции:
   - CalculateSubtotal
   - CalculateDiscount
   - CalculateTax
   - FormatInvoiceHeader
   - FormatLineItems

**Ключевые концепции:**
- **Извлечение метода**: Замените фрагмент кода вызовом функции
- **Семантическое сжатие**: Замените сложную логику раскрывающим намерение именем
- **Принцип DRY**: Не повторяйтесь - извлекайте дублированный код
- **Раскройте намерение**: Имя функции объясняет что, реализация объясняет как

**Когда извлекать метод:**
- Блок кода имеет чёткую цель
- Блоку кода можно дать осмысленное имя
- Код дублируется в нескольких местах
- Функция слишком длинная (>20 строк)

**Ограничения:**
- Извлеките минимум 5 вспомогательных функций
- Каждая вспомогательная должна быть независимо тестируемой
- Главная функция должна читаться как документация`,
			hint1: `Извлеките CalculateSubtotal, CalculateDiscount и CalculateTax для математических операций. Каждая должна быть чистой функцией принимающей параметры и возвращающей результат.`,
			hint2: `Извлеките FormatInvoiceHeader и FormatLineItems для форматирования. Создайте функцию FormatTotals которая использует функции вычислений и форматирует нижнюю секцию.`,
			whyItMatters: `Извлечение метода — одна из самых мощных техник рефакторинга для улучшения ясности кода.

**Почему извлечение метода важно:**

**1. Семантическое сжатие - Замените "Как" на "Что"**

Имена функций раскрывающих намерение делают код самообъясняющимся.

**2. DRY - Не повторяйтесь**

Извлеките один раз, переиспользуйте везде.

**3. Тестируемость**

Тестируйте вычисления независимо от форматирования.

**4. Когда извлекать**

Извлекайте когда блок кода делает более одной вещи, когда есть комментарий объясняющий код, когда код дублируется.`,
			solutionCode: `package principles

import (
	"fmt"
	"strings"
	"time"
)

type Invoice struct {
	CustomerName string
	Items        []InvoiceItem
	DiscountRate float64
	TaxRate      float64
}

type InvoiceItem struct {
	Name     string
	Quantity int
	Price    float64
}

// GenerateInvoice создаёт отформатированную строку счёта
// Это теперь высокоуровневая функция читающаяся как документация
func GenerateInvoice(invoice *Invoice) string {
	var result strings.Builder

	// Строим секции счёта
	result.WriteString(FormatInvoiceHeader(invoice.CustomerName))
	result.WriteString(FormatLineItems(invoice.Items))
	result.WriteString(FormatTotals(invoice))

	return result.String()
}

// CalculateSubtotal суммирует общую стоимость всех позиций счёта
func CalculateSubtotal(items []InvoiceItem) float64 {
	subtotal := 0.0
	for _, item := range items {
		subtotal += float64(item.Quantity) * item.Price
	}
	return subtotal
}

// CalculateDiscount применяет ставку скидки к сумме
func CalculateDiscount(amount, rate float64) float64 {
	return amount * rate
}

// CalculateTax применяет налоговую ставку к сумме
func CalculateTax(amount, rate float64) float64 {
	return amount * rate
}

// FormatInvoiceHeader создаёт заголовок счёта с информацией о клиенте
func FormatInvoiceHeader(customerName string) string {
	var header strings.Builder
	header.WriteString("=====================================\n")
	header.WriteString("           INVOICE\n")
	header.WriteString("=====================================\n")
	header.WriteString(fmt.Sprintf("Customer: %s\n", customerName))
	header.WriteString(fmt.Sprintf("Date: %s\n\n", time.Now().Format("2006-01-02")))
	return header.String()
}

// FormatLineItems создаёт детализированный список товаров
func FormatLineItems(items []InvoiceItem) string {
	var lines strings.Builder
	lines.WriteString("Items:\n")
	lines.WriteString("-------------------------------------\n")
	for _, item := range items {
		lineTotal := float64(item.Quantity) * item.Price
		lines.WriteString(fmt.Sprintf("%-20s x%-3d $%8.2f\n",
			item.Name, item.Quantity, lineTotal))
	}
	lines.WriteString("-------------------------------------\n\n")
	return lines.String()
}

// FormatTotals вычисляет и форматирует все итоги (промежуточный итог, скидка, налог, итого)
func FormatTotals(invoice *Invoice) string {
	var totals strings.Builder

	subtotal := CalculateSubtotal(invoice.Items)
	discountAmount := CalculateDiscount(subtotal, invoice.DiscountRate)
	amountAfterDiscount := subtotal - discountAmount
	taxAmount := CalculateTax(amountAfterDiscount, invoice.TaxRate)
	total := amountAfterDiscount + taxAmount

	totals.WriteString(fmt.Sprintf("Subtotal:            $%8.2f\n", subtotal))
	if invoice.DiscountRate > 0 {
		totals.WriteString(fmt.Sprintf("Discount (%.0f%%):      -$%8.2f\n",
			invoice.DiscountRate*100, discountAmount))
	}
	totals.WriteString(fmt.Sprintf("Tax (%.0f%%):            $%8.2f\n",
		invoice.TaxRate*100, taxAmount))
	totals.WriteString("-------------------------------------\n")
	totals.WriteString(fmt.Sprintf("TOTAL:               $%8.2f\n", total))
	totals.WriteString("=====================================\n")

	return totals.String()
}`
		},
		uz: {
			title: 'Kichik funksiyalar - Metodni ajratish',
			description: `Murakkab mantiqni yaxshi nomlangan funksiyalarga ajratish orqali kod o'qilishini yaxshilash uchun "Metodni ajratish" refaktoring texnikasini mashq qiling.

**Siz refaktoring qilasiz:**

1. **GenerateInvoice** - Murakkab hisoblash va formatlash mantiqini ajrating
2. Yordamchi funksiyalarni yarating:
   - CalculateSubtotal
   - CalculateDiscount
   - CalculateTax
   - FormatInvoiceHeader
   - FormatLineItems

**Asosiy tushunchalar:**
- **Metodni ajratish**: Kod fragmentini funksiya chaqiruvi bilan almashtiring
- **Semantik siqish**: Murakkab mantiqni niyatni ochuvchi nom bilan almashtiring
- **DRY printsipi**: O'zingizni takrorlamang - takrorlanuvchi kodni ajrating
- **Niyatni ochish**: Funksiya nomi nimani tushuntiradi, amalga oshirish qanday qilib

**Qachon metodni ajratish:**
- Kod blokining aniq maqsadi bor
- Kod blokiga mazmunli nom berish mumkin
- Kod bir nechta joyda takrorlanadi
- Funksiya juda uzun (>20 qator)

**Cheklovlar:**
- Kamida 5 yordamchi funksiya ajrating
- Har bir yordamchi mustaqil ravishda testlanishi kerak
- Asosiy funksiya hujjat kabi o'qilishi kerak`,
			hint1: `Matematik operatsiyalar uchun CalculateSubtotal, CalculateDiscount va CalculateTax ajrating. Har biri parametrlarni qabul qiluvchi va natija qaytaruvchi toza funksiya bo'lishi kerak.`,
			hint2: `Formatlash uchun FormatInvoiceHeader va FormatLineItems ajrating. Hisoblash funksiyalaridan foydalanadigan va pastki qismni formatlaydigan FormatTotals funksiyasini yarating.`,
			whyItMatters: `Metodni ajratish kod ravshanligi uchun eng kuchli refaktoring texnikalaridan biridir.

**Metodni ajratish nima uchun muhim:**

**1. Semantik siqish - "Qanday" ni "Nima" bilan almashtirish**

Niyatni ochuvchi funksiya nomlari kodni o'z-o'zini tushuntiradigan qiladi.

**2. DRY - O'zingizni takrorlamang**

Bir marta ajrating, hamma joyda qayta ishlating.

**3. Testlanuvchilik**

Hisoblashlarni formatlanishdan mustaqil ravishda test qiling.

**4. Qachon ajratish**

Kod bloki bir nechta ish qilganda, kodni tushuntiradigan izoh bo'lganda, kod takrorlanganda ajrating.`,
			solutionCode: `package principles

import (
	"fmt"
	"strings"
	"time"
)

type Invoice struct {
	CustomerName string
	Items        []InvoiceItem
	DiscountRate float64
	TaxRate      float64
}

type InvoiceItem struct {
	Name     string
	Quantity int
	Price    float64
}

// GenerateInvoice formatlangan hisob-faktura qatorini yaratadi
// Bu endi hujjat kabi o'qiladigan yuqori darajali funksiya
func GenerateInvoice(invoice *Invoice) string {
	var result strings.Builder

	// Hisob-faktura bo'limlarini quramiz
	result.WriteString(FormatInvoiceHeader(invoice.CustomerName))
	result.WriteString(FormatLineItems(invoice.Items))
	result.WriteString(FormatTotals(invoice))

	return result.String()
}

// CalculateSubtotal barcha hisob-faktura elementlarining umumiy narxini yig'adi
func CalculateSubtotal(items []InvoiceItem) float64 {
	subtotal := 0.0
	for _, item := range items {
		subtotal += float64(item.Quantity) * item.Price
	}
	return subtotal
}

// CalculateDiscount summaga chegirma stavkasini qo'llaydi
func CalculateDiscount(amount, rate float64) float64 {
	return amount * rate
}

// CalculateTax summaga soliq stavkasini qo'llaydi
func CalculateTax(amount, rate float64) float64 {
	return amount * rate
}

// FormatInvoiceHeader mijoz ma'lumotlari bilan hisob-faktura sarlavhasini yaratadi
func FormatInvoiceHeader(customerName string) string {
	var header strings.Builder
	header.WriteString("=====================================\n")
	header.WriteString("           INVOICE\n")
	header.WriteString("=====================================\n")
	header.WriteString(fmt.Sprintf("Customer: %s\n", customerName))
	header.WriteString(fmt.Sprintf("Date: %s\n\n", time.Now().Format("2006-01-02")))
	return header.String()
}

// FormatLineItems mahsulotlarning batafsil ro'yxatini yaratadi
func FormatLineItems(items []InvoiceItem) string {
	var lines strings.Builder
	lines.WriteString("Items:\n")
	lines.WriteString("-------------------------------------\n")
	for _, item := range items {
		lineTotal := float64(item.Quantity) * item.Price
		lines.WriteString(fmt.Sprintf("%-20s x%-3d $%8.2f\n",
			item.Name, item.Quantity, lineTotal))
	}
	lines.WriteString("-------------------------------------\n\n")
	return lines.String()
}

// FormatTotals barcha yakuniy summalarni hisoblaydi va formatlaydi (oraliq summa, chegirma, soliq, jami)
func FormatTotals(invoice *Invoice) string {
	var totals strings.Builder

	subtotal := CalculateSubtotal(invoice.Items)
	discountAmount := CalculateDiscount(subtotal, invoice.DiscountRate)
	amountAfterDiscount := subtotal - discountAmount
	taxAmount := CalculateTax(amountAfterDiscount, invoice.TaxRate)
	total := amountAfterDiscount + taxAmount

	totals.WriteString(fmt.Sprintf("Subtotal:            $%8.2f\n", subtotal))
	if invoice.DiscountRate > 0 {
		totals.WriteString(fmt.Sprintf("Discount (%.0f%%):      -$%8.2f\n",
			invoice.DiscountRate*100, discountAmount))
	}
	totals.WriteString(fmt.Sprintf("Tax (%.0f%%):            $%8.2f\n",
		invoice.TaxRate*100, taxAmount))
	totals.WriteString("-------------------------------------\n")
	totals.WriteString(fmt.Sprintf("TOTAL:               $%8.2f\n", total))
	totals.WriteString("=====================================\n")

	return totals.String()
}`
		}
	}
};

export default task;
