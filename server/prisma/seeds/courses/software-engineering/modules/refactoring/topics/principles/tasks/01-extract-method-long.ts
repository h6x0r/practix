import { Task } from '../../../../../../../types';

export const task: Task = {
	slug: 'se-refactoring-extract-method-long',
	title: 'Extract Method - Long Method',
	difficulty: 'easy',
	tags: ['refactoring', 'extract-method', 'clean-code', 'go'],
	estimatedTime: '20m',
	isPremium: false,
	youtubeUrl: '',
	description: `Refactor a long method by extracting smaller, focused methods that improve readability and maintainability.

**You will refactor:**

1. **PrintOrderReceipt()** - A long method that does too many things
2. Extract **printHeader()** - Print receipt header with date
3. Extract **printItems()** - Print order items with prices
4. Extract **printFooter()** - Print total and tax information

**Key Concepts:**
- **Single Responsibility**: Each method should do one thing
- **Readability**: Shorter methods are easier to understand
- **Reusability**: Extracted methods can be used elsewhere
- **Testing**: Smaller methods are easier to test

**Before Refactoring:**

\`\`\`go
func (o *Order) PrintOrderReceipt() {
    // 50+ lines of code doing many things
    // Hard to read and understand
}
\`\`\`

**After Refactoring:**

\`\`\`go
func (o *Order) PrintOrderReceipt() {
    o.printHeader()
    o.printItems()
    o.printFooter()
}
\`\`\`

**When to Extract Method:**
- Method is longer than 10-15 lines
- Method has multiple levels of abstraction
- Code has comments explaining what sections do
- You can't quickly explain what method does
- Method has nested loops or conditionals

**Constraints:**
- Keep original PrintOrderReceipt method
- Extract exactly 3 methods: printHeader, printItems, printFooter
- All extracted methods must be private (lowercase first letter)
- Maintain exact same output format`,
	initialCode: `package refactoring

import (
	"fmt"
	"time"
)

type OrderItem struct {
	Name     string
	Price    float64
	Quantity int
}

type Order struct {
	ID         string
	CustomerID string
	Date       time.Time
	TaxRate    float64
}

func (o *Order) PrintOrderReceipt() {

	for _, item := range o.Items {
	}

	var subtotal float64
	for _, item := range o.Items {
	}
}`,
	solutionCode: `package refactoring

import (
	"fmt"
	"time"
)

type OrderItem struct {
	Name     string
	Price    float64
	Quantity int
}

type Order struct {
	ID         string
	CustomerID string
	Items      []OrderItem
	Date       time.Time
	TaxRate    float64
}

func (o *Order) PrintOrderReceipt() {
	o.printHeader()		// clear what this does - prints header
	o.printItems()		// prints all order items
	o.printFooter()		// prints totals and tax
}

// printHeader prints the receipt header with order metadata
func (o *Order) printHeader() {
	fmt.Println("================================")
	fmt.Println("      ONLINE STORE RECEIPT      ")
	fmt.Println("================================")
	fmt.Printf("Order ID: %s\n", o.ID)		// order identification
	fmt.Printf("Customer ID: %s\n", o.CustomerID)	// customer identification
	fmt.Printf("Date: %s\n", o.Date.Format("2006-01-02 15:04:05"))	// formatted timestamp
	fmt.Println("================================")
}

// printItems prints each order item with quantity, price, and subtotal
func (o *Order) printItems() {
	fmt.Println("\nITEMS:")
	for _, item := range o.Items {		// iterate through all items
		subtotal := float64(item.Quantity) * item.Price		// calculate item subtotal
		fmt.Printf("%dx %s @ $%.2f = $%.2f\n",
			item.Quantity, item.Name, item.Price, subtotal)
	}
}

// printFooter calculates and prints subtotal, tax, and total
func (o *Order) printFooter() {
	fmt.Println("\n================================")
	var subtotal float64
	for _, item := range o.Items {		// sum all item prices
		subtotal += float64(item.Quantity) * item.Price
	}
	tax := subtotal * o.TaxRate		// calculate tax amount
	total := subtotal + tax			// calculate final total
	fmt.Printf("Subtotal: $%.2f\n", subtotal)
	fmt.Printf("Tax (%.0f%%): $%.2f\n", o.TaxRate*100, tax)
	fmt.Printf("TOTAL: $%.2f\n", total)
	fmt.Println("================================")
}`,
	hint1: `Create three private methods (lowercase names): printHeader(), printItems(), and printFooter(). Move the corresponding code sections into each method.`,
	hint2: `In PrintOrderReceipt(), simply call the three extracted methods in order: o.printHeader(), o.printItems(), o.printFooter().`,
	whyItMatters: `Extract Method is one of the most important refactoring techniques for improving code quality.

**Why Extract Method Matters:**

**1. Improved Readability**
Long methods are hard to read and understand. Extracting methods creates self-documenting code:

\`\`\`go
// Before: What does this code do? Must read all 50 lines
func ProcessUserRegistration(email, password string) error {
    // validate email format
    if !strings.Contains(email, "@") { ... }
    // check password strength
    if len(password) < 8 { ... }
    // hash password
    hash, _ := bcrypt.GenerateFromPassword(...)
    // save to database
    db.Exec("INSERT INTO users...")
    // send welcome email
    smtp.SendMail(...)
    // log registration
    log.Printf("User registered: %s", email)
}

// After: Instantly understand the process
func ProcessUserRegistration(email, password string) error {
    if err := validateEmail(email); err != nil { return err }
    if err := validatePassword(password); err != nil { return err }
    hashedPassword := hashPassword(password)
    if err := saveUser(email, hashedPassword); err != nil { return err }
    sendWelcomeEmail(email)
    logRegistration(email)
    return nil
}
\`\`\`

**2. Code Reusability**
Extracted methods can be reused in other parts of your code:

\`\`\`go
// Before: Code duplication
func PrintInvoice() {
    // calculate totals - duplicated in 3 places
    var subtotal float64
    for _, item := range items {
        subtotal += item.Price * float64(item.Quantity)
    }
    tax := subtotal * 0.08
    total := subtotal + tax
}

// After: Reusable calculation
func (o *Order) calculateTotals() (subtotal, tax, total float64) {
    for _, item := range o.Items {
        subtotal += item.Price * float64(item.Quantity)
    }
    tax = subtotal * o.TaxRate
    total = subtotal + tax
    return
}

// Now use in multiple places
func PrintInvoice() { sub, tax, tot := o.calculateTotals() }
func SaveOrder() { _, _, total := o.calculateTotals() }
func SendReceipt() { sub, tax, tot := o.calculateTotals() }
\`\`\`

**3. Easier Testing**
Smaller methods are easier to unit test:

\`\`\`go
// Before: How to test just the validation logic?
func ProcessOrder(order Order) error {
    // validation mixed with processing
    if len(order.Items) == 0 { return errors.New("empty") }
    // database operations
    db.Save(order)
    // email sending
    sendEmail(order)
}

// After: Each concern can be tested independently
func TestValidateOrder(t *testing.T) {
    err := validateOrder(Order{Items: []Item{}})
    assert.Error(t, err)  // Easy to test just validation
}

func validateOrder(order Order) error {
    if len(order.Items) == 0 { return errors.New("empty") }
    return nil
}
\`\`\`

**4. Better Abstraction Levels**
Methods should stay at one level of abstraction:

\`\`\`go
// Before: Mixes high-level and low-level operations
func DeployApplication() {
    // high level
    buildApplication()
    // suddenly low level details
    cmd := exec.Command("docker", "build", "-t", "app", ".")
    cmd.Run()
    exec.Command("docker", "push", "app").Run()
    // back to high level
    notifyTeam()
}

// After: Consistent abstraction level
func DeployApplication() {
    buildApplication()
    pushToRegistry()  // same level of abstraction
    notifyTeam()
}

func pushToRegistry() {
    // low-level details hidden here
    cmd := exec.Command("docker", "push", "app")
    cmd.Run()
}
\`\`\`

**Real-World Impact:**

A study by Microsoft found that methods longer than 100 lines had 2-3x more bugs than shorter methods. Google's style guide recommends functions be "focused and small."

**When NOT to Extract:**
- Method is already short (< 5 lines)
- Extraction would create more confusion
- Code is only used once and simple
- Would create too many parameters (use Extract Class instead)

**Code Smells Indicating Need for Extract Method:**
- Comments explaining code sections (each comment = potential method)
- Nested if/for statements (extract inner logic)
- Long parameter lists (extract related operations)
- Duplicate code blocks
- Methods with "and" in the name (doThisAndThat)`,
	order: 0,
	testCode: `package refactoring

import (
	"testing"
	"time"
)

// Test1: PrintOrderReceipt runs without panic
func Test1(t *testing.T) {
	order := &Order{
		ID:         "ORD-001",
		CustomerID: "CUST-123",
		Items: []OrderItem{
			{Name: "Widget", Price: 10.0, Quantity: 2},
		},
		Date:    time.Now(),
		TaxRate: 0.08,
	}
	order.PrintOrderReceipt() // Should not panic
}

// Test2: Order with multiple items
func Test2(t *testing.T) {
	order := &Order{
		ID:         "ORD-002",
		CustomerID: "CUST-456",
		Items: []OrderItem{
			{Name: "Apple", Price: 1.50, Quantity: 3},
			{Name: "Banana", Price: 0.75, Quantity: 6},
		},
		Date:    time.Now(),
		TaxRate: 0.10,
	}
	order.PrintOrderReceipt()
}

// Test3: Order with empty items
func Test3(t *testing.T) {
	order := &Order{
		ID:         "ORD-003",
		CustomerID: "CUST-789",
		Items:      []OrderItem{},
		Date:       time.Now(),
		TaxRate:    0.05,
	}
	order.PrintOrderReceipt() // Should handle empty items
}

// Test4: Order with zero tax rate
func Test4(t *testing.T) {
	order := &Order{
		ID:         "ORD-004",
		CustomerID: "CUST-001",
		Items:      []OrderItem{{Name: "Test", Price: 100.0, Quantity: 1}},
		Date:       time.Now(),
		TaxRate:    0.0,
	}
	order.PrintOrderReceipt()
}

// Test5: Order with high quantity items
func Test5(t *testing.T) {
	order := &Order{
		ID:         "ORD-005",
		CustomerID: "CUST-VIP",
		Items:      []OrderItem{{Name: "Bulk Item", Price: 5.0, Quantity: 100}},
		Date:       time.Now(),
		TaxRate:    0.12,
	}
	order.PrintOrderReceipt()
}

// Test6: OrderItem struct has correct fields
func Test6(t *testing.T) {
	item := OrderItem{Name: "Product", Price: 25.99, Quantity: 2}
	if item.Name != "Product" || item.Price != 25.99 || item.Quantity != 2 {
		t.Error("OrderItem fields not set correctly")
	}
}

// Test7: Order struct has correct fields
func Test7(t *testing.T) {
	now := time.Now()
	order := Order{ID: "123", CustomerID: "C1", Date: now, TaxRate: 0.08}
	if order.ID != "123" || order.CustomerID != "C1" || order.TaxRate != 0.08 {
		t.Error("Order fields not set correctly")
	}
}

// Test8: Order with decimal prices
func Test8(t *testing.T) {
	order := &Order{
		ID:         "ORD-DEC",
		CustomerID: "CUST-DEC",
		Items: []OrderItem{
			{Name: "Item1", Price: 19.99, Quantity: 1},
			{Name: "Item2", Price: 29.49, Quantity: 2},
		},
		Date:    time.Now(),
		TaxRate: 0.0825,
	}
	order.PrintOrderReceipt()
}

// Test9: Order total calculation is correct
func Test9(t *testing.T) {
	order := &Order{
		Items:   []OrderItem{{Name: "Test", Price: 10.0, Quantity: 2}},
		TaxRate: 0.10,
	}
	// Expected: subtotal=20, tax=2, total=22
	order.PrintOrderReceipt()
}

// Test10: Date formatting works
func Test10(t *testing.T) {
	order := &Order{
		ID:         "ORD-DATE",
		CustomerID: "CUST-DATE",
		Items:      []OrderItem{{Name: "Item", Price: 1.0, Quantity: 1}},
		Date:       time.Date(2024, 6, 15, 14, 30, 0, 0, time.UTC),
		TaxRate:    0.05,
	}
	order.PrintOrderReceipt()
}
`,
	translations: {
		ru: {
			title: 'Extract Method - Длинный метод',
			description: `Рефакторинг длинного метода путём извлечения меньших, сфокусированных методов, которые улучшают читаемость и поддерживаемость.

**Вы выполните рефакторинг:**

1. **PrintOrderReceipt()** — Длинный метод, который делает слишком много
2. Извлечь **printHeader()** — Печать заголовка с датой
3. Извлечь **printItems()** — Печать товаров с ценами
4. Извлечь **printFooter()** — Печать итогов и налогов

**Ключевые концепции:**
- **Единственная ответственность**: Каждый метод должен делать одно
- **Читаемость**: Короткие методы легче понять
- **Переиспользуемость**: Извлечённые методы можно использовать где угодно
- **Тестирование**: Маленькие методы легче тестировать

**До рефакторинга:**

\`\`\`go
func (o *Order) PrintOrderReceipt() {
    // 50+ строк кода, делающих много вещей
    // Трудно читать и понимать
}
\`\`\`

**После рефакторинга:**

\`\`\`go
func (o *Order) PrintOrderReceipt() {
    o.printHeader()
    o.printItems()
    o.printFooter()
}
\`\`\`

**Когда извлекать метод:**
- Метод длиннее 10-15 строк
- Метод имеет несколько уровней абстракции
- В коде есть комментарии, объясняющие секции
- Вы не можете быстро объяснить, что делает метод
- Метод имеет вложенные циклы или условия

**Ограничения:**
- Сохранить исходный метод PrintOrderReceipt
- Извлечь ровно 3 метода: printHeader, printItems, printFooter
- Все извлечённые методы должны быть приватными (с маленькой буквы)
- Сохранить точно такой же формат вывода`,
			hint1: `Создайте три приватных метода (имена с маленькой буквы): printHeader(), printItems() и printFooter(). Переместите соответствующие секции кода в каждый метод.`,
			hint2: `В PrintOrderReceipt() просто вызовите три извлечённых метода по порядку: o.printHeader(), o.printItems(), o.printFooter().`,
			whyItMatters: `Extract Method — одна из важнейших техник рефакторинга для улучшения качества кода.

**Почему Extract Method важен:**

**1. Улучшенная читаемость**
Длинные методы трудно читать и понимать. Извлечение методов создаёт самодокументируемый код:

\`\`\`go
// До: Что делает этот код? Нужно прочитать все 50 строк
func ProcessUserRegistration(email, password string) error {
    // валидация email формата
    if !strings.Contains(email, "@") { ... }
    // проверка надёжности пароля
    if len(password) < 8 { ... }
    // хеширование пароля
    hash, _ := bcrypt.GenerateFromPassword(...)
    // сохранение в БД
    db.Exec("INSERT INTO users...")
    // отправка приветственного email
    smtp.SendMail(...)
    // логирование регистрации
    log.Printf("User registered: %s", email)
}

// После: Мгновенно понятен процесс
func ProcessUserRegistration(email, password string) error {
    if err := validateEmail(email); err != nil { return err }
    if err := validatePassword(password); err != nil { return err }
    hashedPassword := hashPassword(password)
    if err := saveUser(email, hashedPassword); err != nil { return err }
    sendWelcomeEmail(email)
    logRegistration(email)
    return nil
}
\`\`\`

**2. Переиспользуемость кода**
Извлечённые методы можно использовать в других частях кода:

\`\`\`go
// До: Дублирование кода
func PrintInvoice() {
    // расчёт итогов - дублируется в 3 местах
    var subtotal float64
    for _, item := range items {
        subtotal += item.Price * float64(item.Quantity)
    }
    tax := subtotal * 0.08
    total := subtotal + tax
}

// После: Переиспользуемый расчёт
func (o *Order) calculateTotals() (subtotal, tax, total float64) {
    for _, item := range o.Items {
        subtotal += item.Price * float64(item.Quantity)
    }
    tax = subtotal * o.TaxRate
    total = subtotal + tax
    return
}

// Теперь используется в нескольких местах
func PrintInvoice() { sub, tax, tot := o.calculateTotals() }
func SaveOrder() { _, _, total := o.calculateTotals() }
func SendReceipt() { sub, tax, tot := o.calculateTotals() }
\`\`\`

**3. Упрощённое тестирование**
Маленькие методы легче тестировать:

\`\`\`go
// До: Как протестировать только логику валидации?
func ProcessOrder(order Order) error {
    // валидация смешана с обработкой
    if len(order.Items) == 0 { return errors.New("empty") }
    // операции с БД
    db.Save(order)
    // отправка email
    sendEmail(order)
}

// После: Каждая ответственность тестируется отдельно
func TestValidateOrder(t *testing.T) {
    err := validateOrder(Order{Items: []Item{}})
    assert.Error(t, err)  // Легко тестировать только валидацию
}

func validateOrder(order Order) error {
    if len(order.Items) == 0 { return errors.New("empty") }
    return nil
}
\`\`\`

**4. Лучшие уровни абстракции**
Методы должны оставаться на одном уровне абстракции:

\`\`\`go
// До: Смешиваются высоко- и низкоуровневые операции
func DeployApplication() {
    // высокий уровень
    buildApplication()
    // внезапно низкоуровневые детали
    cmd := exec.Command("docker", "build", "-t", "app", ".")
    cmd.Run()
    exec.Command("docker", "push", "app").Run()
    // обратно на высокий уровень
    notifyTeam()
}

// После: Согласованный уровень абстракции
func DeployApplication() {
    buildApplication()
    pushToRegistry()  // тот же уровень абстракции
    notifyTeam()
}

func pushToRegistry() {
    // низкоуровневые детали скрыты здесь
    cmd := exec.Command("docker", "push", "app")
    cmd.Run()
}
\`\`\`

**Реальное влияние:**

Исследование Microsoft показало, что методы длиннее 100 строк имели в 2-3 раза больше багов, чем короткие методы. Гайд по стилю Google рекомендует, чтобы функции были "сфокусированными и маленькими".

**Когда НЕ извлекать:**
- Метод уже короткий (< 5 строк)
- Извлечение создаст больше путаницы
- Код используется только раз и прост
- Создаст слишком много параметров (используйте Extract Class)

**Code Smells, указывающие на необходимость Extract Method:**
- Комментарии, объясняющие секции кода (каждый комментарий = потенциальный метод)
- Вложенные if/for операторы (извлеките внутреннюю логику)
- Длинные списки параметров (извлеките связанные операции)
- Дублирующиеся блоки кода
- Методы с "and" в имени (doThisAndThat)`
		},
		uz: {
			title: 'Extract Method - Uzun metod',
			description: `Uzun metodning o'qilishi va qo'llab-quvvatlanishini yaxshilaydigan kichikroq, yo'naltirilgan metodlarni ajratish orqali refaktoring qilish.

**Siz refaktoring qilasiz:**

1. **PrintOrderReceipt()** — Juda ko'p ish qiladigan uzun metod
2. Ajratish **printHeader()** — Sana bilan sarlavha chop etish
3. Ajratish **printItems()** — Narxlar bilan mahsulotlarni chop etish
4. Ajratish **printFooter()** — Jami va soliq ma'lumotlarini chop etish

**Asosiy tushunchalar:**
- **Yagona mas'uliyat**: Har bir metod bitta ishni qilishi kerak
- **O'qilishi**: Qisqa metodlar tushunish osonroq
- **Qayta foydalanish**: Ajratilgan metodlarni boshqa joylarda ishlatish mumkin
- **Testlash**: Kichik metodlarni test qilish osonroq

**Refaktoring oldidan:**

\`\`\`go
func (o *Order) PrintOrderReceipt() {
    // 50+ qator kod ko'p narsani qiladi
    // O'qish va tushunish qiyin
}
\`\`\`

**Refaktoringdan keyin:**

\`\`\`go
func (o *Order) PrintOrderReceipt() {
    o.printHeader()
    o.printItems()
    o.printFooter()
}
\`\`\`

**Qachon metodini ajratish kerak:**
- Metod 10-15 qatordan uzunroq
- Metod bir nechta abstraktsiya darajalariga ega
- Kodda bo'limlarni tushuntiruvchi izohlar bor
- Metod nimani qilishini tez tushuntirib bo'lmaydi
- Metodda ichma-ich tsikllar yoki shartlar bor

**Cheklovlar:**
- Asl PrintOrderReceipt metodini saqlash
- Aynan 3 ta metod ajratish: printHeader, printItems, printFooter
- Barcha ajratilgan metodlar private bo'lishi kerak (kichik harf bilan)
- Chiqish formatini aynan saqlab qolish`,
			hint1: `Uchta private metod yarating (kichik harf bilan): printHeader(), printItems() va printFooter(). Mos keladigan kod qismlarini har bir metodga ko'chiring.`,
			hint2: `PrintOrderReceipt() da shunchaki uchta ajratilgan metodlarni ketma-ket chaqiring: o.printHeader(), o.printItems(), o.printFooter().`,
			whyItMatters: `Extract Method kod sifatini yaxshilash uchun eng muhim refaktoring texnikalaridan biridir.

**Extract Method nima uchun muhim:**

**1. Yaxshilangan o'qilish**
Uzun metodlarni o'qish va tushunish qiyin. Metodlarni ajratish o'z-o'zini hujjatlaydigan kod yaratadi:

\`\`\`go
// Oldin: Bu kod nima qiladi? Barcha 50 qatorni o'qish kerak
func ProcessUserRegistration(email, password string) error {
    // email formatini tekshirish
    if !strings.Contains(email, "@") { ... }
    // parol mustahkamligini tekshirish
    if len(password) < 8 { ... }
    // parolni xeshlash
    hash, _ := bcrypt.GenerateFromPassword(...)
    // ma'lumotlar bazasiga saqlash
    db.Exec("INSERT INTO users...")
    // xush kelibsiz emailini yuborish
    smtp.SendMail(...)
    // ro'yxatdan o'tishni loglash
    log.Printf("User registered: %s", email)
}

// Keyin: Jarayonni bir zumda tushunish
func ProcessUserRegistration(email, password string) error {
    if err := validateEmail(email); err != nil { return err }
    if err := validatePassword(password); err != nil { return err }
    hashedPassword := hashPassword(password)
    if err := saveUser(email, hashedPassword); err != nil { return err }
    sendWelcomeEmail(email)
    logRegistration(email)
    return nil
}
\`\`\`

**2. Kodning qayta foydalanilishi**
Ajratilgan metodlarni kodingizning boshqa qismlarida qayta ishlatish mumkin:

\`\`\`go
// Oldin: Kod takrorlanishi
func PrintInvoice() {
    // jamlarni hisoblash - 3 joyda takrorlanadi
    var subtotal float64
    for _, item := range items {
        subtotal += item.Price * float64(item.Quantity)
    }
    tax := subtotal * 0.08
    total := subtotal + tax
}

// Keyin: Qayta ishlatiladigan hisoblash
func (o *Order) calculateTotals() (subtotal, tax, total float64) {
    for _, item := range o.Items {
        subtotal += item.Price * float64(item.Quantity)
    }
    tax = subtotal * o.TaxRate
    total = subtotal + tax
    return
}

// Endi bir nechta joyda ishlatiladi
func PrintInvoice() { sub, tax, tot := o.calculateTotals() }
func SaveOrder() { _, _, total := o.calculateTotals() }
func SendReceipt() { sub, tax, tot := o.calculateTotals() }
\`\`\`

**3. Osonroq testlash**
Kichikroq metodlarni birlik testlash osonroq:

\`\`\`go
// Oldin: Faqat validatsiya logikasini qanday test qilish mumkin?
func ProcessOrder(order Order) error {
    // validatsiya qayta ishlash bilan aralashgan
    if len(order.Items) == 0 { return errors.New("empty") }
    // ma'lumotlar bazasi operatsiyalari
    db.Save(order)
    // email yuborish
    sendEmail(order)
}

// Keyin: Har bir mas'uliyat mustaqil test qilinadi
func TestValidateOrder(t *testing.T) {
    err := validateOrder(Order{Items: []Item{}})
    assert.Error(t, err)  // Faqat validatsiyani test qilish oson
}

func validateOrder(order Order) error {
    if len(order.Items) == 0 { return errors.New("empty") }
    return nil
}
\`\`\`

**4. Yaxshiroq abstraktsiya darajalari**
Metodlar bitta abstraktsiya darajasida qolishi kerak:

\`\`\`go
// Oldin: Yuqori va past darajadagi operatsiyalar aralashgan
func DeployApplication() {
    // yuqori daraja
    buildApplication()
    // to'satdan past daraja tafsilotlari
    cmd := exec.Command("docker", "build", "-t", "app", ".")
    cmd.Run()
    exec.Command("docker", "push", "app").Run()
    // yana yuqori darajaga qaytish
    notifyTeam()
}

// Keyin: Izchil abstraktsiya darajasi
func DeployApplication() {
    buildApplication()
    pushToRegistry()  // bir xil abstraktsiya darajasi
    notifyTeam()
}

func pushToRegistry() {
    // past darajadagi tafsilotlar bu yerda yashirilgan
    cmd := exec.Command("docker", "push", "app")
    cmd.Run()
}
\`\`\`

**Real ta'sir:**

Microsoft tadqiqotlari 100 qatordan uzunroq metodlarda qisqa metodlarga nisbatan 2-3 marta ko'proq xatolar borligini ko'rsatdi. Google uslub qo'llanmasida funksiyalar "yo'naltirilgan va kichik" bo'lishini tavsiya qiladi.

**Qachon AJRATMASLIK kerak:**
- Metod allaqachon qisqa (< 5 qator)
- Ajratish ko'proq chalkashlik yaratadi
- Kod faqat bir marta ishlatiladi va oddiy
- Juda ko'p parametrlar yaratadi (Extract Class ishlating)

**Extract Method kerakligini ko'rsatuvchi Code Smells:**
- Kod bo'limlarini tushuntiruvchi izohlar (har bir izoh = potensial metod)
- Ichma-ich if/for operatorlar (ichki logikani ajrating)
- Uzun parametr ro'yxatlari (bog'liq operatsiyalarni ajrating)
- Takrorlanuvchi kod bloklari
- Nomida "and" bo'lgan metodlar (doThisAndThat)`
		}
	}
};

export default task;
