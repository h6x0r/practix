import { Task } from '../../../../../../../types';

export const task: Task = {
	slug: 'se-clean-code-meaningful-functions',
	title: 'Meaningful Function Names',
	difficulty: 'easy',
	tags: ['go', 'clean-code', 'naming', 'functions'],
	estimatedTime: '25m',
	isPremium: false,
	youtubeUrl: '',
	description: `Refactor functions to have clear, descriptive names that indicate what they do and use appropriate verb conventions.

**You will refactor:**

1. **Rename functions** to follow verb-noun convention and be self-explanatory
2. **Process(user *User) bool** - Check if user can access premium features
3. **Handle(order *Order) error** - Calculate order total with discounts and taxes
4. Make function names describe their complete behavior

**Key Concepts:**
- **Verb-Noun Convention**: Functions do things, names should reflect action
- **Say What You Do**: Name should match what function actually does
- **Avoid Misleading Names**: Don't name functions opposite to what they do
- **Consistent Vocabulary**: Use one word per concept (get vs fetch vs retrieve)

**Example - Before:**

\`\`\`go
func Process(u *User) bool {
    return u.IsPremium && u.ExpiresAt > time.Now().Unix()
}

func Handle(o *Order) error {
    o.Total = calculateSubtotal(o)
    o.Total = applyDiscount(o.Total, o.DiscountCode)
    o.Total = addTax(o.Total, o.TaxRate)
    return saveOrder(o)
}
\`\`\`

**Example - After:**

\`\`\`go
func CanAccessPremiumFeatures(user *User) bool {
    return user.IsPremium && user.ExpiresAt > time.Now().Unix()
}

func CalculateAndSaveOrderTotal(order *Order) error {
    order.Total = calculateSubtotal(order)
    order.Total = applyDiscount(order.Total, order.DiscountCode)
    order.Total = addTax(order.Total, order.TaxRate)
    return saveOrder(order)
}
\`\`\`

**When to use meaningful function names:**
- Every function should reveal its purpose
- Boolean-returning functions should ask a question (Is, Has, Can, Should)
- Action functions should use verbs (Calculate, Process, Validate, Send)

**Constraints:**
- Keep the same implementation logic
- Only rename the functions
- Function names should describe all side effects`,
	initialCode: `package principles

import "time"

type User struct {
	ID        int
	IsPremium bool
	ExpiresAt int64
}

type Order struct {
	ID           int
	Total        float64
	DiscountCode string
	TaxRate      float64
}

type OrderItem struct {
	ProductID int
	Quantity  int
	Price     float64
}

func Process(u *User) bool {
	return u.IsPremium && u.ExpiresAt > currentTime
}

func Handle(o *Order) error {
	for _, item := range o.Items {
	}

	if o.DiscountCode != "" {
	}

	return saveOrderToDB(o)
}

func saveOrderToDB(o *Order) error {
	return nil
}`,
	solutionCode: `package principles

import "time"

type User struct {
	ID        int
	IsPremium bool
	ExpiresAt int64
}

type Order struct {
	ID           int
	Items        []OrderItem
	Total        float64
	DiscountCode string
	TaxRate      float64
}

type OrderItem struct {
	ProductID int
	Quantity  int
	Price     float64
}

// CanAccessPremiumFeatures checks if user has active premium subscription
// Boolean function uses "Can" prefix to form a question
// Name reveals the business logic: checking premium feature access
func CanAccessPremiumFeatures(user *User) bool {
	currentTime := time.Now().Unix()
	return user.IsPremium && user.ExpiresAt > currentTime
}

// CalculateAndSaveOrderTotal computes order total with discounts and taxes, then persists to database
// Name describes both actions: calculation AND saving
// Reveals side effect: this function modifies database state
// Verb-noun convention: Calculate(verb) + OrderTotal(noun) + And + Save(verb)
func CalculateAndSaveOrderTotal(order *Order) error {
	// Calculate subtotal from items
	subtotal := 0.0
	for _, item := range order.Items {
		subtotal += item.Price * float64(item.Quantity)
	}

	// Apply discount
	discountAmount := 0.0
	if order.DiscountCode != "" {
		discountAmount = subtotal * 0.1 // 10% discount
	}

	// Calculate tax
	taxAmount := (subtotal - discountAmount) * order.TaxRate

	// Set final total
	order.Total = subtotal - discountAmount + taxAmount

	// Save to database (simulated)
	return saveOrderToDB(order)
}

// Helper function (simulated database save)
func saveOrderToDB(order *Order) error {
	// Simulate database save
	return nil
}`,
	hint1: `For the first function, use a question format like "CanAccessPremiumFeatures" or "HasActivePremiumSubscription". Boolean functions should read like a yes/no question.`,
	hint2: `For the second function, describe both actions it performs: "CalculateAndSaveOrderTotal". Don't hide the side effect (saving to database) - make it explicit in the name.`,
	testCode: `package principles

import (
	"testing"
	"time"
)

// Test1: Active premium user can access features
func Test1(t *testing.T) {
	user := &User{
		IsPremium: true,
		ExpiresAt: time.Now().Add(24 * time.Hour).Unix(),
	}
	if !CanAccessPremiumFeatures(user) {
		t.Error("active premium user should access features")
	}
}

// Test2: Expired premium user cannot access
func Test2(t *testing.T) {
	user := &User{
		IsPremium: true,
		ExpiresAt: time.Now().Add(-24 * time.Hour).Unix(),
	}
	if CanAccessPremiumFeatures(user) {
		t.Error("expired premium user should not access features")
	}
}

// Test3: Non-premium user cannot access
func Test3(t *testing.T) {
	user := &User{
		IsPremium: false,
		ExpiresAt: time.Now().Add(24 * time.Hour).Unix(),
	}
	if CanAccessPremiumFeatures(user) {
		t.Error("non-premium user should not access features")
	}
}

// Test4: Order total calculated correctly
func Test4(t *testing.T) {
	order := &Order{
		Items: []OrderItem{
			{Price: 10.0, Quantity: 2},
			{Price: 5.0, Quantity: 1},
		},
		TaxRate: 0.0,
	}
	CalculateAndSaveOrderTotal(order)
	if order.Total != 25.0 {
		t.Errorf("Total = %.2f, want 25.00", order.Total)
	}
}

// Test5: Discount applied correctly
func Test5(t *testing.T) {
	order := &Order{
		Items: []OrderItem{
			{Price: 100.0, Quantity: 1},
		},
		DiscountCode: "SAVE10",
		TaxRate:      0.0,
	}
	CalculateAndSaveOrderTotal(order)
	if order.Total != 90.0 {
		t.Errorf("Total = %.2f, want 90.00", order.Total)
	}
}

// Test6: Tax applied correctly
func Test6(t *testing.T) {
	order := &Order{
		Items: []OrderItem{
			{Price: 100.0, Quantity: 1},
		},
		TaxRate: 0.1, // 10%
	}
	CalculateAndSaveOrderTotal(order)
	if order.Total != 110.0 {
		t.Errorf("Total = %.2f, want 110.00", order.Total)
	}
}

// Test7: Empty order
func Test7(t *testing.T) {
	order := &Order{
		Items:   []OrderItem{},
		TaxRate: 0.1,
	}
	err := CalculateAndSaveOrderTotal(order)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
}

// Test8: Function returns nil on success
func Test8(t *testing.T) {
	order := &Order{
		Items: []OrderItem{{Price: 10.0, Quantity: 1}},
	}
	err := CalculateAndSaveOrderTotal(order)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
}

// Test9: Complex order with discount and tax
func Test9(t *testing.T) {
	order := &Order{
		Items: []OrderItem{
			{Price: 100.0, Quantity: 1},
		},
		DiscountCode: "SAVE",
		TaxRate:      0.08,
	}
	CalculateAndSaveOrderTotal(order)
	// Subtotal: 100, Discount: 10, After: 90, Tax: 7.2, Total: 97.2
	expected := 97.2
	if order.Total != expected {
		t.Errorf("Total = %.2f, want %.2f", order.Total, expected)
	}
}

// Test10: User struct fields accessible
func Test10(t *testing.T) {
	user := &User{ID: 123, IsPremium: true, ExpiresAt: 0}
	if user.ID != 123 {
		t.Error("User ID not set correctly")
	}
}
`,
	whyItMatters: `Meaningful function names make code self-documenting and prevent costly misunderstandings.

**Why Function Names Matter:**

**1. Function Names Are Your API Documentation**

\`\`\`go
// BAD: What does "Process" do? You must read implementation
func Process(user *User) bool {
    return user.IsPremium && user.ExpiresAt > time.Now().Unix()
}
// Developer must read code to understand: "Oh, it checks premium status"

// GOOD: Name tells you exactly what it does
func CanAccessPremiumFeatures(user *User) bool {
    return user.IsPremium && user.ExpiresAt > time.Now().Unix()
}
// Developer immediately knows: "checks if user can access premium features"
\`\`\`

**2. Boolean Functions Should Ask Questions**

\`\`\`go
// BAD: Looks like it performs an action
func ValidateUser(user *User) bool { ... }
// Misleading: Does it validate AND modify? Or just check?

// GOOD: Clearly a question, clearly returns bool
func IsUserValid(user *User) bool { ... }
func HasPermission(user *User) bool { ... }
func CanEditDocument(user *User, doc *Document) bool { ... }
\`\`\`

**3. Hidden Side Effects Are Dangerous**

\`\`\`go
// BAD: Name hides that it modifies database
func GetUser(id int) *User {
    user := fetchFromDB(id)
    user.LastLogin = time.Now()
    saveToDBNoError(user)  // SURPRISE! Side effect not in name
    return user
}
// Developer thinks it's read-only, but it writes to DB!

// GOOD: Name reveals all actions
func GetUserAndUpdateLastLogin(id int) *User {
    user := fetchFromDB(id)
    user.LastLogin = time.Now()
    saveToDBNoError(user)
    return user
}
// Or better: separate concerns
func GetUser(id int) *User { ... }              // Pure read
func UpdateUserLastLogin(user *User) { ... }    // Explicit write
\`\`\`

**4. Real Bug from Production**

\`\`\`go
// Production code at company X (anonymized):
func Handle(order *Order) error {
    // Developer A wrote this:
    chargePayment(order)        // charges credit card
    sendConfirmationEmail(order)
    return nil
}

// Developer B calls it:
if !precheck(order) {
    return errors.New("invalid order")
}
err := Handle(order)  // Called twice due to retry logic
// RESULT: Customer charged twice! Cost: $50,000 in refunds

// Fix: Better naming reveals danger
func ChargePaymentAndSendConfirmation(order *Order) error {
    // Name makes it obvious this should only be called once
    chargePayment(order)
    sendConfirmationEmail(order)
    return nil
}
\`\`\`

**5. Verb-Noun Convention for Clarity**

\`\`\`go
// GOOD patterns:
// - Actions: Calculate, Validate, Process, Transform, Send, Fetch
// - Queries (bool): Is, Has, Can, Should, Needs
// - Queries (data): Get, Find, Fetch, Load, Query

// Examples:
func CalculateOrderTotal(order *Order) float64 { ... }
func ValidateEmailFormat(email string) bool { ... }
func TransformUserToDTO(user *User) UserDTO { ... }
func SendWelcomeEmail(user *User) error { ... }

func IsEmailValid(email string) bool { ... }
func HasPermission(user *User) bool { ... }
func CanDeleteAccount(user *User) bool { ... }

func GetUserByID(id int) (*User, error) { ... }
func FindActiveUsers() []*User { ... }
func FetchLatestOrders(limit int) []*Order { ... }
\`\`\`

**6. Consistent Vocabulary Across Codebase**

\`\`\`go
// BAD: Same concept, different words
func GetUser(id int) *User { ... }
func FetchProduct(id int) *Product { ... }
func RetrieveOrder(id int) *Order { ... }
func LoadInvoice(id int) *Invoice { ... }
// Developer must remember: which word for which entity?

// GOOD: Same concept, same word
func GetUser(id int) *User { ... }
func GetProduct(id int) *Product { ... }
func GetOrder(id int) *Order { ... }
func GetInvoice(id int) *Invoice { ... }
// Pattern is predictable: "Get" always means fetch by ID
\`\`\`

**7. Go-Specific Naming Conventions**

\`\`\`go
// Getters: Don't use "Get" prefix (Go convention)
func (u *User) Name() string { ... }        // GOOD
func (u *User) GetName() string { ... }     // BAD (too Java-like)

// Setters: Use "Set" prefix
func (u *User) SetName(name string) { ... }  // GOOD

// Boolean methods: Use "Is/Has/Can"
func (u *User) IsAdmin() bool { ... }        // GOOD
func (u *User) Admin() bool { ... }          // BAD (unclear return type)

// Interfaces: Often named with "er" suffix
type Reader interface { Read(...) }           // GOOD
type Writer interface { Write(...) }          // GOOD
\`\`\`

**8. Length vs Clarity Trade-off**

\`\`\`go
// Too short: Cryptic
func Proc(u *User) bool { ... }

// Too long: Unwieldy
func ProcessUserPremiumAccessEligibilityCheckAndReturnBooleanResult(u *User) bool { ... }

// Just right: Clear and concise
func CanAccessPremiumFeatures(user *User) bool { ... }

// Rule of thumb:
// - Exported functions: Be more descriptive (2-4 words)
// - Package-private: Can be shorter (1-2 words) if context is clear
\`\`\`

**Common Naming Anti-Patterns:**

\`\`\`go
// ❌ Generic names that don't reveal purpose
func DoStuff(data interface{}) { ... }
func HandleRequest(req Request) { ... }
func ProcessData(input []byte) { ... }

// ✅ Specific names that reveal intent
func ValidateEmailFormat(email string) bool { ... }
func AuthenticateUserCredentials(req LoginRequest) error { ... }
func ParseJSONToUserStruct(input []byte) (*User, error) { ... }

// ❌ Misleading names
func GetUser(id int) *User {
    // Actually creates user if not exists!
}

// ✅ Honest names
func GetOrCreateUser(id int) *User { ... }

// ❌ Abbreviations that obscure meaning
func ProcUsrAuth(usr *Usr) bool { ... }

// ✅ Full words that communicate clearly
func ProcessUserAuthentication(user *User) bool { ... }
\`\`\``,
	order: 1,
	translations: {
		ru: {
			title: 'Осмысленные имена функций',
			description: `Выполните рефакторинг функций для использования чётких, описательных имён, которые указывают что они делают и следуют соответствующим глагольным конвенциям.

**Вы выполните рефакторинг:**

1. **Переименуйте функции** для следования конвенции глагол-существительное и самообъясняемости
2. **Process(user *User) bool** - Проверить может ли пользователь получить доступ к премиум функциям
3. **Handle(order *Order) error** - Рассчитать общую стоимость заказа со скидками и налогами
4. Сделайте имена функций описывающими их полное поведение

**Ключевые концепции:**
- **Конвенция глагол-существительное**: Функции делают что-то, имена должны отражать действие
- **Говорите что делаете**: Имя должно соответствовать тому, что функция действительно делает
- **Избегайте вводящих в заблуждение имён**: Не называйте функции противоположно тому, что они делают
- **Последовательный словарь**: Используйте одно слово для концепции (get vs fetch vs retrieve)

**Когда использовать осмысленные имена функций:**
- Каждая функция должна раскрывать свою цель
- Функции возвращающие bool должны задавать вопрос (Is, Has, Can, Should)
- Функции-действия должны использовать глаголы (Calculate, Process, Validate, Send)

**Ограничения:**
- Сохраните ту же логику реализации
- Только переименуйте функции
- Имена функций должны описывать все побочные эффекты`,
			hint1: `Для первой функции используйте вопросительный формат вроде "CanAccessPremiumFeatures" или "HasActivePremiumSubscription". Булевы функции должны читаться как вопрос да/нет.`,
			hint2: `Для второй функции опишите оба действия которые она выполняет: "CalculateAndSaveOrderTotal". Не скрывайте побочный эффект (сохранение в БД) - сделайте его явным в имени.`,
			whyItMatters: `Осмысленные имена функций делают код самодокументируемым и предотвращают дорогостоящие недоразумения.

**Почему имена функций важны:**

**1. Имена функций — это документация вашего API**

Разработчик сразу понимает назначение без чтения кода.

**2. Булевы функции должны задавать вопросы**

IsUserValid, HasPermission, CanEditDocument — ясно что возвращают bool.

**3. Скрытые побочные эффекты опасны**

Функция GetUser не должна изменять базу данных. Если изменяет — назовите GetUserAndUpdateLastLogin.

**4. Последовательный словарь в кодовой базе**

Используйте одно слово для концепции: GetUser, GetProduct, GetOrder — паттерн предсказуем.`,
			solutionCode: `package principles

import "time"

type User struct {
	ID        int
	IsPremium bool
	ExpiresAt int64
}

type Order struct {
	ID           int
	Items        []OrderItem
	Total        float64
	DiscountCode string
	TaxRate      float64
}

type OrderItem struct {
	ProductID int
	Quantity  int
	Price     float64
}

// CanAccessPremiumFeatures проверяет имеет ли пользователь активную премиум подписку
// Булева функция использует префикс "Can" для формирования вопроса
// Имя раскрывает бизнес-логику: проверка доступа к премиум функциям
func CanAccessPremiumFeatures(user *User) bool {
	currentTime := time.Now().Unix()
	return user.IsPremium && user.ExpiresAt > currentTime
}

// CalculateAndSaveOrderTotal вычисляет итоговую стоимость заказа со скидками и налогами, затем сохраняет в БД
// Имя описывает оба действия: вычисление И сохранение
// Раскрывает побочный эффект: эта функция изменяет состояние БД
// Конвенция глагол-существительное: Calculate(глагол) + OrderTotal(существительное) + And + Save(глагол)
func CalculateAndSaveOrderTotal(order *Order) error {
	// Вычисляем промежуточную сумму из позиций
	subtotal := 0.0
	for _, item := range order.Items {
		subtotal += item.Price * float64(item.Quantity)
	}

	// Применяем скидку
	discountAmount := 0.0
	if order.DiscountCode != "" {
		discountAmount = subtotal * 0.1 // скидка 10%
	}

	// Вычисляем налог
	taxAmount := (subtotal - discountAmount) * order.TaxRate

	// Устанавливаем финальную сумму
	order.Total = subtotal - discountAmount + taxAmount

	// Сохраняем в БД (симуляция)
	return saveOrderToDB(order)
}

// Вспомогательная функция (симуляция сохранения в БД)
func saveOrderToDB(order *Order) error {
	// Симулируем сохранение в БД
	return nil
}`
		},
		uz: {
			title: 'Funksiyalarning mazmunli nomlari',
			description: `Funksiyalarni nima qilishini ko'rsatadigan aniq, ta'riflovchi nomlar bilan refaktoring qiling va tegishli fe'l konventsiyalariga amal qiling.

**Siz refaktoring qilasiz:**

1. **Funksiyalarni qayta nomlash** fe'l-ot konventsiyasiga rioya qilish va o'z-o'zini tushuntirish uchun
2. **Process(user *User) bool** - Foydalanuvchi premium funksiyalarga kirish huquqi borligini tekshirish
3. **Handle(order *Order) error** - Buyurtma umumiy narxini chegirmalar va soliqlar bilan hisoblash
4. Funksiya nomlari ularning to'liq harakatini ta'riflashi kerak

**Asosiy tushunchalar:**
- **Fe'l-ot konventsiyasi**: Funksiyalar biror narsa qiladi, nomlar harakatni aks ettirishi kerak
- **Nima qilganingizni ayting**: Nom funksiya aslida nima qilishiga mos kelishi kerak
- **Chalg'ituvchi nomlardan qoching**: Funksiyalarni ular qilgan ishga qarama-qarshi nomlamang
- **Izchil lug'at**: Kontseptsiya uchun bitta so'z ishlating (get vs fetch vs retrieve)

**Qachon funksiyalarning mazmunli nomlarini ishlatish:**
- Har bir funksiya o'z maqsadini ochib berishi kerak
- Boolean qaytaradigan funksiyalar savol berishi kerak (Is, Has, Can, Should)
- Harakat funksiyalari fe'llar ishlatishi kerak (Calculate, Process, Validate, Send)

**Cheklovlar:**
- Bir xil amalga oshirish mantiqini saqlang
- Faqat funksiyalarni qayta nomlang
- Funksiya nomlari barcha yon ta'sirlarni ta'riflashi kerak`,
			hint1: `Birinchi funksiya uchun "CanAccessPremiumFeatures" yoki "HasActivePremiumSubscription" kabi savol formatini ishlating. Boolean funksiyalar ha/yo'q savoli kabi o'qilishi kerak.`,
			hint2: `Ikkinchi funksiya uchun u bajaradigan ikkala harakatni ta'riflang: "CalculateAndSaveOrderTotal". Yon ta'sirni (ma'lumotlar bazasiga saqlash) yashirmang - nomda aniq ko'rsating.`,
			whyItMatters: `Funksiyalarning mazmunli nomlari kodni o'z-o'zini hujjatlaydigan qiladi va qimmat noto'g'ri tushunishlarning oldini oladi.

**Funksiya nomlari nima uchun muhim:**

**1. Funksiya nomlari sizning API hujjatingiz**

Dasturchi kodni o'qimasdan darhol maqsadni tushunadi.

**2. Boolean funksiyalar savol berishi kerak**

IsUserValid, HasPermission, CanEditDocument — boolean qaytarishi aniq.

**3. Yashirin yon ta'sirlar xavfli**

GetUser funksiyasi ma'lumotlar bazasini o'zgartirmasligi kerak. Agar o'zgartirsa — GetUserAndUpdateLastLogin deb nomlang.

**4. Kod bazasida izchil lug'at**

Kontseptsiya uchun bitta so'z ishlating: GetUser, GetProduct, GetOrder — pattern bashorat qilinadigan.`,
			solutionCode: `package principles

import "time"

type User struct {
	ID        int
	IsPremium bool
	ExpiresAt int64
}

type Order struct {
	ID           int
	Items        []OrderItem
	Total        float64
	DiscountCode string
	TaxRate      float64
}

type OrderItem struct {
	ProductID int
	Quantity  int
	Price     float64
}

// CanAccessPremiumFeatures foydalanuvchining faol premium obunasi borligini tekshiradi
// Boolean funksiya savol tuzish uchun "Can" prefiksini ishlatadi
// Nom biznes mantiqni ochadi: premium funksiyalarga kirishni tekshirish
func CanAccessPremiumFeatures(user *User) bool {
	currentTime := time.Now().Unix()
	return user.IsPremium && user.ExpiresAt > currentTime
}

// CalculateAndSaveOrderTotal buyurtma umumiy narxini chegirmalar va soliqlar bilan hisoblaydi, keyin ma'lumotlar bazasiga saqlaydi
// Nom ikkala harakatni ta'riflaydi: hisoblash VA saqlash
// Yon ta'sirni ochadi: bu funksiya ma'lumotlar bazasi holatini o'zgartiradi
// Fe'l-ot konventsiyasi: Calculate(fe'l) + OrderTotal(ot) + And + Save(fe'l)
func CalculateAndSaveOrderTotal(order *Order) error {
	// Elementlardan oraliq summani hisoblaymiz
	subtotal := 0.0
	for _, item := range order.Items {
		subtotal += item.Price * float64(item.Quantity)
	}

	// Chegirmani qo'llaymiz
	discountAmount := 0.0
	if order.DiscountCode != "" {
		discountAmount = subtotal * 0.1 // 10% chegirma
	}

	// Soliqni hisoblaymiz
	taxAmount := (subtotal - discountAmount) * order.TaxRate

	// Yakuniy summani o'rnatamiz
	order.Total = subtotal - discountAmount + taxAmount

	// Ma'lumotlar bazasiga saqlaymiz (simulyatsiya)
	return saveOrderToDB(order)
}

// Yordamchi funksiya (ma'lumotlar bazasiga saqlash simulyatsiyasi)
func saveOrderToDB(order *Order) error {
	// Ma'lumotlar bazasiga saqlashni simulyatsiya qilamiz
	return nil
}`
		}
	}
};

export default task;
