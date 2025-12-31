import { Task } from '../../../../../../../types';

export const task: Task = {
	slug: 'se-ap-spaghetti-code-basic',
	title: 'Spaghetti Code Anti-pattern - Basic',
	difficulty: 'easy',
	tags: ['go', 'anti-patterns', 'spaghetti-code', 'refactoring'],
	estimatedTime: '20m',
	isPremium: false,
	youtubeUrl: '',
	description: `Refactor spaghetti code with nested conditionals into clean, readable functions.

**The Problem:**

Spaghetti code is unstructured, tangled code with complex control flow that's hard to follow. It often has deeply nested if statements and unclear logic flow.

**You will refactor:**

A messy discount calculator with:
- Nested if statements
- Mixed concerns
- Hard-to-follow logic

**Into clean functions:**
1. **IsEligibleForDiscount** - Check eligibility
2. **CalculateDiscount** - Calculate discount percentage
3. **ApplyDiscount** - Apply discount to price

**Example Usage:**

\`\`\`go
price := 100.0
isPremium := true
itemCount := 5

if IsEligibleForDiscount(isPremium, itemCount) {
    discount := CalculateDiscount(isPremium, itemCount)
    finalPrice := ApplyDiscount(price, discount)
    fmt.Printf("Final: $%.2f\\n", finalPrice)
}
\`\`\`

**Your Task:**

Implement clean, focused functions that replace the spaghetti code.`,
	initialCode: `package antipatterns

func IsEligibleForDiscount(isPremium bool, itemCount int) bool {
}

func CalculateDiscount(isPremium bool, itemCount int) float64 {
}

func ApplyDiscount(price float64, discount float64) float64 {
}`,
	solutionCode: `package antipatterns

// IsEligibleForDiscount checks if user qualifies for any discount
// Clear single purpose: eligibility check
func IsEligibleForDiscount(isPremium bool, itemCount int) bool {
	return isPremium || itemCount >= 3	// simple OR condition
}

// CalculateDiscount determines discount percentage based on user status and cart
// Clear single purpose: discount calculation
func CalculateDiscount(isPremium bool, itemCount int) float64 {
	// Early return pattern - handle special cases first
	if isPremium && itemCount >= 5 {
		return 0.20	// best discount: premium + bulk
	}

	if isPremium {
		return 0.15	// premium discount
	}

	if itemCount >= 5 {
		return 0.10	// bulk discount tier 2
	}

	if itemCount >= 3 {
		return 0.05	// bulk discount tier 1
	}

	return 0.0	// no discount
}

// ApplyDiscount applies discount to price
// Clear single purpose: price calculation
func ApplyDiscount(price float64, discount float64) float64 {
	return price * (1 - discount)	// simple formula
}`,
	hint1: `IsEligibleForDiscount: return isPremium || itemCount >= 3. CalculateDiscount: use if-else chain checking conditions from highest discount to lowest.`,
	hint2: `ApplyDiscount: return price * (1 - discount). For CalculateDiscount, check "isPremium && itemCount >= 5" first (0.20), then "isPremium" (0.15), then itemCount conditions.`,
	whyItMatters: `Spaghetti code is one of the most common causes of bugs, maintenance nightmares, and developer frustration.

**Why Spaghetti Code is Dangerous:**

**1. Impossible to Understand**

\`\`\`go
// BAD: Spaghetti code - what does this do?!
func ProcessOrder(user User, items []Item, coupon string) float64 {
	total := 0.0
	for _, item := range items {
		if item.Price > 0 {
			if user.IsPremium {
				if len(items) > 5 {
					if coupon != "" {
						if item.Category == "electronics" {
							total += item.Price * 0.7
						} else {
							if item.InStock {
								total += item.Price * 0.75
							} else {
								total += item.Price * 0.8
							}
						}
					} else {
						if item.Price > 100 {
							total += item.Price * 0.85
						} else {
							total += item.Price * 0.9
						}
					}
				} else {
					if user.YearsActive > 2 {
						total += item.Price * 0.9
					} else {
						total += item.Price * 0.95
					}
				}
			} else {
				if len(items) > 3 {
					total += item.Price * 0.95
				} else {
					total += item.Price
				}
			}
		}
	}
	return total
}
// 7 levels of nesting! Impossible to understand or modify!
\`\`\`

**2. Bug Paradise**

\`\`\`go
// Spaghetti code hides bugs in the nesting
if user.IsActive {
	if user.HasPaid {
		if order.Status == "pending" {
			if inventory.Check(order.Items) {
				ProcessPayment(order)
				// BUG: What if payment fails? No error handling!
				// BUG: Inventory never reserved!
				// BUG: No transaction rollback!
				UpdateInventory(order)
				SendEmail(order)
			}
		}
	}
}
// Missing else clauses - what happens when conditions fail?
// No one knows!
\`\`\`

**3. Impossible to Test**

\`\`\`go
// How do you test all the paths in this spaghetti?
func ComplexLogic(a, b, c, d, e bool) string {
	if a {
		if b {
			if c {
				if d {
					if e {
						return "case1"
					} else {
						return "case2"
					}
				} else {
					return "case3"
				}
			} else {
				if d {
					return "case4"
				}
				return "case5"
			}
		} else {
			return "case6"
		}
	} else {
		if b && c {
			return "case7"
		}
		return "case8"
	}
}
// 8+ paths to test! And this is simplified!
\`\`\`

**Real-World Example - Before Refactoring:**

\`\`\`go
// 200 lines of spaghetti in production
func HandleUserRegistration(data map[string]string) error {
	if data["email"] != "" {
		if validateEmail(data["email"]) {
			if !userExists(data["email"]) {
				if data["password"] != "" {
					if len(data["password"]) >= 8 {
						if data["name"] != "" {
							if data["age"] != "" {
								age, err := strconv.Atoi(data["age"])
								if err == nil {
									if age >= 18 {
										if data["country"] != "" {
											if isValidCountry(data["country"]) {
												// Finally create user...
												user := createUser(data)
												sendWelcomeEmail(user)
												return nil
											} else {
												return errors.New("invalid country")
											}
										} else {
											return errors.New("country required")
										}
									} else {
										return errors.New("must be 18+")
									}
								} else {
									return errors.New("invalid age")
								}
							} else {
								return errors.New("age required")
							}
						} else {
							return errors.New("name required")
						}
					} else {
						return errors.New("password too short")
					}
				} else {
					return errors.New("password required")
				}
			} else {
				return errors.New("user exists")
			}
		} else {
			return errors.New("invalid email")
		}
	} else {
		return errors.New("email required")
	}
}
// 15+ levels of nesting! Arrow anti-pattern!
\`\`\`

**After Refactoring - Clean Code:**

\`\`\`go
// Extract validation to focused functions
func validateRegistrationData(data map[string]string) error {
	// Early returns - fail fast pattern
	if data["email"] == "" {
		return errors.New("email required")
	}

	if !validateEmail(data["email"]) {
		return errors.New("invalid email")
	}

	if userExists(data["email"]) {
		return errors.New("user exists")
	}

	if err := validatePassword(data["password"]); err != nil {
		return err
	}

	if err := validateUserInfo(data); err != nil {
		return err
	}

	return nil
}

func validatePassword(password string) error {
	if password == "" {
		return errors.New("password required")
	}

	if len(password) < 8 {
		return errors.New("password too short")
	}

	return nil
}

func validateUserInfo(data map[string]string) error {
	if data["name"] == "" {
		return errors.New("name required")
	}

	age, err := strconv.Atoi(data["age"])
	if err != nil {
		return errors.New("invalid age")
	}

	if age < 18 {
		return errors.New("must be 18+")
	}

	if !isValidCountry(data["country"]) {
		return errors.New("invalid country")
	}

	return nil
}

// Main handler is now clean and readable
func HandleUserRegistration(data map[string]string) error {
	if err := validateRegistrationData(data); err != nil {
		return err
	}

	user := createUser(data)
	sendWelcomeEmail(user)

	return nil
}
// Each function has ONE clear job!
// No nesting! Easy to read, test, and modify!
\`\`\`

**Benefits of Refactoring:**

1. **Readability**: Each function has a clear, single purpose
2. **Testability**: Test each validation independently
3. **Maintainability**: Easy to add new validations
4. **Debuggability**: Stack traces point to specific validation
5. **Reusability**: Use validatePassword anywhere

**Pattern: Early Returns (Guard Clauses)**

\`\`\`go
// Instead of nesting:
func Process(user User) error {
	if user.IsValid() {
		if user.IsActive() {
			if user.HasPermission() {
				// Do work here
				return doWork(user)
			}
		}
	}
	return errors.New("invalid")
}

// Use early returns:
func Process(user User) error {
	if !user.IsValid() {
		return errors.New("invalid user")
	}

	if !user.IsActive() {
		return errors.New("inactive user")
	}

	if !user.HasPermission() {
		return errors.New("no permission")
	}

	return doWork(user)  // happy path at the end
}
// Flat structure! Easy to read top-to-bottom!
\`\`\`

**Warning Signs of Spaghetti Code:**

- More than 3 levels of nesting
- Functions longer than 50 lines
- Unclear what the function does
- Multiple responsibilities in one function
- Hard to name the function
- Can't explain the logic in simple terms
- Tests require complex setup`,
	order: 2,
	testCode: `package antipatterns

import (
	"testing"
)

// Test1: IsEligibleForDiscount premium user
func Test1(t *testing.T) {
	if !IsEligibleForDiscount(true, 1) {
		t.Error("Premium user should be eligible")
	}
}

// Test2: IsEligibleForDiscount non-premium with 3+ items
func Test2(t *testing.T) {
	if !IsEligibleForDiscount(false, 3) {
		t.Error("3+ items should be eligible")
	}
}

// Test3: IsEligibleForDiscount non-premium with < 3 items
func Test3(t *testing.T) {
	if IsEligibleForDiscount(false, 2) {
		t.Error("Non-premium with < 3 items should not be eligible")
	}
}

// Test4: CalculateDiscount premium with 5+ items (20%)
func Test4(t *testing.T) {
	discount := CalculateDiscount(true, 5)
	if discount != 0.20 {
		t.Errorf("Expected 0.20, got %f", discount)
	}
}

// Test5: CalculateDiscount premium only (15%)
func Test5(t *testing.T) {
	discount := CalculateDiscount(true, 2)
	if discount != 0.15 {
		t.Errorf("Expected 0.15, got %f", discount)
	}
}

// Test6: CalculateDiscount 5+ items non-premium (10%)
func Test6(t *testing.T) {
	discount := CalculateDiscount(false, 5)
	if discount != 0.10 {
		t.Errorf("Expected 0.10, got %f", discount)
	}
}

// Test7: CalculateDiscount 3+ items non-premium (5%)
func Test7(t *testing.T) {
	discount := CalculateDiscount(false, 3)
	if discount != 0.05 {
		t.Errorf("Expected 0.05, got %f", discount)
	}
}

// Test8: CalculateDiscount no discount
func Test8(t *testing.T) {
	discount := CalculateDiscount(false, 2)
	if discount != 0.0 {
		t.Errorf("Expected 0.0, got %f", discount)
	}
}

// Test9: ApplyDiscount 20% off 100
func Test9(t *testing.T) {
	result := ApplyDiscount(100.0, 0.20)
	if result != 80.0 {
		t.Errorf("Expected 80.0, got %f", result)
	}
}

// Test10: ApplyDiscount 0% off
func Test10(t *testing.T) {
	result := ApplyDiscount(100.0, 0.0)
	if result != 100.0 {
		t.Errorf("Expected 100.0, got %f", result)
	}
}
`,
	translations: {
		ru: {
			title: 'Антипаттерн Spaghetti Code - Базовый',
			description: `Рефакторьте спагетти-код с вложенными условиями в чистые, читаемые функции.

**Проблема:**

Спагетти-код — это неструктурированный, запутанный код со сложным потоком управления, который трудно отследить. Он часто имеет глубоко вложенные операторы if и неясную логику.

**Вы выполните рефакторинг:**

Запутанного калькулятора скидок с:
- Вложенными операторами if
- Смешанными обязанностями
- Трудно отслеживаемой логикой

**В чистые функции:**
1. **IsEligibleForDiscount** - Проверка права на скидку
2. **CalculateDiscount** - Расчёт процента скидки
3. **ApplyDiscount** - Применение скидки к цене

**Пример использования:**

\`\`\`go
price := 100.0
isPremium := true
itemCount := 5

if IsEligibleForDiscount(isPremium, itemCount) {
    discount := CalculateDiscount(isPremium, itemCount)
    finalPrice := ApplyDiscount(price, discount)
    fmt.Printf("Final: $%.2f\\n", finalPrice)
}
\`\`\`

**Ваша задача:**

Реализуйте чистые, сфокусированные функции, заменяющие спагетти-код.`,
			hint1: `IsEligibleForDiscount: верните isPremium || itemCount >= 3. CalculateDiscount: используйте цепочку if-else, проверяя условия от наибольшей скидки к наименьшей.`,
			hint2: `ApplyDiscount: верните price * (1 - discount). Для CalculateDiscount сначала проверьте "isPremium && itemCount >= 5" (0.20), затем "isPremium" (0.15), затем условия itemCount.`,
			whyItMatters: `Спагетти-код — одна из самых распространённых причин багов, кошмаров обслуживания и разочарования разработчиков.

**Почему спагетти-код опасен:**

**1. Невозможно понять**
**2. Рай для багов**
**3. Невозможно тестировать**

**Преимущества рефакторинга:**

1. **Читаемость**: Каждая функция имеет ясную, единственную цель
2. **Тестируемость**: Тестируйте каждую валидацию независимо
3. **Поддерживаемость**: Легко добавлять новые валидации
4. **Отладка**: Трассировка стека указывает на конкретную валидацию
5. **Переиспользуемость**: Используйте validatePassword где угодно`,
			solutionCode: `package antipatterns

// IsEligibleForDiscount проверяет, имеет ли пользователь право на скидку
// Ясная единственная цель: проверка права
func IsEligibleForDiscount(isPremium bool, itemCount int) bool {
	return isPremium || itemCount >= 3	// простое условие ИЛИ
}

// CalculateDiscount определяет процент скидки на основе статуса пользователя и корзины
// Ясная единственная цель: расчёт скидки
func CalculateDiscount(isPremium bool, itemCount int) float64 {
	// Паттерн ранних возвратов - сначала обрабатываем особые случаи
	if isPremium && itemCount >= 5 {
		return 0.20	// лучшая скидка: premium + оптом
	}

	if isPremium {
		return 0.15	// скидка premium
	}

	if itemCount >= 5 {
		return 0.10	// оптовая скидка уровень 2
	}

	if itemCount >= 3 {
		return 0.05	// оптовая скидка уровень 1
	}

	return 0.0	// без скидки
}

// ApplyDiscount применяет скидку к цене
// Ясная единственная цель: расчёт цены
func ApplyDiscount(price float64, discount float64) float64 {
	return price * (1 - discount)	// простая формула
}`
		},
		uz: {
			title: 'Spaghetti Code Anti-pattern - Asosiy',
			description: `Ichma-ich shartlarga ega spaghetti code ni toza, o'qiladigan funksiyalarga refaktoring qiling.

**Muammo:**

Spaghetti code — bu murakkab boshqaruv oqimi bilan kuzatish qiyin bo'lgan strukturasiz, chalkash kod. U ko'pincha chuqur ichma-ich if operatorlari va noaniq mantiq oqimiga ega.

**Siz refaktoring qilasiz:**

Quyidagilarga ega chalkash chegirma kalkulyatori:
- Ichma-ich if operatorlari
- Aralash mas'uliyatlar
- Kuzatish qiyin bo'lgan mantiq

**Toza funksiyalarga:**
1. **IsEligibleForDiscount** - Munosiblikni tekshirish
2. **CalculateDiscount** - Chegirma foizini hisoblash
3. **ApplyDiscount** - Narxga chegirma qo'llash

**Foydalanish misoli:**

\`\`\`go
price := 100.0
isPremium := true
itemCount := 5

if IsEligibleForDiscount(isPremium, itemCount) {
    discount := CalculateDiscount(isPremium, itemCount)
    finalPrice := ApplyDiscount(price, discount)
    fmt.Printf("Final: $%.2f\\n", finalPrice)
}
\`\`\`

**Sizning vazifangiz:**

Spaghetti code ni almashtiruvchi toza, fokusli funksiyalarni amalga oshiring.`,
			hint1: `IsEligibleForDiscount: isPremium || itemCount >= 3 ni qaytaring. CalculateDiscount: eng yuqori chegirmadan eng pastiga qadar shartlarni tekshiruvchi if-else zanjiridan foydalaning.`,
			hint2: `ApplyDiscount: price * (1 - discount) ni qaytaring. CalculateDiscount uchun avval "isPremium && itemCount >= 5" (0.20), keyin "isPremium" (0.15), keyin itemCount shartlarini tekshiring.`,
			whyItMatters: `Spaghetti code xatolar, texnik xizmat ko'rsatish dahshatlari va dasturchining umidsizligining eng keng tarqalgan sabablaridan biridir.

**Spaghetti code nima uchun xavfli:**

**1. Tushunish mumkin emas**
**2. Xatolar uchun jannat**
**3. Test qilish mumkin emas**

**Refaktoringning afzalliklari:**

1. **O'qilishi**: Har bir funksiya aniq, yagona maqsadga ega
2. **Test qilish mumkinligi**: Har bir validatsiyani mustaqil test qiling
3. **Qo'llab-quvvatlash**: Yangi validatsiyalar qo'shish oson
4. **Debuglash**: Stack trace aniq validatsiyaga ishora qiladi
5. **Qayta foydalanish**: validatePassword ni istalgan joyda ishlating`,
			solutionCode: `package antipatterns

// IsEligibleForDiscount foydalanuvchining chegirmaga munosibligini tekshiradi
// Aniq yagona maqsad: munosiblik tekshiruvi
func IsEligibleForDiscount(isPremium bool, itemCount int) bool {
	return isPremium || itemCount >= 3	// oddiy YOKI sharti
}

// CalculateDiscount foydalanuvchi holati va savatga asosan chegirma foizini aniqlaydi
// Aniq yagona maqsad: chegirma hisoblash
func CalculateDiscount(isPremium bool, itemCount int) float64 {
	// Erta qaytish patterni - avval maxsus holatlarni qayta ishlaymiz
	if isPremium && itemCount >= 5 {
		return 0.20	// eng yaxshi chegirma: premium + ommaviy
	}

	if isPremium {
		return 0.15	// premium chegirma
	}

	if itemCount >= 5 {
		return 0.10	// ommaviy chegirma 2-daraja
	}

	if itemCount >= 3 {
		return 0.05	// ommaviy chegirma 1-daraja
	}

	return 0.0	// chegirma yo'q
}

// ApplyDiscount narxga chegirma qo'llaydi
// Aniq yagona maqsad: narx hisoblash
func ApplyDiscount(price float64, discount float64) float64 {
	return price * (1 - discount)	// oddiy formula
}`
		}
	}
};

export default task;
