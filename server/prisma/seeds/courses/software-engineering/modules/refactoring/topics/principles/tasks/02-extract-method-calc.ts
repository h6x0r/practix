import { Task } from '../../../../../../../types';

export const task: Task = {
	slug: 'se-refactoring-extract-method-calc',
	title: 'Extract Method - Complex Calculation',
	difficulty: 'easy',
	tags: ['refactoring', 'extract-method', 'clean-code', 'go'],
	estimatedTime: '20m',
	isPremium: false,
	youtubeUrl: '',
	description: `Refactor complex calculations by extracting them into well-named methods that reveal intent.

**You will refactor:**

1. **CalculateDiscount()** - Contains complex pricing logic
2. Extract **isEligibleForBulkDiscount()** - Check if quantity qualifies for bulk discount
3. Extract **calculateBulkDiscountRate()** - Calculate discount percentage based on quantity
4. Extract **applyDiscountToPrice()** - Apply calculated discount to total price

**Key Concepts:**
- **Reveal Intent**: Method names explain what calculations do
- **Separation of Concerns**: Each calculation has its own method
- **Testability**: Individual calculations can be tested separately
- **Maintainability**: Easy to modify discount rules

**Example Usage:**

\`\`\`go
discount := CalculateDiscount(10, 50.0)  // 10 items at $50 each
// Returns discounted total with 15% bulk discount
\`\`\`

**When to Extract Calculations:**
- Complex mathematical formulas
- Business logic calculations
- Calculations used in multiple places
- Formula has magic numbers
- Calculation needs explanation

**Constraints:**
- Keep original CalculateDiscount signature
- Extract exactly 3 helper methods
- All extracted methods must be private
- Maintain exact same calculation logic`,
	initialCode: `package refactoring

func CalculateDiscount(quantity int, pricePerItem float64) float64 {

	if quantity >= 10 {
		var discountRate float64
		if quantity >= 100 {
		}
	}

	return total
}`,
	solutionCode: `package refactoring

func CalculateDiscount(quantity int, pricePerItem float64) float64 {
	total := float64(quantity) * pricePerItem

	if isEligibleForBulkDiscount(quantity) {		// clear intent: check eligibility
		discountRate := calculateBulkDiscountRate(quantity)	// get discount rate
		total = applyDiscountToPrice(total, discountRate)	// apply the discount
	}

	return total
}

// isEligibleForBulkDiscount checks if order qualifies for bulk discount
func isEligibleForBulkDiscount(quantity int) bool {
	return quantity >= 10		// minimum 10 items for bulk discount
}

// calculateBulkDiscountRate returns discount percentage based on quantity tiers
func calculateBulkDiscountRate(quantity int) float64 {
	if quantity >= 100 {
		return 0.20		// 20% discount for 100+ items
	} else if quantity >= 50 {
		return 0.15		// 15% discount for 50-99 items
	} else {
		return 0.10		// 10% discount for 10-49 items
	}
}

// applyDiscountToPrice calculates final price after applying discount rate
func applyDiscountToPrice(price, discountRate float64) float64 {
	return price * (1.0 - discountRate)		// subtract discount percentage from price
}`,
	hint1: `Create isEligibleForBulkDiscount(quantity int) bool that returns true when quantity >= 10.`,
	hint2: `Create calculateBulkDiscountRate(quantity int) float64 with the if-else chain for discount tiers, and applyDiscountToPrice(price, discountRate float64) float64 that returns price * (1.0 - discountRate).`,
	whyItMatters: `Extracting complex calculations into named methods is crucial for maintainable business logic.

**Why Extract Calculations Matters:**

**1. Business Logic Clarity**
Named methods make business rules explicit and understandable:

\`\`\`go
// Before: What are these numbers? Why this calculation?
func CalculateShipping(w float64, d int) float64 {
    base := 5.0
    if w > 10 {
        base += (w - 10) * 0.5
    }
    if d > 100 {
        base *= 1.2
    }
    return base
}

// After: Business rules are crystal clear
func CalculateShipping(weight float64, distance int) float64 {
    baseRate := getBaseShippingRate()
    weightSurcharge := calculateWeightSurcharge(weight)
    distanceSurcharge := calculateDistanceSurcharge(distance)
    return baseRate + weightSurcharge + distanceSurcharge
}

func calculateWeightSurcharge(weight float64) float64 {
    const heavyThreshold = 10.0  // kg
    const surchargePerKg = 0.5   // $ per kg over threshold

    if weight > heavyThreshold {
        return (weight - heavyThreshold) * surchargePerKg
    }
    return 0
}
\`\`\`

**2. Eliminate Magic Numbers**
Extract methods give context to mysterious numbers:

\`\`\`go
// Before: What do these numbers mean?
func IsPrime(n int) bool {
    if n < 2 { return false }
    if n == 2 { return true }
    if n%2 == 0 { return false }
    for i := 3; i*i <= n; i += 2 {
        if n%i == 0 { return false }
    }
    return true
}

// After: Each step is clear
func IsPrime(n int) bool {
    if isLessThanSmallestPrime(n) { return false }
    if isSmallestPrime(n) { return true }
    if isEvenNumber(n) { return false }
    return hasNoOddDivisors(n)
}

func isLessThanSmallestPrime(n int) bool {
    const smallestPrime = 2
    return n < smallestPrime
}
\`\`\`

**3. Enable Easy Testing**
Test complex calculations independently:

\`\`\`go
// Before: How to test just the discount calculation?
func ProcessOrder(items []Item) float64 {
    total := 0.0
    for _, item := range items {
        total += item.Price
    }
    // Can't test this logic in isolation
    if total > 100 { total *= 0.9 }
    return total
}

// After: Each calculation can be tested independently
func TestCalculateVolumeDiscount(t *testing.T) {
    tests := []struct {
        total    float64
        expected float64
    }{
        {100.0, 100.0},  // no discount at threshold
        {101.0, 90.9},   // discount above threshold
    }
    for _, tt := range tests {
        got := calculateVolumeDiscount(tt.total)
        assert.Equal(t, tt.expected, got)
    }
}

func calculateVolumeDiscount(total float64) float64 {
    const discountThreshold = 100.0
    const discountRate = 0.10

    if total > discountThreshold {
        return total * (1.0 - discountRate)
    }
    return total
}
\`\`\`

**4. Reusability Across Contexts**
Extracted calculations can be used in different scenarios:

\`\`\`go
// Shared calculation logic
func calculateCompoundInterest(principal, rate float64, years int) float64 {
    const compoundsPerYear = 12
    return principal * math.Pow(1+rate/compoundsPerYear, float64(years*compoundsPerYear))
}

// Used in multiple contexts
func ProjectInvestmentGrowth(initial float64) float64 {
    return calculateCompoundInterest(initial, 0.07, 10)
}

func CalculateLoanPayment(principal float64) float64 {
    return calculateCompoundInterest(principal, 0.05, 5)
}

func EstimateRetirementSavings(monthly float64) float64 {
    return calculateCompoundInterest(monthly, 0.08, 30)
}
\`\`\`

**5. Performance Optimization Opportunities**
Extracted methods can be memoized or optimized:

\`\`\`go
// Cache expensive calculations
var factorialCache = make(map[int]int)

func factorial(n int) int {
    if result, exists := factorialCache[n]; exists {
        return result  // return cached result
    }

    if n <= 1 {
        return 1
    }

    result := n * factorial(n-1)
    factorialCache[n] = result  // cache for future calls
    return result
}

// Now expensive calculation is reused efficiently
func CalculateCombinations(n, k int) int {
    return factorial(n) / (factorial(k) * factorial(n-k))
}
\`\`\`

**Real-World Example - E-commerce Pricing:**

\`\`\`go
// Before: Incomprehensible pricing logic
func GetPrice(item Item, user User) float64 {
    p := item.BasePrice
    if user.IsPremium && user.Orders > 10 { p *= 0.85 }
    if item.OnSale { p *= 0.9 }
    if time.Now().Hour() >= 22 || time.Now().Hour() < 6 { p *= 1.1 }
    return p
}

// After: Each pricing rule is explicit
func GetPrice(item Item, user User) float64 {
    price := item.BasePrice
    price = applyLoyaltyDiscount(price, user)
    price = applySaleDiscount(price, item)
    price = applyTimeSurcharge(price)
    return price
}

func applyLoyaltyDiscount(price float64, user User) float64 {
    const loyaltyDiscountRate = 0.15
    const minimumOrders = 10

    if user.IsPremium && user.Orders > minimumOrders {
        return price * (1.0 - loyaltyDiscountRate)
    }
    return price
}
\`\`\`

**Code Smells for Complex Calculations:**
- Multiple nested conditions
- Hard-coded numbers without explanation
- Comments explaining what calculation does
- Duplicated calculation logic
- Long variable names trying to explain calculation
- Calculations mixed with other logic

**Benefits Summary:**
- Business rules become self-documenting code
- Easy to modify individual calculation rules
- Can add logging/monitoring per calculation
- Enables A/B testing of different formulas
- Makes code reviews focus on business logic, not syntax`,
	order: 1,
	testCode: `package refactoring

import (
	"testing"
)

// Test1: CalculateDiscount with no bulk discount (quantity < 10)
func Test1(t *testing.T) {
	result := CalculateDiscount(5, 10.0)
	expected := 50.0 // no discount
	if result != expected {
		t.Errorf("expected %f, got %f", expected, result)
	}
}

// Test2: CalculateDiscount with 10% discount (10-49 items)
func Test2(t *testing.T) {
	result := CalculateDiscount(10, 10.0)
	expected := 100.0 * 0.90 // 10% off
	if result != expected {
		t.Errorf("expected %f, got %f", expected, result)
	}
}

// Test3: CalculateDiscount with 15% discount (50-99 items)
func Test3(t *testing.T) {
	result := CalculateDiscount(50, 10.0)
	expected := 500.0 * 0.85 // 15% off
	if result != expected {
		t.Errorf("expected %f, got %f", expected, result)
	}
}

// Test4: CalculateDiscount with 20% discount (100+ items)
func Test4(t *testing.T) {
	result := CalculateDiscount(100, 10.0)
	expected := 1000.0 * 0.80 // 20% off
	if result != expected {
		t.Errorf("expected %f, got %f", expected, result)
	}
}

// Test5: isEligibleForBulkDiscount returns true for >= 10
func Test5(t *testing.T) {
	if !isEligibleForBulkDiscount(10) {
		t.Error("expected true for 10 items")
	}
	if !isEligibleForBulkDiscount(100) {
		t.Error("expected true for 100 items")
	}
}

// Test6: isEligibleForBulkDiscount returns false for < 10
func Test6(t *testing.T) {
	if isEligibleForBulkDiscount(9) {
		t.Error("expected false for 9 items")
	}
	if isEligibleForBulkDiscount(0) {
		t.Error("expected false for 0 items")
	}
}

// Test7: calculateBulkDiscountRate returns correct rates
func Test7(t *testing.T) {
	if calculateBulkDiscountRate(10) != 0.10 {
		t.Error("expected 0.10 for 10 items")
	}
	if calculateBulkDiscountRate(49) != 0.10 {
		t.Error("expected 0.10 for 49 items")
	}
	if calculateBulkDiscountRate(50) != 0.15 {
		t.Error("expected 0.15 for 50 items")
	}
	if calculateBulkDiscountRate(100) != 0.20 {
		t.Error("expected 0.20 for 100 items")
	}
}

// Test8: applyDiscountToPrice calculates correctly
func Test8(t *testing.T) {
	result := applyDiscountToPrice(100.0, 0.10)
	if result != 90.0 {
		t.Errorf("expected 90.0, got %f", result)
	}
}

// Test9: CalculateDiscount with boundary at 99 items
func Test9(t *testing.T) {
	result := CalculateDiscount(99, 10.0)
	expected := 990.0 * 0.85 // 15% off
	if result != expected {
		t.Errorf("expected %f, got %f", expected, result)
	}
}

// Test10: CalculateDiscount with large quantity
func Test10(t *testing.T) {
	result := CalculateDiscount(1000, 5.0)
	expected := 5000.0 * 0.80 // 20% off = 4000
	if result != expected {
		t.Errorf("expected %f, got %f", expected, result)
	}
}
`,
	translations: {
		ru: {
			title: 'Extract Method - Сложные вычисления',
			description: `Рефакторинг сложных вычислений путём извлечения их в хорошо названные методы, которые раскрывают намерение.

**Вы выполните рефакторинг:**

1. **CalculateDiscount()** - Содержит сложную ценовую логику
2. Извлечь **isEligibleForBulkDiscount()** - Проверить, подходит ли количество для оптовой скидки
3. Извлечь **calculateBulkDiscountRate()** - Вычислить процент скидки на основе количества
4. Извлечь **applyDiscountToPrice()** - Применить вычисленную скидку к общей цене

**Ключевые концепции:**
- **Раскрытие намерения**: Имена методов объясняют, что делают вычисления
- **Разделение ответственности**: Каждое вычисление имеет свой метод
- **Тестируемость**: Отдельные вычисления могут быть протестированы отдельно
- **Поддерживаемость**: Легко изменять правила скидок

**Пример использования:**

\`\`\`go
discount := CalculateDiscount(10, 50.0)  // 10 товаров по $50 каждый
// Возвращает итог со скидкой с 15% оптовой скидкой
\`\`\`

**Когда извлекать вычисления:**
- Сложные математические формулы
- Вычисления бизнес-логики
- Вычисления, используемые в нескольких местах
- Формула имеет магические числа
- Вычисление требует объяснения

**Ограничения:**
- Сохранить исходную сигнатуру CalculateDiscount
- Извлечь ровно 3 вспомогательных метода
- Все извлечённые методы должны быть приватными
- Сохранить точно такую же логику вычислений`,
			hint1: `Создайте isEligibleForBulkDiscount(quantity int) bool, который возвращает true когда quantity >= 10.`,
			hint2: `Создайте calculateBulkDiscountRate(quantity int) float64 с цепочкой if-else для уровней скидок, и applyDiscountToPrice(price, discountRate float64) float64, который возвращает price * (1.0 - discountRate).`,
			whyItMatters: `Извлечение сложных вычислений в именованные методы критично для поддерживаемой бизнес-логики.

**Почему извлечение вычислений важно:**

**1. Ясность бизнес-логики**
Именованные методы делают бизнес-правила явными и понятными.

**2. Устранение магических чисел**
Извлечённые методы дают контекст загадочным числам.

**3. Упрощённое тестирование**
Тестируйте сложные вычисления независимо.

**4. Переиспользуемость в контекстах**
Извлечённые вычисления можно использовать в разных сценариях.

**5. Возможности оптимизации производительности**
Извлечённые методы можно мемоизировать или оптимизировать.`
		},
		uz: {
			title: 'Extract Method - Murakkab hisoblash',
			description: `Murakkab hisoblashlarni niyatni ochib beradigan yaxshi nomlangan metodlarga ajratish orqali refaktoring qilish.

**Siz refaktoring qilasiz:**

1. **CalculateDiscount()** - Murakkab narxlash logikasini o'z ichiga oladi
2. Ajratish **isEligibleForBulkDiscount()** - Miqdor ulgurji chegirma uchun mos kelishini tekshirish
3. Ajratish **calculateBulkDiscountRate()** - Miqdorga asoslangan chegirma foizini hisoblash
4. Ajratish **applyDiscountToPrice()** - Hisoblangan chegirmani umumiy narxga qo'llash

**Asosiy tushunchalar:**
- **Niyatni ochish**: Metod nomlari hisoblashlar nimani qilishini tushuntiradi
- **Mas'uliyatlarni ajratish**: Har bir hisoblash o'z metodiga ega
- **Testlanish**: Alohida hisoblashlarni alohida test qilish mumkin
- **Qo'llab-quvvatlash**: Chegirma qoidalarini o'zgartirish oson

**Foydalanish misoli:**

\`\`\`go
discount := CalculateDiscount(10, 50.0)  // $50 dan 10 ta mahsulot
// 15% ulgurji chegirma bilan chegirmali jamni qaytaradi
\`\`\`

**Qachon hisoblashlarni ajratish kerak:**
- Murakkab matematik formulalar
- Biznes logikasi hisoblashlari
- Bir nechta joylarda ishlatiladigan hisoblashlar
- Formulada sehrli raqamlar bor
- Hisoblash tushuntirishni talab qiladi

**Cheklovlar:**
- Asl CalculateDiscount imzosini saqlash
- Aynan 3 ta yordamchi metod ajratish
- Barcha ajratilgan metodlar private bo'lishi kerak
- Aynan bir xil hisoblash logikasini saqlash`,
			hint1: `isEligibleForBulkDiscount(quantity int) bool yarating, u quantity >= 10 bo'lganda true qaytaradi.`,
			hint2: `calculateBulkDiscountRate(quantity int) float64 ni chegirma darajalari uchun if-else zanjiri bilan yarating, va applyDiscountToPrice(price, discountRate float64) float64 ni price * (1.0 - discountRate) qaytaradigan qiling.`,
			whyItMatters: `Murakkab hisoblashlarni nomlangan metodlarga ajratish qo'llab-quvvatlanadigan biznes logikasi uchun juda muhimdir.

**Hisoblashlarni ajratish nima uchun muhim:**

**1. Biznes logikasi ravshanligi**
Nomlangan metodlar biznes qoidalarini aniq va tushunarli qiladi.

**2. Sehrli raqamlarni yo'q qilish**
Ajratilgan metodlar sirli raqamlarga kontekst beradi.

**3. Oson testlash**
Murakkab hisoblashlarni mustaqil test qiling.

**4. Kontekstlarda qayta foydalanish**
Ajratilgan hisoblashlarni turli stsenariylarda ishlatish mumkin.

**5. Ishlashni optimallashtirish imkoniyatlari**
Ajratilgan metodlarni memoizatsiya yoki optimallashtirish mumkin.`
		}
	}
};

export default task;
