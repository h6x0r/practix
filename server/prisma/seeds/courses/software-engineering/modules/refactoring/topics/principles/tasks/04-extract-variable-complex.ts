import { Task } from '../../../../../../../types';

export const task: Task = {
	slug: 'se-refactoring-extract-variable-complex',
	title: 'Extract Variable - Complex Expressions',
	difficulty: 'easy',
	tags: ['refactoring', 'extract-variable', 'clean-code', 'go'],
	estimatedTime: '15m',
	isPremium: false,
	youtubeUrl: '',
	description: `Break down complex expressions into named variables that make the logic clear and testable.

**You will refactor:**

1. **IsEligibleForLoan()** - Contains complex boolean expressions
2. Extract **hasGoodCreditScore** - Credit score check result
3. Extract **hasStableIncome** - Employment duration check result
4. Extract **hasReasonableDebtRatio** - Debt-to-income check result

**Key Concepts:**
- **Intermediate Variables**: Break complex expressions into steps
- **Readability**: Each variable explains one condition
- **Debugging**: Easy to inspect intermediate values
- **Testing**: Can verify individual conditions

**Before Refactoring:**

\`\`\`go
return creditScore >= 650 && employmentMonths >= 24 && monthlyDebt/monthlyIncome < 0.43
\`\`\`

**After Refactoring:**

\`\`\`go
hasGoodCredit := creditScore >= 650
hasStableIncome := employmentMonths >= 24
hasReasonableDebt := monthlyDebt/monthlyIncome < 0.43
return hasGoodCredit && hasStableIncome && hasReasonableDebt
\`\`\`

**When to Extract Variables:**
- Complex boolean conditions
- Multiple chained operations
- Nested function calls
- Calculated values used multiple times
- Expressions that need explanation

**Constraints:**
- Keep original IsEligibleForLoan signature
- Extract exactly 3 intermediate variables
- Maintain exact same logic and return value
- Variables must have meaningful names`,
	initialCode: `package refactoring

func IsEligibleForLoan(creditScore int, employmentMonths int, monthlyIncome, monthlyDebt float64) bool {
	return creditScore >= 650 && employmentMonths >= 24 && monthlyDebt/monthlyIncome < 0.43
}`,
	solutionCode: `package refactoring

func IsEligibleForLoan(creditScore int, employmentMonths int, monthlyIncome, monthlyDebt float64) bool {
	// Break down complex expression into understandable parts
	hasGoodCreditScore := creditScore >= 650		// minimum credit score for approval
	hasStableIncome := employmentMonths >= 24		// at least 2 years employment required
	hasReasonableDebtRatio := monthlyDebt/monthlyIncome < 0.43	// debt-to-income below 43%

	// All three conditions must be true for loan eligibility
	return hasGoodCreditScore && hasStableIncome && hasReasonableDebtRatio
}`,
	hint1: `Create three boolean variables: hasGoodCreditScore, hasStableIncome, and hasReasonableDebtRatio. Assign each the result of its respective condition check.`,
	hint2: `Return the AND of all three variables: return hasGoodCreditScore && hasStableIncome && hasReasonableDebtRatio`,
	whyItMatters: `Extracting complex expressions into named variables dramatically improves code comprehension and maintainability.

**Why Extract Complex Expressions Matters:**

**1. Cognitive Load Reduction**
Human brain can only track 3-5 things at once:

\`\`\`go
// Before: Must mentally parse and remember 7+ conditions
if user.Age >= 18 && user.Country == "US" && !user.Banned &&
   user.EmailVerified && time.Since(user.Created) > 24*time.Hour &&
   user.PurchaseCount >= 1 && user.Balance > 0 {
    // What was the first condition again?
}

// After: Each condition is named and clear
isAdult := user.Age >= 18
isUSResident := user.Country == "US"
isActiveAccount := !user.Banned && user.EmailVerified
isEstablishedUser := time.Since(user.Created) > 24*time.Hour
hasTransactionHistory := user.PurchaseCount >= 1
hasFunds := user.Balance > 0

isEligibleForPremium := isAdult && isUSResident &&
                        isActiveAccount && isEstablishedUser &&
                        hasTransactionHistory && hasFunds
if isEligibleForPremium {
    // Crystal clear what we're checking
}
\`\`\`

**2. Debugging Made Easy**
Inspect intermediate values without debugger:

\`\`\`go
// Before: Can't see which condition failed
func ValidateOrder(order Order) error {
    if len(order.Items) > 0 && order.Total >= order.MinOrder &&
       order.ShippingAddress != "" && order.PaymentMethod != nil {
        return nil
    }
    return errors.New("invalid")  // Which check failed?
}

// After: Can log/print each condition
func ValidateOrder(order Order) error {
    hasItems := len(order.Items) > 0
    meetsMinimum := order.Total >= order.MinOrder
    hasShippingAddr := order.ShippingAddress != ""
    hasPaymentMethod := order.PaymentMethod != nil

    // Can add logging
    log.Printf("Order validation: items=%v, minimum=%v, address=%v, payment=%v",
               hasItems, meetsMinimum, hasShippingAddr, hasPaymentMethod)

    isValid := hasItems && meetsMinimum && hasShippingAddr && hasPaymentMethod
    if !isValid {
        return errors.New("invalid")
    }
    return nil
}
\`\`\`

**3. Better Testing**
Test individual conditions in isolation:

\`\`\`go
// Before: Hard to test specific conditions
func CanAccessFeature(user User, feature Feature) bool {
    return user.Subscription == "premium" &&
           feature.MinPlan <= user.PlanLevel &&
           !feature.Deprecated
}

// After: Can test each rule independently
func TestPremiumSubscriptionCheck(t *testing.T) {
    user := User{Subscription: "premium"}
    hasPremium := isPremiumSubscriber(user)
    assert.True(t, hasPremium)
}

func isPremiumSubscriber(user User) bool {
    return user.Subscription == "premium"
}

func meetsFeaturePlan(feature Feature, user User) bool {
    return feature.MinPlan <= user.PlanLevel
}

func isActiveFeature(feature Feature) bool {
    return !feature.Deprecated
}
\`\`\`

**4. Reusable Conditions**
Extract once, use multiple times:

\`\`\`go
// Before: Same expression duplicated
func GetDiscount(user User) float64 {
    if user.Orders > 10 && time.Since(user.Joined) > 365*24*time.Hour {
        return 0.15
    }
    return 0
}

func ShowLoyaltyBadge(user User) bool {
    return user.Orders > 10 && time.Since(user.Joined) > 365*24*time.Hour
}

// After: Define condition once, reuse everywhere
func isLoyalCustomer(user User) bool {
    hasFrequentOrders := user.Orders > 10
    isMemberForYear := time.Since(user.Joined) > 365*24*time.Hour
    return hasFrequentOrders && isMemberForYear
}

func GetDiscount(user User) float64 {
    if isLoyalCustomer(user) {
        return 0.15
    }
    return 0
}

func ShowLoyaltyBadge(user User) bool {
    return isLoyalCustomer(user)
}
\`\`\`

**5. Performance Optimization**
Avoid recalculating expensive expressions:

\`\`\`go
// Before: calculateRisk() called twice if condition is true
func ProcessTransaction(tx Transaction) {
    if calculateRisk(tx) > 0.7 && calculateRisk(tx) < 0.9 {
        // Medium risk: manual review
        manualReview(tx)
    }
}

// After: Calculate once, use multiple times
func ProcessTransaction(tx Transaction) {
    riskScore := calculateRisk(tx)  // expensive operation done once
    isMediumRisk := riskScore > 0.7 && riskScore < 0.9

    if isMediumRisk {
        manualReview(tx)
    }
}
\`\`\`

**Real-World Example - Date Range Validation:**

\`\`\`go
// Before: Unreadable date logic
func IsValidReservation(start, end time.Time, maxDays int) bool {
    return !start.Before(time.Now()) && end.After(start) &&
           end.Sub(start).Hours()/24 <= float64(maxDays) &&
           start.Weekday() != time.Sunday
}

// After: Clear business rules
func IsValidReservation(start, end time.Time, maxDays int) bool {
    now := time.Now()

    // Each business rule is explicit
    isStartInFuture := !start.Before(now)
    isEndAfterStart := end.After(start)

    reservationDays := end.Sub(start).Hours() / 24
    isWithinMaxDuration := reservationDays <= float64(maxDays)

    isValidStartDay := start.Weekday() != time.Sunday

    // Combine all rules
    return isStartInFuture && isEndAfterStart &&
           isWithinMaxDuration && isValidStartDay
}
\`\`\`

**Benefits Summary:**
- Code reads like documentation
- Easy to modify individual conditions
- Simplifies debugging (see which condition fails)
- Enables condition reuse
- Reduces cognitive load
- Makes testing straightforward
- Documents business rules in code

**Code Smells for Complex Expressions:**
- Boolean expressions with 3+ conditions
- Nested ternary operators
- Multiple && or || in one line
- Arithmetic operations in conditionals
- Expressions requiring comments to explain
- Same sub-expression repeated multiple times`,
	order: 3,
	testCode: `package refactoring

import (
	"testing"
)

// Test1: IsEligibleForLoan with all conditions met
func Test1(t *testing.T) {
	// creditScore=700, employment=36mo, income=5000, debt=1000 (ratio=0.2)
	if !IsEligibleForLoan(700, 36, 5000, 1000) {
		t.Error("expected eligible with all conditions met")
	}
}

// Test2: IsEligibleForLoan with low credit score
func Test2(t *testing.T) {
	if IsEligibleForLoan(600, 36, 5000, 1000) {
		t.Error("expected not eligible with credit score < 650")
	}
}

// Test3: IsEligibleForLoan with boundary credit score (650)
func Test3(t *testing.T) {
	if !IsEligibleForLoan(650, 24, 5000, 1000) {
		t.Error("expected eligible with exactly 650 credit score")
	}
}

// Test4: IsEligibleForLoan with short employment
func Test4(t *testing.T) {
	if IsEligibleForLoan(700, 12, 5000, 1000) {
		t.Error("expected not eligible with employment < 24 months")
	}
}

// Test5: IsEligibleForLoan with boundary employment (24 months)
func Test5(t *testing.T) {
	if !IsEligibleForLoan(700, 24, 5000, 1000) {
		t.Error("expected eligible with exactly 24 months employment")
	}
}

// Test6: IsEligibleForLoan with high debt ratio
func Test6(t *testing.T) {
	// debt/income = 2500/5000 = 0.5 > 0.43
	if IsEligibleForLoan(700, 36, 5000, 2500) {
		t.Error("expected not eligible with debt ratio > 0.43")
	}
}

// Test7: IsEligibleForLoan with boundary debt ratio (just under 0.43)
func Test7(t *testing.T) {
	// debt/income = 2100/5000 = 0.42 < 0.43
	if !IsEligibleForLoan(700, 36, 5000, 2100) {
		t.Error("expected eligible with debt ratio 0.42")
	}
}

// Test8: IsEligibleForLoan with debt ratio exactly 0.43
func Test8(t *testing.T) {
	// debt/income = 2150/5000 = 0.43, should fail (need < 0.43)
	if IsEligibleForLoan(700, 36, 5000, 2150) {
		t.Error("expected not eligible with debt ratio exactly 0.43")
	}
}

// Test9: IsEligibleForLoan with all conditions failing
func Test9(t *testing.T) {
	if IsEligibleForLoan(500, 6, 3000, 2000) {
		t.Error("expected not eligible with all conditions failing")
	}
}

// Test10: IsEligibleForLoan with excellent scores
func Test10(t *testing.T) {
	if !IsEligibleForLoan(800, 120, 10000, 1000) {
		t.Error("expected eligible with excellent scores")
	}
}
`,
	translations: {
		ru: {
			title: 'Extract Variable - Сложные выражения',
			description: `Разбейте сложные выражения на именованные переменные, которые делают логику ясной и тестируемой.

**Вы выполните рефакторинг:**

1. **IsEligibleForLoan()** - Содержит сложные булевы выражения
2. Извлечь **hasGoodCreditScore** - Результат проверки кредитного рейтинга
3. Извлечь **hasStableIncome** - Результат проверки продолжительности занятости
4. Извлечь **hasReasonableDebtRatio** - Проверка отношения долга к доходу

**Ключевые концепции:**
- **Промежуточные переменные**: Разбивайте сложные выражения на шаги
- **Читаемость**: Каждая переменная объясняет одно условие
- **Отладка**: Легко проверить промежуточные значения
- **Тестирование**: Можно проверить отдельные условия`,
			hint1: `Создайте три булевы переменные: hasGoodCreditScore, hasStableIncome и hasReasonableDebtRatio. Присвойте каждой результат соответствующей проверки условия.`,
			hint2: `Верните AND всех трёх переменных: return hasGoodCreditScore && hasStableIncome && hasReasonableDebtRatio`,
			whyItMatters: `Извлечение сложных выражений в именованные переменные драматически улучшает понимание и поддерживаемость кода.`
		},
		uz: {
			title: 'Extract Variable - Murakkab ifodalar',
			description: `Murakkab ifodalarni logikani aniq va testlanadigan qiladigan nomlangan o'zgaruvchilarga bo'ling.

**Siz refaktoring qilasiz:**

1. **IsEligibleForLoan()** - Murakkab boolean ifodalarni o'z ichiga oladi
2. Ajratish **hasGoodCreditScore** - Kredit reytingini tekshirish natijasi
3. Ajratish **hasStableIncome** - Ish muddatini tekshirish natijasi
4. Ajratish **hasReasonableDebtRatio** - Qarz-daromad nisbatini tekshirish

**Asosiy tushunchalar:**
- **Oraliq o'zgaruvchilar**: Murakkab ifodalarni qadamlarga bo'ling
- **O'qilishi**: Har bir o'zgaruvchi bitta shartni tushuntiradi
- **Debugging**: Oraliq qiymatlarni tekshirish oson
- **Testlash**: Alohida shartlarni tekshirish mumkin`,
			hint1: `Uchta boolean o'zgaruvchi yarating: hasGoodCreditScore, hasStableIncome va hasReasonableDebtRatio. Har biriga mos shart tekshiruvi natijasini tayinlang.`,
			hint2: `Barcha uchta o'zgaruvchining AND ni qaytaring: return hasGoodCreditScore && hasStableIncome && hasReasonableDebtRatio`,
			whyItMatters: `Murakkab ifodalarni nomlangan o'zgaruvchilarga ajratish kodni tushunish va qo'llab-quvvatlashni dramatik yaxshilaydi.`
		}
	}
};

export default task;
