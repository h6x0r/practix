import { Task } from '../../../../../../../types';

export const task: Task = {
	slug: 'se-ap-magic-numbers-basic',
	title: 'Magic Numbers Anti-pattern - Basic',
	difficulty: 'easy',
	tags: ['go', 'anti-patterns', 'magic-numbers', 'constants', 'refactoring'],
	estimatedTime: '15m',
	isPremium: false,
	youtubeUrl: '',
	description: `Replace magic numbers with named constants for better code readability and maintainability.

**The Problem:**

Magic numbers are unnamed numerical values scattered throughout code. They make code hard to understand and maintain.

**You will refactor:**

Define meaningful constants and use them in calculations.

**Define these constants:**
- SecondsPerMinute = 60
- MinutesPerHour = 60
- HoursPerDay = 24

**Implement:**
1. **SecondsInDay** - Calculate using constants
2. **MinutesToSeconds** - Convert using constants

**Your Task:**

Replace all magic numbers with named constants.`,
	initialCode: `package antipatterns

func SecondsInDay() int {
}

func MinutesToSeconds(minutes int) int {
}`,
	solutionCode: `package antipatterns

// Named constants make the code self-documenting
// Anyone can understand what these values represent
const (
	SecondsPerMinute = 60	// clear meaning: 60 seconds in a minute
	MinutesPerHour   = 60	// clear meaning: 60 minutes in an hour
	HoursPerDay      = 24	// clear meaning: 24 hours in a day
)

// SecondsInDay uses named constants - self-documenting
// No magic numbers - the calculation is clear
func SecondsInDay() int {
	return HoursPerDay * MinutesPerHour * SecondsPerMinute	// 24 * 60 * 60
}

// MinutesToSeconds uses named constant - clear intent
// If we need to change the conversion (unlikely), we change it once
func MinutesToSeconds(minutes int) int {
	return minutes * SecondsPerMinute	// multiply by 60, but the constant explains WHY
}`,
	hint1: `Define the three constants at package level using const block. SecondsInDay multiplies all three constants. MinutesToSeconds multiplies minutes by SecondsPerMinute.`,
	hint2: `const ( SecondsPerMinute = 60; MinutesPerHour = 60; HoursPerDay = 24 ). SecondsInDay returns HoursPerDay * MinutesPerHour * SecondsPerMinute.`,
	whyItMatters: `Magic numbers make code cryptic and error-prone. Named constants make code self-documenting and maintainable.

**The Problem with Magic Numbers:**

\`\`\`go
// BAD: What does 86400 mean?
func CacheExpiration() time.Duration {
	return time.Duration(86400) * time.Second
}
// Is it seconds? Minutes? Hours? Days? Who knows!

// GOOD: Self-documenting with constants
const SecondsPerDay = 24 * 60 * 60

func CacheExpiration() time.Duration {
	return time.Duration(SecondsPerDay) * time.Second
}
// Immediately clear: cache expires after one day
\`\`\`

**Real-World Example:**

\`\`\`go
// BAD: Magic numbers everywhere
func CalculatePrice(quantity int) float64 {
	basePrice := float64(quantity) * 19.99
	if quantity > 10 {
		basePrice *= 0.9  // What is 0.9?
	}
	tax := basePrice * 0.08  // What is 0.08?
	shipping := 5.99  // Why 5.99?
	if basePrice > 50 {
		shipping = 0  // Free shipping threshold unclear
	}
	return basePrice + tax + shipping
}
// Impossible to understand without context!

// GOOD: Named constants
const (
	ItemPrice              = 19.99
	BulkDiscountThreshold  = 10
	BulkDiscountRate       = 0.10  // 10% discount
	TaxRate                = 0.08  // 8% tax
	ShippingCost           = 5.99
	FreeShippingThreshold  = 50.00
)

func CalculatePrice(quantity int) float64 {
	basePrice := float64(quantity) * ItemPrice

	if quantity > BulkDiscountThreshold {
		discount := basePrice * BulkDiscountRate
		basePrice -= discount
	}

	tax := basePrice * TaxRate

	shipping := ShippingCost
	if basePrice > FreeShippingThreshold {
		shipping = 0
	}

	return basePrice + tax + shipping
}
// Crystal clear! Business rules are self-documenting!
\`\`\`

**Benefits:**
1. **Readability**: Code explains itself
2. **Maintainability**: Change value in one place
3. **Consistency**: Same value used everywhere
4. **Type safety**: Compiler catches mistakes`,
	order: 8,
	testCode: `package antipatterns

import (
	"testing"
)

// Test1: SecondsInDay returns correct value
func Test1(t *testing.T) {
	if SecondsInDay() != 86400 {
		t.Errorf("Expected 86400, got %d", SecondsInDay())
	}
}

// Test2: MinutesToSeconds with 1 minute
func Test2(t *testing.T) {
	if MinutesToSeconds(1) != 60 {
		t.Error("1 minute should be 60 seconds")
	}
}

// Test3: MinutesToSeconds with 60 minutes
func Test3(t *testing.T) {
	if MinutesToSeconds(60) != 3600 {
		t.Error("60 minutes should be 3600 seconds")
	}
}

// Test4: MinutesToSeconds with 0
func Test4(t *testing.T) {
	if MinutesToSeconds(0) != 0 {
		t.Error("0 minutes should be 0 seconds")
	}
}

// Test5: SecondsPerMinute constant
func Test5(t *testing.T) {
	if SecondsPerMinute != 60 {
		t.Error("SecondsPerMinute should be 60")
	}
}

// Test6: MinutesPerHour constant
func Test6(t *testing.T) {
	if MinutesPerHour != 60 {
		t.Error("MinutesPerHour should be 60")
	}
}

// Test7: HoursPerDay constant
func Test7(t *testing.T) {
	if HoursPerDay != 24 {
		t.Error("HoursPerDay should be 24")
	}
}

// Test8: SecondsInDay = HoursPerDay * MinutesPerHour * SecondsPerMinute
func Test8(t *testing.T) {
	expected := HoursPerDay * MinutesPerHour * SecondsPerMinute
	if SecondsInDay() != expected {
		t.Error("SecondsInDay should use constants")
	}
}

// Test9: MinutesToSeconds large number
func Test9(t *testing.T) {
	if MinutesToSeconds(1440) != 86400 {
		t.Error("1440 minutes should be 86400 seconds (one day)")
	}
}

// Test10: MinutesToSeconds with 5
func Test10(t *testing.T) {
	if MinutesToSeconds(5) != 300 {
		t.Error("5 minutes should be 300 seconds")
	}
}
`,
	translations: {
		ru: {
			title: 'Антипаттерн Magic Numbers - Базовый',
			description: `Замените магические числа именованными константами для лучшей читаемости и поддерживаемости кода.

**Проблема:**

Магические числа — это неименованные числовые значения, разбросанные по коду. Они делают код трудным для понимания и поддержки.`,
			hint1: `Определите три константы на уровне пакета используя const блок. SecondsInDay умножает все три константы. MinutesToSeconds умножает minutes на SecondsPerMinute.`,
			hint2: `const ( SecondsPerMinute = 60; MinutesPerHour = 60; HoursPerDay = 24 ). SecondsInDay возвращает HoursPerDay * MinutesPerHour * SecondsPerMinute.`,
			whyItMatters: `Магические числа делают код загадочным и подверженным ошибкам. Именованные константы делают код самодокументируемым и поддерживаемым.`,
			solutionCode: `package antipatterns

const (
	SecondsPerMinute = 60
	MinutesPerHour   = 60
	HoursPerDay      = 24
)

func SecondsInDay() int {
	return HoursPerDay * MinutesPerHour * SecondsPerMinute
}

func MinutesToSeconds(minutes int) int {
	return minutes * SecondsPerMinute
}`
		},
		uz: {
			title: 'Magic Numbers Anti-pattern - Asosiy',
			description: `Yaxshi kod o'qilishi va qo'llab-quvvatlanishi uchun sehrli raqamlarni nomlangan konstantalar bilan almashtiring.

**Muammo:**

Sehrli raqamlar - bu kod bo'ylab tarqalgan nomsiz raqamli qiymatlar. Ular kodni tushunish va qo'llab-quvvatlashni qiyinlashtiradi.`,
			hint1: `Const blokdan foydalanib paket darajasida uchta konstanta aniqlang. SecondsInDay barcha uchta konstantani ko'paytiradi. MinutesToSeconds minutes ni SecondsPerMinute ga ko'paytiradi.`,
			hint2: `const ( SecondsPerMinute = 60; MinutesPerHour = 60; HoursPerDay = 24 ). SecondsInDay HoursPerDay * MinutesPerHour * SecondsPerMinute ni qaytaradi.`,
			whyItMatters: `Sehrli raqamlar kodni sirli va xatolarga moyil qiladi. Nomlangan konstantalar kodni o'z-o'zini hujjatlaydigan va qo'llab-quvvatlanadigan qiladi.`,
			solutionCode: `package antipatterns

const (
	SecondsPerMinute = 60
	MinutesPerHour   = 60
	HoursPerDay      = 24
)

func SecondsInDay() int {
	return HoursPerDay * MinutesPerHour * SecondsPerMinute
}

func MinutesToSeconds(minutes int) int {
	return minutes * SecondsPerMinute
}`
		}
	}
};

export default task;
