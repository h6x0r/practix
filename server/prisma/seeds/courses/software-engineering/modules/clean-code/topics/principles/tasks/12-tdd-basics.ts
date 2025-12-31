import { Task } from '../../../../../../../types';

export const task: Task = {
	slug: 'se-clean-code-tdd',
	title: 'Test-Driven Development Basics',
	difficulty: 'medium',
	tags: ['go', 'clean-code', 'tdd', 'testing'],
	estimatedTime: '35m',
	isPremium: false,
	youtubeUrl: '',
	description: `Practice Test-Driven Development (TDD): write tests first, then implement code to pass the tests. Follow the Red-Green-Refactor cycle.

**You will implement:**

1. Write failing tests first (Red)
2. Write minimal code to pass tests (Green)
3. Refactor while keeping tests green
4. Implement a calculator with TDD approach

**Key Concepts:**
- **Red-Green-Refactor**: TDD cycle
- **Test First**: Write tests before implementation
- **Minimal Code**: Write just enough to pass tests
- **Refactor**: Improve code while tests pass

**Constraints:**
- Write at least 4 test cases
- Follow TDD cycle for each function
- Tests must be comprehensive`,
	initialCode: `package principles

type Calculator struct{}

func (c *Calculator) Add(a, b int) int {
}

func (c *Calculator) Subtract(a, b int) int {
}

func (c *Calculator) Multiply(a, b int) int {
}

func (c *Calculator) Divide(a, b int) (int, error) {
}`,
	solutionCode: `package principles

import (
	"errors"
	"testing"
)

// Calculator performs basic arithmetic operations
type Calculator struct{}

// Add returns sum of two integers
func (c *Calculator) Add(a, b int) int {
	return a + b
}

// Subtract returns difference of two integers
func (c *Calculator) Subtract(a, b int) int {
	return a - b
}

// Multiply returns product of two integers
func (c *Calculator) Multiply(a, b int) int {
	return a * b
}

// Divide returns quotient and error if dividing by zero
func (c *Calculator) Divide(a, b int) (int, error) {
	if b == 0 {
		return 0, errors.New("division by zero")
	}
	return a / b, nil
}

// TDD Tests - Written BEFORE implementation

func TestCalculatorAdd(t *testing.T) {
	calc := &Calculator{}

	tests := []struct {
		name     string
		a, b     int
		expected int
	}{
		{"positive numbers", 5, 3, 8},
		{"negative numbers", -5, -3, -8},
		{"mixed signs", 5, -3, 2},
		{"with zero", 5, 0, 5},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := calc.Add(tt.a, tt.b)
			if result != tt.expected {
				t.Errorf("Add(%d, %d) = %d; want %d", tt.a, tt.b, result, tt.expected)
			}
		})
	}
}

func TestCalculatorSubtract(t *testing.T) {
	calc := &Calculator{}

	tests := []struct {
		name     string
		a, b     int
		expected int
	}{
		{"positive numbers", 5, 3, 2},
		{"negative numbers", -5, -3, -2},
		{"mixed signs", 5, -3, 8},
		{"result zero", 5, 5, 0},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := calc.Subtract(tt.a, tt.b)
			if result != tt.expected {
				t.Errorf("Subtract(%d, %d) = %d; want %d", tt.a, tt.b, result, tt.expected)
			}
		})
	}
}

func TestCalculatorMultiply(t *testing.T) {
	calc := &Calculator{}

	tests := []struct {
		name     string
		a, b     int
		expected int
	}{
		{"positive numbers", 5, 3, 15},
		{"negative numbers", -5, -3, 15},
		{"mixed signs", 5, -3, -15},
		{"multiply by zero", 5, 0, 0},
		{"multiply by one", 5, 1, 5},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := calc.Multiply(tt.a, tt.b)
			if result != tt.expected {
				t.Errorf("Multiply(%d, %d) = %d; want %d", tt.a, tt.b, result, tt.expected)
			}
		})
	}
}

func TestCalculatorDivide(t *testing.T) {
	calc := &Calculator{}

	t.Run("normal division", func(t *testing.T) {
		result, err := calc.Divide(10, 2)
		if err != nil {
			t.Errorf("Unexpected error: %v", err)
		}
		if result != 5 {
			t.Errorf("Divide(10, 2) = %d; want 5", result)
		}
	})

	t.Run("division by zero", func(t *testing.T) {
		_, err := calc.Divide(10, 0)
		if err == nil {
			t.Error("Expected error for division by zero, got nil")
		}
	})

	t.Run("negative division", func(t *testing.T) {
		result, err := calc.Divide(-10, 2)
		if err != nil {
			t.Errorf("Unexpected error: %v", err)
		}
		if result != -5 {
			t.Errorf("Divide(-10, 2) = %d; want -5", result)
		}
	})
}`,
	hint1: `For each method: write test cases first with expected outcomes, run test (it fails - RED), implement the method, run test (it passes - GREEN).`,
	hint2: `Use table-driven tests for multiple cases. Divide should handle division by zero error. Test positive, negative, and edge cases for each operation.`,
	whyItMatters: `TDD ensures code correctness, prevents regressions, and guides design.

**Why TDD Matters:**

**The TDD Cycle:**
\`\`\`
1. RED: Write failing test
2. GREEN: Write minimal code to pass
3. REFACTOR: Improve code while tests pass
4. REPEAT
\`\`\`

**Benefits:**
- Ensures testable design
- Catches bugs early
- Documents expected behavior
- Enables confident refactoring
- Reduces debugging time

**Example TDD Flow:**
\`\`\`go
// 1. RED: Write test first
func TestAdd(t *testing.T) {
    calc := &Calculator{}
    result := calc.Add(2, 3)
    if result != 5 {
        t.Error("Add failed")
    }
}
// Run: FAIL (Add not implemented)

// 2. GREEN: Minimal implementation
func (c *Calculator) Add(a, b int) int {
    return a + b
}
// Run: PASS

// 3. REFACTOR: Improve if needed
// Tests still pass!
\`\`\`

**TDD Best Practices:**
- Write smallest test possible
- One test per behavior
- Test edge cases
- Keep tests fast
- Test behavior, not implementation`,
	order: 11,
	testCode: `package principles

import (
	"testing"
)

// Test1: Add positive numbers
func Test1(t *testing.T) {
	calc := &Calculator{}
	result := calc.Add(5, 3)
	if result != 8 {
		t.Errorf("Add(5, 3) = %d; want 8", result)
	}
}

// Test2: Add negative numbers
func Test2(t *testing.T) {
	calc := &Calculator{}
	result := calc.Add(-5, -3)
	if result != -8 {
		t.Errorf("Add(-5, -3) = %d; want -8", result)
	}
}

// Test3: Subtract positive numbers
func Test3(t *testing.T) {
	calc := &Calculator{}
	result := calc.Subtract(10, 4)
	if result != 6 {
		t.Errorf("Subtract(10, 4) = %d; want 6", result)
	}
}

// Test4: Multiply positive numbers
func Test4(t *testing.T) {
	calc := &Calculator{}
	result := calc.Multiply(6, 7)
	if result != 42 {
		t.Errorf("Multiply(6, 7) = %d; want 42", result)
	}
}

// Test5: Multiply by zero
func Test5(t *testing.T) {
	calc := &Calculator{}
	result := calc.Multiply(100, 0)
	if result != 0 {
		t.Errorf("Multiply(100, 0) = %d; want 0", result)
	}
}

// Test6: Divide normal case
func Test6(t *testing.T) {
	calc := &Calculator{}
	result, err := calc.Divide(20, 4)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if result != 5 {
		t.Errorf("Divide(20, 4) = %d; want 5", result)
	}
}

// Test7: Divide by zero returns error
func Test7(t *testing.T) {
	calc := &Calculator{}
	_, err := calc.Divide(10, 0)
	if err == nil {
		t.Error("expected error for division by zero")
	}
}

// Test8: Add with zero
func Test8(t *testing.T) {
	calc := &Calculator{}
	result := calc.Add(42, 0)
	if result != 42 {
		t.Errorf("Add(42, 0) = %d; want 42", result)
	}
}

// Test9: Subtract resulting in negative
func Test9(t *testing.T) {
	calc := &Calculator{}
	result := calc.Subtract(3, 10)
	if result != -7 {
		t.Errorf("Subtract(3, 10) = %d; want -7", result)
	}
}

// Test10: Divide negative by positive
func Test10(t *testing.T) {
	calc := &Calculator{}
	result, err := calc.Divide(-20, 5)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if result != -4 {
		t.Errorf("Divide(-20, 5) = %d; want -4", result)
	}
}
`,
	translations: {
		ru: {
			title: 'Основы разработки через тестирование',
			description: `Практикуйте разработку через тестирование (TDD): сначала пишите тесты, затем реализуйте код чтобы пройти тесты. Следуйте циклу Красный-Зелёный-Рефакторинг.`,
			hint1: `Для каждого метода: сначала напишите тест-кейсы с ожидаемыми результатами, запустите тест (провал - КРАСНЫЙ), реализуйте метод, запустите тест (успех - ЗЕЛЁНЫЙ).`,
			hint2: `Используйте табличные тесты для множественных случаев. Divide должен обрабатывать ошибку деления на ноль. Тестируйте положительные, отрицательные и граничные случаи.`,
			whyItMatters: `TDD обеспечивает корректность кода, предотвращает регрессии и направляет проектирование.`,
			solutionCode: `package principles

import (
	"errors"
	"testing"
)

// Calculator выполняет базовые арифметические операции
type Calculator struct{}

// Add возвращает сумму двух целых чисел
func (c *Calculator) Add(a, b int) int {
	return a + b
}

// Subtract возвращает разность двух целых чисел
func (c *Calculator) Subtract(a, b int) int {
	return a - b
}

// Multiply возвращает произведение двух целых чисел
func (c *Calculator) Multiply(a, b int) int {
	return a * b
}

// Divide возвращает частное и ошибку при делении на ноль
func (c *Calculator) Divide(a, b int) (int, error) {
	if b == 0 {
		return 0, errors.New("division by zero")
	}
	return a / b, nil
}

// TDD Тесты - Написаны ДО реализации

func TestCalculatorAdd(t *testing.T) {
	calc := &Calculator{}

	tests := []struct {
		name     string
		a, b     int
		expected int
	}{
		{"положительные числа", 5, 3, 8},
		{"отрицательные числа", -5, -3, -8},
		{"смешанные знаки", 5, -3, 2},
		{"с нулём", 5, 0, 5},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := calc.Add(tt.a, tt.b)
			if result != tt.expected {
				t.Errorf("Add(%d, %d) = %d; want %d", tt.a, tt.b, result, tt.expected)
			}
		})
	}
}

func TestCalculatorDivide(t *testing.T) {
	calc := &Calculator{}

	t.Run("нормальное деление", func(t *testing.T) {
		result, err := calc.Divide(10, 2)
		if err != nil {
			t.Errorf("Неожиданная ошибка: %v", err)
		}
		if result != 5 {
			t.Errorf("Divide(10, 2) = %d; want 5", result)
		}
	})

	t.Run("деление на ноль", func(t *testing.T) {
		_, err := calc.Divide(10, 0)
		if err == nil {
			t.Error("Ожидалась ошибка для деления на ноль")
		}
	})
}`
		},
		uz: {
			title: 'Test orqali dasturlash asoslari',
			description: `Test orqali dasturlashni (TDD) mashq qiling: avval testlar yozing, keyin testlarni o'tish uchun kodni amalga oshiring. Qizil-Yashil-Refaktoring siklini bajaring.`,
			hint1: `Har bir metod uchun: avval kutilgan natijalar bilan test holatlarini yozing, testni ishga tushiring (muvaffaqiyatsiz - QIZIL), metodni amalga oshiring, testni ishga tushiring (muvaffaqiyatli - YASHIL).`,
			hint2: `Ko'p holatlar uchun jadval asosidagi testlardan foydalaning. Divide nolga bo'lish xatosini boshqarishi kerak. Har bir operatsiya uchun ijobiy, salbiy va chekka holatlarni test qiling.`,
			whyItMatters: `TDD kod to'g'riligini ta'minlaydi, regressiyalarning oldini oladi va dizaynni yo'naltiradi.`,
			solutionCode: `package principles

import (
	"errors"
	"testing"
)

// Calculator asosiy arifmetik operatsiyalarni bajaradi
type Calculator struct{}

// Add ikki butun sonning yig'indisini qaytaradi
func (c *Calculator) Add(a, b int) int {
	return a + b
}

// Subtract ikki butun sonning ayirmasini qaytaradi
func (c *Calculator) Subtract(a, b int) int {
	return a - b
}

// Multiply ikki butun sonning ko'paytmasini qaytaradi
func (c *Calculator) Multiply(a, b int) int {
	return a * b
}

// Divide bo'linma va nolga bo'lganda xatoni qaytaradi
func (c *Calculator) Divide(a, b int) (int, error) {
	if b == 0 {
		return 0, errors.New("division by zero")
	}
	return a / b, nil
}

// TDD testlari - Amalga oshirishdan OLDIN yozilgan

func TestCalculatorAdd(t *testing.T) {
	calc := &Calculator{}

	tests := []struct {
		name     string
		a, b     int
		expected int
	}{
		{"musbat sonlar", 5, 3, 8},
		{"manfiy sonlar", -5, -3, -8},
		{"aralash belgilar", 5, -3, 2},
		{"nol bilan", 5, 0, 5},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := calc.Add(tt.a, tt.b)
			if result != tt.expected {
				t.Errorf("Add(%d, %d) = %d; want %d", tt.a, tt.b, result, tt.expected)
			}
		})
	}
}

func TestCalculatorDivide(t *testing.T) {
	calc := &Calculator{}

	t.Run("oddiy bo'lish", func(t *testing.T) {
		result, err := calc.Divide(10, 2)
		if err != nil {
			t.Errorf("Kutilmagan xato: %v", err)
		}
		if result != 5 {
			t.Errorf("Divide(10, 2) = %d; want 5", result)
		}
	})

	t.Run("nolga bo'lish", func(t *testing.T) {
		_, err := calc.Divide(10, 0)
		if err == nil {
			t.Error("Nolga bo'lish uchun xato kutilgan edi")
		}
	})
}`
		}
	}
};

export default task;
