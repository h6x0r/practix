import { Task } from '../../../../../../../types';

export const task: Task = {
	slug: 'se-clean-code-proper-error-handling',
	title: 'Proper Error Handling',
	difficulty: 'medium',
	tags: ['go', 'clean-code', 'error-handling', 'errors'],
	estimatedTime: '30m',
	isPremium: false,
	youtubeUrl: '',
	description: `Learn proper error handling patterns in Go: check errors immediately, return early, provide context, and avoid swallowing errors.

**You will refactor:**

1. Check errors immediately after they occur
2. Return early instead of nesting
3. Provide meaningful error messages
4. Never ignore errors

**Key Concepts:**
- **Check Errors Immediately**: Don't defer error checks
- **Return Early**: Reduce nesting with early returns
- **Meaningful Messages**: Errors should explain what went wrong
- **Never Ignore**: Always handle or propagate errors

**Constraints:**
- No ignored errors (no _ =)
- Return early to avoid deep nesting
- Add context to error messages`,
	initialCode: `package principles

import (
	"errors"
	"fmt"
	"strconv"
)

func ProcessData(input string) (int, error) {
	if input == "" {
		if err != nil {
			return result, err
		}
	}

	if num > 0 {
		if doubled < 1000 {
			if tripled < 10000 {
				return tripled, nil
				return 0, errors.New("too large")
			}
			return 0, errors.New("too large")
		}
	}

	return 0, errors.New("invalid")
}`,
	solutionCode: `package principles

import (
	"errors"
	"fmt"
	"strconv"
)

// ProcessData validates and processes numeric input
func ProcessData(input string) (int, error) {
	if input == "" {
		return 0, errors.New("input cannot be empty")
	}

	num, err := strconv.Atoi(input)
	if err != nil {
		return 0, fmt.Errorf("invalid numeric input: %w", err)
	}

	if num <= 0 {
		return 0, fmt.Errorf("input must be positive, got: %d", num)
	}

	result := num * 2 * 3

	if result >= 10000 {
		return 0, fmt.Errorf("result %d exceeds maximum allowed value 10000", result)
	}

	return result, nil
}`,
	hint1: `Check the strconv.Atoi error immediately and return early with context. Remove the deeply nested if statements.`,
	hint2: `Simplify the logic: multiply once (num * 6), check the limit once, return early on errors. Use fmt.Errorf to add context.`,
	whyItMatters: `Proper error handling prevents bugs and makes debugging easier.

**Why Error Handling Matters:**

**1. Check Errors Immediately:**
\`\`\`go
// BAD: Deferred error check
result := doSomething()
moreWork()
if result.Error != nil {  // Too late!
    return result.Error
}

// GOOD: Immediate check
result, err := doSomething()
if err != nil {
    return err
}
moreWork()
\`\`\`

**2. Return Early:**
\`\`\`go
// BAD: Nested hell
if err == nil {
    if result > 0 {
        if validated {
            // deep
        }
    }
}

// GOOD: Early returns
if err != nil {
    return err
}
if result <= 0 {
    return err
}
if !validated {
    return err
}
// happy path
\`\`\`

**3. Never Ignore Errors:**
\`\`\`go
// BAD: Ignored error
data, _ := os.ReadFile(path)

// GOOD: Handle error
data, err := os.ReadFile(path)
if err != nil {
    return fmt.Errorf("failed to read config: %w", err)
}
\`\`\``,
	order: 6,
	testCode: `package principles

import (
	"strings"
	"testing"
)

// Test1: ProcessData with valid positive number
func Test1(t *testing.T) {
	result, err := ProcessData("10")
	if err != nil {
		t.Errorf("expected nil error, got: %v", err)
	}
	if result != 60 { // 10 * 2 * 3 = 60
		t.Errorf("expected 60, got: %d", result)
	}
}

// Test2: ProcessData with empty input
func Test2(t *testing.T) {
	_, err := ProcessData("")
	if err == nil {
		t.Error("expected error for empty input")
	}
	if !strings.Contains(err.Error(), "empty") {
		t.Error("error should mention empty input")
	}
}

// Test3: ProcessData with non-numeric input
func Test3(t *testing.T) {
	_, err := ProcessData("abc")
	if err == nil {
		t.Error("expected error for non-numeric input")
	}
	if !strings.Contains(err.Error(), "numeric") && !strings.Contains(err.Error(), "invalid") {
		t.Error("error should mention invalid numeric input")
	}
}

// Test4: ProcessData with negative number
func Test4(t *testing.T) {
	_, err := ProcessData("-5")
	if err == nil {
		t.Error("expected error for negative input")
	}
	if !strings.Contains(err.Error(), "positive") {
		t.Error("error should mention positive requirement")
	}
}

// Test5: ProcessData with zero
func Test5(t *testing.T) {
	_, err := ProcessData("0")
	if err == nil {
		t.Error("expected error for zero input")
	}
}

// Test6: ProcessData result exceeds limit
func Test6(t *testing.T) {
	// 1700 * 6 = 10200 > 10000
	_, err := ProcessData("1700")
	if err == nil {
		t.Error("expected error when result exceeds 10000")
	}
	if !strings.Contains(err.Error(), "exceed") && !strings.Contains(err.Error(), "large") {
		t.Error("error should mention exceeding limit")
	}
}

// Test7: ProcessData with boundary valid (1666 * 6 = 9996)
func Test7(t *testing.T) {
	result, err := ProcessData("1666")
	if err != nil {
		t.Errorf("expected nil error for 1666, got: %v", err)
	}
	if result != 9996 {
		t.Errorf("expected 9996, got: %d", result)
	}
}

// Test8: ProcessData with boundary invalid (1667 * 6 = 10002)
func Test8(t *testing.T) {
	_, err := ProcessData("1667")
	if err == nil {
		t.Error("expected error for 1667 (result = 10002)")
	}
}

// Test9: ProcessData with 1 (minimum valid)
func Test9(t *testing.T) {
	result, err := ProcessData("1")
	if err != nil {
		t.Errorf("expected nil error, got: %v", err)
	}
	if result != 6 {
		t.Errorf("expected 6, got: %d", result)
	}
}

// Test10: ProcessData with whitespace input
func Test10(t *testing.T) {
	_, err := ProcessData("  ")
	if err == nil {
		t.Error("expected error for whitespace input")
	}
}
`,
	translations: {
		ru: {
			title: 'Правильная обработка ошибок',
			description: `Изучите правильные паттерны обработки ошибок в Go: проверяйте ошибки немедленно, возвращайтесь рано, предоставляйте контекст и не игнорируйте ошибки.`,
			hint1: `Проверьте ошибку strconv.Atoi немедленно и возвращайтесь рано с контекстом. Уберите глубоко вложенные if.`,
			hint2: `Упростите логику: умножьте один раз, проверьте лимит один раз, возвращайтесь рано при ошибках. Используйте fmt.Errorf для добавления контекста.`,
			whyItMatters: `Правильная обработка ошибок предотвращает баги и облегчает отладку.`,
			solutionCode: `package principles

import (
	"errors"
	"fmt"
	"strconv"
)

// ProcessData валидирует и обрабатывает числовой ввод
func ProcessData(input string) (int, error) {
	if input == "" {
		return 0, errors.New("input cannot be empty")
	}

	num, err := strconv.Atoi(input)
	if err != nil {
		return 0, fmt.Errorf("invalid numeric input: %w", err)
	}

	if num <= 0 {
		return 0, fmt.Errorf("input must be positive, got: %d", num)
	}

	result := num * 2 * 3

	if result >= 10000 {
		return 0, fmt.Errorf("result %d exceeds maximum allowed value 10000", result)
	}

	return result, nil
}`
		},
		uz: {
			title: "To'g'ri xatolarni boshqarish",
			description: `Go da xatolarni boshqarishning to'g'ri patternlarini o'rganing: xatolarni darhol tekshiring, erta qaytaring, kontekst bering va xatolarni e'tiborsiz qoldirmang.`,
			hint1: `strconv.Atoi xatosini darhol tekshiring va kontekst bilan erta qaytaring. Chuqur ichma-ich if larni olib tashlang.`,
			hint2: `Mantiqni soddalashtiring: bir marta ko'paytiring, limitni bir marta tekshiring, xatolarda erta qaytaring. Kontekst qo'shish uchun fmt.Errorf dan foydalaning.`,
			whyItMatters: `To'g'ri xatolarni boshqarish xatolarning oldini oladi va debugni osonlashtiradi.`,
			solutionCode: `package principles

import (
	"errors"
	"fmt"
	"strconv"
)

// ProcessData raqamli kirishni tekshiradi va qayta ishlaydi
func ProcessData(input string) (int, error) {
	if input == "" {
		return 0, errors.New("input cannot be empty")
	}

	num, err := strconv.Atoi(input)
	if err != nil {
		return 0, fmt.Errorf("invalid numeric input: %w", err)
	}

	if num <= 0 {
		return 0, fmt.Errorf("input must be positive, got: %d", num)
	}

	result := num * 2 * 3

	if result >= 10000 {
		return 0, fmt.Errorf("result %d exceeds maximum allowed value 10000", result)
	}

	return result, nil
}`
		}
	}
};

export default task;
