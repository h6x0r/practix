import { Task } from '../../../../../../../types';

export const task: Task = {
	slug: 'se-ap-copy-paste-basic',
	title: 'Copy-Paste Programming - Basic',
	difficulty: 'easy',
	tags: ['go', 'anti-patterns', 'copy-paste', 'dry', 'refactoring'],
	estimatedTime: '20m',
	isPremium: false,
	youtubeUrl: '',
	description: `Eliminate copy-paste code duplication by extracting common functionality.

**The Problem:**

Copy-Paste Programming is copying code and modifying it slightly instead of creating reusable functions. This violates the DRY (Don't Repeat Yourself) principle.

**You will refactor duplicated code into:**
1. **CalculateArea** - Generic area calculation function
2. **CalculateRectangleArea** - Uses CalculateArea
3. **CalculateSquareArea** - Uses CalculateArea

**Your Task:**

Extract common logic to eliminate duplication.`,
	initialCode: `package antipatterns

func CalculateArea(length, width float64) float64 {
}

func CalculateRectangleArea(length, width float64) float64 {
}

func CalculateSquareArea(side float64) float64 {
}`,
	solutionCode: `package antipatterns

// CalculateArea is the single source of truth for area calculation
// DRY: Don't Repeat Yourself - write logic once
func CalculateArea(length, width float64) float64 {
	return length * width	// single implementation
}

// CalculateRectangleArea delegates to shared function
// No duplication - reuses existing logic
func CalculateRectangleArea(length, width float64) float64 {
	return CalculateArea(length, width)	// reuse, don't duplicate
}

// CalculateSquareArea also delegates to shared function
// Square is just a rectangle where length == width
func CalculateSquareArea(side float64) float64 {
	return CalculateArea(side, side)	// reuse with same value
}`,
	hint1: `CalculateArea returns length * width. CalculateRectangleArea calls CalculateArea(length, width). CalculateSquareArea calls CalculateArea(side, side).`,
	hint2: `The key is to have one implementation of the multiplication logic in CalculateArea, and have the other functions delegate to it.`,
	whyItMatters: `Copy-paste programming creates maintenance nightmares. When you fix a bug, you have to remember to fix it in ALL copies!

**The Danger of Duplication:**

\`\`\`go
// BAD: Duplicated validation logic
func ValidateUser(u User) error {
	if u.Email == "" {
		return errors.New("email required")
	}
	if !strings.Contains(u.Email, "@") {
		return errors.New("invalid email")
	}
	if len(u.Password) < 8 {
		return errors.New("password too short")
	}
	return nil
}

func ValidateAdmin(a Admin) error {
	if a.Email == "" {  // DUPLICATE!
		return errors.New("email required")
	}
	if !strings.Contains(a.Email, "@") {  // DUPLICATE!
		return errors.New("invalid email")
	}
	if len(a.Password) < 8 {  // DUPLICATE!
		return errors.New("password too short")
	}
	return nil
}
// Found a bug? Have to fix it TWICE!

// GOOD: Extract common logic
func validateEmail(email string) error {
	if email == "" {
		return errors.New("email required")
	}
	if !strings.Contains(email, "@") {
		return errors.New("invalid email")
	}
	return nil
}

func validatePassword(password string) error {
	if len(password) < 8 {
		return errors.New("password too short")
	}
	return nil
}

func ValidateUser(u User) error {
	if err := validateEmail(u.Email); err != nil {
		return err
	}
	return validatePassword(u.Password)
}

func ValidateAdmin(a Admin) error {
	if err := validateEmail(a.Email); err != nil {
		return err
	}
	return validatePassword(a.Password)
}
// Fix bug ONCE, both functions benefit!
\`\`\`

**DRY Principle Benefits:**
1. Fix bugs once
2. Add features once
3. Easier to test
4. Less code to maintain`,
	order: 6,
	testCode: `package antipatterns

import (
	"testing"
)

// Test1: CalculateArea basic
func Test1(t *testing.T) {
	if CalculateArea(5.0, 4.0) != 20.0 {
		t.Error("Expected 20.0")
	}
}

// Test2: CalculateArea with zero
func Test2(t *testing.T) {
	if CalculateArea(5.0, 0) != 0 {
		t.Error("Expected 0")
	}
}

// Test3: CalculateRectangleArea uses CalculateArea
func Test3(t *testing.T) {
	if CalculateRectangleArea(10.0, 3.0) != 30.0 {
		t.Error("Expected 30.0")
	}
}

// Test4: CalculateSquareArea uses CalculateArea with same side
func Test4(t *testing.T) {
	if CalculateSquareArea(5.0) != 25.0 {
		t.Error("Expected 25.0")
	}
}

// Test5: CalculateArea with decimals
func Test5(t *testing.T) {
	result := CalculateArea(2.5, 4.0)
	if result != 10.0 {
		t.Errorf("Expected 10.0, got %f", result)
	}
}

// Test6: CalculateSquareArea with 1
func Test6(t *testing.T) {
	if CalculateSquareArea(1.0) != 1.0 {
		t.Error("Expected 1.0")
	}
}

// Test7: CalculateRectangleArea with zero width
func Test7(t *testing.T) {
	if CalculateRectangleArea(10.0, 0) != 0 {
		t.Error("Expected 0")
	}
}

// Test8: CalculateArea large numbers
func Test8(t *testing.T) {
	if CalculateArea(1000.0, 1000.0) != 1000000.0 {
		t.Error("Expected 1000000.0")
	}
}

// Test9: CalculateSquareArea with zero
func Test9(t *testing.T) {
	if CalculateSquareArea(0) != 0 {
		t.Error("Expected 0")
	}
}

// Test10: Rectangle and Square consistency
func Test10(t *testing.T) {
	side := 7.0
	if CalculateSquareArea(side) != CalculateRectangleArea(side, side) {
		t.Error("Square should equal rectangle with equal sides")
	}
}
`,
	translations: {
		ru: {
			title: 'Copy-Paste Programming - Базовый',
			description: `Устраните дублирование кода copy-paste, извлекая общую функциональность.

**Проблема:**

Copy-Paste Programming — это копирование кода и его небольшая модификация вместо создания переиспользуемых функций. Это нарушает принцип DRY (Don't Repeat Yourself).`,
			hint1: `CalculateArea возвращает length * width. CalculateRectangleArea вызывает CalculateArea(length, width). CalculateSquareArea вызывает CalculateArea(side, side).`,
			hint2: `Ключ в том, чтобы иметь одну реализацию логики умножения в CalculateArea, а другие функции делегируют ей.`,
			whyItMatters: `Copy-paste программирование создаёт кошмары обслуживания. Когда вы исправляете баг, вы должны помнить исправить его во ВСЕХ копиях!`,
			solutionCode: `package antipatterns

func CalculateArea(length, width float64) float64 {
	return length * width
}

func CalculateRectangleArea(length, width float64) float64 {
	return CalculateArea(length, width)
}

func CalculateSquareArea(side float64) float64 {
	return CalculateArea(side, side)
}`
		},
		uz: {
			title: 'Copy-Paste Programming - Asosiy',
			description: `Umumiy funksionallikni chiqarib, copy-paste kod dublikatsiyasini yo'q qiling.

**Muammo:**

Copy-Paste Programming — bu qayta foydalaniladigan funksiyalar yaratish o'rniga kodni nusxalash va uni ozgina o'zgartirish. Bu DRY (Don't Repeat Yourself) printsipini buzadi.`,
			hint1: `CalculateArea length * width ni qaytaradi. CalculateRectangleArea CalculateArea(length, width) ni chaqiradi. CalculateSquareArea CalculateArea(side, side) ni chaqiradi.`,
			hint2: `Kalit CalculateArea da ko'paytirish mantiqining bitta implementatsiyasiga ega bo'lish va boshqa funksiyalar unga topshirishdir.`,
			whyItMatters: `Copy-paste dasturlash texnik xizmat ko'rsatish dahshatlarini yaratadi. Xatoni tuzatganingizda, uni BARCHA nusxalarda tuzatishni eslab qolishingiz kerak!`,
			solutionCode: `package antipatterns

func CalculateArea(length, width float64) float64 {
	return length * width
}

func CalculateRectangleArea(length, width float64) float64 {
	return CalculateArea(length, width)
}

func CalculateSquareArea(side float64) float64 {
	return CalculateArea(side, side)
}`
		}
	}
};

export default task;
