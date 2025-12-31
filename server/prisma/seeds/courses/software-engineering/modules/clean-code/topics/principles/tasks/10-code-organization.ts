import { Task } from '../../../../../../../types';

export const task: Task = {
	slug: 'se-clean-code-organization',
	title: 'Code Organization and Structure',
	difficulty: 'medium',
	tags: ['go', 'clean-code', 'organization', 'structure'],
	estimatedTime: '30m',
	isPremium: false,
	youtubeUrl: '',
	description: `Learn to organize code logically: group related functions, order by abstraction level, and follow the newspaper metaphor.

**You will reorganize:**

1. Group related functions together
2. Order functions by abstraction level (high to low)
3. Place helper functions near their callers
4. Follow the "newspaper" structure

**Key Concepts:**
- **Newspaper Metaphor**: Most important at top, details at bottom
- **Vertical Ordering**: High-level functions first, helpers below
- **Grouping**: Related functions together
- **Stepdown Rule**: Reading code from top to bottom

**Constraints:**
- Main function at top
- Helper functions below callers
- Group related operations`,
	initialCode: `package principles

import "fmt"

func validateName(name string) bool {
	return len(name) >= 2 && len(name) <= 50
}

func CreateUser(name, email string) error {
		return fmt.Errorf("invalid input")
	}
}

func buildUser(name, email string) *User {
}

type User struct {
	Name  string
}

func validateEmail(email string) bool {
}

func saveUser(user *User) error {
	return nil
}

func validateInput(name, email string) bool {
}`,
	solutionCode: `package principles

import "fmt"

type User struct {
	Name  string
	Email string
}

// CreateUser is the main entry point for user creation
// High-level function that orchestrates the process
func CreateUser(name, email string) error {
	if !validateInput(name, email) {
		return fmt.Errorf("invalid input")
	}

	user := buildUser(name, email)
	return saveUser(user)
}

// validateInput checks if both name and email are valid
// Helper function used by CreateUser
func validateInput(name, email string) bool {
	return validateName(name) && validateEmail(email)
}

// validateName checks if name meets length requirements
func validateName(name string) bool {
	return len(name) >= 2 && len(name) <= 50
}

// validateEmail checks if email has minimum length
func validateEmail(email string) bool {
	return len(email) > 3
}

// buildUser constructs a User object from validated inputs
func buildUser(name, email string) *User {
	return &User{Name: name, Email: email}
}

// saveUser persists user to storage
func saveUser(user *User) error {
	fmt.Printf("Saving user: %s\n", user.Name)
	return nil
}`,
	hint1: `Put type definitions first, then CreateUser (main function), then its immediate helpers (validateInput, buildUser, saveUser), then lower-level helpers (validateName, validateEmail).`,
	hint2: `Follow the stepdown rule: each function is followed by functions it calls. Read top to bottom like a newspaper: headline (CreateUser) then details (helpers).`,
	whyItMatters: `Good organization makes code easy to navigate and understand.

**Why Organization Matters:**

**Newspaper Structure:**
\`\`\`go
// HIGH LEVEL (headline)
func ProcessOrder(order *Order) error {
    if err := validateOrder(order); err != nil {
        return err
    }
    return saveOrder(order)
}

// MEDIUM LEVEL (subheading)
func validateOrder(order *Order) error {
    if !hasItems(order) {
        return errors.New("no items")
    }
    return nil
}

// LOW LEVEL (details)
func hasItems(order *Order) bool {
    return len(order.Items) > 0
}
\`\`\`

**Benefits:**
- Read top to bottom naturally
- Find related code quickly
- Understand flow without jumping
- Easy to navigate large files`,
	order: 9,
	testCode: `package principles

import (
	"testing"
)

// Test1: CreateUser with valid inputs
func Test1(t *testing.T) {
	err := CreateUser("John", "john@example.com")
	if err != nil {
		t.Errorf("expected nil error, got: %v", err)
	}
}

// Test2: CreateUser with invalid name (too short)
func Test2(t *testing.T) {
	err := CreateUser("J", "j@e.com")
	if err == nil {
		t.Error("expected error for name too short")
	}
}

// Test3: CreateUser with invalid email (too short)
func Test3(t *testing.T) {
	err := CreateUser("John", "ab")
	if err == nil {
		t.Error("expected error for email too short")
	}
}

// Test4: validateName with valid name
func Test4(t *testing.T) {
	if !validateName("Alice") {
		t.Error("expected true for valid name")
	}
}

// Test5: validateName with boundary minimum (2 chars)
func Test5(t *testing.T) {
	if !validateName("Jo") {
		t.Error("expected true for 2 char name")
	}
}

// Test6: validateName with boundary maximum (50 chars)
func Test6(t *testing.T) {
	name := "AAAAAAAAAABBBBBBBBBBCCCCCCCCCCDDDDDDDDDDEEEEEEEEEE" // 50 chars
	if !validateName(name) {
		t.Error("expected true for 50 char name")
	}
}

// Test7: validateName with too long name (51 chars)
func Test7(t *testing.T) {
	name := "AAAAAAAAAABBBBBBBBBBCCCCCCCCCCDDDDDDDDDDEEEEEEEEEEF" // 51 chars
	if validateName(name) {
		t.Error("expected false for 51 char name")
	}
}

// Test8: validateEmail with valid email
func Test8(t *testing.T) {
	if !validateEmail("a@b.com") {
		t.Error("expected true for valid email")
	}
}

// Test9: validateEmail with boundary (4 chars is valid)
func Test9(t *testing.T) {
	if !validateEmail("a@bc") {
		t.Error("expected true for 4 char email")
	}
}

// Test10: validateInput checks both name and email
func Test10(t *testing.T) {
	if !validateInput("John", "john@example.com") {
		t.Error("expected true for valid name and email")
	}
	if validateInput("J", "john@example.com") {
		t.Error("expected false for invalid name")
	}
	if validateInput("John", "ab") {
		t.Error("expected false for invalid email")
	}
}
`,
	translations: {
		ru: {
			title: 'Организация и структура кода',
			description: `Научитесь логически организовывать код: группируйте связанные функции, упорядочивайте по уровню абстракции и следуйте метафоре газеты.`,
			hint1: `Поместите определения типов первыми, затем CreateUser (главную функцию), затем её непосредственные помощники, затем помощники нижнего уровня.`,
			hint2: `Следуйте правилу понижения: каждая функция следует за функциями которые она вызывает. Читайте сверху вниз как газету.`,
			whyItMatters: `Хорошая организация делает код лёгким для навигации и понимания.`,
			solutionCode: `package principles

import "fmt"

type User struct {
	Name  string
	Email string
}

// CreateUser главная точка входа для создания пользователя
// Высокоуровневая функция оркеструющая процесс
func CreateUser(name, email string) error {
	if !validateInput(name, email) {
		return fmt.Errorf("invalid input")
	}

	user := buildUser(name, email)
	return saveUser(user)
}

// validateInput проверяет валидность имени и email
// Вспомогательная функция используемая CreateUser
func validateInput(name, email string) bool {
	return validateName(name) && validateEmail(email)
}

// validateName проверяет соответствие имени требованиям длины
func validateName(name string) bool {
	return len(name) >= 2 && len(name) <= 50
}

// validateEmail проверяет минимальную длину email
func validateEmail(email string) bool {
	return len(email) > 3
}

// buildUser конструирует объект User из валидированных входов
func buildUser(name, email string) *User {
	return &User{Name: name, Email: email}
}

// saveUser сохраняет пользователя в хранилище
func saveUser(user *User) error {
	fmt.Printf("Saving user: %s\n", user.Name)
	return nil
}`
		},
		uz: {
			title: 'Kod tashkil etish va struktura',
			description: `Kodni mantiqiy tashkil etishni o'rganing: bog'liq funksiyalarni guruhlang, abstraktsiya darajasi bo'yicha tartiblang va gazeta metaforasiga amal qiling.`,
			hint1: `Avval tur ta'riflarini qo'ying, keyin CreateUser (asosiy funksiya), keyin uning to'g'ridan-to'g'ri yordamchilari, keyin pastki darajali yordamchilar.`,
			hint2: `Pasayish qoidasiga amal qiling: har bir funksiya o'zi chaqiradigan funksiyalardan keyin keladi. Yuqoridan pastga gazeta kabi o'qing.`,
			whyItMatters: `Yaxshi tashkilot kodni navigatsiya qilish va tushunishni osonlashtiradi.`,
			solutionCode: `package principles

import "fmt"

type User struct {
	Name  string
	Email string
}

// CreateUser foydalanuvchi yaratish uchun asosiy kirish nuqtasi
// Jarayonni boshqaradigan yuqori darajali funksiya
func CreateUser(name, email string) error {
	if !validateInput(name, email) {
		return fmt.Errorf("invalid input")
	}

	user := buildUser(name, email)
	return saveUser(user)
}

// validateInput ism va email haqiqiyligini tekshiradi
// CreateUser tomonidan ishlatiladigan yordamchi funksiya
func validateInput(name, email string) bool {
	return validateName(name) && validateEmail(email)
}

// validateName ism uzunlik talablariga mos kelishini tekshiradi
func validateName(name string) bool {
	return len(name) >= 2 && len(name) <= 50
}

// validateEmail email minimal uzunligini tekshiradi
func validateEmail(email string) bool {
	return len(email) > 3
}

// buildUser tasdiqlangan kirishlardan User ob'ektini quradi
func buildUser(name, email string) *User {
	return &User{Name: name, Email: email}
}

// saveUser foydalanuvchini xotiraga saqlaydi
func saveUser(user *User) error {
	fmt.Printf("Saving user: %s\n", user.Name)
	return nil
}`
		}
	}
};

export default task;
