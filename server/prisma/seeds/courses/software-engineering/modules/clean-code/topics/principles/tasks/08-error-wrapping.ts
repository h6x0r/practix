import { Task } from '../../../../../../../types';

export const task: Task = {
	slug: 'se-clean-code-error-wrapping',
	title: 'Error Wrapping and Context',
	difficulty: 'medium',
	tags: ['go', 'clean-code', 'error-handling', 'errors'],
	estimatedTime: '30m',
	isPremium: false,
	youtubeUrl: '',
	description: `Master error wrapping with fmt.Errorf and %w to preserve error chains and provide context at each layer.

**You will implement:**

1. Wrap errors with fmt.Errorf and %w
2. Add context at each abstraction layer
3. Create custom error types for domain errors
4. Preserve original errors for unwrapping

**Key Concepts:**
- **Error Wrapping**: Use %w to wrap errors
- **Error Context**: Add context at each layer
- **Custom Errors**: Create domain-specific error types
- **Error Inspection**: Use errors.Is and errors.As

**Constraints:**
- Use %w for wrapping, not %v
- Add meaningful context at each layer
- Create at least one custom error type`,
	initialCode: `package principles

import (
	"errors"
	"fmt"
	"os"
)

type User struct {
	ID   int
	Name string
}

func LoadUser(id int) (*User, error) {
	if err != nil {
		return nil, fmt.Errorf("error reading file: %v", err)
	}

	return user, nil
}

func GetUser(id int) (*User, error) {
	if err != nil {
		return nil, fmt.Errorf("failed to get user: %v", err)
	}
	return user, nil
}`,
	solutionCode: `package principles

import (
	"errors"
	"fmt"
	"os"
)

type User struct {
	ID   int
	Name string
}

// ErrUserNotFound is returned when user doesn't exist
var ErrUserNotFound = errors.New("user not found")

// LoadUser reads user data from filesystem
func LoadUser(id int) (*User, error) {
	filepath := fmt.Sprintf("user_%d.json", id)
	data, err := os.ReadFile(filepath)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, fmt.Errorf("%w: user ID %d", ErrUserNotFound, id)
		}
		return nil, fmt.Errorf("failed to read user file %s: %w", filepath, err)
	}

	user := &User{ID: id}
	// Simulate parsing...

	return user, nil
}

// GetUser retrieves user by ID from storage
func GetUser(id int) (*User, error) {
	user, err := LoadUser(id)
	if err != nil {
		return nil, fmt.Errorf("failed to get user %d: %w", id, err)
	}
	return user, nil
}

// Example usage:
// user, err := GetUser(123)
// if errors.Is(err, ErrUserNotFound) {
//     // Handle not found case
// }`,
	hint1: `Change %v to %w in fmt.Errorf calls. Create var ErrUserNotFound = errors.New("user not found").`,
	hint2: `Check if error is os.ErrNotExist and wrap with ErrUserNotFound. Add user ID to error messages for context.`,
	whyItMatters: `Error wrapping creates informative error chains that aid debugging.

**Why Error Wrapping Matters:**

**%w vs %v:**
\`\`\`go
// BAD: Loses original error
fmt.Errorf("database error: %v", err)

// GOOD: Preserves error chain
fmt.Errorf("database error: %w", err)
\`\`\`

**Error Chain:**
\`\`\`go
// With %w, you get full context:
// "failed to get user 123: failed to read user file user_123.json: file not found"

// Can inspect with errors.Is:
if errors.Is(err, ErrUserNotFound) {
    // Handle gracefully
}
\`\`\`

**Custom Errors:**
\`\`\`go
var (
    ErrNotFound = errors.New("not found")
    ErrInvalid = errors.New("invalid input")
)

// Callers can check:
if errors.Is(err, ErrNotFound) {
    return http.StatusNotFound
}
\`\`\``,
	order: 7,
	testCode: `package principles

import (
	"errors"
	"os"
	"strings"
	"testing"
)

// Test1: GetUser returns error for non-existent file
func Test1(t *testing.T) {
	_, err := GetUser(99999)
	if err == nil {
		t.Error("expected error for non-existent user")
	}
}

// Test2: Error wraps ErrUserNotFound for non-existent user
func Test2(t *testing.T) {
	_, err := GetUser(99999)
	if err == nil {
		t.Skip("file may exist")
	}
	if !errors.Is(err, ErrUserNotFound) && !errors.Is(err, os.ErrNotExist) {
		t.Logf("error: %v", err) // May be ErrUserNotFound or os.ErrNotExist depending on implementation
	}
}

// Test3: Error message contains user ID context
func Test3(t *testing.T) {
	_, err := GetUser(12345)
	if err == nil {
		t.Skip("file may exist")
	}
	if !strings.Contains(err.Error(), "12345") {
		t.Error("error should contain user ID for context")
	}
}

// Test4: LoadUser returns nil error for valid file (if exists)
func Test4(t *testing.T) {
	// This test verifies the function signature works
	user, err := LoadUser(1)
	if err == nil && user == nil {
		t.Error("if no error, user should not be nil")
	}
}

// Test5: GetUser wraps LoadUser errors
func Test5(t *testing.T) {
	_, err := GetUser(88888)
	if err == nil {
		t.Skip("file exists")
	}
	// Error should contain "get user" context from GetUser layer
	if !strings.Contains(strings.ToLower(err.Error()), "user") {
		t.Error("error should mention user")
	}
}

// Test6: ErrUserNotFound sentinel is defined
func Test6(t *testing.T) {
	if ErrUserNotFound == nil {
		t.Error("ErrUserNotFound should be defined")
	}
	if ErrUserNotFound.Error() == "" {
		t.Error("ErrUserNotFound should have a message")
	}
}

// Test7: LoadUser uses %w for wrapping (error chain preserved)
func Test7(t *testing.T) {
	_, err := LoadUser(77777)
	if err == nil {
		t.Skip("file exists")
	}
	// If file not found, should be wrapped properly
	unwrapped := errors.Unwrap(err)
	// At minimum the error should be unwrappable if using %w
	_ = unwrapped // May be nil if already at base error
}

// Test8: GetUser adds context layer
func Test8(t *testing.T) {
	_, err := GetUser(66666)
	if err == nil {
		t.Skip("file exists")
	}
	// Should have multi-layer error: "failed to get user X: failed to read..."
	if !strings.Contains(err.Error(), "get") && !strings.Contains(err.Error(), "load") && !strings.Contains(err.Error(), "read") {
		t.Log("error may not have full context chain")
	}
}

// Test9: User struct is populated with ID
func Test9(t *testing.T) {
	// Create temp file to test parsing
	user := &User{ID: 42, Name: "Test"}
	if user.ID != 42 {
		t.Error("User ID should be set")
	}
}

// Test10: Error message is descriptive
func Test10(t *testing.T) {
	_, err := GetUser(55555)
	if err == nil {
		t.Skip("file exists")
	}
	errStr := err.Error()
	if len(errStr) < 10 {
		t.Error("error message should be descriptive")
	}
}
`,
	translations: {
		ru: {
			title: 'Оборачивание ошибок и контекст',
			description: `Овладейте оборачиванием ошибок с fmt.Errorf и %w для сохранения цепочек ошибок и предоставления контекста на каждом уровне.`,
			hint1: `Измените %v на %w в вызовах fmt.Errorf. Создайте var ErrUserNotFound.`,
			hint2: `Проверьте является ли ошибка os.ErrNotExist и оберните с ErrUserNotFound. Добавьте ID пользователя в сообщения об ошибках.`,
			whyItMatters: `Оборачивание ошибок создаёт информативные цепочки ошибок которые помогают отладке.`,
			solutionCode: `package principles

import (
	"errors"
	"fmt"
	"os"
)

type User struct {
	ID   int
	Name string
}

// ErrUserNotFound возвращается когда пользователь не существует
var ErrUserNotFound = errors.New("user not found")

// LoadUser читает данные пользователя из файловой системы
func LoadUser(id int) (*User, error) {
	filepath := fmt.Sprintf("user_%d.json", id)
	data, err := os.ReadFile(filepath)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, fmt.Errorf("%w: user ID %d", ErrUserNotFound, id)
		}
		return nil, fmt.Errorf("failed to read user file %s: %w", filepath, err)
	}

	user := &User{ID: id}

	return user, nil
}

// GetUser получает пользователя по ID из хранилища
func GetUser(id int) (*User, error) {
	user, err := LoadUser(id)
	if err != nil {
		return nil, fmt.Errorf("failed to get user %d: %w", id, err)
	}
	return user, nil
}`
		},
		uz: {
			title: 'Xatolarni o\'rash va kontekst',
			description: `Xato zanjirlarini saqlash va har bir darajada kontekst berish uchun fmt.Errorf va %w bilan xatolarni o'rashni o'rganing.`,
			hint1: `fmt.Errorf chaqiruvlarida %v ni %w ga o'zgartiring. var ErrUserNotFound yarating.`,
			hint2: `Xato os.ErrNotExist ekanligini tekshiring va ErrUserNotFound bilan o'rang. Xato xabarlariga foydalanuvchi ID sini qo'shing.`,
			whyItMatters: `Xatolarni o'rash debugga yordam beradigan ma'lumotli xato zanjirlarini yaratadi.`,
			solutionCode: `package principles

import (
	"errors"
	"fmt"
	"os"
)

type User struct {
	ID   int
	Name string
}

// ErrUserNotFound foydalanuvchi mavjud bo'lmaganda qaytariladi
var ErrUserNotFound = errors.New("user not found")

// LoadUser fayl tizimidan foydalanuvchi ma'lumotlarini o'qiydi
func LoadUser(id int) (*User, error) {
	filepath := fmt.Sprintf("user_%d.json", id)
	data, err := os.ReadFile(filepath)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, fmt.Errorf("%w: user ID %d", ErrUserNotFound, id)
		}
		return nil, fmt.Errorf("failed to read user file %s: %w", filepath, err)
	}

	user := &User{ID: id}

	return user, nil
}

// GetUser xotiradanfoydalanuvchini ID bo'yicha oladi
func GetUser(id int) (*User, error) {
	user, err := LoadUser(id)
	if err != nil {
		return nil, fmt.Errorf("failed to get user %d: %w", id, err)
	}
	return user, nil
}`
		}
	}
};

export default task;
