import { Task } from '../../../../../../../types';

export const task: Task = {
	slug: 'se-ap-magic-numbers-advanced',
	title: 'Magic Numbers Anti-pattern - Advanced',
	difficulty: 'medium',
	tags: ['go', 'anti-patterns', 'magic-numbers', 'constants', 'refactoring'],
	estimatedTime: '25m',
	isPremium: false,
	youtubeUrl: '',
	description: `Use typed constants and enums for better type safety and code clarity.

**The Problem:**

Using raw numbers for status codes, error codes, or configuration values leads to errors and unclear code.

**You will implement:**

A proper HTTP status system using typed constants.

**Define:**
- Custom type HTTPStatus
- Constants for common status codes
- Helper function to check status types

**Your Task:**

Create a type-safe status code system.`,
	initialCode: `package antipatterns

func IsSuccess(status HTTPStatus) bool {
}

func IsClientError(status HTTPStatus) bool {
}

func IsServerError(status HTTPStatus) bool {
}`,
	solutionCode: `package antipatterns

// HTTPStatus is a typed constant for HTTP status codes
// Using a custom type provides type safety
type HTTPStatus int

// Typed constants - compiler ensures we use HTTPStatus type
// Self-documenting - each constant has a clear meaning
const (
	StatusOK            HTTPStatus = 200	// success
	StatusCreated       HTTPStatus = 201	// resource created
	StatusBadRequest    HTTPStatus = 400	// client error: bad request
	StatusUnauthorized  HTTPStatus = 401	// client error: not authenticated
	StatusNotFound      HTTPStatus = 404	// client error: resource not found
	StatusInternalError HTTPStatus = 500	// server error
)

// IsSuccess checks if status is in 2xx range (success)
// Range constants make the logic clear
func IsSuccess(status HTTPStatus) bool {
	return status >= 200 && status < 300	// 2xx range
}

// IsClientError checks if status is in 4xx range (client errors)
// Range constants make the logic clear
func IsClientError(status HTTPStatus) bool {
	return status >= 400 && status < 500	// 4xx range
}

// IsServerError checks if status is in 5xx range (server errors)
// Range constants make the logic clear
func IsServerError(status HTTPStatus) bool {
	return status >= 500 && status < 600	// 5xx range
}`,
	hint1: `Define HTTPStatus as type HTTPStatus int. Define constants with HTTPStatus type. IsSuccess checks status >= 200 && status < 300.`,
	hint2: `IsClientError checks status >= 400 && status < 500. IsServerError checks status >= 500 && status < 600. All use range comparisons.`,
	whyItMatters: `Typed constants provide compile-time safety and make code more maintainable than raw numbers.

**Type Safety Benefits:**

\`\`\`go
// BAD: Raw integers - no type safety
func HandleResponse(statusCode int) {
	if statusCode == 200 {  // typo: 200 or 201?
		// success
	}
}

HandleResponse(999)  // compiles! But 999 is invalid!
HandleResponse(-1)   // compiles! But -1 is invalid!

// GOOD: Typed constants - compiler enforces correctness
type HTTPStatus int

const (
	StatusOK HTTPStatus = 200
	StatusCreated HTTPStatus = 201
)

func HandleResponse(status HTTPStatus) {
	if status == StatusOK {  // clear and type-safe
		// success
	}
}

HandleResponse(StatusOK)      // correct
HandleResponse(HTTPStatus(999)) // explicit cast required, signals something unusual
// HandleResponse(999)        // compile error! Must be HTTPStatus type
\`\`\`

**Enum Pattern in Go:**

\`\`\`go
// Define custom type for enum
type OrderStatus int

// Use iota for auto-incrementing constants
const (
	OrderPending OrderStatus = iota  // 0
	OrderProcessing                  // 1
	OrderShipped                     // 2
	OrderDelivered                   // 3
	OrderCancelled                   // 4
)

// Add String() method for readable output
func (s OrderStatus) String() string {
	return [...]string{
		"Pending",
		"Processing",
		"Shipped",
		"Delivered",
		"Cancelled",
	}[s]
}

// Now you can use it type-safely
var status OrderStatus = OrderPending
fmt.Println(status)  // prints "Pending"
\`\`\`

**Real-World Example:**

\`\`\`go
// BAD: Magic numbers for permissions
func HasPermission(user User, permission int) bool {
	return user.Permissions & permission != 0
}

if HasPermission(user, 4) {  // What is 4?
	// allow action
}

// GOOD: Typed permission constants
type Permission uint32

const (
	PermissionRead   Permission = 1 << iota  // 1
	PermissionWrite                          // 2
	PermissionDelete                         // 4
	PermissionAdmin                          // 8
)

func (p Permission) String() string {
	var perms []string
	if p&PermissionRead != 0 {
		perms = append(perms, "Read")
	}
	if p&PermissionWrite != 0 {
		perms = append(perms, "Write")
	}
	if p&PermissionDelete != 0 {
		perms = append(perms, "Delete")
	}
	if p&PermissionAdmin != 0 {
		perms = append(perms, "Admin")
	}
	return strings.Join(perms, "|")
}

func HasPermission(user User, permission Permission) bool {
	return user.Permissions&permission != 0
}

if HasPermission(user, PermissionDelete) {  // Clear!
	// allow action
}

userPerms := PermissionRead | PermissionWrite
fmt.Println(userPerms)  // prints "Read|Write"
\`\`\`

**Benefits:**
1. **Type Safety**: Compiler catches misuse
2. **Autocomplete**: IDE suggests valid values
3. **Refactoring**: Easy to find all uses
4. **Documentation**: Self-documenting code`,
	order: 9,
	testCode: `package antipatterns

import (
	"testing"
)

// Test1: StatusOK is 200
func Test1(t *testing.T) {
	if StatusOK != 200 {
		t.Error("StatusOK should be 200")
	}
}

// Test2: IsSuccess with 200
func Test2(t *testing.T) {
	if !IsSuccess(StatusOK) {
		t.Error("200 should be success")
	}
}

// Test3: IsSuccess with 201
func Test3(t *testing.T) {
	if !IsSuccess(StatusCreated) {
		t.Error("201 should be success")
	}
}

// Test4: IsSuccess with 400 is false
func Test4(t *testing.T) {
	if IsSuccess(StatusBadRequest) {
		t.Error("400 should not be success")
	}
}

// Test5: IsClientError with 400
func Test5(t *testing.T) {
	if !IsClientError(StatusBadRequest) {
		t.Error("400 should be client error")
	}
}

// Test6: IsClientError with 404
func Test6(t *testing.T) {
	if !IsClientError(StatusNotFound) {
		t.Error("404 should be client error")
	}
}

// Test7: IsServerError with 500
func Test7(t *testing.T) {
	if !IsServerError(StatusInternalError) {
		t.Error("500 should be server error")
	}
}

// Test8: IsServerError with 200 is false
func Test8(t *testing.T) {
	if IsServerError(StatusOK) {
		t.Error("200 should not be server error")
	}
}

// Test9: IsClientError with 500 is false
func Test9(t *testing.T) {
	if IsClientError(StatusInternalError) {
		t.Error("500 should not be client error")
	}
}

// Test10: HTTPStatus type works as expected
func Test10(t *testing.T) {
	var status HTTPStatus = 299
	if !IsSuccess(status) {
		t.Error("299 should be success (2xx range)")
	}
}
`,
	translations: {
		ru: {
			title: 'Антипаттерн Magic Numbers - Продвинутый',
			description: `Используйте типизированные константы и перечисления для лучшей типобезопасности и ясности кода.`,
			hint1: `Определите HTTPStatus как type HTTPStatus int. Определите константы с типом HTTPStatus. IsSuccess проверяет status >= 200 && status < 300.`,
			hint2: `IsClientError проверяет status >= 400 && status < 500. IsServerError проверяет status >= 500 && status < 600. Все используют сравнения диапазонов.`,
			whyItMatters: `Типизированные константы обеспечивают безопасность на этапе компиляции и делают код более поддерживаемым, чем сырые числа.`,
			solutionCode: `package antipatterns

type HTTPStatus int

const (
	StatusOK            HTTPStatus = 200
	StatusCreated       HTTPStatus = 201
	StatusBadRequest    HTTPStatus = 400
	StatusUnauthorized  HTTPStatus = 401
	StatusNotFound      HTTPStatus = 404
	StatusInternalError HTTPStatus = 500
)

func IsSuccess(status HTTPStatus) bool {
	return status >= 200 && status < 300
}

func IsClientError(status HTTPStatus) bool {
	return status >= 400 && status < 500
}

func IsServerError(status HTTPStatus) bool {
	return status >= 500 && status < 600
}`
		},
		uz: {
			title: 'Magic Numbers Anti-pattern - Ilg\'or',
			description: `Yaxshiroq tip xavfsizligi va kod ravshanlik uchun tiplangan konstantalar va enum lardan foydalaning.`,
			hint1: `HTTPStatus ni type HTTPStatus int sifatida aniqlang. HTTPStatus tipi bilan konstantalar aniqlang. IsSuccess status >= 200 && status < 300 ni tekshiradi.`,
			hint2: `IsClientError status >= 400 && status < 500 ni tekshiradi. IsServerError status >= 500 && status < 600 ni tekshiradi. Barcha diapazon solishtirmalaridan foydalanadi.`,
			whyItMatters: `Tiplangan konstantalar kompilyatsiya vaqtida xavfsizlik beradi va kodni xom raqamlarga qaraganda ko'proq qo'llab-quvvatlanadigan qiladi.`,
			solutionCode: `package antipatterns

type HTTPStatus int

const (
	StatusOK            HTTPStatus = 200
	StatusCreated       HTTPStatus = 201
	StatusBadRequest    HTTPStatus = 400
	StatusUnauthorized  HTTPStatus = 401
	StatusNotFound      HTTPStatus = 404
	StatusInternalError HTTPStatus = 500
)

func IsSuccess(status HTTPStatus) bool {
	return status >= 200 && status < 300
}

func IsClientError(status HTTPStatus) bool {
	return status >= 400 && status < 500
}

func IsServerError(status HTTPStatus) bool {
	return status >= 500 && status < 600
}`
		}
	}
};

export default task;
