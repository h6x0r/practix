import { Task } from '../../../../../../../types';

export const task: Task = {
	slug: 'se-refactoring-rename-functions',
	title: 'Rename - Function Naming',
	difficulty: 'easy',
	tags: ['refactoring', 'rename', 'clean-code', 'go'],
	estimatedTime: '15m',
	isPremium: false,
	youtubeUrl: '',
	description: `Rename poorly named functions to clearly communicate their purpose and behavior.

**You will refactor:**

1. **Do()** - Rename to **SendWelcomeEmail()**
2. **Check()** - Rename to **IsEmailValid()**
3. **Handle()** - Rename to **FormatUserDisplayName()**

**Key Concepts:**
- **Verb-Noun Pattern**: Functions do actions (sendEmail, calculateTotal)
- **Specific Names**: Describe what, not how
- **Consistent Vocabulary**: Use same terms throughout codebase
- **Boolean Functions**: Start with is/has/can for predicates

**Before Refactoring:**

\`\`\`go
func Do(user User) { ... }        // What does it do?
func Check(s string) bool { ... } // Check what?
func Handle(u User) string { ... } // Handle how?
\`\`\`

**After Refactoring:**

\`\`\`go
func SendWelcomeEmail(user User) { ... }
func IsEmailValid(email string) bool { ... }
func FormatUserDisplayName(user User) string { ... }
\`\`\`

**When to Rename Functions:**
- Generic verbs (do, handle, process, manage)
- Name doesn't match implementation
- Similar names for different purposes
- Name is too vague or too detailed
- Boolean functions without is/has/can prefix

**Constraints:**
- Rename exactly 3 functions
- Update all function calls to use new names
- Maintain exact same functionality
- Follow Go naming conventions`,
	initialCode: `package refactoring

import (
	"fmt"
	"strings"
)

type User struct {
	Email     string
	FirstName string
	LastName  string
}

func Do(user User) {
	if Check(user.Email) {
	}
}

func Check(s string) bool {
	return strings.Contains(s, "@") && strings.Contains(s, ".")
}

func Handle(u User) string {
	return u.FirstName + " " + u.LastName
}`,
	solutionCode: `package refactoring

import (
	"fmt"
	"strings"
)

type User struct {
	Email     string
	FirstName string
	LastName  string
}

// SendWelcomeEmail sends a welcome email to the user if their email is valid
func SendWelcomeEmail(user User) {
	if IsEmailValid(user.Email) {		// clear: checking email validity
		displayName := FormatUserDisplayName(user)	// clear: formatting name
		message := fmt.Sprintf("Welcome %s!", displayName)
		fmt.Println("Sending email to:", user.Email)
		fmt.Println("Message:", message)
	}
}

// IsEmailValid checks if email has required @ and . characters
func IsEmailValid(email string) bool {
	return strings.Contains(email, "@") && strings.Contains(email, ".")
}

// FormatUserDisplayName combines first and last name with space
func FormatUserDisplayName(user User) string {
	return user.FirstName + " " + user.LastName
}`,
	hint1: `Rename the Do function to SendWelcomeEmail. Remember to update its definition, not just the name - the body stays the same.`,
	hint2: `Rename Check to IsEmailValid (update definition and the call in SendWelcomeEmail), and Handle to FormatUserDisplayName (update definition and the call in SendWelcomeEmail).`,
	whyItMatters: `Function names are the primary way developers understand what code does. Poor names lead to confusion and bugs.

**Why Function Naming Matters:**

**1. Immediate Understanding**
Good names eliminate need to read implementation:

\`\`\`go
// Before: Must read code to understand
func Process(data []byte) error {
    // 100 lines of JSON parsing, validation, and DB saving
}

// After: Intent is clear from signature
func ParseAndSaveUserProfile(jsonData []byte) error {
    // Same 100 lines, but you know what it does
}
\`\`\`

**2. Prevent Misuse**
Clear names prevent incorrect usage:

\`\`\`go
// Before: Misleading name
func GetUser(id int) *User {
    user := db.QueryUser(id)
    user.LastAccessed = time.Now()
    db.UpdateUser(user)  // Surprise! It modifies data
    return user
}

// After: Name reveals side effect
func GetUserAndUpdateLastAccessed(id int) *User {
    user := db.QueryUser(id)
    user.LastAccessed = time.Now()
    db.UpdateUser(user)
    return user
}

// Or better: Separate concerns
func GetUser(id int) *User {
    return db.QueryUser(id)
}

func UpdateUserLastAccessed(userID int) error {
    return db.UpdateLastAccessed(userID, time.Now())
}
\`\`\`

**3. Code Reviews Focus on Logic**
Reviewers understand intent without explanation:

\`\`\`go
// Before: Reviewer must ask "what does transform do?"
func transform(orders []Order) []Order {
    var result []Order
    for _, o := range orders {
        if o.Status == "pending" && time.Since(o.Created) > 24*time.Hour {
            result = append(result, o)
        }
    }
    return result
}

// After: Self-explanatory
func FilterStaleOrders(orders []Order) []Order {
    const staleThreshold = 24 * time.Hour
    var staleOrders []Order

    for _, order := range orders {
        isPending := order.Status == "pending"
        isOlderThan24Hours := time.Since(order.Created) > staleThreshold

        if isPending && isOlderThan24Hours {
            staleOrders = append(staleOrders, order)
        }
    }
    return staleOrders
}
\`\`\`

**4. Searchability and Refactoring**
Specific names make codebase navigation easy:

\`\`\`go
// Before: Generic names - hard to search
func Get(id string) interface{} { ... }
func Set(id string, v interface{}) { ... }
func Update(data map[string]interface{}) { ... }

// Searching for "Get" returns 1000+ results

// After: Specific names - easy to find
func GetUserByID(userID string) (*User, error) { ... }
func SetUserPreferences(userID string, prefs *Preferences) error { ... }
func UpdateUserProfile(userID string, profile *Profile) error { ... }

// Searching for "GetUserByID" returns exact matches
\`\`\`

**5. Prevents "Comment Crutches"**
Good names eliminate need for comments:

\`\`\`go
// Before: Comment needed to explain
// Converts temperature from Celsius to Fahrenheit
func convert(c float64) float64 {
    return c*1.8 + 32
}

// After: Name explains itself
func CelsiusToFahrenheit(celsius float64) float64 {
    const conversionFactor = 1.8
    const offset = 32
    return celsius*conversionFactor + offset
}
\`\`\`

**Function Naming Patterns:**

**Commands (no return value):**
- Verb + Noun: \`SendEmail\`, \`DeleteUser\`, \`SaveOrder\`
- Action: \`Initialize\`, \`Execute\`, \`Cleanup\`

**Queries (return value):**
- Get + Noun: \`GetUser\`, \`GetBalance\`, \`GetOrders\`
- Calculate + Noun: \`CalculateTotal\`, \`CalculateTax\`
- Find + Noun: \`FindUserByEmail\`, \`FindActiveOrders\`

**Predicates (return bool):**
- Is + Adjective: \`IsValid\`, \`IsEmpty\`, \`IsExpired\`
- Has + Noun: \`HasPermission\`, \`HasDiscount\`, \`HasItems\`
- Can + Verb: \`CanEdit\`, \`CanDelete\`, \`CanProcess\`

**Converters:**
- Type + To + Type: \`StringToInt\`, \`JSONToUser\`
- From + Type: \`FromJSON\`, \`FromString\`

**Real-World Examples:**

\`\`\`go
// Authentication
func AuthenticateUser(email, password string) (*User, error)
func IsUserAuthenticated(token string) bool
func RefreshAuthToken(oldToken string) (string, error)
func RevokeUserSession(sessionID string) error

// Payment Processing
func ChargeCustomer(amount float64, customerID string) error
func RefundPayment(transactionID string) error
func CalculateOrderTotal(items []Item) float64
func ValidatePaymentMethod(method PaymentMethod) error

// Email Operations
func SendPasswordResetEmail(userEmail string) error
func QueueWelcomeEmail(user User) error
func FormatEmailSubject(template string, data map[string]string) string
func IsEmailDeliverable(email string) bool
\`\`\`

**Function Names to Avoid:**
- Generic verbs: \`do\`, \`handle\`, \`process\`, \`manage\`, \`execute\`
- Vague actions: \`doStuff\`, \`handleData\`, \`processInfo\`
- Manager/Handler: \`UserManager\`, \`DataHandler\` (what do they manage?)
- Redundant type: \`getUserUser\`, \`createOrderOrder\`

**Side Effect Communication:**
Name should reveal if function modifies state:

\`\`\`go
// Pure function (no side effects)
func CalculateDiscount(price float64, rate float64) float64

// Side effect in name
func SaveAndCalculateDiscount(order *Order) float64

// Better: Separate concerns
func CalculateOrderDiscount(order Order) float64
func SaveOrder(order Order) error
\`\`\`

**Length Guidelines:**
- Public functions: Descriptive, 2-4 words OK
- Private helpers: Can be shorter if obvious in context
- Avoid: Single letter functions (except very common like \`f\` in functional programming)

Remember: "There are only two hard things in Computer Science: cache invalidation and naming things." - Phil Karlton`,
	order: 5,
	testCode: `package refactoring

import (
	"testing"
)

// Test1: IsEmailValid with valid email
func Test1(t *testing.T) {
	if !IsEmailValid("test@example.com") {
		t.Error("expected valid for test@example.com")
	}
}

// Test2: IsEmailValid without @
func Test2(t *testing.T) {
	if IsEmailValid("testexample.com") {
		t.Error("expected invalid without @")
	}
}

// Test3: IsEmailValid without .
func Test3(t *testing.T) {
	if IsEmailValid("test@examplecom") {
		t.Error("expected invalid without .")
	}
}

// Test4: IsEmailValid with minimal valid
func Test4(t *testing.T) {
	if !IsEmailValid("a@b.c") {
		t.Error("expected valid for a@b.c")
	}
}

// Test5: FormatUserDisplayName with normal names
func Test5(t *testing.T) {
	user := User{FirstName: "John", LastName: "Doe", Email: "john@example.com"}
	result := FormatUserDisplayName(user)
	if result != "John Doe" {
		t.Errorf("expected 'John Doe', got '%s'", result)
	}
}

// Test6: FormatUserDisplayName with empty first name
func Test6(t *testing.T) {
	user := User{FirstName: "", LastName: "Doe", Email: "a@b.com"}
	result := FormatUserDisplayName(user)
	if result != " Doe" {
		t.Errorf("expected ' Doe', got '%s'", result)
	}
}

// Test7: FormatUserDisplayName with empty last name
func Test7(t *testing.T) {
	user := User{FirstName: "John", LastName: "", Email: "a@b.com"}
	result := FormatUserDisplayName(user)
	if result != "John " {
		t.Errorf("expected 'John ', got '%s'", result)
	}
}

// Test8: SendWelcomeEmail with valid email runs without panic
func Test8(t *testing.T) {
	user := User{FirstName: "Test", LastName: "User", Email: "test@example.com"}
	SendWelcomeEmail(user) // Should not panic
}

// Test9: SendWelcomeEmail with invalid email (no crash, just skips)
func Test9(t *testing.T) {
	user := User{FirstName: "Test", LastName: "User", Email: "invalid"}
	SendWelcomeEmail(user) // Should not panic, just skip
}

// Test10: User struct fields work correctly
func Test10(t *testing.T) {
	user := User{Email: "e@x.com", FirstName: "F", LastName: "L"}
	if user.Email != "e@x.com" || user.FirstName != "F" || user.LastName != "L" {
		t.Error("User fields not set correctly")
	}
}
`,
	translations: {
		ru: {
			title: 'Rename - Именование функций',
			description: `Переименуйте плохо названные функции, чтобы чётко передать их назначение и поведение.

**Вы выполните рефакторинг:**

1. **Do()** - Переименовать в **SendWelcomeEmail()**
2. **Check()** - Переименовать в **IsEmailValid()**
3. **Handle()** - Переименовать в **FormatUserDisplayName()**`,
			hint1: `Переименуйте функцию Do в SendWelcomeEmail. Не забудьте обновить её определение, а не только имя - тело остаётся прежним.`,
			hint2: `Переименуйте Check в IsEmailValid (обновите определение и вызов в SendWelcomeEmail), и Handle в FormatUserDisplayName (обновите определение и вызов в SendWelcomeEmail).`,
			whyItMatters: `Имена функций — основной способ, которым разработчики понимают, что делает код. Плохие имена приводят к путанице и багам.`
		},
		uz: {
			title: 'Rename - Funktsiya nomlash',
			description: `Yomon nomlangan funktsiyalarni ularning maqsadi va xatti-harakatini aniq etkazadigan nomlarga o'zgartiring.

**Siz refaktoring qilasiz:**

1. **Do()** - **SendWelcomeEmail()** ga qayta nomlash
2. **Check()** - **IsEmailValid()** ga qayta nomlash
3. **Handle()** - **FormatUserDisplayName()** ga qayta nomlash`,
			hint1: `Do funktsiyasini SendWelcomeEmail ga qayta nomlang. Uning ta'rifini yangilashni unutmang, faqat nomini emas - tanasi o'zgarmaydi.`,
			hint2: `Check ni IsEmailValid ga (ta'rif va SendWelcomeEmail dagi chaqiruvni yangilang), va Handle ni FormatUserDisplayName ga (ta'rif va SendWelcomeEmail dagi chaqiruvni yangilang) qayta nomlang.`,
			whyItMatters: `Funktsiya nomlari dasturchilarga kod nimani qilishini tushunishning asosiy usuli. Yomon nomlar chalkashlik va xatolarga olib keladi.`
		}
	}
};

export default task;
