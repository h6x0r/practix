import { Task } from '../../../../../../../types';

export const task: Task = {
	slug: 'se-refactoring-param-object-long',
	title: 'Introduce Parameter Object - Long Parameter List',
	difficulty: 'medium',
	tags: ['refactoring', 'parameter-object', 'clean-code', 'go'],
	estimatedTime: '25m',
	isPremium: false,
	youtubeUrl: '',
	description: `Replace a long parameter list with a single parameter object that groups related data together.

**You will refactor:**

1. **CreateUser()** - Has 6 parameters (too many!)
2. Create **UserData struct** to group user fields
3. Update **CreateUser** to accept UserData parameter
4. Update **SaveUser** to use the same struct

**Key Concepts:**
- **Preserve Whole Object**: Pass object instead of many fields
- **Data Clumping**: Parameters that travel together belong together
- **Readability**: Fewer parameters are easier to understand
- **Maintainability**: Add new fields without changing signatures

**Before Refactoring:**

\`\`\`go
func CreateUser(firstName, lastName, email, phone, address, city string) User {
    // 6 parameters - hard to remember order
}
\`\`\`

**After Refactoring:**

\`\`\`go
type UserData struct {
    FirstName string
    LastName  string
    Email     string
    Phone     string
    Address   string
    City      string
}

func CreateUser(data UserData) User {
    // Single parameter - clear and extensible
}
\`\`\`

**When to Introduce Parameter Object:**
- 3+ parameters that always travel together
- Same parameter group in multiple functions
- Parameters represent a cohesive concept
- Adding related parameters frequently
- Hard to remember parameter order

**Constraints:**
- Create UserData struct with all 6 fields
- Update CreateUser to accept UserData
- Update SaveUser to accept UserData
- Maintain exact same user creation logic`,
	initialCode: `package refactoring

import "fmt"

type User struct {
	ID        int
	FirstName string
	LastName  string
	Email     string
	Phone     string
	Address   string
	City      string
}

func CreateUser(firstName, lastName, email, phone, address, city string) User {
	}
}

func SaveUser(firstName, lastName, email, phone, address, city string) {
}`,
	solutionCode: `package refactoring

import "fmt"

type User struct {
	ID        int
	FirstName string
	LastName  string
	Email     string
	Phone     string
	Address   string
	City      string
}

// UserData groups related user information together
type UserData struct {
	FirstName string
	LastName  string
	Email     string
	Phone     string
	Address   string
	City      string
}

var nextID = 1

// CreateUser now takes single parameter object instead of 6 individual parameters
func CreateUser(data UserData) User {
	user := User{
		ID:        nextID,
		FirstName: data.FirstName,	// access through data object
		LastName:  data.LastName,
		Email:     data.Email,
		Phone:     data.Phone,
		Address:   data.Address,
		City:      data.City,
	}
	nextID++
	return user
}

// SaveUser also uses the parameter object for consistency
func SaveUser(data UserData) {
	fmt.Printf("Saving user: %s %s, %s, %s, %s, %s\n",
		data.FirstName, data.LastName, data.Email,
		data.Phone, data.Address, data.City)
}`,
	hint1: `Create a UserData struct with six string fields matching the parameter names: FirstName, LastName, Email, Phone, Address, City.`,
	hint2: `Change both function signatures to accept 'data UserData' instead of the six string parameters. Update the function bodies to access fields via 'data.FirstName', 'data.LastName', etc.`,
	whyItMatters: `Parameter objects reduce complexity, improve readability, and make code more maintainable and extensible.

**Why Introduce Parameter Object Matters:**

**1. Reduced Cognitive Load**
Fewer parameters are easier to understand:

\`\`\`go
// Before: Which parameter is which? Easy to mix up order
func SendEmail(from, to, cc, bcc, subject, body, replyTo, priority string, attachments []string) error {
    // Called like this - error-prone:
    SendEmail("me@example.com", "you@example.com", "", "", "Hello", "World", "", "high", nil)
    // Wait, which empty string is which?
}

// After: Crystal clear
type EmailMessage struct {
    From        string
    To          string
    CC          string
    BCC         string
    Subject     string
    Body        string
    ReplyTo     string
    Priority    string
    Attachments []string
}

func SendEmail(message EmailMessage) error {
    // Called like this - self-documenting:
    SendEmail(EmailMessage{
        From:     "me@example.com",
        To:       "you@example.com",
        Subject:  "Hello",
        Body:     "World",
        Priority: "high",
    })
    // Missing fields are obvious, order doesn't matter
}
\`\`\`

**2. Easy to Extend**
Add new fields without changing signatures:

\`\`\`go
// Before: Need to add timezone? Must change every caller!
func ScheduleMeeting(title, location string, start, end time.Time) error {
    // Used in 50 places
}

// Adding timezone parameter breaks all 50 call sites:
func ScheduleMeeting(title, location string, start, end time.Time, timezone string) error {
    // 50 compile errors to fix!
}

// After: Add fields without breaking callers
type MeetingDetails struct {
    Title    string
    Location string
    Start    time.Time
    End      time.Time
    Timezone string  // NEW: existing code still compiles, uses zero value
}

func ScheduleMeeting(details MeetingDetails) error {
    // Existing callers don't break
}
\`\`\`

**3. Eliminate Data Clumps**
Parameters that travel together belong together:

\`\`\`go
// Before: Same parameters in multiple functions (data clump smell)
func CreateInvoice(custName, custEmail, custAddr, custCity string, items []Item) Invoice {
    // customer parameters always together
}

func SendInvoice(custName, custEmail, custAddr, custCity string, invoice Invoice) {
    // same customer parameters again
}

func UpdateBilling(custName, custEmail, custAddr, custCity string, amount float64) {
    // and again...
}

// After: Group related data
type Customer struct {
    Name    string
    Email   string
    Address string
    City    string
}

func CreateInvoice(customer Customer, items []Item) Invoice {
    // Clean signature
}

func SendInvoice(customer Customer, invoice Invoice) {
    // Consistent usage
}

func UpdateBilling(customer Customer, amount float64) {
    // Easy to understand
}
\`\`\`

**4. Validation in One Place**
Validate parameter object once:

\`\`\`go
// Before: Duplicate validation in every function
func RegisterUser(email, password, phone string) error {
    if !isValidEmail(email) { return errors.New("invalid email") }
    if len(password) < 8 { return errors.New("password too short") }
    if !isValidPhone(phone) { return errors.New("invalid phone") }
    // register user
}

func UpdateUser(email, password, phone string) error {
    if !isValidEmail(email) { return errors.New("invalid email") }
    if len(password) < 8 { return errors.New("password too short") }
    if !isValidPhone(phone) { return errors.New("invalid phone") }
    // update user
}

// After: Validate once in parameter object
type UserCredentials struct {
    Email    string
    Password string
    Phone    string
}

func (uc UserCredentials) Validate() error {
    if !isValidEmail(uc.Email) {
        return errors.New("invalid email")
    }
    if len(uc.Password) < 8 {
        return errors.New("password too short")
    }
    if !isValidPhone(uc.Phone) {
        return errors.New("invalid phone")
    }
    return nil
}

func RegisterUser(creds UserCredentials) error {
    if err := creds.Validate(); err != nil {
        return err
    }
    // register user
}

func UpdateUser(creds UserCredentials) error {
    if err := creds.Validate(); err != nil {
        return err
    }
    // update user
}
\`\`\`

**5. Named Parameters Effect**
Go doesn't have named parameters, but structs simulate them:

\`\`\`go
// Before: Position-based, error-prone
func CreateProduct(name string, price float64, stock int, category string, featured bool) Product {
    // Easy to mix up: CreateProduct("Widget", 10, 99.99, "Tools", true)
    // Wait, is 10 the stock or price?
}

// After: Named fields like named parameters
type ProductInfo struct {
    Name     string
    Price    float64
    Stock    int
    Category string
    Featured bool
}

func CreateProduct(info ProductInfo) Product {
    // Crystal clear:
    CreateProduct(ProductInfo{
        Name:     "Widget",
        Price:    99.99,
        Stock:    10,
        Category: "Tools",
        Featured: true,
    })
    // No confusion possible
}
\`\`\`

**6. Default Values**
Struct fields have automatic zero values:

\`\`\`go
// Before: Need overloads or optional parameters (Go doesn't support)
func Connect(host string, port int, timeout time.Duration, retries int) error {
    // Must provide all parameters even if you want defaults
}

// After: Omit optional fields for defaults
type ConnectionConfig struct {
    Host    string
    Port    int            // default 0, can check and use 80
    Timeout time.Duration  // default 0, can check and use 30*time.Second
    Retries int            // default 0, can check and use 3
}

func Connect(config ConnectionConfig) error {
    port := config.Port
    if port == 0 {
        port = 80  // default
    }

    timeout := config.Timeout
    if timeout == 0 {
        timeout = 30 * time.Second  // default
    }

    // Use with defaults:
    Connect(ConnectionConfig{Host: "example.com"})
    // Only specify what differs:
    Connect(ConnectionConfig{Host: "example.com", Timeout: 5 * time.Second})
}
\`\`\`

**Real-World Example - HTTP Handler:**

\`\`\`go
// Before: Long parameter list
func HandleRequest(method, path, body string, headers map[string]string,
                   queryParams map[string]string, cookies []*Cookie,
                   timeout time.Duration, maxRetries int) Response {
    // 8 parameters - hard to manage
}

// After: Clean parameter object
type HTTPRequest struct {
    Method      string
    Path        string
    Body        string
    Headers     map[string]string
    QueryParams map[string]string
    Cookies     []*Cookie
    Timeout     time.Duration
    MaxRetries  int
}

func HandleRequest(req HTTPRequest) Response {
    // Single parameter - much cleaner
}

// Easy to add middleware that modifies request:
func LoggingMiddleware(req HTTPRequest) HTTPRequest {
    log.Printf("Request: %s %s", req.Method, req.Path)
    return req
}

func AuthMiddleware(req HTTPRequest) (HTTPRequest, error) {
    if req.Headers["Authorization"] == "" {
        return req, errors.New("unauthorized")
    }
    return req, nil
}
\`\`\`

**When NOT to Introduce Parameter Object:**
- Only 1-2 related parameters
- Parameters are unrelated
- Object would have no meaningful name
- Parameters rarely used together
- Simple utility functions`,
	order: 10,
	testCode: `package refactoring

import (
	"testing"
)

// Test1: UserData struct has all required fields
func Test1(t *testing.T) {
	data := UserData{
		FirstName: "John",
		LastName:  "Doe",
		Email:     "john@example.com",
		Phone:     "123-456-7890",
		Address:   "123 Main St",
		City:      "NYC",
	}
	if data.FirstName != "John" || data.LastName != "Doe" {
		t.Error("UserData fields not set correctly")
	}
}

// Test2: CreateUser returns User with correct fields
func Test2(t *testing.T) {
	data := UserData{
		FirstName: "Jane",
		LastName:  "Smith",
		Email:     "jane@example.com",
		Phone:     "555-1234",
		Address:   "456 Oak Ave",
		City:      "LA",
	}
	user := CreateUser(data)
	if user.FirstName != "Jane" || user.Email != "jane@example.com" {
		t.Error("CreateUser did not set fields correctly")
	}
}

// Test3: CreateUser assigns incrementing ID
func Test3(t *testing.T) {
	data1 := UserData{FirstName: "A"}
	data2 := UserData{FirstName: "B"}
	user1 := CreateUser(data1)
	user2 := CreateUser(data2)
	if user2.ID <= user1.ID {
		t.Error("IDs should be incrementing")
	}
}

// Test4: SaveUser runs without panic
func Test4(t *testing.T) {
	data := UserData{
		FirstName: "Test",
		LastName:  "User",
		Email:     "test@test.com",
		Phone:     "000-000-0000",
		Address:   "Test St",
		City:      "Test City",
	}
	SaveUser(data) // Should not panic
}

// Test5: User struct has ID field
func Test5(t *testing.T) {
	data := UserData{FirstName: "X"}
	user := CreateUser(data)
	if user.ID == 0 {
		t.Error("User should have non-zero ID")
	}
}

// Test6: CreateUser copies all fields from UserData
func Test6(t *testing.T) {
	data := UserData{
		FirstName: "F",
		LastName:  "L",
		Email:     "e@e.com",
		Phone:     "111",
		Address:   "A",
		City:      "C",
	}
	user := CreateUser(data)
	if user.Phone != "111" || user.Address != "A" || user.City != "C" {
		t.Error("CreateUser did not copy all fields")
	}
}

// Test7: Multiple CreateUser calls work independently
func Test7(t *testing.T) {
	data1 := UserData{FirstName: "One"}
	data2 := UserData{FirstName: "Two"}
	user1 := CreateUser(data1)
	user2 := CreateUser(data2)
	if user1.FirstName == user2.FirstName {
		t.Error("Users should have different first names")
	}
}

// Test8: UserData with empty fields
func Test8(t *testing.T) {
	data := UserData{}
	user := CreateUser(data)
	if user.FirstName != "" || user.Email != "" {
		t.Error("Empty UserData should create User with empty fields")
	}
}

// Test9: SaveUser handles all fields
func Test9(t *testing.T) {
	data := UserData{
		FirstName: "Full",
		LastName:  "Name",
		Email:     "full@name.com",
		Phone:     "999",
		Address:   "Full Address",
		City:      "Full City",
	}
	SaveUser(data) // Verify it runs with all fields
}

// Test10: User has same field structure as UserData (minus ID)
func Test10(t *testing.T) {
	data := UserData{
		FirstName: "Same",
		LastName:  "Fields",
		Email:     "same@fields.com",
		Phone:     "123",
		Address:   "Addr",
		City:      "City",
	}
	user := CreateUser(data)
	if user.FirstName != data.FirstName || user.LastName != data.LastName ||
		user.Email != data.Email || user.Phone != data.Phone ||
		user.Address != data.Address || user.City != data.City {
		t.Error("User fields should match UserData fields")
	}
}
`,
	translations: {
		ru: {
			title: 'Introduce Parameter Object - Длинный список параметров',
			description: `Замените длинный список параметров одним объектом-параметром, который группирует связанные данные вместе.

**Вы выполните рефакторинг:**

1. **CreateUser()** - Имеет 6 параметров (слишком много!)
2. Создать **UserData struct** для группировки полей пользователя
3. Обновить **CreateUser** для принятия параметра UserData
4. Обновить **SaveUser** для использования той же структуры`,
			hint1: `Создайте структуру UserData с шестью строковыми полями, соответствующими именам параметров: FirstName, LastName, Email, Phone, Address, City.`,
			hint2: `Измените обе сигнатуры функций, чтобы принимать 'data UserData' вместо шести строковых параметров. Обновите тела функций для доступа к полям через 'data.FirstName', 'data.LastName' и т.д.`,
			whyItMatters: `Объекты-параметры уменьшают сложность, улучшают читаемость и делают код более поддерживаемым и расширяемым.`
		},
		uz: {
			title: 'Introduce Parameter Object - Uzun parametrlar ro\'yxati',
			description: `Uzun parametrlar ro'yxatini bog'liq ma'lumotlarni birgalikda guruhlash uchun bitta parametr ob'ekti bilan almashtiring.

**Siz refaktoring qilasiz:**

1. **CreateUser()** - 6 ta parametrga ega (juda ko'p!)
2. Yaratish **UserData struct** foydalanuvchi maydonlarini guruhlash uchun
3. **CreateUser** ni UserData parametrini qabul qilishi uchun yangilash
4. **SaveUser** ni bir xil strukturadan foydalanishi uchun yangilash`,
			hint1: `UserData strukturasini parametr nomlariga mos keladigan oltita string maydoni bilan yarating: FirstName, LastName, Email, Phone, Address, City.`,
			hint2: `Ikkala funksiya imzosini oltita string parametr o'rniga 'data UserData' qabul qilish uchun o'zgartiring. Funksiya tanalarini 'data.FirstName', 'data.LastName' va hokazo orqali maydonlarga kirish uchun yangilang.`,
			whyItMatters: `Parametr ob'ektlari murakkablikni kamaytiradi, o'qilishni yaxshilaydi va kodni yanada qo'llab-quvvatlanadigan va kengaytiriladigan qiladi.`
		}
	}
};

export default task;
