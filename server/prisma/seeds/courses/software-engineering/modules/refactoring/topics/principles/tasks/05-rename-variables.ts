import { Task } from '../../../../../../../types';

export const task: Task = {
	slug: 'se-refactoring-rename-variables',
	title: 'Rename - Unclear Variable Names',
	difficulty: 'easy',
	tags: ['refactoring', 'rename', 'clean-code', 'go'],
	estimatedTime: '15m',
	isPremium: false,
	youtubeUrl: '',
	description: `Rename unclear variable names to reveal their purpose and improve code readability.

**You will refactor:**

1. **ProcessData()** - Contains cryptic variable names
2. Rename **d** to **data** - Clear input parameter name
3. Rename **r** to **validRecords** - Filtered valid records
4. Rename **t** to **totalAmount** - Sum of all amounts

**Key Concepts:**
- **Reveal Intent**: Names should explain what, not how
- **Avoid Abbreviations**: Use full words unless universally known
- **Be Specific**: Precise names over generic ones
- **Consistent Naming**: Follow team conventions

**Before Refactoring:**

\`\`\`go
func ProcessData(d []Record) float64 {
    r := []Record{}
    t := 0.0
    // What are d, r, and t?
}
\`\`\`

**After Refactoring:**

\`\`\`go
func ProcessData(data []Record) float64 {
    validRecords := []Record{}
    totalAmount := 0.0
    // Clear and self-documenting
}
\`\`\`

**When to Rename:**
- Single letter variables (except i, j, k in loops)
- Abbreviations that aren't obvious
- Generic names (data, temp, value)
- Misleading names
- Names that don't match current use

**Constraints:**
- Keep original function name and logic
- Rename exactly 3 variables
- Use descriptive, meaningful names
- Follow Go naming conventions (camelCase)`,
	initialCode: `package refactoring

type Record struct {
	ID     int
	Amount float64
	Valid  bool
}

func ProcessData(d []Record) float64 {
	for _, rec := range d {
		if rec.Valid {
		}
	}

	for _, rec := range r {
	}

	return t
}`,
	solutionCode: `package refactoring

type Record struct {
	ID     int
	Amount float64
	Valid  bool
}

func ProcessData(data []Record) float64 {
	// Filter to keep only valid records
	validRecords := []Record{}
	for _, rec := range data {
		if rec.Valid {
			validRecords = append(validRecords, rec)
		}
	}

	// Calculate sum of all valid record amounts
	totalAmount := 0.0
	for _, rec := range validRecords {
		totalAmount += rec.Amount
	}

	return totalAmount
}`,
	hint1: `Change the function parameter name from 'd' to 'data', and update its usage in the first loop.`,
	hint2: `Rename 'r' to 'validRecords' (both declaration and usage), and 't' to 'totalAmount' (both declaration and usage).`,
	whyItMatters: `Good naming is one of the hardest and most important aspects of programming. Names are how we communicate intent.

**Why Renaming Variables Matters:**

**1. Code as Documentation**
Names explain purpose without comments:

\`\`\`go
// Before: Need comments to understand
func calc(x, y int) int {
    z := x * y  // calculate area
    return z
}

// After: Self-explanatory
func calculateRectangleArea(width, height int) int {
    area := width * height
    return area
}
\`\`\`

**2. Prevent Misunderstandings**
Clear names prevent bugs:

\`\`\`go
// Before: What does 'd' represent? Days? Distance? Data?
func processOrder(d int) {
    if d > 5 {
        applyDiscount()  // Discount after 5... what?
    }
}

// After: Crystal clear
func processOrder(daysSinceLastOrder int) {
    const loyaltyThresholdDays = 5
    if daysSinceLastOrder > loyaltyThresholdDays {
        applyDiscount()  // Ah, loyalty discount!
    }
}
\`\`\`

**3. Reduce Cognitive Load**
Brain doesn't have to remember what variables mean:

\`\`\`go
// Before: Must mentally track what each variable represents
func calculatePrice(q, p, d, t float64) float64 {
    s := q * p
    sd := s * (1 - d)
    return sd * (1 + t)
}

// After: No mental mapping needed
func calculatePrice(quantity, pricePerUnit, discountRate, taxRate float64) float64 {
    subtotal := quantity * pricePerUnit
    discountedPrice := subtotal * (1 - discountRate)
    finalPrice := discountedPrice * (1 + taxRate)
    return finalPrice
}
\`\`\`

**4. Enable Refactoring**
Good names make code restructuring safer:

\`\`\`go
// Before: If you need to change logic, what is 'tmp'?
func process() {
    tmp := getData()
    // 50 lines later...
    save(tmp)  // tmp contains... what again?
}

// After: Clear even after many lines
func process() {
    userProfile := getUserProfile()
    // 50 lines of processing...
    saveUserProfile(userProfile)  // Obviously saving profile
}
\`\`\`

**5. Code Reviews Are Faster**
Reviewers spend time on logic, not deciphering names:

\`\`\`go
// Before: Reviewer must ask questions
func handleRequest(r Request) {
    x := r.getData()
    y := process(x)
    z := validate(y)
    if !z {
        return errors.New("err")  // What error?
    }
}

// After: Self-reviewing
func handleRequest(request Request) {
    userData := request.getUserData()
    sanitizedData := sanitizeUserInput(userData)
    isValid := validateUserData(sanitizedData)
    if !isValid {
        return errors.New("invalid user data")
    }
}
\`\`\`

**Real-World Example - Database Operations:**

\`\`\`go
// Before: Cryptic database code
func getU(id int) (*User, error) {
    q := "SELECT * FROM users WHERE id = ?"
    r, err := db.Query(q, id)
    if err != nil {
        return nil, err
    }
    defer r.Close()

    u := &User{}
    if r.Next() {
        r.Scan(&u.ID, &u.Name, &u.Email)
    }
    return u, nil
}

// After: Clear intent at every step
func getUserByID(userID int) (*User, error) {
    query := "SELECT * FROM users WHERE id = ?"
    rows, err := db.Query(query, userID)
    if err != nil {
        return nil, err
    }
    defer rows.Close()

    user := &User{}
    if rows.Next() {
        rows.Scan(&user.ID, &user.Name, &user.Email)
    }
    return user, nil
}
\`\`\`

**Naming Conventions:**

**Variables:**
- Use nouns: \`userAccount\`, \`orderTotal\`, \`emailAddress\`
- Be specific: \`customerEmail\` not \`email\`
- Avoid generic: \`data\`, \`info\`, \`item\`, \`thing\`, \`object\`

**Booleans:**
- Use predicates: \`isValid\`, \`hasPermission\`, \`canEdit\`
- Affirmative: \`isEnabled\` not \`isNotDisabled\`

**Collections:**
- Use plurals: \`users\`, \`orders\`, \`products\`
- Or descriptive: \`userList\`, \`activeOrders\`, \`pendingRequests\`

**Loop Indices:**
- Short names OK for small loops: \`i\`, \`j\`, \`k\`
- Descriptive for complex loops: \`userIndex\`, \`rowNumber\`

**Acronyms:**
- Known acronyms OK: \`ID\`, \`URL\`, \`HTTP\`, \`API\`
- In Go: \`userID\` not \`userId\`, \`httpClient\` not \`HTTPClient\`

**Examples of Good Naming:**

\`\`\`go
// Time durations
requestTimeout := 30 * time.Second
retryDelay := 5 * time.Second
sessionExpiration := 24 * time.Hour

// Counts and indices
activeUserCount := 0
currentPageNumber := 1
maxRetryAttempts := 3

// State flags
isAuthenticated := true
hasAdminRole := false
canModifySettings := user.IsAdmin

// Computed values
discountedPrice := originalPrice * 0.9
averageRating := totalRating / ratingCount
remainingBalance := totalBalance - withdrawalAmount
\`\`\`

**Names to Avoid:**
- Single letters (except loop indices): \`a\`, \`b\`, \`x\`, \`y\`
- Unclear abbreviations: \`usr\`, \`ord\`, \`mgr\`, \`proc\`
- Type in name: \`userString\`, \`amountInt\` (Go is typed)
- Numbered variables: \`user1\`, \`user2\` (use array/slice)
- Noise words: \`theUser\`, \`myVariable\`, \`dataValue\`

**The Two-Week Rule:**
If you can't understand your variable name after two weeks away from the code, it's a bad name. Rename it!`,
	order: 4,
	testCode: `package refactoring

import (
	"testing"
)

// Test1: ProcessData with all valid records
func Test1(t *testing.T) {
	data := []Record{
		{ID: 1, Amount: 10.0, Valid: true},
		{ID: 2, Amount: 20.0, Valid: true},
	}
	result := ProcessData(data)
	if result != 30.0 {
		t.Errorf("expected 30.0, got %f", result)
	}
}

// Test2: ProcessData with no valid records
func Test2(t *testing.T) {
	data := []Record{
		{ID: 1, Amount: 10.0, Valid: false},
		{ID: 2, Amount: 20.0, Valid: false},
	}
	result := ProcessData(data)
	if result != 0.0 {
		t.Errorf("expected 0.0, got %f", result)
	}
}

// Test3: ProcessData with mixed valid/invalid
func Test3(t *testing.T) {
	data := []Record{
		{ID: 1, Amount: 10.0, Valid: true},
		{ID: 2, Amount: 20.0, Valid: false},
		{ID: 3, Amount: 30.0, Valid: true},
	}
	result := ProcessData(data)
	if result != 40.0 {
		t.Errorf("expected 40.0, got %f", result)
	}
}

// Test4: ProcessData with empty slice
func Test4(t *testing.T) {
	result := ProcessData([]Record{})
	if result != 0.0 {
		t.Errorf("expected 0.0, got %f", result)
	}
}

// Test5: ProcessData with single valid record
func Test5(t *testing.T) {
	data := []Record{{ID: 1, Amount: 99.99, Valid: true}}
	result := ProcessData(data)
	if result != 99.99 {
		t.Errorf("expected 99.99, got %f", result)
	}
}

// Test6: ProcessData with negative amounts (valid)
func Test6(t *testing.T) {
	data := []Record{
		{ID: 1, Amount: -10.0, Valid: true},
		{ID: 2, Amount: 30.0, Valid: true},
	}
	result := ProcessData(data)
	if result != 20.0 {
		t.Errorf("expected 20.0, got %f", result)
	}
}

// Test7: ProcessData with zero amounts
func Test7(t *testing.T) {
	data := []Record{
		{ID: 1, Amount: 0.0, Valid: true},
		{ID: 2, Amount: 0.0, Valid: true},
	}
	result := ProcessData(data)
	if result != 0.0 {
		t.Errorf("expected 0.0, got %f", result)
	}
}

// Test8: Record struct fields work correctly
func Test8(t *testing.T) {
	r := Record{ID: 5, Amount: 15.5, Valid: true}
	if r.ID != 5 || r.Amount != 15.5 || !r.Valid {
		t.Error("Record fields not set correctly")
	}
}

// Test9: ProcessData with many records
func Test9(t *testing.T) {
	data := make([]Record, 100)
	for i := 0; i < 100; i++ {
		data[i] = Record{ID: i, Amount: 1.0, Valid: true}
	}
	result := ProcessData(data)
	if result != 100.0 {
		t.Errorf("expected 100.0, got %f", result)
	}
}

// Test10: ProcessData with decimal amounts
func Test10(t *testing.T) {
	data := []Record{
		{ID: 1, Amount: 10.25, Valid: true},
		{ID: 2, Amount: 20.75, Valid: true},
	}
	result := ProcessData(data)
	if result != 31.0 {
		t.Errorf("expected 31.0, got %f", result)
	}
}
`,
	translations: {
		ru: {
			title: 'Rename - Неясные имена переменных',
			description: `Переименуйте неясные имена переменных, чтобы раскрыть их назначение и улучшить читаемость кода.

**Вы выполните рефакторинг:**

1. **ProcessData()** - Содержит криптичные имена переменных
2. Переименовать **d** в **data** - Ясное имя входного параметра
3. Переименовать **r** в **validRecords** - Отфильтрованные валидные записи
4. Переименовать **t** в **totalAmount** - Сумма всех значений`,
			hint1: `Измените имя параметра функции с 'd' на 'data', и обновите его использование в первом цикле.`,
			hint2: `Переименуйте 'r' в 'validRecords' (и объявление и использование), и 't' в 'totalAmount' (и объявление и использование).`,
			whyItMatters: `Хорошее именование — один из самых сложных и важных аспектов программирования. Имена — это способ передачи намерений.`
		},
		uz: {
			title: 'Rename - Noaniq o\'zgaruvchi nomlari',
			description: `Noaniq o'zgaruvchi nomlarini ularning maqsadini ochib beradigan va kod o'qilishini yaxshilaydigan nomlarga o'zgartiring.

**Siz refaktoring qilasiz:**

1. **ProcessData()** - Kriptik o'zgaruvchi nomlarini o'z ichiga oladi
2. Qayta nomlash **d** dan **data** ga - Aniq kirish parametri nomi
3. Qayta nomlash **r** dan **validRecords** ga - Filtrlangan yaroqli yozuvlar
4. Qayta nomlash **t** dan **totalAmount** ga - Barcha qiymatlar yig'indisi`,
			hint1: `Funksiya parametri nomini 'd' dan 'data' ga o'zgartiring va birinchi tsikldagi ishlatilishini yangilang.`,
			hint2: `'r' ni 'validRecords' ga (ham e'lon ham ishlatish), va 't' ni 'totalAmount' ga (ham e'lon ham ishlatish) qayta nomlang.`,
			whyItMatters: `Yaxshi nomlash dasturlashning eng qiyin va muhim jihatlaridan biridir. Nomlar niyatni etkazish usuli.`
		}
	}
};

export default task;
