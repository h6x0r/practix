import { Task } from '../../../../../../../types';

export const task: Task = {
	slug: 'se-refactoring-param-object-clumps',
	title: 'Introduce Parameter Object - Data Clumps',
	difficulty: 'medium',
	tags: ['refactoring', 'parameter-object', 'clean-code', 'go'],
	estimatedTime: '25m',
	isPremium: false,
	youtubeUrl: '',
	description: `Eliminate data clumps by grouping parameters that always appear together into a cohesive object.

**You will refactor:**

1. **Multiple functions** with recurring coordinate parameters (x, y)
2. Create **Point struct** to represent coordinates
3. Update **CalculateDistance, IsInBounds, DrawLine** to use Point
4. Remove duplicate x, y parameter pairs

**Key Concepts:**
- **Data Clumps**: Same parameters appearing together repeatedly
- **Cohesion**: Related data belongs in one structure
- **DRY Principle**: Don't repeat parameter groups
- **Domain Modeling**: Point represents a concept

**Before Refactoring:**

\`\`\`go
func CalculateDistance(x1, y1, x2, y2 float64) float64 { }
func IsInBounds(x, y, minX, minY, maxX, maxY float64) bool { }
func DrawLine(startX, startY, endX, endY float64) { }
// x, y parameters clump together everywhere!
\`\`\`

**After Refactoring:**

\`\`\`go
type Point struct { X, Y float64 }
type Bounds struct { Min, Max Point }

func CalculateDistance(p1, p2 Point) float64 { }
func IsInBounds(p Point, bounds Bounds) bool { }
func DrawLine(start, end Point) { }
\`\`\`

**When to Introduce Parameter Object:**
- Same parameters in multiple functions
- Parameters represent a cohesive concept
- Parameters are always used together
- Would benefit from methods on the object
- Parameters describe domain entity

**Constraints:**
- Create Point struct with X, Y fields
- Create Bounds struct with Min, Max Point fields
- Update all three functions to use these types
- Maintain exact same calculation logic`,
	initialCode: `package refactoring

import "math"

func CalculateDistance(x1, y1, x2, y2 float64) float64 {
	return math.Sqrt(dx*dx + dy*dy)
}

func IsInBounds(x, y, minX, minY, maxX, maxY float64) bool {
	return x >= minX && x <= maxX && y >= minY && y <= maxY
}

func DrawLine(startX, startY, endX, endY float64) {
}`,
	solutionCode: `package refactoring

import "math"

// Point represents a 2D coordinate
type Point struct {
	X float64
	Y float64
}

// Bounds represents a rectangular boundary area
type Bounds struct {
	Min Point	// minimum corner (top-left)
	Max Point	// maximum corner (bottom-right)
}

// CalculateDistance calculates Euclidean distance between two points
func CalculateDistance(p1, p2 Point) float64 {
	dx := p2.X - p1.X	// difference in x coordinates
	dy := p2.Y - p1.Y	// difference in y coordinates
	return math.Sqrt(dx*dx + dy*dy)	// Pythagorean theorem
}

// IsInBounds checks if point is within rectangular bounds
func IsInBounds(p Point, bounds Bounds) bool {
	return p.X >= bounds.Min.X && p.X <= bounds.Max.X &&
		p.Y >= bounds.Min.Y && p.Y <= bounds.Max.Y
}

// DrawLine draws a line between two points
func DrawLine(start, end Point) {
	// Simplified line drawing
	distance := CalculateDistance(start, end)	// clean call with Point objects
	println("Drawing line of length:", distance)
}`,
	hint1: `Create two structs: Point with X, Y float64 fields, and Bounds with Min, Max Point fields.`,
	hint2: `Update function signatures to use Point and Bounds types. Replace parameter references: x1 becomes p1.X, y1 becomes p1.Y, minX becomes bounds.Min.X, etc.`,
	whyItMatters: `Eliminating data clumps improves code organization and reveals domain concepts that were hidden in parameter lists.

**Why Eliminating Data Clumps Matters:**

**1. Reveals Domain Concepts**
Parameter clumps often represent domain objects:

\`\`\`go
// Before: Hidden domain concept
func CreateReservation(startYear, startMonth, startDay, endYear, endMonth, endDay int) {
    // 6 parameters - but it's really two dates!
}

func IsOverlapping(start1Y, start1M, start1D, end1Y, end1M, end1D,
                   start2Y, start2M, start2D, end2Y, end2M, end2D int) bool {
    // 12 parameters - impossible to read!
}

// After: Domain concept is explicit
type Date struct {
    Year  int
    Month int
    Day   int
}

type DateRange struct {
    Start Date
    End   Date
}

func CreateReservation(period DateRange) {
    // Clear: one reservation period
}

func IsOverlapping(period1, period2 DateRange) bool {
    // Clear: checking if two periods overlap
}
\`\`\`

**2. Enable Object Behavior**
Parameter objects can have methods:

\`\`\`go
// Before: Utility functions for operations
func RectangleArea(width, height float64) float64 {
    return width * height
}

func RectanglePerimeter(width, height float64) float64 {
    return 2 * (width + height)
}

func IsSquare(width, height float64) bool {
    return width == height
}

// width, height clump everywhere

// After: Rectangle knows its own operations
type Rectangle struct {
    Width  float64
    Height float64
}

func (r Rectangle) Area() float64 {
    return r.Width * r.Height
}

func (r Rectangle) Perimeter() float64 {
    return 2 * (r.Width + r.Height)
}

func (r Rectangle) IsSquare() bool {
    return r.Width == r.Height
}

// Usage is cleaner and more intuitive
rect := Rectangle{Width: 10, Height: 5}
fmt.Println(rect.Area())       // 50
fmt.Println(rect.Perimeter())  // 30
fmt.Println(rect.IsSquare())   // false
\`\`\`

**3. Reduce Parameter Count**
Functions become simpler:

\`\`\`go
// Before: Too many parameters
func FormatAddress(street, city, state, zip, country string) string {
    return fmt.Sprintf("%s, %s, %s %s, %s", street, city, state, zip, country)
}

func ValidateAddress(street, city, state, zip, country string) error {
    // same 5 parameters again
}

func SaveAddress(userID int, street, city, state, zip, country string) error {
    // same 5 parameters + userID
}

func CompareAddresses(street1, city1, state1, zip1, country1,
                      street2, city2, state2, zip2, country2 string) bool {
    // 10 parameters - nightmare!
}

// After: Clean parameter objects
type Address struct {
    Street  string
    City    string
    State   string
    Zip     string
    Country string
}

func (a Address) Format() string {
    return fmt.Sprintf("%s, %s, %s %s, %s", a.Street, a.City, a.State, a.Zip, a.Country)
}

func (a Address) Validate() error {
    // validation logic
}

func SaveAddress(userID int, address Address) error {
    // 2 parameters instead of 6
}

func (a Address) Equals(other Address) bool {
    return a == other  // simple comparison
}
\`\`\`

**4. Centralized Validation**
Validate once in constructor or method:

\`\`\`go
// Before: Duplicate validation
func ProcessPayment(cardNumber, cvv, expMonth, expYear string) error {
    if len(cardNumber) != 16 { return errors.New("invalid card") }
    if len(cvv) != 3 { return errors.New("invalid cvv") }
    // process payment
}

func SaveCard(userID int, cardNumber, cvv, expMonth, expYear string) error {
    if len(cardNumber) != 16 { return errors.New("invalid card") }
    if len(cvv) != 3 { return errors.New("invalid cvv") }
    // save card
}

// After: Single validation point
type CreditCard struct {
    Number   string
    CVV      string
    ExpMonth string
    ExpYear  string
}

func NewCreditCard(number, cvv, expMonth, expYear string) (*CreditCard, error) {
    if len(number) != 16 {
        return nil, errors.New("invalid card number")
    }
    if len(cvv) != 3 {
        return nil, errors.New("invalid cvv")
    }
    return &CreditCard{
        Number:   number,
        CVV:      cvv,
        ExpMonth: expMonth,
        ExpYear:  expYear,
    }, nil
}

func ProcessPayment(card CreditCard) error {
    // card is already validated
}

func SaveCard(userID int, card CreditCard) error {
    // card is already validated
}
\`\`\`

**5. Type Safety**
Prevent mixing up parameter order:

\`\`\`go
// Before: Easy to mix up parameters
func Transfer(fromAccountID, toAccountID, amount string) error {
    // Called with: Transfer("123", "456", "100.00")
    // Or was it: Transfer("456", "123", "100.00")?
    // Disaster if mixed up!
}

// After: Type safety prevents mistakes
type AccountID string
type Amount struct {
    Value    float64
    Currency string
}

type Transfer struct {
    From   AccountID
    To     AccountID
    Amount Amount
}

func ProcessTransfer(t Transfer) error {
    // Called with:
    ProcessTransfer(Transfer{
        From:   AccountID("123"),
        To:     AccountID("456"),
        Amount: Amount{Value: 100.00, Currency: "USD"},
    })
    // Impossible to mix up From and To - they're named!
}
\`\`\`

**6. Testability**
Create reusable test fixtures:

\`\`\`go
// Before: Must setup parameters in every test
func TestCalculateShipping(t *testing.T) {
    cost := CalculateShipping("USA", "NY", "10001", 5.0)
    assert.Equal(t, 10.0, cost)
}

func TestValidateAddress(t *testing.T) {
    err := ValidateAddress("USA", "NY", "10001")
    assert.NoError(t, err)
}

// After: Reusable test fixtures
func TestCalculateShipping(t *testing.T) {
    nyAddress := Address{Country: "USA", State: "NY", Zip: "10001"}
    cost := CalculateShipping(nyAddress, 5.0)
    assert.Equal(t, 10.0, cost)
}

func TestValidateAddress(t *testing.T) {
    nyAddress := Address{Country: "USA", State: "NY", Zip: "10001"}
    err := nyAddress.Validate()
    assert.NoError(t, err)
}

// Can even create test helpers
func TestAddress() Address {
    return Address{Country: "USA", State: "NY", Zip: "10001"}
}
\`\`\`

**Real-World Example - Color System:**

\`\`\`go
// Before: RGB values as separate parameters everywhere
func CreatePixel(r, g, b, a int) Pixel {
    return Pixel{Red: r, Green: g, Blue: b, Alpha: a}
}

func MixColors(r1, g1, b1, r2, g2, b2 int) (int, int, int) {
    return (r1+r2)/2, (g1+g2)/2, (b1+b2)/2
}

func ToHex(r, g, b int) string {
    return fmt.Sprintf("#%02x%02x%02x", r, g, b)
}

func Lighten(r, g, b int, percent float64) (int, int, int) {
    // lighten color
}

// After: Color as first-class concept
type Color struct {
    R, G, B, A int
}

func NewColor(r, g, b, a int) Color {
    return Color{R: r, G: g, B: b, A: a}
}

func (c Color) Mix(other Color) Color {
    return Color{
        R: (c.R + other.R) / 2,
        G: (c.G + other.G) / 2,
        B: (c.B + other.B) / 2,
        A: (c.A + other.A) / 2,
    }
}

func (c Color) ToHex() string {
    return fmt.Sprintf("#%02x%02x%02x", c.R, c.G, c.B)
}

func (c Color) Lighten(percent float64) Color {
    // lighten and return new color
}

// Much more intuitive API
red := NewColor(255, 0, 0, 255)
blue := NewColor(0, 0, 255, 255)
purple := red.Mix(blue)
fmt.Println(purple.ToHex())
\`\`\`

**Identifying Data Clumps:**
- Same 2-3+ parameters in multiple functions
- Parameters have cohesive meaning (coordinates, date ranges, etc.)
- Parameters are always used together
- Changing one parameter often requires changing others
- Parameters represent a concept that could be named

**Benefits:**
- Clearer code structure
- Fewer parameters
- Domain concepts become explicit
- Methods can be added to objects
- Better type safety
- Easier testing with fixtures
- Single point for validation`,
	order: 11,
	testCode: `package refactoring

import (
	"math"
	"testing"
)

// Test1: Point struct has X and Y fields
func Test1(t *testing.T) {
	p := Point{X: 10.0, Y: 20.0}
	if p.X != 10.0 || p.Y != 20.0 {
		t.Error("Point fields not set correctly")
	}
}

// Test2: Bounds struct has Min and Max Point fields
func Test2(t *testing.T) {
	bounds := Bounds{
		Min: Point{X: 0, Y: 0},
		Max: Point{X: 100, Y: 100},
	}
	if bounds.Min.X != 0 || bounds.Max.Y != 100 {
		t.Error("Bounds fields not set correctly")
	}
}

// Test3: CalculateDistance for horizontal line
func Test3(t *testing.T) {
	p1 := Point{X: 0, Y: 0}
	p2 := Point{X: 10, Y: 0}
	result := CalculateDistance(p1, p2)
	if result != 10.0 {
		t.Errorf("expected 10.0, got %f", result)
	}
}

// Test4: CalculateDistance for vertical line
func Test4(t *testing.T) {
	p1 := Point{X: 0, Y: 0}
	p2 := Point{X: 0, Y: 5}
	result := CalculateDistance(p1, p2)
	if result != 5.0 {
		t.Errorf("expected 5.0, got %f", result)
	}
}

// Test5: CalculateDistance for diagonal (3-4-5 triangle)
func Test5(t *testing.T) {
	p1 := Point{X: 0, Y: 0}
	p2 := Point{X: 3, Y: 4}
	result := CalculateDistance(p1, p2)
	if result != 5.0 {
		t.Errorf("expected 5.0, got %f", result)
	}
}

// Test6: IsInBounds with point inside
func Test6(t *testing.T) {
	p := Point{X: 50, Y: 50}
	bounds := Bounds{Min: Point{X: 0, Y: 0}, Max: Point{X: 100, Y: 100}}
	if !IsInBounds(p, bounds) {
		t.Error("point (50,50) should be in bounds (0,0)-(100,100)")
	}
}

// Test7: IsInBounds with point outside
func Test7(t *testing.T) {
	p := Point{X: 150, Y: 50}
	bounds := Bounds{Min: Point{X: 0, Y: 0}, Max: Point{X: 100, Y: 100}}
	if IsInBounds(p, bounds) {
		t.Error("point (150,50) should be outside bounds (0,0)-(100,100)")
	}
}

// Test8: IsInBounds with point on boundary
func Test8(t *testing.T) {
	p := Point{X: 100, Y: 100}
	bounds := Bounds{Min: Point{X: 0, Y: 0}, Max: Point{X: 100, Y: 100}}
	if !IsInBounds(p, bounds) {
		t.Error("point on boundary should be in bounds")
	}
}

// Test9: DrawLine runs without panic
func Test9(t *testing.T) {
	start := Point{X: 0, Y: 0}
	end := Point{X: 10, Y: 10}
	DrawLine(start, end) // Should not panic
}

// Test10: CalculateDistance with same point returns 0
func Test10(t *testing.T) {
	p := Point{X: 5, Y: 5}
	result := CalculateDistance(p, p)
	if result != 0.0 {
		t.Errorf("expected 0.0, got %f", result)
	}
}

// Helper for float comparison
func almostEqual(a, b, epsilon float64) bool {
	return math.Abs(a-b) < epsilon
}
`,
	translations: {
		ru: {
			title: 'Introduce Parameter Object - Сгустки данных',
			description: `Устраните сгустки данных, группируя параметры, которые всегда появляются вместе, в связанный объект.

**Вы выполните рефакторинг:**

1. **Несколько функций** с повторяющимися параметрами координат (x, y)
2. Создать **Point struct** для представления координат
3. Обновить **CalculateDistance, IsInBounds, DrawLine** для использования Point
4. Удалить дублирующиеся пары параметров x, y`,
			hint1: `Создайте две структуры: Point с полями X, Y float64, и Bounds с полями Min, Max Point.`,
			hint2: `Обновите сигнатуры функций для использования типов Point и Bounds. Замените ссылки на параметры: x1 становится p1.X, y1 становится p1.Y, minX становится bounds.Min.X и т.д.`,
			whyItMatters: `Устранение сгустков данных улучшает организацию кода и раскрывает доменные концепции, которые были скрыты в списках параметров.`
		},
		uz: {
			title: 'Introduce Parameter Object - Ma\'lumot to\'plamlari',
			description: `Har doim birgalikda paydo bo'ladigan parametrlarni birlashtirilgan ob'ektga guruhlash orqali ma'lumot to'plamlarini yo'q qiling.

**Siz refaktoring qilasiz:**

1. **Bir nechta funktsiyalar** takrorlanuvchi koordinata parametrlari (x, y) bilan
2. Yaratish **Point struct** koordinatalarni ifodalash uchun
3. **CalculateDistance, IsInBounds, DrawLine** ni Point ishlatish uchun yangilash
4. Takrorlanuvchi x, y parametr juftlarini o'chirish`,
			hint1: `Ikkita struktura yarating: Point X, Y float64 maydonlari bilan, va Bounds Min, Max Point maydonlari bilan.`,
			hint2: `Funksiya imzolarini Point va Bounds turlaridan foydalanish uchun yangilang. Parametr havolalarini almashtiring: x1 p1.X ga aylanadi, y1 p1.Y ga aylanadi, minX bounds.Min.X ga aylanadi va hokazo.`,
			whyItMatters: `Ma'lumot to'plamlarini yo'q qilish kod tashkilotini yaxshilaydi va parametr ro'yxatlarida yashiringan domen tushunchalarini ochib beradi.`
		}
	}
};

export default task;
