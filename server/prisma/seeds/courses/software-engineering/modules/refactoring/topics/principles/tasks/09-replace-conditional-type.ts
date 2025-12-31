import { Task } from '../../../../../../../types';

export const task: Task = {
	slug: 'se-refactoring-replace-conditional-type',
	title: 'Replace Conditional with Polymorphism - Type Checking',
	difficulty: 'medium',
	tags: ['refactoring', 'polymorphism', 'clean-code', 'go'],
	estimatedTime: '30m',
	isPremium: false,
	youtubeUrl: '',
	description: `Replace type-checking conditionals with polymorphism using interfaces for cleaner, extensible code.

**You will refactor:**

1. **CalculatePrice()** - Has type-checking switch statement
2. Create **Vehicle interface** with GetPrice() method
3. Implement interface on **Car, Truck, Motorcycle** types
4. Remove switch statement, use polymorphic call

**Key Concepts:**
- **Polymorphism**: Different types, same interface
- **Open/Closed Principle**: Open for extension, closed for modification
- **Eliminate Type Checks**: Let types handle their own behavior
- **Interface Segregation**: Small, focused interfaces

**Before Refactoring:**

\`\`\`go
func CalculatePrice(vehicleType string, days int) float64 {
    switch vehicleType {
    case "car": return float64(days) * 50
    case "truck": return float64(days) * 100
    case "motorcycle": return float64(days) * 30
    }
}
\`\`\`

**After Refactoring:**

\`\`\`go
type Vehicle interface {
    GetPrice(days int) float64
}

func CalculatePrice(vehicle Vehicle, days int) float64 {
    return vehicle.GetPrice(days)
}
\`\`\`

**When to Replace with Polymorphism:**
- Switch/if-else on type or category
- Same switch appears in multiple places
- Adding new types requires changing many places
- Behavior varies by type
- Open/Closed Principle violation

**Constraints:**
- Create Vehicle interface with GetPrice(days int) float64
- Implement on Car, Truck, Motorcycle structs
- Remove switch statement from CalculateRentalPrice
- Maintain exact same pricing logic`,
	initialCode: `package refactoring

func CalculateRentalPrice(vehicleType string, days int) float64 {
	switch vehicleType {
		return float64(days) * 50.0
		return float64(days) * 100.0
		return float64(days) * 30.0
		return 0
	}
}`,
	solutionCode: `package refactoring

// Vehicle interface defines behavior all vehicles must implement
type Vehicle interface {
	GetPrice(days int) float64
}

// Car implements Vehicle interface
type Car struct{}

func (c Car) GetPrice(days int) float64 {
	const carDailyRate = 50.0
	return float64(days) * carDailyRate	// $50 per day for cars
}

// Truck implements Vehicle interface
type Truck struct{}

func (t Truck) GetPrice(days int) float64 {
	const truckDailyRate = 100.0
	return float64(days) * truckDailyRate	// $100 per day for trucks
}

// Motorcycle implements Vehicle interface
type Motorcycle struct{}

func (m Motorcycle) GetPrice(days int) float64 {
	const motorcycleDailyRate = 30.0
	return float64(days) * motorcycleDailyRate	// $30 per day for motorcycles
}

// CalculateRentalPrice uses polymorphism - no type checking needed
func CalculateRentalPrice(vehicle Vehicle, days int) float64 {
	return vehicle.GetPrice(days)	// polymorphic call - each type handles its own pricing
}`,
	hint1: `Define the Vehicle interface with one method: GetPrice(days int) float64. Then create three empty structs: Car, Truck, and Motorcycle.`,
	hint2: `Implement GetPrice method on each struct with the appropriate rate (Car: 50, Truck: 100, Motorcycle: 30). Change CalculateRentalPrice parameter from string to Vehicle interface and replace the switch with return vehicle.GetPrice(days).`,
	whyItMatters: `Replacing conditionals with polymorphism makes code more maintainable, extensible, and follows the Open/Closed Principle.

**Why Replace Conditional with Polymorphism Matters:**

**1. Open/Closed Principle**
Add new types without modifying existing code:

\`\`\`go
// Before: Adding new payment type requires modifying existing function
func ProcessPayment(paymentType string, amount float64) error {
    switch paymentType {
    case "credit_card":
        return processCreditCard(amount)
    case "paypal":
        return processPayPal(amount)
    case "bitcoin":  // NEW: must modify this function
        return processBitcoin(amount)
    }
}

// Must also update validation, formatting, and 5 other places!

// After: Adding new payment type is just a new struct
type PaymentMethod interface {
    Process(amount float64) error
}

type CreditCard struct {}
func (cc CreditCard) Process(amount float64) error { /* implementation */ }

type PayPal struct {}
func (pp PayPal) Process(amount float64) error { /* implementation */ }

// NEW: Just add new struct, no existing code changes
type Bitcoin struct {}
func (b Bitcoin) Process(amount float64) error { /* implementation */ }

func ProcessPayment(method PaymentMethod, amount float64) error {
    return method.Process(amount)  // works with any PaymentMethod
}
\`\`\`

**2. Eliminate Duplicate Switches**
Same switch scattered everywhere leads to bugs:

\`\`\`go
// Before: Same switch in multiple places
func CalculateShipping(itemType string, weight float64) float64 {
    switch itemType {
    case "book": return weight * 2.0
    case "electronics": return weight * 5.0
    case "furniture": return weight * 10.0
    }
}

func GetHandlingTime(itemType string) int {
    switch itemType {  // Duplicate switch!
    case "book": return 1
    case "electronics": return 3
    case "furniture": return 7
    }
}

func RequiresInsurance(itemType string) bool {
    switch itemType {  // Another duplicate!
    case "book": return false
    case "electronics": return true
    case "furniture": return true
    }
}

// Bug: If you add "clothing" type, must update 3+ places!

// After: Each type knows its own behavior
type Item interface {
    CalculateShipping(weight float64) float64
    GetHandlingTime() int
    RequiresInsurance() bool
}

type Book struct{}
func (b Book) CalculateShipping(weight float64) float64 { return weight * 2.0 }
func (b Book) GetHandlingTime() int { return 1 }
func (b Book) RequiresInsurance() bool { return false }

type Electronics struct{}
func (e Electronics) CalculateShipping(weight float64) float64 { return weight * 5.0 }
func (e Electronics) GetHandlingTime() int { return 3 }
func (e Electronics) RequiresInsurance() bool { return true }

// Add Clothing: just one new struct, impossible to forget methods
type Clothing struct{}
func (c Clothing) CalculateShipping(weight float64) float64 { return weight * 3.0 }
func (c Clothing) GetHandlingTime() int { return 2 }
func (c Clothing) RequiresInsurance() bool { return false }
\`\`\`

**3. Single Responsibility**
Each type manages its own behavior:

\`\`\`go
// Before: One function knows all discount rules
func ApplyDiscount(customerType string, amount float64) float64 {
    switch customerType {
    case "regular":
        if amount > 100 { return amount * 0.95 }
        return amount
    case "premium":
        if amount > 50 { return amount * 0.90 }
        return amount
    case "vip":
        if amount > 0 { return amount * 0.80 }
        return amount
    }
}

// After: Each customer type knows its own discount logic
type Customer interface {
    ApplyDiscount(amount float64) float64
}

type RegularCustomer struct{}
func (r RegularCustomer) ApplyDiscount(amount float64) float64 {
    if amount > 100 {
        return amount * 0.95  // 5% off over $100
    }
    return amount
}

type PremiumCustomer struct{}
func (p PremiumCustomer) ApplyDiscount(amount float64) float64 {
    if amount > 50 {
        return amount * 0.90  // 10% off over $50
    }
    return amount
}

type VIPCustomer struct{}
func (v VIPCustomer) ApplyDiscount(amount float64) float64 {
    return amount * 0.80  // Always 20% off
}
\`\`\`

**4. Testability**
Test each type independently:

\`\`\`go
// Before: Hard to test specific type logic
func TestDiscounts(t *testing.T) {
    // Must test all types in one function
    assert.Equal(t, 95.0, ApplyDiscount("regular", 100))
    assert.Equal(t, 90.0, ApplyDiscount("premium", 100))
    assert.Equal(t, 80.0, ApplyDiscount("vip", 100))
    // Messy, all mixed together
}

// After: Clean, focused tests
func TestRegularCustomerDiscount(t *testing.T) {
    customer := RegularCustomer{}
    assert.Equal(t, 95.0, customer.ApplyDiscount(100))
    assert.Equal(t, 100.0, customer.ApplyDiscount(50))  // no discount under $100
}

func TestPremiumCustomerDiscount(t *testing.T) {
    customer := PremiumCustomer{}
    assert.Equal(t, 90.0, customer.ApplyDiscount(100))
    assert.Equal(t, 45.0, customer.ApplyDiscount(50))  // discount starts at $50
}

func TestVIPCustomerDiscount(t *testing.T) {
    customer := VIPCustomer{}
    assert.Equal(t, 80.0, customer.ApplyDiscount(100))
    assert.Equal(t, 8.0, customer.ApplyDiscount(10))  // always 20% off
}
\`\`\`

**5. Maintainability**
Changes localized to specific types:

\`\`\`go
// Before: Changing premium discount affects this function
func CalculatePrice(customerType string, items []Item) float64 {
    total := sumItems(items)
    switch customerType {
    case "regular":
        return total
    case "premium":
        return total * 0.85  // Need to change this to 0.90
    case "vip":
        return total * 0.70
    }
}

// Also need to update in GetLoyaltyPoints, ShowDiscountBadge, etc.

// After: Change only PremiumCustomer
type PremiumCustomer struct{}

func (p PremiumCustomer) CalculatePrice(items []Item) float64 {
    total := sumItems(items)
    return total * 0.90  // Change in ONE place
}
// Automatically affects all uses of PremiumCustomer
\`\`\`

**Real-World Example - Notification System:**

\`\`\`go
// Before: Messy conditionals
func SendNotification(notifType, recipient, message string) error {
    switch notifType {
    case "email":
        return smtp.Send(recipient, message)
    case "sms":
        return twillio.SMS(recipient, message)
    case "push":
        return fcm.Push(recipient, message)
    case "slack":
        return slack.Post(recipient, message)
    }
}

// After: Clean polymorphism
type NotificationChannel interface {
    Send(recipient, message string) error
}

type EmailChannel struct{ smtp *SMTPClient }
func (e EmailChannel) Send(recipient, message string) error {
    return e.smtp.Send(recipient, message)
}

type SMSChannel struct{ client *TwillioClient }
func (s SMSChannel) Send(recipient, message string) error {
    return s.client.SMS(recipient, message)
}

type PushChannel struct{ fcm *FCMClient }
func (p PushChannel) Send(recipient, message string) error {
    return p.fcm.Push(recipient, message)
}

type SlackChannel struct{ client *SlackClient }
func (s SlackChannel) Send(recipient, message string) error {
    return s.client.Post(recipient, message)
}

func SendNotification(channel NotificationChannel, recipient, message string) error {
    return channel.Send(recipient, message)
}

// Adding WhatsApp? Just add WhatsAppChannel struct!
\`\`\`

**When NOT to Use Polymorphism:**
- Only 2-3 simple cases that won't change
- Cases are truly mutually exclusive and stable
- Simple lookup/mapping (use map[string]func)
- Performance-critical code (interface calls have small overhead)
- Configuration-based behavior (use strategy pattern with dependency injection)`,
	order: 8,
	testCode: `package refactoring

import (
	"testing"
)

// Test1: Car.GetPrice returns correct rate
func Test1(t *testing.T) {
	car := Car{}
	result := car.GetPrice(1)
	if result != 50.0 {
		t.Errorf("expected 50.0, got %f", result)
	}
}

// Test2: Truck.GetPrice returns correct rate
func Test2(t *testing.T) {
	truck := Truck{}
	result := truck.GetPrice(1)
	if result != 100.0 {
		t.Errorf("expected 100.0, got %f", result)
	}
}

// Test3: Motorcycle.GetPrice returns correct rate
func Test3(t *testing.T) {
	motorcycle := Motorcycle{}
	result := motorcycle.GetPrice(1)
	if result != 30.0 {
		t.Errorf("expected 30.0, got %f", result)
	}
}

// Test4: CalculateRentalPrice with Car for 5 days
func Test4(t *testing.T) {
	car := Car{}
	result := CalculateRentalPrice(car, 5)
	if result != 250.0 {
		t.Errorf("expected 250.0, got %f", result)
	}
}

// Test5: CalculateRentalPrice with Truck for 3 days
func Test5(t *testing.T) {
	truck := Truck{}
	result := CalculateRentalPrice(truck, 3)
	if result != 300.0 {
		t.Errorf("expected 300.0, got %f", result)
	}
}

// Test6: CalculateRentalPrice with Motorcycle for 7 days
func Test6(t *testing.T) {
	motorcycle := Motorcycle{}
	result := CalculateRentalPrice(motorcycle, 7)
	if result != 210.0 {
		t.Errorf("expected 210.0, got %f", result)
	}
}

// Test7: Car implements Vehicle interface
func Test7(t *testing.T) {
	var vehicle Vehicle = Car{}
	result := vehicle.GetPrice(2)
	if result != 100.0 {
		t.Errorf("expected 100.0, got %f", result)
	}
}

// Test8: Truck implements Vehicle interface
func Test8(t *testing.T) {
	var vehicle Vehicle = Truck{}
	result := vehicle.GetPrice(2)
	if result != 200.0 {
		t.Errorf("expected 200.0, got %f", result)
	}
}

// Test9: Motorcycle implements Vehicle interface
func Test9(t *testing.T) {
	var vehicle Vehicle = Motorcycle{}
	result := vehicle.GetPrice(2)
	if result != 60.0 {
		t.Errorf("expected 60.0, got %f", result)
	}
}

// Test10: CalculateRentalPrice with zero days
func Test10(t *testing.T) {
	car := Car{}
	result := CalculateRentalPrice(car, 0)
	if result != 0.0 {
		t.Errorf("expected 0.0, got %f", result)
	}
}
`,
	translations: {
		ru: {
			title: 'Replace Conditional with Polymorphism - Проверка типов',
			description: `Замените условные операторы проверки типов полиморфизмом с использованием интерфейсов для более чистого, расширяемого кода.

**Вы выполните рефакторинг:**

1. **CalculatePrice()** - Имеет switch с проверкой типов
2. Создать **Vehicle интерфейс** с методом GetPrice()
3. Реализовать интерфейс на типах **Car, Truck, Motorcycle**
4. Удалить switch, использовать полиморфный вызов`,
			hint1: `Определите интерфейс Vehicle с одним методом: GetPrice(days int) float64. Затем создайте три пустые структуры: Car, Truck и Motorcycle.`,
			hint2: `Реализуйте метод GetPrice на каждой структуре с соответствующей ставкой (Car: 50, Truck: 100, Motorcycle: 30). Измените параметр CalculateRentalPrice со string на интерфейс Vehicle и замените switch на return vehicle.GetPrice(days).`,
			whyItMatters: `Замена условных операторов полиморфизмом делает код более поддерживаемым, расширяемым и следует принципу открытости/закрытости.`
		},
		uz: {
			title: 'Replace Conditional with Polymorphism - Tur tekshiruvi',
			description: `Tur tekshiruv shartli operatorlarini toza va kengaytiriladigan kod uchun interfeyslar yordamida polimorfizm bilan almashtiring.

**Siz refaktoring qilasiz:**

1. **CalculatePrice()** - Tur tekshiruv switch operatoriga ega
2. Yaratish **Vehicle interfeysi** GetPrice() metodi bilan
3. Interfeysni **Car, Truck, Motorcycle** turlarida amalga oshirish
4. Switch operatorini o'chirish, polimorfik chaqiruvdan foydalanish`,
			hint1: `Vehicle interfeysini bitta metod bilan aniqlang: GetPrice(days int) float64. Keyin uchta bo'sh struktura yarating: Car, Truck va Motorcycle.`,
			hint2: `Har bir strukturada GetPrice metodini mos stavka bilan amalga oshiring (Car: 50, Truck: 100, Motorcycle: 30). CalculateRentalPrice parametrini string dan Vehicle interfeysiga o'zgartiring va switch ni return vehicle.GetPrice(days) bilan almashtiring.`,
			whyItMatters: `Shartli operatorlarni polimorfizm bilan almashtirish kodni yanada qo'llab-quvvatlanadigan, kengaytiriladigan qiladi va ochiqlik/yopiqlik printsipiga amal qiladi.`
		}
	}
};

export default task;
