import { Task } from '../../../../../../../types';

export const task: Task = {
	slug: 'se-refactoring-move-method-envy',
	title: 'Move Method - Feature Envy',
	difficulty: 'medium',
	tags: ['refactoring', 'move-method', 'clean-code', 'go'],
	estimatedTime: '25m',
	isPremium: false,
	youtubeUrl: '',
	description: `Move a method that's more interested in another struct's data to where it belongs - the "feature envy" smell.

**You will refactor:**

1. **OrderProcessor.CalculateShippingCost()** - Envies Address data
2. Move to **Address.CalculateShippingCost()** - Method belongs with its data
3. Update **OrderProcessor.ProcessOrder()** to use moved method

**Key Concepts:**
- **Feature Envy**: Method uses another struct's data more than its own
- **Data and Behavior Together**: Methods should be near the data they use
- **Encapsulation**: Each struct manages its own behavior
- **Tell, Don't Ask**: Objects should do their own work

**Before Refactoring:**

\`\`\`go
type OrderProcessor struct {}

func (op *OrderProcessor) CalculateShippingCost(addr Address) float64 {
    // Uses only Address data, not OrderProcessor data
    distance := calculateDistance(addr.Country, addr.City)
    return distance * 0.5
}
\`\`\`

**After Refactoring:**

\`\`\`go
type Address struct { ... }

func (a *Address) CalculateShippingCost() float64 {
    // Method is with its data
    distance := calculateDistance(a.Country, a.City)
    return distance * 0.5
}
\`\`\`

**When to Move Method:**
- Method uses another struct's data extensively
- Method doesn't use its own struct's data
- Same calculation needed in multiple places
- Better encapsulation by moving
- Follows "Tell, Don't Ask" principle

**Constraints:**
- Move CalculateShippingCost to Address struct
- Update ProcessOrder to call the moved method
- Keep calculateDistance as package-level helper
- Maintain exact same calculation logic`,
	initialCode: `package refactoring

type Address struct {
	Country string
	City    string
	Street  string
	ZipCode string
}

type OrderProcessor struct {
	// OrderProcessor has no fields related to shipping calculation
}

func (op *OrderProcessor) CalculateShippingCost(address Address) float64 {
	return distance * 0.5
}

func (op *OrderProcessor) ProcessOrder(address Address, itemsCost float64) float64 {
	return total
}

func calculateDistance(country, city string) float64 {
	if country == "USA" {
		return 100.0
	}
	return 500.0
}`,
	solutionCode: `package refactoring

type Address struct {
	Country string
	City    string
	Street  string
	ZipCode string
}

type OrderProcessor struct {
	// OrderProcessor has no fields related to shipping calculation
}

// CalculateShippingCost now belongs to Address - it uses Address data
func (a *Address) CalculateShippingCost() float64 {
	distance := calculateDistance(a.Country, a.City)	// uses own data
	return distance * 0.5
}

func (op *OrderProcessor) ProcessOrder(address Address, itemsCost float64) float64 {
	shippingCost := address.CalculateShippingCost()	// Tell, Don't Ask - let Address calculate
	total := itemsCost + shippingCost
	return total
}

// Helper function remains at package level - shared utility
func calculateDistance(country, city string) float64 {
	// Simplified distance calculation
	if country == "USA" {
		return 100.0
	}
	return 500.0
}`,
	hint1: `Move the CalculateShippingCost method to the Address struct. Change the receiver from (op *OrderProcessor) to (a *Address) and update 'address.Country' to 'a.Country' and 'address.City' to 'a.City' inside the method.`,
	hint2: `In ProcessOrder, change the call from 'op.CalculateShippingCost(address)' to 'address.CalculateShippingCost()' since it's now a method of Address.`,
	whyItMatters: `Feature Envy is a code smell where a method is more interested in another class's data than its own. Moving such methods improves cohesion and encapsulation.

**Why Moving Methods (Feature Envy) Matters:**

**1. Improved Cohesion**
Methods belong with the data they manipulate:

\`\`\`go
// Before: Feature envy - Invoice class accessing Order internals
type InvoiceGenerator struct {}

func (ig *InvoiceGenerator) CalculateTotal(order Order) float64 {
    subtotal := 0.0
    for _, item := range order.Items {  // accessing Order's items
        subtotal += item.Price * float64(item.Quantity)
    }
    tax := subtotal * order.TaxRate  // accessing Order's tax rate
    return subtotal + tax
}

// After: Order calculates its own total
type Order struct {
    Items   []Item
    TaxRate float64
}

func (o *Order) CalculateTotal() float64 {
    subtotal := 0.0
    for _, item := range o.Items {
        subtotal += item.Price * float64(item.Quantity)
    }
    tax := subtotal * o.TaxRate
    return subtotal + tax
}

type InvoiceGenerator struct {}

func (ig *InvoiceGenerator) GenerateInvoice(order Order) Invoice {
    total := order.CalculateTotal()  // Tell, Don't Ask
    return Invoice{OrderID: order.ID, Total: total}
}
\`\`\`

**2. Better Encapsulation**
Keep data and behavior together:

\`\`\`go
// Before: Password validation logic scattered
type PasswordValidator struct {}

func (pv *PasswordValidator) IsValid(user User) bool {
    // Reaching into User internals
    return len(user.Password) >= 8 &&
           containsUppercase(user.Password) &&
           containsDigit(user.Password)
}

// After: User validates its own password
type User struct {
    Email    string
    Password string
}

func (u *User) HasValidPassword() bool {
    // User knows how to validate itself
    return len(u.Password) >= 8 &&
           containsUppercase(u.Password) &&
           containsDigit(u.Password)
}

// Validator just coordinates
type PasswordValidator struct {}

func (pv *PasswordValidator) ValidateUser(user User) error {
    if !user.HasValidPassword() {
        return errors.New("invalid password")
    }
    return nil
}
\`\`\`

**3. Reusability**
Method available wherever the data is:

\`\`\`go
// Before: Rectangle area calculation in separate class
type GeometryUtils struct {}

func (gu *GeometryUtils) CalculateArea(rect Rectangle) float64 {
    return rect.Width * rect.Height
}

// Must carry GeometryUtils everywhere you need area
func PrintRectangle(rect Rectangle, utils GeometryUtils) {
    fmt.Printf("Area: %.2f\n", utils.CalculateArea(rect))
}

// After: Rectangle calculates its own area
type Rectangle struct {
    Width  float64
    Height float64
}

func (r *Rectangle) Area() float64 {
    return r.Width * r.Height
}

// Available anywhere you have a Rectangle
func PrintRectangle(rect Rectangle) {
    fmt.Printf("Area: %.2f\n", rect.Area())
}
\`\`\`

**4. Easier Testing**
Test methods with their data:

\`\`\`go
// Before: Must create two objects to test
type PriceCalculator struct {}

func (pc *PriceCalculator) GetDiscountedPrice(product Product) float64 {
    if product.Quantity > 10 {
        return product.Price * 0.9
    }
    return product.Price
}

func TestDiscounting(t *testing.T) {
    pc := PriceCalculator{}
    product := Product{Price: 100, Quantity: 15}
    price := pc.GetDiscountedPrice(product)  // two objects involved
}

// After: Test Product directly
type Product struct {
    Price    float64
    Quantity int
}

func (p *Product) GetDiscountedPrice() float64 {
    if p.Quantity > 10 {
        return p.Price * 0.9
    }
    return p.Price
}

func TestProductDiscount(t *testing.T) {
    product := Product{Price: 100, Quantity: 15}
    price := product.GetDiscountedPrice()  // single, focused test
}
\`\`\`

**5. Tell, Don't Ask Principle**
Objects should do their own work:

\`\`\`go
// Before: Asking for data, then processing it
type ShoppingCart struct {
    Items []Item
}

type CheckoutService struct {}

func (cs *CheckoutService) Process(cart ShoppingCart) {
    // Asking cart for data, then calculating
    total := 0.0
    for _, item := range cart.Items {
        total += item.Price
    }
    if len(cart.Items) > 5 {
        total *= 0.95  // 5% discount
    }
    charge(total)
}

// After: Telling cart to do its work
type ShoppingCart struct {
    Items []Item
}

func (cart *ShoppingCart) CalculateTotal() float64 {
    total := 0.0
    for _, item := range cart.Items {
        total += item.Price
    }
    if len(cart.Items) > 5 {
        total *= 0.95  // cart knows its own discount rules
    }
    return total
}

type CheckoutService struct {}

func (cs *CheckoutService) Process(cart ShoppingCart) {
    total := cart.CalculateTotal()  // Tell cart to calculate
    charge(total)
}
\`\`\`

**Real-World Example - User Account:**

\`\`\`go
// Before: AccountManager accessing User internals
type AccountManager struct {
    db Database
}

func (am *AccountManager) CanUpgradeToPremium(user User) bool {
    // Feature envy - only uses User data
    return user.AccountAge > 30*24*time.Hour &&
           user.TotalPurchases > 100.0 &&
           user.Tier == "basic"
}

func (am *AccountManager) GetMonthlyFee(user User) float64 {
    // More feature envy
    if user.Tier == "premium" {
        return 29.99
    }
    return 0
}

// After: User methods encapsulate User logic
type User struct {
    AccountAge      time.Duration
    TotalPurchases  float64
    Tier            string
}

func (u *User) CanUpgradeToPremium() bool {
    const minAccountAge = 30 * 24 * time.Hour
    const minPurchases = 100.0

    return u.AccountAge > minAccountAge &&
           u.TotalPurchases > minPurchases &&
           u.Tier == "basic"
}

func (u *User) GetMonthlyFee() float64 {
    if u.Tier == "premium" {
        return 29.99
    }
    return 0
}

type AccountManager struct {
    db Database
}

func (am *AccountManager) OfferUpgrade(user User) error {
    if user.CanUpgradeToPremium() {  // Clean delegation
        return am.sendUpgradeEmail(user)
    }
    return nil
}
\`\`\`

**How to Identify Feature Envy:**
1. Method uses more foreign data than its own
2. Method has many calls to another object's getters
3. Method would make more sense on another struct
4. Method name includes name of another struct
5. Changing another struct requires changing this method

**When NOT to Move:**
- Method coordinates between multiple objects
- Moving would break encapsulation
- Method is a utility function used by many structs
- Method implements a design pattern (Strategy, Visitor)
- Current placement follows architectural boundaries`,
	order: 6,
	testCode: `package refactoring

import (
	"testing"
)

// Test1: Address.CalculateShippingCost for USA
func Test1(t *testing.T) {
	addr := &Address{Country: "USA", City: "NYC", Street: "123 Main", ZipCode: "10001"}
	result := addr.CalculateShippingCost()
	expected := 100.0 * 0.5 // distance=100 for USA
	if result != expected {
		t.Errorf("expected %f, got %f", expected, result)
	}
}

// Test2: Address.CalculateShippingCost for non-USA
func Test2(t *testing.T) {
	addr := &Address{Country: "UK", City: "London", Street: "456 Elm", ZipCode: "SW1A"}
	result := addr.CalculateShippingCost()
	expected := 500.0 * 0.5 // distance=500 for non-USA
	if result != expected {
		t.Errorf("expected %f, got %f", expected, result)
	}
}

// Test3: ProcessOrder calculates total correctly for USA
func Test3(t *testing.T) {
	processor := &OrderProcessor{}
	addr := Address{Country: "USA", City: "LA", Street: "789 Oak", ZipCode: "90001"}
	result := processor.ProcessOrder(addr, 100.0)
	expected := 100.0 + 50.0 // items + shipping (100*0.5)
	if result != expected {
		t.Errorf("expected %f, got %f", expected, result)
	}
}

// Test4: ProcessOrder calculates total correctly for non-USA
func Test4(t *testing.T) {
	processor := &OrderProcessor{}
	addr := Address{Country: "Canada", City: "Toronto", Street: "1 Maple", ZipCode: "M5V"}
	result := processor.ProcessOrder(addr, 200.0)
	expected := 200.0 + 250.0 // items + shipping (500*0.5)
	if result != expected {
		t.Errorf("expected %f, got %f", expected, result)
	}
}

// Test5: calculateDistance returns 100 for USA
func Test5(t *testing.T) {
	result := calculateDistance("USA", "Chicago")
	if result != 100.0 {
		t.Errorf("expected 100.0, got %f", result)
	}
}

// Test6: calculateDistance returns 500 for non-USA
func Test6(t *testing.T) {
	result := calculateDistance("Germany", "Berlin")
	if result != 500.0 {
		t.Errorf("expected 500.0, got %f", result)
	}
}

// Test7: Address struct fields work correctly
func Test7(t *testing.T) {
	addr := Address{Country: "FR", City: "Paris", Street: "Champs", ZipCode: "75000"}
	if addr.Country != "FR" || addr.City != "Paris" {
		t.Error("Address fields not set correctly")
	}
}

// Test8: ProcessOrder with zero items cost
func Test8(t *testing.T) {
	processor := &OrderProcessor{}
	addr := Address{Country: "USA", City: "Boston"}
	result := processor.ProcessOrder(addr, 0.0)
	if result != 50.0 {
		t.Errorf("expected 50.0 (just shipping), got %f", result)
	}
}

// Test9: Multiple addresses can calculate shipping
func Test9(t *testing.T) {
	usa := &Address{Country: "USA", City: "NYC"}
	uk := &Address{Country: "UK", City: "London"}

	if usa.CalculateShippingCost() != 50.0 {
		t.Error("USA shipping should be 50.0")
	}
	if uk.CalculateShippingCost() != 250.0 {
		t.Error("UK shipping should be 250.0")
	}
}

// Test10: OrderProcessor can process multiple orders
func Test10(t *testing.T) {
	processor := &OrderProcessor{}
	addr1 := Address{Country: "USA", City: "NYC"}
	addr2 := Address{Country: "UK", City: "London"}

	total1 := processor.ProcessOrder(addr1, 50.0)
	total2 := processor.ProcessOrder(addr2, 50.0)

	if total1 != 100.0 || total2 != 300.0 {
		t.Errorf("expected 100.0 and 300.0, got %f and %f", total1, total2)
	}
}
`,
	translations: {
		ru: {
			title: 'Move Method - Feature Envy',
			description: `Переместите метод, который больше интересуется данными другой структуры, туда, где он должен быть - запах кода "feature envy".

**Вы выполните рефакторинг:**

1. **OrderProcessor.CalculateShippingCost()** - Завидует данным Address
2. Переместить в **Address.CalculateShippingCost()** - Метод принадлежит своим данным
3. Обновить **OrderProcessor.ProcessOrder()** для использования перемещённого метода`,
			hint1: `Переместите метод CalculateShippingCost в структуру Address. Измените получатель с (op *OrderProcessor) на (a *Address) и обновите 'address.Country' на 'a.Country' и 'address.City' на 'a.City' внутри метода.`,
			hint2: `В ProcessOrder измените вызов с 'op.CalculateShippingCost(address)' на 'address.CalculateShippingCost()', так как теперь это метод Address.`,
			whyItMatters: `Feature Envy — это запах кода, когда метод больше интересуется данными другого класса, чем своими. Перемещение таких методов улучшает связность и инкапсуляцию.`
		},
		uz: {
			title: 'Move Method - Feature Envy',
			description: `Boshqa strukturaning ma'lumotlariga ko'proq qiziqadigan metodini uning bo'lishi kerak bo'lgan joyga ko'chiring - "feature envy" code smell.

**Siz refaktoring qilasiz:**

1. **OrderProcessor.CalculateShippingCost()** - Address ma'lumotlariga hasad qiladi
2. Ko'chirish **Address.CalculateShippingCost()** ga - Metod o'z ma'lumotlariga tegishli
3. **OrderProcessor.ProcessOrder()** ni ko'chirilgan metoddan foydalanish uchun yangilash`,
			hint1: `CalculateShippingCost metodini Address strukturasiga ko'chiring. Qabul qiluvchini (op *OrderProcessor) dan (a *Address) ga o'zgartiring va metod ichida 'address.Country' ni 'a.Country' ga va 'address.City' ni 'a.City' ga yangilang.`,
			hint2: `ProcessOrder da chaqiruvni 'op.CalculateShippingCost(address)' dan 'address.CalculateShippingCost()' ga o'zgartiring, chunki endi bu Address metodi.`,
			whyItMatters: `Feature Envy - bu metod o'z ma'lumotlaridan ko'ra boshqa klassning ma'lumotlariga ko'proq qiziqadigan code smell. Bunday metodlarni ko'chirish bog'liqlik va inkapsulatsiyani yaxshilaydi.`
		}
	}
};

export default task;
