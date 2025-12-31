import { Task } from '../../../../../../../types';

export const task: Task = {
	slug: 'se-grasp-information-expert',
	title: 'Information Expert',
	difficulty: 'easy',
	tags: ['go', 'software-engineering', 'grasp', 'information-expert'],
	estimatedTime: '30m',
	isPremium: false,
	youtubeUrl: '',
	description: `Implement the Information Expert principle - assign responsibility to the class that has the information necessary to fulfill it.

**You will implement:**

1. **Order struct** - Contains order items with prices
2. **AddItem(name string, price float64)** - Add an item to the order
3. **CalculateTotal() float64** - Calculate total price (Order has the information!)
4. **GetItemCount() int** - Return number of items (Order knows its items!)

**Key Concepts:**
- **Information Expert**: The object that has the data should do the work
- **Encapsulation**: Keep data and behavior together
- **Low Coupling**: Other objects don't need to know Order's internal structure

**Example Usage:**

\`\`\`go
order := NewOrder()
order.AddItem("Laptop", 999.99)
order.AddItem("Mouse", 29.99)

// Order calculates its own total (it has the price information)
total := order.CalculateTotal()
fmt.Println(total) // 1029.98

// Order knows its own item count
count := order.GetItemCount()
fmt.Println(count) // 2
\`\`\`

**Why Information Expert?**
- **Encapsulation**: Each object is responsible for its own data
- **Maintainability**: Changes to Order structure only affect Order
- **Single Source of Truth**: Order is the expert on order calculations

**Anti-pattern (Don't do this):**
\`\`\`go
// BAD: External object calculating Order's total
func CalculateOrderTotal(order *Order) float64 {
    total := 0.0
    for _, item := range order.items { // exposes internal structure!
        total += item.price
    }
    return total
}
\`\`\`

**Constraints:**
- Order must calculate its own total
- Order must track its own item count
- No external access to internal items slice`,
	initialCode: `package principles

type OrderItem struct {
	Name  string
	Price float64
}

type Order struct {
	items []OrderItem
}

func NewOrder() *Order {
	}
}

func (o *Order) AddItem(name string, price float64) {
}

func (o *Order) CalculateTotal() float64 {
}

func (o *Order) GetItemCount() int {
}`,
	solutionCode: `package principles

type OrderItem struct {
	Name  string
	Price float64
}

type Order struct {
	items []OrderItem	// private: only Order manipulates this data
}

func NewOrder() *Order {
	return &Order{
		items: make([]OrderItem, 0),	// initialize empty slice
	}
}

func (o *Order) AddItem(name string, price float64) {
	// Order is the expert on managing items - no one else should do this
	o.items = append(o.items, OrderItem{
		Name:  name,
		Price: price,
	})
}

func (o *Order) CalculateTotal() float64 {
	total := 0.0
	// Order has the price information, so it calculates the total
	// No other object needs to know how items are stored
	for _, item := range o.items {
		total += item.Price	// sum up all item prices
	}
	return total
}

func (o *Order) GetItemCount() int {
	// Order knows its items, so it provides the count
	// Simple and direct - the expert provides the answer
	return len(o.items)
}`,
	hint1: `AddItem should create an OrderItem and append it to o.items. Use append(o.items, OrderItem{Name: name, Price: price}).`,
	hint2: `CalculateTotal should loop through o.items and sum up all the Price values. GetItemCount should return len(o.items).`,
	whyItMatters: `Information Expert is the most fundamental GRASP principle - it tells you where to put responsibility.

**Why Information Expert Matters:**

**1. Natural Responsibility Assignment**
The object with the data should do the work - it's intuitive and maintainable:

\`\`\`go
// GOOD: Order calculates its own total (it has the prices)
type Order struct {
    items []OrderItem
}

func (o *Order) CalculateTotal() float64 {
    total := 0.0
    for _, item := range o.items {
        total += item.Price
    }
    return total
}

// BAD: External calculator needs access to Order's internal data
type OrderCalculator struct{}

func (oc *OrderCalculator) CalculateTotal(order *Order) float64 {
    total := 0.0
    for _, item := range order.Items { // requires Items to be public!
        total += item.Price
    }
    return total
}
// Now Order's items must be public, breaking encapsulation
\`\`\`

**2. Encapsulation and Data Hiding**
When the expert does the work, you can keep data private:

\`\`\`go
// Information Expert allows private data
type BankAccount struct {
    balance float64  // private - no one else needs to see this
}

func (b *BankAccount) Deposit(amount float64) {
    b.balance += amount  // BankAccount manages its own balance
}

func (b *BankAccount) GetBalance() float64 {
    return b.balance  // controlled access
}

// Without Information Expert, you'd need public fields
type BadBankAccount struct {
    Balance float64  // public - anyone can modify!
}
// Now external code does: account.Balance += 100
// No validation, no control, no encapsulation!
\`\`\`

**3. Easier to Maintain and Change**
When logic lives with data, changes are localized:

\`\`\`go
// Easy to add tax calculation - change only Order
func (o *Order) CalculateTotal() float64 {
    subtotal := 0.0
    for _, item := range o.items {
        subtotal += item.Price
    }
    return subtotal * 1.08  // added 8% tax - one place to change!
}

// With external calculator, you'd need to:
// 1. Expose more Order data
// 2. Update calculator logic
// 3. Risk breaking other code that accesses Order.Items
\`\`\`

**4. Real-World Example: Shopping Cart**
\`\`\`go
type ShoppingCart struct {
    items        []CartItem
    discountCode string
}

// Cart is the expert on its total
func (c *ShoppingCart) GetTotal() float64 {
    subtotal := 0.0
    for _, item := range c.items {
        subtotal += item.Price * float64(item.Quantity)
    }
    return c.applyDiscount(subtotal)  // private method
}

// Cart is the expert on discount calculation
func (c *ShoppingCart) applyDiscount(amount float64) float64 {
    if c.discountCode == "SAVE10" {
        return amount * 0.9
    }
    return amount
}

// Cart is the expert on item management
func (c *ShoppingCart) AddItem(productID string, quantity int) {
    // Check if item exists, update quantity or add new
    // Complex logic that only Cart needs to know about
}
\`\`\`

**5. Preventing Feature Envy Code Smell**
Information Expert prevents "feature envy" - when one object is overly interested in another object's data:

\`\`\`go
// FEATURE ENVY - BAD! Service is envious of Customer's data
type CustomerService struct{}

func (cs *CustomerService) IsVIP(customer *Customer) bool {
    // This logic uses only Customer data - should be in Customer!
    return customer.TotalSpent > 10000 && customer.YearsActive > 5
}

// INFORMATION EXPERT - GOOD! Customer knows if it's VIP
type Customer struct {
    totalSpent  float64
    yearsActive int
}

func (c *Customer) IsVIP() bool {
    return c.totalSpent > 10000 && c.yearsActive > 5
}
\`\`\`

**Common Mistakes:**
- Creating "Manager" or "Helper" classes that manipulate other objects' data
- Making fields public just so external code can do calculations
- Scattering logic about one object across multiple classes

**Rule of Thumb:**
Ask yourself: "Which object has the information needed to perform this task?"
That object should do the work!`,
	order: 0,
	testCode: `package principles

import (
	"testing"
)

// Test1: NewOrder creates empty order
func Test1(t *testing.T) {
	order := NewOrder()
	if order.GetItemCount() != 0 {
		t.Error("New order should have 0 items")
	}
}

// Test2: AddItem increases item count
func Test2(t *testing.T) {
	order := NewOrder()
	order.AddItem("Laptop", 999.99)
	if order.GetItemCount() != 1 {
		t.Errorf("expected 1 item, got %d", order.GetItemCount())
	}
}

// Test3: CalculateTotal returns correct sum
func Test3(t *testing.T) {
	order := NewOrder()
	order.AddItem("Laptop", 999.99)
	order.AddItem("Mouse", 29.99)
	expected := 1029.98
	if order.CalculateTotal() != expected {
		t.Errorf("expected %.2f, got %.2f", expected, order.CalculateTotal())
	}
}

// Test4: Empty order has zero total
func Test4(t *testing.T) {
	order := NewOrder()
	if order.CalculateTotal() != 0.0 {
		t.Error("Empty order should have total of 0")
	}
}

// Test5: Multiple items counted correctly
func Test5(t *testing.T) {
	order := NewOrder()
	order.AddItem("A", 10.0)
	order.AddItem("B", 20.0)
	order.AddItem("C", 30.0)
	if order.GetItemCount() != 3 {
		t.Errorf("expected 3 items, got %d", order.GetItemCount())
	}
}

// Test6: Single item total
func Test6(t *testing.T) {
	order := NewOrder()
	order.AddItem("Single", 50.0)
	if order.CalculateTotal() != 50.0 {
		t.Errorf("expected 50.0, got %.2f", order.CalculateTotal())
	}
}

// Test7: Items with zero price
func Test7(t *testing.T) {
	order := NewOrder()
	order.AddItem("Free", 0.0)
	order.AddItem("Paid", 100.0)
	if order.CalculateTotal() != 100.0 {
		t.Errorf("expected 100.0, got %.2f", order.CalculateTotal())
	}
}

// Test8: OrderItem struct fields
func Test8(t *testing.T) {
	item := OrderItem{Name: "Test", Price: 25.0}
	if item.Name != "Test" || item.Price != 25.0 {
		t.Error("OrderItem fields not set correctly")
	}
}

// Test9: Large number of items
func Test9(t *testing.T) {
	order := NewOrder()
	for i := 0; i < 100; i++ {
		order.AddItem("Item", 1.0)
	}
	if order.GetItemCount() != 100 {
		t.Errorf("expected 100 items, got %d", order.GetItemCount())
	}
	if order.CalculateTotal() != 100.0 {
		t.Errorf("expected 100.0, got %.2f", order.CalculateTotal())
	}
}

// Test10: Decimal precision
func Test10(t *testing.T) {
	order := NewOrder()
	order.AddItem("A", 0.01)
	order.AddItem("B", 0.02)
	order.AddItem("C", 0.03)
	expected := 0.06
	if order.CalculateTotal() != expected {
		t.Errorf("expected %.2f, got %.2f", expected, order.CalculateTotal())
	}
}
`,
	translations: {
		ru: {
			title: 'Информационный эксперт',
			description: `Реализуйте принцип Информационного эксперта — назначьте ответственность классу, который имеет необходимую информацию для её выполнения.

**Вы реализуете:**

1. **Order struct** — Содержит товары заказа с ценами
2. **AddItem(name string, price float64)** — Добавить товар в заказ
3. **CalculateTotal() float64** — Рассчитать общую стоимость (Order имеет информацию!)
4. **GetItemCount() int** — Вернуть количество товаров (Order знает свои товары!)

**Ключевые концепции:**
- **Информационный эксперт**: Объект, имеющий данные, должен выполнять работу
- **Инкапсуляция**: Держите данные и поведение вместе
- **Низкая связанность**: Другим объектам не нужно знать внутреннюю структуру Order

**Пример использования:**

\`\`\`go
order := NewOrder()
order.AddItem("Laptop", 999.99)
order.AddItem("Mouse", 29.99)

// Order рассчитывает свою общую стоимость (у него есть информация о ценах)
total := order.CalculateTotal()
fmt.Println(total) // 1029.98

// Order знает количество своих товаров
count := order.GetItemCount()
fmt.Println(count) // 2
\`\`\`

**Зачем нужен Информационный эксперт?**
- **Инкапсуляция**: Каждый объект отвечает за свои данные
- **Поддерживаемость**: Изменения в структуре Order влияют только на Order
- **Единый источник истины**: Order — эксперт по расчётам заказа

**Анти-паттерн (Не делайте так):**
\`\`\`go
// ПЛОХО: Внешний объект рассчитывает сумму Order
func CalculateOrderTotal(order *Order) float64 {
    total := 0.0
    for _, item := range order.items { // раскрывает внутреннюю структуру!
        total += item.price
    }
    return total
}
\`\`\`

**Ограничения:**
- Order должен сам рассчитывать свою сумму
- Order должен отслеживать количество своих товаров
- Нет внешнего доступа к внутреннему срезу items`,
			hint1: `AddItem должен создать OrderItem и добавить его в o.items. Используйте append(o.items, OrderItem{Name: name, Price: price}).`,
			hint2: `CalculateTotal должен пройти по o.items и просуммировать все значения Price. GetItemCount должен вернуть len(o.items).`,
			whyItMatters: `Информационный эксперт — самый фундаментальный принцип GRASP, который указывает, где разместить ответственность.

**Почему Информационный эксперт важен:**

**1. Естественное назначение ответственности**
Объект с данными должен выполнять работу — это интуитивно и поддерживаемо:

\`\`\`go
// ХОРОШО: Order рассчитывает свою сумму (у него есть цены)
type Order struct {
    items []OrderItem
}

func (o *Order) CalculateTotal() float64 {
    total := 0.0
    for _, item := range o.items {
        total += item.Price
    }
    return total
}

// ПЛОХО: Внешний калькулятор нуждается в доступе к внутренним данным Order
type OrderCalculator struct{}

func (oc *OrderCalculator) CalculateTotal(order *Order) float64 {
    total := 0.0
    for _, item := range order.Items { // требует публичности Items!
        total += item.Price
    }
    return total
}
// Теперь items Order должны быть публичными, нарушая инкапсуляцию
\`\`\`

**2. Инкапсуляция и сокрытие данных**
Когда эксперт выполняет работу, данные можно держать приватными:

\`\`\`go
// Информационный эксперт позволяет приватные данные
type BankAccount struct {
    balance float64  // приватное - никому не нужно это видеть
}

func (b *BankAccount) Deposit(amount float64) {
    b.balance += amount  // BankAccount управляет своим балансом
}

func (b *BankAccount) GetBalance() float64 {
    return b.balance  // контролируемый доступ
}

// Без информационного эксперта нужны публичные поля
type BadBankAccount struct {
    Balance float64  // публичное - кто угодно может изменить!
}
// Теперь внешний код делает: account.Balance += 100
// Нет валидации, нет контроля, нет инкапсуляции!
\`\`\`

**Распространённые ошибки:**
- Создание "Manager" или "Helper" классов, манипулирующих данными других объектов
- Публичные поля только для внешних вычислений
- Разбрасывание логики одного объекта по разным классам`,
			solutionCode: `package principles

type OrderItem struct {
	Name  string
	Price float64
}

type Order struct {
	items []OrderItem	// приватное: только Order манипулирует этими данными
}

func NewOrder() *Order {
	return &Order{
		items: make([]OrderItem, 0),	// инициализируем пустой срез
	}
}

func (o *Order) AddItem(name string, price float64) {
	// Order — эксперт по управлению товарами - никто другой не должен этим заниматься
	o.items = append(o.items, OrderItem{
		Name:  name,
		Price: price,
	})
}

func (o *Order) CalculateTotal() float64 {
	total := 0.0
	// Order имеет информацию о ценах, поэтому он рассчитывает сумму
	// Никакому другому объекту не нужно знать, как хранятся товары
	for _, item := range o.items {
		total += item.Price	// суммируем все цены товаров
	}
	return total
}

func (o *Order) GetItemCount() int {
	// Order знает свои товары, поэтому он предоставляет их количество
	// Просто и понятно - эксперт предоставляет ответ
	return len(o.items)
}`
		},
		uz: {
			title: 'Information Expert (Ma\'lumot eksperti)',
			description: `Information Expert prinsipini amalga oshiring — mas'uliyatni vazifani bajarish uchun zarur ma'lumotga ega bo'lgan klassga belgilang.

**Siz amalga oshirasiz:**

1. **Order struct** — Narxlar bilan buyurtma elementlarini o'z ichiga oladi
2. **AddItem(name string, price float64)** — Buyurtmaga element qo'shish
3. **CalculateTotal() float64** — Umumiy narxni hisoblash (Order ma'lumotga ega!)
4. **GetItemCount() int** — Elementlar sonini qaytarish (Order o'z elementlarini biladi!)

**Asosiy tushunchalar:**
- **Information Expert**: Ma'lumotga ega bo'lgan ob'ekt ishni bajarishi kerak
- **Inkapsulyatsiya**: Ma'lumot va xatti-harakatni birga saqlang
- **Past bog'lanish**: Boshqa ob'ektlar Order ning ichki tuzilishini bilishi shart emas

**Foydalanish misoli:**

\`\`\`go
order := NewOrder()
order.AddItem("Laptop", 999.99)
order.AddItem("Mouse", 29.99)

// Order o'zining umumiy narxini hisoblaydi (unda narx ma'lumoti bor)
total := order.CalculateTotal()
fmt.Println(total) // 1029.98

// Order o'z elementlari sonini biladi
count := order.GetItemCount()
fmt.Println(count) // 2
\`\`\`

**Nima uchun Information Expert?**
- **Inkapsulyatsiya**: Har bir ob'ekt o'z ma'lumoti uchun javobgar
- **Parvarish qilish qulayligi**: Order strukturasidagi o'zgarishlar faqat Order ga ta'sir qiladi
- **Yagona haqiqat manbai**: Order buyurtma hisob-kitoblari bo'yicha ekspert

**Anti-pattern (Buni qilmang):**
\`\`\`go
// YOMON: Tashqi ob'ekt Order summani hisoblaydi
func CalculateOrderTotal(order *Order) float64 {
    total := 0.0
    for _, item := range order.items { // ichki tuzilishni ochib beradi!
        total += item.price
    }
    return total
}
\`\`\`

**Cheklovlar:**
- Order o'z summasini hisoblashi kerak
- Order o'z elementlari sonini kuzatishi kerak
- Ichki items slice ga tashqi kirish yo'q`,
			hint1: `AddItem OrderItem yaratishi va uni o.items ga qo'shishi kerak. append(o.items, OrderItem{Name: name, Price: price}) dan foydalaning.`,
			hint2: `CalculateTotal o.items bo'ylab aylanib, barcha Price qiymatlarini yig'ishi kerak. GetItemCount len(o.items) qaytarishi kerak.`,
			whyItMatters: `Information Expert eng asosiy GRASP printsipi - bu sizga mas'uliyatni qaerga joylashtirish kerakligini aytadi.

**Information Expert nima uchun muhim:**

**1. Tabiiy mas'uliyat tayinlash**
Ma'lumotga ega ob'ekt ishni bajarishi kerak - bu intuitiv va parvarish qilish mumkin:

\`\`\`go
// YAXSHI: Order o'z summasini hisoblaydi (unda narxlar bor)
type Order struct {
    items []OrderItem
}

func (o *Order) CalculateTotal() float64 {
    total := 0.0
    for _, item := range o.items {
        total += item.Price
    }
    return total
}

// YOMON: Tashqi kalkulyator Order ichki ma'lumotlariga kirishni talab qiladi
type OrderCalculator struct{}

func (oc *OrderCalculator) CalculateTotal(order *Order) float64 {
    total := 0.0
    for _, item := range order.Items { // Items ommaviy bo'lishini talab qiladi!
        total += item.Price
    }
    return total
}
// Endi Order ning items lari ommaviy bo'lishi kerak, inkapsulyatsiyani buzadi
\`\`\`

**Umumiy xatolar:**
- Boshqa ob'ektlar ma'lumotlarini manipulyatsiya qiluvchi "Manager" yoki "Helper" klasslari yaratish
- Tashqi hisob-kitoblar uchun faqat ommaviy maydonlar qilish
- Bir ob'ekt mantiqini bir nechta klasslar bo'ylab tarqatish`,
			solutionCode: `package principles

type OrderItem struct {
	Name  string
	Price float64
}

type Order struct {
	items []OrderItem	// privat: faqat Order bu ma'lumotlarni boshqaradi
}

func NewOrder() *Order {
	return &Order{
		items: make([]OrderItem, 0),	// bo'sh slice ni initsializatsiya qilamiz
	}
}

func (o *Order) AddItem(name string, price float64) {
	// Order elementlarni boshqarish bo'yicha ekspert - hech kim buni qilmasligi kerak
	o.items = append(o.items, OrderItem{
		Name:  name,
		Price: price,
	})
}

func (o *Order) CalculateTotal() float64 {
	total := 0.0
	// Order narx ma'lumotiga ega, shuning uchun u summani hisoblaydi
	// Boshqa hech bir ob'ekt elementlar qanday saqlanishini bilishi shart emas
	for _, item := range o.items {
		total += item.Price	// barcha element narxlarini yig'amiz
	}
	return total
}

func (o *Order) GetItemCount() int {
	// Order o'z elementlarini biladi, shuning uchun u sonni taqdim etadi
	// Oddiy va aniq - ekspert javobni beradi
	return len(o.items)
}`
		}
	}
};

export default task;
