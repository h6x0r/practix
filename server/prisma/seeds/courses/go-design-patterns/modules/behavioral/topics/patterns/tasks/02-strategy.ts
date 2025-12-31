import { Task } from '../../../../../../../types';

export const task: Task = {
	slug: 'go-dp-strategy',
	title: 'Strategy Pattern',
	difficulty: 'easy',
	tags: ['go', 'design-patterns', 'behavioral', 'strategy'],
	estimatedTime: '25m',
	isPremium: false,
	youtubeUrl: '',
	description: `Implement the Strategy pattern in Go - define a family of algorithms, encapsulate each one, and make them interchangeable at runtime.

The Strategy pattern lets you define a family of algorithms (strategies), put each of them into a separate struct, and make their objects interchangeable. The context object delegates the algorithm execution to a linked strategy object instead of implementing multiple versions directly.

**You will implement:**

1. **PaymentStrategy interface** - Common interface with Pay(amount) method
2. **CreditCardStrategy struct** - Concrete strategy for credit card payments
3. **PayPalStrategy struct** - Concrete strategy for PayPal payments
4. **ShoppingCart struct** - Context that uses a strategy to process payment

**Example Usage:**

\`\`\`go
cart := NewShoppingCart()	// create context (shopping cart)
cart.AddItem(100)	// add item worth 100
cart.AddItem(50)	// add item worth 50

// Set credit card strategy
cart.SetPaymentStrategy(&CreditCardStrategy{CardNumber: "1234-5678"})	// inject strategy
result1 := cart.Checkout()	// execute strategy
// result1: "Paid 150 using credit card 1234-5678"

// Switch to PayPal strategy at runtime
cart.SetPaymentStrategy(&PayPalStrategy{Email: "user@example.com"})	// change strategy
result2 := cart.Checkout()	// execute new strategy
// result2: "Paid 150 using PayPal user@example.com"
\`\`\``,
	initialCode: `package patterns

import "fmt"

type PaymentStrategy interface {
}

type CreditCardStrategy struct {
	CardNumber string
}

func (c *CreditCardStrategy) Pay(amount int) string {
}

type PayPalStrategy struct {
	Email string
}

func (p *PayPalStrategy) Pay(amount int) string {
}

type ShoppingCart struct {
	items    []int
	strategy PaymentStrategy
}

func NewShoppingCart() *ShoppingCart {
}

func (s *ShoppingCart) AddItem(price int) {
}

func (s *ShoppingCart) SetPaymentStrategy(strategy PaymentStrategy) {
}

func (s *ShoppingCart) Checkout() string {
}`,
	solutionCode: `package patterns

import "fmt"	// import for string formatting

// PaymentStrategy defines payment algorithm interface
type PaymentStrategy interface {	// strategy interface - all strategies implement this
	Pay(amount int) string	// method signature for payment processing
}

// CreditCardStrategy pays with credit card
type CreditCardStrategy struct {	// concrete strategy for credit card
	CardNumber string	// card number for payment
}

// Pay processes payment using credit card
func (c *CreditCardStrategy) Pay(amount int) string {	// implements PaymentStrategy
	return fmt.Sprintf("Paid %d using credit card %s", amount, c.CardNumber)	// format payment confirmation
}

// PayPalStrategy pays with PayPal
type PayPalStrategy struct {	// concrete strategy for PayPal
	Email string	// PayPal email account
}

// Pay processes payment using PayPal
func (p *PayPalStrategy) Pay(amount int) string {	// implements PaymentStrategy
	return fmt.Sprintf("Paid %d using PayPal %s", amount, p.Email)	// format payment confirmation
}

// ShoppingCart uses payment strategy (context)
type ShoppingCart struct {	// context that uses strategies
	items    []int	// list of item prices
	strategy PaymentStrategy	// current payment strategy
}

// NewShoppingCart creates empty cart
func NewShoppingCart() *ShoppingCart {	// factory function for context
	return &ShoppingCart{items: make([]int, 0)}	// initialize with empty items
}

// AddItem adds item price to cart
func (s *ShoppingCart) AddItem(price int) {	// method to add items
	s.items = append(s.items, price)	// append price to items slice
}

// SetPaymentStrategy sets the payment strategy
func (s *ShoppingCart) SetPaymentStrategy(strategy PaymentStrategy) {	// strategy injection
	s.strategy = strategy	// store strategy reference
}

// Checkout processes payment using current strategy
func (s *ShoppingCart) Checkout() string {	// execute strategy
	if s.strategy == nil {	// check if strategy is set
		return "No payment strategy set"	// return error message if no strategy
	}
	total := 0	// accumulator for total price
	for _, price := range s.items {	// iterate through all items
		total += price	// sum up prices
	}
	return s.strategy.Pay(total)	// delegate to strategy
}`,
	hint1: `Each concrete strategy (CreditCardStrategy, PayPalStrategy) implements the Pay method by formatting a string with the amount and the payment-specific detail (card number or email). Use fmt.Sprintf for formatting.`,
	hint2: `In ShoppingCart: AddItem appends to items slice, SetPaymentStrategy stores the strategy reference. Checkout first checks if strategy is nil (return error message), then calculates total by summing items, and finally calls strategy.Pay(total).`,
	whyItMatters: `**Why the Strategy Pattern Exists**

Without Strategy, you might use conditionals to select payment methods:

\`\`\`go
// Problem: Hard to add new payment methods, violates Open/Closed principle
func (cart *Cart) Checkout(method string, details map[string]string) string {
    total := cart.calculateTotal()
    switch method {
    case "credit":
        return fmt.Sprintf("Paid %d with card %s", total, details["card"])
    case "paypal":
        return fmt.Sprintf("Paid %d via PayPal %s", total, details["email"])
    case "bitcoin":  // Adding new method requires modifying this function
        return fmt.Sprintf("Paid %d with Bitcoin %s", total, details["wallet"])
    }
    return "Unknown method"
}
\`\`\`

With Strategy, adding new payment methods is easy:

\`\`\`go
// Solution: Each payment method is a separate strategy
type PaymentStrategy interface {
    Pay(amount int) string
}

func (cart *Cart) Checkout() string {
    return cart.strategy.Pay(cart.total)  // delegate to strategy
}

// Adding Bitcoin is just a new struct implementing PaymentStrategy
type BitcoinStrategy struct { Wallet string }
func (b *BitcoinStrategy) Pay(amount int) string { ... }
\`\`\`

**Real-World Strategy Examples in Go**

1. **Compression Strategies**:
\`\`\`go
type Compressor interface {
    Compress(data []byte) []byte
    GetExtension() string
}

type GzipCompressor struct{}
func (g *GzipCompressor) Compress(data []byte) []byte { /* gzip logic */ }
func (g *GzipCompressor) GetExtension() string { return ".gz" }

type ZipCompressor struct{}
func (z *ZipCompressor) Compress(data []byte) []byte { /* zip logic */ }
func (z *ZipCompressor) GetExtension() string { return ".zip" }

type FileArchiver struct {
    compressor Compressor
}

func (f *FileArchiver) Archive(filename string, data []byte) {
    compressed := f.compressor.Compress(data)
    os.WriteFile(filename + f.compressor.GetExtension(), compressed, 0644)
}
\`\`\`

2. **Sorting Strategies**:
\`\`\`go
type SortStrategy interface {
    Sort(data []int) []int
}

type QuickSort struct{}
func (q *QuickSort) Sort(data []int) []int { /* quicksort */ }

type MergeSort struct{}
func (m *MergeSort) Sort(data []int) []int { /* mergesort */ }

type DataProcessor struct {
    sorter SortStrategy
}

func (d *DataProcessor) Process(data []int) []int {
    return d.sorter.Sort(data)
}
\`\`\`

**Production Pattern: Rate Limiter with Multiple Strategies**

\`\`\`go
package main

import (
    "sync"
    "time"
)

// RateLimitStrategy defines the rate limiting algorithm
type RateLimitStrategy interface {
    Allow(key string) bool
    GetName() string
}

// TokenBucketStrategy implements token bucket algorithm
type TokenBucketStrategy struct {
    capacity   int
    refillRate int
    buckets    map[string]*bucket
    mu         sync.Mutex
}

type bucket struct {
    tokens     int
    lastRefill time.Time
}

func NewTokenBucket(capacity, refillRate int) *TokenBucketStrategy {
    return &TokenBucketStrategy{
        capacity:   capacity,
        refillRate: refillRate,
        buckets:    make(map[string]*bucket),
    }
}

func (t *TokenBucketStrategy) Allow(key string) bool {
    t.mu.Lock()
    defer t.mu.Unlock()

    b, exists := t.buckets[key]
    if !exists {
        b = &bucket{tokens: t.capacity, lastRefill: time.Now()}
        t.buckets[key] = b
    }

    // Refill tokens based on time elapsed
    elapsed := time.Since(b.lastRefill)
    refillAmount := int(elapsed.Seconds()) * t.refillRate
    b.tokens = min(t.capacity, b.tokens + refillAmount)
    b.lastRefill = time.Now()

    if b.tokens > 0 {
        b.tokens--
        return true
    }
    return false
}

func (t *TokenBucketStrategy) GetName() string { return "TokenBucket" }

// SlidingWindowStrategy implements sliding window counter
type SlidingWindowStrategy struct {
    windowSize time.Duration
    maxRequests int
    windows    map[string][]time.Time
    mu         sync.Mutex
}

func NewSlidingWindow(windowSize time.Duration, maxRequests int) *SlidingWindowStrategy {
    return &SlidingWindowStrategy{
        windowSize:  windowSize,
        maxRequests: maxRequests,
        windows:     make(map[string][]time.Time),
    }
}

func (s *SlidingWindowStrategy) Allow(key string) bool {
    s.mu.Lock()
    defer s.mu.Unlock()

    now := time.Now()
    windowStart := now.Add(-s.windowSize)

    // Remove expired timestamps
    var valid []time.Time
    for _, t := range s.windows[key] {
        if t.After(windowStart) {
            valid = append(valid, t)
        }
    }
    s.windows[key] = valid

    if len(valid) < s.maxRequests {
        s.windows[key] = append(s.windows[key], now)
        return true
    }
    return false
}

func (s *SlidingWindowStrategy) GetName() string { return "SlidingWindow" }

// RateLimiter is the context that uses strategies
type RateLimiter struct {
    strategy RateLimitStrategy
}

func NewRateLimiter(strategy RateLimitStrategy) *RateLimiter {
    return &RateLimiter{strategy: strategy}
}

func (r *RateLimiter) SetStrategy(strategy RateLimitStrategy) {
    r.strategy = strategy
}

func (r *RateLimiter) AllowRequest(clientID string) bool {
    return r.strategy.Allow(clientID)
}

func min(a, b int) int {
    if a < b { return a }
    return b
}
\`\`\`

**Common Mistakes to Avoid**

1. **Strategy with too many dependencies** - Strategies should be self-contained and focused
2. **Context knowing concrete strategies** - Context should only depend on the interface
3. **Not validating strategy before use** - Always check if strategy is nil before calling
4. **Overusing Strategy** - Simple conditionals may be better for 2-3 fixed cases
5. **Strategies sharing mutable state** - Each strategy should be independent`,
	order: 1,
	testCode: `package patterns

import (
	"strings"
	"testing"
)

// Test1: CreditCardStrategy.Pay returns correct format
func Test1(t *testing.T) {
	c := &CreditCardStrategy{CardNumber: "1234-5678"}
	result := c.Pay(100)
	if !strings.Contains(result, "100") {
		t.Error("Should contain amount")
	}
	if !strings.Contains(result, "1234-5678") {
		t.Error("Should contain card number")
	}
}

// Test2: PayPalStrategy.Pay returns correct format
func Test2(t *testing.T) {
	p := &PayPalStrategy{Email: "user@example.com"}
	result := p.Pay(50)
	if !strings.Contains(result, "50") {
		t.Error("Should contain amount")
	}
	if !strings.Contains(result, "user@example.com") {
		t.Error("Should contain email")
	}
}

// Test3: NewShoppingCart creates empty cart
func Test3(t *testing.T) {
	cart := NewShoppingCart()
	if cart == nil {
		t.Error("NewShoppingCart should return non-nil")
	}
}

// Test4: AddItem adds items to cart
func Test4(t *testing.T) {
	cart := NewShoppingCart()
	cart.AddItem(100)
	cart.AddItem(50)
	cart.SetPaymentStrategy(&CreditCardStrategy{CardNumber: "1234"})
	result := cart.Checkout()
	if !strings.Contains(result, "150") {
		t.Error("Should sum items to 150")
	}
}

// Test5: SetPaymentStrategy sets the strategy
func Test5(t *testing.T) {
	cart := NewShoppingCart()
	cart.AddItem(100)
	cart.SetPaymentStrategy(&PayPalStrategy{Email: "test@test.com"})
	result := cart.Checkout()
	if !strings.Contains(result, "PayPal") {
		t.Error("Should use PayPal strategy")
	}
}

// Test6: Checkout without strategy returns error message
func Test6(t *testing.T) {
	cart := NewShoppingCart()
	cart.AddItem(100)
	result := cart.Checkout()
	if !strings.Contains(result, "No payment strategy") {
		t.Error("Should return error when no strategy set")
	}
}

// Test7: Strategy can be changed at runtime
func Test7(t *testing.T) {
	cart := NewShoppingCart()
	cart.AddItem(100)
	cart.SetPaymentStrategy(&CreditCardStrategy{CardNumber: "1234"})
	r1 := cart.Checkout()
	cart.SetPaymentStrategy(&PayPalStrategy{Email: "a@a.com"})
	r2 := cart.Checkout()
	if !strings.Contains(r1, "credit card") || !strings.Contains(r2, "PayPal") {
		t.Error("Should switch strategy at runtime")
	}
}

// Test8: Empty cart checkout works
func Test8(t *testing.T) {
	cart := NewShoppingCart()
	cart.SetPaymentStrategy(&CreditCardStrategy{CardNumber: "1234"})
	result := cart.Checkout()
	if !strings.Contains(result, "0") {
		t.Error("Empty cart should have total 0")
	}
}

// Test9: Multiple items sum correctly
func Test9(t *testing.T) {
	cart := NewShoppingCart()
	cart.AddItem(10)
	cart.AddItem(20)
	cart.AddItem(30)
	cart.AddItem(40)
	cart.SetPaymentStrategy(&CreditCardStrategy{CardNumber: "test"})
	result := cart.Checkout()
	if !strings.Contains(result, "100") {
		t.Error("Should sum to 100")
	}
}

// Test10: CreditCardStrategy implements PaymentStrategy
func Test10(t *testing.T) {
	var s PaymentStrategy = &CreditCardStrategy{CardNumber: "test"}
	result := s.Pay(25)
	if result == "" {
		t.Error("Should return non-empty string")
	}
}
`,
	translations: {
		ru: {
			title: 'Паттерн Strategy (Стратегия)',
			description: `Реализуйте паттерн Strategy на Go — определите семейство алгоритмов, инкапсулируйте каждый из них и сделайте их взаимозаменяемыми во время выполнения.

Паттерн Strategy позволяет определить семейство алгоритмов (стратегий), поместить каждый из них в отдельную структуру и сделать их объекты взаимозаменяемыми. Контекстный объект делегирует выполнение алгоритма связанному объекту стратегии вместо реализации нескольких версий напрямую.

**Вы реализуете:**

1. **Интерфейс PaymentStrategy** — Общий интерфейс с методом Pay(amount)
2. **Структура CreditCardStrategy** — Конкретная стратегия для оплаты кредитной картой
3. **Структура PayPalStrategy** — Конкретная стратегия для оплаты через PayPal
4. **Структура ShoppingCart** — Контекст, использующий стратегию для обработки платежа

**Пример использования:**

\`\`\`go
cart := NewShoppingCart()	// создаём контекст (корзину)
cart.AddItem(100)	// добавляем товар стоимостью 100
cart.AddItem(50)	// добавляем товар стоимостью 50

// Устанавливаем стратегию кредитной карты
cart.SetPaymentStrategy(&CreditCardStrategy{CardNumber: "1234-5678"})	// внедряем стратегию
result1 := cart.Checkout()	// выполняем стратегию
// result1: "Paid 150 using credit card 1234-5678"

// Переключаемся на стратегию PayPal во время выполнения
cart.SetPaymentStrategy(&PayPalStrategy{Email: "user@example.com"})	// меняем стратегию
result2 := cart.Checkout()	// выполняем новую стратегию
// result2: "Paid 150 using PayPal user@example.com"
\`\`\``,
			hint1: `Каждая конкретная стратегия (CreditCardStrategy, PayPalStrategy) реализует метод Pay, форматируя строку с суммой и деталью оплаты (номер карты или email). Используйте fmt.Sprintf для форматирования.`,
			hint2: `В ShoppingCart: AddItem добавляет в срез items, SetPaymentStrategy сохраняет ссылку на стратегию. Checkout сначала проверяет, равна ли strategy nil (возвращает сообщение об ошибке), затем вычисляет total суммированием items, и наконец вызывает strategy.Pay(total).`,
			whyItMatters: `**Зачем нужен паттерн Strategy**

Без Strategy вы можете использовать условия для выбора способа оплаты:

\`\`\`go
// Проблема: Трудно добавить новые способы оплаты, нарушает принцип Open/Closed
func (cart *Cart) Checkout(method string, details map[string]string) string {
    total := cart.calculateTotal()
    switch method {
    case "credit":
        return fmt.Sprintf("Paid %d with card %s", total, details["card"])
    case "paypal":
        return fmt.Sprintf("Paid %d via PayPal %s", total, details["email"])
    case "bitcoin":  // Добавление нового метода требует изменения этой функции
        return fmt.Sprintf("Paid %d with Bitcoin %s", total, details["wallet"])
    }
    return "Unknown method"
}
\`\`\`

Со Strategy добавление новых способов оплаты просто:

\`\`\`go
// Решение: Каждый способ оплаты — отдельная стратегия
type PaymentStrategy interface {
    Pay(amount int) string
}

func (cart *Cart) Checkout() string {
    return cart.strategy.Pay(cart.total)  // делегируем стратегии
}

// Добавление Bitcoin — просто новая структура, реализующая PaymentStrategy
type BitcoinStrategy struct { Wallet string }
func (b *BitcoinStrategy) Pay(amount int) string { ... }
\`\`\`

**Реальные примеры Strategy в Go**

1. **Стратегии сжатия**:
\`\`\`go
type Compressor interface {
    Compress(data []byte) []byte
    GetExtension() string
}

type GzipCompressor struct{}
func (g *GzipCompressor) Compress(data []byte) []byte { /* логика gzip */ }
func (g *GzipCompressor) GetExtension() string { return ".gz" }

type ZipCompressor struct{}
func (z *ZipCompressor) Compress(data []byte) []byte { /* логика zip */ }
func (z *ZipCompressor) GetExtension() string { return ".zip" }

type FileArchiver struct {
    compressor Compressor
}

func (f *FileArchiver) Archive(filename string, data []byte) {
    compressed := f.compressor.Compress(data)
    os.WriteFile(filename + f.compressor.GetExtension(), compressed, 0644)
}
\`\`\`

2. **Стратегии сортировки**:
\`\`\`go
type SortStrategy interface {
    Sort(data []int) []int
}

type QuickSort struct{}
func (q *QuickSort) Sort(data []int) []int { /* quicksort */ }

type MergeSort struct{}
func (m *MergeSort) Sort(data []int) []int { /* mergesort */ }

type DataProcessor struct {
    sorter SortStrategy
}

func (d *DataProcessor) Process(data []int) []int {
    return d.sorter.Sort(data)
}
\`\`\`

**Продакшен паттерн: Rate Limiter с несколькими стратегиями**

\`\`\`go
package main

import (
    "sync"
    "time"
)

// RateLimitStrategy определяет алгоритм ограничения скорости
type RateLimitStrategy interface {
    Allow(key string) bool
    GetName() string
}

// TokenBucketStrategy реализует алгоритм token bucket
type TokenBucketStrategy struct {
    capacity   int
    refillRate int
    buckets    map[string]*bucket
    mu         sync.Mutex
}

type bucket struct {
    tokens     int
    lastRefill time.Time
}

func NewTokenBucket(capacity, refillRate int) *TokenBucketStrategy {
    return &TokenBucketStrategy{
        capacity:   capacity,
        refillRate: refillRate,
        buckets:    make(map[string]*bucket),
    }
}

func (t *TokenBucketStrategy) Allow(key string) bool {
    t.mu.Lock()
    defer t.mu.Unlock()

    b, exists := t.buckets[key]
    if !exists {
        b = &bucket{tokens: t.capacity, lastRefill: time.Now()}
        t.buckets[key] = b
    }

    // Пополняем токены на основе прошедшего времени
    elapsed := time.Since(b.lastRefill)
    refillAmount := int(elapsed.Seconds()) * t.refillRate
    b.tokens = min(t.capacity, b.tokens + refillAmount)
    b.lastRefill = time.Now()

    if b.tokens > 0 {
        b.tokens--
        return true
    }
    return false
}

func (t *TokenBucketStrategy) GetName() string { return "TokenBucket" }

// SlidingWindowStrategy реализует счётчик скользящего окна
type SlidingWindowStrategy struct {
    windowSize time.Duration
    maxRequests int
    windows    map[string][]time.Time
    mu         sync.Mutex
}

func NewSlidingWindow(windowSize time.Duration, maxRequests int) *SlidingWindowStrategy {
    return &SlidingWindowStrategy{
        windowSize:  windowSize,
        maxRequests: maxRequests,
        windows:     make(map[string][]time.Time),
    }
}

func (s *SlidingWindowStrategy) Allow(key string) bool {
    s.mu.Lock()
    defer s.mu.Unlock()

    now := time.Now()
    windowStart := now.Add(-s.windowSize)

    // Удаляем истёкшие временные метки
    var valid []time.Time
    for _, t := range s.windows[key] {
        if t.After(windowStart) {
            valid = append(valid, t)
        }
    }
    s.windows[key] = valid

    if len(valid) < s.maxRequests {
        s.windows[key] = append(s.windows[key], now)
        return true
    }
    return false
}

func (s *SlidingWindowStrategy) GetName() string { return "SlidingWindow" }

// RateLimiter — контекст, использующий стратегии
type RateLimiter struct {
    strategy RateLimitStrategy
}

func NewRateLimiter(strategy RateLimitStrategy) *RateLimiter {
    return &RateLimiter{strategy: strategy}
}

func (r *RateLimiter) SetStrategy(strategy RateLimitStrategy) {
    r.strategy = strategy
}

func (r *RateLimiter) AllowRequest(clientID string) bool {
    return r.strategy.Allow(clientID)
}

func min(a, b int) int {
    if a < b { return a }
    return b
}
\`\`\`

**Распространённые ошибки**

1. **Стратегия с слишком многими зависимостями** — Стратегии должны быть самодостаточными и сфокусированными
2. **Контекст знает конкретные стратегии** — Контекст должен зависеть только от интерфейса
3. **Не проверяют стратегию перед использованием** — Всегда проверяйте, равна ли strategy nil перед вызовом
4. **Чрезмерное использование Strategy** — Простые условия могут быть лучше для 2-3 фиксированных случаев
5. **Стратегии разделяют изменяемое состояние** — Каждая стратегия должна быть независимой`,
			solutionCode: `package patterns

import "fmt"	// импорт для форматирования строк

// PaymentStrategy определяет интерфейс алгоритма оплаты
type PaymentStrategy interface {	// интерфейс стратегии — все стратегии реализуют его
	Pay(amount int) string	// сигнатура метода для обработки платежа
}

// CreditCardStrategy оплачивает кредитной картой
type CreditCardStrategy struct {	// конкретная стратегия для кредитной карты
	CardNumber string	// номер карты для оплаты
}

// Pay обрабатывает платёж кредитной картой
func (c *CreditCardStrategy) Pay(amount int) string {	// реализует PaymentStrategy
	return fmt.Sprintf("Paid %d using credit card %s", amount, c.CardNumber)	// форматируем подтверждение платежа
}

// PayPalStrategy оплачивает через PayPal
type PayPalStrategy struct {	// конкретная стратегия для PayPal
	Email string	// email аккаунта PayPal
}

// Pay обрабатывает платёж через PayPal
func (p *PayPalStrategy) Pay(amount int) string {	// реализует PaymentStrategy
	return fmt.Sprintf("Paid %d using PayPal %s", amount, p.Email)	// форматируем подтверждение платежа
}

// ShoppingCart использует стратегию оплаты (контекст)
type ShoppingCart struct {	// контекст, использующий стратегии
	items    []int	// список цен товаров
	strategy PaymentStrategy	// текущая стратегия оплаты
}

// NewShoppingCart создаёт пустую корзину
func NewShoppingCart() *ShoppingCart {	// фабричная функция для контекста
	return &ShoppingCart{items: make([]int, 0)}	// инициализируем пустыми товарами
}

// AddItem добавляет цену товара в корзину
func (s *ShoppingCart) AddItem(price int) {	// метод добавления товаров
	s.items = append(s.items, price)	// добавляем цену в срез items
}

// SetPaymentStrategy устанавливает стратегию оплаты
func (s *ShoppingCart) SetPaymentStrategy(strategy PaymentStrategy) {	// внедрение стратегии
	s.strategy = strategy	// сохраняем ссылку на стратегию
}

// Checkout обрабатывает платёж используя текущую стратегию
func (s *ShoppingCart) Checkout() string {	// выполнение стратегии
	if s.strategy == nil {	// проверяем, установлена ли стратегия
		return "No payment strategy set"	// возвращаем сообщение об ошибке если нет стратегии
	}
	total := 0	// накопитель для общей цены
	for _, price := range s.items {	// перебираем все товары
		total += price	// суммируем цены
	}
	return s.strategy.Pay(total)	// делегируем стратегии
}`
		},
		uz: {
			title: 'Strategy (Strategiya) Pattern',
			description: `Go tilida Strategy patternini amalga oshiring — algoritmlar oilasini aniqlang, har birini inkapsulyatsiya qiling va ularni ishlash vaqtida almashtirib turadigan qiling.

Strategy patterni algoritmlar oilasini (strategiyalar) aniqlash, ularning har birini alohida structga joylashtirish va ularning ob'ektlarini almashtirib turish imkonini beradi. Kontekst ob'ekti algoritm bajarilishini bir nechta versiyalarni to'g'ridan-to'g'ri amalga oshirish o'rniga bog'langan strategiya ob'ektiga delegatsiya qiladi.

**Siz amalga oshirasiz:**

1. **PaymentStrategy interfeysi** — Pay(amount) metodi bilan umumiy interfeys
2. **CreditCardStrategy struct** — Kredit karta to'lovlari uchun konkret strategiya
3. **PayPalStrategy struct** — PayPal to'lovlari uchun konkret strategiya
4. **ShoppingCart struct** — To'lovni qayta ishlash uchun strategiya ishlatadigan kontekst

**Foydalanish namunasi:**

\`\`\`go
cart := NewShoppingCart()	// kontekst (savat) yaratamiz
cart.AddItem(100)	// 100 qiymatli mahsulot qo'shamiz
cart.AddItem(50)	// 50 qiymatli mahsulot qo'shamiz

// Kredit karta strategiyasini o'rnatamiz
cart.SetPaymentStrategy(&CreditCardStrategy{CardNumber: "1234-5678"})	// strategiyani kiritamiz
result1 := cart.Checkout()	// strategiyani bajaramiz
// result1: "Paid 150 using credit card 1234-5678"

// Ishlash vaqtida PayPal strategiyasiga o'tamiz
cart.SetPaymentStrategy(&PayPalStrategy{Email: "user@example.com"})	// strategiyani almashtiramiz
result2 := cart.Checkout()	// yangi strategiyani bajaramiz
// result2: "Paid 150 using PayPal user@example.com"
\`\`\``,
			hint1: `Har bir konkret strategiya (CreditCardStrategy, PayPalStrategy) summa va to'lovga xos tafsilot (karta raqami yoki email) bilan satr formatlash orqali Pay metodini amalga oshiradi. Formatlash uchun fmt.Sprintf dan foydalaning.`,
			hint2: `ShoppingCart da: AddItem items slice ga qo'shadi, SetPaymentStrategy strategiya havolasini saqlaydi. Checkout avval strategy nil ekanligini tekshiradi (xato xabarini qaytaradi), keyin items ni jamlash orqali total ni hisoblaydi va nihoyat strategy.Pay(total) ni chaqiradi.`,
			whyItMatters: `**Strategy Pattern nima uchun kerak**

Strategy siz to'lov usullarini tanlash uchun shartlardan foydalanishingiz mumkin:

\`\`\`go
// Muammo: Yangi to'lov usullarini qo'shish qiyin, Open/Closed printsipini buzadi
func (cart *Cart) Checkout(method string, details map[string]string) string {
    total := cart.calculateTotal()
    switch method {
    case "credit":
        return fmt.Sprintf("Paid %d with card %s", total, details["card"])
    case "paypal":
        return fmt.Sprintf("Paid %d via PayPal %s", total, details["email"])
    case "bitcoin":  // Yangi usul qo'shish bu funksiyani o'zgartirishni talab qiladi
        return fmt.Sprintf("Paid %d with Bitcoin %s", total, details["wallet"])
    }
    return "Unknown method"
}
\`\`\`

Strategy bilan yangi to'lov usullarini qo'shish oson:

\`\`\`go
// Yechim: Har bir to'lov usuli alohida strategiya
type PaymentStrategy interface {
    Pay(amount int) string
}

func (cart *Cart) Checkout() string {
    return cart.strategy.Pay(cart.total)  // strategiyaga delegatsiya
}

// Bitcoin qo'shish — shunchaki PaymentStrategy ni amalga oshiruvchi yangi struct
type BitcoinStrategy struct { Wallet string }
func (b *BitcoinStrategy) Pay(amount int) string { ... }
\`\`\`

**Go da Strategy ning real dunyo misollari**

1. **Siqish strategiyalari**:
\`\`\`go
type Compressor interface {
    Compress(data []byte) []byte
    GetExtension() string
}

type GzipCompressor struct{}
func (g *GzipCompressor) Compress(data []byte) []byte { /* gzip mantig'i */ }
func (g *GzipCompressor) GetExtension() string { return ".gz" }

type ZipCompressor struct{}
func (z *ZipCompressor) Compress(data []byte) []byte { /* zip mantig'i */ }
func (z *ZipCompressor) GetExtension() string { return ".zip" }

type FileArchiver struct {
    compressor Compressor
}

func (f *FileArchiver) Archive(filename string, data []byte) {
    compressed := f.compressor.Compress(data)
    os.WriteFile(filename + f.compressor.GetExtension(), compressed, 0644)
}
\`\`\`

2. **Saralash strategiyalari**:
\`\`\`go
type SortStrategy interface {
    Sort(data []int) []int
}

type QuickSort struct{}
func (q *QuickSort) Sort(data []int) []int { /* quicksort */ }

type MergeSort struct{}
func (m *MergeSort) Sort(data []int) []int { /* mergesort */ }

type DataProcessor struct {
    sorter SortStrategy
}

func (d *DataProcessor) Process(data []int) []int {
    return d.sorter.Sort(data)
}
\`\`\`

**Prodakshen pattern: Ko'p strategiyali Rate Limiter**

\`\`\`go
package main

import (
    "sync"
    "time"
)

// RateLimitStrategy tezlikni cheklash algoritmini aniqlaydi
type RateLimitStrategy interface {
    Allow(key string) bool
    GetName() string
}

// TokenBucketStrategy token bucket algoritmini amalga oshiradi
type TokenBucketStrategy struct {
    capacity   int
    refillRate int
    buckets    map[string]*bucket
    mu         sync.Mutex
}

type bucket struct {
    tokens     int
    lastRefill time.Time
}

func NewTokenBucket(capacity, refillRate int) *TokenBucketStrategy {
    return &TokenBucketStrategy{
        capacity:   capacity,
        refillRate: refillRate,
        buckets:    make(map[string]*bucket),
    }
}

func (t *TokenBucketStrategy) Allow(key string) bool {
    t.mu.Lock()
    defer t.mu.Unlock()

    b, exists := t.buckets[key]
    if !exists {
        b = &bucket{tokens: t.capacity, lastRefill: time.Now()}
        t.buckets[key] = b
    }

    // O'tgan vaqt asosida tokenlarni to'ldiramiz
    elapsed := time.Since(b.lastRefill)
    refillAmount := int(elapsed.Seconds()) * t.refillRate
    b.tokens = min(t.capacity, b.tokens + refillAmount)
    b.lastRefill = time.Now()

    if b.tokens > 0 {
        b.tokens--
        return true
    }
    return false
}

func (t *TokenBucketStrategy) GetName() string { return "TokenBucket" }

// SlidingWindowStrategy sliding window hisoblagichini amalga oshiradi
type SlidingWindowStrategy struct {
    windowSize time.Duration
    maxRequests int
    windows    map[string][]time.Time
    mu         sync.Mutex
}

func NewSlidingWindow(windowSize time.Duration, maxRequests int) *SlidingWindowStrategy {
    return &SlidingWindowStrategy{
        windowSize:  windowSize,
        maxRequests: maxRequests,
        windows:     make(map[string][]time.Time),
    }
}

func (s *SlidingWindowStrategy) Allow(key string) bool {
    s.mu.Lock()
    defer s.mu.Unlock()

    now := time.Now()
    windowStart := now.Add(-s.windowSize)

    // Muddati o'tgan vaqt belgilarini olib tashlaymiz
    var valid []time.Time
    for _, t := range s.windows[key] {
        if t.After(windowStart) {
            valid = append(valid, t)
        }
    }
    s.windows[key] = valid

    if len(valid) < s.maxRequests {
        s.windows[key] = append(s.windows[key], now)
        return true
    }
    return false
}

func (s *SlidingWindowStrategy) GetName() string { return "SlidingWindow" }

// RateLimiter strategiyalardan foydalanadigan kontekst
type RateLimiter struct {
    strategy RateLimitStrategy
}

func NewRateLimiter(strategy RateLimitStrategy) *RateLimiter {
    return &RateLimiter{strategy: strategy}
}

func (r *RateLimiter) SetStrategy(strategy RateLimitStrategy) {
    r.strategy = strategy
}

func (r *RateLimiter) AllowRequest(clientID string) bool {
    return r.strategy.Allow(clientID)
}

func min(a, b int) int {
    if a < b { return a }
    return b
}
\`\`\`

**Oldini olish kerak bo'lgan keng tarqalgan xatolar**

1. **Juda ko'p bog'liqliklarga ega strategiya** — Strategiyalar o'z-o'zini ta'minlovchi va diqqat markazida bo'lishi kerak
2. **Kontekst konkret strategiyalarni biladi** — Kontekst faqat interfeysga bog'liq bo'lishi kerak
3. **Ishlatishdan oldin strategiyani tekshirmaslik** — Chaqirishdan oldin doimo strategy nil ekanligini tekshiring
4. **Strategy ni haddan tashqari ishlatish** — Oddiy shartlar 2-3 ta belgilangan holatlar uchun yaxshiroq bo'lishi mumkin
5. **Strategiyalar o'zgaruvchan holatni ulashadi** — Har bir strategiya mustaqil bo'lishi kerak`,
			solutionCode: `package patterns

import "fmt"	// satrlarni formatlash uchun import

// PaymentStrategy to'lov algoritmi interfeysini aniqlaydi
type PaymentStrategy interface {	// strategiya interfeysi — barcha strategiyalar buni amalga oshiradi
	Pay(amount int) string	// to'lovni qayta ishlash uchun metod signaturasi
}

// CreditCardStrategy kredit karta bilan to'laydi
type CreditCardStrategy struct {	// kredit karta uchun konkret strategiya
	CardNumber string	// to'lov uchun karta raqami
}

// Pay kredit karta bilan to'lovni qayta ishlaydi
func (c *CreditCardStrategy) Pay(amount int) string {	// PaymentStrategy ni amalga oshiradi
	return fmt.Sprintf("Paid %d using credit card %s", amount, c.CardNumber)	// to'lov tasdig'ini formatlaymiz
}

// PayPalStrategy PayPal orqali to'laydi
type PayPalStrategy struct {	// PayPal uchun konkret strategiya
	Email string	// PayPal email akkaunt
}

// Pay PayPal orqali to'lovni qayta ishlaydi
func (p *PayPalStrategy) Pay(amount int) string {	// PaymentStrategy ni amalga oshiradi
	return fmt.Sprintf("Paid %d using PayPal %s", amount, p.Email)	// to'lov tasdig'ini formatlaymiz
}

// ShoppingCart to'lov strategiyasidan foydalanadi (kontekst)
type ShoppingCart struct {	// strategiyalardan foydalanadigan kontekst
	items    []int	// mahsulot narxlari ro'yxati
	strategy PaymentStrategy	// joriy to'lov strategiyasi
}

// NewShoppingCart bo'sh savat yaratadi
func NewShoppingCart() *ShoppingCart {	// kontekst uchun fabrika funksiyasi
	return &ShoppingCart{items: make([]int, 0)}	// bo'sh mahsulotlar bilan initsializatsiya qilamiz
}

// AddItem savatga mahsulot narxini qo'shadi
func (s *ShoppingCart) AddItem(price int) {	// mahsulot qo'shish metodi
	s.items = append(s.items, price)	// narxni items slice ga qo'shamiz
}

// SetPaymentStrategy to'lov strategiyasini o'rnatadi
func (s *ShoppingCart) SetPaymentStrategy(strategy PaymentStrategy) {	// strategiya kiritish
	s.strategy = strategy	// strategiya havolasini saqlaymiz
}

// Checkout joriy strategiya yordamida to'lovni qayta ishlaydi
func (s *ShoppingCart) Checkout() string {	// strategiyani bajarish
	if s.strategy == nil {	// strategiya o'rnatilganligini tekshiramiz
		return "No payment strategy set"	// strategiya yo'q bo'lsa xato xabarini qaytaramiz
	}
	total := 0	// umumiy narx uchun yig'uvchi
	for _, price := range s.items {	// barcha mahsulotlarni takrorlaymiz
		total += price	// narxlarni jamlaymiz
	}
	return s.strategy.Pay(total)	// strategiyaga delegatsiya qilamiz
}`
		}
	}
};

export default task;
