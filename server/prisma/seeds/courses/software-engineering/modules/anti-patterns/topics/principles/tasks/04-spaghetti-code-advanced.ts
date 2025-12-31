import { Task } from '../../../../../../../types';

export const task: Task = {
	slug: 'se-ap-spaghetti-code-advanced',
	title: 'Spaghetti Code Anti-pattern - Advanced',
	difficulty: 'medium',
	tags: ['go', 'anti-patterns', 'spaghetti-code', 'refactoring'],
	estimatedTime: '25m',
	isPremium: false,
	youtubeUrl: '',
	description: `Refactor complex spaghetti code with state machine pattern for clean control flow.

**The Problem:**

Complex state transitions with tangled conditionals make the code impossible to understand and maintain.

**You will refactor:**

An order status checker with nested conditionals into a clean state machine.

**Implement:**
1. **CanTransition** - Check if state transition is valid
2. **GetNextStates** - Get valid next states for current state
3. **TransitionTo** - Perform state transition

**Valid Transitions:**
- pending -> processing, cancelled
- processing -> shipped, cancelled
- shipped -> delivered
- cancelled -> (no transitions)
- delivered -> (no transitions)

**Your Task:**

Implement a clean state machine that replaces tangled conditional logic.`,
	initialCode: `package antipatterns

type OrderStatus string

)

func CanTransition(from, to OrderStatus) bool {
}

func GetNextStates(current OrderStatus) []OrderStatus {
}

type Order struct {
	ID     int
	Status OrderStatus
}

func (o *Order) TransitionTo(newStatus OrderStatus) bool {
}`,
	solutionCode: `package antipatterns

// OrderStatus represents order states
type OrderStatus string

const (
	StatusPending    OrderStatus = "pending"
	StatusProcessing OrderStatus = "processing"
	StatusShipped    OrderStatus = "shipped"
	StatusDelivered  OrderStatus = "delivered"
	StatusCancelled  OrderStatus = "cancelled"
)

// State machine: valid transitions map
// Each state maps to its allowed next states
var validTransitions = map[OrderStatus][]OrderStatus{
	StatusPending:    {StatusProcessing, StatusCancelled},	// pending can go to processing or cancelled
	StatusProcessing: {StatusShipped, StatusCancelled},		// processing can go to shipped or cancelled
	StatusShipped:    {StatusDelivered},					// shipped can only go to delivered
	StatusDelivered:  {},									// delivered is terminal state
	StatusCancelled:  {},									// cancelled is terminal state
}

// CanTransition checks if state transition is valid
// Uses state machine map for O(1) lookup
func CanTransition(from, to OrderStatus) bool {
	allowedStates, exists := validTransitions[from]	// get allowed transitions for current state
	if !exists {
		return false	// unknown state
	}

	// Check if 'to' is in the allowed states
	for _, allowed := range allowedStates {
		if allowed == to {
			return true	// valid transition
		}
	}

	return false	// transition not allowed
}

// GetNextStates returns all valid next states for current state
// Returns empty slice for terminal states
func GetNextStates(current OrderStatus) []OrderStatus {
	states, exists := validTransitions[current]
	if !exists {
		return []OrderStatus{}	// unknown state, no transitions
	}
	return states	// return allowed transitions
}

// Order represents an order with status
type Order struct {
	ID     int
	Status OrderStatus
}

// TransitionTo attempts to transition order to new status
// Only succeeds if transition is valid according to state machine
func (o *Order) TransitionTo(newStatus OrderStatus) bool {
	if !CanTransition(o.Status, newStatus) {
		return false	// invalid transition
	}

	o.Status = newStatus	// perform transition
	return true				// success
}`,
	hint1: `Create a map[OrderStatus][]OrderStatus to define valid transitions. CanTransition looks up the 'from' state and checks if 'to' is in the allowed slice.`,
	hint2: `GetNextStates simply returns validTransitions[current]. TransitionTo calls CanTransition first, and only updates o.Status if it returns true.`,
	whyItMatters: `State machines eliminate complex conditional logic and make valid transitions explicit and testable.

**Why State Machines Matter:**

**1. Replace Tangled Conditionals**

\`\`\`go
// BAD: Spaghetti state transitions
func (o *Order) UpdateStatus(newStatus string) error {
	if o.Status == "pending" {
		if newStatus == "processing" || newStatus == "cancelled" {
			o.Status = newStatus
			return nil
		} else if newStatus == "shipped" {
			return errors.New("cannot ship pending order")
		} else if newStatus == "delivered" {
			return errors.New("cannot deliver pending order")
		}
	} else if o.Status == "processing" {
		if newStatus == "shipped" || newStatus == "cancelled" {
			o.Status = newStatus
			return nil
		} else if newStatus == "pending" {
			return errors.New("cannot go back to pending")
		} else if newStatus == "delivered" {
			return errors.New("must ship before deliver")
		}
	} else if o.Status == "shipped" {
		if newStatus == "delivered" {
			o.Status = newStatus
			return nil
		} else {
			return errors.New("can only deliver shipped orders")
		}
	} else if o.Status == "delivered" {
		return errors.New("order already delivered")
	} else if o.Status == "cancelled" {
		return errors.New("order is cancelled")
	}
	return errors.New("unknown status")
}
// 30+ lines of nested ifs! Hard to verify all transitions!
\`\`\`

\`\`\`go
// GOOD: State machine - clear and declarative
var transitions = map[OrderStatus][]OrderStatus{
	Pending:    {Processing, Cancelled},
	Processing: {Shipped, Cancelled},
	Shipped:    {Delivered},
	Delivered:  {},
	Cancelled:  {},
}

func (o *Order) UpdateStatus(newStatus OrderStatus) error {
	allowed := transitions[o.Status]

	for _, s := range allowed {
		if s == newStatus {
			o.Status = newStatus
			return nil
		}
	}

	return fmt.Errorf("cannot transition from %s to %s", o.Status, newStatus)
}
// 10 lines! All transitions visible at a glance!
\`\`\`

**2. Self-Documenting Code**

\`\`\`go
// The state machine map IS the documentation!
var userStateTransitions = map[UserState][]UserState{
	Guest:      {SigningUp, SigningIn},           // guests can sign up or sign in
	SigningUp:  {Active, Guest},                  // can complete signup or cancel
	SigningIn:  {Active, Guest},                  // can complete signin or cancel
	Active:     {Suspended, Deleted, SigningOut}, // active users can be suspended, deleted, or sign out
	Suspended:  {Active, Deleted},                // suspended can be reactivated or deleted
	SigningOut: {Guest},                          // signing out goes back to guest
	Deleted:    {},                               // deleted is terminal
}
// Anyone can understand the user lifecycle by reading this!
\`\`\`

**3. Easy to Test**

\`\`\`go
func TestOrderStateTransitions(t *testing.T) {
	tests := []struct {
		from    OrderStatus
		to      OrderStatus
		allowed bool
	}{
		{Pending, Processing, true},
		{Pending, Shipped, false},      // can't skip processing
		{Processing, Cancelled, true},
		{Shipped, Pending, false},      // can't go backwards
		{Delivered, Cancelled, false},  // terminal state
		{Cancelled, Processing, false}, // terminal state
	}

	for _, tt := range tests {
		got := CanTransition(tt.from, tt.to)
		if got != tt.allowed {
			t.Errorf("CanTransition(%s, %s) = %v, want %v",
				tt.from, tt.to, got, tt.allowed)
		}
	}
}
// Table-driven test covers all transitions!
\`\`\`

**Real-World Example - Payment Processing:**

\`\`\`go
// Payment state machine
type PaymentState string

const (
	PaymentInitiated  PaymentState = "initiated"
	PaymentAuthorized PaymentState = "authorized"
	PaymentCaptured   PaymentState = "captured"
	PaymentRefunded   PaymentState = "refunded"
	PaymentFailed     PaymentState = "failed"
	PaymentExpired    PaymentState = "expired"
)

var paymentTransitions = map[PaymentState][]PaymentState{
	PaymentInitiated:  {PaymentAuthorized, PaymentFailed, PaymentExpired},
	PaymentAuthorized: {PaymentCaptured, PaymentExpired},
	PaymentCaptured:   {PaymentRefunded},
	PaymentRefunded:   {}, // terminal
	PaymentFailed:     {}, // terminal
	PaymentExpired:    {}, // terminal
}

type Payment struct {
	ID     string
	Amount float64
	State  PaymentState
}

func (p *Payment) Authorize() error {
	if !canTransition(p.State, PaymentAuthorized) {
		return fmt.Errorf("cannot authorize payment in %s state", p.State)
	}

	// Call payment gateway...
	p.State = PaymentAuthorized
	return nil
}

func (p *Payment) Capture() error {
	if !canTransition(p.State, PaymentCaptured) {
		return fmt.Errorf("cannot capture payment in %s state", p.State)
	}

	// Call payment gateway...
	p.State = PaymentCaptured
	return nil
}

func (p *Payment) Refund() error {
	if !canTransition(p.State, PaymentRefunded) {
		return fmt.Errorf("cannot refund payment in %s state", p.State)
	}

	// Call payment gateway...
	p.State = PaymentRefunded
	return nil
}
// Each transition is guarded by the state machine!
// Impossible to perform invalid transitions!
\`\`\`

**4. Visual Representation**

State machines can be easily visualized:

\`\`\`
Pending ──> Processing ──> Shipped ──> Delivered
   │            │
   └──> Cancelled <──┘
\`\`\`

**5. Adding New States is Easy**

\`\`\`go
// Need to add "returned" state? Just update the map!
var transitions = map[OrderStatus][]OrderStatus{
	Pending:    {Processing, Cancelled},
	Processing: {Shipped, Cancelled},
	Shipped:    {Delivered, Returned},  // NEW: can be returned
	Delivered:  {Returned},             // NEW: can return after delivery
	Returned:   {},                     // NEW: terminal state
	Cancelled:  {},
}
// No changes to the transition logic needed!
// The map drives everything!
\`\`\`

**Benefits:**

1. **Clarity**: All valid transitions in one place
2. **Safety**: Invalid transitions are impossible
3. **Testability**: Easy to test all transition rules
4. **Maintainability**: Add new states without touching logic
5. **Documentation**: State machine is self-documenting`,
	order: 3,
	testCode: `package antipatterns

import (
	"testing"
)

// Test1: CanTransition pending to processing
func Test1(t *testing.T) {
	if !CanTransition(StatusPending, StatusProcessing) {
		t.Error("Should allow pending -> processing")
	}
}

// Test2: CanTransition pending to cancelled
func Test2(t *testing.T) {
	if !CanTransition(StatusPending, StatusCancelled) {
		t.Error("Should allow pending -> cancelled")
	}
}

// Test3: CanTransition pending to shipped (invalid)
func Test3(t *testing.T) {
	if CanTransition(StatusPending, StatusShipped) {
		t.Error("Should not allow pending -> shipped")
	}
}

// Test4: CanTransition shipped to delivered
func Test4(t *testing.T) {
	if !CanTransition(StatusShipped, StatusDelivered) {
		t.Error("Should allow shipped -> delivered")
	}
}

// Test5: CanTransition delivered to anything (terminal)
func Test5(t *testing.T) {
	if CanTransition(StatusDelivered, StatusCancelled) {
		t.Error("Delivered should be terminal state")
	}
}

// Test6: GetNextStates for pending
func Test6(t *testing.T) {
	states := GetNextStates(StatusPending)
	if len(states) != 2 {
		t.Errorf("Pending should have 2 next states, got %d", len(states))
	}
}

// Test7: GetNextStates for cancelled (terminal)
func Test7(t *testing.T) {
	states := GetNextStates(StatusCancelled)
	if len(states) != 0 {
		t.Error("Cancelled should have no next states")
	}
}

// Test8: Order.TransitionTo valid transition
func Test8(t *testing.T) {
	order := &Order{ID: 1, Status: StatusPending}
	if !order.TransitionTo(StatusProcessing) {
		t.Error("Should succeed for valid transition")
	}
	if order.Status != StatusProcessing {
		t.Error("Status should be updated")
	}
}

// Test9: Order.TransitionTo invalid transition
func Test9(t *testing.T) {
	order := &Order{ID: 1, Status: StatusPending}
	if order.TransitionTo(StatusDelivered) {
		t.Error("Should fail for invalid transition")
	}
	if order.Status != StatusPending {
		t.Error("Status should not change on failed transition")
	}
}

// Test10: Full transition chain
func Test10(t *testing.T) {
	order := &Order{ID: 1, Status: StatusPending}
	order.TransitionTo(StatusProcessing)
	order.TransitionTo(StatusShipped)
	order.TransitionTo(StatusDelivered)
	if order.Status != StatusDelivered {
		t.Error("Should reach delivered state")
	}
}
`,
	translations: {
		ru: {
			title: 'Антипаттерн Spaghetti Code - Продвинутый',
			description: `Рефакторьте сложный спагетти-код с паттерном state machine для чистого потока управления.

**Проблема:**

Сложные переходы состояний с запутанными условиями делают код невозможным для понимания и поддержки.

**Вы выполните рефакторинг:**

Проверяльщика статуса заказа с вложенными условиями в чистый state machine.

**Реализуйте:**
1. **CanTransition** - Проверка валидности перехода состояния
2. **GetNextStates** - Получение валидных следующих состояний
3. **TransitionTo** - Выполнение перехода состояния

**Валидные переходы:**
- pending -> processing, cancelled
- processing -> shipped, cancelled
- shipped -> delivered
- cancelled -> (нет переходов)
- delivered -> (нет переходов)`,
			hint1: `Создайте map[OrderStatus][]OrderStatus для определения валидных переходов. CanTransition ищет состояние 'from' и проверяет, есть ли 'to' в разрешённом slice.`,
			hint2: `GetNextStates просто возвращает validTransitions[current]. TransitionTo сначала вызывает CanTransition и обновляет o.Status только если возвращает true.`,
			whyItMatters: `State machines устраняют сложную условную логику и делают валидные переходы явными и тестируемыми.`,
			solutionCode: `package antipatterns

type OrderStatus string

const (
	StatusPending    OrderStatus = "pending"
	StatusProcessing OrderStatus = "processing"
	StatusShipped    OrderStatus = "shipped"
	StatusDelivered  OrderStatus = "delivered"
	StatusCancelled  OrderStatus = "cancelled"
)

var validTransitions = map[OrderStatus][]OrderStatus{
	StatusPending:    {StatusProcessing, StatusCancelled},
	StatusProcessing: {StatusShipped, StatusCancelled},
	StatusShipped:    {StatusDelivered},
	StatusDelivered:  {},
	StatusCancelled:  {},
}

func CanTransition(from, to OrderStatus) bool {
	allowedStates, exists := validTransitions[from]
	if !exists {
		return false
	}

	for _, allowed := range allowedStates {
		if allowed == to {
			return true
		}
	}

	return false
}

func GetNextStates(current OrderStatus) []OrderStatus {
	states, exists := validTransitions[current]
	if !exists {
		return []OrderStatus{}
	}
	return states
}

type Order struct {
	ID     int
	Status OrderStatus
}

func (o *Order) TransitionTo(newStatus OrderStatus) bool {
	if !CanTransition(o.Status, newStatus) {
		return false
	}

	o.Status = newStatus
	return true
}`
		},
		uz: {
			title: 'Spaghetti Code Anti-pattern - Ilg\'or',
			description: `Toza boshqaruv oqimi uchun state machine patterni bilan murakkab spaghetti code ni refaktoring qiling.

**Muammo:**

Chalkash shartlarga ega murakkab holat o'tishlari kodni tushunish va qo'llab-quvvatlash uchun imkonsiz qiladi.

**Siz refaktoring qilasiz:**

Ichma-ich shartlarga ega buyurtma holati tekshiruvchisini toza state machine ga.

**Amalga oshiring:**
1. **CanTransition** - Holat o'tishining to'g'riligini tekshirish
2. **GetNextStates** - Joriy holat uchun to'g'ri keyingi holatlarni olish
3. **TransitionTo** - Holat o'tishini amalga oshirish

**To'g'ri o'tishlar:**
- pending -> processing, cancelled
- processing -> shipped, cancelled
- shipped -> delivered
- cancelled -> (o'tishlar yo'q)
- delivered -> (o'tishlar yo'q)`,
			hint1: `To'g'ri o'tishlarni aniqlash uchun map[OrderStatus][]OrderStatus yarating. CanTransition 'from' holatini qidiradi va 'to' ruxsat etilgan slice da bor-yo'qligini tekshiradi.`,
			hint2: `GetNextStates oddiy validTransitions[current] ni qaytaradi. TransitionTo avval CanTransition ni chaqiradi va faqat true qaytarsa o.Status ni yangilaydi.`,
			whyItMatters: `State machinelar murakkab shartli mantiqni yo'q qiladi va to'g'ri o'tishlarni aniq va test qilinadigan qiladi.`,
			solutionCode: `package antipatterns

type OrderStatus string

const (
	StatusPending    OrderStatus = "pending"
	StatusProcessing OrderStatus = "processing"
	StatusShipped    OrderStatus = "shipped"
	StatusDelivered  OrderStatus = "delivered"
	StatusCancelled  OrderStatus = "cancelled"
)

var validTransitions = map[OrderStatus][]OrderStatus{
	StatusPending:    {StatusProcessing, StatusCancelled},
	StatusProcessing: {StatusShipped, StatusCancelled},
	StatusShipped:    {StatusDelivered},
	StatusDelivered:  {},
	StatusCancelled:  {},
}

func CanTransition(from, to OrderStatus) bool {
	allowedStates, exists := validTransitions[from]
	if !exists {
		return false
	}

	for _, allowed := range allowedStates {
		if allowed == to {
			return true
		}
	}

	return false
}

func GetNextStates(current OrderStatus) []OrderStatus {
	states, exists := validTransitions[current]
	if !exists {
		return []OrderStatus{}
	}
	return states
}

type Order struct {
	ID     int
	Status OrderStatus
}

func (o *Order) TransitionTo(newStatus OrderStatus) bool {
	if !CanTransition(o.Status, newStatus) {
		return false
	}

	o.Status = newStatus
	return true
}`
		}
	}
};

export default task;
