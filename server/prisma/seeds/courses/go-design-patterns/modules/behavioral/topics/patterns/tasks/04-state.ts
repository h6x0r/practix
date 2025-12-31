import { Task } from '../../../../../../../types';

export const task: Task = {
	slug: 'go-dp-state',
	title: 'State Pattern',
	difficulty: 'medium',
	tags: ['go', 'design-patterns', 'behavioral', 'state'],
	estimatedTime: '35m',
	isPremium: false,
	youtubeUrl: '',
	description: `Implement the State pattern in Go - allow an object to alter its behavior when its internal state changes.

**You will implement:**

1. **State interface** - InsertCoin, SelectProduct, Dispense
2. **VendingMachine** - Context holding current state
3. **IdleState, HasCoinState, DispensingState** - Concrete states

**Example Usage:**

\`\`\`go
vm := NewVendingMachine()	// create vending machine in idle state
vm.InsertCoin()	// "Coin inserted" - transitions to HasCoinState
vm.SelectProduct()	// "Product selected" - transitions to DispensingState
vm.Dispense()	// "Dispensing product" - transitions back to IdleState

// Try invalid operations
vm2 := NewVendingMachine()	// create fresh machine
vm2.SelectProduct()	// "Please insert coin first" - invalid in IdleState
vm2.Dispense()	// "Please insert coin first" - invalid in IdleState
vm2.InsertCoin()	// "Coin inserted" - now in HasCoinState
vm2.InsertCoin()	// "Coin already inserted" - duplicate rejected
\`\`\``,
	initialCode: `package patterns

type State interface {
}

type VendingMachine struct {
	state State
}

func NewVendingMachine() *VendingMachine {
	return vm
}

func (vm *VendingMachine) SetState(state State) {
}

func (vm *VendingMachine) InsertCoin() string {
	return vm.state.InsertCoin(vm)
}

func (vm *VendingMachine) SelectProduct() string {
	return vm.state.SelectProduct(vm)
}

func (vm *VendingMachine) Dispense() string {
	return vm.state.Dispense(vm)
}

type IdleState struct{}

func (s *IdleState) InsertCoin(vm *VendingMachine) string {
}

func (s *IdleState) SelectProduct(vm *VendingMachine) string {
}

func (s *IdleState) Dispense(vm *VendingMachine) string {
}

type HasCoinState struct{}

func (s *HasCoinState) InsertCoin(vm *VendingMachine) string {
}

func (s *HasCoinState) SelectProduct(vm *VendingMachine) string {
}

func (s *HasCoinState) Dispense(vm *VendingMachine) string {
}

type DispensingState struct{}

func (s *DispensingState) InsertCoin(vm *VendingMachine) string {
}

func (s *DispensingState) SelectProduct(vm *VendingMachine) string {
}

func (s *DispensingState) Dispense(vm *VendingMachine) string {
}`,
	solutionCode: `package patterns

type State interface {	// defines all possible actions
	InsertCoin(vm *VendingMachine) string	// handle coin insertion
	SelectProduct(vm *VendingMachine) string	// handle product selection
	Dispense(vm *VendingMachine) string	// handle product dispensing
}

type VendingMachine struct {	// context that delegates to current state
	state State	// holds reference to current state object
}

func NewVendingMachine() *VendingMachine {	// factory constructor
	vm := &VendingMachine{}	// create new machine instance
	vm.state = &IdleState{}	// start in idle state waiting for coin
	return vm	// return configured machine
}

func (vm *VendingMachine) SetState(state State) {	// allows states to trigger transitions
	vm.state = state	// update current state reference
}

func (vm *VendingMachine) InsertCoin() string {	// public interface - insert coin
	return vm.state.InsertCoin(vm)	// delegate to current state's handler
}

func (vm *VendingMachine) SelectProduct() string {	// public interface - select product
	return vm.state.SelectProduct(vm)	// delegate to current state's handler
}

func (vm *VendingMachine) Dispense() string {	// public interface - dispense product
	return vm.state.Dispense(vm)	// delegate to current state's handler
}

type IdleState struct{}	// waiting for coin - initial state

func (s *IdleState) InsertCoin(vm *VendingMachine) string {	// coin inserted while idle
	vm.SetState(&HasCoinState{})	// transition to has-coin state
	return "Coin inserted"	// confirm coin accepted
}

func (s *IdleState) SelectProduct(vm *VendingMachine) string {	// selection while idle
	return "Please insert coin first"	// reject - need coin first
}

func (s *IdleState) Dispense(vm *VendingMachine) string {	// dispense while idle
	return "Please insert coin first"	// reject - need coin first
}

type HasCoinState struct{}	// coin accepted - ready for selection

func (s *HasCoinState) InsertCoin(vm *VendingMachine) string {	// another coin while has coin
	return "Coin already inserted"	// reject duplicate coin
}

func (s *HasCoinState) SelectProduct(vm *VendingMachine) string {	// product selected
	vm.SetState(&DispensingState{})	// transition to dispensing state
	return "Product selected"	// confirm selection
}

func (s *HasCoinState) Dispense(vm *VendingMachine) string {	// dispense without selection
	return "Please select product first"	// reject - need selection first
}

type DispensingState struct{}	// actively dispensing product

func (s *DispensingState) InsertCoin(vm *VendingMachine) string {	// coin during dispense
	return "Please wait, dispensing"	// reject - machine busy
}

func (s *DispensingState) SelectProduct(vm *VendingMachine) string {	// selection during dispense
	return "Please wait, dispensing"	// reject - machine busy
}

func (s *DispensingState) Dispense(vm *VendingMachine) string {	// complete dispense
	vm.SetState(&IdleState{})	// transition back to idle state
	return "Dispensing product"	// confirm product dispensed
}`,
	hint1: `**State Transition Logic:**

Each state handles the same actions differently based on what's valid in that state:

- **IdleState**: Only InsertCoin is valid - transitions to HasCoinState
- **HasCoinState**: Only SelectProduct is valid - transitions to DispensingState
- **DispensingState**: Only Dispense is valid - transitions back to IdleState

Invalid operations return error messages without changing state.

\`\`\`go
// IdleState example
func (s *IdleState) InsertCoin(vm *VendingMachine) string {
	vm.SetState(&HasCoinState{})	// change state
	return "Coin inserted"	// success message
}

func (s *IdleState) SelectProduct(vm *VendingMachine) string {
	return "Please insert coin first"	// no state change - invalid
}
\`\`\``,
	hint2: `**State Transitions with SetState:**

States trigger their own transitions using the context's SetState method:

\`\`\`go
// HasCoinState - product selected, transition to dispensing
func (s *HasCoinState) SelectProduct(vm *VendingMachine) string {
	vm.SetState(&DispensingState{})	// transition happens here
	return "Product selected"
}

// DispensingState - dispense complete, back to idle
func (s *DispensingState) Dispense(vm *VendingMachine) string {
	vm.SetState(&IdleState{})	// cycle complete, reset
	return "Dispensing product"
}
\`\`\`

The VendingMachine (context) exposes SetState so states can trigger transitions.`,
	whyItMatters: `## Why State Pattern Exists

**Problem:** Complex conditionals that change behavior based on state become unmaintainable.

\`\`\`go
// Without State - messy switch/if chains
func (vm *VendingMachine) InsertCoin() string {
	switch vm.currentState {	// state check everywhere
	case "idle":
		vm.currentState = "hasCoin"
		return "Coin inserted"
	case "hasCoin":
		return "Coin already inserted"
	case "dispensing":
		return "Please wait"
	}
	return ""
}
// Every method needs similar switch statements!
\`\`\`

**Solution:** Encapsulate state-specific behavior in state objects:

\`\`\`go
// With State - clean delegation
func (vm *VendingMachine) InsertCoin() string {
	return vm.state.InsertCoin(vm)	// state handles it
}
// Each state knows only its own behavior
\`\`\`

---

## Real-World State Machines in Go

**1. Order Processing System:**
- States: Created, Paid, Processing, Shipped, Delivered, Cancelled
- Each state has different valid operations

**2. TCP Connection:**
- States: Closed, Listen, SynSent, SynReceived, Established, FinWait, etc.
- Complex transition rules based on packets received

**3. Document Workflow:**
- States: Draft, Review, Approved, Published, Archived
- Different users can perform different actions per state

**4. Game Character:**
- States: Idle, Walking, Running, Jumping, Attacking, Dead
- Input handling varies by current state

---

## Production Pattern: Order State Machine

\`\`\`go
package order

import (
	"fmt"
	"time"
)

// OrderState defines valid operations per state
type OrderState interface {
	Pay(o *Order) error	// process payment
	Ship(o *Order) error	// ship order
	Deliver(o *Order) error	// mark delivered
	Cancel(o *Order) error	// cancel order
	String() string	// state name for logging
}

// Order is the context holding state
type Order struct {
	ID        string	// order identifier
	State     OrderState	// current state
	CreatedAt time.Time	// creation timestamp
	PaidAt    *time.Time	// payment timestamp
	ShippedAt *time.Time	// shipping timestamp
	Events    []string	// audit log
}

func NewOrder(id string) *Order {	// factory with initial state
	o := &Order{
		ID:        id,
		State:     &PendingState{},	// start in pending
		CreatedAt: time.Now(),
		Events:    []string{},
	}
	o.addEvent("Order created")
	return o
}

func (o *Order) addEvent(event string) {	// audit trail
	o.Events = append(o.Events, fmt.Sprintf("[%s] %s", time.Now().Format("15:04:05"), event))
}

func (o *Order) SetState(state OrderState) {	// state transition
	o.addEvent(fmt.Sprintf("State: %s -> %s", o.State, state))
	o.State = state
}

// Pending - waiting for payment
type PendingState struct{}

func (s *PendingState) String() string { return "Pending" }

func (s *PendingState) Pay(o *Order) error {	// payment received
	now := time.Now()
	o.PaidAt = &now
	o.SetState(&PaidState{})	// transition to paid
	o.addEvent("Payment processed")
	return nil
}

func (s *PendingState) Ship(o *Order) error {	// can't ship unpaid
	return fmt.Errorf("cannot ship: payment required")
}

func (s *PendingState) Deliver(o *Order) error {	// can't deliver
	return fmt.Errorf("cannot deliver: not shipped")
}

func (s *PendingState) Cancel(o *Order) error {	// can cancel pending
	o.SetState(&CancelledState{})
	o.addEvent("Order cancelled by customer")
	return nil
}

// Paid - payment received, ready to ship
type PaidState struct{}

func (s *PaidState) String() string { return "Paid" }

func (s *PaidState) Pay(o *Order) error {	// already paid
	return fmt.Errorf("order already paid")
}

func (s *PaidState) Ship(o *Order) error {	// ship the order
	now := time.Now()
	o.ShippedAt = &now
	o.SetState(&ShippedState{})
	o.addEvent("Order shipped")
	return nil
}

func (s *PaidState) Deliver(o *Order) error {	// can't deliver unshipped
	return fmt.Errorf("cannot deliver: not shipped yet")
}

func (s *PaidState) Cancel(o *Order) error {	// cancel with refund
	o.SetState(&CancelledState{})
	o.addEvent("Order cancelled - refund initiated")
	return nil
}

// Shipped - in transit
type ShippedState struct{}

func (s *ShippedState) String() string { return "Shipped" }

func (s *ShippedState) Pay(o *Order) error {
	return fmt.Errorf("order already paid")
}

func (s *ShippedState) Ship(o *Order) error {
	return fmt.Errorf("order already shipped")
}

func (s *ShippedState) Deliver(o *Order) error {	// delivery confirmed
	o.SetState(&DeliveredState{})
	o.addEvent("Order delivered successfully")
	return nil
}

func (s *ShippedState) Cancel(o *Order) error {	// too late to cancel
	return fmt.Errorf("cannot cancel: order in transit")
}

// Delivered - final happy state
type DeliveredState struct{}

func (s *DeliveredState) String() string { return "Delivered" }

func (s *DeliveredState) Pay(o *Order) error {
	return fmt.Errorf("order complete")
}

func (s *DeliveredState) Ship(o *Order) error {
	return fmt.Errorf("order complete")
}

func (s *DeliveredState) Deliver(o *Order) error {
	return fmt.Errorf("order already delivered")
}

func (s *DeliveredState) Cancel(o *Order) error {
	return fmt.Errorf("cannot cancel delivered order")
}

// Cancelled - final sad state
type CancelledState struct{}

func (s *CancelledState) String() string { return "Cancelled" }

func (s *CancelledState) Pay(o *Order) error {
	return fmt.Errorf("order cancelled")
}

func (s *CancelledState) Ship(o *Order) error {
	return fmt.Errorf("order cancelled")
}

func (s *CancelledState) Deliver(o *Order) error {
	return fmt.Errorf("order cancelled")
}

func (s *CancelledState) Cancel(o *Order) error {
	return fmt.Errorf("order already cancelled")
}

// Usage:
// order := NewOrder("ORD-001")
// order.State.Pay(order)      // Pending -> Paid
// order.State.Ship(order)     // Paid -> Shipped
// order.State.Deliver(order)  // Shipped -> Delivered
\`\`\`

---

## Common Mistakes to Avoid

**1. State logic in context instead of state objects:**
\`\`\`go
// Wrong - defeats the purpose
func (vm *VendingMachine) InsertCoin() string {
	if vm.stateType == "idle" {	// back to conditionals
		vm.stateType = "hasCoin"
		return "Coin inserted"
	}
	return "Error"
}

// Right - delegate to state
func (vm *VendingMachine) InsertCoin() string {
	return vm.state.InsertCoin(vm)	// state handles logic
}
\`\`\`

**2. Not validating transitions:**
\`\`\`go
// Wrong - allows invalid transitions
func (s *IdleState) Dispense(vm *VendingMachine) string {
	vm.SetState(&DispensingState{})	// invalid!
	return "Dispensing"
}

// Right - reject invalid operations
func (s *IdleState) Dispense(vm *VendingMachine) string {
	return "Please insert coin first"	// stay in current state
}
\`\`\`

**3. Creating new state instances on every transition:**
\`\`\`go
// Wasteful - creates garbage
vm.SetState(&IdleState{})	// new allocation every time

// Better for stateless states - use singletons
var (
	idleState       = &IdleState{}
	hasCoinState    = &HasCoinState{}
	dispensingState = &DispensingState{}
)

func (s *DispensingState) Dispense(vm *VendingMachine) string {
	vm.SetState(idleState)	// reuse singleton
	return "Dispensing product"
}
\`\`\``,
	order: 3,
	testCode: `package patterns

import (
	"testing"
)

// Test1: NewVendingMachine starts in IdleState
func Test1(t *testing.T) {
	vm := NewVendingMachine()
	result := vm.SelectProduct()
	if result != "Please insert coin first" {
		t.Error("Should start in idle state")
	}
}

// Test2: InsertCoin in IdleState transitions to HasCoinState
func Test2(t *testing.T) {
	vm := NewVendingMachine()
	result := vm.InsertCoin()
	if result != "Coin inserted" {
		t.Errorf("Expected 'Coin inserted', got '%s'", result)
	}
}

// Test3: SelectProduct in HasCoinState transitions to DispensingState
func Test3(t *testing.T) {
	vm := NewVendingMachine()
	vm.InsertCoin()
	result := vm.SelectProduct()
	if result != "Product selected" {
		t.Errorf("Expected 'Product selected', got '%s'", result)
	}
}

// Test4: Dispense in DispensingState transitions to IdleState
func Test4(t *testing.T) {
	vm := NewVendingMachine()
	vm.InsertCoin()
	vm.SelectProduct()
	result := vm.Dispense()
	if result != "Dispensing product" {
		t.Errorf("Expected 'Dispensing product', got '%s'", result)
	}
}

// Test5: InsertCoin in HasCoinState returns error
func Test5(t *testing.T) {
	vm := NewVendingMachine()
	vm.InsertCoin()
	result := vm.InsertCoin()
	if result != "Coin already inserted" {
		t.Error("Should reject duplicate coin")
	}
}

// Test6: Dispense in IdleState returns error
func Test6(t *testing.T) {
	vm := NewVendingMachine()
	result := vm.Dispense()
	if result != "Please insert coin first" {
		t.Error("Should require coin before dispense")
	}
}

// Test7: SelectProduct in DispensingState returns error
func Test7(t *testing.T) {
	vm := NewVendingMachine()
	vm.InsertCoin()
	vm.SelectProduct()
	result := vm.SelectProduct()
	if result != "Please wait, dispensing" {
		t.Error("Should wait during dispensing")
	}
}

// Test8: Full cycle returns to IdleState
func Test8(t *testing.T) {
	vm := NewVendingMachine()
	vm.InsertCoin()
	vm.SelectProduct()
	vm.Dispense()
	// Now should be back in IdleState
	result := vm.InsertCoin()
	if result != "Coin inserted" {
		t.Error("Should accept coin after full cycle")
	}
}

// Test9: Dispense in HasCoinState returns error
func Test9(t *testing.T) {
	vm := NewVendingMachine()
	vm.InsertCoin()
	result := vm.Dispense()
	if result != "Please select product first" {
		t.Error("Should require selection before dispense")
	}
}

// Test10: InsertCoin in DispensingState returns error
func Test10(t *testing.T) {
	vm := NewVendingMachine()
	vm.InsertCoin()
	vm.SelectProduct()
	result := vm.InsertCoin()
	if result != "Please wait, dispensing" {
		t.Error("Should wait during dispensing")
	}
}
`,
	translations: {
		ru: {
			title: 'Паттерн State (Состояние)',
			description: `Реализуйте паттерн State на Go — позвольте объекту изменять поведение при изменении внутреннего состояния.

**Вы реализуете:**

1. **Интерфейс State** - InsertCoin, SelectProduct, Dispense
2. **VendingMachine** - Контекст с текущим состоянием
3. **IdleState, HasCoinState, DispensingState** - Конкретные состояния

**Пример использования:**

\`\`\`go
vm := NewVendingMachine()	// создаём автомат в состоянии ожидания
vm.InsertCoin()	// "Coin inserted" - переход в HasCoinState
vm.SelectProduct()	// "Product selected" - переход в DispensingState
vm.Dispense()	// "Dispensing product" - возврат в IdleState

// Попытка недопустимых операций
vm2 := NewVendingMachine()	// создаём новый автомат
vm2.SelectProduct()	// "Please insert coin first" - недопустимо в IdleState
vm2.Dispense()	// "Please insert coin first" - недопустимо в IdleState
vm2.InsertCoin()	// "Coin inserted" - теперь в HasCoinState
vm2.InsertCoin()	// "Coin already inserted" - дубликат отклонён
\`\`\``,
			hint1: `**Логика переходов состояний:**

Каждое состояние обрабатывает одни и те же действия по-разному в зависимости от допустимости:

- **IdleState**: Только InsertCoin допустим - переход в HasCoinState
- **HasCoinState**: Только SelectProduct допустим - переход в DispensingState
- **DispensingState**: Только Dispense допустим - возврат в IdleState

Недопустимые операции возвращают сообщения об ошибке без изменения состояния.

\`\`\`go
// Пример IdleState
func (s *IdleState) InsertCoin(vm *VendingMachine) string {
	vm.SetState(&HasCoinState{})	// изменить состояние
	return "Coin inserted"	// сообщение об успехе
}

func (s *IdleState) SelectProduct(vm *VendingMachine) string {
	return "Please insert coin first"	// без изменения состояния - недопустимо
}
\`\`\``,
			hint2: `**Переходы состояний с SetState:**

Состояния сами инициируют переходы через метод SetState контекста:

\`\`\`go
// HasCoinState - товар выбран, переход к выдаче
func (s *HasCoinState) SelectProduct(vm *VendingMachine) string {
	vm.SetState(&DispensingState{})	// переход происходит здесь
	return "Product selected"
}

// DispensingState - выдача завершена, возврат в ожидание
func (s *DispensingState) Dispense(vm *VendingMachine) string {
	vm.SetState(&IdleState{})	// цикл завершён, сброс
	return "Dispensing product"
}
\`\`\`

VendingMachine (контекст) предоставляет SetState, чтобы состояния могли инициировать переходы.`,
			whyItMatters: `## Зачем нужен паттерн State

**Проблема:** Сложные условия, изменяющие поведение в зависимости от состояния, становятся неподдерживаемыми.

\`\`\`go
// Без State - запутанные цепочки switch/if
func (vm *VendingMachine) InsertCoin() string {
	switch vm.currentState {	// проверка состояния повсюду
	case "idle":
		vm.currentState = "hasCoin"
		return "Coin inserted"
	case "hasCoin":
		return "Coin already inserted"
	case "dispensing":
		return "Please wait"
	}
	return ""
}
// Каждый метод требует похожих switch!
\`\`\`

**Решение:** Инкапсулируйте поведение, специфичное для состояния, в объектах состояний:

\`\`\`go
// С State - чистое делегирование
func (vm *VendingMachine) InsertCoin() string {
	return vm.state.InsertCoin(vm)	// состояние обрабатывает
}
// Каждое состояние знает только своё поведение
\`\`\`

---

## Реальные конечные автоматы в Go

**1. Система обработки заказов:**
- Состояния: Created, Paid, Processing, Shipped, Delivered, Cancelled
- Каждое состояние имеет разные допустимые операции

**2. TCP-соединение:**
- Состояния: Closed, Listen, SynSent, SynReceived, Established, FinWait и др.
- Сложные правила переходов на основе полученных пакетов

**3. Документооборот:**
- Состояния: Draft, Review, Approved, Published, Archived
- Разные пользователи могут выполнять разные действия в каждом состоянии

**4. Игровой персонаж:**
- Состояния: Idle, Walking, Running, Jumping, Attacking, Dead
- Обработка ввода зависит от текущего состояния

---

## Production-паттерн: Конечный автомат заказа

\`\`\`go
package order

import (
	"fmt"
	"time"
)

// OrderState определяет допустимые операции для каждого состояния
type OrderState interface {
	Pay(o *Order) error	// обработать платёж
	Ship(o *Order) error	// отправить заказ
	Deliver(o *Order) error	// отметить доставленным
	Cancel(o *Order) error	// отменить заказ
	String() string	// имя состояния для логирования
}

// Order - контекст с состоянием
type Order struct {
	ID        string	// идентификатор заказа
	State     OrderState	// текущее состояние
	CreatedAt time.Time	// время создания
	PaidAt    *time.Time	// время оплаты
	ShippedAt *time.Time	// время отправки
	Events    []string	// журнал аудита
}

func NewOrder(id string) *Order {	// фабрика с начальным состоянием
	o := &Order{
		ID:        id,
		State:     &PendingState{},	// начинаем с ожидания
		CreatedAt: time.Now(),
		Events:    []string{},
	}
	o.addEvent("Order created")
	return o
}

func (o *Order) addEvent(event string) {	// журнал действий
	o.Events = append(o.Events, fmt.Sprintf("[%s] %s", time.Now().Format("15:04:05"), event))
}

func (o *Order) SetState(state OrderState) {	// переход состояния
	o.addEvent(fmt.Sprintf("State: %s -> %s", o.State, state))
	o.State = state
}

// Pending - ожидание оплаты
type PendingState struct{}

func (s *PendingState) String() string { return "Pending" }

func (s *PendingState) Pay(o *Order) error {	// платёж получен
	now := time.Now()
	o.PaidAt = &now
	o.SetState(&PaidState{})	// переход в оплачено
	o.addEvent("Payment processed")
	return nil
}

func (s *PendingState) Ship(o *Order) error {	// нельзя отправить неоплаченный
	return fmt.Errorf("cannot ship: payment required")
}

func (s *PendingState) Deliver(o *Order) error {	// нельзя доставить
	return fmt.Errorf("cannot deliver: not shipped")
}

func (s *PendingState) Cancel(o *Order) error {	// можно отменить ожидающий
	o.SetState(&CancelledState{})
	o.addEvent("Order cancelled by customer")
	return nil
}

// Paid - оплата получена, готов к отправке
type PaidState struct{}

func (s *PaidState) String() string { return "Paid" }

func (s *PaidState) Pay(o *Order) error {	// уже оплачен
	return fmt.Errorf("order already paid")
}

func (s *PaidState) Ship(o *Order) error {	// отправить заказ
	now := time.Now()
	o.ShippedAt = &now
	o.SetState(&ShippedState{})
	o.addEvent("Order shipped")
	return nil
}

func (s *PaidState) Deliver(o *Order) error {	// нельзя доставить неотправленный
	return fmt.Errorf("cannot deliver: not shipped yet")
}

func (s *PaidState) Cancel(o *Order) error {	// отмена с возвратом
	o.SetState(&CancelledState{})
	o.addEvent("Order cancelled - refund initiated")
	return nil
}

// Shipped - в пути
type ShippedState struct{}

func (s *ShippedState) String() string { return "Shipped" }

func (s *ShippedState) Pay(o *Order) error {
	return fmt.Errorf("order already paid")
}

func (s *ShippedState) Ship(o *Order) error {
	return fmt.Errorf("order already shipped")
}

func (s *ShippedState) Deliver(o *Order) error {	// доставка подтверждена
	o.SetState(&DeliveredState{})
	o.addEvent("Order delivered successfully")
	return nil
}

func (s *ShippedState) Cancel(o *Order) error {	// слишком поздно отменять
	return fmt.Errorf("cannot cancel: order in transit")
}

// Delivered - финальное успешное состояние
type DeliveredState struct{}

func (s *DeliveredState) String() string { return "Delivered" }

func (s *DeliveredState) Pay(o *Order) error {
	return fmt.Errorf("order complete")
}

func (s *DeliveredState) Ship(o *Order) error {
	return fmt.Errorf("order complete")
}

func (s *DeliveredState) Deliver(o *Order) error {
	return fmt.Errorf("order already delivered")
}

func (s *DeliveredState) Cancel(o *Order) error {
	return fmt.Errorf("cannot cancel delivered order")
}

// Cancelled - финальное неуспешное состояние
type CancelledState struct{}

func (s *CancelledState) String() string { return "Cancelled" }

func (s *CancelledState) Pay(o *Order) error {
	return fmt.Errorf("order cancelled")
}

func (s *CancelledState) Ship(o *Order) error {
	return fmt.Errorf("order cancelled")
}

func (s *CancelledState) Deliver(o *Order) error {
	return fmt.Errorf("order cancelled")
}

func (s *CancelledState) Cancel(o *Order) error {
	return fmt.Errorf("order already cancelled")
}

// Использование:
// order := NewOrder("ORD-001")
// order.State.Pay(order)      // Pending -> Paid
// order.State.Ship(order)     // Paid -> Shipped
// order.State.Deliver(order)  // Shipped -> Delivered
\`\`\`

---

## Распространённые ошибки

**1. Логика состояний в контексте вместо объектов состояний:**
\`\`\`go
// Неправильно - противоречит цели
func (vm *VendingMachine) InsertCoin() string {
	if vm.stateType == "idle" {	// обратно к условиям
		vm.stateType = "hasCoin"
		return "Coin inserted"
	}
	return "Error"
}

// Правильно - делегируем состоянию
func (vm *VendingMachine) InsertCoin() string {
	return vm.state.InsertCoin(vm)	// состояние обрабатывает логику
}
\`\`\`

**2. Отсутствие валидации переходов:**
\`\`\`go
// Неправильно - допускает недопустимые переходы
func (s *IdleState) Dispense(vm *VendingMachine) string {
	vm.SetState(&DispensingState{})	// недопустимо!
	return "Dispensing"
}

// Правильно - отклоняем недопустимые операции
func (s *IdleState) Dispense(vm *VendingMachine) string {
	return "Please insert coin first"	// остаёмся в текущем состоянии
}
\`\`\`

**3. Создание новых экземпляров состояний при каждом переходе:**
\`\`\`go
// Расточительно - создаёт мусор
vm.SetState(&IdleState{})	// новое выделение каждый раз

// Лучше для stateless-состояний - используйте синглтоны
var (
	idleState       = &IdleState{}
	hasCoinState    = &HasCoinState{}
	dispensingState = &DispensingState{}
)

func (s *DispensingState) Dispense(vm *VendingMachine) string {
	vm.SetState(idleState)	// переиспользуем синглтон
	return "Dispensing product"
}
\`\`\``
		},
		uz: {
			title: 'State (Holat) Pattern',
			description: `Go tilida State patternini amalga oshiring — ob'ektga ichki holati o'zgarganda xatti-harakatini o'zgartirishga ruxsat bering.

**Siz amalga oshirasiz:**

1. **State interfeysi** - InsertCoin, SelectProduct, Dispense
2. **VendingMachine** - Joriy holatni saqlovchi kontekst
3. **IdleState, HasCoinState, DispensingState** - Aniq holatlar

**Foydalanish namunasi:**

\`\`\`go
vm := NewVendingMachine()	// kutish holatida avtomat yaratish
vm.InsertCoin()	// "Coin inserted" - HasCoinState ga o'tish
vm.SelectProduct()	// "Product selected" - DispensingState ga o'tish
vm.Dispense()	// "Dispensing product" - IdleState ga qaytish

// Noto'g'ri operatsiyalarni sinash
vm2 := NewVendingMachine()	// yangi avtomat yaratish
vm2.SelectProduct()	// "Please insert coin first" - IdleState da noto'g'ri
vm2.Dispense()	// "Please insert coin first" - IdleState da noto'g'ri
vm2.InsertCoin()	// "Coin inserted" - endi HasCoinState da
vm2.InsertCoin()	// "Coin already inserted" - dublikat rad etildi
\`\`\``,
			hint1: `**Holat o'tish mantiq:**

Har bir holat bir xil harakatlarni turlicha boshqaradi:

- **IdleState**: Faqat InsertCoin to'g'ri - HasCoinState ga o'tish
- **HasCoinState**: Faqat SelectProduct to'g'ri - DispensingState ga o'tish
- **DispensingState**: Faqat Dispense to'g'ri - IdleState ga qaytish

Noto'g'ri operatsiyalar holatni o'zgartirmasdan xato xabarlarini qaytaradi.

\`\`\`go
// IdleState namunasi
func (s *IdleState) InsertCoin(vm *VendingMachine) string {
	vm.SetState(&HasCoinState{})	// holatni o'zgartirish
	return "Coin inserted"	// muvaffaqiyat xabari
}

func (s *IdleState) SelectProduct(vm *VendingMachine) string {
	return "Please insert coin first"	// holat o'zgarmadi - noto'g'ri
}
\`\`\``,
			hint2: `**SetState bilan holat o'tishlari:**

Holatlar kontekstning SetState metodidan foydalanib o'tishlarni boshlaydi:

\`\`\`go
// HasCoinState - mahsulot tanlangan, berish holatiga o'tish
func (s *HasCoinState) SelectProduct(vm *VendingMachine) string {
	vm.SetState(&DispensingState{})	// o'tish bu yerda sodir bo'ladi
	return "Product selected"
}

// DispensingState - berish tugadi, kutishga qaytish
func (s *DispensingState) Dispense(vm *VendingMachine) string {
	vm.SetState(&IdleState{})	// sikl tugadi, qayta o'rnatish
	return "Dispensing product"
}
\`\`\`

VendingMachine (kontekst) SetState ni taqdim etadi, holatlar o'tishlarni boshlay olsin.`,
			whyItMatters: `## State Pattern nima uchun kerak

**Muammo:** Holatga qarab xatti-harakatni o'zgartiruvchi murakkab shartlar qo'llab-quvvatlashga qiyin.

\`\`\`go
// State siz - chalkash switch/if zanjirlari
func (vm *VendingMachine) InsertCoin() string {
	switch vm.currentState {	// hamma joyda holat tekshiruvi
	case "idle":
		vm.currentState = "hasCoin"
		return "Coin inserted"
	case "hasCoin":
		return "Coin already inserted"
	case "dispensing":
		return "Please wait"
	}
	return ""
}
// Har bir metod shunga o'xshash switch talab qiladi!
\`\`\`

**Yechim:** Holatga xos xatti-harakatni holat ob'ektlarida inkapsulyatsiya qiling:

\`\`\`go
// State bilan - toza delegatsiya
func (vm *VendingMachine) InsertCoin() string {
	return vm.state.InsertCoin(vm)	// holat boshqaradi
}
// Har bir holat faqat o'z xatti-harakatini biladi
\`\`\`

---

## Go da haqiqiy holat mashinalari

**1. Buyurtma qayta ishlash tizimi:**
- Holatlar: Created, Paid, Processing, Shipped, Delivered, Cancelled
- Har bir holatda turli amallar ruxsat etilgan

**2. TCP ulanishi:**
- Holatlar: Closed, Listen, SynSent, SynReceived, Established, FinWait va boshqalar
- Olingan paketlarga asoslangan murakkab o'tish qoidalari

**3. Hujjat ish oqimi:**
- Holatlar: Draft, Review, Approved, Published, Archived
- Har bir holatda turli foydalanuvchilar turli amallarni bajara oladi

**4. O'yin personaji:**
- Holatlar: Idle, Walking, Running, Jumping, Attacking, Dead
- Kiritish boshqaruvi joriy holatga bog'liq

---

## Production pattern: Buyurtma holat mashinasi

\`\`\`go
package order

import (
	"fmt"
	"time"
)

// OrderState har bir holat uchun ruxsat etilgan operatsiyalarni belgilaydi
type OrderState interface {
	Pay(o *Order) error	// to'lovni qayta ishlash
	Ship(o *Order) error	// buyurtmani jo'natish
	Deliver(o *Order) error	// yetkazilgan deb belgilash
	Cancel(o *Order) error	// buyurtmani bekor qilish
	String() string	// log uchun holat nomi
}

// Order - holatni saqlovchi kontekst
type Order struct {
	ID        string	// buyurtma identifikatori
	State     OrderState	// joriy holat
	CreatedAt time.Time	// yaratilgan vaqt
	PaidAt    *time.Time	// to'langan vaqt
	ShippedAt *time.Time	// jo'natilgan vaqt
	Events    []string	// audit jurnali
}

func NewOrder(id string) *Order {	// boshlang'ich holatli fabrika
	o := &Order{
		ID:        id,
		State:     &PendingState{},	// kutishdan boshlaymiz
		CreatedAt: time.Now(),
		Events:    []string{},
	}
	o.addEvent("Order created")
	return o
}

func (o *Order) addEvent(event string) {	// harakat jurnali
	o.Events = append(o.Events, fmt.Sprintf("[%s] %s", time.Now().Format("15:04:05"), event))
}

func (o *Order) SetState(state OrderState) {	// holat o'tishi
	o.addEvent(fmt.Sprintf("State: %s -> %s", o.State, state))
	o.State = state
}

// Pending - to'lovni kutish
type PendingState struct{}

func (s *PendingState) String() string { return "Pending" }

func (s *PendingState) Pay(o *Order) error {	// to'lov qabul qilindi
	now := time.Now()
	o.PaidAt = &now
	o.SetState(&PaidState{})	// to'langan holatga o'tish
	o.addEvent("Payment processed")
	return nil
}

func (s *PendingState) Ship(o *Order) error {	// to'lanmagan jo'natib bo'lmaydi
	return fmt.Errorf("cannot ship: payment required")
}

func (s *PendingState) Deliver(o *Order) error {	// yetkazib bo'lmaydi
	return fmt.Errorf("cannot deliver: not shipped")
}

func (s *PendingState) Cancel(o *Order) error {	// kutayotganini bekor qilsa bo'ladi
	o.SetState(&CancelledState{})
	o.addEvent("Order cancelled by customer")
	return nil
}

// Paid - to'lov qabul qilindi, jo'natishga tayyor
type PaidState struct{}

func (s *PaidState) String() string { return "Paid" }

func (s *PaidState) Pay(o *Order) error {	// allaqachon to'langan
	return fmt.Errorf("order already paid")
}

func (s *PaidState) Ship(o *Order) error {	// buyurtmani jo'natish
	now := time.Now()
	o.ShippedAt = &now
	o.SetState(&ShippedState{})
	o.addEvent("Order shipped")
	return nil
}

func (s *PaidState) Deliver(o *Order) error {	// jo'natilmaganini yetkazib bo'lmaydi
	return fmt.Errorf("cannot deliver: not shipped yet")
}

func (s *PaidState) Cancel(o *Order) error {	// qaytarish bilan bekor qilish
	o.SetState(&CancelledState{})
	o.addEvent("Order cancelled - refund initiated")
	return nil
}

// Shipped - yo'lda
type ShippedState struct{}

func (s *ShippedState) String() string { return "Shipped" }

func (s *ShippedState) Pay(o *Order) error {
	return fmt.Errorf("order already paid")
}

func (s *ShippedState) Ship(o *Order) error {
	return fmt.Errorf("order already shipped")
}

func (s *ShippedState) Deliver(o *Order) error {	// yetkazish tasdiqlandi
	o.SetState(&DeliveredState{})
	o.addEvent("Order delivered successfully")
	return nil
}

func (s *ShippedState) Cancel(o *Order) error {	// bekor qilish uchun kech
	return fmt.Errorf("cannot cancel: order in transit")
}

// Delivered - yakuniy muvaffaqiyatli holat
type DeliveredState struct{}

func (s *DeliveredState) String() string { return "Delivered" }

func (s *DeliveredState) Pay(o *Order) error {
	return fmt.Errorf("order complete")
}

func (s *DeliveredState) Ship(o *Order) error {
	return fmt.Errorf("order complete")
}

func (s *DeliveredState) Deliver(o *Order) error {
	return fmt.Errorf("order already delivered")
}

func (s *DeliveredState) Cancel(o *Order) error {
	return fmt.Errorf("cannot cancel delivered order")
}

// Cancelled - yakuniy muvaffaqiyatsiz holat
type CancelledState struct{}

func (s *CancelledState) String() string { return "Cancelled" }

func (s *CancelledState) Pay(o *Order) error {
	return fmt.Errorf("order cancelled")
}

func (s *CancelledState) Ship(o *Order) error {
	return fmt.Errorf("order cancelled")
}

func (s *CancelledState) Deliver(o *Order) error {
	return fmt.Errorf("order cancelled")
}

func (s *CancelledState) Cancel(o *Order) error {
	return fmt.Errorf("order already cancelled")
}

// Foydalanish:
// order := NewOrder("ORD-001")
// order.State.Pay(order)      // Pending -> Paid
// order.State.Ship(order)     // Paid -> Shipped
// order.State.Deliver(order)  // Shipped -> Delivered
\`\`\`

---

## Keng tarqalgan xatolar

**1. Holat mantiqini kontekstda yozish holat ob'ektlari o'rniga:**
\`\`\`go
// Noto'g'ri - maqsadga zid
func (vm *VendingMachine) InsertCoin() string {
	if vm.stateType == "idle" {	// yana shartlarga qaytdik
		vm.stateType = "hasCoin"
		return "Coin inserted"
	}
	return "Error"
}

// To'g'ri - holatga delegatsiya
func (vm *VendingMachine) InsertCoin() string {
	return vm.state.InsertCoin(vm)	// holat mantiqni boshqaradi
}
\`\`\`

**2. O'tishlarni tekshirmaslik:**
\`\`\`go
// Noto'g'ri - noto'g'ri o'tishlarga ruxsat beradi
func (s *IdleState) Dispense(vm *VendingMachine) string {
	vm.SetState(&DispensingState{})	// noto'g'ri!
	return "Dispensing"
}

// To'g'ri - noto'g'ri operatsiyalarni rad etish
func (s *IdleState) Dispense(vm *VendingMachine) string {
	return "Please insert coin first"	// joriy holatda qolish
}
\`\`\`

**3. Har bir o'tishda yangi holat namunalarini yaratish:**
\`\`\`go
// Isrofgarchilik - axlat yaratadi
vm.SetState(&IdleState{})	// har safar yangi ajratish

// Stateless holatlar uchun yaxshiroq - singletonlardan foydalaning
var (
	idleState       = &IdleState{}
	hasCoinState    = &HasCoinState{}
	dispensingState = &DispensingState{}
)

func (s *DispensingState) Dispense(vm *VendingMachine) string {
	vm.SetState(idleState)	// singletonni qayta ishlatish
	return "Dispensing product"
}
\`\`\``
		}
	}
};

export default task;
