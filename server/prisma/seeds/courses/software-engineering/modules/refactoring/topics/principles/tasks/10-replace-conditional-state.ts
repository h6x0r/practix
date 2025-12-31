import { Task } from '../../../../../../../types';

export const task: Task = {
	slug: 'se-refactoring-replace-conditional-state',
	title: 'Replace Conditional with Polymorphism - State Pattern',
	difficulty: 'medium',
	tags: ['refactoring', 'polymorphism', 'state-pattern', 'go'],
	estimatedTime: '30m',
	isPremium: false,
	youtubeUrl: '',
	description: `Replace state-checking conditionals with the State pattern for cleaner state management.

**You will refactor:**

1. **Document.Publish()** - Has state-checking if-else chain
2. Create **DocumentState interface** with Publish() method
3. Implement **Draft, Review, Published** state types
4. Replace conditionals with state-based behavior

**Key Concepts:**
- **State Pattern**: Object changes behavior when state changes
- **Eliminate State Checks**: Let states handle their own transitions
- **Encapsulation**: Each state encapsulates its behavior
- **Cleaner Logic**: No nested if-else for state checks

**Before Refactoring:**

\`\`\`go
func (d *Document) Publish() string {
    if d.Status == "draft" {
        d.Status = "review"
        return "Moved to review"
    }
    if d.Status == "review" {
        d.Status = "published"
        return "Published"
    }
    return "Already published"
}
\`\`\`

**After Refactoring:**

\`\`\`go
type DocumentState interface {
    Publish(doc *Document) string
}

func (d *Document) Publish() string {
    return d.State.Publish(d)
}
\`\`\`

**When to Use State Pattern:**
- Object behavior changes based on state
- Multiple state-checking conditionals
- State transitions are complex
- Same state checks in multiple methods
- States have different behavior

**Constraints:**
- Create DocumentState interface with Publish(*Document) string
- Implement Draft, Review, Published state structs
- Update Document to use State field
- Remove if-else chain, delegate to state`,
	initialCode: `package refactoring

type Document struct {
	Title  string
	Status string // "draft", "review", "published"
}

func NewDocument(title string) *Document {
	}
}

func (d *Document) Publish() string {
	}
}`,
	solutionCode: `package refactoring

// DocumentState interface defines state behavior
type DocumentState interface {
	Publish(doc *Document) string
}

// DraftState represents draft status
type DraftState struct{}

func (ds DraftState) Publish(doc *Document) string {
	doc.State = ReviewState{}		// transition to review state
	return "Sent for review"
}

// ReviewState represents review status
type ReviewState struct{}

func (rs ReviewState) Publish(doc *Document) string {
	doc.State = PublishedState{}	// transition to published state
	return "Document published successfully"
}

// PublishedState represents published status
type PublishedState struct{}

func (ps PublishedState) Publish(doc *Document) string {
	// Already published, no state change
	return "Document is already published"
}

type Document struct {
	Title string
	State DocumentState	// state object, not string
}

func NewDocument(title string) *Document {
	return &Document{
		Title: title,
		State: DraftState{},	// start in draft state
	}
}

// Publish delegates to current state - no conditionals!
func (d *Document) Publish() string {
	return d.State.Publish(d)	// state handles its own transitions
}`,
	hint1: `Define DocumentState interface with Publish(*Document) string. Create three state structs: DraftState, ReviewState, PublishedState (empty structs).`,
	hint2: `Implement Publish on each state. DraftState sets doc.State = ReviewState{}, ReviewState sets doc.State = PublishedState{}, PublishedState does nothing. Update Document struct to have State DocumentState field instead of Status string, and initialize with DraftState{} in NewDocument.`,
	whyItMatters: `State pattern eliminates complex conditionals and makes state transitions explicit and maintainable.

**Why State Pattern Matters:**

**1. Eliminate Complex State Conditionals**
Replace nested if-else with polymorphism:

\`\`\`go
// Before: Complex nested conditionals
type Order struct {
    Status string
}

func (o *Order) Cancel() error {
    if o.Status == "pending" {
        o.Status = "cancelled"
        return nil
    } else if o.Status == "processing" {
        return errors.New("cannot cancel processing order")
    } else if o.Status == "shipped" {
        return errors.New("cannot cancel shipped order")
    } else if o.Status == "delivered" {
        return errors.New("cannot cancel delivered order")
    } else if o.Status == "cancelled" {
        return errors.New("already cancelled")
    }
    return errors.New("unknown status")
}

func (o *Order) Ship() error {
    if o.Status == "pending" {
        return errors.New("cannot ship pending order")
    } else if o.Status == "processing" {
        o.Status = "shipped"
        return nil
    } else if o.Status == "shipped" {
        return errors.New("already shipped")
    }
    // ... more nested conditions
}

// After: Clean state pattern
type OrderState interface {
    Cancel(order *Order) error
    Ship(order *Order) error
}

type PendingState struct{}
func (ps PendingState) Cancel(order *Order) error {
    order.State = CancelledState{}
    return nil
}
func (ps PendingState) Ship(order *Order) error {
    return errors.New("cannot ship pending order")
}

type ProcessingState struct{}
func (ps ProcessingState) Cancel(order *Order) error {
    return errors.New("cannot cancel processing order")
}
func (ps ProcessingState) Ship(order *Order) error {
    order.State = ShippedState{}
    return nil
}

type Order struct {
    State OrderState
}

func (o *Order) Cancel() error { return o.State.Cancel(o) }
func (o *Order) Ship() error { return o.State.Ship(o) }
\`\`\`

**2. Clear State Transitions**
State changes are explicit and documented:

\`\`\`go
// Before: Hidden state transitions scattered in code
func (conn *Connection) Send(data []byte) error {
    if conn.status == "connected" {
        // send data
        if err != nil {
            conn.status = "error"  // Hidden transition!
        }
    } else if conn.status == "disconnected" {
        conn.status = "connecting"  // Another hidden transition
        // reconnect logic
    }
}

// After: Explicit state transitions
type ConnectedState struct{}
func (cs ConnectedState) Send(conn *Connection, data []byte) error {
    err := conn.socket.Write(data)
    if err != nil {
        conn.State = ErrorState{}  // Explicit transition
        return err
    }
    return nil
}

type DisconnectedState struct{}
func (ds DisconnectedState) Send(conn *Connection, data []byte) error {
    conn.State = ConnectingState{}  // Explicit transition
    return conn.reconnect()
}
\`\`\`

**3. Easier Testing**
Test each state independently:

\`\`\`go
// Before: Must test all states in one big function
func TestOrderOperations(t *testing.T) {
    order := &Order{Status: "pending"}
    err := order.Ship()
    assert.Error(t, err)  // pending can't ship

    order.Status = "processing"
    err = order.Ship()
    assert.NoError(t, err)  // processing can ship
    // Messy, testing multiple states together
}

// After: Clean, focused state tests
func TestPendingStateShip(t *testing.T) {
    state := PendingState{}
    order := &Order{}
    err := state.Ship(order)
    assert.Error(t, err)
    assert.Contains(t, err.Error(), "cannot ship pending")
}

func TestProcessingStateShip(t *testing.T) {
    state := ProcessingState{}
    order := &Order{State: state}
    err := state.Ship(order)
    assert.NoError(t, err)
    assert.IsType(t, ShippedState{}, order.State)
}
\`\`\`

**4. Add New States Easily**
Extend without modifying existing code:

\`\`\`go
// Before: Adding "on-hold" state requires modifying 5+ methods
func (o *Order) Process() error {
    if o.Status == "pending" { /* ... */ }
    else if o.Status == "processing" { /* ... */ }
    else if o.Status == "on-hold" { /* NEW: must add everywhere */ }
}

func (o *Order) Cancel() error {
    if o.Status == "pending" { /* ... */ }
    else if o.Status == "processing" { /* ... */ }
    else if o.Status == "on-hold" { /* NEW: must add here too */ }
}

// After: Just add new state struct
type OnHoldState struct{}

func (ohs OnHoldState) Process(order *Order) error {
    return errors.New("order is on hold")
}

func (ohs OnHoldState) Cancel(order *Order) error {
    order.State = CancelledState{}
    return nil
}

// No existing code needs modification!
\`\`\`

**5. Encapsulate State Behavior**
Each state knows what it can and can't do:

\`\`\`go
// Before: Validation logic duplicated
func (doc *Document) Publish() error {
    if doc.Status != "review" {
        return errors.New("can only publish from review")
    }
    doc.Status = "published"
    return nil
}

func (doc *Document) Archive() error {
    if doc.Status != "published" {
        return errors.New("can only archive published documents")
    }
    doc.Status = "archived"
    return nil
}

// After: Each state knows its capabilities
type ReviewState struct{}
func (rs ReviewState) Publish(doc *Document) error {
    doc.State = PublishedState{}  // review can publish
    return nil
}
func (rs ReviewState) Archive(doc *Document) error {
    return errors.New("cannot archive review documents")
}

type PublishedState struct{}
func (ps PublishedState) Publish(doc *Document) error {
    return errors.New("already published")
}
func (ps PublishedState) Archive(doc *Document) error {
    doc.State = ArchivedState{}  // published can archive
    return nil
}
\`\`\`

**Real-World Example - TCP Connection:**

\`\`\`go
// Before: Complex state management
type TCPConnection struct {
    state string
}

func (c *TCPConnection) Connect() error {
    if c.state == "closed" {
        c.state = "connecting"
        // connect logic
        c.state = "connected"
    } else if c.state == "connecting" {
        return errors.New("already connecting")
    } else if c.state == "connected" {
        return errors.New("already connected")
    }
}

func (c *TCPConnection) Send(data []byte) error {
    if c.state == "connected" {
        // send
    } else if c.state == "connecting" {
        return errors.New("not connected yet")
    } else if c.state == "closed" {
        return errors.New("connection closed")
    }
}

// After: State pattern
type ConnectionState interface {
    Connect(conn *TCPConnection) error
    Send(conn *TCPConnection, data []byte) error
    Close(conn *TCPConnection) error
}

type ClosedState struct{}
func (cs ClosedState) Connect(conn *TCPConnection) error {
    conn.State = ConnectingState{}
    return conn.doConnect()
}
func (cs ClosedState) Send(conn *TCPConnection, data []byte) error {
    return errors.New("connection closed")
}

type ConnectingState struct{}
func (cs ConnectingState) Connect(conn *TCPConnection) error {
    return errors.New("already connecting")
}
func (cs ConnectingState) Send(conn *TCPConnection, data []byte) error {
    return errors.New("not connected yet")
}

type ConnectedState struct{}
func (cs ConnectedState) Connect(conn *TCPConnection) error {
    return errors.New("already connected")
}
func (cs ConnectedState) Send(conn *TCPConnection, data []byte) error {
    return conn.socket.Write(data)
}

type TCPConnection struct {
    State ConnectionState
}

func (c *TCPConnection) Connect() error { return c.State.Connect(c) }
func (c *TCPConnection) Send(data []byte) error { return c.State.Send(c, data) }
\`\`\`

**Benefits:**
- Eliminates complex conditionals
- State transitions are explicit
- Easy to add new states
- Each state tested independently
- Follows Open/Closed Principle
- Code is more maintainable`,
	order: 9,
	testCode: `package refactoring

import (
	"testing"
)

// Test1: NewDocument starts in DraftState
func Test1(t *testing.T) {
	doc := NewDocument("Test")
	_, isDraft := doc.State.(DraftState)
	if !isDraft {
		t.Error("new document should start in DraftState")
	}
}

// Test2: DraftState.Publish transitions to ReviewState
func Test2(t *testing.T) {
	doc := NewDocument("Test")
	doc.Publish()
	_, isReview := doc.State.(ReviewState)
	if !isReview {
		t.Error("after publish from draft, should be in ReviewState")
	}
}

// Test3: ReviewState.Publish transitions to PublishedState
func Test3(t *testing.T) {
	doc := NewDocument("Test")
	doc.Publish() // draft -> review
	doc.Publish() // review -> published
	_, isPublished := doc.State.(PublishedState)
	if !isPublished {
		t.Error("after second publish, should be in PublishedState")
	}
}

// Test4: PublishedState.Publish stays in PublishedState
func Test4(t *testing.T) {
	doc := NewDocument("Test")
	doc.Publish() // draft -> review
	doc.Publish() // review -> published
	doc.Publish() // should stay published
	_, isPublished := doc.State.(PublishedState)
	if !isPublished {
		t.Error("after third publish, should still be in PublishedState")
	}
}

// Test5: DraftState.Publish returns correct message
func Test5(t *testing.T) {
	doc := NewDocument("Test")
	msg := doc.Publish()
	if msg != "Sent for review" {
		t.Errorf("expected 'Sent for review', got '%s'", msg)
	}
}

// Test6: ReviewState.Publish returns correct message
func Test6(t *testing.T) {
	doc := NewDocument("Test")
	doc.Publish() // draft -> review
	msg := doc.Publish()
	if msg != "Document published successfully" {
		t.Errorf("expected 'Document published successfully', got '%s'", msg)
	}
}

// Test7: PublishedState.Publish returns correct message
func Test7(t *testing.T) {
	doc := NewDocument("Test")
	doc.Publish() // draft -> review
	doc.Publish() // review -> published
	msg := doc.Publish()
	if msg != "Document is already published" {
		t.Errorf("expected 'Document is already published', got '%s'", msg)
	}
}

// Test8: Document stores title correctly
func Test8(t *testing.T) {
	doc := NewDocument("My Document")
	if doc.Title != "My Document" {
		t.Errorf("expected title 'My Document', got '%s'", doc.Title)
	}
}

// Test9: Multiple documents have independent state
func Test9(t *testing.T) {
	doc1 := NewDocument("Doc1")
	doc2 := NewDocument("Doc2")

	doc1.Publish() // doc1 -> review

	_, isDraft := doc2.State.(DraftState)
	if !isDraft {
		t.Error("doc2 should still be in DraftState")
	}
}

// Test10: Full state transition cycle
func Test10(t *testing.T) {
	doc := NewDocument("Test")

	msg1 := doc.Publish()
	msg2 := doc.Publish()
	msg3 := doc.Publish()

	if msg1 != "Sent for review" || msg2 != "Document published successfully" || msg3 != "Document is already published" {
		t.Error("full state transition messages incorrect")
	}
}
`,
	translations: {
		ru: {
			title: 'Replace Conditional with Polymorphism - Паттерн состояний',
			description: `Замените условные операторы проверки состояния паттерном State для более чистого управления состоянием.

**Вы выполните рефакторинг:**

1. **Document.Publish()** - Имеет цепочку if-else для проверки состояния
2. Создать **DocumentState интерфейс** с методом Publish()
3. Реализовать типы состояний **Draft, Review, Published**
4. Заменить условные операторы поведением на основе состояния`,
			hint1: `Определите интерфейс DocumentState с Publish(*Document) string. Создайте три структуры состояний: DraftState, ReviewState, PublishedState (пустые структуры).`,
			hint2: `Реализуйте Publish на каждом состоянии. DraftState устанавливает doc.State = ReviewState{}, ReviewState устанавливает doc.State = PublishedState{}, PublishedState ничего не делает. Обновите структуру Document, чтобы иметь поле State DocumentState вместо Status string, и инициализируйте с DraftState{} в NewDocument.`,
			whyItMatters: `Паттерн State устраняет сложные условные операторы и делает переходы состояний явными и поддерживаемыми.`
		},
		uz: {
			title: 'Replace Conditional with Polymorphism - Holat patterni',
			description: `Holat tekshiruv shartli operatorlarini toza holat boshqaruvi uchun State patterni bilan almashtiring.

**Siz refaktoring qilasiz:**

1. **Document.Publish()** - Holat tekshiruv if-else zanjiriga ega
2. Yaratish **DocumentState interfeysi** Publish() metodi bilan
3. **Draft, Review, Published** holat turlarini amalga oshirish
4. Shartli operatorlarni holat asosidagi xatti-harakat bilan almashtirish`,
			hint1: `DocumentState interfeysini Publish(*Document) string bilan aniqlang. Uchta holat strukturasini yarating: DraftState, ReviewState, PublishedState (bo'sh strukturalar).`,
			hint2: `Har bir holatda Publish ni amalga oshiring. DraftState doc.State = ReviewState{} ni o'rnatadi, ReviewState doc.State = PublishedState{} ni o'rnatadi, PublishedState hech narsa qilmaydi. Document strukturasini Status string o'rniga State DocumentState maydoniga ega bo'lishi uchun yangilang va NewDocument da DraftState{} bilan ishga tushiring.`,
			whyItMatters: `State patterni murakkab shartli operatorlarni yo'q qiladi va holat o'tishlarini aniq va qo'llab-quvvatlanadigan qiladi.`
		}
	}
};

export default task;
