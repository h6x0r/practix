import { Task } from '../../../../../../../types';

export const task: Task = {
	slug: 'se-solid-isp',
	title: 'Interface Segregation Principle',
	difficulty: 'medium',
	tags: ['go', 'solid', 'isp', 'interfaces'],
	estimatedTime: '30m',
	isPremium: false,
	youtubeUrl: '',
	description: `Implement the Interface Segregation Principle (ISP) - clients should not be forced to depend on interfaces they don't use.

**Current Problem:**

A Worker interface with methods that not all workers need, forcing implementations to have empty methods.

**Your task:**

Refactor to segregated interfaces where each client depends only on methods it uses:

1. **Worker interface** - Basic work capability (Work method)
2. **Eater interface** - Eating capability (Eat method)
3. **Sleeper interface** - Sleeping capability (Sleep method)
4. **Human** - Implements Worker, Eater, Sleeper
5. **Robot** - Implements only Worker (no eat/sleep)
6. **Manager** - Works with appropriate interface types

**Key Concepts:**
- **Interface Segregation**: Many specific interfaces better than one general
- **Role-based interfaces**: Interfaces represent roles/capabilities
- **No fat interfaces**: Clients see only what they need

**Example Usage:**

\`\`\`go
// Humans need all capabilities
human := &Human{Name: "John"}
human.Work()   // can work
human.Eat()    // can eat
human.Sleep()  // can sleep

// Robots only work
robot := &Robot{Model: "T-800"}
robot.Work()   // can work
// robot.Eat() doesn't exist - compile error!

// Functions use specific interfaces
workers := []Worker{human, robot}
ManageWorkers(workers)  // both can work

eaters := []Eater{human}
FeedWorkers(eaters)  // only humans need feeding
\`\`\`

**Why ISP matters:**
- Prevents forced implementations of unused methods
- Makes dependencies explicit and minimal
- Easier to understand and test
- Reduces coupling between components

**Constraints:**
- Robot must NOT implement Eater or Sleeper
- Human must implement all three interfaces
- Functions should depend on smallest interface needed`,
	initialCode: `package principles

import "fmt"

type Worker interface {
}

type Human struct {
	Name string
}

func (h *Human) Work() error {
	return nil
}

func (h *Human) Eat() error {
	return nil
}

func (h *Human) Sleep() error {
	return nil
}

type Robot struct {
	Model string
}

func (r *Robot) Work() error {
	return nil
}

func (r *Robot) Eat() error {
	return nil
}

func (r *Robot) Sleep() error {
	return nil
}

type WorkerInterface interface {
}

type Eater interface {
}

type Sleeper interface {
}

type HumanRefactored struct {
	Name string
}

func (h *HumanRefactored) Work() error {
}

func (h *HumanRefactored) Eat() error {
}

func (h *HumanRefactored) Sleep() error {
}

type RobotRefactored struct {
	Model string
}

func (r *RobotRefactored) Work() error {
}

func ManageWorkers(workers []WorkerInterface) {
}

func FeedWorkers(eaters []Eater) {
}

func RestWorkers(sleepers []Sleeper) {
}`,
	solutionCode: `package principles

import "fmt"

// WorkerInterface - segregated interface for work capability
// ISP compliant: contains only one method, one responsibility
type WorkerInterface interface {
	Work() error	// only work-related functionality
}

// Eater - segregated interface for eating capability
// ISP compliant: only implementations that eat implement this
type Eater interface {
	Eat() error	// only eating-related functionality
}

// Sleeper - segregated interface for sleeping capability
// ISP compliant: only implementations that sleep implement this
type Sleeper interface {
	Sleep() error	// only sleeping-related functionality
}

// HumanRefactored implements all three interfaces
// Humans can work, eat, and sleep
type HumanRefactored struct {
	Name string	// human's name
}

// Work implementation for humans
func (h *HumanRefactored) Work() error {
	fmt.Printf("%s is working\\n", h.Name)	// human-specific work
	return nil	// work successful
}

// Eat implementation for humans
func (h *HumanRefactored) Eat() error {
	fmt.Printf("%s is eating lunch\\n", h.Name)	// humans need food
	return nil	// eating successful
}

// Sleep implementation for humans
func (h *HumanRefactored) Sleep() error {
	fmt.Printf("%s is sleeping\\n", h.Name)	// humans need rest
	return nil	// sleeping successful
}

// RobotRefactored implements ONLY WorkerInterface
// ISP compliant: doesn't implement unused methods
type RobotRefactored struct {
	Model string	// robot model identifier
}

// Work implementation for robots
func (r *RobotRefactored) Work() error {
	fmt.Printf("Robot %s is working 24/7\\n", r.Model)	// robots work continuously
	return nil	// work successful
}

// No Eat method - robots don't eat!
// No Sleep method - robots don't sleep!
// ISP compliant: only implements what it actually does

// ManageWorkers works with any WorkerInterface
// ISP compliant: depends only on Work capability
func ManageWorkers(workers []WorkerInterface) {
	for _, worker := range workers {	// iterate all workers
		worker.Work()	// call Work - all WorkerInterface implementations have this
	}
}

// FeedWorkers works only with Eater interface
// ISP compliant: only feeds things that can eat
func FeedWorkers(eaters []Eater) {
	for _, eater := range eaters {	// iterate all eaters
		eater.Eat()	// call Eat - only Eater implementations have this
	}
}

// RestWorkers works only with Sleeper interface
// ISP compliant: only rests things that can sleep
func RestWorkers(sleepers []Sleeper) {
	for _, sleeper := range sleepers {	// iterate all sleepers
		sleeper.Sleep()	// call Sleep - only Sleeper implementations have this
	}
}

// Usage demonstrates ISP:
// human := &HumanRefactored{Name: "Alice"}
// robot := &RobotRefactored{Model: "T-800"}
//
// ManageWorkers([]WorkerInterface{human, robot})  // both can work
// FeedWorkers([]Eater{human})  // only human needs food
// RestWorkers([]Sleeper{human})  // only human needs sleep
//
// Trying to pass robot to FeedWorkers or RestWorkers = COMPILE ERROR
// Type system enforces ISP!`,
	hint1: `For Human, implement all three methods (Work, Eat, Sleep) that print messages with h.Name and return nil. For Robot, implement only Work that prints a message with r.Model and returns nil.`,
	hint2: `For ManageWorkers, loop through workers and call worker.Work() on each. For FeedWorkers, loop through eaters and call eater.Eat(). For RestWorkers, loop through sleepers and call sleeper.Sleep().`,
	testCode: `package principles

import "testing"

// Test1: Human implements Work
func Test1(t *testing.T) {
	h := &HumanRefactored{Name: "John"}
	if err := h.Work(); err != nil {
		t.Errorf("Human.Work() error: %v", err)
	}
}

// Test2: Human implements Eat
func Test2(t *testing.T) {
	h := &HumanRefactored{Name: "John"}
	if err := h.Eat(); err != nil {
		t.Errorf("Human.Eat() error: %v", err)
	}
}

// Test3: Human implements Sleep
func Test3(t *testing.T) {
	h := &HumanRefactored{Name: "John"}
	if err := h.Sleep(); err != nil {
		t.Errorf("Human.Sleep() error: %v", err)
	}
}

// Test4: Robot implements Work only
func Test4(t *testing.T) {
	r := &RobotRefactored{Model: "T-800"}
	if err := r.Work(); err != nil {
		t.Errorf("Robot.Work() error: %v", err)
	}
}

// Test5: ManageWorkers accepts both Human and Robot
func Test5(t *testing.T) {
	workers := []WorkerInterface{
		&HumanRefactored{Name: "Alice"},
		&RobotRefactored{Model: "R2D2"},
	}
	ManageWorkers(workers)
}

// Test6: FeedWorkers accepts only Eaters
func Test6(t *testing.T) {
	eaters := []Eater{
		&HumanRefactored{Name: "Bob"},
	}
	FeedWorkers(eaters)
}

// Test7: RestWorkers accepts only Sleepers
func Test7(t *testing.T) {
	sleepers := []Sleeper{
		&HumanRefactored{Name: "Charlie"},
	}
	RestWorkers(sleepers)
}

// Test8: Human implements all three interfaces
func Test8(t *testing.T) {
	h := &HumanRefactored{Name: "Test"}
	var _ WorkerInterface = h
	var _ Eater = h
	var _ Sleeper = h
}

// Test9: Robot implements only WorkerInterface
func Test9(t *testing.T) {
	r := &RobotRefactored{Model: "Test"}
	var _ WorkerInterface = r
}

// Test10: Empty slices work
func Test10(t *testing.T) {
	ManageWorkers([]WorkerInterface{})
	FeedWorkers([]Eater{})
	RestWorkers([]Sleeper{})
}
`,
	whyItMatters: `The Interface Segregation Principle prevents clients from depending on methods they don't use.

**Why ISP Matters:**

**1. The Problem with Fat Interfaces**

\`\`\`go
// VIOLATES ISP - fat interface
type Document interface {
	Open() error
	Close() error
	Save() error
	Print() error
	Fax() error      // Not all documents can be faxed!
	Email() error    // Not all documents can be emailed!
	Encrypt() error  // Not all documents need encryption!
}

// PDF forced to implement everything
type PDFDocument struct{}

func (p *PDFDocument) Open() error { return nil }
func (p *PDFDocument) Close() error { return nil }
func (p *PDFDocument) Save() error { return nil }
func (p *PDFDocument) Print() error { return nil }
func (p *PDFDocument) Fax() error { return nil }
func (p *PDFDocument) Email() error { return nil }
func (p *PDFDocument) Encrypt() error { return nil }

// ReadOnlyDocument can't save/email/fax but forced to implement!
type ReadOnlyDocument struct{}

func (r *ReadOnlyDocument) Open() error { return nil }
func (r *ReadOnlyDocument) Close() error { return nil }
func (r *ReadOnlyDocument) Save() error {
	return fmt.Errorf("read-only")  // forced to implement
}
func (r *ReadOnlyDocument) Print() error { return nil }
func (r *ReadOnlyDocument) Fax() error {
	return fmt.Errorf("cannot fax")  // forced to implement
}
func (r *ReadOnlyDocument) Email() error {
	return fmt.Errorf("cannot email")  // forced to implement
}
func (r *ReadOnlyDocument) Encrypt() error {
	return fmt.Errorf("cannot encrypt")  // forced to implement
}

// FOLLOWS ISP - segregated interfaces
type Readable interface {
	Open() error
	Close() error
}

type Writable interface {
	Save() error
}

type Printable interface {
	Print() error
}

type Faxable interface {
	Fax() error
}

type Emailable interface {
	Email() error
}

type Encryptable interface {
	Encrypt() error
}

// PDF implements all capabilities
type PDFDoc struct{}

func (p *PDFDoc) Open() error { return nil }
func (p *PDFDoc) Close() error { return nil }
func (p *PDFDoc) Save() error { return nil }
func (p *PDFDoc) Print() error { return nil }
func (p *PDFDoc) Fax() error { return nil }
func (p *PDFDoc) Email() error { return nil }
func (p *PDFDoc) Encrypt() error { return nil }

// ReadOnly implements only what it can do
type ReadOnlyDoc struct{}

func (r *ReadOnlyDoc) Open() error { return nil }
func (r *ReadOnlyDoc) Close() error { return nil }
func (r *ReadOnlyDoc) Print() error { return nil }

// No Save, Fax, Email, Encrypt - clean!
\`\`\`

**2. Real-World: Database Connections**

\`\`\`go
// VIOLATES ISP
type Database interface {
	Connect() error
	Disconnect() error
	Query(sql string) ([]Row, error)
	Execute(sql string) error
	BeginTransaction() error
	Commit() error
	Rollback() error
	Backup() error        // Not all DBs support backup!
	Replicate() error     // Not all DBs support replication!
}

// FOLLOWS ISP
type Connector interface {
	Connect() error
	Disconnect() error
}

type Querier interface {
	Query(sql string) ([]Row, error)
}

type Executor interface {
	Execute(sql string) error
}

type Transactional interface {
	BeginTransaction() error
	Commit() error
	Rollback() error
}

type Backupable interface {
	Backup() error
}

type Replicatable interface {
	Replicate() error
}

// PostgreSQL implements all
type PostgreSQL struct{}
// implements Connector, Querier, Executor, Transactional, Backupable, Replicatable

// SQLite implements subset
type SQLite struct{}
// implements Connector, Querier, Executor, Transactional, Backupable
// doesn't implement Replicatable - no forced empty methods!
\`\`\`

**3. Testing Benefits**

\`\`\`go
// WITH ISP - easy to mock only what's needed
type Saver interface {
	Save(data string) error
}

type MockSaver struct {
	SaveCalled bool
}

func (m *MockSaver) Save(data string) error {
	m.SaveCalled = true
	return nil
}

func TestBackup(t *testing.T) {
	mock := &MockSaver{}
	service := &BackupService{Storage: mock}

	service.Backup("data")

	if !mock.SaveCalled {
		t.Error("Save was not called")
	}
	// Simple mock - only implements what's needed
}
\`\`\`

**Signs of ISP Violations:**
- Empty method implementations
- Methods that return errors like "not supported"
- Methods with panic("not implemented")
- Comments saying "doesn't apply to this type"
- Large interfaces with many methods
- Client code uses only subset of interface`,
	order: 6,
	translations: {
		ru: {
			title: 'Принцип разделения интерфейсов',
			description: `Реализуйте принцип разделения интерфейсов (ISP) - клиенты не должны зависеть от интерфейсов, которые они не используют.`,
			hint1: `Для Human реализуйте все три метода (Work, Eat, Sleep), которые выводят сообщения с h.Name и возвращают nil. Для Robot реализуйте только Work.`,
			hint2: `Для ManageWorkers переберите workers и вызовите worker.Work() на каждом. Для FeedWorkers переберите eaters и вызовите eater.Eat().`,
			whyItMatters: `Принцип разделения интерфейсов предотвращает зависимость клиентов от методов, которые они не используют.`,
			solutionCode: `package principles

import "fmt"

type WorkerInterface interface {
	Work() error
}

type Eater interface {
	Eat() error
}

type Sleeper interface {
	Sleep() error
}

type HumanRefactored struct {
	Name string
}

func (h *HumanRefactored) Work() error {
	fmt.Printf("%s работает\\n", h.Name)
	return nil
}

func (h *HumanRefactored) Eat() error {
	fmt.Printf("%s обедает\\n", h.Name)
	return nil
}

func (h *HumanRefactored) Sleep() error {
	fmt.Printf("%s спит\\n", h.Name)
	return nil
}

type RobotRefactored struct {
	Model string
}

func (r *RobotRefactored) Work() error {
	fmt.Printf("Робот %s работает 24/7\\n", r.Model)
	return nil
}

func ManageWorkers(workers []WorkerInterface) {
	for _, worker := range workers {
		worker.Work()
	}
}

func FeedWorkers(eaters []Eater) {
	for _, eater := range eaters {
		eater.Eat()
	}
}

func RestWorkers(sleepers []Sleeper) {
	for _, sleeper := range sleepers {
		sleeper.Sleep()
	}
}`
		},
		uz: {
			title: 'Interfeys ajratish printsipi',
			description: `Interfeys ajratish prinsipini (ISP) amalga oshiring - mijozlar ular ishlatmaydigan interfeysga bog'liq bo'lmasligi kerak.`,
			hint1: `Human uchun h.Name bilan xabarlarni chiqaruvchi va nil qaytaruvchi barcha uchta metodlarni (Work, Eat, Sleep) amalga oshiring. Robot uchun faqat Work ni amalga oshiring.`,
			hint2: `ManageWorkers uchun workers ni aylanib o'ting va har birida worker.Work() ni chaqiring. FeedWorkers uchun eaters ni aylanib o'ting va eater.Eat() ni chaqiring.`,
			whyItMatters: `Interfeys ajratish printsipi mijozlarning ular ishlatmaydigan metodlarga bog'liqligini oldini oladi.`,
			solutionCode: `package principles

import "fmt"

type WorkerInterface interface {
	Work() error
}

type Eater interface {
	Eat() error
}

type Sleeper interface {
	Sleep() error
}

type HumanRefactored struct {
	Name string
}

func (h *HumanRefactored) Work() error {
	fmt.Printf("%s ishlayapti\\n", h.Name)
	return nil
}

func (h *HumanRefactored) Eat() error {
	fmt.Printf("%s ovqatlanmoqda\\n", h.Name)
	return nil
}

func (h *HumanRefactored) Sleep() error {
	fmt.Printf("%s uxlamoqda\\n", h.Name)
	return nil
}

type RobotRefactored struct {
	Model string
}

func (r *RobotRefactored) Work() error {
	fmt.Printf("Robot %s 24/7 ishlayapti\\n", r.Model)
	return nil
}

func ManageWorkers(workers []WorkerInterface) {
	for _, worker := range workers {
		worker.Work()
	}
}

func FeedWorkers(eaters []Eater) {
	for _, eater := range eaters {
		eater.Eat()
	}
}

func RestWorkers(sleepers []Sleeper) {
	for _, sleeper := range sleepers {
		sleeper.Sleep()
	}
}`
		}
	}
};

export default task;
