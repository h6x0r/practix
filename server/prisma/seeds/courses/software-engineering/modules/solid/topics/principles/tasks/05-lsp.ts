import { Task } from '../../../../../../../types';

export const task: Task = {
	slug: 'se-solid-lsp',
	title: 'Liskov Substitution Principle',
	difficulty: 'medium',
	tags: ['go', 'solid', 'lsp', 'polymorphism'],
	estimatedTime: '30m',
	isPremium: false,
	youtubeUrl: '',
	description: `Implement the Liskov Substitution Principle (LSP) - objects of a superclass should be replaceable with objects of its subclasses without breaking the application.

**Current Problem:**

A Bird hierarchy where Penguin breaks LSP because it can't fly, causing runtime errors when treated as a Bird.

**Your task:**

Refactor to follow LSP by creating proper abstractions:

1. **Bird interface** - Base contract for all birds (Eat method)
2. **FlyingBird interface** - Extends Bird, adds Fly capability
3. **Sparrow** - Implements both Bird and FlyingBird
4. **Eagle** - Implements both Bird and FlyingBird
5. **Penguin** - Implements only Bird (doesn't fly)

**Key Concepts:**
- **Behavioral Subtyping**: Subtypes must be behaviorally compatible
- **Contract Compliance**: Don't strengthen preconditions or weaken postconditions
- **No Surprises**: Substitution should not cause unexpected behavior

**Example Usage:**

\`\`\`go
// All birds can eat
birds := []Bird{
    &Sparrow{},
    &Eagle{},
    &Penguin{},  // OK - Penguin is a Bird
}

for _, bird := range birds {
    bird.Eat()  // Works for all birds
}

// Only flying birds can fly
flyingBirds := []FlyingBird{
    &Sparrow{},
    &Eagle{},
    // &Penguin{} would not compile - Penguin doesn't implement FlyingBird
}

for _, bird := range flyingBirds {
    bird.Fly()  // Safe - all implement Fly
}
\`\`\`

**Why LSP matters:**
- Prevents runtime errors from broken assumptions
- Enables safe polymorphism
- Makes inheritance hierarchies reliable
- Avoids unexpected behavior

**Constraints:**
- Penguin must NOT have a Fly method
- All birds must implement Bird interface
- Only flying birds implement FlyingBird interface
- Code using Bird interface should work with any bird`,
	initialCode: `package principles

import "fmt"

type Bird interface {
}

type Sparrow struct{}

func (s *Sparrow) Fly() error {
	return nil
}

func (s *Sparrow) Eat() error {
	return nil
}

type Penguin struct{}

func (p *Penguin) Fly() error {
	return fmt.Errorf("penguins cannot fly")
}

func (p *Penguin) Eat() error {
	return nil
}

func MakeBirdsFly(birds []Bird) {
	for _, bird := range birds {
		if err := bird.Fly(); err != nil {
		}
	}
}

type BirdInterface interface {
}

type FlyingBird interface {
}

type SparrowRefactored struct{}

func (s *SparrowRefactored) Eat() error {
}

func (s *SparrowRefactored) Fly() error {
}

type Eagle struct{}

func (e *Eagle) Eat() error {
}

func (e *Eagle) Fly() error {
}

type PenguinRefactored struct{}

func (p *PenguinRefactored) Eat() error {
}

func FeedBirds(birds []BirdInterface) {
}

func MakeFly(birds []FlyingBird) {
}`,
	solutionCode: `package principles

import "fmt"

// BirdInterface defines capabilities ALL birds share
// LSP compliant: all implementations can fulfill this contract
type BirdInterface interface {
	Eat() error	// every bird must eat
}

// FlyingBird interface for birds that can fly
// Separates flying capability from being a bird
// LSP compliant: only birds that actually fly implement this
type FlyingBird interface {
	BirdInterface	// compose Bird - flying birds are also birds
	Fly() error	// add flying capability
}

// Sparrow implements both interfaces - it's a bird that can fly
type SparrowRefactored struct{}

// Eat implementation for Sparrow
func (s *SparrowRefactored) Eat() error {
	fmt.Println("Sparrow eating seeds")	// sparrow-specific eating behavior
	return nil	// successful eating
}

// Fly implementation for Sparrow
func (s *SparrowRefactored) Fly() error {
	fmt.Println("Sparrow flying swiftly")	// sparrow can fly
	return nil	// successful flight
}

// Eagle implements both interfaces - it's a bird that can fly
type Eagle struct{}

// Eat implementation for Eagle
func (e *Eagle) Eat() error {
	fmt.Println("Eagle eating prey")	// eagle-specific eating behavior
	return nil	// successful eating
}

// Fly implementation for Eagle
func (e *Eagle) Fly() error {
	fmt.Println("Eagle soaring high")	// eagle can fly
	return nil	// successful flight
}

// Penguin implements ONLY BirdInterface - it's a bird that cannot fly
// LSP compliant: doesn't pretend to fly, doesn't have broken Fly method
type PenguinRefactored struct{}

// Eat implementation for Penguin
func (p *PenguinRefactored) Eat() error {
	fmt.Println("Penguin eating fish")	// penguin-specific eating behavior
	return nil	// successful eating
}

// No Fly method - penguin is Bird but NOT FlyingBird
// This is correct! Penguin doesn't implement FlyingBird interface

// FeedBirds works with any bird - LSP compliant
// Can substitute any bird type without issues
func FeedBirds(birds []BirdInterface) {
	for _, bird := range birds {	// iterate all birds
		bird.Eat()	// safe - all birds can eat
	}
}

// MakeFly works only with flying birds - LSP compliant
// Type system prevents non-flying birds from being passed
func MakeFly(birds []FlyingBird) {
	for _, bird := range birds {	// iterate flying birds
		bird.Fly()	// safe - all FlyingBirds can fly
	}
}

// Usage demonstrates LSP:
// allBirds := []BirdInterface{&SparrowRefactored{}, &Eagle{}, &PenguinRefactored{}}
// FeedBirds(allBirds)  // works with all birds
//
// flyingBirds := []FlyingBird{&SparrowRefactored{}, &Eagle{}}
// MakeFly(flyingBirds)  // works only with birds that can actually fly`,
	hint1: `For Sparrow and Eagle, implement both Eat() and Fly() methods that print messages and return nil. For Penguin, implement only Eat() - do NOT add a Fly() method.`,
	hint2: `For FeedBirds, loop through the birds slice and call bird.Eat() on each. For MakeFly, loop through the birds slice and call bird.Fly() on each. The type system ensures only flying birds are passed to MakeFly.`,
	testCode: `package principles

import "testing"

// Test1: Sparrow implements Eat
func Test1(t *testing.T) {
	sparrow := &SparrowRefactored{}
	if err := sparrow.Eat(); err != nil {
		t.Errorf("Sparrow.Eat() error: %v", err)
	}
}

// Test2: Sparrow implements Fly
func Test2(t *testing.T) {
	sparrow := &SparrowRefactored{}
	if err := sparrow.Fly(); err != nil {
		t.Errorf("Sparrow.Fly() error: %v", err)
	}
}

// Test3: Eagle implements Eat
func Test3(t *testing.T) {
	eagle := &Eagle{}
	if err := eagle.Eat(); err != nil {
		t.Errorf("Eagle.Eat() error: %v", err)
	}
}

// Test4: Eagle implements Fly
func Test4(t *testing.T) {
	eagle := &Eagle{}
	if err := eagle.Fly(); err != nil {
		t.Errorf("Eagle.Fly() error: %v", err)
	}
}

// Test5: Penguin implements Eat only
func Test5(t *testing.T) {
	penguin := &PenguinRefactored{}
	if err := penguin.Eat(); err != nil {
		t.Errorf("Penguin.Eat() error: %v", err)
	}
}

// Test6: All birds can be treated as BirdInterface
func Test6(t *testing.T) {
	birds := []BirdInterface{
		&SparrowRefactored{},
		&Eagle{},
		&PenguinRefactored{},
	}
	for _, bird := range birds {
		if err := bird.Eat(); err != nil {
			t.Errorf("bird.Eat() error: %v", err)
		}
	}
}

// Test7: FeedBirds works with all birds
func Test7(t *testing.T) {
	birds := []BirdInterface{
		&SparrowRefactored{},
		&Eagle{},
		&PenguinRefactored{},
	}
	FeedBirds(birds)
}

// Test8: MakeFly works with flying birds
func Test8(t *testing.T) {
	flyingBirds := []FlyingBird{
		&SparrowRefactored{},
		&Eagle{},
	}
	MakeFly(flyingBirds)
}

// Test9: Sparrow and Eagle implement FlyingBird
func Test9(t *testing.T) {
	var fb FlyingBird
	fb = &SparrowRefactored{}
	_ = fb
	fb = &Eagle{}
	_ = fb
}

// Test10: Empty slices work correctly
func Test10(t *testing.T) {
	FeedBirds([]BirdInterface{})
	MakeFly([]FlyingBird{})
}
`,
	whyItMatters: `The Liskov Substitution Principle prevents subtle bugs from broken inheritance hierarchies.

**Why LSP Matters:**

**1. The Classic Square-Rectangle Problem**

\`\`\`go
// VIOLATES LSP
type Rectangle struct {
	Width, Height int
}

func (r *Rectangle) SetWidth(w int) { r.Width = w }
func (r *Rectangle) SetHeight(h int) { r.Height = h }
func (r *Rectangle) Area() int { return r.Width * r.Height }

type Square struct {
	Rectangle  // Square IS-A Rectangle?
}

// Square must maintain width == height
func (s *Square) SetWidth(w int) {
	s.Width = w
	s.Height = w  // violates LSP!
}

func (s *Square) SetHeight(h int) {
	s.Width = h  // violates LSP!
	s.Height = h
}

// This code works with Rectangle but BREAKS with Square!
func TestRectangle(r *Rectangle) {
	r.SetWidth(5)
	r.SetHeight(4)
	expected := 20
	actual := r.Area()
	// expected == actual for Rectangle (20 == 20) ✓
	// expected != actual for Square (20 != 16) ✗ BREAKS LSP!
}

// FOLLOWS LSP - Separate abstractions
type Shape interface {
	Area() int
}

type RectangleShape struct {
	Width, Height int
}

func (r *RectangleShape) Area() int {
	return r.Width * r.Height
}

type SquareShape struct {
	Side int
}

func (s *SquareShape) Area() int {
	return s.Side * s.Side
}

// Works with any Shape without surprises
func PrintArea(s Shape) {
	fmt.Println(s.Area())  // safe for both Rectangle and Square
}
\`\`\`

**2. Real-World: Storage Backends**

\`\`\`go
// VIOLATES LSP
type Storage interface {
	Save(key, value string) error
	Delete(key string) error
}

type DatabaseStorage struct{}

func (ds *DatabaseStorage) Save(key, value string) error {
	// saves to database
	return nil
}

func (ds *DatabaseStorage) Delete(key string) error {
	// deletes from database
	return nil
}

type ReadOnlyStorage struct{}

func (rs *ReadOnlyStorage) Save(key, value string) error {
	// Read-only storage can't save - VIOLATES LSP!
	return fmt.Errorf("read-only storage")
}

func (rs *ReadOnlyStorage) Delete(key string) error {
	// Read-only storage can't delete - VIOLATES LSP!
	return fmt.Errorf("read-only storage")
}

// This function expects Storage to be writable
// ReadOnlyStorage breaks this assumption!
func CacheData(storage Storage, data map[string]string) error {
	for k, v := range data {
		if err := storage.Save(k, v); err != nil {
			return err  // Unexpected error for ReadOnlyStorage!
		}
	}
	return nil
}

// FOLLOWS LSP - Separate readable and writable
type ReadableStorage interface {
	Get(key string) (string, error)
}

type WritableStorage interface {
	ReadableStorage  // writable is also readable
	Save(key, value string) error
	Delete(key string) error
}

type FullStorage struct{}  // implements WritableStorage
type CachedStorage struct{}  // implements ReadableStorage only

// Functions use appropriate interface
func ReadData(storage ReadableStorage, key string) string {
	value, _ := storage.Get(key)
	return value  // works with both FullStorage and CachedStorage
}

func WriteData(storage WritableStorage, key, value string) error {
	return storage.Save(key, value)  // only accepts writable storage
}
\`\`\`

**3. Testing Benefits**

\`\`\`go
// LSP enables reliable mocking
type Logger interface {
	Log(message string) error
}

type ProductionLogger struct{}

func (pl *ProductionLogger) Log(message string) error {
	// writes to file
	return nil
}

// Test mock must follow same contract
type TestLogger struct {
	Messages []string
}

func (tl *TestLogger) Log(message string) error {
	tl.Messages = append(tl.Messages, message)
	return nil  // same return behavior as ProductionLogger
}

// Code works identically with both
func ProcessOrder(logger Logger) {
	logger.Log("Order processed")
	// No surprises whether using ProductionLogger or TestLogger
}
\`\`\`

**Signs of LSP Violations:**
- Subtype throws exceptions base type doesn't
- Subtype has empty/no-op implementations
- Subtype strengthens preconditions (requires more)
- Subtype weakens postconditions (guarantees less)
- Need type checking (if type == X) to handle subtypes
- Documentation says "don't use X method on Y subtype"`,
	order: 4,
	translations: {
		ru: {
			title: 'Принцип подстановки Барбары Лисков',
			description: `Реализуйте принцип подстановки Лисков (LSP) - объекты суперкласса должны заменяться объектами его подклассов без нарушения работы приложения.`,
			hint1: `Для Sparrow и Eagle реализуйте оба метода Eat() и Fly(), которые выводят сообщения и возвращают nil. Для Penguin реализуйте только Eat() - НЕ добавляйте метод Fly().`,
			hint2: `Для FeedBirds переберите slice birds и вызовите bird.Eat() на каждом. Для MakeFly переберите slice birds и вызовите bird.Fly() на каждом.`,
			whyItMatters: `Принцип подстановки Лисков предотвращает тонкие баги от неправильных иерархий наследования.`,
			solutionCode: `package principles

import "fmt"

type BirdInterface interface {
	Eat() error
}

type FlyingBird interface {
	BirdInterface
	Fly() error
}

type SparrowRefactored struct{}

func (s *SparrowRefactored) Eat() error {
	fmt.Println("Воробей ест зёрна")
	return nil
}

func (s *SparrowRefactored) Fly() error {
	fmt.Println("Воробей летит быстро")
	return nil
}

type Eagle struct{}

func (e *Eagle) Eat() error {
	fmt.Println("Орёл ест добычу")
	return nil
}

func (e *Eagle) Fly() error {
	fmt.Println("Орёл парит высоко")
	return nil
}

type PenguinRefactored struct{}

func (p *PenguinRefactored) Eat() error {
	fmt.Println("Пингвин ест рыбу")
	return nil
}

func FeedBirds(birds []BirdInterface) {
	for _, bird := range birds {
		bird.Eat()
	}
}

func MakeFly(birds []FlyingBird) {
	for _, bird := range birds {
		bird.Fly()
	}
}`
		},
		uz: {
			title: 'Liskov almashtirish printsipi',
			description: `Liskov almashtirish prinsipini (LSP) amalga oshiring - superklassning ob'ektlari ilovaning buzilishisiz uning subklasslari ob'ektlari bilan almashtirilishi mumkin bo'lishi kerak.`,
			hint1: `Sparrow va Eagle uchun xabarlarni chiqaruvchi va nil qaytaruvchi Eat() va Fly() metodlarini amalga oshiring. Penguin uchun faqat Eat() ni amalga oshiring - Fly() metodini qo'shmang.`,
			hint2: `FeedBirds uchun birds slice ni aylanib o'ting va har birida bird.Eat() ni chaqiring. MakeFly uchun birds slice ni aylanib o'ting va har birida bird.Fly() ni chaqiring.`,
			whyItMatters: `Liskov almashtirish printsipi noto'g'ri meros ierarxiyalaridan kelib chiqadigan nozik xatolarning oldini oladi.`,
			solutionCode: `package principles

import "fmt"

type BirdInterface interface {
	Eat() error
}

type FlyingBird interface {
	BirdInterface
	Fly() error
}

type SparrowRefactored struct{}

func (s *SparrowRefactored) Eat() error {
	fmt.Println("Chumchuq urug' yemoqda")
	return nil
}

func (s *SparrowRefactored) Fly() error {
	fmt.Println("Chumchuq tez uchmoqda")
	return nil
}

type Eagle struct{}

func (e *Eagle) Eat() error {
	fmt.Println("Burgut o'ljasini yemoqda")
	return nil
}

func (e *Eagle) Fly() error {
	fmt.Println("Burgut baland parvoz qilmoqda")
	return nil
}

type PenguinRefactored struct{}

func (p *PenguinRefactored) Eat() error {
	fmt.Println("Pingvin baliq yemoqda")
	return nil
}

func FeedBirds(birds []BirdInterface) {
	for _, bird := range birds {
		bird.Eat()
	}
}

func MakeFly(birds []FlyingBird) {
	for _, bird := range birds {
		bird.Fly()
	}
}`
		}
	}
};

export default task;
