import { Task } from '../../../../../../../types';

export const task: Task = {
	slug: 'go-dp-composite',
	title: 'Composite Pattern',
	difficulty: 'medium',
	tags: ['go', 'design-patterns', 'structural', 'composite'],
	estimatedTime: '35m',
	isPremium: false,
	youtubeUrl: '',
	description: `Implement the Composite pattern in Go - compose objects into tree structures to represent part-whole hierarchies.

The Composite pattern lets clients treat individual objects and compositions of objects uniformly. This is essential for building tree-like structures where both simple elements (leaves) and complex containers (composites) share a common interface.

**You will implement:**

1. **Component interface** - Common interface for leaves and composites
2. **File struct** - Leaf node with name and size
3. **Directory struct** - Composite node containing children

**Example Usage:**

\`\`\`go
// Create file system structure
root := NewDirectory("root")	// composite node
docs := NewDirectory("docs")	// nested composite
file1 := &File{Name: "readme.txt", Size: 100}	// leaf node
file2 := &File{Name: "notes.txt", Size: 50}	// leaf node

// Build tree structure
docs.Add(file2)	// add leaf to composite
root.Add(docs)	// add composite to composite
root.Add(file1)	// add leaf to composite

// Uniform operations on any Component
root.GetSize()	// returns 150 (recursive sum)
docs.GetSize()	// returns 50
file1.GetSize()	// returns 100

// Print entire tree structure
root.Print("")
// Output:
// Directory: root
//   Directory: docs
//     File: notes.txt (50 bytes)
//   File: readme.txt (100 bytes)
\`\`\``,
	initialCode: `package patterns

import "fmt"

type Component interface {
}

type File struct {
	Name string
	Size int
}

func (f *File) GetName() string {
}

func (f *File) GetSize() int {
}

func (f *File) Print(indent string) {
}

type Directory struct {
	name     string
}

func NewDirectory(name string) *Directory {
}

func (d *Directory) GetName() string {
}

func (d *Directory) GetSize() int {
}

func (d *Directory) Print(indent string) {
}

func (d *Directory) Add(component Component) {
}

func (d *Directory) Remove(name string) {
}`,
	solutionCode: `package patterns

import "fmt"

// Component interface for both files and directories
type Component interface {	// common interface for uniform treatment
	GetName() string	// returns element name
	GetSize() int	// returns size (leaf) or calculated size (composite)
	Print(indent string)	// displays element with indentation
}

// File is a leaf component
type File struct {	// leaf - cannot contain other components
	Name string	// file name
	Size int	// file size in bytes
}

// GetName returns file name
func (f *File) GetName() string {	// implements Component interface
	return f.Name	// simply return the name field
}

// GetSize returns file size
func (f *File) GetSize() int {	// implements Component interface
	return f.Size	// leaf returns its own size directly
}

// Print displays file info
func (f *File) Print(indent string) {	// implements Component interface
	fmt.Printf("%sFile: %s (%d bytes)\\n", indent, f.Name, f.Size)	// formatted output with indentation
}

// Directory is a composite component
type Directory struct {	// composite - can contain other components
	name     string	// directory name
	children []Component	// slice of child components (files or directories)
}

// NewDirectory creates a new directory
func NewDirectory(name string) *Directory {	// factory function for directories
	return &Directory{name: name, children: make([]Component, 0)}	// initialize with empty children slice
}

// GetName returns directory name
func (d *Directory) GetName() string {	// implements Component interface
	return d.name	// return directory name
}

// GetSize returns sum of all children sizes
func (d *Directory) GetSize() int {	// implements Component interface - recursive calculation
	total := 0	// accumulator for total size
	for _, child := range d.children {	// iterate over all children
		total += child.GetSize()	// recursively get size (works for both files and directories)
	}
	return total	// return calculated total
}

// Print displays directory and all children
func (d *Directory) Print(indent string) {	// implements Component interface - recursive display
	fmt.Printf("%sDirectory: %s\\n", indent, d.name)	// print directory with current indent
	for _, child := range d.children {	// iterate over all children
		child.Print(indent + "  ")	// recurse with increased indent for visual hierarchy
	}
}

// Add adds a component to directory
func (d *Directory) Add(component Component) {	// composite-specific method
	d.children = append(d.children, component)	// append to children slice
}

// Remove removes a component by name
func (d *Directory) Remove(name string) {	// composite-specific method
	for i, child := range d.children {	// search through children
		if child.GetName() == name {	// found matching component
			d.children = append(d.children[:i], d.children[i+1:]...)	// remove by slicing
			return	// exit after first removal
		}
	}
}`,
	hint1: `File methods are simple:
- GetName() returns the Name field directly
- GetSize() returns the Size field directly
- Print() uses fmt.Printf with format "%sFile: %s (%d bytes)\\n" where %s is indent, %s is name, %d is size

The key insight is that File (leaf) contains actual data, while Directory (composite) delegates to children.`,
	hint2: `Directory methods use recursion through the Component interface:
- GetSize() iterates d.children and sums child.GetSize() - polymorphism handles both files and subdirectories
- Print() outputs directory name, then calls child.Print(indent + "  ") for each child
- Add() uses append to add component to children slice
- Remove() finds component by name and removes using slice append trick: d.children[:i], d.children[i+1:]...`,
	whyItMatters: `## Why Composite Exists

The Composite pattern solves the problem of treating individual objects and groups of objects uniformly. Without it, client code must distinguish between simple elements and containers, leading to complex conditional logic.

**Problem - Without Composite:**
\`\`\`go
// Client code must know about every type
func calculateSize(item interface{}) int {
    switch v := item.(type) {
    case *File:
        return v.Size
    case *Directory:
        total := 0
        for _, child := range v.Children {
            total += calculateSize(child)  // type switch again
        }
        return total
    default:
        return 0
    }
}
\`\`\`

**Solution - With Composite:**
\`\`\`go
// Uniform interface - client doesn't care about type
func calculateSize(c Component) int {
    return c.GetSize()  // polymorphism handles recursion
}
\`\`\`

## Real-World Go Examples

**1. UI Widget Tree:**
\`\`\`go
type Widget interface {
    Render(canvas *Canvas)
    GetBounds() Rectangle
}

type Button struct {
    text   string
    bounds Rectangle
}

func (b *Button) Render(canvas *Canvas) {
    canvas.DrawButton(b.bounds, b.text)
}

type Panel struct {
    children []Widget
    bounds   Rectangle
}

func (p *Panel) Render(canvas *Canvas) {
    canvas.DrawPanel(p.bounds)
    for _, child := range p.children {
        child.Render(canvas)  // recursive rendering
    }
}
\`\`\`

**2. Expression Tree (Calculator):**
\`\`\`go
type Expression interface {
    Evaluate() float64
}

type Number struct{ value float64 }

func (n *Number) Evaluate() float64 { return n.value }

type Addition struct {
    left, right Expression
}

func (a *Addition) Evaluate() float64 {
    return a.left.Evaluate() + a.right.Evaluate()
}

// Usage: (3 + 4) + 5
expr := &Addition{
    left:  &Addition{left: &Number{3}, right: &Number{4}},
    right: &Number{5},
}
result := expr.Evaluate()  // returns 12
\`\`\`

## Production Pattern: Organization Chart

\`\`\`go
type Employee interface {
    GetName() string
    GetSalary() float64
    GetSubordinates() []Employee
    Accept(visitor EmployeeVisitor)  // visitor support
}

type Developer struct {
    name   string
    salary float64
}

func (d *Developer) GetName() string           { return d.name }
func (d *Developer) GetSalary() float64        { return d.salary }
func (d *Developer) GetSubordinates() []Employee { return nil }
func (d *Developer) Accept(v EmployeeVisitor)  { v.VisitDeveloper(d) }

type Manager struct {
    name         string
    salary       float64
    subordinates []Employee
}

func (m *Manager) GetName() string             { return m.name }
func (m *Manager) GetSalary() float64          { return m.salary }
func (m *Manager) GetSubordinates() []Employee { return m.subordinates }
func (m *Manager) Accept(v EmployeeVisitor)    { v.VisitManager(m) }

func (m *Manager) AddSubordinate(e Employee) {
    m.subordinates = append(m.subordinates, e)
}

// Calculate total salary of a team
func GetTotalSalary(e Employee) float64 {
    total := e.GetSalary()
    for _, sub := range e.GetSubordinates() {
        total += GetTotalSalary(sub)  // recursive calculation
    }
    return total
}

// Count all employees under a manager
func CountEmployees(e Employee) int {
    count := 1  // count self
    for _, sub := range e.GetSubordinates() {
        count += CountEmployees(sub)
    }
    return count
}
\`\`\`

## Common Mistakes

**1. Breaking Type Safety:**
\`\`\`go
// Bad - using interface{} loses type safety
type BadComponent interface {
    Add(child interface{})  // accepts anything
}

// Good - strongly typed children
type Component interface {
    Add(child Component)  // only accepts Components
}
\`\`\`

**2. Adding Composite Methods to Leaf:**
\`\`\`go
// Bad - leaf has Add/Remove that don't make sense
type File struct { /* ... */ }
func (f *File) Add(c Component) {
    panic("files can't have children")  // runtime error
}

// Good - only composite has child management
type Directory struct {
    children []Component
}
func (d *Directory) Add(c Component) {
    d.children = append(d.children, c)
}
// File doesn't have Add method at all
\`\`\`

**3. Infinite Recursion (Adding Parent to Child):**
\`\`\`go
// Dangerous - can create cycles
parent := NewDirectory("parent")
child := NewDirectory("child")
parent.Add(child)
child.Add(parent)  // creates infinite loop in GetSize()!

// Solution - check for cycles before adding
func (d *Directory) Add(c Component) error {
    if d.containsAncestor(c) {
        return errors.New("cannot add ancestor as child")
    }
    d.children = append(d.children, c)
    return nil
}
\`\`\`

**Key Principles:**
- Component interface should be minimal - only operations meaningful for both leaves and composites
- Composite holds and delegates to children; leaves do the actual work
- Client code should work with Component interface, not concrete types
- Consider cycle detection for mutable tree structures`,
	order: 2,
	testCode: `package patterns

import (
	"testing"
)

// Test1: File.GetName returns name
func Test1(t *testing.T) {
	f := &File{Name: "test.txt", Size: 100}
	if f.GetName() != "test.txt" {
		t.Error("File.GetName should return name")
	}
}

// Test2: File.GetSize returns size
func Test2(t *testing.T) {
	f := &File{Name: "test.txt", Size: 100}
	if f.GetSize() != 100 {
		t.Error("File.GetSize should return size")
	}
}

// Test3: Directory.GetName returns name
func Test3(t *testing.T) {
	d := NewDirectory("docs")
	if d.GetName() != "docs" {
		t.Error("Directory.GetName should return name")
	}
}

// Test4: Directory.Add adds component
func Test4(t *testing.T) {
	d := NewDirectory("docs")
	f := &File{Name: "test.txt", Size: 100}
	d.Add(f)
	if d.GetSize() != 100 {
		t.Error("Directory should contain added file")
	}
}

// Test5: Directory.GetSize sums children
func Test5(t *testing.T) {
	d := NewDirectory("docs")
	d.Add(&File{Name: "a.txt", Size: 50})
	d.Add(&File{Name: "b.txt", Size: 30})
	if d.GetSize() != 80 {
		t.Error("Directory.GetSize should sum children sizes")
	}
}

// Test6: Nested directories calculate size recursively
func Test6(t *testing.T) {
	root := NewDirectory("root")
	child := NewDirectory("child")
	child.Add(&File{Name: "a.txt", Size: 100})
	root.Add(child)
	root.Add(&File{Name: "b.txt", Size: 50})
	if root.GetSize() != 150 {
		t.Error("Nested directory size should be 150")
	}
}

// Test7: Directory.Remove removes by name
func Test7(t *testing.T) {
	d := NewDirectory("docs")
	d.Add(&File{Name: "a.txt", Size: 50})
	d.Add(&File{Name: "b.txt", Size: 30})
	d.Remove("a.txt")
	if d.GetSize() != 30 {
		t.Error("Remove should remove file by name")
	}
}

// Test8: File implements Component interface
func Test8(t *testing.T) {
	var c Component = &File{Name: "test.txt", Size: 10}
	if c.GetName() != "test.txt" {
		t.Error("File should implement Component")
	}
}

// Test9: Directory implements Component interface
func Test9(t *testing.T) {
	var c Component = NewDirectory("test")
	if c.GetName() != "test" {
		t.Error("Directory should implement Component")
	}
}

// Test10: Empty directory has size 0
func Test10(t *testing.T) {
	d := NewDirectory("empty")
	if d.GetSize() != 0 {
		t.Error("Empty directory should have size 0")
	}
}
`,
	translations: {
		ru: {
			title: 'Паттерн Composite (Компоновщик)',
			description: `Реализуйте паттерн Composite на Go — компонуйте объекты в древовидные структуры для представления иерархий часть-целое.

Паттерн Composite позволяет клиентам единообразно работать с отдельными объектами и композициями объектов. Это важно для построения древовидных структур, где и простые элементы (листья), и сложные контейнеры (композиты) имеют общий интерфейс.

**Вы реализуете:**

1. **Интерфейс Component** - Общий интерфейс для листьев и композитов
2. **Структура File** - Листовой узел с именем и размером
3. **Структура Directory** - Композитный узел, содержащий детей

**Пример использования:**

\`\`\`go
// Создание структуры файловой системы
root := NewDirectory("root")	// композитный узел
docs := NewDirectory("docs")	// вложенный композит
file1 := &File{Name: "readme.txt", Size: 100}	// листовой узел
file2 := &File{Name: "notes.txt", Size: 50}	// листовой узел

// Построение древовидной структуры
docs.Add(file2)	// добавить лист в композит
root.Add(docs)	// добавить композит в композит
root.Add(file1)	// добавить лист в композит

// Единообразные операции над любым Component
root.GetSize()	// возвращает 150 (рекурсивная сумма)
docs.GetSize()	// возвращает 50
file1.GetSize()	// возвращает 100

// Вывод всей структуры дерева
root.Print("")
// Вывод:
// Directory: root
//   Directory: docs
//     File: notes.txt (50 bytes)
//   File: readme.txt (100 bytes)
\`\`\``,
			hint1: `Методы File простые:
- GetName() возвращает поле Name напрямую
- GetSize() возвращает поле Size напрямую
- Print() использует fmt.Printf с форматом "%sFile: %s (%d bytes)\\n", где %s — отступ, %s — имя, %d — размер

Ключевая идея: File (лист) содержит реальные данные, а Directory (композит) делегирует детям.`,
			hint2: `Методы Directory используют рекурсию через интерфейс Component:
- GetSize() итерирует d.children и суммирует child.GetSize() — полиморфизм обрабатывает и файлы, и поддиректории
- Print() выводит имя директории, затем вызывает child.Print(indent + "  ") для каждого ребёнка
- Add() использует append для добавления компонента в слайс children
- Remove() находит компонент по имени и удаляет трюком со слайсами: d.children[:i], d.children[i+1:]...`,
			whyItMatters: `## Зачем нужен Composite

Паттерн Composite решает проблему единообразной работы с отдельными объектами и группами объектов. Без него клиентский код должен различать простые элементы и контейнеры, что приводит к сложной условной логике.

**Проблема — без Composite:**
\`\`\`go
// Клиентский код должен знать о каждом типе
func calculateSize(item interface{}) int {
    switch v := item.(type) {
    case *File:
        return v.Size
    case *Directory:
        total := 0
        for _, child := range v.Children {
            total += calculateSize(child)  // снова switch
        }
        return total
    default:
        return 0
    }
}
\`\`\`

**Решение — с Composite:**
\`\`\`go
// Единый интерфейс — клиенту не важен тип
func calculateSize(c Component) int {
    return c.GetSize()  // полиморфизм обрабатывает рекурсию
}
\`\`\`

## Реальные примеры на Go

**1. Дерево UI-виджетов:**
\`\`\`go
type Widget interface {
    Render(canvas *Canvas)
    GetBounds() Rectangle
}

type Button struct {
    text   string
    bounds Rectangle
}

func (b *Button) Render(canvas *Canvas) {
    canvas.DrawButton(b.bounds, b.text)
}

type Panel struct {
    children []Widget
    bounds   Rectangle
}

func (p *Panel) Render(canvas *Canvas) {
    canvas.DrawPanel(p.bounds)
    for _, child := range p.children {
        child.Render(canvas)  // рекурсивный рендеринг
    }
}
\`\`\`

**2. Дерево выражений (калькулятор):**
\`\`\`go
type Expression interface {
    Evaluate() float64
}

type Number struct{ value float64 }

func (n *Number) Evaluate() float64 { return n.value }

type Addition struct {
    left, right Expression
}

func (a *Addition) Evaluate() float64 {
    return a.left.Evaluate() + a.right.Evaluate()
}

// Использование: (3 + 4) + 5
expr := &Addition{
    left:  &Addition{left: &Number{3}, right: &Number{4}},
    right: &Number{5},
}
result := expr.Evaluate()  // возвращает 12
\`\`\`

## Продакшен паттерн: Организационная структура

\`\`\`go
type Employee interface {
    GetName() string
    GetSalary() float64
    GetSubordinates() []Employee
    Accept(visitor EmployeeVisitor)  // поддержка visitor
}

type Developer struct {
    name   string
    salary float64
}

func (d *Developer) GetName() string           { return d.name }
func (d *Developer) GetSalary() float64        { return d.salary }
func (d *Developer) GetSubordinates() []Employee { return nil }
func (d *Developer) Accept(v EmployeeVisitor)  { v.VisitDeveloper(d) }

type Manager struct {
    name         string
    salary       float64
    subordinates []Employee
}

func (m *Manager) GetName() string             { return m.name }
func (m *Manager) GetSalary() float64          { return m.salary }
func (m *Manager) GetSubordinates() []Employee { return m.subordinates }
func (m *Manager) Accept(v EmployeeVisitor)    { v.VisitManager(m) }

func (m *Manager) AddSubordinate(e Employee) {
    m.subordinates = append(m.subordinates, e)
}

// Расчёт общей зарплаты команды
func GetTotalSalary(e Employee) float64 {
    total := e.GetSalary()
    for _, sub := range e.GetSubordinates() {
        total += GetTotalSalary(sub)  // рекурсивный расчёт
    }
    return total
}

// Подсчёт всех сотрудников под менеджером
func CountEmployees(e Employee) int {
    count := 1  // считаем себя
    for _, sub := range e.GetSubordinates() {
        count += CountEmployees(sub)
    }
    return count
}
\`\`\`

## Частые ошибки

**1. Нарушение типобезопасности:**
\`\`\`go
// Плохо — использование interface{} теряет типобезопасность
type BadComponent interface {
    Add(child interface{})  // принимает что угодно
}

// Хорошо — строго типизированные дети
type Component interface {
    Add(child Component)  // принимает только Component
}
\`\`\`

**2. Добавление методов композита в лист:**
\`\`\`go
// Плохо — у листа есть Add/Remove, которые не имеют смысла
type File struct { /* ... */ }
func (f *File) Add(c Component) {
    panic("files can't have children")  // ошибка времени выполнения
}

// Хорошо — только композит управляет детьми
type Directory struct {
    children []Component
}
func (d *Directory) Add(c Component) {
    d.children = append(d.children, c)
}
// File вообще не имеет метода Add
\`\`\`

**3. Бесконечная рекурсия (добавление родителя в ребёнка):**
\`\`\`go
// Опасно — может создать циклы
parent := NewDirectory("parent")
child := NewDirectory("child")
parent.Add(child)
child.Add(parent)  // создаёт бесконечный цикл в GetSize()!

// Решение — проверка циклов перед добавлением
func (d *Directory) Add(c Component) error {
    if d.containsAncestor(c) {
        return errors.New("cannot add ancestor as child")
    }
    d.children = append(d.children, c)
    return nil
}
\`\`\`

**Ключевые принципы:**
- Интерфейс Component должен быть минимальным — только операции, значимые для листьев и композитов
- Композит хранит и делегирует детям; листья выполняют реальную работу
- Клиентский код должен работать с интерфейсом Component, а не с конкретными типами
- Рассмотрите обнаружение циклов для изменяемых древовидных структур`,
			solutionCode: `package patterns

import "fmt"

// Component интерфейс для файлов и директорий
type Component interface {	// общий интерфейс для единообразной работы
	GetName() string	// возвращает имя элемента
	GetSize() int	// возвращает размер (лист) или вычисленный размер (композит)
	Print(indent string)	// отображает элемент с отступом
}

// File — листовой компонент
type File struct {	// лист — не может содержать другие компоненты
	Name string	// имя файла
	Size int	// размер файла в байтах
}

// GetName возвращает имя файла
func (f *File) GetName() string {	// реализует интерфейс Component
	return f.Name	// просто возвращаем поле name
}

// GetSize возвращает размер файла
func (f *File) GetSize() int {	// реализует интерфейс Component
	return f.Size	// лист возвращает свой размер напрямую
}

// Print отображает информацию о файле
func (f *File) Print(indent string) {	// реализует интерфейс Component
	fmt.Printf("%sFile: %s (%d bytes)\\n", indent, f.Name, f.Size)	// форматированный вывод с отступом
}

// Directory — составной компонент
type Directory struct {	// композит — может содержать другие компоненты
	name     string	// имя директории
	children []Component	// слайс дочерних компонентов (файлы или директории)
}

// NewDirectory создаёт новую директорию
func NewDirectory(name string) *Directory {	// фабричная функция для директорий
	return &Directory{name: name, children: make([]Component, 0)}	// инициализация с пустым слайсом детей
}

// GetName возвращает имя директории
func (d *Directory) GetName() string {	// реализует интерфейс Component
	return d.name	// возвращаем имя директории
}

// GetSize возвращает сумму размеров всех детей
func (d *Directory) GetSize() int {	// реализует интерфейс Component — рекурсивный расчёт
	total := 0	// аккумулятор для общего размера
	for _, child := range d.children {	// итерируем по всем детям
		total += child.GetSize()	// рекурсивно получаем размер (работает для файлов и директорий)
	}
	return total	// возвращаем вычисленную сумму
}

// Print отображает директорию и всех детей
func (d *Directory) Print(indent string) {	// реализует интерфейс Component — рекурсивное отображение
	fmt.Printf("%sDirectory: %s\\n", indent, d.name)	// выводим директорию с текущим отступом
	for _, child := range d.children {	// итерируем по всем детям
		child.Print(indent + "  ")	// рекурсия с увеличенным отступом для визуальной иерархии
	}
}

// Add добавляет компонент в директорию
func (d *Directory) Add(component Component) {	// метод только для композита
	d.children = append(d.children, component)	// добавляем в слайс детей
}

// Remove удаляет компонент по имени
func (d *Directory) Remove(name string) {	// метод только для композита
	for i, child := range d.children {	// ищем среди детей
		if child.GetName() == name {	// нашли совпадающий компонент
			d.children = append(d.children[:i], d.children[i+1:]...)	// удаляем через срез
			return	// выходим после первого удаления
		}
	}
}`
		},
		uz: {
			title: 'Composite (Kompozit) Pattern',
			description: `Go tilida Composite patternini amalga oshiring — ob'ektlarni qism-butun ierarxiyalarini ifodalash uchun daraxt strukturalarida jamlang.

Composite patterni mijozlarga individual ob'ektlar va ob'ektlar kompozitsiyalari bilan bir xil tarzda ishlashga imkon beradi. Bu oddiy elementlar (barglar) ham, murakkab konteynerlar (kompozitlar) ham umumiy interfeysga ega bo'lgan daraxt shaklidagi strukturalarni qurish uchun muhim.

**Siz amalga oshirasiz:**

1. **Component interfeysi** - Barglar va kompozitlar uchun umumiy interfeys
2. **File strukturasi** - Nomi va hajmi bilan barg tugunlari
3. **Directory strukturasi** - Bolalarni o'z ichiga olgan kompozit tugun

**Foydalanish namunasi:**

\`\`\`go
// Fayl tizimi strukturasini yaratish
root := NewDirectory("root")	// kompozit tugun
docs := NewDirectory("docs")	// ichki kompozit
file1 := &File{Name: "readme.txt", Size: 100}	// barg tugun
file2 := &File{Name: "notes.txt", Size: 50}	// barg tugun

// Daraxt strukturasini qurish
docs.Add(file2)	// kompozitga barg qo'shish
root.Add(docs)	// kompozitga kompozit qo'shish
root.Add(file1)	// kompozitga barg qo'shish

// Har qanday Component da bir xil operatsiyalar
root.GetSize()	// 150 qaytaradi (rekursiv yig'indi)
docs.GetSize()	// 50 qaytaradi
file1.GetSize()	// 100 qaytaradi

// Butun daraxt strukturasini chiqarish
root.Print("")
// Chiqish:
// Directory: root
//   Directory: docs
//     File: notes.txt (50 bytes)
//   File: readme.txt (100 bytes)
\`\`\``,
			hint1: `File metodlari oddiy:
- GetName() to'g'ridan-to'g'ri Name maydonini qaytaradi
- GetSize() to'g'ridan-to'g'ri Size maydonini qaytaradi
- Print() "%sFile: %s (%d bytes)\\n" formati bilan fmt.Printf ishlatadi, bu yerda %s — bo'sh joy, %s — nom, %d — hajm

Asosiy tushuncha: File (barg) haqiqiy ma'lumotlarni o'z ichiga oladi, Directory (kompozit) esa bolalarga delegatsiya qiladi.`,
			hint2: `Directory metodlari Component interfeysi orqali rekursiyani ishlatadi:
- GetSize() d.children ni iteratsiya qiladi va child.GetSize() ni jamlaydi — polimorfizm fayllar va pastki direktoriyalarni ishlaydi
- Print() direktoriya nomini chiqaradi, keyin har bir bola uchun child.Print(indent + "  ") ni chaqiradi
- Add() komponentni children slice ga qo'shish uchun append ishlatadi
- Remove() komponentni nom bo'yicha topadi va slice usuli bilan olib tashlaydi: d.children[:i], d.children[i+1:]...`,
			whyItMatters: `## Nega Composite kerak

Composite patterni individual ob'ektlar va ob'ektlar guruhlari bilan bir xil tarzda ishlash muammosini hal qiladi. Busiz mijoz kodi oddiy elementlar va konteynerlarni farqlashi kerak, bu murakkab shartli mantiqqa olib keladi.

**Muammo — Composite siz:**
\`\`\`go
// Mijoz kodi har bir tur haqida bilishi kerak
func calculateSize(item interface{}) int {
    switch v := item.(type) {
    case *File:
        return v.Size
    case *Directory:
        total := 0
        for _, child := range v.Children {
            total += calculateSize(child)  // yana switch
        }
        return total
    default:
        return 0
    }
}
\`\`\`

**Yechim — Composite bilan:**
\`\`\`go
// Yagona interfeys — mijozga tur muhim emas
func calculateSize(c Component) int {
    return c.GetSize()  // polimorfizm rekursiyani boshqaradi
}
\`\`\`

## Go dagi real dunyo misollar

**1. UI Widget daraxti:**
\`\`\`go
type Widget interface {
    Render(canvas *Canvas)
    GetBounds() Rectangle
}

type Button struct {
    text   string
    bounds Rectangle
}

func (b *Button) Render(canvas *Canvas) {
    canvas.DrawButton(b.bounds, b.text)
}

type Panel struct {
    children []Widget
    bounds   Rectangle
}

func (p *Panel) Render(canvas *Canvas) {
    canvas.DrawPanel(p.bounds)
    for _, child := range p.children {
        child.Render(canvas)  // rekursiv render
    }
}
\`\`\`

**2. Ifoda daraxti (kalkulyator):**
\`\`\`go
type Expression interface {
    Evaluate() float64
}

type Number struct{ value float64 }

func (n *Number) Evaluate() float64 { return n.value }

type Addition struct {
    left, right Expression
}

func (a *Addition) Evaluate() float64 {
    return a.left.Evaluate() + a.right.Evaluate()
}

// Foydalanish: (3 + 4) + 5
expr := &Addition{
    left:  &Addition{left: &Number{3}, right: &Number{4}},
    right: &Number{5},
}
result := expr.Evaluate()  // 12 qaytaradi
\`\`\`

## Prodakshen pattern: Tashkilot strukturasi

\`\`\`go
type Employee interface {
    GetName() string
    GetSalary() float64
    GetSubordinates() []Employee
    Accept(visitor EmployeeVisitor)  // visitor qo'llab-quvvatlash
}

type Developer struct {
    name   string
    salary float64
}

func (d *Developer) GetName() string           { return d.name }
func (d *Developer) GetSalary() float64        { return d.salary }
func (d *Developer) GetSubordinates() []Employee { return nil }
func (d *Developer) Accept(v EmployeeVisitor)  { v.VisitDeveloper(d) }

type Manager struct {
    name         string
    salary       float64
    subordinates []Employee
}

func (m *Manager) GetName() string             { return m.name }
func (m *Manager) GetSalary() float64          { return m.salary }
func (m *Manager) GetSubordinates() []Employee { return m.subordinates }
func (m *Manager) Accept(v EmployeeVisitor)    { v.VisitManager(m) }

func (m *Manager) AddSubordinate(e Employee) {
    m.subordinates = append(m.subordinates, e)
}

// Jamoa umumiy maoshini hisoblash
func GetTotalSalary(e Employee) float64 {
    total := e.GetSalary()
    for _, sub := range e.GetSubordinates() {
        total += GetTotalSalary(sub)  // rekursiv hisoblash
    }
    return total
}

// Menejer ostidagi barcha xodimlarni sanash
func CountEmployees(e Employee) int {
    count := 1  // o'zimizni sanash
    for _, sub := range e.GetSubordinates() {
        count += CountEmployees(sub)
    }
    return count
}
\`\`\`

## Keng tarqalgan xatolar

**1. Tur xavfsizligini buzish:**
\`\`\`go
// Yomon — interface{} ishlatish tur xavfsizligini yo'qotadi
type BadComponent interface {
    Add(child interface{})  // hamma narsani qabul qiladi
}

// Yaxshi — qat'iy tiplashtilgan bolalar
type Component interface {
    Add(child Component)  // faqat Component qabul qiladi
}
\`\`\`

**2. Bargga kompozit metodlarini qo'shish:**
\`\`\`go
// Yomon — bargda Add/Remove bor, ular mantiqsiz
type File struct { /* ... */ }
func (f *File) Add(c Component) {
    panic("files can't have children")  // runtime xato
}

// Yaxshi — faqat kompozit bolalarni boshqaradi
type Directory struct {
    children []Component
}
func (d *Directory) Add(c Component) {
    d.children = append(d.children, c)
}
// File umuman Add metodiga ega emas
\`\`\`

**3. Cheksiz rekursiya (bolaga ota-onani qo'shish):**
\`\`\`go
// Xavfli — sikl yaratishi mumkin
parent := NewDirectory("parent")
child := NewDirectory("child")
parent.Add(child)
child.Add(parent)  // GetSize() da cheksiz sikl yaratadi!

// Yechim — qo'shishdan oldin sikllarni tekshirish
func (d *Directory) Add(c Component) error {
    if d.containsAncestor(c) {
        return errors.New("cannot add ancestor as child")
    }
    d.children = append(d.children, c)
    return nil
}
\`\`\`

**Asosiy tamoyillar:**
- Component interfeysi minimal bo'lishi kerak — faqat barglar va kompozitlar uchun mantiqiy operatsiyalar
- Kompozit bolalarga ega va delegatsiya qiladi; barglar haqiqiy ishni bajaradi
- Mijoz kodi Component interfeysi bilan ishlashi kerak, konkret turlar bilan emas
- O'zgaruvchan daraxt strukturalari uchun sikl aniqlashni ko'rib chiqing`,
			solutionCode: `package patterns

import "fmt"

// Component fayllar va direktoriyalar uchun interfeys
type Component interface {	// bir xil munosabat uchun umumiy interfeys
	GetName() string	// element nomini qaytaradi
	GetSize() int	// hajmni (barg) yoki hisoblangan hajmni (kompozit) qaytaradi
	Print(indent string)	// elementni bo'sh joy bilan ko'rsatadi
}

// File — barg komponenti
type File struct {	// barg — boshqa komponentlarni o'z ichiga olmaydi
	Name string	// fayl nomi
	Size int	// fayl hajmi baytlarda
}

// GetName fayl nomini qaytaradi
func (f *File) GetName() string {	// Component interfeysini amalga oshiradi
	return f.Name	// shunchaki name maydonini qaytaramiz
}

// GetSize fayl hajmini qaytaradi
func (f *File) GetSize() int {	// Component interfeysini amalga oshiradi
	return f.Size	// barg o'z hajmini to'g'ridan-to'g'ri qaytaradi
}

// Print fayl ma'lumotini ko'rsatadi
func (f *File) Print(indent string) {	// Component interfeysini amalga oshiradi
	fmt.Printf("%sFile: %s (%d bytes)\\n", indent, f.Name, f.Size)	// bo'sh joy bilan formatlangan chiqish
}

// Directory — kompozit komponenti
type Directory struct {	// kompozit — boshqa komponentlarni o'z ichiga olishi mumkin
	name     string	// direktoriya nomi
	children []Component	// bola komponentlar slice (fayllar yoki direktoriyalar)
}

// NewDirectory yangi direktoriya yaratadi
func NewDirectory(name string) *Directory {	// direktoriyalar uchun fabrika funksiyasi
	return &Directory{name: name, children: make([]Component, 0)}	// bo'sh bolalar slice bilan ishga tushirish
}

// GetName direktoriya nomini qaytaradi
func (d *Directory) GetName() string {	// Component interfeysini amalga oshiradi
	return d.name	// direktoriya nomini qaytaramiz
}

// GetSize barcha bolalar hajmlarining yig'indisini qaytaradi
func (d *Directory) GetSize() int {	// Component interfeysini amalga oshiradi — rekursiv hisoblash
	total := 0	// umumiy hajm uchun akkumulyator
	for _, child := range d.children {	// barcha bolalar bo'yicha iteratsiya
		total += child.GetSize()	// rekursiv hajm olish (fayllar va direktoriyalar uchun ishlaydi)
	}
	return total	// hisoblangan yig'indini qaytarish
}

// Print direktoriya va barcha bolalarni ko'rsatadi
func (d *Directory) Print(indent string) {	// Component interfeysini amalga oshiradi — rekursiv ko'rsatish
	fmt.Printf("%sDirectory: %s\\n", indent, d.name)	// joriy bo'sh joy bilan direktoriyani chiqarish
	for _, child := range d.children {	// barcha bolalar bo'yicha iteratsiya
		child.Print(indent + "  ")	// vizual ierarxiya uchun ko'paytirilgan bo'sh joy bilan rekursiya
	}
}

// Add komponentni direktoriyaga qo'shadi
func (d *Directory) Add(component Component) {	// faqat kompozit uchun metod
	d.children = append(d.children, component)	// bolalar slice ga qo'shish
}

// Remove komponentni nom bo'yicha olib tashlaydi
func (d *Directory) Remove(name string) {	// faqat kompozit uchun metod
	for i, child := range d.children {	// bolalar orasidan qidirish
		if child.GetName() == name {	// mos komponent topildi
			d.children = append(d.children[:i], d.children[i+1:]...)	// slice orqali olib tashlash
			return	// birinchi olib tashlashdan keyin chiqish
		}
	}
}`
		}
	}
};

export default task;
