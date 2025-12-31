import { Task } from '../../../../../../../types';

export const task: Task = {
	slug: 'go-dp-visitor',
	title: 'Visitor Pattern',
	difficulty: 'hard',
	tags: ['go', 'design-patterns', 'behavioral', 'visitor'],
	estimatedTime: '40m',
	isPremium: false,
	youtubeUrl: '',
	description: `Implement the Visitor pattern in Go - separate algorithms from the objects on which they operate.

**You will implement:**

1. **Shape interface** - Accept(visitor) method
2. **Circle, Rectangle** - Concrete elements
3. **ShapeVisitor interface** - VisitCircle, VisitRectangle
4. **AreaCalculator, JSONExporter** - Concrete visitors

**Example Usage:**

\`\`\`go
circle := &Circle{Radius: 5}	// create circle element
rectangle := &Rectangle{Width: 4, Height: 3}	// create rectangle element

areaCalc := &AreaCalculator{}	// create area visitor
result1 := circle.Accept(areaCalc)	// "Circle area: 78.54"
result2 := rectangle.Accept(areaCalc)	// "Rectangle area: 12.00"

jsonExp := &JSONExporter{}	// create JSON visitor
json1 := circle.Accept(jsonExp)	// {"type":"circle","radius":5.00}
json2 := rectangle.Accept(jsonExp)	// {"type":"rectangle","width":4.00,"height":3.00}
\`\`\``,
	initialCode: `package patterns

import "fmt"

type Shape interface {
}

type ShapeVisitor interface {
}

type Circle struct {
	Radius float64
}

func (c *Circle) Accept(visitor ShapeVisitor) string {
}

type Rectangle struct {
	Width  float64
	Height float64
}

func (r *Rectangle) Accept(visitor ShapeVisitor) string {
}

type AreaCalculator struct{}

func (a *AreaCalculator) VisitCircle(c *Circle) string {
}

func (a *AreaCalculator) VisitRectangle(r *Rectangle) string {
}

type JSONExporter struct{}

func (j *JSONExporter) VisitCircle(c *Circle) string {
}

func (j *JSONExporter) VisitRectangle(r *Rectangle) string {
}`,
	solutionCode: `package patterns

import "fmt"	// format package for string formatting

// Shape defines the element interface with Accept method
type Shape interface {	// element interface
	Accept(visitor ShapeVisitor) string	// double dispatch entry point
}

// ShapeVisitor defines visit methods for each element type
type ShapeVisitor interface {	// visitor interface
	VisitCircle(c *Circle) string	// operation for circles
	VisitRectangle(r *Rectangle) string	// operation for rectangles
}

// Circle is a concrete element
type Circle struct {	// concrete element type
	Radius float64	// circle's radius
}

// Accept calls the visitor's VisitCircle method
func (c *Circle) Accept(visitor ShapeVisitor) string {	// double dispatch implementation
	return visitor.VisitCircle(c)	// pass self to visitor
}

// Rectangle is a concrete element
type Rectangle struct {	// concrete element type
	Width  float64	// rectangle's width
	Height float64	// rectangle's height
}

// Accept calls the visitor's VisitRectangle method
func (r *Rectangle) Accept(visitor ShapeVisitor) string {	// double dispatch implementation
	return visitor.VisitRectangle(r)	// pass self to visitor
}

// AreaCalculator is a concrete visitor for area calculation
type AreaCalculator struct{}	// visitor for computing areas

// VisitCircle calculates area of a circle
func (a *AreaCalculator) VisitCircle(c *Circle) string {	// operation for Circle
	area := 3.14159 * c.Radius * c.Radius	// pi * r^2 formula
	return fmt.Sprintf("Circle area: %.2f", area)	// format result
}

// VisitRectangle calculates area of a rectangle
func (a *AreaCalculator) VisitRectangle(r *Rectangle) string {	// operation for Rectangle
	area := r.Width * r.Height	// width * height formula
	return fmt.Sprintf("Rectangle area: %.2f", area)	// format result
}

// JSONExporter is a concrete visitor for JSON serialization
type JSONExporter struct{}	// visitor for JSON export

// VisitCircle exports circle as JSON
func (j *JSONExporter) VisitCircle(c *Circle) string {	// JSON operation for Circle
	return fmt.Sprintf("{\"type\":\"circle\",\"radius\":%.2f}", c.Radius)	// JSON format
}

// VisitRectangle exports rectangle as JSON
func (j *JSONExporter) VisitRectangle(r *Rectangle) string {	// JSON operation for Rectangle
	return fmt.Sprintf("{\"type\":\"rectangle\",\"width\":%.2f,\"height\":%.2f}", r.Width, r.Height)	// JSON format
}`,
	hint1: `**Understanding Visitor's Double Dispatch:**

The key to Visitor pattern is "double dispatch" - the Accept method delegates to the visitor:

\`\`\`go
// Element interface - has Accept method
type Shape interface {
	Accept(visitor ShapeVisitor) string	// entry point for visitors
}

// Each element type knows which Visit method to call
func (c *Circle) Accept(visitor ShapeVisitor) string {
	return visitor.VisitCircle(c)	// pass "self" to visitor
}

func (r *Rectangle) Accept(visitor ShapeVisitor) string {
	return visitor.VisitRectangle(r)	// pass "self" to visitor
}
\`\`\`

This allows the visitor to operate on the concrete element type with full access to its data.`,
	hint2: `**Complete Visitor Implementation:**

\`\`\`go
// Visitor interface - one method per element type
type ShapeVisitor interface {
	VisitCircle(c *Circle) string	// handle circles
	VisitRectangle(r *Rectangle) string	// handle rectangles
}

// AreaCalculator - concrete visitor
func (a *AreaCalculator) VisitCircle(c *Circle) string {
	area := 3.14159 * c.Radius * c.Radius	// pi * r^2
	return fmt.Sprintf("Circle area: %.2f", area)
}

func (a *AreaCalculator) VisitRectangle(r *Rectangle) string {
	area := r.Width * r.Height	// w * h
	return fmt.Sprintf("Rectangle area: %.2f", area)
}

// JSONExporter - another concrete visitor
func (j *JSONExporter) VisitCircle(c *Circle) string {
	return fmt.Sprintf("{\"type\":\"circle\",\"radius\":%.2f}", c.Radius)
}

func (j *JSONExporter) VisitRectangle(r *Rectangle) string {
	return fmt.Sprintf("{\"type\":\"rectangle\",\"width\":%.2f,\"height\":%.2f}", r.Width, r.Height)
}
\`\`\`

Each visitor defines operations for all element types without modifying elements.`,
	whyItMatters: `## Why Visitor Exists

**The Problem: Adding Operations Without Modifying Classes**

Without Visitor, every new operation requires modifying all element classes:

\`\`\`go
// ❌ WITHOUT VISITOR - operations scattered in elements
type Circle struct {
	Radius float64
}

func (c *Circle) Area() float64 { return 3.14 * c.Radius * c.Radius }
func (c *Circle) ToJSON() string { return fmt.Sprintf("{...}") }
func (c *Circle) ToXML() string { return fmt.Sprintf("<.../>") }
// Need PDF export? Must modify Circle!
// Need perimeter? Must modify Circle!

type Rectangle struct {
	Width, Height float64
}

func (r *Rectangle) Area() float64 { return r.Width * r.Height }
func (r *Rectangle) ToJSON() string { return fmt.Sprintf("{...}") }
func (r *Rectangle) ToXML() string { return fmt.Sprintf("<.../>") }
// Same operations duplicated in every shape!

// ✅ WITH VISITOR - operations separated from elements
type ShapeVisitor interface {
	VisitCircle(c *Circle) string
	VisitRectangle(r *Rectangle) string
}

// New operation? Just add a new visitor!
type PerimeterCalculator struct{}
type PDFExporter struct{}
// No changes to Circle or Rectangle!
\`\`\`

---

## Real-World Examples in Go

**1. AST Traversal (go/ast):**
\`\`\`go
// Visitor: ast.Walk traverses syntax trees
// Elements: ast.Node types (FuncDecl, IfStmt, etc.)
ast.Walk(visitor, node)
\`\`\`

**2. File System Operations:**
\`\`\`go
// Visitor: Operations like size calculation, permission check
// Elements: File, Directory, Symlink
filepath.Walk(root, walkFunc)
\`\`\`

**3. Document Processing:**
\`\`\`go
// Visitor: PDF export, HTML render, word count
// Elements: Paragraph, Table, Image, Heading
\`\`\`

**4. Code Analysis Tools:**
\`\`\`go
// Visitor: Lint rules, metrics collectors
// Elements: Code constructs (functions, loops, etc.)
\`\`\`

---

## Production Pattern: AST Node Visitor

\`\`\`go
package main

import (
	"fmt"
	"strings"
)

// Node represents an AST node
type Node interface {
	Accept(visitor NodeVisitor) interface{}
}

// NodeVisitor defines operations for AST traversal
type NodeVisitor interface {
	VisitProgram(p *Program) interface{}
	VisitFunction(f *Function) interface{}
	VisitVariable(v *Variable) interface{}
	VisitBinaryOp(b *BinaryOp) interface{}
	VisitNumber(n *Number) interface{}
}

// Program is the root AST node
type Program struct {
	Functions []*Function
}

func (p *Program) Accept(visitor NodeVisitor) interface{} {
	return visitor.VisitProgram(p)
}

// Function represents a function definition
type Function struct {
	Name   string
	Params []string
	Body   Node
}

func (f *Function) Accept(visitor NodeVisitor) interface{} {
	return visitor.VisitFunction(f)
}

// Variable represents a variable reference
type Variable struct {
	Name string
}

func (v *Variable) Accept(visitor NodeVisitor) interface{} {
	return visitor.VisitVariable(v)
}

// BinaryOp represents a binary operation
type BinaryOp struct {
	Left     Node
	Operator string
	Right    Node
}

func (b *BinaryOp) Accept(visitor NodeVisitor) interface{} {
	return visitor.VisitBinaryOp(b)
}

// Number represents a numeric literal
type Number struct {
	Value float64
}

func (n *Number) Accept(visitor NodeVisitor) interface{} {
	return visitor.VisitNumber(n)
}

// CodePrinter is a visitor that prints code
type CodePrinter struct {
	Indent int
}

func (cp *CodePrinter) VisitProgram(p *Program) interface{} {
	var result []string
	for _, fn := range p.Functions {
		result = append(result, fn.Accept(cp).(string))
	}
	return strings.Join(result, "\\n\\n")
}

func (cp *CodePrinter) VisitFunction(f *Function) interface{} {
	params := strings.Join(f.Params, ", ")
	body := f.Body.Accept(cp).(string)
	return fmt.Sprintf("func %s(%s) {\\n  return %s\\n}", f.Name, params, body)
}

func (cp *CodePrinter) VisitVariable(v *Variable) interface{} {
	return v.Name
}

func (cp *CodePrinter) VisitBinaryOp(b *BinaryOp) interface{} {
	left := b.Left.Accept(cp).(string)
	right := b.Right.Accept(cp).(string)
	return fmt.Sprintf("(%s %s %s)", left, b.Operator, right)
}

func (cp *CodePrinter) VisitNumber(n *Number) interface{} {
	return fmt.Sprintf("%.0f", n.Value)
}

// Evaluator is a visitor that evaluates expressions
type Evaluator struct {
	Variables map[string]float64
}

func (e *Evaluator) VisitProgram(p *Program) interface{} {
	var lastResult interface{}
	for _, fn := range p.Functions {
		lastResult = fn.Accept(e)
	}
	return lastResult
}

func (e *Evaluator) VisitFunction(f *Function) interface{} {
	// Store function for later calls (simplified)
	return f.Body.Accept(e)
}

func (e *Evaluator) VisitVariable(v *Variable) interface{} {
	if val, ok := e.Variables[v.Name]; ok {
		return val
	}
	return 0.0
}

func (e *Evaluator) VisitBinaryOp(b *BinaryOp) interface{} {
	left := b.Left.Accept(e).(float64)
	right := b.Right.Accept(e).(float64)

	switch b.Operator {
	case "+":
		return left + right
	case "-":
		return left - right
	case "*":
		return left * right
	case "/":
		if right != 0 {
			return left / right
		}
		return 0.0
	}
	return 0.0
}

func (e *Evaluator) VisitNumber(n *Number) interface{} {
	return n.Value
}

// TypeChecker is a visitor that checks types
type TypeChecker struct {
	Errors []string
}

func (tc *TypeChecker) VisitProgram(p *Program) interface{} {
	for _, fn := range p.Functions {
		fn.Accept(tc)
	}
	return tc.Errors
}

func (tc *TypeChecker) VisitFunction(f *Function) interface{} {
	return f.Body.Accept(tc)
}

func (tc *TypeChecker) VisitVariable(v *Variable) interface{} {
	return "number"	// simplified: all variables are numbers
}

func (tc *TypeChecker) VisitBinaryOp(b *BinaryOp) interface{} {
	leftType := b.Left.Accept(tc).(string)
	rightType := b.Right.Accept(tc).(string)

	if leftType != rightType {
		tc.Errors = append(tc.Errors, fmt.Sprintf("Type mismatch: %s vs %s", leftType, rightType))
	}
	return "number"
}

func (tc *TypeChecker) VisitNumber(n *Number) interface{} {
	return "number"
}

// Usage
func main() {
	// Build AST: func add(x, y) { return x + y }
	program := &Program{
		Functions: []*Function{
			{
				Name:   "add",
				Params: []string{"x", "y"},
				Body: &BinaryOp{
					Left:     &Variable{Name: "x"},
					Operator: "+",
					Right:    &Variable{Name: "y"},
				},
			},
		},
	}

	// Visitor 1: Print code
	printer := &CodePrinter{}
	code := program.Accept(printer).(string)
	fmt.Println("Generated code:")
	fmt.Println(code)

	// Visitor 2: Evaluate with values
	evaluator := &Evaluator{
		Variables: map[string]float64{"x": 5, "y": 3},
	}
	result := program.Accept(evaluator).(float64)
	fmt.Printf("\\nEvaluation result: %.0f\\n", result)

	// Visitor 3: Type check
	checker := &TypeChecker{}
	errors := program.Accept(checker).([]string)
	fmt.Printf("Type errors: %v\\n", errors)
}
\`\`\`

---

## Common Mistakes to Avoid

**1. Modifying Elements in Visitor:**
\`\`\`go
// ❌ WRONG - visitor shouldn't modify elements
func (v *MyVisitor) VisitCircle(c *Circle) string {
	c.Radius = c.Radius * 2	// Don't modify!
	return fmt.Sprintf("...")
}

// ✅ RIGHT - visitor only reads and computes
func (v *MyVisitor) VisitCircle(c *Circle) string {
	scaledRadius := c.Radius * 2	// compute new value
	return fmt.Sprintf("Scaled radius: %.2f", scaledRadius)
}
\`\`\`

**2. Missing Visit Methods:**
\`\`\`go
// ❌ WRONG - missing element type
type ShapeVisitor interface {
	VisitCircle(c *Circle) string
	// Forgot VisitRectangle!
}

// ✅ RIGHT - all element types covered
type ShapeVisitor interface {
	VisitCircle(c *Circle) string
	VisitRectangle(r *Rectangle) string
	VisitTriangle(t *Triangle) string	// don't forget new types
}
\`\`\`

**3. Breaking Double Dispatch:**
\`\`\`go
// ❌ WRONG - type switch in visitor (defeats purpose)
func (v *BadVisitor) Visit(s Shape) string {
	switch shape := s.(type) {
	case *Circle:
		return "circle"
	case *Rectangle:
		return "rectangle"
	}
	return ""
}

// ✅ RIGHT - use Accept for proper dispatch
func ProcessShapes(shapes []Shape, visitor ShapeVisitor) []string {
	var results []string
	for _, shape := range shapes {
		results = append(results, shape.Accept(visitor))
	}
	return results
}
\`\`\``,
	order: 9,
	testCode: `package patterns

import (
	"strings"
	"testing"
)

// Test1: Circle.Accept calls visitor.VisitCircle
func Test1(t *testing.T) {
	circle := &Circle{Radius: 5}
	calc := &AreaCalculator{}
	result := circle.Accept(calc)
	if !strings.Contains(result, "Circle") {
		t.Error("Circle.Accept should call VisitCircle")
	}
}

// Test2: Rectangle.Accept calls visitor.VisitRectangle
func Test2(t *testing.T) {
	rect := &Rectangle{Width: 4, Height: 3}
	calc := &AreaCalculator{}
	result := rect.Accept(calc)
	if !strings.Contains(result, "Rectangle") {
		t.Error("Rectangle.Accept should call VisitRectangle")
	}
}

// Test3: AreaCalculator calculates circle area correctly
func Test3(t *testing.T) {
	circle := &Circle{Radius: 5}
	calc := &AreaCalculator{}
	result := circle.Accept(calc)
	if !strings.Contains(result, "78.5") {
		t.Error("Circle area with radius 5 should be ~78.5")
	}
}

// Test4: AreaCalculator calculates rectangle area correctly
func Test4(t *testing.T) {
	rect := &Rectangle{Width: 4, Height: 3}
	calc := &AreaCalculator{}
	result := rect.Accept(calc)
	if !strings.Contains(result, "12") {
		t.Error("Rectangle area 4x3 should be 12")
	}
}

// Test5: JSONExporter exports circle as JSON
func Test5(t *testing.T) {
	circle := &Circle{Radius: 5}
	exp := &JSONExporter{}
	result := circle.Accept(exp)
	if !strings.Contains(result, "circle") || !strings.Contains(result, "radius") {
		t.Error("JSONExporter should export circle with type and radius")
	}
}

// Test6: JSONExporter exports rectangle as JSON
func Test6(t *testing.T) {
	rect := &Rectangle{Width: 4, Height: 3}
	exp := &JSONExporter{}
	result := rect.Accept(exp)
	if !strings.Contains(result, "rectangle") || !strings.Contains(result, "width") {
		t.Error("JSONExporter should export rectangle with type, width, height")
	}
}

// Test7: Same element accepts different visitors
func Test7(t *testing.T) {
	circle := &Circle{Radius: 10}
	calc := &AreaCalculator{}
	exp := &JSONExporter{}
	areaResult := circle.Accept(calc)
	jsonResult := circle.Accept(exp)
	if areaResult == jsonResult {
		t.Error("Same element should produce different results for different visitors")
	}
}

// Test8: Circle struct has Radius field
func Test8(t *testing.T) {
	circle := &Circle{Radius: 7.5}
	if circle.Radius != 7.5 {
		t.Error("Circle.Radius should be set correctly")
	}
}

// Test9: Rectangle struct has Width and Height
func Test9(t *testing.T) {
	rect := &Rectangle{Width: 10, Height: 20}
	if rect.Width != 10 || rect.Height != 20 {
		t.Error("Rectangle Width and Height should be set correctly")
	}
}

// Test10: Visitor pattern allows adding new operations
func Test10(t *testing.T) {
	shapes := []Shape{&Circle{Radius: 5}, &Rectangle{Width: 4, Height: 3}}
	calc := &AreaCalculator{}
	for _, shape := range shapes {
		result := shape.Accept(calc)
		if result == "" {
			t.Error("All shapes should accept visitors and return results")
		}
	}
}
`,
	translations: {
		ru: {
			title: 'Паттерн Visitor (Посетитель)',
			description: `Реализуйте паттерн Visitor на Go — отделите алгоритмы от объектов, над которыми они работают.

**Вы реализуете:**

1. **Интерфейс Shape** - Метод Accept(visitor)
2. **Circle, Rectangle** - Конкретные элементы
3. **Интерфейс ShapeVisitor** - VisitCircle, VisitRectangle
4. **AreaCalculator, JSONExporter** - Конкретные посетители

**Пример использования:**

\`\`\`go
circle := &Circle{Radius: 5}	// создаём элемент круг
rectangle := &Rectangle{Width: 4, Height: 3}	// создаём элемент прямоугольник

areaCalc := &AreaCalculator{}	// создаём посетителя для площади
result1 := circle.Accept(areaCalc)	// "Circle area: 78.54"
result2 := rectangle.Accept(areaCalc)	// "Rectangle area: 12.00"

jsonExp := &JSONExporter{}	// создаём JSON посетителя
json1 := circle.Accept(jsonExp)	// {"type":"circle","radius":5.00}
json2 := rectangle.Accept(jsonExp)	// {"type":"rectangle","width":4.00,"height":3.00}
\`\`\``,
			hint1: `**Понимание двойной диспетчеризации Visitor:**

Ключ к паттерну Visitor — "двойная диспетчеризация" — метод Accept делегирует посетителю:

\`\`\`go
// Интерфейс элемента - имеет метод Accept
type Shape interface {
	Accept(visitor ShapeVisitor) string	// точка входа для посетителей
}

// Каждый тип элемента знает какой Visit метод вызывать
func (c *Circle) Accept(visitor ShapeVisitor) string {
	return visitor.VisitCircle(c)	// передаём "себя" посетителю
}

func (r *Rectangle) Accept(visitor ShapeVisitor) string {
	return visitor.VisitRectangle(r)	// передаём "себя" посетителю
}
\`\`\`

Это позволяет посетителю работать с конкретным типом элемента с полным доступом к его данным.`,
			hint2: `**Полная реализация Visitor:**

\`\`\`go
// Интерфейс посетителя - один метод на тип элемента
type ShapeVisitor interface {
	VisitCircle(c *Circle) string	// обработка кругов
	VisitRectangle(r *Rectangle) string	// обработка прямоугольников
}

// AreaCalculator - конкретный посетитель
func (a *AreaCalculator) VisitCircle(c *Circle) string {
	area := 3.14159 * c.Radius * c.Radius	// pi * r^2
	return fmt.Sprintf("Circle area: %.2f", area)
}

func (a *AreaCalculator) VisitRectangle(r *Rectangle) string {
	area := r.Width * r.Height	// w * h
	return fmt.Sprintf("Rectangle area: %.2f", area)
}

// JSONExporter - другой конкретный посетитель
func (j *JSONExporter) VisitCircle(c *Circle) string {
	return fmt.Sprintf("{\"type\":\"circle\",\"radius\":%.2f}", c.Radius)
}

func (j *JSONExporter) VisitRectangle(r *Rectangle) string {
	return fmt.Sprintf("{\"type\":\"rectangle\",\"width\":%.2f,\"height\":%.2f}", r.Width, r.Height)
}
\`\`\`

Каждый посетитель определяет операции для всех типов элементов без их модификации.`,
			whyItMatters: `## Почему существует Visitor

**Проблема: Добавление операций без модификации классов**

Без Visitor каждая новая операция требует модификации всех классов элементов:

\`\`\`go
// ❌ БЕЗ VISITOR - операции разбросаны по элементам
type Circle struct {
	Radius float64
}

func (c *Circle) Area() float64 { return 3.14 * c.Radius * c.Radius }
func (c *Circle) ToJSON() string { return fmt.Sprintf("{...}") }
func (c *Circle) ToXML() string { return fmt.Sprintf("<.../>") }
// Нужен PDF экспорт? Нужно изменить Circle!
// Нужен периметр? Нужно изменить Circle!

type Rectangle struct {
	Width, Height float64
}

func (r *Rectangle) Area() float64 { return r.Width * r.Height }
func (r *Rectangle) ToJSON() string { return fmt.Sprintf("{...}") }
func (r *Rectangle) ToXML() string { return fmt.Sprintf("<.../>") }
// Те же операции дублируются в каждой фигуре!

// ✅ С VISITOR - операции отделены от элементов
type ShapeVisitor interface {
	VisitCircle(c *Circle) string
	VisitRectangle(r *Rectangle) string
}

// Новая операция? Просто добавьте нового посетителя!
type PerimeterCalculator struct{}
type PDFExporter struct{}
// Никаких изменений в Circle или Rectangle!
\`\`\`

---

## Примеры из реального мира в Go

**1. Обход AST (go/ast):**
\`\`\`go
// Посетитель: ast.Walk обходит синтаксические деревья
// Элементы: типы ast.Node (FuncDecl, IfStmt, и т.д.)
ast.Walk(visitor, node)
\`\`\`

**2. Операции с файловой системой:**
\`\`\`go
// Посетитель: Операции вроде подсчёта размера, проверки прав
// Элементы: File, Directory, Symlink
filepath.Walk(root, walkFunc)
\`\`\`

**3. Обработка документов:**
\`\`\`go
// Посетитель: PDF экспорт, HTML рендеринг, подсчёт слов
// Элементы: Paragraph, Table, Image, Heading
\`\`\`

**4. Инструменты анализа кода:**
\`\`\`go
// Посетитель: Правила линтера, сборщики метрик
// Элементы: Конструкции кода (функции, циклы, и т.д.)
\`\`\`

---

## Продакшн паттерн: AST Node Visitor

\`\`\`go
package main

import (
	"fmt"
	"strings"
)

// Node представляет узел AST
type Node interface {
	Accept(visitor NodeVisitor) interface{}
}

// NodeVisitor определяет операции для обхода AST
type NodeVisitor interface {
	VisitProgram(p *Program) interface{}
	VisitFunction(f *Function) interface{}
	VisitVariable(v *Variable) interface{}
	VisitBinaryOp(b *BinaryOp) interface{}
	VisitNumber(n *Number) interface{}
}

// Program это корневой узел AST
type Program struct {
	Functions []*Function
}

func (p *Program) Accept(visitor NodeVisitor) interface{} {
	return visitor.VisitProgram(p)
}

// Function представляет определение функции
type Function struct {
	Name   string
	Params []string
	Body   Node
}

func (f *Function) Accept(visitor NodeVisitor) interface{} {
	return visitor.VisitFunction(f)
}

// Variable представляет ссылку на переменную
type Variable struct {
	Name string
}

func (v *Variable) Accept(visitor NodeVisitor) interface{} {
	return visitor.VisitVariable(v)
}

// BinaryOp представляет бинарную операцию
type BinaryOp struct {
	Left     Node
	Operator string
	Right    Node
}

func (b *BinaryOp) Accept(visitor NodeVisitor) interface{} {
	return visitor.VisitBinaryOp(b)
}

// Number представляет числовой литерал
type Number struct {
	Value float64
}

func (n *Number) Accept(visitor NodeVisitor) interface{} {
	return visitor.VisitNumber(n)
}

// CodePrinter это посетитель который печатает код
type CodePrinter struct {
	Indent int
}

func (cp *CodePrinter) VisitProgram(p *Program) interface{} {
	var result []string
	for _, fn := range p.Functions {
		result = append(result, fn.Accept(cp).(string))
	}
	return strings.Join(result, "\\n\\n")
}

func (cp *CodePrinter) VisitFunction(f *Function) interface{} {
	params := strings.Join(f.Params, ", ")
	body := f.Body.Accept(cp).(string)
	return fmt.Sprintf("func %s(%s) {\\n  return %s\\n}", f.Name, params, body)
}

func (cp *CodePrinter) VisitVariable(v *Variable) interface{} {
	return v.Name
}

func (cp *CodePrinter) VisitBinaryOp(b *BinaryOp) interface{} {
	left := b.Left.Accept(cp).(string)
	right := b.Right.Accept(cp).(string)
	return fmt.Sprintf("(%s %s %s)", left, b.Operator, right)
}

func (cp *CodePrinter) VisitNumber(n *Number) interface{} {
	return fmt.Sprintf("%.0f", n.Value)
}

// Evaluator это посетитель который вычисляет выражения
type Evaluator struct {
	Variables map[string]float64
}

func (e *Evaluator) VisitProgram(p *Program) interface{} {
	var lastResult interface{}
	for _, fn := range p.Functions {
		lastResult = fn.Accept(e)
	}
	return lastResult
}

func (e *Evaluator) VisitFunction(f *Function) interface{} {
	// Сохраняем функцию для последующих вызовов (упрощённо)
	return f.Body.Accept(e)
}

func (e *Evaluator) VisitVariable(v *Variable) interface{} {
	if val, ok := e.Variables[v.Name]; ok {
		return val
	}
	return 0.0
}

func (e *Evaluator) VisitBinaryOp(b *BinaryOp) interface{} {
	left := b.Left.Accept(e).(float64)
	right := b.Right.Accept(e).(float64)

	switch b.Operator {
	case "+":
		return left + right
	case "-":
		return left - right
	case "*":
		return left * right
	case "/":
		if right != 0 {
			return left / right
		}
		return 0.0
	}
	return 0.0
}

func (e *Evaluator) VisitNumber(n *Number) interface{} {
	return n.Value
}

// TypeChecker это посетитель который проверяет типы
type TypeChecker struct {
	Errors []string
}

func (tc *TypeChecker) VisitProgram(p *Program) interface{} {
	for _, fn := range p.Functions {
		fn.Accept(tc)
	}
	return tc.Errors
}

func (tc *TypeChecker) VisitFunction(f *Function) interface{} {
	return f.Body.Accept(tc)
}

func (tc *TypeChecker) VisitVariable(v *Variable) interface{} {
	return "number"	// упрощённо: все переменные числа
}

func (tc *TypeChecker) VisitBinaryOp(b *BinaryOp) interface{} {
	leftType := b.Left.Accept(tc).(string)
	rightType := b.Right.Accept(tc).(string)

	if leftType != rightType {
		tc.Errors = append(tc.Errors, fmt.Sprintf("Type mismatch: %s vs %s", leftType, rightType))
	}
	return "number"
}

func (tc *TypeChecker) VisitNumber(n *Number) interface{} {
	return "number"
}

// Использование
func main() {
	// Строим AST: func add(x, y) { return x + y }
	program := &Program{
		Functions: []*Function{
			{
				Name:   "add",
				Params: []string{"x", "y"},
				Body: &BinaryOp{
					Left:     &Variable{Name: "x"},
					Operator: "+",
					Right:    &Variable{Name: "y"},
				},
			},
		},
	}

	// Посетитель 1: Печать кода
	printer := &CodePrinter{}
	code := program.Accept(printer).(string)
	fmt.Println("Generated code:")
	fmt.Println(code)

	// Посетитель 2: Вычисление со значениями
	evaluator := &Evaluator{
		Variables: map[string]float64{"x": 5, "y": 3},
	}
	result := program.Accept(evaluator).(float64)
	fmt.Printf("\\nEvaluation result: %.0f\\n", result)

	// Посетитель 3: Проверка типов
	checker := &TypeChecker{}
	errors := program.Accept(checker).([]string)
	fmt.Printf("Type errors: %v\\n", errors)
}
\`\`\`

---

## Распространённые ошибки

**1. Модификация элементов в посетителе:**
\`\`\`go
// ❌ НЕПРАВИЛЬНО - посетитель не должен изменять элементы
func (v *MyVisitor) VisitCircle(c *Circle) string {
	c.Radius = c.Radius * 2	// Не изменяйте!
	return fmt.Sprintf("...")
}

// ✅ ПРАВИЛЬНО - посетитель только читает и вычисляет
func (v *MyVisitor) VisitCircle(c *Circle) string {
	scaledRadius := c.Radius * 2	// вычисляем новое значение
	return fmt.Sprintf("Scaled radius: %.2f", scaledRadius)
}
\`\`\`

**2. Отсутствующие Visit методы:**
\`\`\`go
// ❌ НЕПРАВИЛЬНО - отсутствует тип элемента
type ShapeVisitor interface {
	VisitCircle(c *Circle) string
	// Забыли VisitRectangle!
}

// ✅ ПРАВИЛЬНО - все типы элементов покрыты
type ShapeVisitor interface {
	VisitCircle(c *Circle) string
	VisitRectangle(r *Rectangle) string
	VisitTriangle(t *Triangle) string	// не забывайте новые типы
}
\`\`\`

**3. Нарушение двойной диспетчеризации:**
\`\`\`go
// ❌ НЕПРАВИЛЬНО - type switch в посетителе (нарушает цель)
func (v *BadVisitor) Visit(s Shape) string {
	switch shape := s.(type) {
	case *Circle:
		return "circle"
	case *Rectangle:
		return "rectangle"
	}
	return ""
}

// ✅ ПРАВИЛЬНО - используйте Accept для правильной диспетчеризации
func ProcessShapes(shapes []Shape, visitor ShapeVisitor) []string {
	var results []string
	for _, shape := range shapes {
		results = append(results, shape.Accept(visitor))
	}
	return results
}
\`\`\``
		},
		uz: {
			title: 'Visitor (Tashrif buyuruvchi) Pattern',
			description: `Go tilida Visitor patternini amalga oshiring — algoritmlarni ular ishlaydigan ob'ektlardan ajrating.

**Siz amalga oshirasiz:**

1. **Shape interfeysi** - Accept(visitor) metodi
2. **Circle, Rectangle** - Aniq elementlar
3. **ShapeVisitor interfeysi** - VisitCircle, VisitRectangle
4. **AreaCalculator, JSONExporter** - Aniq tashrif buyuruvchilar

**Foydalanish namunasi:**

\`\`\`go
circle := &Circle{Radius: 5}	// aylana elementi yaratamiz
rectangle := &Rectangle{Width: 4, Height: 3}	// to'rtburchak elementi yaratamiz

areaCalc := &AreaCalculator{}	// maydon uchun visitor yaratamiz
result1 := circle.Accept(areaCalc)	// "Circle area: 78.54"
result2 := rectangle.Accept(areaCalc)	// "Rectangle area: 12.00"

jsonExp := &JSONExporter{}	// JSON visitor yaratamiz
json1 := circle.Accept(jsonExp)	// {"type":"circle","radius":5.00}
json2 := rectangle.Accept(jsonExp)	// {"type":"rectangle","width":4.00,"height":3.00}
\`\`\``,
			hint1: `**Visitor ning ikki marta dispetcherlashni tushunish:**

Visitor patternining kaliti "ikki marta dispetcherlash" - Accept metodi visitorga delegatsiya qiladi:

\`\`\`go
// Element interfeysi - Accept metodi bor
type Shape interface {
	Accept(visitor ShapeVisitor) string	// visitorlar uchun kirish nuqtasi
}

// Har bir element turi qaysi Visit metodini chaqirishni biladi
func (c *Circle) Accept(visitor ShapeVisitor) string {
	return visitor.VisitCircle(c)	// "o'zini" visitorga uzatadi
}

func (r *Rectangle) Accept(visitor ShapeVisitor) string {
	return visitor.VisitRectangle(r)	// "o'zini" visitorga uzatadi
}
\`\`\`

Bu visitorga aniq element turi bilan uning ma'lumotlariga to'liq kirish imkonini beradi.`,
			hint2: `**To'liq Visitor realizatsiyasi:**

\`\`\`go
// Visitor interfeysi - har bir element turi uchun bitta metod
type ShapeVisitor interface {
	VisitCircle(c *Circle) string	// aylanalarni qayta ishlash
	VisitRectangle(r *Rectangle) string	// to'rtburchaklarni qayta ishlash
}

// AreaCalculator - aniq visitor
func (a *AreaCalculator) VisitCircle(c *Circle) string {
	area := 3.14159 * c.Radius * c.Radius	// pi * r^2
	return fmt.Sprintf("Circle area: %.2f", area)
}

func (a *AreaCalculator) VisitRectangle(r *Rectangle) string {
	area := r.Width * r.Height	// w * h
	return fmt.Sprintf("Rectangle area: %.2f", area)
}

// JSONExporter - boshqa aniq visitor
func (j *JSONExporter) VisitCircle(c *Circle) string {
	return fmt.Sprintf("{\"type\":\"circle\",\"radius\":%.2f}", c.Radius)
}

func (j *JSONExporter) VisitRectangle(r *Rectangle) string {
	return fmt.Sprintf("{\"type\":\"rectangle\",\"width\":%.2f,\"height\":%.2f}", r.Width, r.Height)
}
\`\`\`

Har bir visitor elementlarni o'zgartirmasdan barcha element turlari uchun operatsiyalarni aniqlaydi.`,
			whyItMatters: `## Nima uchun Visitor mavjud

**Muammo: Sinflarni o'zgartirmasdan operatsiyalar qo'shish**

Visitor'siz har bir yangi operatsiya barcha element sinflarini o'zgartirishni talab qiladi:

\`\`\`go
// ❌ VISITOR'SIZ - operatsiyalar elementlarga tarqalgan
type Circle struct {
	Radius float64
}

func (c *Circle) Area() float64 { return 3.14 * c.Radius * c.Radius }
func (c *Circle) ToJSON() string { return fmt.Sprintf("{...}") }
func (c *Circle) ToXML() string { return fmt.Sprintf("<.../>") }
// PDF eksport kerakmi? Circle'ni o'zgartirish kerak!
// Perimetr kerakmi? Circle'ni o'zgartirish kerak!

type Rectangle struct {
	Width, Height float64
}

func (r *Rectangle) Area() float64 { return r.Width * r.Height }
func (r *Rectangle) ToJSON() string { return fmt.Sprintf("{...}") }
func (r *Rectangle) ToXML() string { return fmt.Sprintf("<.../>") }
// Xuddi shu operatsiyalar har bir shaklda takrorlanadi!

// ✅ VISITOR BILAN - operatsiyalar elementlardan ajratilgan
type ShapeVisitor interface {
	VisitCircle(c *Circle) string
	VisitRectangle(r *Rectangle) string
}

// Yangi operatsiya? Faqat yangi visitor qo'shing!
type PerimeterCalculator struct{}
type PDFExporter struct{}
// Circle yoki Rectangle'da hech qanday o'zgarish yo'q!
\`\`\`

---

## Go'da haqiqiy dunyo misollari

**1. AST obhodi (go/ast):**
\`\`\`go
// Visitor: ast.Walk sintaktik daraxtlarni o'tadi
// Elementlar: ast.Node turlari (FuncDecl, IfStmt, va h.k.)
ast.Walk(visitor, node)
\`\`\`

**2. Fayl tizimi operatsiyalari:**
\`\`\`go
// Visitor: Hajmni hisoblash, ruxsatlarni tekshirish kabi operatsiyalar
// Elementlar: File, Directory, Symlink
filepath.Walk(root, walkFunc)
\`\`\`

**3. Hujjatlarni qayta ishlash:**
\`\`\`go
// Visitor: PDF eksport, HTML render, so'z soni
// Elementlar: Paragraph, Table, Image, Heading
\`\`\`

**4. Kod tahlil vositalari:**
\`\`\`go
// Visitor: Linter qoidalari, metrika yig'uvchilar
// Elementlar: Kod konstruktsiyalari (funksiyalar, tsikllar, va h.k.)
\`\`\`

---

## Production Pattern: AST Node Visitor

\`\`\`go
package main

import (
	"fmt"
	"strings"
)

// Node AST tugunini ifodalaydi
type Node interface {
	Accept(visitor NodeVisitor) interface{}
}

// NodeVisitor AST o'tish uchun operatsiyalarni aniqlaydi
type NodeVisitor interface {
	VisitProgram(p *Program) interface{}
	VisitFunction(f *Function) interface{}
	VisitVariable(v *Variable) interface{}
	VisitBinaryOp(b *BinaryOp) interface{}
	VisitNumber(n *Number) interface{}
}

// Program bu AST ning ildiz tuguni
type Program struct {
	Functions []*Function
}

func (p *Program) Accept(visitor NodeVisitor) interface{} {
	return visitor.VisitProgram(p)
}

// Function funksiya ta'rifini ifodalaydi
type Function struct {
	Name   string
	Params []string
	Body   Node
}

func (f *Function) Accept(visitor NodeVisitor) interface{} {
	return visitor.VisitFunction(f)
}

// Variable o'zgaruvchi havolasini ifodalaydi
type Variable struct {
	Name string
}

func (v *Variable) Accept(visitor NodeVisitor) interface{} {
	return visitor.VisitVariable(v)
}

// BinaryOp ikkilik operatsiyani ifodalaydi
type BinaryOp struct {
	Left     Node
	Operator string
	Right    Node
}

func (b *BinaryOp) Accept(visitor NodeVisitor) interface{} {
	return visitor.VisitBinaryOp(b)
}

// Number raqamli literalni ifodalaydi
type Number struct {
	Value float64
}

func (n *Number) Accept(visitor NodeVisitor) interface{} {
	return visitor.VisitNumber(n)
}

// CodePrinter kod chop etuvchi visitor
type CodePrinter struct {
	Indent int
}

func (cp *CodePrinter) VisitProgram(p *Program) interface{} {
	var result []string
	for _, fn := range p.Functions {
		result = append(result, fn.Accept(cp).(string))
	}
	return strings.Join(result, "\\n\\n")
}

func (cp *CodePrinter) VisitFunction(f *Function) interface{} {
	params := strings.Join(f.Params, ", ")
	body := f.Body.Accept(cp).(string)
	return fmt.Sprintf("func %s(%s) {\\n  return %s\\n}", f.Name, params, body)
}

func (cp *CodePrinter) VisitVariable(v *Variable) interface{} {
	return v.Name
}

func (cp *CodePrinter) VisitBinaryOp(b *BinaryOp) interface{} {
	left := b.Left.Accept(cp).(string)
	right := b.Right.Accept(cp).(string)
	return fmt.Sprintf("(%s %s %s)", left, b.Operator, right)
}

func (cp *CodePrinter) VisitNumber(n *Number) interface{} {
	return fmt.Sprintf("%.0f", n.Value)
}

// Evaluator ifodalarni hisoblovchi visitor
type Evaluator struct {
	Variables map[string]float64
}

func (e *Evaluator) VisitProgram(p *Program) interface{} {
	var lastResult interface{}
	for _, fn := range p.Functions {
		lastResult = fn.Accept(e)
	}
	return lastResult
}

func (e *Evaluator) VisitFunction(f *Function) interface{} {
	// Funksiyani keyingi chaqiruvlar uchun saqlaymiz (soddalashtirilgan)
	return f.Body.Accept(e)
}

func (e *Evaluator) VisitVariable(v *Variable) interface{} {
	if val, ok := e.Variables[v.Name]; ok {
		return val
	}
	return 0.0
}

func (e *Evaluator) VisitBinaryOp(b *BinaryOp) interface{} {
	left := b.Left.Accept(e).(float64)
	right := b.Right.Accept(e).(float64)

	switch b.Operator {
	case "+":
		return left + right
	case "-":
		return left - right
	case "*":
		return left * right
	case "/":
		if right != 0 {
			return left / right
		}
		return 0.0
	}
	return 0.0
}

func (e *Evaluator) VisitNumber(n *Number) interface{} {
	return n.Value
}

// TypeChecker turlarni tekshiruvchi visitor
type TypeChecker struct {
	Errors []string
}

func (tc *TypeChecker) VisitProgram(p *Program) interface{} {
	for _, fn := range p.Functions {
		fn.Accept(tc)
	}
	return tc.Errors
}

func (tc *TypeChecker) VisitFunction(f *Function) interface{} {
	return f.Body.Accept(tc)
}

func (tc *TypeChecker) VisitVariable(v *Variable) interface{} {
	return "number"	// soddalashtirilgan: barcha o'zgaruvchilar raqam
}

func (tc *TypeChecker) VisitBinaryOp(b *BinaryOp) interface{} {
	leftType := b.Left.Accept(tc).(string)
	rightType := b.Right.Accept(tc).(string)

	if leftType != rightType {
		tc.Errors = append(tc.Errors, fmt.Sprintf("Type mismatch: %s vs %s", leftType, rightType))
	}
	return "number"
}

func (tc *TypeChecker) VisitNumber(n *Number) interface{} {
	return "number"
}

// Foydalanish
func main() {
	// AST quramiz: func add(x, y) { return x + y }
	program := &Program{
		Functions: []*Function{
			{
				Name:   "add",
				Params: []string{"x", "y"},
				Body: &BinaryOp{
					Left:     &Variable{Name: "x"},
					Operator: "+",
					Right:    &Variable{Name: "y"},
				},
			},
		},
	}

	// Visitor 1: Kodni chop etish
	printer := &CodePrinter{}
	code := program.Accept(printer).(string)
	fmt.Println("Generated code:")
	fmt.Println(code)

	// Visitor 2: Qiymatlar bilan hisoblash
	evaluator := &Evaluator{
		Variables: map[string]float64{"x": 5, "y": 3},
	}
	result := program.Accept(evaluator).(float64)
	fmt.Printf("\\nEvaluation result: %.0f\\n", result)

	// Visitor 3: Turni tekshirish
	checker := &TypeChecker{}
	errors := program.Accept(checker).([]string)
	fmt.Printf("Type errors: %v\\n", errors)
}
\`\`\`

---

## Keng tarqalgan xatolar

**1. Visitorda elementlarni o'zgartirish:**
\`\`\`go
// ❌ NOTO'G'RI - visitor elementlarni o'zgartirmasligi kerak
func (v *MyVisitor) VisitCircle(c *Circle) string {
	c.Radius = c.Radius * 2	// O'zgartirmang!
	return fmt.Sprintf("...")
}

// ✅ TO'G'RI - visitor faqat o'qiydi va hisoblaydi
func (v *MyVisitor) VisitCircle(c *Circle) string {
	scaledRadius := c.Radius * 2	// yangi qiymatni hisoblaymiz
	return fmt.Sprintf("Scaled radius: %.2f", scaledRadius)
}
\`\`\`

**2. Yo'qolgan Visit metodlar:**
\`\`\`go
// ❌ NOTO'G'RI - element turi yo'qolgan
type ShapeVisitor interface {
	VisitCircle(c *Circle) string
	// VisitRectangle ni unutdik!
}

// ✅ TO'G'RI - barcha element turlari qamrab olingan
type ShapeVisitor interface {
	VisitCircle(c *Circle) string
	VisitRectangle(r *Rectangle) string
	VisitTriangle(t *Triangle) string	// yangi turlarni unutmang
}
\`\`\`

**3. Ikki marta dispetcherlashni buzish:**
\`\`\`go
// ❌ NOTO'G'RI - visitorda type switch (maqsadni buzadi)
func (v *BadVisitor) Visit(s Shape) string {
	switch shape := s.(type) {
	case *Circle:
		return "circle"
	case *Rectangle:
		return "rectangle"
	}
	return ""
}

// ✅ TO'G'RI - to'g'ri dispetcherlash uchun Accept ishlating
func ProcessShapes(shapes []Shape, visitor ShapeVisitor) []string {
	var results []string
	for _, shape := range shapes {
		results = append(results, shape.Accept(visitor))
	}
	return results
}
\`\`\``
		}
	}
};

export default task;
