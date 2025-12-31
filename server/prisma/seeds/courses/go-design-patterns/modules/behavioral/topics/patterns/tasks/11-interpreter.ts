import { Task } from '../../../../../../../types';

export const task: Task = {
	slug: 'go-dp-interpreter',
	title: 'Interpreter Pattern',
	difficulty: 'hard',
	tags: ['go', 'design-patterns', 'behavioral', 'interpreter'],
	estimatedTime: '45m',
	isPremium: false,
	youtubeUrl: '',
	description: `Implement the Interpreter pattern in Go - define a grammar representation and an interpreter for it.

**You will implement:**

1. **Expression interface** - Interpret method
2. **NumberExpression** - Terminal expression
3. **AddExpression, SubtractExpression** - Non-terminal expressions
4. **Parser** - Builds expression tree

**Example Usage:**

\`\`\`go
expr := Parse("5 + 3 - 2")	// parse expression string
result := expr.Interpret()	// evaluate expression tree
fmt.Println(result)	// 6

expr2 := Parse("10 * 2 + 5")	// another expression
result2 := expr2.Interpret()	// evaluate
fmt.Println(result2)	// 25

expr3 := Parse("100 - 50 * 2")	// with subtraction and multiply
result3 := expr3.Interpret()	// evaluate
fmt.Println(result3)	// 100
\`\`\``,
	initialCode: `package patterns

import (
	"strconv"
	"strings"
)

type Expression interface {
}

type NumberExpression struct {
	value int
}

func (n *NumberExpression) Interpret() int {
}

type AddExpression struct {
	left  Expression
	right Expression
}

func (a *AddExpression) Interpret() int {
}

type SubtractExpression struct {
	left  Expression
	right Expression
}

func (s *SubtractExpression) Interpret() int {
}

type MultiplyExpression struct {
	left  Expression
	right Expression
}

func (m *MultiplyExpression) Interpret() int {
}

func Parse(expression string) Expression {
}`,
	solutionCode: `package patterns

import (
	"strconv"	// string conversion utilities
	"strings"	// string manipulation utilities
)

// Expression defines the interface for all expression types
type Expression interface {	// abstract expression interface
	Interpret() int	// evaluate and return integer result
}

// NumberExpression is a terminal expression that holds a value
type NumberExpression struct {	// terminal expression (leaf node)
	value int	// the numeric value
}

// Interpret returns the stored number value
func (n *NumberExpression) Interpret() int {	// terminal: returns value directly
	return n.value	// leaf nodes just return their value
}

// AddExpression represents addition of two expressions
type AddExpression struct {	// non-terminal expression for addition
	left  Expression	// left operand
	right Expression	// right operand
}

// Interpret evaluates left and right, then adds them
func (a *AddExpression) Interpret() int {	// recursive evaluation
	return a.left.Interpret() + a.right.Interpret()	// add both results
}

// SubtractExpression represents subtraction of two expressions
type SubtractExpression struct {	// non-terminal expression for subtraction
	left  Expression	// left operand (minuend)
	right Expression	// right operand (subtrahend)
}

// Interpret evaluates left and right, then subtracts
func (s *SubtractExpression) Interpret() int {	// recursive evaluation
	return s.left.Interpret() - s.right.Interpret()	// subtract right from left
}

// MultiplyExpression represents multiplication of two expressions
type MultiplyExpression struct {	// non-terminal expression for multiplication
	left  Expression	// left operand (multiplicand)
	right Expression	// right operand (multiplier)
}

// Interpret evaluates left and right, then multiplies
func (m *MultiplyExpression) Interpret() int {	// recursive evaluation
	return m.left.Interpret() * m.right.Interpret()	// multiply both results
}

// Parse creates an expression tree from a string
func Parse(expression string) Expression {	// parser builds the AST
	tokens := strings.Fields(expression)	// split by whitespace
	if len(tokens) == 0 {	// handle empty input
		return &NumberExpression{value: 0}	// return zero for empty
	}

	num, _ := strconv.Atoi(tokens[0])	// parse first number
	result := Expression(&NumberExpression{value: num})	// start with first number

	for i := 1; i < len(tokens)-1; i += 2 {	// process operator-number pairs
		operator := tokens[i]	// get operator (+, -, *)
		num, _ := strconv.Atoi(tokens[i+1])	// parse next number
		right := &NumberExpression{value: num}	// create right operand

		switch operator {	// build appropriate expression node
		case "+":	// addition operator
			result = &AddExpression{left: result, right: right}	// wrap in add
		case "-":	// subtraction operator
			result = &SubtractExpression{left: result, right: right}	// wrap in subtract
		case "*":	// multiplication operator
			result = &MultiplyExpression{left: result, right: right}	// wrap in multiply
		}
	}
	return result	// return root of expression tree
}`,
	hint1: `**Understanding Interpreter Structure:**

The Interpreter pattern has two types of expressions:

\`\`\`go
// 1. Terminal Expression - leaf nodes with values
type NumberExpression struct {
	value int	// holds the actual number
}

func (n *NumberExpression) Interpret() int {
	return n.value	// simply return the stored value
}

// 2. Non-Terminal Expressions - composite nodes with children
type AddExpression struct {
	left  Expression	// can be Number or another operation
	right Expression	// can be Number or another operation
}

func (a *AddExpression) Interpret() int {
	// Recursively interpret children then combine
	return a.left.Interpret() + a.right.Interpret()
}
\`\`\`

Terminal expressions hold values, non-terminal expressions combine children.`,
	hint2: `**Complete Parser Implementation:**

\`\`\`go
func Parse(expression string) Expression {
	tokens := strings.Fields(expression)	// ["5", "+", "3", "-", "2"]
	if len(tokens) == 0 {
		return &NumberExpression{value: 0}
	}

	// Start with first number
	num, _ := strconv.Atoi(tokens[0])
	result := Expression(&NumberExpression{value: num})

	// Process pairs: (operator, number)
	for i := 1; i < len(tokens)-1; i += 2 {
		operator := tokens[i]	// "+", "-", or "*"
		num, _ := strconv.Atoi(tokens[i+1])
		right := &NumberExpression{value: num}

		switch operator {
		case "+":
			result = &AddExpression{left: result, right: right}
		case "-":
			result = &SubtractExpression{left: result, right: right}
		case "*":
			result = &MultiplyExpression{left: result, right: right}
		}
	}
	return result
}

// "5 + 3 - 2" becomes:
//       Subtract
//      /        \\
//    Add        2
//   /   \\
//  5     3
\`\`\`

The parser builds a left-associative tree.`,
	whyItMatters: `## Why Interpreter Exists

**The Problem: Evaluating Domain-Specific Languages**

Without Interpreter, evaluating expressions requires complex, monolithic code:

\`\`\`go
// ❌ WITHOUT INTERPRETER - tangled evaluation logic
func Evaluate(expr string) int {
	// Manual parsing and evaluation mixed together
	parts := strings.Split(expr, " ")
	result := parseNumber(parts[0])
	for i := 1; i < len(parts); i += 2 {
		op := parts[i]
		num := parseNumber(parts[i+1])
		if op == "+" {
			result += num
		} else if op == "-" {
			result -= num
		}
		// Adding new operators = modifying this mess
	}
	return result
}

// ✅ WITH INTERPRETER - clean, extensible structure
type Expression interface {
	Interpret() int	// each expression knows how to evaluate itself
}

// Adding new operations = just add new expression type!
type ModuloExpression struct {
	left, right Expression
}

func (m *ModuloExpression) Interpret() int {
	return m.left.Interpret() % m.right.Interpret()
}
\`\`\`

---

## Real-World Examples in Go

**1. SQL Query Execution (database/sql):**
\`\`\`go
// Grammar: SELECT, WHERE, JOIN, etc.
// Expressions: Column, Table, Condition
db.Query("SELECT name FROM users WHERE age > 18")
\`\`\`

**2. Regular Expressions (regexp):**
\`\`\`go
// Grammar: patterns, quantifiers, character classes
// Expressions: Literal, Star, Plus, Alternation
re := regexp.MustCompile("[a-z]+")
\`\`\`

**3. Template Engines (text/template):**
\`\`\`go
// Grammar: variables, conditionals, loops
// Expressions: Text, Variable, Range, If
tmpl.Execute(w, data)
\`\`\`

**4. JSON Path Queries:**
\`\`\`go
// Grammar: paths, filters, wildcards
// Expressions: Root, Child, Filter, Wildcard
jsonpath.Get(data, "$.users[*].name")
\`\`\`

---

## Production Pattern: Boolean Expression Evaluator

\`\`\`go
package main

import (
	"fmt"
	"strings"
)

// BoolExpression evaluates to a boolean result
type BoolExpression interface {
	Interpret(context map[string]bool) bool
}

// Variable represents a boolean variable
type Variable struct {
	Name string
}

func (v *Variable) Interpret(context map[string]bool) bool {
	return context[v.Name]
}

// Constant represents a boolean constant (true/false)
type Constant struct {
	Value bool
}

func (c *Constant) Interpret(context map[string]bool) bool {
	return c.Value
}

// And represents logical AND
type And struct {
	Left  BoolExpression
	Right BoolExpression
}

func (a *And) Interpret(context map[string]bool) bool {
	return a.Left.Interpret(context) && a.Right.Interpret(context)
}

// Or represents logical OR
type Or struct {
	Left  BoolExpression
	Right BoolExpression
}

func (o *Or) Interpret(context map[string]bool) bool {
	return o.Left.Interpret(context) || o.Right.Interpret(context)
}

// Not represents logical NOT
type Not struct {
	Operand BoolExpression
}

func (n *Not) Interpret(context map[string]bool) bool {
	return !n.Operand.Interpret(context)
}

// Rule represents a named business rule with an expression
type Rule struct {
	Name       string
	Expression BoolExpression
}

func (r *Rule) Evaluate(context map[string]bool) bool {
	return r.Expression.Interpret(context)
}

// RuleEngine manages and evaluates business rules
type RuleEngine struct {
	rules []Rule
}

func NewRuleEngine() *RuleEngine {
	return &RuleEngine{rules: make([]Rule, 0)}
}

func (e *RuleEngine) AddRule(name string, expr BoolExpression) {
	e.rules = append(e.rules, Rule{Name: name, Expression: expr})
}

func (e *RuleEngine) EvaluateAll(context map[string]bool) map[string]bool {
	results := make(map[string]bool)
	for _, rule := range e.rules {
		results[rule.Name] = rule.Evaluate(context)
	}
	return results
}

func (e *RuleEngine) GetPassingRules(context map[string]bool) []string {
	var passing []string
	for _, rule := range e.rules {
		if rule.Evaluate(context) {
			passing = append(passing, rule.Name)
		}
	}
	return passing
}

// ParseBoolExpr parses simple boolean expressions
// Supports: variables, AND, OR, NOT, true, false
func ParseBoolExpr(expr string) BoolExpression {
	expr = strings.TrimSpace(expr)

	// Handle NOT
	if strings.HasPrefix(expr, "NOT ") {
		operand := ParseBoolExpr(expr[4:])
		return &Not{Operand: operand}
	}

	// Handle OR (lowest precedence)
	if idx := strings.Index(expr, " OR "); idx != -1 {
		left := ParseBoolExpr(expr[:idx])
		right := ParseBoolExpr(expr[idx+4:])
		return &Or{Left: left, Right: right}
	}

	// Handle AND
	if idx := strings.Index(expr, " AND "); idx != -1 {
		left := ParseBoolExpr(expr[:idx])
		right := ParseBoolExpr(expr[idx+5:])
		return &And{Left: left, Right: right}
	}

	// Handle constants
	if expr == "true" {
		return &Constant{Value: true}
	}
	if expr == "false" {
		return &Constant{Value: false}
	}

	// Handle variable
	return &Variable{Name: expr}
}

// Usage
func main() {
	// Create rule engine
	engine := NewRuleEngine()

	// Define business rules using expressions
	// Rule 1: User can access premium content
	premiumAccess := &Or{
		Left:  &Variable{Name: "isPremium"},
		Right: &Variable{Name: "isAdmin"},
	}
	engine.AddRule("premium_access", premiumAccess)

	// Rule 2: User can download files
	downloadAllowed := &And{
		Left:  &Variable{Name: "isVerified"},
		Right: &Not{Operand: &Variable{Name: "isBanned"}},
	}
	engine.AddRule("download_allowed", downloadAllowed)

	// Rule 3: Show ads (not premium and not admin)
	showAds := &And{
		Left:  &Not{Operand: &Variable{Name: "isPremium"}},
		Right: &Not{Operand: &Variable{Name: "isAdmin"}},
	}
	engine.AddRule("show_ads", showAds)

	// Parse expression from string
	rule4 := ParseBoolExpr("isActive AND isVerified OR isAdmin")
	engine.AddRule("can_comment", rule4)

	// Evaluate rules for a user
	userContext := map[string]bool{
		"isPremium":  false,
		"isAdmin":    false,
		"isVerified": true,
		"isBanned":   false,
		"isActive":   true,
	}

	fmt.Println("User Context:", userContext)
	fmt.Println()

	// Evaluate all rules
	results := engine.EvaluateAll(userContext)
	for name, passed := range results {
		status := "DENIED"
		if passed {
			status = "GRANTED"
		}
		fmt.Printf("Rule '%s': %s\\n", name, status)
	}

	// Get passing rules
	fmt.Println("\\nPassing rules:", engine.GetPassingRules(userContext))
}
\`\`\`

---

## Common Mistakes to Avoid

**1. No Shared Grammar Definition:**
\`\`\`go
// ❌ WRONG - inconsistent interpretation
type BadExpr1 struct { value int }
func (e *BadExpr1) Interpret() int { return e.value }

type BadExpr2 struct { value int }
func (e *BadExpr2) Interpret() float64 { return float64(e.value) }

// ✅ RIGHT - consistent interface
type Expression interface {
	Interpret() int	// all expressions return same type
}

type NumberExpr struct { value int }
func (n *NumberExpr) Interpret() int { return n.value }

type AddExpr struct { left, right Expression }
func (a *AddExpr) Interpret() int {
	return a.left.Interpret() + a.right.Interpret()
}
\`\`\`

**2. Missing Context Parameter:**
\`\`\`go
// ❌ WRONG - can't evaluate variables
type Variable struct { name string }
func (v *Variable) Interpret() int {
	// How to get variable value???
	return 0
}

// ✅ RIGHT - use context for variables
type Variable struct { name string }
func (v *Variable) Interpret(ctx map[string]int) int {
	return ctx[v.name]	// lookup in context
}
\`\`\`

**3. Parser Creates Wrong Tree Structure:**
\`\`\`go
// ❌ WRONG - ignores operator precedence
// "2 + 3 * 4" should be 2 + (3 * 4) = 14
// But naive left-to-right gives (2 + 3) * 4 = 20

// ✅ RIGHT - handle precedence properly
func Parse(expr string) Expression {
	// First handle * and /
	// Then handle + and -
	// Or use parentheses in grammar
}
\`\`\``,
	order: 10,
	testCode: `package patterns

import "testing"

// Test1: NumberExpression returns its value
func Test1(t *testing.T) {
	num := &NumberExpression{value: 42}
	if num.Interpret() != 42 {
		t.Errorf("Expected 42, got %d", num.Interpret())
	}
}

// Test2: AddExpression adds two numbers
func Test2(t *testing.T) {
	left := &NumberExpression{value: 5}
	right := &NumberExpression{value: 3}
	add := &AddExpression{left: left, right: right}
	if add.Interpret() != 8 {
		t.Errorf("Expected 8, got %d", add.Interpret())
	}
}

// Test3: SubtractExpression subtracts two numbers
func Test3(t *testing.T) {
	left := &NumberExpression{value: 10}
	right := &NumberExpression{value: 4}
	sub := &SubtractExpression{left: left, right: right}
	if sub.Interpret() != 6 {
		t.Errorf("Expected 6, got %d", sub.Interpret())
	}
}

// Test4: MultiplyExpression multiplies two numbers
func Test4(t *testing.T) {
	left := &NumberExpression{value: 6}
	right := &NumberExpression{value: 7}
	mul := &MultiplyExpression{left: left, right: right}
	if mul.Interpret() != 42 {
		t.Errorf("Expected 42, got %d", mul.Interpret())
	}
}

// Test5: Parse handles simple addition
func Test5(t *testing.T) {
	expr := Parse("5 + 3")
	if expr.Interpret() != 8 {
		t.Errorf("Expected 8, got %d", expr.Interpret())
	}
}

// Test6: Parse handles simple subtraction
func Test6(t *testing.T) {
	expr := Parse("10 - 4")
	if expr.Interpret() != 6 {
		t.Errorf("Expected 6, got %d", expr.Interpret())
	}
}

// Test7: Parse handles simple multiplication
func Test7(t *testing.T) {
	expr := Parse("6 * 7")
	if expr.Interpret() != 42 {
		t.Errorf("Expected 42, got %d", expr.Interpret())
	}
}

// Test8: Parse handles chained operations
func Test8(t *testing.T) {
	expr := Parse("5 + 3 - 2")
	if expr.Interpret() != 6 {
		t.Errorf("Expected 6, got %d", expr.Interpret())
	}
}

// Test9: Parse handles single number
func Test9(t *testing.T) {
	expr := Parse("42")
	if expr.Interpret() != 42 {
		t.Errorf("Expected 42, got %d", expr.Interpret())
	}
}

// Test10: Nested expressions work correctly
func Test10(t *testing.T) {
	// (5 + 3) * 2 = 16 (left-to-right without precedence)
	expr := Parse("5 + 3 * 2")
	result := expr.Interpret()
	if result != 16 {
		t.Errorf("Expected 16 (left-to-right), got %d", result)
	}
}
`,
	translations: {
		ru: {
			title: 'Паттерн Interpreter (Интерпретатор)',
			description: `Реализуйте паттерн Interpreter на Go — определите представление грамматики и интерпретатор для неё.

**Вы реализуете:**

1. **Интерфейс Expression** - Метод Interpret
2. **NumberExpression** - Терминальное выражение
3. **AddExpression, SubtractExpression** - Нетерминальные выражения
4. **Parser** - Строит дерево выражений

**Пример использования:**

\`\`\`go
expr := Parse("5 + 3 - 2")	// парсим строку выражения
result := expr.Interpret()	// вычисляем дерево выражений
fmt.Println(result)	// 6

expr2 := Parse("10 * 2 + 5")	// другое выражение
result2 := expr2.Interpret()	// вычисляем
fmt.Println(result2)	// 25

expr3 := Parse("100 - 50 * 2")	// с вычитанием и умножением
result3 := expr3.Interpret()	// вычисляем
fmt.Println(result3)	// 100
\`\`\``,
			hint1: `**Понимание структуры Interpreter:**

Паттерн Interpreter имеет два типа выражений:

\`\`\`go
// 1. Терминальное выражение - листовые узлы со значениями
type NumberExpression struct {
	value int	// хранит реальное число
}

func (n *NumberExpression) Interpret() int {
	return n.value	// просто возвращает сохранённое значение
}

// 2. Нетерминальные выражения - составные узлы с дочерними
type AddExpression struct {
	left  Expression	// может быть Number или другая операция
	right Expression	// может быть Number или другая операция
}

func (a *AddExpression) Interpret() int {
	// Рекурсивно интерпретируем дочерние, затем комбинируем
	return a.left.Interpret() + a.right.Interpret()
}
\`\`\`

Терминальные выражения хранят значения, нетерминальные комбинируют дочерние.`,
			hint2: `**Полная реализация Parser:**

\`\`\`go
func Parse(expression string) Expression {
	tokens := strings.Fields(expression)	// ["5", "+", "3", "-", "2"]
	if len(tokens) == 0 {
		return &NumberExpression{value: 0}
	}

	// Начинаем с первого числа
	num, _ := strconv.Atoi(tokens[0])
	result := Expression(&NumberExpression{value: num})

	// Обрабатываем пары: (оператор, число)
	for i := 1; i < len(tokens)-1; i += 2 {
		operator := tokens[i]	// "+", "-", или "*"
		num, _ := strconv.Atoi(tokens[i+1])
		right := &NumberExpression{value: num}

		switch operator {
		case "+":
			result = &AddExpression{left: result, right: right}
		case "-":
			result = &SubtractExpression{left: result, right: right}
		case "*":
			result = &MultiplyExpression{left: result, right: right}
		}
	}
	return result
}

// "5 + 3 - 2" становится:
//       Subtract
//      /        \\
//    Add        2
//   /   \\
//  5     3
\`\`\`

Парсер строит левоассоциативное дерево.`,
			whyItMatters: `## Почему существует Interpreter

**Проблема: Вычисление предметно-ориентированных языков**

Без Interpreter вычисление выражений требует сложного монолитного кода:

\`\`\`go
// ❌ БЕЗ INTERPRETER - запутанная логика вычисления
func Evaluate(expr string) int {
	// Ручной парсинг и вычисление смешаны вместе
	parts := strings.Split(expr, " ")
	result := parseNumber(parts[0])
	for i := 1; i < len(parts); i += 2 {
		op := parts[i]
		num := parseNumber(parts[i+1])
		if op == "+" {
			result += num
		} else if op == "-" {
			result -= num
		}
		// Добавление новых операторов = изменение этого беспорядка
	}
	return result
}

// ✅ С INTERPRETER - чистая, расширяемая структура
type Expression interface {
	Interpret() int	// каждое выражение знает как себя вычислить
}

// Добавление новых операций = просто добавьте новый тип выражения!
type ModuloExpression struct {
	left, right Expression
}

func (m *ModuloExpression) Interpret() int {
	return m.left.Interpret() % m.right.Interpret()
}
\`\`\`

---

## Примеры из реального мира в Go

**1. Выполнение SQL запросов (database/sql):**
\`\`\`go
// Грамматика: SELECT, WHERE, JOIN, и т.д.
// Выражения: Column, Table, Condition
db.Query("SELECT name FROM users WHERE age > 18")
\`\`\`

**2. Регулярные выражения (regexp):**
\`\`\`go
// Грамматика: паттерны, квантификаторы, классы символов
// Выражения: Literal, Star, Plus, Alternation
re := regexp.MustCompile("[a-z]+")
\`\`\`

**3. Движки шаблонов (text/template):**
\`\`\`go
// Грамматика: переменные, условия, циклы
// Выражения: Text, Variable, Range, If
tmpl.Execute(w, data)
\`\`\`

**4. JSON Path запросы:**
\`\`\`go
// Грамматика: пути, фильтры, подстановочные знаки
// Выражения: Root, Child, Filter, Wildcard
jsonpath.Get(data, "$.users[*].name")
\`\`\`

---

## Продакшн паттерн: Boolean Expression Evaluator

\`\`\`go
package main

import (
	"fmt"
	"strings"
)

// BoolExpression вычисляется в булев результат
type BoolExpression interface {
	Interpret(context map[string]bool) bool
}

// Variable представляет булеву переменную
type Variable struct {
	Name string
}

func (v *Variable) Interpret(context map[string]bool) bool {
	return context[v.Name]
}

// Constant представляет булеву константу (true/false)
type Constant struct {
	Value bool
}

func (c *Constant) Interpret(context map[string]bool) bool {
	return c.Value
}

// And представляет логическое И
type And struct {
	Left  BoolExpression
	Right BoolExpression
}

func (a *And) Interpret(context map[string]bool) bool {
	return a.Left.Interpret(context) && a.Right.Interpret(context)
}

// Or представляет логическое ИЛИ
type Or struct {
	Left  BoolExpression
	Right BoolExpression
}

func (o *Or) Interpret(context map[string]bool) bool {
	return o.Left.Interpret(context) || o.Right.Interpret(context)
}

// Not представляет логическое НЕ
type Not struct {
	Operand BoolExpression
}

func (n *Not) Interpret(context map[string]bool) bool {
	return !n.Operand.Interpret(context)
}

// Rule представляет именованное бизнес-правило с выражением
type Rule struct {
	Name       string
	Expression BoolExpression
}

func (r *Rule) Evaluate(context map[string]bool) bool {
	return r.Expression.Interpret(context)
}

// RuleEngine управляет и вычисляет бизнес-правила
type RuleEngine struct {
	rules []Rule
}

func NewRuleEngine() *RuleEngine {
	return &RuleEngine{rules: make([]Rule, 0)}
}

func (e *RuleEngine) AddRule(name string, expr BoolExpression) {
	e.rules = append(e.rules, Rule{Name: name, Expression: expr})
}

func (e *RuleEngine) EvaluateAll(context map[string]bool) map[string]bool {
	results := make(map[string]bool)
	for _, rule := range e.rules {
		results[rule.Name] = rule.Evaluate(context)
	}
	return results
}

func (e *RuleEngine) GetPassingRules(context map[string]bool) []string {
	var passing []string
	for _, rule := range e.rules {
		if rule.Evaluate(context) {
			passing = append(passing, rule.Name)
		}
	}
	return passing
}

// ParseBoolExpr парсит простые булевы выражения
// Поддерживает: переменные, AND, OR, NOT, true, false
func ParseBoolExpr(expr string) BoolExpression {
	expr = strings.TrimSpace(expr)

	// Обработка NOT
	if strings.HasPrefix(expr, "NOT ") {
		operand := ParseBoolExpr(expr[4:])
		return &Not{Operand: operand}
	}

	// Обработка OR (низший приоритет)
	if idx := strings.Index(expr, " OR "); idx != -1 {
		left := ParseBoolExpr(expr[:idx])
		right := ParseBoolExpr(expr[idx+4:])
		return &Or{Left: left, Right: right}
	}

	// Обработка AND
	if idx := strings.Index(expr, " AND "); idx != -1 {
		left := ParseBoolExpr(expr[:idx])
		right := ParseBoolExpr(expr[idx+5:])
		return &And{Left: left, Right: right}
	}

	// Обработка констант
	if expr == "true" {
		return &Constant{Value: true}
	}
	if expr == "false" {
		return &Constant{Value: false}
	}

	// Обработка переменной
	return &Variable{Name: expr}
}

// Использование
func main() {
	// Создаём движок правил
	engine := NewRuleEngine()

	// Определяем бизнес-правила используя выражения
	// Правило 1: Пользователь может получить доступ к премиум контенту
	premiumAccess := &Or{
		Left:  &Variable{Name: "isPremium"},
		Right: &Variable{Name: "isAdmin"},
	}
	engine.AddRule("premium_access", premiumAccess)

	// Правило 2: Пользователь может скачивать файлы
	downloadAllowed := &And{
		Left:  &Variable{Name: "isVerified"},
		Right: &Not{Operand: &Variable{Name: "isBanned"}},
	}
	engine.AddRule("download_allowed", downloadAllowed)

	// Правило 3: Показывать рекламу (не премиум и не админ)
	showAds := &And{
		Left:  &Not{Operand: &Variable{Name: "isPremium"}},
		Right: &Not{Operand: &Variable{Name: "isAdmin"}},
	}
	engine.AddRule("show_ads", showAds)

	// Парсим выражение из строки
	rule4 := ParseBoolExpr("isActive AND isVerified OR isAdmin")
	engine.AddRule("can_comment", rule4)

	// Вычисляем правила для пользователя
	userContext := map[string]bool{
		"isPremium":  false,
		"isAdmin":    false,
		"isVerified": true,
		"isBanned":   false,
		"isActive":   true,
	}

	fmt.Println("User Context:", userContext)
	fmt.Println()

	// Вычисляем все правила
	results := engine.EvaluateAll(userContext)
	for name, passed := range results {
		status := "DENIED"
		if passed {
			status = "GRANTED"
		}
		fmt.Printf("Rule '%s': %s\\n", name, status)
	}

	// Получаем прошедшие правила
	fmt.Println("\\nPassing rules:", engine.GetPassingRules(userContext))
}
\`\`\`

---

## Распространённые ошибки

**1. Нет общего определения грамматики:**
\`\`\`go
// ❌ НЕПРАВИЛЬНО - несогласованная интерпретация
type BadExpr1 struct { value int }
func (e *BadExpr1) Interpret() int { return e.value }

type BadExpr2 struct { value int }
func (e *BadExpr2) Interpret() float64 { return float64(e.value) }

// ✅ ПРАВИЛЬНО - согласованный интерфейс
type Expression interface {
	Interpret() int	// все выражения возвращают один тип
}

type NumberExpr struct { value int }
func (n *NumberExpr) Interpret() int { return n.value }

type AddExpr struct { left, right Expression }
func (a *AddExpr) Interpret() int {
	return a.left.Interpret() + a.right.Interpret()
}
\`\`\`

**2. Отсутствует параметр контекста:**
\`\`\`go
// ❌ НЕПРАВИЛЬНО - не можем вычислить переменные
type Variable struct { name string }
func (v *Variable) Interpret() int {
	// Как получить значение переменной???
	return 0
}

// ✅ ПРАВИЛЬНО - используем контекст для переменных
type Variable struct { name string }
func (v *Variable) Interpret(ctx map[string]int) int {
	return ctx[v.name]	// поиск в контексте
}
\`\`\`

**3. Парсер создаёт неправильную структуру дерева:**
\`\`\`go
// ❌ НЕПРАВИЛЬНО - игнорирует приоритет операторов
// "2 + 3 * 4" должно быть 2 + (3 * 4) = 14
// Но наивный слева-направо даёт (2 + 3) * 4 = 20

// ✅ ПРАВИЛЬНО - правильно обрабатываем приоритет
func Parse(expr string) Expression {
	// Сначала обрабатываем * и /
	// Затем обрабатываем + и -
	// Или используем скобки в грамматике
}
\`\`\``
		},
		uz: {
			title: 'Interpreter (Tarjimon) Pattern',
			description: `Go tilida Interpreter patternini amalga oshiring — grammatika tasvirini va uning tarjimoni aniqlang.

**Siz amalga oshirasiz:**

1. **Expression interfeysi** - Interpret metodi
2. **NumberExpression** - Terminal ifoda
3. **AddExpression, SubtractExpression** - Noterminal ifodalar
4. **Parser** - Ifoda daraxtini quradi

**Foydalanish namunasi:**

\`\`\`go
expr := Parse("5 + 3 - 2")	// ifoda satrini tahlil qilamiz
result := expr.Interpret()	// ifoda daraxtini hisoblaymiz
fmt.Println(result)	// 6

expr2 := Parse("10 * 2 + 5")	// boshqa ifoda
result2 := expr2.Interpret()	// hisoblaymiz
fmt.Println(result2)	// 25

expr3 := Parse("100 - 50 * 2")	// ayirish va ko'paytirish bilan
result3 := expr3.Interpret()	// hisoblaymiz
fmt.Println(result3)	// 100
\`\`\``,
			hint1: `**Interpreter tuzilmasini tushunish:**

Interpreter patterni ikki turdagi ifodalarga ega:

\`\`\`go
// 1. Terminal ifoda - qiymatli barg tugunlar
type NumberExpression struct {
	value int	// haqiqiy raqamni saqlaydi
}

func (n *NumberExpression) Interpret() int {
	return n.value	// saqlangan qiymatni oddiy qaytaradi
}

// 2. Noterminal ifodalar - bolali murakkab tugunlar
type AddExpression struct {
	left  Expression	// Number yoki boshqa operatsiya bo'lishi mumkin
	right Expression	// Number yoki boshqa operatsiya bo'lishi mumkin
}

func (a *AddExpression) Interpret() int {
	// Bolalarni rekursiv interpret qilib, keyin birlashtiramiz
	return a.left.Interpret() + a.right.Interpret()
}
\`\`\`

Terminal ifodalar qiymatlarni saqlaydi, noterminallar bolalarni birlashtiradi.`,
			hint2: `**To'liq Parser realizatsiyasi:**

\`\`\`go
func Parse(expression string) Expression {
	tokens := strings.Fields(expression)	// ["5", "+", "3", "-", "2"]
	if len(tokens) == 0 {
		return &NumberExpression{value: 0}
	}

	// Birinchi raqamdan boshlaymiz
	num, _ := strconv.Atoi(tokens[0])
	result := Expression(&NumberExpression{value: num})

	// Juftlarni qayta ishlaymiz: (operator, raqam)
	for i := 1; i < len(tokens)-1; i += 2 {
		operator := tokens[i]	// "+", "-", yoki "*"
		num, _ := strconv.Atoi(tokens[i+1])
		right := &NumberExpression{value: num}

		switch operator {
		case "+":
			result = &AddExpression{left: result, right: right}
		case "-":
			result = &SubtractExpression{left: result, right: right}
		case "*":
			result = &MultiplyExpression{left: result, right: right}
		}
	}
	return result
}

// "5 + 3 - 2" ga aylanadi:
//       Subtract
//      /        \\
//    Add        2
//   /   \\
//  5     3
\`\`\`

Parser chap-assotsiativ daraxt quradi.`,
			whyItMatters: `## Nima uchun Interpreter mavjud

**Muammo: Soha-yo'naltirilgan tillarni baholash**

Interpreter'siz ifodalarni baholash murakkab, monolit kodini talab qiladi:

\`\`\`go
// ❌ INTERPRETER'SIZ - chalkash baholash mantiqi
func Evaluate(expr string) int {
	// Qo'lda tahlil va baholash birlashtirilgan
	parts := strings.Split(expr, " ")
	result := parseNumber(parts[0])
	for i := 1; i < len(parts); i += 2 {
		op := parts[i]
		num := parseNumber(parts[i+1])
		if op == "+" {
			result += num
		} else if op == "-" {
			result -= num
		}
		// Yangi operatorlar qo'shish = bu tartibsizlikni o'zgartirish
	}
	return result
}

// ✅ INTERPRETER BILAN - toza, kengaytiriladigan tuzilma
type Expression interface {
	Interpret() int	// har bir ifoda o'zini qanday baholashni biladi
}

// Yangi operatsiyalar qo'shish = faqat yangi ifoda turini qo'shing!
type ModuloExpression struct {
	left, right Expression
}

func (m *ModuloExpression) Interpret() int {
	return m.left.Interpret() % m.right.Interpret()
}
\`\`\`

---

## Go'da haqiqiy dunyo misollari

**1. SQL so'rovlarini bajarish (database/sql):**
\`\`\`go
// Grammatika: SELECT, WHERE, JOIN, va h.k.
// Ifodalar: Column, Table, Condition
db.Query("SELECT name FROM users WHERE age > 18")
\`\`\`

**2. Muntazam ifodalar (regexp):**
\`\`\`go
// Grammatika: patternlar, kvantifikatorlar, belgi sinflari
// Ifodalar: Literal, Star, Plus, Alternation
re := regexp.MustCompile("[a-z]+")
\`\`\`

**3. Shablon dvigatellari (text/template):**
\`\`\`go
// Grammatika: o'zgaruvchilar, shartlar, tsikllar
// Ifodalar: Text, Variable, Range, If
tmpl.Execute(w, data)
\`\`\`

**4. JSON Path so'rovlar:**
\`\`\`go
// Grammatika: yo'llar, filtrlar, joker belgilar
// Ifodalar: Root, Child, Filter, Wildcard
jsonpath.Get(data, "$.users[*].name")
\`\`\`

---

## Production Pattern: Boolean Expression Evaluator

\`\`\`go
package main

import (
	"fmt"
	"strings"
)

// BoolExpression mantiqiy natijaga baholanadi
type BoolExpression interface {
	Interpret(context map[string]bool) bool
}

// Variable mantiqiy o'zgaruvchini ifodalaydi
type Variable struct {
	Name string
}

func (v *Variable) Interpret(context map[string]bool) bool {
	return context[v.Name]
}

// Constant mantiqiy konstantani ifodalaydi (true/false)
type Constant struct {
	Value bool
}

func (c *Constant) Interpret(context map[string]bool) bool {
	return c.Value
}

// And mantiqiy VA ni ifodalaydi
type And struct {
	Left  BoolExpression
	Right BoolExpression
}

func (a *And) Interpret(context map[string]bool) bool {
	return a.Left.Interpret(context) && a.Right.Interpret(context)
}

// Or mantiqiy YOKI ni ifodalaydi
type Or struct {
	Left  BoolExpression
	Right BoolExpression
}

func (o *Or) Interpret(context map[string]bool) bool {
	return o.Left.Interpret(context) || o.Right.Interpret(context)
}

// Not mantiqiy EMAS ni ifodalaydi
type Not struct {
	Operand BoolExpression
}

func (n *Not) Interpret(context map[string]bool) bool {
	return !n.Operand.Interpret(context)
}

// Rule ifoda bilan nomlangan biznes qoidasini ifodalaydi
type Rule struct {
	Name       string
	Expression BoolExpression
}

func (r *Rule) Evaluate(context map[string]bool) bool {
	return r.Expression.Interpret(context)
}

// RuleEngine biznes qoidalarini boshqaradi va baholaydi
type RuleEngine struct {
	rules []Rule
}

func NewRuleEngine() *RuleEngine {
	return &RuleEngine{rules: make([]Rule, 0)}
}

func (e *RuleEngine) AddRule(name string, expr BoolExpression) {
	e.rules = append(e.rules, Rule{Name: name, Expression: expr})
}

func (e *RuleEngine) EvaluateAll(context map[string]bool) map[string]bool {
	results := make(map[string]bool)
	for _, rule := range e.rules {
		results[rule.Name] = rule.Evaluate(context)
	}
	return results
}

func (e *RuleEngine) GetPassingRules(context map[string]bool) []string {
	var passing []string
	for _, rule := range e.rules {
		if rule.Evaluate(context) {
			passing = append(passing, rule.Name)
		}
	}
	return passing
}

// ParseBoolExpr oddiy mantiqiy ifodalarni tahlil qiladi
// Qo'llab-quvvatlaydi: o'zgaruvchilar, AND, OR, NOT, true, false
func ParseBoolExpr(expr string) BoolExpression {
	expr = strings.TrimSpace(expr)

	// NOT ni qayta ishlash
	if strings.HasPrefix(expr, "NOT ") {
		operand := ParseBoolExpr(expr[4:])
		return &Not{Operand: operand}
	}

	// OR ni qayta ishlash (eng past ustuvorlik)
	if idx := strings.Index(expr, " OR "); idx != -1 {
		left := ParseBoolExpr(expr[:idx])
		right := ParseBoolExpr(expr[idx+4:])
		return &Or{Left: left, Right: right}
	}

	// AND ni qayta ishlash
	if idx := strings.Index(expr, " AND "); idx != -1 {
		left := ParseBoolExpr(expr[:idx])
		right := ParseBoolExpr(expr[idx+5:])
		return &And{Left: left, Right: right}
	}

	// Konstantalarni qayta ishlash
	if expr == "true" {
		return &Constant{Value: true}
	}
	if expr == "false" {
		return &Constant{Value: false}
	}

	// O'zgaruvchini qayta ishlash
	return &Variable{Name: expr}
}

// Foydalanish
func main() {
	// Qoidalar dvigatelini yaratamiz
	engine := NewRuleEngine()

	// Ifodalar yordamida biznes qoidalarini aniqlaymiz
	// Qoida 1: Foydalanuvchi premium kontentga kira oladi
	premiumAccess := &Or{
		Left:  &Variable{Name: "isPremium"},
		Right: &Variable{Name: "isAdmin"},
	}
	engine.AddRule("premium_access", premiumAccess)

	// Qoida 2: Foydalanuvchi fayllarni yuklab olishi mumkin
	downloadAllowed := &And{
		Left:  &Variable{Name: "isVerified"},
		Right: &Not{Operand: &Variable{Name: "isBanned"}},
	}
	engine.AddRule("download_allowed", downloadAllowed)

	// Qoida 3: Reklama ko'rsatish (premium emas va admin emas)
	showAds := &And{
		Left:  &Not{Operand: &Variable{Name: "isPremium"}},
		Right: &Not{Operand: &Variable{Name: "isAdmin"}},
	}
	engine.AddRule("show_ads", showAds)

	// Satrdan ifodani tahlil qilamiz
	rule4 := ParseBoolExpr("isActive AND isVerified OR isAdmin")
	engine.AddRule("can_comment", rule4)

	// Foydalanuvchi uchun qoidalarni baholaymiz
	userContext := map[string]bool{
		"isPremium":  false,
		"isAdmin":    false,
		"isVerified": true,
		"isBanned":   false,
		"isActive":   true,
	}

	fmt.Println("User Context:", userContext)
	fmt.Println()

	// Barcha qoidalarni baholaymiz
	results := engine.EvaluateAll(userContext)
	for name, passed := range results {
		status := "DENIED"
		if passed {
			status = "GRANTED"
		}
		fmt.Printf("Rule '%s': %s\\n", name, status)
	}

	// O'tgan qoidalarni olamiz
	fmt.Println("\\nPassing rules:", engine.GetPassingRules(userContext))
}
\`\`\`

---

## Keng tarqalgan xatolar

**1. Umumiy grammatika ta'rifi yo'q:**
\`\`\`go
// ❌ NOTO'G'RI - mos kelmaydigan interpretatsiya
type BadExpr1 struct { value int }
func (e *BadExpr1) Interpret() int { return e.value }

type BadExpr2 struct { value int }
func (e *BadExpr2) Interpret() float64 { return float64(e.value) }

// ✅ TO'G'RI - mos interfeys
type Expression interface {
	Interpret() int	// barcha ifodalar bir xil turni qaytaradi
}

type NumberExpr struct { value int }
func (n *NumberExpr) Interpret() int { return n.value }

type AddExpr struct { left, right Expression }
func (a *AddExpr) Interpret() int {
	return a.left.Interpret() + a.right.Interpret()
}
\`\`\`

**2. Kontekst parametri yo'qolgan:**
\`\`\`go
// ❌ NOTO'G'RI - o'zgaruvchilarni hisoblab bo'lmaydi
type Variable struct { name string }
func (v *Variable) Interpret() int {
	// O'zgaruvchi qiymatini qanday olish???
	return 0
}

// ✅ TO'G'RI - o'zgaruvchilar uchun kontekst ishlatamiz
type Variable struct { name string }
func (v *Variable) Interpret(ctx map[string]int) int {
	return ctx[v.name]	// kontekstda qidirish
}
\`\`\`

**3. Parser noto'g'ri daraxt tuzilmasini yaratadi:**
\`\`\`go
// ❌ NOTO'G'RI - operator ustuvorligini e'tiborsiz qoldiradi
// "2 + 3 * 4" 2 + (3 * 4) = 14 bo'lishi kerak
// Lekin sodda chapdan-o'ngga (2 + 3) * 4 = 20 beradi

// ✅ TO'G'RI - ustuvorlikni to'g'ri qayta ishlaymiz
func Parse(expr string) Expression {
	// Avval * va / ni qayta ishlaymiz
	// Keyin + va - ni qayta ishlaymiz
	// Yoki grammatikada qavslar ishlatamiz
}
\`\`\``
		}
	}
};

export default task;
