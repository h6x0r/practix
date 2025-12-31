import { Task } from '../../../../../../../types';

export const task: Task = {
	slug: 'java-dp-visitor',
	title: 'Visitor Pattern',
	difficulty: 'hard',
	tags: ['java', 'design-patterns', 'behavioral', 'visitor'],
	estimatedTime: '40m',
	isPremium: false,
	youtubeUrl: '',
	description: `## Visitor Pattern

The **Visitor Pattern** lets you define new operations without changing the classes of the elements on which it operates. It achieves this through "double dispatch" - the operation executed depends on both the visitor type and the element type.

---

### Key Components

| Component | Role |
|-----------|------|
| **Visitor** | Interface declaring visit methods for each element type |
| **ConcreteVisitor** | Implements operations for each element type |
| **Element** | Interface with accept(visitor) method |
| **ConcreteElement** | Calls visitor.visit(this) in accept() |

---

### Your Task

Implement a **Shape Calculator System** using the Visitor pattern:

1. **Shape** (Element): Interface with accept() method
2. **Circle, Rectangle** (ConcreteElements): Shapes that accept visitors
3. **AreaCalculator, PerimeterCalculator** (ConcreteVisitors): Calculate different properties

---

### Example Usage

\`\`\`java
Shape circle = new Circle(5.0);	// create circle with radius 5
Shape rectangle = new Rectangle(4.0, 6.0);	// create 4x6 rectangle

ShapeVisitor areaCalc = new AreaCalculator();	// visitor for areas
ShapeVisitor perimeterCalc = new PerimeterCalculator();	// visitor for perimeters

// Calculate areas using visitor
System.out.println(circle.accept(areaCalc));	// "Circle area: 78.54"
System.out.println(rectangle.accept(areaCalc));	// "Rectangle area: 24.00"

// Calculate perimeters using same shapes, different visitor
System.out.println(circle.accept(perimeterCalc));	// "Circle perimeter: 31.42"
System.out.println(rectangle.accept(perimeterCalc));	// "Rectangle perimeter: 20.00"
\`\`\`

---

### Key Insight

> The key trick is **double dispatch**: when \`circle.accept(areaCalc)\` is called, it becomes \`areaCalc.visit(this)\` inside Circle. This way, the correct visit() method is chosen based on BOTH the visitor type (AreaCalculator) AND the element type (Circle).`,
	initialCode: `interface ShapeVisitor {
    String visit(Circle circle);
    String visit(Rectangle rectangle);
}

interface Shape {
    String accept(ShapeVisitor visitor);
}

class Circle implements Shape {
    private double radius;

    public Circle(double radius) {
    }

    public double getRadius() { return radius; }

    @Override
    public String accept(ShapeVisitor visitor) {
        throw new UnsupportedOperationException("TODO");
    }
}

class Rectangle implements Shape {
    private double width, height;

    public Rectangle(double width, double height) {
    }

    public double getWidth() { return width; }
    public double getHeight() { return height; }

    @Override
    public String accept(ShapeVisitor visitor) {
        throw new UnsupportedOperationException("TODO");
    }
}

class AreaCalculator implements ShapeVisitor {
    @Override
    public String visit(Circle circle) {
        throw new UnsupportedOperationException("TODO");
    }

    @Override
    public String visit(Rectangle rectangle) {
        throw new UnsupportedOperationException("TODO");
    }
}

class PerimeterCalculator implements ShapeVisitor {
    @Override
    public String visit(Circle circle) {
        throw new UnsupportedOperationException("TODO");
    }

    @Override
    public String visit(Rectangle rectangle) {
        throw new UnsupportedOperationException("TODO");
    }
}`,
	solutionCode: `interface ShapeVisitor {	// Visitor interface - declares visit for each element type
    String visit(Circle circle);	// visit method for Circle elements
    String visit(Rectangle rectangle);	// visit method for Rectangle elements
}

interface Shape {	// Element interface - declares accept method
    String accept(ShapeVisitor visitor);	// accept visitor and return result
}

class Circle implements Shape {	// ConcreteElement - circle shape
    private double radius;	// circle's radius

    public Circle(double radius) {	// constructor with radius
        this.radius = radius;	// store radius value
    }

    public double getRadius() { return radius; }	// getter for radius

    @Override
    public String accept(ShapeVisitor visitor) {	// accept method - key to double dispatch
        return visitor.visit(this);	// call visitor's visit method, passing self
    }
}

class Rectangle implements Shape {	// ConcreteElement - rectangle shape
    private double width, height;	// rectangle dimensions

    public Rectangle(double width, double height) {	// constructor with dimensions
        this.width = width;	// store width
        this.height = height;	// store height
    }

    public double getWidth() { return width; }	// getter for width
    public double getHeight() { return height; }	// getter for height

    @Override
    public String accept(ShapeVisitor visitor) {	// accept method - key to double dispatch
        return visitor.visit(this);	// call visitor's visit method, passing self
    }
}

class AreaCalculator implements ShapeVisitor {	// ConcreteVisitor - calculates area
    @Override
    public String visit(Circle circle) {	// calculate circle area
        double area = Math.PI * circle.getRadius() * circle.getRadius();	// pi * r^2
        return String.format("Circle area: %.2f", area);	// format result
    }

    @Override
    public String visit(Rectangle rectangle) {	// calculate rectangle area
        double area = rectangle.getWidth() * rectangle.getHeight();	// width * height
        return String.format("Rectangle area: %.2f", area);	// format result
    }
}

class PerimeterCalculator implements ShapeVisitor {	// ConcreteVisitor - calculates perimeter
    @Override
    public String visit(Circle circle) {	// calculate circle perimeter (circumference)
        double perimeter = 2 * Math.PI * circle.getRadius();	// 2 * pi * r
        return String.format("Circle perimeter: %.2f", perimeter);	// format result
    }

    @Override
    public String visit(Rectangle rectangle) {	// calculate rectangle perimeter
        double perimeter = 2 * (rectangle.getWidth() + rectangle.getHeight());	// 2 * (w + h)
        return String.format("Rectangle perimeter: %.2f", perimeter);	// format result
    }
}`,
	hint1: `## Hint 1: The accept() Method - Double Dispatch

The accept() method is the key to the Visitor pattern. Each element simply calls the visitor's visit method with itself:

\`\`\`java
class Circle implements Shape {
    @Override
    public String accept(ShapeVisitor visitor) {	// accept any visitor
        return visitor.visit(this);	// call visit with "this" Circle
    }
}

class Rectangle implements Shape {
    @Override
    public String accept(ShapeVisitor visitor) {	// accept any visitor
        return visitor.visit(this);	// call visit with "this" Rectangle
    }
}
\`\`\`

By passing \`this\`, we enable the compiler to select the correct overloaded visit() method based on the actual type of the element.`,
	hint2: `## Hint 2: Implementing Visitor Operations

Each visitor implements specific logic for each element type:

\`\`\`java
class AreaCalculator implements ShapeVisitor {
    @Override
    public String visit(Circle circle) {	// area of circle
        double area = Math.PI * circle.getRadius() * circle.getRadius();	// pi * r^2
        return String.format("Circle area: %.2f", area);	// format result
    }

    @Override
    public String visit(Rectangle rectangle) {	// area of rectangle
        double area = rectangle.getWidth() * rectangle.getHeight();	// width * height
        return String.format("Rectangle area: %.2f", area);	// format result
    }
}
\`\`\`

Use getters to access element data. Format results with \`String.format("%.2f", value)\` for two decimal places.`,
	whyItMatters: `## Why Visitor Pattern Matters

### The Problem: Adding Operations Requires Modifying Element Classes

Without Visitor, adding new operations means modifying every element class:

\`\`\`java
// ❌ Without Visitor - operations inside element classes
interface Shape {	// every new operation changes this interface
    double getArea();	// operation 1
    double getPerimeter();	// operation 2
    String toJSON();	// operation 3 - must add to interface
    String toXML();	// operation 4 - must add to interface
    // Every new operation = change interface + all implementations!
}

class Circle implements Shape {	// circle must implement ALL operations
    private double radius;	// data

    @Override
    public double getArea() {	// operation 1
        return Math.PI * radius * radius;	// circle-specific
    }

    @Override
    public double getPerimeter() {	// operation 2
        return 2 * Math.PI * radius;	// circle-specific
    }

    @Override
    public String toJSON() {	// operation 3 - ADDED
        return "{\\"type\\":\\"circle\\",\\"radius\\":" + radius + "}";	// must add here
    }

    @Override
    public String toXML() {	// operation 4 - ADDED
        return "<circle radius=\\"" + radius + "\\"/>";	// must add here
    }
    // Adding toYAML() requires modifying Circle, Rectangle, Triangle...
}

class Rectangle implements Shape {	// rectangle must also implement ALL
    private double width, height;

    // Must implement ALL the same operations...
    // Every new shape class must implement ALL operations!
}

// Problems:
// 1. Adding operation = modify Shape + ALL ConcreteShapes
// 2. Classes grow with unrelated responsibilities
// 3. Violates Open/Closed Principle
// 4. Hard to add both new shapes AND new operations
\`\`\`

\`\`\`java
// ✅ With Visitor - operations separate from elements
interface ShapeVisitor {	// add operations here, not in Shape
    String visit(Circle circle);	// operation for Circle
    String visit(Rectangle rectangle);	// operation for Rectangle
}

interface Shape {	// stable interface - rarely changes
    String accept(ShapeVisitor visitor);	// only one method!
}

class Circle implements Shape {	// just data and accept
    private double radius;	// data

    public double getRadius() { return radius; }	// expose data via getter

    @Override
    public String accept(ShapeVisitor visitor) {	// same for all elements
        return visitor.visit(this);	// delegate to visitor
    }
    // Circle doesn't know about area, perimeter, JSON, XML!
}

// Adding new operation = add new visitor class
class AreaCalculator implements ShapeVisitor {	// operation 1
    public String visit(Circle c) { return "Area: " + Math.PI * c.getRadius() * c.getRadius(); }
    public String visit(Rectangle r) { return "Area: " + r.getWidth() * r.getHeight(); }
}

class JSONExporter implements ShapeVisitor {	// operation 3 - NEW CLASS, no changes to shapes!
    public String visit(Circle c) { return "{\\"radius\\":" + c.getRadius() + "}"; }
    public String visit(Rectangle r) { return "{\\"w\\":" + r.getWidth() + ",\\"h\\":" + r.getHeight() + "}"; }
}

// Benefits:
// 1. New operation = new visitor class (no changes to elements)
// 2. Operations grouped in one class
// 3. Open for extension (new visitors)
// 4. Elements stay simple (just data + accept)
\`\`\`

---

### Real-World Applications

| Application | Elements | Visitors | Use Case |
|-------------|----------|----------|----------|
| **Compiler AST** | Nodes (If, While, Call) | TypeChecker, CodeGenerator | Separate analysis from tree |
| **Document Export** | Paragraphs, Tables, Images | HTMLExporter, PDFExporter | Multiple export formats |
| **File System** | Files, Directories | SizeCalculator, SearchVisitor | Operations on file tree |
| **Shopping Cart** | Items, Discounts | TaxCalculator, Renderer | Calculate totals, display |
| **GUI Components** | Buttons, Inputs, Panels | Renderer, Validator | Theme rendering, validation |

---

### Production Pattern: AST Processor for Expression Language

\`\`\`java
import java.util.*;	// import utilities

// Element interface - AST node that accepts visitors
interface Expression {	// base for all expression nodes
    <T> T accept(ExpressionVisitor<T> visitor);	// generic return type for flexibility
}

// Visitor interface - generic to support different return types
interface ExpressionVisitor<T> {	// parameterized by return type
    T visitNumber(NumberExpr expr);	// visit literal number
    T visitBinaryOp(BinaryOpExpr expr);	// visit binary operation
    T visitVariable(VariableExpr expr);	// visit variable reference
    T visitFunctionCall(FunctionCallExpr expr);	// visit function call
}

// ConcreteElement - number literal
class NumberExpr implements Expression {	// represents a number like 42
    private final double value;	// the numeric value

    public NumberExpr(double value) {	// constructor
        this.value = value;	// store value
    }

    public double getValue() { return value; }	// getter

    @Override
    public <T> T accept(ExpressionVisitor<T> visitor) {	// accept visitor
        return visitor.visitNumber(this);	// dispatch to visitNumber
    }
}

// ConcreteElement - binary operation (a + b, a * b, etc.)
class BinaryOpExpr implements Expression {	// represents a op b
    private final String operator;	// +, -, *, /
    private final Expression left;	// left operand
    private final Expression right;	// right operand

    public BinaryOpExpr(String op, Expression left, Expression right) {	// constructor
        this.operator = op;	// store operator
        this.left = left;	// store left expression
        this.right = right;	// store right expression
    }

    public String getOperator() { return operator; }	// getter
    public Expression getLeft() { return left; }	// getter
    public Expression getRight() { return right; }	// getter

    @Override
    public <T> T accept(ExpressionVisitor<T> visitor) {	// accept visitor
        return visitor.visitBinaryOp(this);	// dispatch to visitBinaryOp
    }
}

// ConcreteElement - variable reference
class VariableExpr implements Expression {	// represents variable like 'x'
    private final String name;	// variable name

    public VariableExpr(String name) {	// constructor
        this.name = name;	// store name
    }

    public String getName() { return name; }	// getter

    @Override
    public <T> T accept(ExpressionVisitor<T> visitor) {	// accept visitor
        return visitor.visitVariable(this);	// dispatch to visitVariable
    }
}

// ConcreteElement - function call
class FunctionCallExpr implements Expression {	// represents fn(args)
    private final String functionName;	// function name
    private final List<Expression> arguments;	// arguments list

    public FunctionCallExpr(String name, List<Expression> args) {	// constructor
        this.functionName = name;	// store function name
        this.arguments = new ArrayList<>(args);	// defensive copy
    }

    public String getFunctionName() { return functionName; }	// getter
    public List<Expression> getArguments() { return arguments; }	// getter

    @Override
    public <T> T accept(ExpressionVisitor<T> visitor) {	// accept visitor
        return visitor.visitFunctionCall(this);	// dispatch to visitFunctionCall
    }
}

// ConcreteVisitor 1 - evaluates expression to a number
class Evaluator implements ExpressionVisitor<Double> {	// returns Double
    private final Map<String, Double> variables;	// variable bindings
    private final Map<String, java.util.function.Function<List<Double>, Double>> functions;	// function bindings

    public Evaluator() {	// constructor with empty bindings
        this.variables = new HashMap<>();	// empty variables
        this.functions = new HashMap<>();	// empty functions
        setupBuiltinFunctions();	// add standard functions
    }

    private void setupBuiltinFunctions() {	// add math functions
        functions.put("sqrt", args -> Math.sqrt(args.get(0)));	// square root
        functions.put("abs", args -> Math.abs(args.get(0)));	// absolute value
        functions.put("max", args -> Math.max(args.get(0), args.get(1)));	// maximum
        functions.put("min", args -> Math.min(args.get(0), args.get(1)));	// minimum
    }

    public void setVariable(String name, double value) {	// bind variable
        variables.put(name, value);	// store in map
    }

    @Override
    public Double visitNumber(NumberExpr expr) {	// evaluate number
        return expr.getValue();	// just return the value
    }

    @Override
    public Double visitBinaryOp(BinaryOpExpr expr) {	// evaluate binary op
        double left = expr.getLeft().accept(this);	// evaluate left recursively
        double right = expr.getRight().accept(this);	// evaluate right recursively
        switch (expr.getOperator()) {	// apply operator
            case "+": return left + right;	// addition
            case "-": return left - right;	// subtraction
            case "*": return left * right;	// multiplication
            case "/": return left / right;	// division
            default: throw new RuntimeException("Unknown operator: " + expr.getOperator());	// error
        }
    }

    @Override
    public Double visitVariable(VariableExpr expr) {	// evaluate variable
        if (!variables.containsKey(expr.getName())) {	// check if defined
            throw new RuntimeException("Undefined variable: " + expr.getName());	// error
        }
        return variables.get(expr.getName());	// return value
    }

    @Override
    public Double visitFunctionCall(FunctionCallExpr expr) {	// evaluate function call
        if (!functions.containsKey(expr.getFunctionName())) {	// check if defined
            throw new RuntimeException("Undefined function: " + expr.getFunctionName());	// error
        }
        List<Double> args = new ArrayList<>();	// evaluate arguments
        for (Expression arg : expr.getArguments()) {	// for each argument
            args.add(arg.accept(this));	// evaluate and collect
        }
        return functions.get(expr.getFunctionName()).apply(args);	// call function
    }
}

// ConcreteVisitor 2 - converts AST to string representation
class PrettyPrinter implements ExpressionVisitor<String> {	// returns String
    @Override
    public String visitNumber(NumberExpr expr) {	// print number
        return String.valueOf(expr.getValue());	// convert to string
    }

    @Override
    public String visitBinaryOp(BinaryOpExpr expr) {	// print binary op
        String left = expr.getLeft().accept(this);	// print left
        String right = expr.getRight().accept(this);	// print right
        return "(" + left + " " + expr.getOperator() + " " + right + ")";	// parenthesized
    }

    @Override
    public String visitVariable(VariableExpr expr) {	// print variable
        return expr.getName();	// just the name
    }

    @Override
    public String visitFunctionCall(FunctionCallExpr expr) {	// print function call
        List<String> argStrings = new ArrayList<>();	// collect arg strings
        for (Expression arg : expr.getArguments()) {	// for each argument
            argStrings.add(arg.accept(this));	// print and collect
        }
        return expr.getFunctionName() + "(" + String.join(", ", argStrings) + ")";	// fn(a, b, c)
    }
}

// ConcreteVisitor 3 - collects all variable names in expression
class VariableCollector implements ExpressionVisitor<Set<String>> {	// returns Set<String>
    @Override
    public Set<String> visitNumber(NumberExpr expr) {	// no variables in number
        return new HashSet<>();	// empty set
    }

    @Override
    public Set<String> visitBinaryOp(BinaryOpExpr expr) {	// collect from both sides
        Set<String> vars = new HashSet<>();	// collect here
        vars.addAll(expr.getLeft().accept(this));	// collect from left
        vars.addAll(expr.getRight().accept(this));	// collect from right
        return vars;	// return all
    }

    @Override
    public Set<String> visitVariable(VariableExpr expr) {	// found a variable
        Set<String> vars = new HashSet<>();	// create set
        vars.add(expr.getName());	// add this variable
        return vars;	// return set with this var
    }

    @Override
    public Set<String> visitFunctionCall(FunctionCallExpr expr) {	// collect from arguments
        Set<String> vars = new HashSet<>();	// collect here
        for (Expression arg : expr.getArguments()) {	// for each argument
            vars.addAll(arg.accept(this));	// collect variables
        }
        return vars;	// return all
    }
}

// Usage:
// Build AST for: sqrt(x * x + y * y)  (distance formula)
Expression xSquared = new BinaryOpExpr("*",	// x * x
    new VariableExpr("x"), new VariableExpr("x"));	// x times x
Expression ySquared = new BinaryOpExpr("*",	// y * y
    new VariableExpr("y"), new VariableExpr("y"));	// y times y
Expression sum = new BinaryOpExpr("+", xSquared, ySquared);	// x*x + y*y
Expression distance = new FunctionCallExpr("sqrt", Arrays.asList(sum));	// sqrt(sum)

// Use different visitors on same AST
PrettyPrinter printer = new PrettyPrinter();	// visitor 1
System.out.println(distance.accept(printer));	// "sqrt(((x * x) + (y * y)))"

VariableCollector collector = new VariableCollector();	// visitor 2
System.out.println(distance.accept(collector));	// [x, y]

Evaluator evaluator = new Evaluator();	// visitor 3
evaluator.setVariable("x", 3.0);	// x = 3
evaluator.setVariable("y", 4.0);	// y = 4
System.out.println(distance.accept(evaluator));	// 5.0 (3-4-5 triangle!)
\`\`\`

---

### Common Mistakes to Avoid

| Mistake | Problem | Solution |
|---------|---------|----------|
| Missing visit method | Incomplete visitor | Add visit for every element type |
| Not using double dispatch | Loses type information | Always call visitor.visit(this) |
| Element contains logic | Defeats purpose of visitor | Keep elements as data, logic in visitors |
| Visitor modifies elements | Side effects, hard to reason | Return results instead of mutating |
| Forgetting new element types | Runtime errors | Add to visitor interface, compiler catches missing implementations |`,
	order: 9,
	testCode: `import static org.junit.Assert.*;
import org.junit.Test;

// Test1: Circle accept calls AreaCalculator correctly
class Test1 {
    @Test
    public void test() {
        Circle circle = new Circle(5.0);
        ShapeVisitor areaCalc = new AreaCalculator();
        String result = circle.accept(areaCalc);
        assertTrue(result.contains("Circle area:"));
    }
}

// Test2: Rectangle accept calls AreaCalculator correctly
class Test2 {
    @Test
    public void test() {
        Rectangle rectangle = new Rectangle(4.0, 6.0);
        ShapeVisitor areaCalc = new AreaCalculator();
        String result = rectangle.accept(areaCalc);
        assertTrue(result.contains("Rectangle area:"));
    }
}

// Test3: Circle area calculation is correct (pi * r^2)
class Test3 {
    @Test
    public void test() {
        Circle circle = new Circle(5.0);
        ShapeVisitor areaCalc = new AreaCalculator();
        String result = circle.accept(areaCalc);
        assertTrue(result.contains("78.5"));
    }
}

// Test4: Rectangle area calculation is correct (w * h)
class Test4 {
    @Test
    public void test() {
        Rectangle rectangle = new Rectangle(4.0, 6.0);
        ShapeVisitor areaCalc = new AreaCalculator();
        String result = rectangle.accept(areaCalc);
        assertTrue(result.contains("24.00"));
    }
}

// Test5: Circle perimeter calculation is correct (2 * pi * r)
class Test5 {
    @Test
    public void test() {
        Circle circle = new Circle(5.0);
        ShapeVisitor perimeterCalc = new PerimeterCalculator();
        String result = circle.accept(perimeterCalc);
        assertTrue(result.contains("31.4"));
    }
}

// Test6: Rectangle perimeter calculation is correct (2 * (w + h))
class Test6 {
    @Test
    public void test() {
        Rectangle rectangle = new Rectangle(4.0, 6.0);
        ShapeVisitor perimeterCalc = new PerimeterCalculator();
        String result = rectangle.accept(perimeterCalc);
        assertTrue(result.contains("20.00"));
    }
}

// Test7: Circle getRadius returns correct value
class Test7 {
    @Test
    public void test() {
        Circle circle = new Circle(7.5);
        assertEquals(7.5, circle.getRadius(), 0.01);
    }
}

// Test8: Rectangle getWidth and getHeight work
class Test8 {
    @Test
    public void test() {
        Rectangle rectangle = new Rectangle(3.0, 5.0);
        assertEquals(3.0, rectangle.getWidth(), 0.01);
        assertEquals(5.0, rectangle.getHeight(), 0.01);
    }
}

// Test9: Same shape with different visitors
class Test9 {
    @Test
    public void test() {
        Circle circle = new Circle(10.0);
        String area = circle.accept(new AreaCalculator());
        String perimeter = circle.accept(new PerimeterCalculator());
        assertTrue(area.contains("area"));
        assertTrue(perimeter.contains("perimeter"));
    }
}

// Test10: Different shapes with same visitor
class Test10 {
    @Test
    public void test() {
        ShapeVisitor areaCalc = new AreaCalculator();
        String circleResult = new Circle(1.0).accept(areaCalc);
        String rectResult = new Rectangle(2.0, 3.0).accept(areaCalc);
        assertTrue(circleResult.contains("Circle"));
        assertTrue(rectResult.contains("Rectangle"));
    }
}
`,
	translations: {
		ru: {
			title: 'Паттерн Visitor (Посетитель)',
			description: `## Паттерн Visitor (Посетитель)

Паттерн **Visitor** позволяет определять новые операции без изменения классов элементов, над которыми они работают. Он достигает этого через "двойную диспетчеризацию" - выполняемая операция зависит как от типа посетителя, так и от типа элемента.

---

### Ключевые компоненты

| Компонент | Роль |
|-----------|------|
| **Visitor** | Интерфейс, объявляющий методы visit для каждого типа элемента |
| **ConcreteVisitor** | Реализует операции для каждого типа элемента |
| **Element** | Интерфейс с методом accept(visitor) |
| **ConcreteElement** | Вызывает visitor.visit(this) в accept() |

---

### Ваша задача

Реализуйте **Систему расчёта фигур** используя паттерн Visitor:

1. **Shape** (Element): Интерфейс с методом accept()
2. **Circle, Rectangle** (ConcreteElements): Фигуры, принимающие посетителей
3. **AreaCalculator, PerimeterCalculator** (ConcreteVisitors): Рассчитывают разные свойства

---

### Пример использования

\`\`\`java
Shape circle = new Circle(5.0);	// создать круг с радиусом 5
Shape rectangle = new Rectangle(4.0, 6.0);	// создать прямоугольник 4x6

ShapeVisitor areaCalc = new AreaCalculator();	// посетитель для площадей
ShapeVisitor perimeterCalc = new PerimeterCalculator();	// посетитель для периметров

// Вычислить площади используя посетителя
System.out.println(circle.accept(areaCalc));	// "Circle area: 78.54"
System.out.println(rectangle.accept(areaCalc));	// "Rectangle area: 24.00"

// Вычислить периметры используя те же фигуры, другой посетитель
System.out.println(circle.accept(perimeterCalc));	// "Circle perimeter: 31.42"
System.out.println(rectangle.accept(perimeterCalc));	// "Rectangle perimeter: 20.00"
\`\`\`

---

### Ключевая идея

> Ключевой приём - **двойная диспетчеризация**: когда вызывается \`circle.accept(areaCalc)\`, внутри Circle это становится \`areaCalc.visit(this)\`. Таким образом, правильный метод visit() выбирается на основе ОБОИХ типов - и посетителя (AreaCalculator) И элемента (Circle).`,
			hint1: `## Подсказка 1: Метод accept() - Двойная диспетчеризация

Метод accept() - ключ к паттерну Visitor. Каждый элемент просто вызывает метод visit посетителя с самим собой:

\`\`\`java
class Circle implements Shape {
    @Override
    public String accept(ShapeVisitor visitor) {	// принять любого посетителя
        return visitor.visit(this);	// вызвать visit с "этим" Circle
    }
}

class Rectangle implements Shape {
    @Override
    public String accept(ShapeVisitor visitor) {	// принять любого посетителя
        return visitor.visit(this);	// вызвать visit с "этим" Rectangle
    }
}
\`\`\`

Передавая \`this\`, мы позволяем компилятору выбрать правильный перегруженный метод visit() на основе фактического типа элемента.`,
			hint2: `## Подсказка 2: Реализация операций посетителя

Каждый посетитель реализует специфичную логику для каждого типа элемента:

\`\`\`java
class AreaCalculator implements ShapeVisitor {
    @Override
    public String visit(Circle circle) {	// площадь круга
        double area = Math.PI * circle.getRadius() * circle.getRadius();	// pi * r^2
        return String.format("Circle area: %.2f", area);	// форматировать результат
    }

    @Override
    public String visit(Rectangle rectangle) {	// площадь прямоугольника
        double area = rectangle.getWidth() * rectangle.getHeight();	// ширина * высота
        return String.format("Rectangle area: %.2f", area);	// форматировать результат
    }
}
\`\`\`

Используйте геттеры для доступа к данным элемента. Форматируйте результаты через \`String.format("%.2f", value)\` для двух знаков после запятой.`,
			whyItMatters: `## Почему паттерн Visitor важен

### Проблема: Добавление операций требует модификации классов элементов

Без Visitor добавление новых операций означает модификацию каждого класса элемента:

\`\`\`java
// ❌ Без Visitor - операции внутри классов элементов
interface Shape {	// каждая новая операция меняет этот интерфейс
    double getArea();	// операция 1
    double getPerimeter();	// операция 2
    String toJSON();	// операция 3 - нужно добавить в интерфейс
    String toXML();	// операция 4 - нужно добавить в интерфейс
    // Каждая новая операция = изменить интерфейс + все реализации!
}

class Circle implements Shape {	// круг должен реализовать ВСЕ операции
    private double radius;	// данные

    @Override
    public double getArea() {	// операция 1
        return Math.PI * radius * radius;	// специфично для круга
    }

    @Override
    public double getPerimeter() {	// операция 2
        return 2 * Math.PI * radius;	// специфично для круга
    }

    @Override
    public String toJSON() {	// операция 3 - ДОБАВЛЕНА
        return "{\\"type\\":\\"circle\\",\\"radius\\":" + radius + "}";	// нужно добавить сюда
    }

    @Override
    public String toXML() {	// операция 4 - ДОБАВЛЕНА
        return "<circle radius=\\"" + radius + "\\"/>";	// нужно добавить сюда
    }
    // Добавление toYAML() требует модификации Circle, Rectangle, Triangle...
}

class Rectangle implements Shape {	// прямоугольник тоже должен реализовать ВСЕ
    private double width, height;

    // Должен реализовать ВСЕ те же операции...
    // Каждый новый класс фигуры должен реализовать ВСЕ операции!
}

// Проблемы:
// 1. Добавление операции = модификация Shape + ВСЕХ ConcreteShape
// 2. Классы растут с несвязанными обязанностями
// 3. Нарушает принцип Open/Closed
// 4. Трудно добавлять и новые фигуры И новые операции
\`\`\`

\`\`\`java
// ✅ С Visitor - операции отделены от элементов
interface ShapeVisitor {	// добавлять операции сюда, не в Shape
    String visit(Circle circle);	// операция для Circle
    String visit(Rectangle rectangle);	// операция для Rectangle
}

interface Shape {	// стабильный интерфейс - редко меняется
    String accept(ShapeVisitor visitor);	// только один метод!
}

class Circle implements Shape {	// только данные и accept
    private double radius;	// данные

    public double getRadius() { return radius; }	// открыть данные через геттер

    @Override
    public String accept(ShapeVisitor visitor) {	// одинаково для всех элементов
        return visitor.visit(this);	// делегировать посетителю
    }
    // Circle не знает о площади, периметре, JSON, XML!
}

// Добавление новой операции = добавить новый класс посетителя
class AreaCalculator implements ShapeVisitor {	// операция 1
    public String visit(Circle c) { return "Area: " + Math.PI * c.getRadius() * c.getRadius(); }
    public String visit(Rectangle r) { return "Area: " + r.getWidth() * r.getHeight(); }
}

class JSONExporter implements ShapeVisitor {	// операция 3 - НОВЫЙ КЛАСС, без изменений фигур!
    public String visit(Circle c) { return "{\\"radius\\":" + c.getRadius() + "}"; }
    public String visit(Rectangle r) { return "{\\"w\\":" + r.getWidth() + ",\\"h\\":" + r.getHeight() + "}"; }
}

// Преимущества:
// 1. Новая операция = новый класс посетителя (без изменений элементов)
// 2. Операции сгруппированы в одном классе
// 3. Открыт для расширения (новые посетители)
// 4. Элементы остаются простыми (только данные + accept)
\`\`\`

---

### Применение в реальном мире

| Применение | Элементы | Посетители | Случай использования |
|------------|----------|------------|----------------------|
| **AST компилятора** | Узлы (If, While, Call) | TypeChecker, CodeGenerator | Отделить анализ от дерева |
| **Экспорт документов** | Параграфы, Таблицы, Изображения | HTMLExporter, PDFExporter | Множество форматов экспорта |
| **Файловая система** | Файлы, Директории | SizeCalculator, SearchVisitor | Операции над файловым деревом |
| **Корзина покупок** | Товары, Скидки | TaxCalculator, Renderer | Расчёт итогов, отображение |
| **GUI компоненты** | Кнопки, Поля ввода, Панели | Renderer, Validator | Отрисовка темы, валидация |

---

### Продакшн паттерн: AST-процессор для языка выражений

\`\`\`java
import java.util.*;	// импорт утилит

// Интерфейс Element - узел AST, принимающий посетителей
interface Expression {	// база для всех узлов выражений
    <T> T accept(ExpressionVisitor<T> visitor);	// обобщённый тип возврата для гибкости
}

// Интерфейс Visitor - обобщённый для поддержки разных типов возврата
interface ExpressionVisitor<T> {	// параметризован типом возврата
    T visitNumber(NumberExpr expr);	// посетить числовой литерал
    T visitBinaryOp(BinaryOpExpr expr);	// посетить бинарную операцию
    T visitVariable(VariableExpr expr);	// посетить ссылку на переменную
    T visitFunctionCall(FunctionCallExpr expr);	// посетить вызов функции
}

// ConcreteElement - числовой литерал
class NumberExpr implements Expression {	// представляет число типа 42
    private final double value;	// числовое значение

    public NumberExpr(double value) {	// конструктор
        this.value = value;	// сохранить значение
    }

    public double getValue() { return value; }	// геттер

    @Override
    public <T> T accept(ExpressionVisitor<T> visitor) {	// принять посетителя
        return visitor.visitNumber(this);	// диспетчеризация в visitNumber
    }
}

// ConcreteElement - бинарная операция (a + b, a * b и т.д.)
class BinaryOpExpr implements Expression {	// представляет a op b
    private final String operator;	// +, -, *, /
    private final Expression left;	// левый операнд
    private final Expression right;	// правый операнд

    public BinaryOpExpr(String op, Expression left, Expression right) {	// конструктор
        this.operator = op;	// сохранить оператор
        this.left = left;	// сохранить левое выражение
        this.right = right;	// сохранить правое выражение
    }

    public String getOperator() { return operator; }	// геттер
    public Expression getLeft() { return left; }	// геттер
    public Expression getRight() { return right; }	// геттер

    @Override
    public <T> T accept(ExpressionVisitor<T> visitor) {	// принять посетителя
        return visitor.visitBinaryOp(this);	// диспетчеризация в visitBinaryOp
    }
}

// ConcreteElement - ссылка на переменную
class VariableExpr implements Expression {	// представляет переменную типа 'x'
    private final String name;	// имя переменной

    public VariableExpr(String name) {	// конструктор
        this.name = name;	// сохранить имя
    }

    public String getName() { return name; }	// геттер

    @Override
    public <T> T accept(ExpressionVisitor<T> visitor) {	// принять посетителя
        return visitor.visitVariable(this);	// диспетчеризация в visitVariable
    }
}

// ConcreteElement - вызов функции
class FunctionCallExpr implements Expression {	// представляет fn(args)
    private final String functionName;	// имя функции
    private final List<Expression> arguments;	// список аргументов

    public FunctionCallExpr(String name, List<Expression> args) {	// конструктор
        this.functionName = name;	// сохранить имя функции
        this.arguments = new ArrayList<>(args);	// защитная копия
    }

    public String getFunctionName() { return functionName; }	// геттер
    public List<Expression> getArguments() { return arguments; }	// геттер

    @Override
    public <T> T accept(ExpressionVisitor<T> visitor) {	// принять посетителя
        return visitor.visitFunctionCall(this);	// диспетчеризация в visitFunctionCall
    }
}

// ConcreteVisitor 1 - вычисляет выражение в число
class Evaluator implements ExpressionVisitor<Double> {	// возвращает Double
    private final Map<String, Double> variables;	// привязки переменных
    private final Map<String, java.util.function.Function<List<Double>, Double>> functions;	// привязки функций

    public Evaluator() {	// конструктор с пустыми привязками
        this.variables = new HashMap<>();	// пустые переменные
        this.functions = new HashMap<>();	// пустые функции
        setupBuiltinFunctions();	// добавить стандартные функции
    }

    private void setupBuiltinFunctions() {	// добавить математические функции
        functions.put("sqrt", args -> Math.sqrt(args.get(0)));	// квадратный корень
        functions.put("abs", args -> Math.abs(args.get(0)));	// абсолютное значение
        functions.put("max", args -> Math.max(args.get(0), args.get(1)));	// максимум
        functions.put("min", args -> Math.min(args.get(0), args.get(1)));	// минимум
    }

    public void setVariable(String name, double value) {	// привязать переменную
        variables.put(name, value);	// сохранить в карте
    }

    @Override
    public Double visitNumber(NumberExpr expr) {	// вычислить число
        return expr.getValue();	// просто вернуть значение
    }

    @Override
    public Double visitBinaryOp(BinaryOpExpr expr) {	// вычислить бинарную операцию
        double left = expr.getLeft().accept(this);	// вычислить левое рекурсивно
        double right = expr.getRight().accept(this);	// вычислить правое рекурсивно
        switch (expr.getOperator()) {	// применить оператор
            case "+": return left + right;	// сложение
            case "-": return left - right;	// вычитание
            case "*": return left * right;	// умножение
            case "/": return left / right;	// деление
            default: throw new RuntimeException("Unknown operator: " + expr.getOperator());	// ошибка
        }
    }

    @Override
    public Double visitVariable(VariableExpr expr) {	// вычислить переменную
        if (!variables.containsKey(expr.getName())) {	// проверить определена ли
            throw new RuntimeException("Undefined variable: " + expr.getName());	// ошибка
        }
        return variables.get(expr.getName());	// вернуть значение
    }

    @Override
    public Double visitFunctionCall(FunctionCallExpr expr) {	// вычислить вызов функции
        if (!functions.containsKey(expr.getFunctionName())) {	// проверить определена ли
            throw new RuntimeException("Undefined function: " + expr.getFunctionName());	// ошибка
        }
        List<Double> args = new ArrayList<>();	// вычислить аргументы
        for (Expression arg : expr.getArguments()) {	// для каждого аргумента
            args.add(arg.accept(this));	// вычислить и собрать
        }
        return functions.get(expr.getFunctionName()).apply(args);	// вызвать функцию
    }
}

// ConcreteVisitor 2 - преобразует AST в строковое представление
class PrettyPrinter implements ExpressionVisitor<String> {	// возвращает String
    @Override
    public String visitNumber(NumberExpr expr) {	// вывести число
        return String.valueOf(expr.getValue());	// преобразовать в строку
    }

    @Override
    public String visitBinaryOp(BinaryOpExpr expr) {	// вывести бинарную операцию
        String left = expr.getLeft().accept(this);	// вывести левое
        String right = expr.getRight().accept(this);	// вывести правое
        return "(" + left + " " + expr.getOperator() + " " + right + ")";	// в скобках
    }

    @Override
    public String visitVariable(VariableExpr expr) {	// вывести переменную
        return expr.getName();	// просто имя
    }

    @Override
    public String visitFunctionCall(FunctionCallExpr expr) {	// вывести вызов функции
        List<String> argStrings = new ArrayList<>();	// собрать строки аргументов
        for (Expression arg : expr.getArguments()) {	// для каждого аргумента
            argStrings.add(arg.accept(this));	// вывести и собрать
        }
        return expr.getFunctionName() + "(" + String.join(", ", argStrings) + ")";	// fn(a, b, c)
    }
}

// ConcreteVisitor 3 - собирает все имена переменных в выражении
class VariableCollector implements ExpressionVisitor<Set<String>> {	// возвращает Set<String>
    @Override
    public Set<String> visitNumber(NumberExpr expr) {	// нет переменных в числе
        return new HashSet<>();	// пустое множество
    }

    @Override
    public Set<String> visitBinaryOp(BinaryOpExpr expr) {	// собрать с обеих сторон
        Set<String> vars = new HashSet<>();	// собирать сюда
        vars.addAll(expr.getLeft().accept(this));	// собрать слева
        vars.addAll(expr.getRight().accept(this));	// собрать справа
        return vars;	// вернуть все
    }

    @Override
    public Set<String> visitVariable(VariableExpr expr) {	// нашли переменную
        Set<String> vars = new HashSet<>();	// создать множество
        vars.add(expr.getName());	// добавить эту переменную
        return vars;	// вернуть множество с этой переменной
    }

    @Override
    public Set<String> visitFunctionCall(FunctionCallExpr expr) {	// собрать из аргументов
        Set<String> vars = new HashSet<>();	// собирать сюда
        for (Expression arg : expr.getArguments()) {	// для каждого аргумента
            vars.addAll(arg.accept(this));	// собрать переменные
        }
        return vars;	// вернуть все
    }
}

// Использование:
// Построить AST для: sqrt(x * x + y * y)  (формула расстояния)
Expression xSquared = new BinaryOpExpr("*",	// x * x
    new VariableExpr("x"), new VariableExpr("x"));	// x умножить на x
Expression ySquared = new BinaryOpExpr("*",	// y * y
    new VariableExpr("y"), new VariableExpr("y"));	// y умножить на y
Expression sum = new BinaryOpExpr("+", xSquared, ySquared);	// x*x + y*y
Expression distance = new FunctionCallExpr("sqrt", Arrays.asList(sum));	// sqrt(sum)

// Использовать разных посетителей на том же AST
PrettyPrinter printer = new PrettyPrinter();	// посетитель 1
System.out.println(distance.accept(printer));	// "sqrt(((x * x) + (y * y)))"

VariableCollector collector = new VariableCollector();	// посетитель 2
System.out.println(distance.accept(collector));	// [x, y]

Evaluator evaluator = new Evaluator();	// посетитель 3
evaluator.setVariable("x", 3.0);	// x = 3
evaluator.setVariable("y", 4.0);	// y = 4
System.out.println(distance.accept(evaluator));	// 5.0 (треугольник 3-4-5!)
\`\`\`

---

### Распространённые ошибки

| Ошибка | Проблема | Решение |
|--------|----------|---------|
| Отсутствует метод visit | Неполный посетитель | Добавить visit для каждого типа элемента |
| Не используется двойная диспетчеризация | Теряется информация о типе | Всегда вызывать visitor.visit(this) |
| Элемент содержит логику | Нарушает цель посетителя | Держать элементы как данные, логику в посетителях |
| Посетитель модифицирует элементы | Побочные эффекты, трудно понимать | Возвращать результаты вместо мутации |
| Забыли новые типы элементов | Ошибки времени выполнения | Добавить в интерфейс посетителя, компилятор поймает отсутствующие реализации |`
		},
		uz: {
			title: 'Visitor Pattern',
			description: `## Visitor Pattern

**Visitor Pattern** sizga u ishlaydigan elementlar klasslarini o'zgartirmasdan yangi operatsiyalarni belgilash imkonini beradi. U buni "ikki marta jo'natish" orqali amalga oshiradi - bajariladigan operatsiya ham visitor turi, ham element turiga bog'liq.

---

### Asosiy komponentlar

| Komponent | Vazifa |
|-----------|--------|
| **Visitor** | Har bir element turi uchun visit metodlarini e'lon qiluvchi interfeys |
| **ConcreteVisitor** | Har bir element turi uchun operatsiyalarni amalga oshiradi |
| **Element** | accept(visitor) metodiga ega interfeys |
| **ConcreteElement** | accept() da visitor.visit(this) ni chaqiradi |

---

### Vazifangiz

Visitor patternidan foydalanib **Shakl hisoblagich tizimi** amalga oshiring:

1. **Shape** (Element): accept() metodiga ega interfeys
2. **Circle, Rectangle** (ConcreteElements): Visitorlarni qabul qiluvchi shakllar
3. **AreaCalculator, PerimeterCalculator** (ConcreteVisitors): Turli xususiyatlarni hisoblaydi

---

### Foydalanish namunasi

\`\`\`java
Shape circle = new Circle(5.0);	// radius 5 li aylana yaratish
Shape rectangle = new Rectangle(4.0, 6.0);	// 4x6 to'rtburchak yaratish

ShapeVisitor areaCalc = new AreaCalculator();	// yuzalar uchun visitor
ShapeVisitor perimeterCalc = new PerimeterCalculator();	// perimetrlar uchun visitor

// Visitor yordamida yuzalarni hisoblash
System.out.println(circle.accept(areaCalc));	// "Circle area: 78.54"
System.out.println(rectangle.accept(areaCalc));	// "Rectangle area: 24.00"

// Perimetrlarni bir xil shakllar, boshqa visitor bilan hisoblash
System.out.println(circle.accept(perimeterCalc));	// "Circle perimeter: 31.42"
System.out.println(rectangle.accept(perimeterCalc));	// "Rectangle perimeter: 20.00"
\`\`\`

---

### Asosiy tushuncha

> Asosiy hiyla - **ikki marta jo'natish**: \`circle.accept(areaCalc)\` chaqirilganda, Circle ichida u \`areaCalc.visit(this)\` ga aylanadi. Shu tarzda, to'g'ri visit() metodi HAM visitor turi (AreaCalculator) HAM element turi (Circle) asosida tanlanadi.`,
			hint1: `## Maslahat 1: accept() metodi - Ikki marta jo'natish

accept() metodi Visitor patternining kaliti. Har bir element shunchaki visitorning visit metodini o'zi bilan chaqiradi:

\`\`\`java
class Circle implements Shape {
    @Override
    public String accept(ShapeVisitor visitor) {	// har qanday visitorni qabul qilish
        return visitor.visit(this);	// "bu" Circle bilan visit chaqirish
    }
}

class Rectangle implements Shape {
    @Override
    public String accept(ShapeVisitor visitor) {	// har qanday visitorni qabul qilish
        return visitor.visit(this);	// "bu" Rectangle bilan visit chaqirish
    }
}
\`\`\`

\`this\` ni uzatish orqali biz kompilyatorga elementning haqiqiy turi asosida to'g'ri overload qilingan visit() metodini tanlash imkonini beramiz.`,
			hint2: `## Maslahat 2: Visitor operatsiyalarini amalga oshirish

Har bir visitor har bir element turi uchun maxsus mantiqni amalga oshiradi:

\`\`\`java
class AreaCalculator implements ShapeVisitor {
    @Override
    public String visit(Circle circle) {	// aylana yuzasi
        double area = Math.PI * circle.getRadius() * circle.getRadius();	// pi * r^2
        return String.format("Circle area: %.2f", area);	// natijani formatlash
    }

    @Override
    public String visit(Rectangle rectangle) {	// to'rtburchak yuzasi
        double area = rectangle.getWidth() * rectangle.getHeight();	// eni * bo'yi
        return String.format("Rectangle area: %.2f", area);	// natijani formatlash
    }
}
\`\`\`

Element ma'lumotlariga kirish uchun getterlardan foydalaning. Ikki o'nlik raqam uchun natijalarni \`String.format("%.2f", value)\` bilan formatlang.`,
			whyItMatters: `## Nima uchun Visitor Pattern muhim

### Muammo: Operatsiyalar qo'shish element klasslarini o'zgartirishni talab qiladi

Visitorsiz yangi operatsiyalar qo'shish har bir element klassini o'zgartirishni anglatadi:

\`\`\`java
// ❌ Visitorsiz - operatsiyalar element klasslari ichida
interface Shape {	// har bir yangi operatsiya bu interfeysni o'zgartiradi
    double getArea();	// operatsiya 1
    double getPerimeter();	// operatsiya 2
    String toJSON();	// operatsiya 3 - interfeysga qo'shish kerak
    String toXML();	// operatsiya 4 - interfeysga qo'shish kerak
    // Har bir yangi operatsiya = interfeys + barcha realizatsiyalarni o'zgartirish!
}

class Circle implements Shape {	// aylana BARCHA operatsiyalarni amalga oshirishi kerak
    private double radius;	// ma'lumotlar

    @Override
    public double getArea() {	// operatsiya 1
        return Math.PI * radius * radius;	// aylanaga xos
    }

    @Override
    public double getPerimeter() {	// operatsiya 2
        return 2 * Math.PI * radius;	// aylanaga xos
    }

    @Override
    public String toJSON() {	// operatsiya 3 - QO'SHILDI
        return "{\\"type\\":\\"circle\\",\\"radius\\":" + radius + "}";	// bu yerga qo'shish kerak
    }

    @Override
    public String toXML() {	// operatsiya 4 - QO'SHILDI
        return "<circle radius=\\"" + radius + "\\"/>";	// bu yerga qo'shish kerak
    }
    // toYAML() qo'shish Circle, Rectangle, Triangle ni o'zgartirishni talab qiladi...
}

class Rectangle implements Shape {	// to'rtburchak ham BARCHASINI amalga oshirishi kerak
    private double width, height;

    // BARCHA bir xil operatsiyalarni amalga oshirishi kerak...
    // Har bir yangi shakl klassi BARCHA operatsiyalarni amalga oshirishi kerak!
}

// Muammolar:
// 1. Operatsiya qo'shish = Shape + BARCHA ConcreteShape larni o'zgartirish
// 2. Klasslar bog'liq bo'lmagan mas'uliyatlar bilan o'sadi
// 3. Open/Closed printsipini buzadi
// 4. Ham yangi shakllar HAM yangi operatsiyalar qo'shish qiyin
\`\`\`

\`\`\`java
// ✅ Visitor bilan - operatsiyalar elementlardan ajratilgan
interface ShapeVisitor {	// operatsiyalarni bu yerga qo'shish, Shape ga emas
    String visit(Circle circle);	// Circle uchun operatsiya
    String visit(Rectangle rectangle);	// Rectangle uchun operatsiya
}

interface Shape {	// barqaror interfeys - kamdan-kam o'zgaradi
    String accept(ShapeVisitor visitor);	// faqat bitta metod!
}

class Circle implements Shape {	// faqat ma'lumotlar va accept
    private double radius;	// ma'lumotlar

    public double getRadius() { return radius; }	// ma'lumotlarni getter orqali ochish

    @Override
    public String accept(ShapeVisitor visitor) {	// barcha elementlar uchun bir xil
        return visitor.visit(this);	// visitorga delegatsiya
    }
    // Circle yuza, perimetr, JSON, XML haqida bilmaydi!
}

// Yangi operatsiya qo'shish = yangi visitor klassi qo'shish
class AreaCalculator implements ShapeVisitor {	// operatsiya 1
    public String visit(Circle c) { return "Area: " + Math.PI * c.getRadius() * c.getRadius(); }
    public String visit(Rectangle r) { return "Area: " + r.getWidth() * r.getHeight(); }
}

class JSONExporter implements ShapeVisitor {	// operatsiya 3 - YANGI KLASS, shakllarga o'zgartirish yo'q!
    public String visit(Circle c) { return "{\\"radius\\":" + c.getRadius() + "}"; }
    public String visit(Rectangle r) { return "{\\"w\\":" + r.getWidth() + ",\\"h\\":" + r.getHeight() + "}"; }
}

// Afzalliklar:
// 1. Yangi operatsiya = yangi visitor klassi (elementlarga o'zgartirish yo'q)
// 2. Operatsiyalar bitta klassda guruhlangan
// 3. Kengaytirish uchun ochiq (yangi visitorlar)
// 4. Elementlar oddiy qoladi (faqat ma'lumotlar + accept)
\`\`\`

---

### Haqiqiy dunyo qo'llanilishi

| Qo'llanilish | Elementlar | Visitorlar | Foydalanish holati |
|--------------|------------|------------|---------------------|
| **Kompilyator AST** | Tugunlar (If, While, Call) | TypeChecker, CodeGenerator | Tahlilni daraxtdan ajratish |
| **Hujjat eksporti** | Paragraflar, Jadvallar, Rasmlar | HTMLExporter, PDFExporter | Ko'p eksport formatlari |
| **Fayl tizimi** | Fayllar, Kataloglar | SizeCalculator, SearchVisitor | Fayl daraxti ustida operatsiyalar |
| **Xarid savatchasi** | Mahsulotlar, Chegirmalar | TaxCalculator, Renderer | Jami hisoblash, ko'rsatish |
| **GUI komponentlari** | Tugmalar, Kiritish maydonlari, Panellar | Renderer, Validator | Mavzu ko'rsatish, validatsiya |

---

### Production Pattern: Ifoda tili uchun AST protsessor

\`\`\`java
import java.util.*;	// utilitalarni import qilish

// Element interfeysi - visitorlarni qabul qiluvchi AST tuguni
interface Expression {	// barcha ifoda tugunlari uchun baza
    <T> T accept(ExpressionVisitor<T> visitor);	// moslashuvchanlik uchun umumiy qaytish turi
}

// Visitor interfeysi - turli qaytish turlarini qo'llab-quvvatlash uchun umumiy
interface ExpressionVisitor<T> {	// qaytish turi bo'yicha parametrlangan
    T visitNumber(NumberExpr expr);	// son literalini tashrif buyurish
    T visitBinaryOp(BinaryOpExpr expr);	// ikkilik operatsiyani tashrif buyurish
    T visitVariable(VariableExpr expr);	// o'zgaruvchi referensini tashrif buyurish
    T visitFunctionCall(FunctionCallExpr expr);	// funksiya chaqiruvini tashrif buyurish
}

// ConcreteElement - son literali
class NumberExpr implements Expression {	// 42 kabi sonni ifodalaydi
    private final double value;	// raqamli qiymat

    public NumberExpr(double value) {	// konstruktor
        this.value = value;	// qiymatni saqlash
    }

    public double getValue() { return value; }	// getter

    @Override
    public <T> T accept(ExpressionVisitor<T> visitor) {	// visitorni qabul qilish
        return visitor.visitNumber(this);	// visitNumber ga jo'natish
    }
}

// ConcreteElement - ikkilik operatsiya (a + b, a * b va h.k.)
class BinaryOpExpr implements Expression {	// a op b ni ifodalaydi
    private final String operator;	// +, -, *, /
    private final Expression left;	// chap operand
    private final Expression right;	// o'ng operand

    public BinaryOpExpr(String op, Expression left, Expression right) {	// konstruktor
        this.operator = op;	// operatorni saqlash
        this.left = left;	// chap ifodani saqlash
        this.right = right;	// o'ng ifodani saqlash
    }

    public String getOperator() { return operator; }	// getter
    public Expression getLeft() { return left; }	// getter
    public Expression getRight() { return right; }	// getter

    @Override
    public <T> T accept(ExpressionVisitor<T> visitor) {	// visitorni qabul qilish
        return visitor.visitBinaryOp(this);	// visitBinaryOp ga jo'natish
    }
}

// ConcreteElement - o'zgaruvchi referensi
class VariableExpr implements Expression {	// 'x' kabi o'zgaruvchini ifodalaydi
    private final String name;	// o'zgaruvchi nomi

    public VariableExpr(String name) {	// konstruktor
        this.name = name;	// nomni saqlash
    }

    public String getName() { return name; }	// getter

    @Override
    public <T> T accept(ExpressionVisitor<T> visitor) {	// visitorni qabul qilish
        return visitor.visitVariable(this);	// visitVariable ga jo'natish
    }
}

// ConcreteElement - funksiya chaqiruvi
class FunctionCallExpr implements Expression {	// fn(args) ni ifodalaydi
    private final String functionName;	// funksiya nomi
    private final List<Expression> arguments;	// argumentlar ro'yxati

    public FunctionCallExpr(String name, List<Expression> args) {	// konstruktor
        this.functionName = name;	// funksiya nomini saqlash
        this.arguments = new ArrayList<>(args);	// himoyaviy nusxa
    }

    public String getFunctionName() { return functionName; }	// getter
    public List<Expression> getArguments() { return arguments; }	// getter

    @Override
    public <T> T accept(ExpressionVisitor<T> visitor) {	// visitorni qabul qilish
        return visitor.visitFunctionCall(this);	// visitFunctionCall ga jo'natish
    }
}

// ConcreteVisitor 1 - ifodani songa hisoblaydi
class Evaluator implements ExpressionVisitor<Double> {	// Double qaytaradi
    private final Map<String, Double> variables;	// o'zgaruvchi bog'lanishlari
    private final Map<String, java.util.function.Function<List<Double>, Double>> functions;	// funksiya bog'lanishlari

    public Evaluator() {	// bo'sh bog'lanishlar bilan konstruktor
        this.variables = new HashMap<>();	// bo'sh o'zgaruvchilar
        this.functions = new HashMap<>();	// bo'sh funksiyalar
        setupBuiltinFunctions();	// standart funksiyalarni qo'shish
    }

    private void setupBuiltinFunctions() {	// matematik funksiyalarni qo'shish
        functions.put("sqrt", args -> Math.sqrt(args.get(0)));	// kvadrat ildiz
        functions.put("abs", args -> Math.abs(args.get(0)));	// absolyut qiymat
        functions.put("max", args -> Math.max(args.get(0), args.get(1)));	// maksimum
        functions.put("min", args -> Math.min(args.get(0), args.get(1)));	// minimum
    }

    public void setVariable(String name, double value) {	// o'zgaruvchini bog'lash
        variables.put(name, value);	// xaritada saqlash
    }

    @Override
    public Double visitNumber(NumberExpr expr) {	// sonni hisoblash
        return expr.getValue();	// shunchaki qiymatni qaytarish
    }

    @Override
    public Double visitBinaryOp(BinaryOpExpr expr) {	// ikkilik operatsiyani hisoblash
        double left = expr.getLeft().accept(this);	// chapni rekursiv hisoblash
        double right = expr.getRight().accept(this);	// o'ngni rekursiv hisoblash
        switch (expr.getOperator()) {	// operatorni qo'llash
            case "+": return left + right;	// qo'shish
            case "-": return left - right;	// ayirish
            case "*": return left * right;	// ko'paytirish
            case "/": return left / right;	// bo'lish
            default: throw new RuntimeException("Unknown operator: " + expr.getOperator());	// xato
        }
    }

    @Override
    public Double visitVariable(VariableExpr expr) {	// o'zgaruvchini hisoblash
        if (!variables.containsKey(expr.getName())) {	// aniqlangan-aniqlanmaganini tekshirish
            throw new RuntimeException("Undefined variable: " + expr.getName());	// xato
        }
        return variables.get(expr.getName());	// qiymatni qaytarish
    }

    @Override
    public Double visitFunctionCall(FunctionCallExpr expr) {	// funksiya chaqiruvini hisoblash
        if (!functions.containsKey(expr.getFunctionName())) {	// aniqlangan-aniqlanmaganini tekshirish
            throw new RuntimeException("Undefined function: " + expr.getFunctionName());	// xato
        }
        List<Double> args = new ArrayList<>();	// argumentlarni hisoblash
        for (Expression arg : expr.getArguments()) {	// har bir argument uchun
            args.add(arg.accept(this));	// hisoblash va yig'ish
        }
        return functions.get(expr.getFunctionName()).apply(args);	// funksiyani chaqirish
    }
}

// ConcreteVisitor 2 - ASTni satr ko'rinishiga o'zgartiradi
class PrettyPrinter implements ExpressionVisitor<String> {	// String qaytaradi
    @Override
    public String visitNumber(NumberExpr expr) {	// sonni chop etish
        return String.valueOf(expr.getValue());	// satrga o'zgartirish
    }

    @Override
    public String visitBinaryOp(BinaryOpExpr expr) {	// ikkilik operatsiyani chop etish
        String left = expr.getLeft().accept(this);	// chapni chop etish
        String right = expr.getRight().accept(this);	// o'ngni chop etish
        return "(" + left + " " + expr.getOperator() + " " + right + ")";	// qavsda
    }

    @Override
    public String visitVariable(VariableExpr expr) {	// o'zgaruvchini chop etish
        return expr.getName();	// shunchaki nom
    }

    @Override
    public String visitFunctionCall(FunctionCallExpr expr) {	// funksiya chaqiruvini chop etish
        List<String> argStrings = new ArrayList<>();	// argument satrlarini yig'ish
        for (Expression arg : expr.getArguments()) {	// har bir argument uchun
            argStrings.add(arg.accept(this));	// chop etish va yig'ish
        }
        return expr.getFunctionName() + "(" + String.join(", ", argStrings) + ")";	// fn(a, b, c)
    }
}

// ConcreteVisitor 3 - ifodadagi barcha o'zgaruvchi nomlarini yig'adi
class VariableCollector implements ExpressionVisitor<Set<String>> {	// Set<String> qaytaradi
    @Override
    public Set<String> visitNumber(NumberExpr expr) {	// sonda o'zgaruvchilar yo'q
        return new HashSet<>();	// bo'sh to'plam
    }

    @Override
    public Set<String> visitBinaryOp(BinaryOpExpr expr) {	// ikki tomondan yig'ish
        Set<String> vars = new HashSet<>();	// bu yerga yig'ish
        vars.addAll(expr.getLeft().accept(this));	// chapdan yig'ish
        vars.addAll(expr.getRight().accept(this));	// o'ngdan yig'ish
        return vars;	// hammasini qaytarish
    }

    @Override
    public Set<String> visitVariable(VariableExpr expr) {	// o'zgaruvchi topildi
        Set<String> vars = new HashSet<>();	// to'plam yaratish
        vars.add(expr.getName());	// bu o'zgaruvchini qo'shish
        return vars;	// bu o'zgaruvchi bilan to'plamni qaytarish
    }

    @Override
    public Set<String> visitFunctionCall(FunctionCallExpr expr) {	// argumentlardan yig'ish
        Set<String> vars = new HashSet<>();	// bu yerga yig'ish
        for (Expression arg : expr.getArguments()) {	// har bir argument uchun
            vars.addAll(arg.accept(this));	// o'zgaruvchilarni yig'ish
        }
        return vars;	// hammasini qaytarish
    }
}

// Foydalanish:
// AST qurish: sqrt(x * x + y * y) (masofa formulasi)
Expression xSquared = new BinaryOpExpr("*",	// x * x
    new VariableExpr("x"), new VariableExpr("x"));	// x ko'paytirish x
Expression ySquared = new BinaryOpExpr("*",	// y * y
    new VariableExpr("y"), new VariableExpr("y"));	// y ko'paytirish y
Expression sum = new BinaryOpExpr("+", xSquared, ySquared);	// x*x + y*y
Expression distance = new FunctionCallExpr("sqrt", Arrays.asList(sum));	// sqrt(sum)

// Bir xil AST da turli visitorlardan foydalanish
PrettyPrinter printer = new PrettyPrinter();	// visitor 1
System.out.println(distance.accept(printer));	// "sqrt(((x * x) + (y * y)))"

VariableCollector collector = new VariableCollector();	// visitor 2
System.out.println(distance.accept(collector));	// [x, y]

Evaluator evaluator = new Evaluator();	// visitor 3
evaluator.setVariable("x", 3.0);	// x = 3
evaluator.setVariable("y", 4.0);	// y = 4
System.out.println(distance.accept(evaluator));	// 5.0 (3-4-5 uchburchak!)
\`\`\`

---

### Oldini olish kerak bo'lgan keng tarqalgan xatolar

| Xato | Muammo | Yechim |
|------|--------|--------|
| visit metodi yo'q | To'liq bo'lmagan visitor | Har bir element turi uchun visit qo'shish |
| Ikki marta jo'natish ishlatilmayapti | Tur ma'lumoti yo'qoladi | Doimo visitor.visit(this) chaqirish |
| Element mantiq saqlaydi | Visitor maqsadini buzadi | Elementlarni ma'lumot sifatida saqlash, mantiqni visitorlarda |
| Visitor elementlarni o'zgartiradi | Yon ta'sirlar, tushunish qiyin | Mutatsiya o'rniga natijalarni qaytarish |
| Yangi element turlari unutilgan | Runtime xatolar | Visitor interfeysiga qo'shish, kompilyator yetishmayotgan realizatsiyalarni ushlaydi |`
		}
	}
};

export default task;
