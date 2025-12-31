import { Task } from '../../../../types';

export const task: Task = {
    slug: 'java-factory-pattern',
    title: 'Factory Pattern',
    difficulty: 'medium',
    tags: ['java', 'design-patterns', 'creational', 'factory', 'abstract-factory'],
    estimatedTime: '35m',
    isPremium: false,
    youtubeUrl: '',
    description: `# Factory Pattern

The Factory pattern provides an interface for creating objects in a superclass, but allows subclasses to alter the type of objects that will be created. The Abstract Factory provides an interface for creating families of related objects without specifying their concrete classes.

## Requirements:
1. Create Shape interface and implementations:
   1.1. Circle, Rectangle, Triangle classes
   1.2. Common draw() method

2. Implement Factory Method pattern:
   2.1. ShapeFactory with createShape(String type) method
   2.2. Returns appropriate shape based on type

3. Implement Abstract Factory pattern:
   3.1. UIFactory interface for creating UI components
   3.2. WindowsFactory and MacFactory implementations
   3.3. Create Button and Checkbox components for each platform

4. Demonstrate both patterns with examples

## Example Output:
\`\`\`
=== Factory Method Pattern ===
Drawing: Circle with radius 5
Drawing: Rectangle with width 10 and height 6
Drawing: Triangle with base 8 and height 7

=== Abstract Factory Pattern ===
Windows UI:
Windows Button clicked!
Windows Checkbox checked!

Mac UI:
Mac Button clicked!
Mac Checkbox checked!
\`\`\``,
    initialCode: `// TODO: Create Shape interface and implementations

// TODO: Create ShapeFactory with Factory Method

// TODO: Create Abstract Factory for UI components

public class FactoryPattern {
    public static void main(String[] args) {
        // TODO: Demonstrate Factory Method

        // TODO: Demonstrate Abstract Factory
    }
}`,
    solutionCode: `// ===== Factory Method Pattern =====

// Product interface
interface Shape {
    void draw();
}

// Concrete products
class Circle implements Shape {
    private int radius;

    public Circle(int radius) {
        this.radius = radius;
    }

    @Override
    public void draw() {
        System.out.println("Drawing: Circle with radius " + radius);
    }
}

class Rectangle implements Shape {
    private int width, height;

    public Rectangle(int width, int height) {
        this.width = width;
        this.height = height;
    }

    @Override
    public void draw() {
        System.out.println("Drawing: Rectangle with width " + width + " and height " + height);
    }
}

class Triangle implements Shape {
    private int base, height;

    public Triangle(int base, int height) {
        this.base = base;
        this.height = height;
    }

    @Override
    public void draw() {
        System.out.println("Drawing: Triangle with base " + base + " and height " + height);
    }
}

// Factory class with Factory Method
class ShapeFactory {
    // Factory Method - creates objects based on input
    public Shape createShape(String type, int... dimensions) {
        if (type == null) {
            return null;
        }
        switch (type.toLowerCase()) {
            case "circle":
                return new Circle(dimensions[0]);
            case "rectangle":
                return new Rectangle(dimensions[0], dimensions[1]);
            case "triangle":
                return new Triangle(dimensions[0], dimensions[1]);
            default:
                throw new IllegalArgumentException("Unknown shape: " + type);
        }
    }
}

// ===== Abstract Factory Pattern =====

// Abstract products
interface Button {
    void click();
}

interface Checkbox {
    void check();
}

// Concrete products for Windows
class WindowsButton implements Button {
    @Override
    public void click() {
        System.out.println("Windows Button clicked!");
    }
}

class WindowsCheckbox implements Checkbox {
    @Override
    public void check() {
        System.out.println("Windows Checkbox checked!");
    }
}

// Concrete products for Mac
class MacButton implements Button {
    @Override
    public void click() {
        System.out.println("Mac Button clicked!");
    }
}

class MacCheckbox implements Checkbox {
    @Override
    public void check() {
        System.out.println("Mac Checkbox checked!");
    }
}

// Abstract Factory interface
interface UIFactory {
    Button createButton();
    Checkbox createCheckbox();
}

// Concrete Factory for Windows
class WindowsFactory implements UIFactory {
    @Override
    public Button createButton() {
        return new WindowsButton();
    }

    @Override
    public Checkbox createCheckbox() {
        return new WindowsCheckbox();
    }
}

// Concrete Factory for Mac
class MacFactory implements UIFactory {
    @Override
    public Button createButton() {
        return new MacButton();
    }

    @Override
    public Checkbox createCheckbox() {
        return new MacCheckbox();
    }
}

public class FactoryPattern {
    public static void main(String[] args) {
        System.out.println("=== Factory Method Pattern ===");

        // Using Factory Method
        ShapeFactory shapeFactory = new ShapeFactory();

        Shape circle = shapeFactory.createShape("circle", 5);
        circle.draw();

        Shape rectangle = shapeFactory.createShape("rectangle", 10, 6);
        rectangle.draw();

        Shape triangle = shapeFactory.createShape("triangle", 8, 7);
        triangle.draw();

        System.out.println("\\n=== Abstract Factory Pattern ===");

        // Using Abstract Factory for Windows
        UIFactory windowsFactory = new WindowsFactory();
        System.out.println("Windows UI:");
        Button windowsButton = windowsFactory.createButton();
        Checkbox windowsCheckbox = windowsFactory.createCheckbox();
        windowsButton.click();
        windowsCheckbox.check();

        System.out.println("\\nMac UI:");
        // Using Abstract Factory for Mac
        UIFactory macFactory = new MacFactory();
        Button macButton = macFactory.createButton();
        Checkbox macCheckbox = macFactory.createCheckbox();
        macButton.click();
        macCheckbox.check();
    }
}`,
    hint1: `Factory Method: Create a factory class with a method that returns different objects based on input parameters. Use switch/if statements to decide which concrete class to instantiate.`,
    hint2: `Abstract Factory: Create an interface with methods for creating related objects. Implement concrete factories that return platform-specific implementations of these objects.`,
    whyItMatters: `Factory patterns are fundamental for decoupling object creation from usage. Factory Method provides flexibility in object creation, while Abstract Factory ensures consistency across families of related objects. These patterns are essential in frameworks, dependency injection, and plugin architectures. Understanding them is crucial for writing flexible, maintainable, and testable code.

**Production Pattern:**
\`\`\`java
// Factory Method - centralized object creation
ShapeFactory factory = new ShapeFactory();
Shape circle = factory.createShape("circle", 5);
circle.draw();

// Abstract Factory - families of related objects
UIFactory windowsFactory = new WindowsFactory();
Button button = windowsFactory.createButton();
Checkbox checkbox = windowsFactory.createCheckbox();

// Easy to switch between implementations
UIFactory macFactory = new MacFactory();
\`\`\`

**Practical Benefits:**
- Separates creation logic from business logic
- Easy to add new types without changing client code
- Abstract Factory ensures component compatibility
- Used in Spring Framework, JDBC, and many APIs`,
    order: 3,
    testCode: `import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;
import java.io.*;

// Test 1: ShapeFactory creates Circle correctly
class Test1 {
    @Test
    void testShapeFactoryCreatesCircle() {
        ShapeFactory factory = new ShapeFactory();
        Shape circle = factory.createShape("circle", 5);
        assertNotNull(circle);
        assertTrue(circle instanceof Circle);
    }
}

// Test 2: ShapeFactory creates Rectangle correctly
class Test2 {
    @Test
    void testShapeFactoryCreatesRectangle() {
        ShapeFactory factory = new ShapeFactory();
        Shape rectangle = factory.createShape("rectangle", 10, 6);
        assertNotNull(rectangle);
        assertTrue(rectangle instanceof Rectangle);
    }
}

// Test 3: ShapeFactory creates Triangle correctly
class Test3 {
    @Test
    void testShapeFactoryCreatesTriangle() {
        ShapeFactory factory = new ShapeFactory();
        Shape triangle = factory.createShape("triangle", 8, 7);
        assertNotNull(triangle);
        assertTrue(triangle instanceof Triangle);
    }
}

// Test 4: ShapeFactory returns null for null type
class Test4 {
    @Test
    void testShapeFactoryNullType() {
        ShapeFactory factory = new ShapeFactory();
        Shape shape = factory.createShape(null);
        assertNull(shape);
    }
}

// Test 5: ShapeFactory throws exception for unknown type
class Test5 {
    @Test
    void testShapeFactoryUnknownType() {
        ShapeFactory factory = new ShapeFactory();
        assertThrows(IllegalArgumentException.class, () -> {
            factory.createShape("hexagon", 5);
        });
    }
}

// Test 6: WindowsFactory creates WindowsButton
class Test6 {
    @Test
    void testWindowsFactoryCreatesButton() {
        UIFactory factory = new WindowsFactory();
        Button button = factory.createButton();
        assertNotNull(button);
        assertTrue(button instanceof WindowsButton);
    }
}

// Test 7: WindowsFactory creates WindowsCheckbox
class Test7 {
    @Test
    void testWindowsFactoryCreatesCheckbox() {
        UIFactory factory = new WindowsFactory();
        Checkbox checkbox = factory.createCheckbox();
        assertNotNull(checkbox);
        assertTrue(checkbox instanceof WindowsCheckbox);
    }
}

// Test 8: MacFactory creates MacButton
class Test8 {
    @Test
    void testMacFactoryCreatesButton() {
        UIFactory factory = new MacFactory();
        Button button = factory.createButton();
        assertNotNull(button);
        assertTrue(button instanceof MacButton);
    }
}

// Test 9: MacFactory creates MacCheckbox
class Test9 {
    @Test
    void testMacFactoryCreatesCheckbox() {
        UIFactory factory = new MacFactory();
        Checkbox checkbox = factory.createCheckbox();
        assertNotNull(checkbox);
        assertTrue(checkbox instanceof MacCheckbox);
    }
}

// Test 10: Shape draw method produces output
class Test10 {
    @Test
    void testShapeDrawProducesOutput() {
        ShapeFactory factory = new ShapeFactory();
        Shape circle = factory.createShape("circle", 5);

        ByteArrayOutputStream outContent = new ByteArrayOutputStream();
        PrintStream originalOut = System.out;
        System.setOut(new PrintStream(outContent));

        circle.draw();

        System.setOut(originalOut);
        String output = outContent.toString();
        assertTrue(output.contains("Circle") && output.contains("5"));
    }
}`,
    translations: {
        ru: {
            title: 'Паттерн Factory',
            solutionCode: `// ===== Паттерн Factory Method =====

// Интерфейс продукта
interface Shape {
    void draw();
}

// Конкретные продукты
class Circle implements Shape {
    private int radius;

    public Circle(int radius) {
        this.radius = radius;
    }

    @Override
    public void draw() {
        System.out.println("Drawing: Circle with radius " + radius);
    }
}

class Rectangle implements Shape {
    private int width, height;

    public Rectangle(int width, int height) {
        this.width = width;
        this.height = height;
    }

    @Override
    public void draw() {
        System.out.println("Drawing: Rectangle with width " + width + " and height " + height);
    }
}

class Triangle implements Shape {
    private int base, height;

    public Triangle(int base, int height) {
        this.base = base;
        this.height = height;
    }

    @Override
    public void draw() {
        System.out.println("Drawing: Triangle with base " + base + " and height " + height);
    }
}

// Класс фабрики с Factory Method
class ShapeFactory {
    // Factory Method - создает объекты на основе входных данных
    public Shape createShape(String type, int... dimensions) {
        if (type == null) {
            return null;
        }
        switch (type.toLowerCase()) {
            case "circle":
                return new Circle(dimensions[0]);
            case "rectangle":
                return new Rectangle(dimensions[0], dimensions[1]);
            case "triangle":
                return new Triangle(dimensions[0], dimensions[1]);
            default:
                throw new IllegalArgumentException("Unknown shape: " + type);
        }
    }
}

// ===== Паттерн Abstract Factory =====

// Абстрактные продукты
interface Button {
    void click();
}

interface Checkbox {
    void check();
}

// Конкретные продукты для Windows
class WindowsButton implements Button {
    @Override
    public void click() {
        System.out.println("Windows Button clicked!");
    }
}

class WindowsCheckbox implements Checkbox {
    @Override
    public void check() {
        System.out.println("Windows Checkbox checked!");
    }
}

// Конкретные продукты для Mac
class MacButton implements Button {
    @Override
    public void click() {
        System.out.println("Mac Button clicked!");
    }
}

class MacCheckbox implements Checkbox {
    @Override
    public void check() {
        System.out.println("Mac Checkbox checked!");
    }
}

// Интерфейс Abstract Factory
interface UIFactory {
    Button createButton();
    Checkbox createCheckbox();
}

// Конкретная фабрика для Windows
class WindowsFactory implements UIFactory {
    @Override
    public Button createButton() {
        return new WindowsButton();
    }

    @Override
    public Checkbox createCheckbox() {
        return new WindowsCheckbox();
    }
}

// Конкретная фабрика для Mac
class MacFactory implements UIFactory {
    @Override
    public Button createButton() {
        return new MacButton();
    }

    @Override
    public Checkbox createCheckbox() {
        return new MacCheckbox();
    }
}

public class FactoryPattern {
    public static void main(String[] args) {
        System.out.println("=== Паттерн Factory Method ===");

        // Использование Factory Method
        ShapeFactory shapeFactory = new ShapeFactory();

        Shape circle = shapeFactory.createShape("circle", 5);
        circle.draw();

        Shape rectangle = shapeFactory.createShape("rectangle", 10, 6);
        rectangle.draw();

        Shape triangle = shapeFactory.createShape("triangle", 8, 7);
        triangle.draw();

        System.out.println("\\n=== Паттерн Abstract Factory ===");

        // Использование Abstract Factory для Windows
        UIFactory windowsFactory = new WindowsFactory();
        System.out.println("Windows UI:");
        Button windowsButton = windowsFactory.createButton();
        Checkbox windowsCheckbox = windowsFactory.createCheckbox();
        windowsButton.click();
        windowsCheckbox.check();

        System.out.println("\\nMac UI:");
        // Использование Abstract Factory для Mac
        UIFactory macFactory = new MacFactory();
        Button macButton = macFactory.createButton();
        Checkbox macCheckbox = macFactory.createCheckbox();
        macButton.click();
        macCheckbox.check();
    }
}`,
            description: `# Паттерн Factory

Паттерн Factory предоставляет интерфейс для создания объектов в суперклассе, но позволяет подклассам изменять тип создаваемых объектов. Abstract Factory предоставляет интерфейс для создания семейств связанных объектов без указания их конкретных классов.

## Требования:
1. Создайте интерфейс Shape и реализации:
   1.1. Классы Circle, Rectangle, Triangle
   1.2. Общий метод draw()

2. Реализуйте паттерн Factory Method:
   2.1. ShapeFactory с методом createShape(String type)
   2.2. Возвращает соответствующую фигуру на основе типа

3. Реализуйте паттерн Abstract Factory:
   3.1. Интерфейс UIFactory для создания UI компонентов
   3.2. Реализации WindowsFactory и MacFactory
   3.3. Создайте компоненты Button и Checkbox для каждой платформы

4. Продемонстрируйте оба паттерна с примерами

## Пример вывода:
\`\`\`
=== Factory Method Pattern ===
Drawing: Circle with radius 5
Drawing: Rectangle with width 10 and height 6
Drawing: Triangle with base 8 and height 7

=== Abstract Factory Pattern ===
Windows UI:
Windows Button clicked!
Windows Checkbox checked!

Mac UI:
Mac Button clicked!
Mac Checkbox checked!
\`\`\``,
            hint1: `Factory Method: Создайте класс фабрики с методом, который возвращает разные объекты на основе входных параметров. Используйте switch/if операторы для решения, какой конкретный класс создать.`,
            hint2: `Abstract Factory: Создайте интерфейс с методами для создания связанных объектов. Реализуйте конкретные фабрики, которые возвращают специфичные для платформы реализации этих объектов.`,
            whyItMatters: `Паттерны Factory являются фундаментальными для разделения создания объектов от их использования. Factory Method обеспечивает гибкость в создании объектов, в то время как Abstract Factory обеспечивает согласованность между семействами связанных объектов.

**Продакшен паттерн:**
\`\`\`java
// Factory Method - централизованное создание объектов
ShapeFactory factory = new ShapeFactory();
Shape circle = factory.createShape("circle", 5);
circle.draw();

// Abstract Factory - семейства связанных объектов
UIFactory windowsFactory = new WindowsFactory();
Button button = windowsFactory.createButton();
Checkbox checkbox = windowsFactory.createCheckbox();

// Легко переключаться между реализациями
UIFactory macFactory = new MacFactory();
\`\`\`

**Практические преимущества:**
- Отделяет логику создания от бизнес-логики
- Легко добавлять новые типы без изменения клиентского кода
- Abstract Factory гарантирует совместимость компонентов
- Используется в Spring Framework, JDBC и многих API`
        },
        uz: {
            title: `Factory namunasi`,
            solutionCode: `// ===== Factory Method namunasi =====

// Mahsulot interfeysi
interface Shape {
    void draw();
}

// Aniq mahsulotlar
class Circle implements Shape {
    private int radius;

    public Circle(int radius) {
        this.radius = radius;
    }

    @Override
    public void draw() {
        System.out.println("Drawing: Circle with radius " + radius);
    }
}

class Rectangle implements Shape {
    private int width, height;

    public Rectangle(int width, int height) {
        this.width = width;
        this.height = height;
    }

    @Override
    public void draw() {
        System.out.println("Drawing: Rectangle with width " + width + " and height " + height);
    }
}

class Triangle implements Shape {
    private int base, height;

    public Triangle(int base, int height) {
        this.base = base;
        this.height = height;
    }

    @Override
    public void draw() {
        System.out.println("Drawing: Triangle with base " + base + " and height " + height);
    }
}

// Factory Method bilan fabrika klassi
class ShapeFactory {
    // Factory Method - kirish ma'lumotlari asosida obyektlar yaratadi
    public Shape createShape(String type, int... dimensions) {
        if (type == null) {
            return null;
        }
        switch (type.toLowerCase()) {
            case "circle":
                return new Circle(dimensions[0]);
            case "rectangle":
                return new Rectangle(dimensions[0], dimensions[1]);
            case "triangle":
                return new Triangle(dimensions[0], dimensions[1]);
            default:
                throw new IllegalArgumentException("Unknown shape: " + type);
        }
    }
}

// ===== Abstract Factory namunasi =====

// Abstrakt mahsulotlar
interface Button {
    void click();
}

interface Checkbox {
    void check();
}

// Windows uchun aniq mahsulotlar
class WindowsButton implements Button {
    @Override
    public void click() {
        System.out.println("Windows Button clicked!");
    }
}

class WindowsCheckbox implements Checkbox {
    @Override
    public void check() {
        System.out.println("Windows Checkbox checked!");
    }
}

// Mac uchun aniq mahsulotlar
class MacButton implements Button {
    @Override
    public void click() {
        System.out.println("Mac Button clicked!");
    }
}

class MacCheckbox implements Checkbox {
    @Override
    public void check() {
        System.out.println("Mac Checkbox checked!");
    }
}

// Abstract Factory interfeysi
interface UIFactory {
    Button createButton();
    Checkbox createCheckbox();
}

// Windows uchun aniq fabrika
class WindowsFactory implements UIFactory {
    @Override
    public Button createButton() {
        return new WindowsButton();
    }

    @Override
    public Checkbox createCheckbox() {
        return new WindowsCheckbox();
    }
}

// Mac uchun aniq fabrika
class MacFactory implements UIFactory {
    @Override
    public Button createButton() {
        return new MacButton();
    }

    @Override
    public Checkbox createCheckbox() {
        return new MacCheckbox();
    }
}

public class FactoryPattern {
    public static void main(String[] args) {
        System.out.println("=== Factory Method namunasi ===");

        // Factory Method-dan foydalanish
        ShapeFactory shapeFactory = new ShapeFactory();

        Shape circle = shapeFactory.createShape("circle", 5);
        circle.draw();

        Shape rectangle = shapeFactory.createShape("rectangle", 10, 6);
        rectangle.draw();

        Shape triangle = shapeFactory.createShape("triangle", 8, 7);
        triangle.draw();

        System.out.println("\\n=== Abstract Factory namunasi ===");

        // Windows uchun Abstract Factory dan foydalanish
        UIFactory windowsFactory = new WindowsFactory();
        System.out.println("Windows UI:");
        Button windowsButton = windowsFactory.createButton();
        Checkbox windowsCheckbox = windowsFactory.createCheckbox();
        windowsButton.click();
        windowsCheckbox.check();

        System.out.println("\\nMac UI:");
        // Mac uchun Abstract Factory dan foydalanish
        UIFactory macFactory = new MacFactory();
        Button macButton = macFactory.createButton();
        Checkbox macCheckbox = macFactory.createCheckbox();
        macButton.click();
        macCheckbox.check();
    }
}`,
            description: `# Factory namunasi

Factory namunasi superklass-da obyektlarni yaratish uchun interfeys taqdim etadi, lekin subklass-larga yaratilgan obyektlar turini o'zgartirishga imkon beradi. Abstract Factory aniq klasslarini ko'rsatmasdan bog'liq obyektlar oilalarini yaratish uchun interfeys taqdim etadi.

## Talablar:
1. Shape interfeysi va amalga oshirishlarini yarating:
   1.1. Circle, Rectangle, Triangle klasslari
   1.2. Umumiy draw() metodi

2. Factory Method namunasini amalga oshiring:
   2.1. createShape(String type) metodi bilan ShapeFactory
   2.2. Turga asoslangan tegishli shaklni qaytaradi

3. Abstract Factory namunasini amalga oshiring:
   3.1. UI komponentlarini yaratish uchun UIFactory interfeysi
   3.2. WindowsFactory va MacFactory amalga oshirishlari
   3.3. Har bir platforma uchun Button va Checkbox komponentlarini yarating

4. Ikkala namunani misollar bilan namoyish eting

## Chiqish namunasi:
\`\`\`
=== Factory Method Pattern ===
Drawing: Circle with radius 5
Drawing: Rectangle with width 10 and height 6
Drawing: Triangle with base 8 and height 7

=== Abstract Factory Pattern ===
Windows UI:
Windows Button clicked!
Windows Checkbox checked!

Mac UI:
Mac Button clicked!
Mac Checkbox checked!
\`\`\``,
            hint1: `Factory Method: Kirish parametrlariga asoslangan turli obyektlarni qaytaradigan metod bilan fabrika klassini yarating. Qaysi aniq klassni yaratishni hal qilish uchun switch/if operatorlaridan foydalaning.`,
            hint2: `Abstract Factory: Bog'liq obyektlarni yaratish uchun metodlar bilan interfeys yarating. Bu obyektlarning platformaga xos amalga oshirishlarini qaytaradigan aniq fabrikalarni amalga oshiring.`,
            whyItMatters: `Factory namunalari obyekt yaratishni foydalanishdan ajratish uchun asosiy hisoblanadi. Factory Method obyektlarni yaratishda moslashuvchanlikni ta'minlaydi, Abstract Factory esa bog'liq obyektlar oilalari o'rtasida izchillikni ta'minlaydi.

**Ishlab chiqarish patterni:**
\`\`\`java
// Factory Method - markazlashtirilgan obyekt yaratish
ShapeFactory factory = new ShapeFactory();
Shape circle = factory.createShape("circle", 5);
circle.draw();

// Abstract Factory - bog'liq obyektlar oilalari
UIFactory windowsFactory = new WindowsFactory();
Button button = windowsFactory.createButton();
Checkbox checkbox = windowsFactory.createCheckbox();

// Amalga oshirishlar o'rtasida osongina o'tish
UIFactory macFactory = new MacFactory();
\`\`\`

**Amaliy foydalari:**
- Yaratish mantiqini biznes mantiqidan ajratadi
- Mijoz kodini o'zgartirmasdan yangi turlarni qo'shish oson
- Abstract Factory komponentlar muvofiqligini kafolatlaydi
- Spring Framework, JDBC va ko'plab API larda ishlatiladi`
        }
    }
};

export default task;
