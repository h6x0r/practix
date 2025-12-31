import { Task } from '../../../../types';

export const task: Task = {
    slug: 'java-interface-basics',
    title: 'Interface Declaration and Implementation',
    difficulty: 'easy',
    tags: ['java', 'interfaces', 'oop', 'basics'],
    estimatedTime: '20m',
    isPremium: false,
    youtubeUrl: '',
    description: `# Interface Declaration and Implementation

Interfaces in Java define a contract that classes must follow. They contain abstract methods (without implementation) that implementing classes must provide.

## Requirements:
1. Create a \`Shape\` interface with methods:
   1.1. \`double calculateArea()\`
   1.2. \`double calculatePerimeter()\`
   1.3. \`String getShapeName()\`

2. Create a \`Circle\` class that implements \`Shape\`:
   2.1. Has a \`radius\` field
   2.2. Implements all interface methods
   2.3. Use Math.PI for calculations

3. Create a \`Rectangle\` class that implements \`Shape\`:
   3.1. Has \`width\` and \`height\` fields
   3.2. Implements all interface methods

4. In \`main()\`, create instances and display their properties

## Example Output:
\`\`\`
Circle: Area = 78.54, Perimeter = 31.42
Rectangle: Area = 24.00, Perimeter = 20.00
\`\`\``,
    initialCode: `// TODO: Create Shape interface

// TODO: Create Circle class implementing Shape

// TODO: Create Rectangle class implementing Shape

public class InterfaceBasics {
    public static void main(String[] args) {
        // TODO: Create Circle with radius 5

        // TODO: Create Rectangle with width 4 and height 6

        // TODO: Display information for both shapes
    }
}`,
    solutionCode: `// Shape interface defines the contract for all shapes
interface Shape {
    double calculateArea();
    double calculatePerimeter();
    String getShapeName();
}

// Circle class implements the Shape interface
class Circle implements Shape {
    private double radius;

    public Circle(double radius) {
        this.radius = radius;
    }

    @Override
    public double calculateArea() {
        return Math.PI * radius * radius;
    }

    @Override
    public double calculatePerimeter() {
        return 2 * Math.PI * radius;
    }

    @Override
    public String getShapeName() {
        return "Circle";
    }
}

// Rectangle class implements the Shape interface
class Rectangle implements Shape {
    private double width;
    private double height;

    public Rectangle(double width, double height) {
        this.width = width;
        this.height = height;
    }

    @Override
    public double calculateArea() {
        return width * height;
    }

    @Override
    public double calculatePerimeter() {
        return 2 * (width + height);
    }

    @Override
    public String getShapeName() {
        return "Rectangle";
    }
}

public class InterfaceBasics {
    public static void main(String[] args) {
        // Using interface as type reference
        Shape circle = new Circle(5);
        Shape rectangle = new Rectangle(4, 6);

        // Display information using interface methods
        System.out.printf("%s: Area = %.2f, Perimeter = %.2f%n",
            circle.getShapeName(),
            circle.calculateArea(),
            circle.calculatePerimeter());

        System.out.printf("%s: Area = %.2f, Perimeter = %.2f%n",
            rectangle.getShapeName(),
            rectangle.calculateArea(),
            rectangle.calculatePerimeter());
    }
}`,
    hint1: `Start by declaring the interface with the three abstract methods. Remember, interface methods are public and abstract by default.`,
    hint2: `When implementing the interface, use the @Override annotation and make sure to implement ALL methods defined in the interface.`,
    whyItMatters: `Interfaces are fundamental to Java's approach to abstraction and polymorphism. They allow you to define contracts that multiple classes can implement, enabling flexible and maintainable code design. Interfaces are used extensively in Java frameworks like Spring, Android, and Java Collections.

**Production Pattern:**

\`\`\`java
// Interface definition for different payment methods
interface PaymentProcessor {
    boolean processPayment(double amount);
    String getPaymentMethod();
}

// Different implementations
class CreditCardProcessor implements PaymentProcessor {
    public boolean processPayment(double amount) {
        // Credit card processing logic
        return true;
    }

    public String getPaymentMethod() { return "Credit Card"; }
}

class PayPalProcessor implements PaymentProcessor {
    public boolean processPayment(double amount) {
        // PayPal processing logic
        return true;
    }

    public String getPaymentMethod() { return "PayPal"; }
}

// Usage through interface
PaymentProcessor processor = getProcessor(userChoice);
processor.processPayment(100.0);	// Polymorphism in action
\`\`\`

**Practical Benefits:**

1. **Flexibility**: Easy to add new implementations without changing existing code
2. **Testability**: Simplifies creating mock objects for testing
3. **Loose Coupling**: Code depends on interfaces, not concrete implementations
4. **Polymorphism**: One interface, multiple implementations for different scenarios`,
    order: 1,
    testCode: `import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

// Test 1: Circle implements Shape interface
class Test1 {
    @Test
    void testCircleImplementsShape() {
        Shape circle = new Circle(5);
        assertNotNull(circle);
        assertTrue(circle instanceof Shape);
    }
}

// Test 2: Rectangle implements Shape interface
class Test2 {
    @Test
    void testRectangleImplementsShape() {
        Shape rectangle = new Rectangle(4, 6);
        assertNotNull(rectangle);
        assertTrue(rectangle instanceof Shape);
    }
}

// Test 3: Circle area calculation
class Test3 {
    @Test
    void testCircleArea() {
        Circle circle = new Circle(5);
        double expected = Math.PI * 25;
        assertEquals(expected, circle.calculateArea(), 0.01);
    }
}

// Test 4: Circle perimeter calculation
class Test4 {
    @Test
    void testCirclePerimeter() {
        Circle circle = new Circle(5);
        double expected = 2 * Math.PI * 5;
        assertEquals(expected, circle.calculatePerimeter(), 0.01);
    }
}

// Test 5: Rectangle area calculation
class Test5 {
    @Test
    void testRectangleArea() {
        Rectangle rectangle = new Rectangle(4, 6);
        assertEquals(24.0, rectangle.calculateArea(), 0.01);
    }
}

// Test 6: Rectangle perimeter calculation
class Test6 {
    @Test
    void testRectanglePerimeter() {
        Rectangle rectangle = new Rectangle(4, 6);
        assertEquals(20.0, rectangle.calculatePerimeter(), 0.01);
    }
}

// Test 7: Circle getShapeName returns Circle
class Test7 {
    @Test
    void testCircleShapeName() {
        Circle circle = new Circle(5);
        assertEquals("Circle", circle.getShapeName());
    }
}

// Test 8: Rectangle getShapeName returns Rectangle
class Test8 {
    @Test
    void testRectangleShapeName() {
        Rectangle rectangle = new Rectangle(4, 6);
        assertEquals("Rectangle", rectangle.getShapeName());
    }
}

// Test 9: Polymorphism with Shape reference
class Test9 {
    @Test
    void testPolymorphism() {
        Shape[] shapes = { new Circle(3), new Rectangle(2, 4) };
        assertNotNull(shapes[0].getShapeName());
        assertNotNull(shapes[1].getShapeName());
        assertTrue(shapes[0].calculateArea() > 0);
        assertTrue(shapes[1].calculateArea() > 0);
    }
}

// Test 10: Different circle sizes
class Test10 {
    @Test
    void testDifferentCircleSizes() {
        Circle small = new Circle(1);
        Circle large = new Circle(10);
        assertTrue(large.calculateArea() > small.calculateArea());
        assertTrue(large.calculatePerimeter() > small.calculatePerimeter());
    }
}`,
    translations: {
        ru: {
            title: 'Объявление и реализация интерфейса',
            solutionCode: `// Интерфейс Shape определяет контракт для всех фигур
interface Shape {
    double calculateArea();
    double calculatePerimeter();
    String getShapeName();
}

// Класс Circle реализует интерфейс Shape
class Circle implements Shape {
    private double radius;

    public Circle(double radius) {
        this.radius = radius;
    }

    @Override
    public double calculateArea() {
        return Math.PI * radius * radius;
    }

    @Override
    public double calculatePerimeter() {
        return 2 * Math.PI * radius;
    }

    @Override
    public String getShapeName() {
        return "Circle";
    }
}

// Класс Rectangle реализует интерфейс Shape
class Rectangle implements Shape {
    private double width;
    private double height;

    public Rectangle(double width, double height) {
        this.width = width;
        this.height = height;
    }

    @Override
    public double calculateArea() {
        return width * height;
    }

    @Override
    public double calculatePerimeter() {
        return 2 * (width + height);
    }

    @Override
    public String getShapeName() {
        return "Rectangle";
    }
}

public class InterfaceBasics {
    public static void main(String[] args) {
        // Используем интерфейс как ссылочный тип
        Shape circle = new Circle(5);
        Shape rectangle = new Rectangle(4, 6);

        // Отображаем информацию используя методы интерфейса
        System.out.printf("%s: Area = %.2f, Perimeter = %.2f%n",
            circle.getShapeName(),
            circle.calculateArea(),
            circle.calculatePerimeter());

        System.out.printf("%s: Area = %.2f, Perimeter = %.2f%n",
            rectangle.getShapeName(),
            rectangle.calculateArea(),
            rectangle.calculatePerimeter());
    }
}`,
            description: `# Объявление и реализация интерфейса

Интерфейсы в Java определяют контракт, которому должны следовать классы. Они содержат абстрактные методы (без реализации), которые реализующие классы должны предоставить.

## Требования:
1. Создайте интерфейс \`Shape\` с методами:
   1.1. \`double calculateArea()\`
   1.2. \`double calculatePerimeter()\`
   1.3. \`String getShapeName()\`

2. Создайте класс \`Circle\`, реализующий \`Shape\`:
   2.1. Имеет поле \`radius\`
   2.2. Реализует все методы интерфейса
   2.3. Используйте Math.PI для вычислений

3. Создайте класс \`Rectangle\`, реализующий \`Shape\`:
   3.1. Имеет поля \`width\` и \`height\`
   3.2. Реализует все методы интерфейса

4. В методе \`main()\` создайте экземпляры и отобразите их свойства

## Пример вывода:
\`\`\`
Circle: Area = 78.54, Perimeter = 31.42
Rectangle: Area = 24.00, Perimeter = 20.00
\`\`\``,
            hint1: `Начните с объявления интерфейса с тремя абстрактными методами. Помните, методы интерфейса по умолчанию public и abstract.`,
            hint2: `При реализации интерфейса используйте аннотацию @Override и убедитесь, что реализовали ВСЕ методы, определенные в интерфейсе.`,
            whyItMatters: `Интерфейсы являются фундаментальными для подхода Java к абстракции и полиморфизму. Они позволяют определять контракты, которые могут реализовать несколько классов, обеспечивая гибкий и поддерживаемый дизайн кода. Интерфейсы широко используются в Java-фреймворках, таких как Spring, Android и Java Collections.

**Продакшен паттерн:**

\`\`\`java
// Определение интерфейса для разных способов оплаты
interface PaymentProcessor {
    boolean processPayment(double amount);
    String getPaymentMethod();
}

// Разные реализации
class CreditCardProcessor implements PaymentProcessor {
    public boolean processPayment(double amount) {
        // Логика обработки кредитной карты
        return true;
    }

    public String getPaymentMethod() { return "Credit Card"; }
}

class PayPalProcessor implements PaymentProcessor {
    public boolean processPayment(double amount) {
        // Логика обработки PayPal
        return true;
    }

    public String getPaymentMethod() { return "PayPal"; }
}

// Использование через интерфейс
PaymentProcessor processor = getProcessor(userChoice);
processor.processPayment(100.0);	// Полиморфизм в действии
\`\`\`

**Практические преимущества:**

1. **Гибкость**: Легко добавлять новые реализации без изменения существующего кода
2. **Тестируемость**: Упрощает создание mock-объектов для тестирования
3. **Слабая связанность**: Код зависит от интерфейса, а не от конкретных реализаций
4. **Полиморфизм**: Один интерфейс, множество реализаций для разных сценариев`
        },
        uz: {
            title: `Interface e'lon qilish va amalga oshirish`,
            solutionCode: `// Shape interfeysi barcha shakllar uchun shartnomani belgilaydi
interface Shape {
    double calculateArea();
    double calculatePerimeter();
    String getShapeName();
}

// Circle klassi Shape interfeysini amalga oshiradi
class Circle implements Shape {
    private double radius;

    public Circle(double radius) {
        this.radius = radius;
    }

    @Override
    public double calculateArea() {
        return Math.PI * radius * radius;
    }

    @Override
    public double calculatePerimeter() {
        return 2 * Math.PI * radius;
    }

    @Override
    public String getShapeName() {
        return "Circle";
    }
}

// Rectangle klassi Shape interfeysini amalga oshiradi
class Rectangle implements Shape {
    private double width;
    private double height;

    public Rectangle(double width, double height) {
        this.width = width;
        this.height = height;
    }

    @Override
    public double calculateArea() {
        return width * height;
    }

    @Override
    public double calculatePerimeter() {
        return 2 * (width + height);
    }

    @Override
    public String getShapeName() {
        return "Rectangle";
    }
}

public class InterfaceBasics {
    public static void main(String[] args) {
        // Interfeysni havola turi sifatida ishlatamiz
        Shape circle = new Circle(5);
        Shape rectangle = new Rectangle(4, 6);

        // Interfeys metodlaridan foydalanib ma'lumotlarni ko'rsatamiz
        System.out.printf("%s: Area = %.2f, Perimeter = %.2f%n",
            circle.getShapeName(),
            circle.calculateArea(),
            circle.calculatePerimeter());

        System.out.printf("%s: Area = %.2f, Perimeter = %.2f%n",
            rectangle.getShapeName(),
            rectangle.calculateArea(),
            rectangle.calculatePerimeter());
    }
}`,
            description: `# Interface e'lon qilish va amalga oshirish

Java-da interfeyslar klasslar amal qilishi kerak bo'lgan shartnomani belgilaydi. Ular amalga oshiruvchi klasslar taqdim etishi kerak bo'lgan abstrakt metodlarni (implementatsiyasiz) o'z ichiga oladi.

## Talablar:
1. \`Shape\` interfeysini metodlar bilan yarating:
   1.1. \`double calculateArea()\`
   1.2. \`double calculatePerimeter()\`
   1.3. \`String getShapeName()\`

2. \`Shape\` ni amalga oshiruvchi \`Circle\` klassini yarating:
   2.1. \`radius\` maydoniga ega
   2.2. Barcha interfeys metodlarini amalga oshiradi
   2.3. Hisoblashlar uchun Math.PI dan foydalaning

3. \`Shape\` ni amalga oshiruvchi \`Rectangle\` klassini yarating:
   3.1. \`width\` va \`height\` maydonlariga ega
   3.2. Barcha interfeys metodlarini amalga oshiradi

4. \`main()\` metodida misollar yarating va ularning xususiyatlarini ko'rsating

## Chiqish namunasi:
\`\`\`
Circle: Area = 78.54, Perimeter = 31.42
Rectangle: Area = 24.00, Perimeter = 20.00
\`\`\``,
            hint1: `Uchta abstrakt metod bilan interfeysni e'lon qilishdan boshlang. Esda tuting, interfeys metodlari standart bo'yicha public va abstract hisoblanadi.`,
            hint2: `Interfeysni amalga oshirishda @Override annotatsiyasidan foydalaning va interfeysda belgilangan BARCHA metodlarni amalga oshirganingizga ishonch hosil qiling.`,
            whyItMatters: `Interfeyslar Java-ning abstraktsiya va polimorfizmga yondashuvining asosiy qismidir. Ular bir nechta klasslar amalga oshirishi mumkin bo'lgan shartnomalarni belgilashga imkon beradi, bu esa moslashuvchan va boshqariladigan kod dizaynini ta'minlaydi. Interfeyslar Spring, Android va Java Collections kabi Java frameworklarida keng qo'llaniladi.

**Ishlab chiqarish patterni:**

\`\`\`java
// Turli to'lov usullari uchun interfeys ta'rifi
interface PaymentProcessor {
    boolean processPayment(double amount);
    String getPaymentMethod();
}

// Turli implementatsiyalar
class CreditCardProcessor implements PaymentProcessor {
    public boolean processPayment(double amount) {
        // Kredit karta qayta ishlash mantiqi
        return true;
    }

    public String getPaymentMethod() { return "Credit Card"; }
}

class PayPalProcessor implements PaymentProcessor {
    public boolean processPayment(double amount) {
        // PayPal qayta ishlash mantiqi
        return true;
    }

    public String getPaymentMethod() { return "PayPal"; }
}

// Interfeys orqali foydalanish
PaymentProcessor processor = getProcessor(userChoice);
processor.processPayment(100.0);	// Polimorfizm amalda
\`\`\`

**Amaliy foydalari:**

1. **Moslashuvchanlik**: Mavjud kodni o'zgartirmasdan yangi implementatsiyalarni qo'shish oson
2. **Testlanishi**: Test uchun mock-obyektlarni yaratishni soddalashtiradi
3. **Zaif bog'lanish**: Kod aniq implementatsiyalarga emas, interfeyslarga bog'liq
4. **Polimorfizm**: Bitta interfeys, turli stsenariylar uchun ko'plab implementatsiyalar`
        }
    }
};

export default task;
