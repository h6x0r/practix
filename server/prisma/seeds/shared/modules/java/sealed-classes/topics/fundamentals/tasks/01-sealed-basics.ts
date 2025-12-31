import { Task } from '../../../../../../../../types';

export const task: Task = {
    slug: 'java-sealed-basics',
    title: 'Sealed Classes Basics',
    difficulty: 'easy',
    tags: ['java', 'sealed', 'permits', 'non-sealed', 'final'],
    estimatedTime: '25m',
    isPremium: false,
    youtubeUrl: '',
    description: `Learn the basics of sealed classes in Java 17+.

**Requirements:**
1. Create a sealed class Shape that permits Circle, Rectangle, and Triangle
2. Create final class Circle that extends Shape with a radius field
3. Create final class Rectangle that extends Shape with width and height fields
4. Create non-sealed class Triangle that extends Shape with base and height fields
5. Add a calculateArea() abstract method to Shape and implement it in all subclasses
6. Demonstrate that you cannot create unauthorized subclasses of Shape

Sealed classes restrict which classes can extend or implement them using the permits clause. Permitted subclasses must be final, sealed, or non-sealed.`,
    initialCode: `// Create sealed Shape class that permits Circle, Rectangle, Triangle
// Add abstract double calculateArea() method

// Create final Circle class
// - extends Shape
// - double radius field
// - constructor
// - calculateArea() implementation

// Create final Rectangle class
// - extends Shape
// - double width, height fields
// - constructor
// - calculateArea() implementation

// Create non-sealed Triangle class
// - extends Shape
// - double base, height fields
// - constructor
// - calculateArea() implementation

public class SealedBasics {
    public static void main(String[] args) {
        // Create instances of all shapes

        // Print their areas

        // Try to create unauthorized subclass (will fail at compile time)
    }
}`,
    solutionCode: `// Sealed class restricts which classes can extend it
sealed class Shape permits Circle, Rectangle, Triangle {
    // Abstract method must be implemented by all permitted subclasses
    abstract double calculateArea();
}

// Final class - cannot be extended further
final class Circle extends Shape {
    private final double radius;

    public Circle(double radius) {
        this.radius = radius;
    }

    @Override
    double calculateArea() {
        return Math.PI * radius * radius;
    }
}

// Final class - cannot be extended further
final class Rectangle extends Shape {
    private final double width;
    private final double height;

    public Rectangle(double width, double height) {
        this.width = width;
        this.height = height;
    }

    @Override
    double calculateArea() {
        return width * height;
    }
}

// Non-sealed class - can be extended by other classes
non-sealed class Triangle extends Shape {
    private final double base;
    private final double height;

    public Triangle(double base, double height) {
        this.base = base;
        this.height = height;
    }

    @Override
    double calculateArea() {
        return 0.5 * base * height;
    }
}

public class SealedBasics {
    public static void main(String[] args) {
        // Create instances of permitted subclasses
        Shape circle = new Circle(5.0);
        Shape rectangle = new Rectangle(4.0, 6.0);
        Shape triangle = new Triangle(3.0, 4.0);

        // Calculate and print areas
        System.out.println("Circle area: " + circle.calculateArea());
        System.out.println("Rectangle area: " + rectangle.calculateArea());
        System.out.println("Triangle area: " + triangle.calculateArea());

        // This would cause a compile-time error:
        // class Square extends Shape { } // Error: not permitted

        // Non-sealed Triangle can be extended by other classes
        // class IsoscelesTriangle extends Triangle { } // This is allowed
    }
}`,
    hint1: `Use the sealed keyword before class and add permits clause: sealed class Shape permits Circle, Rectangle, Triangle { ... }`,
    hint2: `Permitted subclasses must be: final (cannot be extended), sealed (restricted extension), or non-sealed (unrestricted extension).`,
    whyItMatters: `Sealed classes give you fine-grained control over class hierarchies. They're essential for domain modeling where you want a fixed set of subtypes, making your code more maintainable and enabling exhaustive pattern matching.

**Production Pattern:**
\`\`\`java
sealed interface PaymentStatus permits Pending, Approved, Rejected {
    String getMessage();
}

final class Pending implements PaymentStatus {
    public String getMessage() { return "Awaiting approval"; }
}

final class Approved implements PaymentStatus {
    public String getMessage() { return "Payment approved"; }
}

final class Rejected implements PaymentStatus {
    private final String reason;
    public Rejected(String reason) { this.reason = reason; }
    public String getMessage() { return "Rejected: " + reason; }
}
\`\`\`

**Practical Benefits:**
- Compiler guarantees handling of all payment statuses
- Impossible to add uncontrolled statuses`,
    order: 0,
    testCode: `import static org.junit.Assert.*;
import org.junit.Test;

// Define sealed class and permitted subclasses
sealed class Shape permits Circle, Rectangle, Triangle {
    abstract double area();
}

final class Circle extends Shape {
    final double radius;
    Circle(double radius) { this.radius = radius; }
    double area() { return Math.PI * radius * radius; }
}

final class Rectangle extends Shape {
    final double width, height;
    Rectangle(double width, double height) { this.width = width; this.height = height; }
    double area() { return width * height; }
}

final class Triangle extends Shape {
    final double base, height;
    Triangle(double base, double height) { this.base = base; this.height = height; }
    double area() { return 0.5 * base * height; }
}

// Test1: Test Circle area calculation
class Test1 {
    @Test
    public void test() {
        Circle circle = new Circle(5);
        assertEquals(Math.PI * 25, circle.area(), 0.01);
    }
}

// Test2: Test Rectangle area calculation
class Test2 {
    @Test
    public void test() {
        Rectangle rect = new Rectangle(4, 5);
        assertEquals(20.0, rect.area(), 0.01);
    }
}

// Test3: Test Triangle area calculation
class Test3 {
    @Test
    public void test() {
        Triangle triangle = new Triangle(6, 4);
        assertEquals(12.0, triangle.area(), 0.01);
    }
}

// Test4: Test Shape polymorphism with Circle
class Test4 {
    @Test
    public void test() {
        Shape shape = new Circle(3);
        assertTrue(shape instanceof Circle);
        assertEquals(Math.PI * 9, shape.area(), 0.01);
    }
}

// Test5: Test Shape polymorphism with Rectangle
class Test5 {
    @Test
    public void test() {
        Shape shape = new Rectangle(3, 4);
        assertTrue(shape instanceof Rectangle);
        assertEquals(12.0, shape.area(), 0.01);
    }
}

// Test6: Test Shape polymorphism with Triangle
class Test6 {
    @Test
    public void test() {
        Shape shape = new Triangle(5, 6);
        assertTrue(shape instanceof Triangle);
        assertEquals(15.0, shape.area(), 0.01);
    }
}

// Test7: Test Circle with different radius
class Test7 {
    @Test
    public void test() {
        Circle circle = new Circle(10);
        assertEquals(Math.PI * 100, circle.area(), 0.01);
    }
}

// Test8: Test Rectangle with square dimensions
class Test8 {
    @Test
    public void test() {
        Rectangle square = new Rectangle(5, 5);
        assertEquals(25.0, square.area(), 0.01);
    }
}

// Test9: Test instanceof for all shapes
class Test9 {
    @Test
    public void test() {
        Shape c = new Circle(1);
        Shape r = new Rectangle(1, 1);
        Shape t = new Triangle(1, 1);
        assertTrue(c instanceof Circle);
        assertTrue(r instanceof Rectangle);
        assertTrue(t instanceof Triangle);
    }
}

// Test10: Test area calculation for various shapes
class Test10 {
    @Test
    public void test() {
        Shape[] shapes = {
            new Circle(2),
            new Rectangle(3, 4),
            new Triangle(4, 5)
        };
        assertTrue(shapes.length == 3);
        assertTrue(shapes[0].area() > 0);
        assertTrue(shapes[1].area() == 12.0);
        assertTrue(shapes[2].area() == 10.0);
    }
}
`,
    translations: {
        ru: {
            title: 'Основы запечатанных классов',
            solutionCode: `// Запечатанный класс ограничивает, какие классы могут его расширить
sealed class Shape permits Circle, Rectangle, Triangle {
    // Абстрактный метод должен быть реализован всеми разрешенными подклассами
    abstract double calculateArea();
}

// Финальный класс - не может быть расширен дальше
final class Circle extends Shape {
    private final double radius;

    public Circle(double radius) {
        this.radius = radius;
    }

    @Override
    double calculateArea() {
        return Math.PI * radius * radius;
    }
}

// Финальный класс - не может быть расширен дальше
final class Rectangle extends Shape {
    private final double width;
    private final double height;

    public Rectangle(double width, double height) {
        this.width = width;
        this.height = height;
    }

    @Override
    double calculateArea() {
        return width * height;
    }
}

// Незапечатанный класс - может быть расширен другими классами
non-sealed class Triangle extends Shape {
    private final double base;
    private final double height;

    public Triangle(double base, double height) {
        this.base = base;
        this.height = height;
    }

    @Override
    double calculateArea() {
        return 0.5 * base * height;
    }
}

public class SealedBasics {
    public static void main(String[] args) {
        // Создаем экземпляры разрешенных подклассов
        Shape circle = new Circle(5.0);
        Shape rectangle = new Rectangle(4.0, 6.0);
        Shape triangle = new Triangle(3.0, 4.0);

        // Вычисляем и выводим площади
        System.out.println("Площадь круга: " + circle.calculateArea());
        System.out.println("Площадь прямоугольника: " + rectangle.calculateArea());
        System.out.println("Площадь треугольника: " + triangle.calculateArea());

        // Это вызовет ошибку компиляции:
        // class Square extends Shape { } // Ошибка: не разрешено

        // Незапечатанный Triangle может быть расширен другими классами
        // class IsoscelesTriangle extends Triangle { } // Это разрешено
    }
}`,
            description: `Изучите основы запечатанных классов в Java 17+.

**Требования:**
1. Создайте запечатанный класс Shape, который разрешает Circle, Rectangle и Triangle
2. Создайте финальный класс Circle, расширяющий Shape с полем radius
3. Создайте финальный класс Rectangle, расширяющий Shape с полями width и height
4. Создайте незапечатанный класс Triangle, расширяющий Shape с полями base и height
5. Добавьте абстрактный метод calculateArea() в Shape и реализуйте его во всех подклассах
6. Продемонстрируйте, что невозможно создать неавторизованные подклассы Shape

Запечатанные классы ограничивают, какие классы могут их расширять или реализовывать, используя предложение permits. Разрешенные подклассы должны быть final, sealed или non-sealed.`,
            hint1: `Используйте ключевое слово sealed перед class и добавьте предложение permits: sealed class Shape permits Circle, Rectangle, Triangle { ... }`,
            hint2: `Разрешенные подклассы должны быть: final (не может быть расширен), sealed (ограниченное расширение) или non-sealed (неограниченное расширение).`,
            whyItMatters: `Запечатанные классы дают вам точный контроль над иерархиями классов. Они необходимы для моделирования доменов, где требуется фиксированный набор подтипов, делая код более поддерживаемым и обеспечивая исчерпывающее сопоставление с образцом.

**Продакшен паттерн:**
\`\`\`java
sealed interface PaymentStatus permits Pending, Approved, Rejected {
    String getMessage();
}

final class Pending implements PaymentStatus {
    public String getMessage() { return "Awaiting approval"; }
}

final class Approved implements PaymentStatus {
    public String getMessage() { return "Payment approved"; }
}

final class Rejected implements PaymentStatus {
    private final String reason;
    public Rejected(String reason) { this.reason = reason; }
    public String getMessage() { return "Rejected: " + reason; }
}
\`\`\`

**Практические преимущества:**
- Компилятор гарантирует обработку всех статусов платежей
- Невозможно добавить неконтролируемые статусы`
        },
        uz: {
            title: 'Muhrlangan klasslar asoslari',
            solutionCode: `// Muhrlangan klass qaysi klasslar uni kengaytirishi mumkinligini cheklaydi
sealed class Shape permits Circle, Rectangle, Triangle {
    // Abstrakt metod barcha ruxsat etilgan subklasslar tomonidan amalga oshirilishi kerak
    abstract double calculateArea();
}

// Final klass - boshqa kengaytirilishi mumkin emas
final class Circle extends Shape {
    private final double radius;

    public Circle(double radius) {
        this.radius = radius;
    }

    @Override
    double calculateArea() {
        return Math.PI * radius * radius;
    }
}

// Final klass - boshqa kengaytirilishi mumkin emas
final class Rectangle extends Shape {
    private final double width;
    private final double height;

    public Rectangle(double width, double height) {
        this.width = width;
        this.height = height;
    }

    @Override
    double calculateArea() {
        return width * height;
    }
}

// Non-sealed klass - boshqa klasslar tomonidan kengaytirilishi mumkin
non-sealed class Triangle extends Shape {
    private final double base;
    private final double height;

    public Triangle(double base, double height) {
        this.base = base;
        this.height = height;
    }

    @Override
    double calculateArea() {
        return 0.5 * base * height;
    }
}

public class SealedBasics {
    public static void main(String[] args) {
        // Ruxsat etilgan subklasslarning misollarini yaratamiz
        Shape circle = new Circle(5.0);
        Shape rectangle = new Rectangle(4.0, 6.0);
        Shape triangle = new Triangle(3.0, 4.0);

        // Yuzalarni hisoblaymiz va chiqaramiz
        System.out.println("Doira yuzi: " + circle.calculateArea());
        System.out.println("To'rtburchak yuzi: " + rectangle.calculateArea());
        System.out.println("Uchburchak yuzi: " + triangle.calculateArea());

        // Bu kompilyatsiya xatosiga olib keladi:
        // class Square extends Shape { } // Xato: ruxsat etilmagan

        // Non-sealed Triangle boshqa klasslar tomonidan kengaytirilishi mumkin
        // class IsoscelesTriangle extends Triangle { } // Bu ruxsat etilgan
    }
}`,
            description: `Java 17+ da muhrlangan klasslar asoslarini o'rganing.

**Talablar:**
1. Circle, Rectangle va Triangle ga ruxsat beruvchi muhrlangan Shape klassini yarating
2. Shape ni kengaytiruvchi radius maydonli final Circle klassini yarating
3. Shape ni kengaytiruvchi width va height maydonli final Rectangle klassini yarating
4. Shape ni kengaytiruvchi base va height maydonli non-sealed Triangle klassini yarating
5. Shape ga abstrakt calculateArea() metodini qo'shing va uni barcha subklasslarda amalga oshiring
6. Shape ning ruxsatsiz subklasslarini yaratib bo'lmasligini ko'rsating

Muhrlangan klasslar permits bandidan foydalanib, qaysi klasslar ularni kengaytirishi yoki amalga oshirishi mumkinligini cheklaydi. Ruxsat etilgan subklasslar final, sealed yoki non-sealed bo'lishi kerak.`,
            hint1: `Class oldidan sealed kalit so'zini ishlating va permits bandini qo'shing: sealed class Shape permits Circle, Rectangle, Triangle { ... }`,
            hint2: `Ruxsat etilgan subklasslar quyidagilardan biri bo'lishi kerak: final (kengaytirilishi mumkin emas), sealed (cheklangan kengaytirish) yoki non-sealed (cheklanmagan kengaytirish).`,
            whyItMatters: `Muhrlangan klasslar sizga klass iyerarxiyalari ustidan aniq nazorat beradi. Ular belgilangan subtiplar to'plami kerak bo'lgan domen modellash uchun zarur, kodingizni yanada boshqariladigan qiladi va to'liq pattern matching ni ta'minlaydi.

**Ishlab chiqarish patterni:**
\`\`\`java
sealed interface PaymentStatus permits Pending, Approved, Rejected {
    String getMessage();
}

final class Pending implements PaymentStatus {
    public String getMessage() { return "Awaiting approval"; }
}

final class Approved implements PaymentStatus {
    public String getMessage() { return "Payment approved"; }
}

final class Rejected implements PaymentStatus {
    private final String reason;
    public Rejected(String reason) { this.reason = reason; }
    public String getMessage() { return "Rejected: " + reason; }
}
\`\`\`

**Amaliy foydalari:**
- Kompilyator barcha to'lov holatlari qayta ishlanishini kafolatlaydi
- Nazoratsiz holatlarni qo'shish mumkin emas`
        }
    }
};

export default task;
