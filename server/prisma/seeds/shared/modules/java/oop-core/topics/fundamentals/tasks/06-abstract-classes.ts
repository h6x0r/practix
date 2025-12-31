import { Task } from '../../../../types';

export const task: Task = {
    slug: 'java-oop-abstract-classes',
    title: 'Abstract Classes and Methods',
    difficulty: 'medium',
    tags: ['java', 'oop', 'abstract', 'abstract-classes', 'design-patterns'],
    estimatedTime: '25m',
    isPremium: false,
    youtubeUrl: '',
    description: `Create a shape hierarchy using abstract classes to understand when and how to use abstraction in Java.

**Requirements:**
1. Create an abstract **Shape** class with:
   1.1. Protected field: color (String)
   1.2. Constructor to initialize color
   1.3. Abstract methods: calculateArea(), calculatePerimeter()
   1.4. Concrete method: displayInfo() that shows shape details
   1.5. A final method: printColor() that cannot be overridden

2. Create **Rectangle** class extending Shape:
   2.1. Private fields: width, height
   2.2. Constructor initializing all fields
   2.3. Implement abstract methods with rectangle-specific formulas
   2.4. Add a specific method: isSquare()

3. Create **Circle** class extending Shape:
   3.1. Private field: radius
   3.2. Constructor initializing all fields
   3.3. Implement abstract methods with circle-specific formulas
   3.4. Use Math.PI for calculations

4. Create **Triangle** class extending Shape:
   4.1. Private fields: side1, side2, side3
   4.2. Constructor initializing all fields
   4.3. Implement abstract methods (use Heron's formula for area)
   4.4. Add validation method: isValidTriangle()

5. In main method:
   5.1. Create instances of all concrete shapes
   5.2. Store them in a Shape array (demonstrate polymorphism)
   5.3. Calculate and display area/perimeter for each
   5.4. Show that you cannot instantiate abstract class

**Learning Goals:**
- Understand when to use abstract classes vs concrete classes
- Learn the difference between abstract and concrete methods
- Practice implementing abstract methods
- Understand the final keyword
- See how abstract classes enable polymorphism`,
    initialCode: `// TODO: Create Shape abstract class

// TODO: Create Rectangle class

// TODO: Create Circle class

// TODO: Create Triangle class

public class AbstractDemo {
    public static void main(String[] args) {
        // TODO: Create and test shape hierarchy
    }
}`,
    solutionCode: `// Abstract class - cannot be instantiated, serves as a template
abstract class Shape {
    protected String color;

    public Shape(String color) {
        this.color = color;
    }

    // Abstract methods - must be implemented by concrete subclasses
    public abstract double calculateArea();
    public abstract double calculatePerimeter();

    // Concrete method - inherited by all subclasses
    public void displayInfo() {
        System.out.println("Shape: " + this.getClass().getSimpleName());
        System.out.println("Color: " + color);
        System.out.println("Area: " + String.format("%.2f", calculateArea()));
        System.out.println("Perimeter: " + String.format("%.2f", calculatePerimeter()));
    }

    // Final method - cannot be overridden by subclasses
    public final void printColor() {
        System.out.println("This shape is " + color);
    }
}

// Concrete class implementing abstract methods
class Rectangle extends Shape {
    private double width;
    private double height;

    public Rectangle(String color, double width, double height) {
        super(color);
        this.width = width;
        this.height = height;
    }

    // Implementation of abstract method
    @Override
    public double calculateArea() {
        return width * height;
    }

    // Implementation of abstract method
    @Override
    public double calculatePerimeter() {
        return 2 * (width + height);
    }

    // Rectangle-specific method
    public boolean isSquare() {
        return width == height;
    }

    @Override
    public void displayInfo() {
        super.displayInfo();
        System.out.println("Width: " + width);
        System.out.println("Height: " + height);
        System.out.println("Is Square: " + (isSquare() ? "Yes" : "No"));
    }
}

// Concrete class implementing abstract methods
class Circle extends Shape {
    private double radius;

    public Circle(String color, double radius) {
        super(color);
        this.radius = radius;
    }

    // Implementation of abstract method using Math.PI
    @Override
    public double calculateArea() {
        return Math.PI * radius * radius;
    }

    // Implementation of abstract method
    @Override
    public double calculatePerimeter() {
        return 2 * Math.PI * radius;
    }

    @Override
    public void displayInfo() {
        super.displayInfo();
        System.out.println("Radius: " + radius);
    }
}

// Concrete class implementing abstract methods
class Triangle extends Shape {
    private double side1;
    private double side2;
    private double side3;

    public Triangle(String color, double side1, double side2, double side3) {
        super(color);
        if (!isValidTriangle(side1, side2, side3)) {
            throw new IllegalArgumentException("Invalid triangle sides");
        }
        this.side1 = side1;
        this.side2 = side2;
        this.side3 = side3;
    }

    // Validate triangle using triangle inequality theorem
    private boolean isValidTriangle(double a, double b, double c) {
        return (a + b > c) && (a + c > b) && (b + c > a);
    }

    // Implementation using Heron's formula
    @Override
    public double calculateArea() {
        double s = calculatePerimeter() / 2; // semi-perimeter
        return Math.sqrt(s * (s - side1) * (s - side2) * (s - side3));
    }

    // Implementation of abstract method
    @Override
    public double calculatePerimeter() {
        return side1 + side2 + side3;
    }

    @Override
    public void displayInfo() {
        super.displayInfo();
        System.out.println("Sides: " + side1 + ", " + side2 + ", " + side3);
    }
}

public class AbstractDemo {
    public static void main(String[] args) {
        // Cannot instantiate abstract class - this would cause error:
        // Shape shape = new Shape("red"); // COMPILATION ERROR

        // Create concrete shape objects
        Rectangle rectangle = new Rectangle("Blue", 5, 3);
        Circle circle = new Circle("Red", 4);
        Triangle triangle = new Triangle("Green", 3, 4, 5);

        System.out.println("=== Rectangle ===");
        rectangle.displayInfo();
        rectangle.printColor();
        System.out.println();

        System.out.println("=== Circle ===");
        circle.displayInfo();
        circle.printColor();
        System.out.println();

        System.out.println("=== Triangle ===");
        triangle.displayInfo();
        triangle.printColor();
        System.out.println();

        // Polymorphism with abstract class - array of Shape references
        System.out.println("=== Polymorphic Array ===");
        Shape[] shapes = {
            new Rectangle("Yellow", 4, 4),
            new Circle("Purple", 3),
            new Triangle("Orange", 5, 5, 5),
            new Rectangle("Pink", 6, 2)
        };

        // Calculate total area of all shapes
        double totalArea = 0;
        for (Shape shape : shapes) {
            System.out.println(shape.getClass().getSimpleName() +
                             " - Area: " + String.format("%.2f", shape.calculateArea()));
            totalArea += shape.calculateArea();
        }
        System.out.println("\nTotal area of all shapes: " +
                          String.format("%.2f", totalArea));

        System.out.println("\n=== Testing Rectangle-specific method ===");
        Rectangle square = new Rectangle("White", 5, 5);
        System.out.println("Is this rectangle a square? " + square.isSquare());

        // Demonstrate downcasting to access subclass-specific methods
        System.out.println("\n=== Downcasting Example ===");
        for (Shape shape : shapes) {
            if (shape instanceof Rectangle) {
                Rectangle rect = (Rectangle) shape;
                System.out.println("Rectangle is square: " + rect.isSquare());
            }
        }

        // Try to create invalid triangle - will throw exception
        System.out.println("\n=== Testing Triangle Validation ===");
        try {
            Triangle invalidTriangle = new Triangle("Black", 1, 2, 10);
        } catch (IllegalArgumentException e) {
            System.out.println("Caught exception: " + e.getMessage());
        }
    }
}`,
    hint1: `Abstract methods have no body - they end with a semicolon. Concrete subclasses must provide implementations for all abstract methods.`,
    hint2: `Use the abstract keyword for both the class and the methods. Remember that abstract classes can have both abstract and concrete methods.`,
    whyItMatters: `Abstract classes are essential for creating well-designed object hierarchies. They allow you to define a common interface and shared behavior while forcing subclasses to implement specific details. This is perfect for situations where you have a general concept (like Shape) but the specific implementation varies (Rectangle, Circle). Abstract classes provide a middle ground between interfaces (pure abstraction) and concrete classes (full implementation), enabling better code organization and enforcing design contracts.`,
    order: 5,
    testCode: `import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;
import java.lang.reflect.*;

// Test 1: Shape is abstract class
class Test1 {
    @Test
    void testShapeIsAbstract() throws Exception {
        Class<?> cls = Class.forName("Shape");
        assertTrue(Modifier.isAbstract(cls.getModifiers()));
    }
}

// Test 2: Rectangle extends Shape
class Test2 {
    @Test
    void testRectangleExtendsShape() throws Exception {
        Class<?> shape = Class.forName("Shape");
        Class<?> rect = Class.forName("Rectangle");
        assertTrue(shape.isAssignableFrom(rect));
    }
}

// Test 3: Circle extends Shape
class Test3 {
    @Test
    void testCircleExtendsShape() throws Exception {
        Class<?> shape = Class.forName("Shape");
        Class<?> circle = Class.forName("Circle");
        assertTrue(shape.isAssignableFrom(circle));
    }
}

// Test 4: Triangle extends Shape
class Test4 {
    @Test
    void testTriangleExtendsShape() throws Exception {
        Class<?> shape = Class.forName("Shape");
        Class<?> triangle = Class.forName("Triangle");
        assertTrue(shape.isAssignableFrom(triangle));
    }
}

// Test 5: calculateArea is abstract in Shape
class Test5 {
    @Test
    void testCalculateAreaAbstract() throws Exception {
        Class<?> cls = Class.forName("Shape");
        Method method = cls.getMethod("calculateArea");
        assertTrue(Modifier.isAbstract(method.getModifiers()));
    }
}

// Test 6: calculatePerimeter is abstract in Shape
class Test6 {
    @Test
    void testCalculatePerimeterAbstract() throws Exception {
        Class<?> cls = Class.forName("Shape");
        Method method = cls.getMethod("calculatePerimeter");
        assertTrue(Modifier.isAbstract(method.getModifiers()));
    }
}

// Test 7: Rectangle calculateArea works correctly
class Test7 {
    @Test
    void testRectangleArea() throws Exception {
        Class<?> cls = Class.forName("Rectangle");
        Constructor<?> con = cls.getConstructor(String.class, double.class, double.class);
        Object rect = con.newInstance("Blue", 5.0, 3.0);
        Method calcArea = cls.getMethod("calculateArea");
        assertEquals(15.0, (double) calcArea.invoke(rect), 0.01);
    }
}

// Test 8: Circle calculateArea works correctly
class Test8 {
    @Test
    void testCircleArea() throws Exception {
        Class<?> cls = Class.forName("Circle");
        Constructor<?> con = cls.getConstructor(String.class, double.class);
        Object circle = con.newInstance("Red", 2.0);
        Method calcArea = cls.getMethod("calculateArea");
        assertEquals(Math.PI * 4, (double) calcArea.invoke(circle), 0.01);
    }
}

// Test 9: printColor is final method
class Test9 {
    @Test
    void testPrintColorIsFinal() throws Exception {
        Class<?> cls = Class.forName("Shape");
        Method method = cls.getMethod("printColor");
        assertTrue(Modifier.isFinal(method.getModifiers()));
    }
}

// Test 10: Rectangle has isSquare method
class Test10 {
    @Test
    void testRectangleIsSquare() throws Exception {
        Class<?> cls = Class.forName("Rectangle");
        Constructor<?> con = cls.getConstructor(String.class, double.class, double.class);
        Object square = con.newInstance("White", 5.0, 5.0);
        Object rect = con.newInstance("Black", 5.0, 3.0);
        Method isSquare = cls.getMethod("isSquare");
        assertTrue((boolean) isSquare.invoke(square));
        assertFalse((boolean) isSquare.invoke(rect));
    }
}`,
    translations: {
        ru: {
            title: 'Абстрактные Классы и Методы',
            solutionCode: `// Абстрактный класс - не может быть инстанцирован, служит шаблоном
abstract class Shape {
    protected String color;

    public Shape(String color) {
        this.color = color;
    }

    // Абстрактные методы - должны быть реализованы конкретными подклассами
    public abstract double calculateArea();
    public abstract double calculatePerimeter();

    // Конкретный метод - наследуется всеми подклассами
    public void displayInfo() {
        System.out.println("Фигура: " + this.getClass().getSimpleName());
        System.out.println("Цвет: " + color);
        System.out.println("Площадь: " + String.format("%.2f", calculateArea()));
        System.out.println("Периметр: " + String.format("%.2f", calculatePerimeter()));
    }

    // Финальный метод - не может быть переопределен подклассами
    public final void printColor() {
        System.out.println("Эта фигура " + color + " цвета");
    }
}

// Конкретный класс, реализующий абстрактные методы
class Rectangle extends Shape {
    private double width;
    private double height;

    public Rectangle(String color, double width, double height) {
        super(color);
        this.width = width;
        this.height = height;
    }

    // Реализация абстрактного метода
    @Override
    public double calculateArea() {
        return width * height;
    }

    // Реализация абстрактного метода
    @Override
    public double calculatePerimeter() {
        return 2 * (width + height);
    }

    // Метод специфичный для Rectangle
    public boolean isSquare() {
        return width == height;
    }

    @Override
    public void displayInfo() {
        super.displayInfo();
        System.out.println("Ширина: " + width);
        System.out.println("Высота: " + height);
        System.out.println("Является квадратом: " + (isSquare() ? "Да" : "Нет"));
    }
}

// Конкретный класс, реализующий абстрактные методы
class Circle extends Shape {
    private double radius;

    public Circle(String color, double radius) {
        super(color);
        this.radius = radius;
    }

    // Реализация абстрактного метода с использованием Math.PI
    @Override
    public double calculateArea() {
        return Math.PI * radius * radius;
    }

    // Реализация абстрактного метода
    @Override
    public double calculatePerimeter() {
        return 2 * Math.PI * radius;
    }

    @Override
    public void displayInfo() {
        super.displayInfo();
        System.out.println("Радиус: " + radius);
    }
}

// Конкретный класс, реализующий абстрактные методы
class Triangle extends Shape {
    private double side1;
    private double side2;
    private double side3;

    public Triangle(String color, double side1, double side2, double side3) {
        super(color);
        if (!isValidTriangle(side1, side2, side3)) {
            throw new IllegalArgumentException("Недопустимые стороны треугольника");
        }
        this.side1 = side1;
        this.side2 = side2;
        this.side3 = side3;
    }

    // Проверка треугольника с использованием теоремы о неравенстве треугольника
    private boolean isValidTriangle(double a, double b, double c) {
        return (a + b > c) && (a + c > b) && (b + c > a);
    }

    // Реализация с использованием формулы Герона
    @Override
    public double calculateArea() {
        double s = calculatePerimeter() / 2; // полупериметр
        return Math.sqrt(s * (s - side1) * (s - side2) * (s - side3));
    }

    // Реализация абстрактного метода
    @Override
    public double calculatePerimeter() {
        return side1 + side2 + side3;
    }

    @Override
    public void displayInfo() {
        super.displayInfo();
        System.out.println("Стороны: " + side1 + ", " + side2 + ", " + side3);
    }
}

public class AbstractDemo {
    public static void main(String[] args) {
        // Невозможно инстанцировать абстрактный класс - это вызовет ошибку:
        // Shape shape = new Shape("red"); // ОШИБКА КОМПИЛЯЦИИ

        // Создание объектов конкретных фигур
        Rectangle rectangle = new Rectangle("Синий", 5, 3);
        Circle circle = new Circle("Красный", 4);
        Triangle triangle = new Triangle("Зеленый", 3, 4, 5);

        System.out.println("=== Прямоугольник ===");
        rectangle.displayInfo();
        rectangle.printColor();
        System.out.println();

        System.out.println("=== Круг ===");
        circle.displayInfo();
        circle.printColor();
        System.out.println();

        System.out.println("=== Треугольник ===");
        triangle.displayInfo();
        triangle.printColor();
        System.out.println();

        // Полиморфизм с абстрактным классом - массив ссылок Shape
        System.out.println("=== Полиморфный массив ===");
        Shape[] shapes = {
            new Rectangle("Желтый", 4, 4),
            new Circle("Фиолетовый", 3),
            new Triangle("Оранжевый", 5, 5, 5),
            new Rectangle("Розовый", 6, 2)
        };

        // Вычисление общей площади всех фигур
        double totalArea = 0;
        for (Shape shape : shapes) {
            System.out.println(shape.getClass().getSimpleName() +
                             " - Площадь: " + String.format("%.2f", shape.calculateArea()));
            totalArea += shape.calculateArea();
        }
        System.out.println("\nОбщая площадь всех фигур: " +
                          String.format("%.2f", totalArea));

        System.out.println("\n=== Тестирование метода специфичного для Rectangle ===");
        Rectangle square = new Rectangle("Белый", 5, 5);
        System.out.println("Является ли этот прямоугольник квадратом? " + square.isSquare());

        // Демонстрация нисходящего приведения для доступа к методам подкласса
        System.out.println("\n=== Пример нисходящего приведения ===");
        for (Shape shape : shapes) {
            if (shape instanceof Rectangle) {
                Rectangle rect = (Rectangle) shape;
                System.out.println("Прямоугольник является квадратом: " + rect.isSquare());
            }
        }

        // Попытка создать недопустимый треугольник - вызовет исключение
        System.out.println("\n=== Тестирование валидации треугольника ===");
        try {
            Triangle invalidTriangle = new Triangle("Черный", 1, 2, 10);
        } catch (IllegalArgumentException e) {
            System.out.println("Поймано исключение: " + e.getMessage());
        }
    }
}`,
            description: `Создайте иерархию фигур, используя абстрактные классы, чтобы понять, когда и как использовать абстракцию в Java.

**Требования:**
1. Создайте абстрактный класс **Shape** с:
   1.1. Защищенным полем: color (String)
   1.2. Конструктором для инициализации color
   1.3. Абстрактными методами: calculateArea(), calculatePerimeter()
   1.4. Конкретным методом: displayInfo(), который показывает детали фигуры
   1.5. Финальным методом: printColor(), который не может быть переопределен

2. Создайте класс **Rectangle**, расширяющий Shape:
   2.1. Приватные поля: width, height
   2.2. Конструктор, инициализирующий все поля
   2.3. Реализуйте абстрактные методы с формулами для прямоугольника
   2.4. Добавьте специфичный метод: isSquare()

3. Создайте класс **Circle**, расширяющий Shape:
   3.1. Приватное поле: radius
   3.2. Конструктор, инициализирующий все поля
   3.3. Реализуйте абстрактные методы с формулами для круга
   3.4. Используйте Math.PI для вычислений

4. Создайте класс **Triangle**, расширяющий Shape:
   4.1. Приватные поля: side1, side2, side3
   4.2. Конструктор, инициализирующий все поля
   4.3. Реализуйте абстрактные методы (используйте формулу Герона для площади)
   4.4. Добавьте метод валидации: isValidTriangle()

5. В методе main:
   5.1. Создайте экземпляры всех конкретных фигур
   5.2. Сохраните их в массиве Shape (продемонстрируйте полиморфизм)
   5.3. Вычислите и отобразите площадь/периметр для каждой
   5.4. Покажите, что нельзя инстанцировать абстрактный класс

**Цели обучения:**
- Понять, когда использовать абстрактные классы против конкретных классов
- Изучить разницу между абстрактными и конкретными методами
- Практиковать реализацию абстрактных методов
- Понять ключевое слово final
- Увидеть, как абстрактные классы обеспечивают полиморфизм`,
            hint1: `Абстрактные методы не имеют тела - они заканчиваются точкой с запятой. Конкретные подклассы должны предоставить реализации для всех абстрактных методов.`,
            hint2: `Используйте ключевое слово abstract как для класса, так и для методов. Помните, что абстрактные классы могут иметь как абстрактные, так и конкретные методы.`,
            whyItMatters: `Абстрактные классы необходимы для создания хорошо спроектированных иерархий объектов. Они позволяют определить общий интерфейс и общее поведение, заставляя подклассы реализовывать конкретные детали. Это идеально подходит для ситуаций, когда у вас есть общая концепция (например, Shape), но конкретная реализация варьируется (Rectangle, Circle). Абстрактные классы обеспечивают золотую середину между интерфейсами (чистая абстракция) и конкретными классами (полная реализация), обеспечивая лучшую организацию кода и соблюдение контрактов проектирования.`
        },
        uz: {
            title: 'Abstrakt Sinflar va Metodlar',
            solutionCode: `// Abstrakt sinf - nusxa yaratib bo'lmaydi, shablon vazifasini bajaradi
abstract class Shape {
    protected String color;

    public Shape(String color) {
        this.color = color;
    }

    // Abstrakt metodlar - konkret subsinflar tomonidan amalga oshirilishi kerak
    public abstract double calculateArea();
    public abstract double calculatePerimeter();

    // Konkret metod - barcha subsinflar tomonidan meros olinadi
    public void displayInfo() {
        System.out.println("Shakl: " + this.getClass().getSimpleName());
        System.out.println("Rang: " + color);
        System.out.println("Maydoni: " + String.format("%.2f", calculateArea()));
        System.out.println("Perimetri: " + String.format("%.2f", calculatePerimeter()));
    }

    // Final metod - subsinflar tomonidan qayta yozib bo'lmaydi
    public final void printColor() {
        System.out.println("Bu shakl " + color + " rangda");
    }
}

// Abstrakt metodlarni amalga oshiruvchi konkret sinf
class Rectangle extends Shape {
    private double width;
    private double height;

    public Rectangle(String color, double width, double height) {
        super(color);
        this.width = width;
        this.height = height;
    }

    // Abstrakt metodning amalga oshirilishi
    @Override
    public double calculateArea() {
        return width * height;
    }

    // Abstrakt metodning amalga oshirilishi
    @Override
    public double calculatePerimeter() {
        return 2 * (width + height);
    }

    // Rectangle ga xos metod
    public boolean isSquare() {
        return width == height;
    }

    @Override
    public void displayInfo() {
        super.displayInfo();
        System.out.println("Kenglik: " + width);
        System.out.println("Balandlik: " + height);
        System.out.println("Kvadratmi: " + (isSquare() ? "Ha" : "Yo'q"));
    }
}

// Abstrakt metodlarni amalga oshiruvchi konkret sinf
class Circle extends Shape {
    private double radius;

    public Circle(String color, double radius) {
        super(color);
        this.radius = radius;
    }

    // Math.PI dan foydalanib abstrakt metodning amalga oshirilishi
    @Override
    public double calculateArea() {
        return Math.PI * radius * radius;
    }

    // Abstrakt metodning amalga oshirilishi
    @Override
    public double calculatePerimeter() {
        return 2 * Math.PI * radius;
    }

    @Override
    public void displayInfo() {
        super.displayInfo();
        System.out.println("Radius: " + radius);
    }
}

// Abstrakt metodlarni amalga oshiruvchi konkret sinf
class Triangle extends Shape {
    private double side1;
    private double side2;
    private double side3;

    public Triangle(String color, double side1, double side2, double side3) {
        super(color);
        if (!isValidTriangle(side1, side2, side3)) {
            throw new IllegalArgumentException("Noto'g'ri uchburchak tomonlari");
        }
        this.side1 = side1;
        this.side2 = side2;
        this.side3 = side3;
    }

    // Uchburchak tengsizligi teoremasidan foydalanib tekshirish
    private boolean isValidTriangle(double a, double b, double c) {
        return (a + b > c) && (a + c > b) && (b + c > a);
    }

    // Heron formulasidan foydalanib amalga oshirish
    @Override
    public double calculateArea() {
        double s = calculatePerimeter() / 2; // yarim perimetr
        return Math.sqrt(s * (s - side1) * (s - side2) * (s - side3));
    }

    // Abstrakt metodning amalga oshirilishi
    @Override
    public double calculatePerimeter() {
        return side1 + side2 + side3;
    }

    @Override
    public void displayInfo() {
        super.displayInfo();
        System.out.println("Tomonlar: " + side1 + ", " + side2 + ", " + side3);
    }
}

public class AbstractDemo {
    public static void main(String[] args) {
        // Abstrakt sinfdan nusxa yaratib bo'lmaydi - bu xatoga sabab bo'ladi:
        // Shape shape = new Shape("red"); // KOMPILYATSIYA XATOSI

        // Konkret shakl obyektlarini yaratish
        Rectangle rectangle = new Rectangle("Ko'k", 5, 3);
        Circle circle = new Circle("Qizil", 4);
        Triangle triangle = new Triangle("Yashil", 3, 4, 5);

        System.out.println("=== To'rtburchak ===");
        rectangle.displayInfo();
        rectangle.printColor();
        System.out.println();

        System.out.println("=== Doira ===");
        circle.displayInfo();
        circle.printColor();
        System.out.println();

        System.out.println("=== Uchburchak ===");
        triangle.displayInfo();
        triangle.printColor();
        System.out.println();

        // Abstrakt sinf bilan polimorfizm - Shape havolalari massivi
        System.out.println("=== Polimorf Massiv ===");
        Shape[] shapes = {
            new Rectangle("Sariq", 4, 4),
            new Circle("Binafsha", 3),
            new Triangle("To'q sariq", 5, 5, 5),
            new Rectangle("Pushti", 6, 2)
        };

        // Barcha shakllarning umumiy maydonini hisoblash
        double totalArea = 0;
        for (Shape shape : shapes) {
            System.out.println(shape.getClass().getSimpleName() +
                             " - Maydoni: " + String.format("%.2f", shape.calculateArea()));
            totalArea += shape.calculateArea();
        }
        System.out.println("\nBarcha shakllarning umumiy maydoni: " +
                          String.format("%.2f", totalArea));

        System.out.println("\n=== Rectangle ga xos metodini sinash ===");
        Rectangle square = new Rectangle("Oq", 5, 5);
        System.out.println("Bu to'rtburchak kvadratmi? " + square.isSquare());

        // Subsinf metodlariga kirish uchun pastga kastingni namoyish etish
        System.out.println("\n=== Pastga Kasting Misoli ===");
        for (Shape shape : shapes) {
            if (shape instanceof Rectangle) {
                Rectangle rect = (Rectangle) shape;
                System.out.println("To'rtburchak kvadrat: " + rect.isSquare());
            }
        }

        // Noto'g'ri uchburchak yaratishga harakat - istisno beradi
        System.out.println("\n=== Uchburchak Tekshiruvini Sinash ===");
        try {
            Triangle invalidTriangle = new Triangle("Qora", 1, 2, 10);
        } catch (IllegalArgumentException e) {
            System.out.println("Istisno ushlandi: " + e.getMessage());
        }
    }
}`,
            description: `Java-da abstraktsiyadan qachon va qanday foydalanishni tushunish uchun abstrakt sinflardan foydalanib shakllar ierarxiyasini yarating.

**Talablar:**
1. Quyidagilar bilan abstrakt **Shape** sinfini yarating:
   1.1. Himoyalangan maydon: color (String)
   1.2. color ni ishga tushirish uchun konstruktor
   1.3. Abstrakt metodlar: calculateArea(), calculatePerimeter()
   1.4. Konkret metod: shakl tafsilotlarini ko'rsatadigan displayInfo()
   1.5. Final metod: qayta yozib bo'lmaydigan printColor()

2. Shape ni kengaytiradigan **Rectangle** sinfini yarating:
   2.1. Xususiy maydonlar: width, height
   2.2. Barcha maydonlarni ishga tushiruvchi konstruktor
   2.3. To'rtburchakka xos formulalar bilan abstrakt metodlarni amalga oshiring
   2.4. Maxsus metod qo'shing: isSquare()

3. Shape ni kengaytiradigan **Circle** sinfini yarating:
   3.1. Xususiy maydon: radius
   3.2. Barcha maydonlarni ishga tushiruvchi konstruktor
   3.3. Doiraga xos formulalar bilan abstrakt metodlarni amalga oshiring
   3.4. Hisoblashlar uchun Math.PI dan foydalaning

4. Shape ni kengaytiradigan **Triangle** sinfini yarating:
   4.1. Xususiy maydonlar: side1, side2, side3
   4.2. Barcha maydonlarni ishga tushiruvchi konstruktor
   4.3. Abstrakt metodlarni amalga oshiring (maydon uchun Heron formulasidan foydalaning)
   4.4. Tekshirish metodini qo'shing: isValidTriangle()

5. Main metodida:
   5.1. Barcha konkret shakllarning nusxalarini yarating
   5.2. Ularni Shape massivida saqlang (polimorfizmni ko'rsating)
   5.3. Har biri uchun maydon/perimetrni hisoblang va ko'rsating
   5.4. Abstrakt sinfdan nusxa yaratib bo'lmasligini ko'rsating

**O'rganish maqsadlari:**
- Abstrakt sinflar va konkret sinflardan qachon foydalanishni tushunish
- Abstrakt va konkret metodlar o'rtasidagi farqni o'rganish
- Abstrakt metodlarni amalga oshirishda amaliyot
- final kalit so'zini tushunish
- Abstrakt sinflar polimorfizmni qanday ta'minlashini ko'rish`,
            hint1: `Abstrakt metodlar tanaga ega emas - ular nuqta-vergul bilan tugaydi. Konkret subsinflar barcha abstrakt metodlar uchun amalga oshirishlarni taqdim etishi kerak.`,
            hint2: `Ham sinf, ham metodlar uchun abstract kalit so'zidan foydalaning. Abstrakt sinflar ham abstrakt, ham konkret metodlarga ega bo'lishi mumkinligini eslang.`,
            whyItMatters: `Abstrakt sinflar yaxshi loyihalangan obyekt ierarxiyalarini yaratish uchun zarurdir. Ular umumiy interfeys va umumiy xatti-harakatni belgilash imkonini beradi, subsinflarni ma'lum tafsilotlarni amalga oshirishga majbur qiladi. Bu umumiy tushunchaga ega bo'lgan (masalan, Shape), lekin aniq amalga oshirish turlicha bo'lgan vaziyatlar uchun juda mos keladi (Rectangle, Circle). Abstrakt sinflar interfeyslar (sof abstraktsiya) va konkret sinflar (to'liq amalga oshirish) o'rtasida o'rta yo'lni ta'minlaydi, bu yaxshiroq kod tashkilotini ta'minlaydi va dizayn shartnomalarini amalga oshirishni majbur qiladi.`
        }
    }
};

export default task;
