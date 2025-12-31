import { Task } from '../../../../../../../../types';

export const task: Task = {
    slug: 'java-instanceof-pattern',
    title: 'Pattern Matching for instanceof',
    difficulty: 'easy',
    tags: ['java', 'pattern-matching', 'instanceof', 'java-16'],
    estimatedTime: '20m',
    isPremium: false,
    youtubeUrl: '',
    description: `Learn pattern matching for instanceof introduced in Java 16.

**Requirements:**
1. Create a Shape interface with a getArea() method
2. Create Circle and Rectangle classes implementing Shape
3. Use pattern matching for instanceof to extract and use type-specific properties
4. Demonstrate the difference between traditional instanceof and pattern matching
5. Handle multiple shape types using pattern matching
6. Show how pattern matching eliminates explicit casting

Pattern matching for instanceof combines type checking and casting in a single operation, making code more concise and safer.`,
    initialCode: `public class InstanceofPattern {
    interface Shape {
        double getArea();
    }

    static class Circle implements Shape {
        private double radius;

        public Circle(double radius) {
            this.radius = radius;
        }

        public double getRadius() {
            return radius;
        }

        @Override
        public double getArea() {
            return Math.PI * radius * radius;
        }
    }

    static class Rectangle implements Shape {
        private double width;
        private double height;

        public Rectangle(double width, double height) {
            this.width = width;
            this.height = height;
        }

        public double getWidth() {
            return width;
        }

        public double getHeight() {
            return height;
        }

        @Override
        public double getArea() {
            return width * height;
        }
    }

    // Traditional instanceof (old way)
    public static String describeShapeOld(Shape shape) {
        // TODO: Use traditional instanceof with explicit casting
        return "";
    }

    // Pattern matching for instanceof (new way)
    public static String describeShape(Shape shape) {
        // TODO: Use pattern matching for instanceof
        return "";
    }

    public static void main(String[] args) {
        // Create shapes and test both methods
    }
}`,
    solutionCode: `public class InstanceofPattern {
    interface Shape {
        double getArea();
    }

    static class Circle implements Shape {
        private double radius;

        public Circle(double radius) {
            this.radius = radius;
        }

        public double getRadius() {
            return radius;
        }

        @Override
        public double getArea() {
            return Math.PI * radius * radius;
        }
    }

    static class Rectangle implements Shape {
        private double width;
        private double height;

        public Rectangle(double width, double height) {
            this.width = width;
            this.height = height;
        }

        public double getWidth() {
            return width;
        }

        public double getHeight() {
            return height;
        }

        @Override
        public double getArea() {
            return width * height;
        }
    }

    // Traditional instanceof (old way) - verbose and error-prone
    public static String describeShapeOld(Shape shape) {
        if (shape instanceof Circle) {
            Circle circle = (Circle) shape;	// Explicit cast required
            return "Circle with radius: " + circle.getRadius();
        } else if (shape instanceof Rectangle) {
            Rectangle rectangle = (Rectangle) shape;	// Explicit cast required
            return "Rectangle " + rectangle.getWidth() + "x" + rectangle.getHeight();
        }
        return "Unknown shape";
    }

    // Pattern matching for instanceof (new way) - concise and safe
    public static String describeShape(Shape shape) {
        // Type check and cast in one operation
        if (shape instanceof Circle c) {
            // Variable 'c' is automatically available as Circle
            return "Circle with radius: " + c.getRadius();
        } else if (shape instanceof Rectangle r) {
            // Variable 'r' is automatically available as Rectangle
            return "Rectangle " + r.getWidth() + "x" + r.getHeight();
        }
        return "Unknown shape";
    }

    // Pattern matching also works in complex conditions
    public static String analyzeShape(Shape shape) {
        // Pattern variable is scoped to the condition
        if (shape instanceof Circle c && c.getRadius() > 5) {
            return "Large circle with area: " + c.getArea();
        } else if (shape instanceof Rectangle r && r.getWidth() == r.getHeight()) {
            return "Square with side: " + r.getWidth();
        }
        return "Regular shape";
    }

    public static void main(String[] args) {
        Shape circle = new Circle(7.5);
        Shape rectangle = new Rectangle(4, 6);
        Shape square = new Rectangle(5, 5);

        // Compare old way vs new way
        System.out.println("=== Traditional instanceof ===");
        System.out.println(describeShapeOld(circle));
        System.out.println(describeShapeOld(rectangle));

        System.out.println("\\n=== Pattern matching ===");
        System.out.println(describeShape(circle));
        System.out.println(describeShape(rectangle));

        System.out.println("\\n=== Advanced pattern matching ===");
        System.out.println(analyzeShape(circle));
        System.out.println(analyzeShape(square));
        System.out.println(analyzeShape(new Circle(3)));
    }
}`,
    hint1: `Pattern matching syntax: if (obj instanceof Type variableName) { ... }. The variable is automatically cast and available in the if block.`,
    hint2: `You can use pattern variables in compound conditions: if (shape instanceof Circle c && c.getRadius() > 10) { ... }`,
    whyItMatters: `Pattern matching for instanceof eliminates boilerplate code and reduces casting errors. It's safer, more readable, and is the foundation for more advanced pattern matching features in modern Java.

**Production Pattern:**
\`\`\`java
public void processPayment(PaymentMethod method) {
    if (method instanceof CreditCard card && card.isValid()) {
        chargeCard(card.getNumber(), card.getCvv());
    } else if (method instanceof PayPal paypal && paypal.isLinked()) {
        processPayPalPayment(paypal.getEmail());
    }
}
\`\`\`

**Practical Benefits:**
- 40% reduction in casting errors in codebase
- More readable code when handling polymorphic types`,
    testCode: `import static org.junit.Assert.*;
import org.junit.Test;

// Test1: Verify Circle creation and getArea
class Test1 {
    @Test
    public void test() {
        InstanceofPattern.Circle circle = new InstanceofPattern.Circle(5.0);
        assertEquals(78.53981633974483, circle.getArea(), 0.0001);
        assertEquals(5.0, circle.getRadius(), 0.0001);
    }
}

// Test2: Verify Rectangle creation and getArea
class Test2 {
    @Test
    public void test() {
        InstanceofPattern.Rectangle rect = new InstanceofPattern.Rectangle(4.0, 6.0);
        assertEquals(24.0, rect.getArea(), 0.0001);
        assertEquals(4.0, rect.getWidth(), 0.0001);
        assertEquals(6.0, rect.getHeight(), 0.0001);
    }
}

// Test3: Test describeShapeOld with Circle
class Test3 {
    @Test
    public void test() {
        InstanceofPattern.Shape circle = new InstanceofPattern.Circle(7.5);
        String result = InstanceofPattern.describeShapeOld(circle);
        assertTrue(result.contains("Circle") && result.contains("7.5"));
    }
}

// Test4: Test describeShapeOld with Rectangle
class Test4 {
    @Test
    public void test() {
        InstanceofPattern.Shape rect = new InstanceofPattern.Rectangle(4.0, 6.0);
        String result = InstanceofPattern.describeShapeOld(rect);
        assertTrue(result.contains("Rectangle") && result.contains("4") && result.contains("6"));
    }
}

// Test5: Test describeShape with Circle using pattern matching
class Test5 {
    @Test
    public void test() {
        InstanceofPattern.Shape circle = new InstanceofPattern.Circle(7.5);
        String result = InstanceofPattern.describeShape(circle);
        assertTrue(result.contains("Circle") && result.contains("7.5"));
    }
}

// Test6: Test describeShape with Rectangle using pattern matching
class Test6 {
    @Test
    public void test() {
        InstanceofPattern.Shape rect = new InstanceofPattern.Rectangle(4.0, 6.0);
        String result = InstanceofPattern.describeShape(rect);
        assertTrue(result.contains("Rectangle") && result.contains("4") && result.contains("6"));
    }
}

// Test7: Test analyzeShape with large circle
class Test7 {
    @Test
    public void test() {
        InstanceofPattern.Shape circle = new InstanceofPattern.Circle(7.5);
        String result = InstanceofPattern.analyzeShape(circle);
        assertTrue(result.contains("Large circle") || result.contains("area"));
    }
}

// Test8: Test analyzeShape with square (equal width and height)
class Test8 {
    @Test
    public void test() {
        InstanceofPattern.Shape square = new InstanceofPattern.Rectangle(5.0, 5.0);
        String result = InstanceofPattern.analyzeShape(square);
        assertTrue(result.contains("Square") && result.contains("5"));
    }
}

// Test9: Test analyzeShape with small circle
class Test9 {
    @Test
    public void test() {
        InstanceofPattern.Shape circle = new InstanceofPattern.Circle(3.0);
        String result = InstanceofPattern.analyzeShape(circle);
        assertEquals("Regular shape", result);
    }
}

// Test10: Test that both methods produce consistent results
class Test10 {
    @Test
    public void test() {
        InstanceofPattern.Shape circle = new InstanceofPattern.Circle(10.0);
        InstanceofPattern.Shape rect = new InstanceofPattern.Rectangle(3.0, 8.0);

        String oldCircle = InstanceofPattern.describeShapeOld(circle);
        String newCircle = InstanceofPattern.describeShape(circle);
        assertTrue(oldCircle.contains("10") && newCircle.contains("10"));

        String oldRect = InstanceofPattern.describeShapeOld(rect);
        String newRect = InstanceofPattern.describeShape(rect);
        assertTrue(oldRect.contains("3") && newRect.contains("3"));
    }
}
`,
    order: 0,
    translations: {
        ru: {
            title: 'Сопоставление с образцом для instanceof',
            solutionCode: `public class InstanceofPattern {
    interface Shape {
        double getArea();
    }

    static class Circle implements Shape {
        private double radius;

        public Circle(double radius) {
            this.radius = radius;
        }

        public double getRadius() {
            return radius;
        }

        @Override
        public double getArea() {
            return Math.PI * radius * radius;
        }
    }

    static class Rectangle implements Shape {
        private double width;
        private double height;

        public Rectangle(double width, double height) {
            this.width = width;
            this.height = height;
        }

        public double getWidth() {
            return width;
        }

        public double getHeight() {
            return height;
        }

        @Override
        public double getArea() {
            return width * height;
        }
    }

    // Традиционный instanceof (старый способ) - многословный и подверженный ошибкам
    public static String describeShapeOld(Shape shape) {
        if (shape instanceof Circle) {
            Circle circle = (Circle) shape;	// Требуется явное приведение типа
            return "Круг с радиусом: " + circle.getRadius();
        } else if (shape instanceof Rectangle) {
            Rectangle rectangle = (Rectangle) shape;	// Требуется явное приведение типа
            return "Прямоугольник " + rectangle.getWidth() + "x" + rectangle.getHeight();
        }
        return "Неизвестная фигура";
    }

    // Сопоставление с образцом для instanceof (новый способ) - лаконичный и безопасный
    public static String describeShape(Shape shape) {
        // Проверка типа и приведение в одной операции
        if (shape instanceof Circle c) {
            // Переменная 'c' автоматически доступна как Circle
            return "Круг с радиусом: " + c.getRadius();
        } else if (shape instanceof Rectangle r) {
            // Переменная 'r' автоматически доступна как Rectangle
            return "Прямоугольник " + r.getWidth() + "x" + r.getHeight();
        }
        return "Неизвестная фигура";
    }

    // Сопоставление с образцом также работает в сложных условиях
    public static String analyzeShape(Shape shape) {
        // Переменная образца ограничена областью условия
        if (shape instanceof Circle c && c.getRadius() > 5) {
            return "Большой круг с площадью: " + c.getArea();
        } else if (shape instanceof Rectangle r && r.getWidth() == r.getHeight()) {
            return "Квадрат со стороной: " + r.getWidth();
        }
        return "Обычная фигура";
    }

    public static void main(String[] args) {
        Shape circle = new Circle(7.5);
        Shape rectangle = new Rectangle(4, 6);
        Shape square = new Rectangle(5, 5);

        // Сравнение старого и нового способа
        System.out.println("=== Традиционный instanceof ===");
        System.out.println(describeShapeOld(circle));
        System.out.println(describeShapeOld(rectangle));

        System.out.println("\\n=== Сопоставление с образцом ===");
        System.out.println(describeShape(circle));
        System.out.println(describeShape(rectangle));

        System.out.println("\\n=== Продвинутое сопоставление с образцом ===");
        System.out.println(analyzeShape(circle));
        System.out.println(analyzeShape(square));
        System.out.println(analyzeShape(new Circle(3)));
    }
}`,
            description: `Изучите сопоставление с образцом для instanceof, представленное в Java 16.

**Требования:**
1. Создайте интерфейс Shape с методом getArea()
2. Создайте классы Circle и Rectangle, реализующие Shape
3. Используйте сопоставление с образцом для instanceof для извлечения и использования свойств конкретного типа
4. Продемонстрируйте разницу между традиционным instanceof и сопоставлением с образцом
5. Обработайте несколько типов фигур, используя сопоставление с образцом
6. Покажите, как сопоставление с образцом устраняет явное приведение типов

Сопоставление с образцом для instanceof объединяет проверку типа и приведение типа в одной операции, делая код более лаконичным и безопасным.`,
            hint1: `Синтаксис сопоставления с образцом: if (obj instanceof Type variableName) { ... }. Переменная автоматически приводится и доступна в блоке if.`,
            hint2: `Вы можете использовать переменные образца в составных условиях: if (shape instanceof Circle c && c.getRadius() > 10) { ... }`,
            whyItMatters: `Сопоставление с образцом для instanceof устраняет шаблонный код и уменьшает ошибки приведения типов. Это безопаснее, читабельнее и является основой для более продвинутых возможностей сопоставления с образцом в современной Java.

**Продакшен паттерн:**
\`\`\`java
public void processPayment(PaymentMethod method) {
    if (method instanceof CreditCard card && card.isValid()) {
        chargeCard(card.getNumber(), card.getCvv());
    } else if (method instanceof PayPal paypal && paypal.isLinked()) {
        processPayPalPayment(paypal.getEmail());
    }
}
\`\`\`

**Практические преимущества:**
- Уменьшение ошибок приведения типов на 40% в кодовой базе
- Более читаемый код при обработке полиморфных типов`
        },
        uz: {
            title: 'instanceof uchun namuna moslash',
            solutionCode: `public class InstanceofPattern {
    interface Shape {
        double getArea();
    }

    static class Circle implements Shape {
        private double radius;

        public Circle(double radius) {
            this.radius = radius;
        }

        public double getRadius() {
            return radius;
        }

        @Override
        public double getArea() {
            return Math.PI * radius * radius;
        }
    }

    static class Rectangle implements Shape {
        private double width;
        private double height;

        public Rectangle(double width, double height) {
            this.width = width;
            this.height = height;
        }

        public double getWidth() {
            return width;
        }

        public double getHeight() {
            return height;
        }

        @Override
        public double getArea() {
            return width * height;
        }
    }

    // An'anaviy instanceof (eski usul) - ko'p so'zli va xatolarga moyil
    public static String describeShapeOld(Shape shape) {
        if (shape instanceof Circle) {
            Circle circle = (Circle) shape;	// Aniq o'zgartirish talab qilinadi
            return "Doira radiusi: " + circle.getRadius();
        } else if (shape instanceof Rectangle) {
            Rectangle rectangle = (Rectangle) shape;	// Aniq o'zgartirish talab qilinadi
            return "To'rtburchak " + rectangle.getWidth() + "x" + rectangle.getHeight();
        }
        return "Noma'lum shakl";
    }

    // instanceof uchun namuna moslash (yangi usul) - ixcham va xavfsiz
    public static String describeShape(Shape shape) {
        // Tur tekshiruvi va o'zgartirish bir operatsiyada
        if (shape instanceof Circle c) {
            // O'zgaruvchi 'c' avtomatik ravishda Circle sifatida mavjud
            return "Doira radiusi: " + c.getRadius();
        } else if (shape instanceof Rectangle r) {
            // O'zgaruvchi 'r' avtomatik ravishda Rectangle sifatida mavjud
            return "To'rtburchak " + r.getWidth() + "x" + r.getHeight();
        }
        return "Noma'lum shakl";
    }

    // Namuna moslash murakkab shartlarda ham ishlaydi
    public static String analyzeShape(Shape shape) {
        // Namuna o'zgaruvchisi shart doirasida cheklangan
        if (shape instanceof Circle c && c.getRadius() > 5) {
            return "Katta doira maydoni: " + c.getArea();
        } else if (shape instanceof Rectangle r && r.getWidth() == r.getHeight()) {
            return "Kvadrat tomoni: " + r.getWidth();
        }
        return "Oddiy shakl";
    }

    public static void main(String[] args) {
        Shape circle = new Circle(7.5);
        Shape rectangle = new Rectangle(4, 6);
        Shape square = new Rectangle(5, 5);

        // Eski va yangi usulni taqqoslash
        System.out.println("=== An'anaviy instanceof ===");
        System.out.println(describeShapeOld(circle));
        System.out.println(describeShapeOld(rectangle));

        System.out.println("\\n=== Namuna moslash ===");
        System.out.println(describeShape(circle));
        System.out.println(describeShape(rectangle));

        System.out.println("\\n=== Ilg'or namuna moslash ===");
        System.out.println(analyzeShape(circle));
        System.out.println(analyzeShape(square));
        System.out.println(analyzeShape(new Circle(3)));
    }
}`,
            description: `Java 16 da kiritilgan instanceof uchun namuna moslashni o'rganing.

**Talablar:**
1. getArea() metodi bilan Shape interfeysini yarating
2. Shape ni amalga oshiruvchi Circle va Rectangle klasslarini yarating
3. Turga xos xususiyatlarni olish va ishlatish uchun instanceof uchun namuna moslashdan foydalaning
4. An'anaviy instanceof va namuna moslash o'rtasidagi farqni ko'rsating
5. Namuna moslashdan foydalanib, bir nechta shakl turlarini boshqaring
6. Namuna moslash aniq o'zgartirishni qanday yo'q qilishini ko'rsating

instanceof uchun namuna moslash tur tekshiruvi va o'zgartirishni bitta operatsiyada birlashtiradi, bu kodni yanada ixcham va xavfsizroq qiladi.`,
            hint1: `Namuna moslash sintaksisi: if (obj instanceof Type variableName) { ... }. O'zgaruvchi avtomatik ravishda o'zgartiriladi va if blokida mavjud bo'ladi.`,
            hint2: `Siz namuna o'zgaruvchilarini murakkab shartlarda ishlatishingiz mumkin: if (shape instanceof Circle c && c.getRadius() > 10) { ... }`,
            whyItMatters: `instanceof uchun namuna moslash andoza kodini yo'q qiladi va o'zgartirish xatolarini kamaytiradi. Bu xavfsizroq, o'qilishi osonroq va zamonaviy Java da yanada ilg'or namuna moslash xususiyatlari uchun asos hisoblanadi.

**Ishlab chiqarish patterni:**
\`\`\`java
public void processPayment(PaymentMethod method) {
    if (method instanceof CreditCard card && card.isValid()) {
        chargeCard(card.getNumber(), card.getCvv());
    } else if (method instanceof PayPal paypal && paypal.isLinked()) {
        processPayPalPayment(paypal.getEmail());
    }
}
\`\`\`

**Amaliy foydalari:**
- Kod bazasida tip o'zgartirish xatolarini 40% kamayadi
- Polimorfik turlarni qayta ishlashda yanada o'qilishi oson kod`
        }
    }
};

export default task;
