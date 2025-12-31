import { Task } from '../../../../types';

export const task: Task = {
    slug: 'java-record-customization',
    title: 'Record Customization',
    difficulty: 'medium',
    tags: ['java', 'records', 'methods', 'static', 'customization'],
    estimatedTime: '30m',
    isPremium: false,
    youtubeUrl: '',
    description: `# Record Customization

While records are immutable and auto-generate most methods, you can still add custom instance methods, static methods, static fields, and even override the auto-generated methods to provide custom implementations.

## Requirements:
1. Create a Rectangle record with width and height:
   1. Add instance method: area()
   2. Add instance method: perimeter()
   3. Add instance method: isSquare()

2. Add static methods and fields:
   1. Static factory method: createSquare(int size)
   2. Static constant: ZERO (a rectangle with 0 width and height)

3. Override auto-generated methods:
   1. Override toString() to provide custom format
   2. Keep equals() and hashCode() as generated

4. Create a Product record with custom methods:
   1. discountedPrice(double percentage)
   2. isFree()
   3. Static method: mostExpensive(Product... products)

## Example Output:
\`\`\`
=== Rectangle Methods ===
Rectangle: Rectangle(10x20)
Area: 200
Perimeter: 60
Is square? false

Square: Rectangle(15x15)
Is square? true

Zero rectangle: Rectangle(0x0)

=== Product Methods ===
Laptop: Product[name=Laptop, price=1000.0]
Original price: $1000.0
With 20% discount: $800.0
Is free? false

Free item: Product[name=Sample, price=0.0]
Is free? true

Most expensive: Product[name=Phone, price=1500.0]
\`\`\``,
    initialCode: `// TODO: Create Rectangle record with custom methods

// TODO: Create Product record with custom methods

public class RecordCustomization {
    public static void main(String[] args) {
        // TODO: Demonstrate Rectangle methods

        // TODO: Demonstrate static methods

        // TODO: Demonstrate Product methods
    }
}`,
    solutionCode: `// Record with custom instance methods and static members
record Rectangle(int width, int height) {
    // Custom instance methods
    public int area() {
        return width * height;
    }

    public int perimeter() {
        return 2 * (width + height);
    }

    public boolean isSquare() {
        return width == height;
    }

    // Static factory method
    public static Rectangle createSquare(int size) {
        return new Rectangle(size, size);
    }

    // Static constant
    public static final Rectangle ZERO = new Rectangle(0, 0);

    // Override auto-generated toString() with custom format
    @Override
    public String toString() {
        return "Rectangle(" + width + "x" + height + ")";
    }
    // equals() and hashCode() remain auto-generated
}

// Record with business logic methods
record Product(String name, double price) {
    // Custom instance method with calculation
    public double discountedPrice(double percentage) {
        if (percentage < 0 || percentage > 100) {
            throw new IllegalArgumentException(
                "Discount must be between 0 and 100");
        }
        return price * (1 - percentage / 100);
    }

    public boolean isFree() {
        return price == 0.0;
    }

    // Static method with varargs
    public static Product mostExpensive(Product... products) {
        if (products.length == 0) {
            throw new IllegalArgumentException("No products provided");
        }

        Product max = products[0];
        for (Product product : products) {
            if (product.price() > max.price()) {
                max = product;
            }
        }
        return max;
    }
}

public class RecordCustomization {
    public static void main(String[] args) {
        System.out.println("=== Rectangle Methods ===");

        // Create rectangle and use custom methods
        Rectangle rect = new Rectangle(10, 20);
        System.out.println("Rectangle: " + rect);
        System.out.println("Area: " + rect.area());
        System.out.println("Perimeter: " + rect.perimeter());
        System.out.println("Is square? " + rect.isSquare());

        System.out.println();

        // Use static factory method
        Rectangle square = Rectangle.createSquare(15);
        System.out.println("Square: " + square);
        System.out.println("Is square? " + square.isSquare());

        System.out.println();

        // Use static constant
        System.out.println("Zero rectangle: " + Rectangle.ZERO);

        System.out.println("\\n=== Product Methods ===");

        // Create products and use custom methods
        Product laptop = new Product("Laptop", 1000.0);
        System.out.println("Laptop: " + laptop);
        System.out.println("Original price: $" + laptop.price());
        System.out.println("With 20% discount: $" +
            laptop.discountedPrice(20));
        System.out.println("Is free? " + laptop.isFree());

        System.out.println();

        Product freeItem = new Product("Sample", 0.0);
        System.out.println("Free item: " + freeItem);
        System.out.println("Is free? " + freeItem.isFree());

        System.out.println();

        // Use static method
        Product phone = new Product("Phone", 1500.0);
        Product tablet = new Product("Tablet", 800.0);
        Product mostExpensive = Product.mostExpensive(laptop, phone, tablet);
        System.out.println("Most expensive: " + mostExpensive);
    }
}`,
    hint1: `Add methods to records just like regular classes: public returnType methodName() { }. Static methods and fields work the same way too.`,
    hint2: `You can override toString(), equals(), or hashCode() by adding @Override and your implementation. The compiler will use your version instead of the auto-generated one.`,
    whyItMatters: `Record customization allows you to combine the benefits of auto-generated code with domain-specific logic. You can add business methods, factory methods, and utility functions while keeping the conciseness of records. This makes records suitable for more than just data carriers - they can encapsulate behavior too.

**Production Pattern:**
\`\`\`java
record Price(BigDecimal amount, String currency) {
    public Price withTax(BigDecimal taxRate) {
        return new Price(amount.multiply(BigDecimal.ONE.add(taxRate)), currency);
    }

    public static Price zero(String currency) {
        return new Price(BigDecimal.ZERO, currency);
    }
}
\`\`\`

**Practical Benefits:**
- Encapsulation of business logic with data immutability
- Factory methods for commonly used values`,
    testCode: `import static org.junit.Assert.*;
import org.junit.Test;

// Test1: Verify Rectangle area calculation
class Test1 {
    @Test
    public void testRectangleArea() {
        Rectangle rect = new Rectangle(10, 20);
        assertEquals(200, rect.area());
    }
}

// Test2: Verify Rectangle perimeter calculation
class Test2 {
    @Test
    public void testRectanglePerimeter() {
        Rectangle rect = new Rectangle(10, 20);
        assertEquals(60, rect.perimeter());
    }
}

// Test3: Verify Rectangle isSquare method
class Test3 {
    @Test
    public void testIsSquare() {
        Rectangle rect = new Rectangle(15, 15);
        assertTrue(rect.isSquare());
        Rectangle nonSquare = new Rectangle(10, 20);
        assertFalse(nonSquare.isSquare());
    }
}

// Test4: Verify Rectangle createSquare factory method
class Test4 {
    @Test
    public void testCreateSquare() {
        Rectangle square = Rectangle.createSquare(25);
        assertEquals(25, square.width());
        assertEquals(25, square.height());
        assertTrue(square.isSquare());
    }
}

// Test5: Verify Rectangle ZERO constant
class Test5 {
    @Test
    public void testZeroConstant() {
        Rectangle zero = Rectangle.ZERO;
        assertEquals(0, zero.width());
        assertEquals(0, zero.height());
        assertEquals(0, zero.area());
    }
}

// Test6: Verify Rectangle custom toString
class Test6 {
    @Test
    public void testCustomToString() {
        Rectangle rect = new Rectangle(10, 20);
        String str = rect.toString();
        assertEquals("Rectangle(10x20)", str);
    }
}

// Test7: Verify Product discountedPrice method
class Test7 {
    @Test
    public void testDiscountedPrice() {
        Product product = new Product("Laptop", 1000.0);
        double discounted = product.discountedPrice(20);
        assertEquals(800.0, discounted, 0.01);
    }
}

// Test8: Verify Product isFree method
class Test8 {
    @Test
    public void testIsFree() {
        Product freeItem = new Product("Sample", 0.0);
        assertTrue(freeItem.isFree());
        Product paidItem = new Product("Book", 50.0);
        assertFalse(paidItem.isFree());
    }
}

// Test9: Verify Product mostExpensive static method
class Test9 {
    @Test
    public void testMostExpensive() {
        Product laptop = new Product("Laptop", 1000.0);
        Product phone = new Product("Phone", 1500.0);
        Product tablet = new Product("Tablet", 800.0);
        Product result = Product.mostExpensive(laptop, phone, tablet);
        assertEquals(phone, result);
    }
}

// Test10: Verify Product discountedPrice validation
class Test10 {
    @Test(expected = IllegalArgumentException.class)
    public void testInvalidDiscount() {
        Product product = new Product("Item", 100.0);
        product.discountedPrice(150);
    }
}`,
    order: 3,
    translations: {
        ru: {
            title: 'Настройка записей',
            solutionCode: `// Record с пользовательскими методами экземпляра и статическими членами
record Rectangle(int width, int height) {
    // Пользовательские методы экземпляра
    public int area() {
        return width * height;
    }

    public int perimeter() {
        return 2 * (width + height);
    }

    public boolean isSquare() {
        return width == height;
    }

    // Статический фабричный метод
    public static Rectangle createSquare(int size) {
        return new Rectangle(size, size);
    }

    // Статическая константа
    public static final Rectangle ZERO = new Rectangle(0, 0);

    // Переопределение автогенерированного toString() с пользовательским форматом
    @Override
    public String toString() {
        return "Rectangle(" + width + "x" + height + ")";
    }
    // equals() и hashCode() остаются автогенерированными
}

// Record с методами бизнес-логики
record Product(String name, double price) {
    // Пользовательский метод экземпляра с вычислением
    public double discountedPrice(double percentage) {
        if (percentage < 0 || percentage > 100) {
            throw new IllegalArgumentException(
                "Discount must be between 0 and 100");
        }
        return price * (1 - percentage / 100);
    }

    public boolean isFree() {
        return price == 0.0;
    }

    // Статический метод с varargs
    public static Product mostExpensive(Product... products) {
        if (products.length == 0) {
            throw new IllegalArgumentException("No products provided");
        }

        Product max = products[0];
        for (Product product : products) {
            if (product.price() > max.price()) {
                max = product;
            }
        }
        return max;
    }
}

public class RecordCustomization {
    public static void main(String[] args) {
        System.out.println("=== Методы Rectangle ===");

        // Создание прямоугольника и использование пользовательских методов
        Rectangle rect = new Rectangle(10, 20);
        System.out.println("Rectangle: " + rect);
        System.out.println("Area: " + rect.area());
        System.out.println("Perimeter: " + rect.perimeter());
        System.out.println("Is square? " + rect.isSquare());

        System.out.println();

        // Использование статического фабричного метода
        Rectangle square = Rectangle.createSquare(15);
        System.out.println("Square: " + square);
        System.out.println("Is square? " + square.isSquare());

        System.out.println();

        // Использование статической константы
        System.out.println("Zero rectangle: " + Rectangle.ZERO);

        System.out.println("\\n=== Методы Product ===");

        // Создание продуктов и использование пользовательских методов
        Product laptop = new Product("Laptop", 1000.0);
        System.out.println("Laptop: " + laptop);
        System.out.println("Original price: $" + laptop.price());
        System.out.println("With 20% discount: $" +
            laptop.discountedPrice(20));
        System.out.println("Is free? " + laptop.isFree());

        System.out.println();

        Product freeItem = new Product("Sample", 0.0);
        System.out.println("Free item: " + freeItem);
        System.out.println("Is free? " + freeItem.isFree());

        System.out.println();

        // Использование статического метода
        Product phone = new Product("Phone", 1500.0);
        Product tablet = new Product("Tablet", 800.0);
        Product mostExpensive = Product.mostExpensive(laptop, phone, tablet);
        System.out.println("Most expensive: " + mostExpensive);
    }
}`,
            description: `# Настройка записей

Хотя records неизменяемы и автоматически генерируют большинство методов, вы все равно можете добавлять пользовательские методы экземпляра, статические методы, статические поля и даже переопределять автогенерированные методы для предоставления пользовательских реализаций.

## Требования:
1. Создайте record Rectangle с шириной и высотой:
   1. Добавьте метод экземпляра: area()
   2. Добавьте метод экземпляра: perimeter()
   3. Добавьте метод экземпляра: isSquare()

2. Добавьте статические методы и поля:
   1. Статический фабричный метод: createSquare(int size)
   2. Статическая константа: ZERO (прямоугольник с нулевой шириной и высотой)

3. Переопределите автогенерированные методы:
   1. Переопределите toString() для предоставления пользовательского формата
   2. Оставьте equals() и hashCode() сгенерированными

4. Создайте record Product с пользовательскими методами:
   1. discountedPrice(double percentage)
   2. isFree()
   3. Статический метод: mostExpensive(Product... products)

## Пример вывода:
\`\`\`
=== Rectangle Methods ===
Rectangle: Rectangle(10x20)
Area: 200
Perimeter: 60
Is square? false

Square: Rectangle(15x15)
Is square? true

Zero rectangle: Rectangle(0x0)

=== Product Methods ===
Laptop: Product[name=Laptop, price=1000.0]
Original price: $1000.0
With 20% discount: $800.0
Is free? false

Free item: Product[name=Sample, price=0.0]
Is free? true

Most expensive: Product[name=Phone, price=1500.0]
\`\`\``,
            hint1: `Добавляйте методы к records так же, как к обычным классам: public returnType methodName() { }. Статические методы и поля работают точно так же.`,
            hint2: `Вы можете переопределить toString(), equals() или hashCode(), добавив @Override и вашу реализацию. Компилятор будет использовать вашу версию вместо автогенерированной.`,
            whyItMatters: `Настройка records позволяет сочетать преимущества автогенерированного кода с доменной логикой. Вы можете добавлять бизнес-методы, фабричные методы и утилитарные функции, сохраняя при этом краткость records. Это делает records подходящими не только для носителей данных - они могут инкапсулировать поведение тоже.

**Продакшен паттерн:**
\`\`\`java
record Price(BigDecimal amount, String currency) {
    public Price withTax(BigDecimal taxRate) {
        return new Price(amount.multiply(BigDecimal.ONE.add(taxRate)), currency);
    }

    public static Price zero(String currency) {
        return new Price(BigDecimal.ZERO, currency);
    }
}
\`\`\`

**Практические преимущества:**
- Инкапсуляция бизнес-логики с неизменяемостью данных
- Фабричные методы для часто используемых значений`
        },
        uz: {
            title: `Recordlarni moslash`,
            solutionCode: `// Maxsus namuna metodlari va statik a'zolar bilan record
record Rectangle(int width, int height) {
    // Maxsus namuna metodlari
    public int area() {
        return width * height;
    }

    public int perimeter() {
        return 2 * (width + height);
    }

    public boolean isSquare() {
        return width == height;
    }

    // Statik fabrika metodi
    public static Rectangle createSquare(int size) {
        return new Rectangle(size, size);
    }

    // Statik o'zgarmas
    public static final Rectangle ZERO = new Rectangle(0, 0);

    // Maxsus format bilan avtomatik yaratilgan toString()ni bekor qilish
    @Override
    public String toString() {
        return "Rectangle(" + width + "x" + height + ")";
    }
    // equals() va hashCode() avtomatik yaratilgan holatda qoladi
}

// Biznes mantiq metodlari bilan record
record Product(String name, double price) {
    // Hisoblash bilan maxsus namuna metodi
    public double discountedPrice(double percentage) {
        if (percentage < 0 || percentage > 100) {
            throw new IllegalArgumentException(
                "Discount must be between 0 and 100");
        }
        return price * (1 - percentage / 100);
    }

    public boolean isFree() {
        return price == 0.0;
    }

    // Varargs bilan statik metod
    public static Product mostExpensive(Product... products) {
        if (products.length == 0) {
            throw new IllegalArgumentException("No products provided");
        }

        Product max = products[0];
        for (Product product : products) {
            if (product.price() > max.price()) {
                max = product;
            }
        }
        return max;
    }
}

public class RecordCustomization {
    public static void main(String[] args) {
        System.out.println("=== Rectangle metodlari ===");

        // To'rtburchak yaratish va maxsus metodlardan foydalanish
        Rectangle rect = new Rectangle(10, 20);
        System.out.println("Rectangle: " + rect);
        System.out.println("Area: " + rect.area());
        System.out.println("Perimeter: " + rect.perimeter());
        System.out.println("Is square? " + rect.isSquare());

        System.out.println();

        // Statik fabrika metodidan foydalanish
        Rectangle square = Rectangle.createSquare(15);
        System.out.println("Square: " + square);
        System.out.println("Is square? " + square.isSquare());

        System.out.println();

        // Statik o'zgarmasdan foydalanish
        System.out.println("Zero rectangle: " + Rectangle.ZERO);

        System.out.println("\\n=== Product metodlari ===");

        // Mahsulotlar yaratish va maxsus metodlardan foydalanish
        Product laptop = new Product("Laptop", 1000.0);
        System.out.println("Laptop: " + laptop);
        System.out.println("Original price: $" + laptop.price());
        System.out.println("With 20% discount: $" +
            laptop.discountedPrice(20));
        System.out.println("Is free? " + laptop.isFree());

        System.out.println();

        Product freeItem = new Product("Sample", 0.0);
        System.out.println("Free item: " + freeItem);
        System.out.println("Is free? " + freeItem.isFree());

        System.out.println();

        // Statik metoddan foydalanish
        Product phone = new Product("Phone", 1500.0);
        Product tablet = new Product("Tablet", 800.0);
        Product mostExpensive = Product.mostExpensive(laptop, phone, tablet);
        System.out.println("Most expensive: " + mostExpensive);
    }
}`,
            description: `# Recordlarni moslash

Recordlar o'zgarmas bo'lsa-da va ko'pchilik metodlarni avtomatik yaratsa-da, siz hali ham maxsus namuna metodlarini, statik metodlarni, statik maydonlarni qo'shishingiz va hatto maxsus implementatsiyalarni taqdim etish uchun avtomatik yaratilgan metodlarni bekor qilishingiz mumkin.

## Talablar:
1. Kenglik va balandlik bilan Rectangle recordini yarating:
   1. Namuna metodini qo'shing: area()
   2. Namuna metodini qo'shing: perimeter()
   3. Namuna metodini qo'shing: isSquare()

2. Statik metodlar va maydonlarni qo'shing:
   1. Statik fabrika metodi: createSquare(int size)
   2. Statik o'zgarmas: ZERO (0 kenglik va balandlikdagi to'rtburchak)

3. Avtomatik yaratilgan metodlarni bekor qiling:
   1. Maxsus formatni taqdim etish uchun toString()ni bekor qiling
   2. equals() va hashCode()ni yaratilgan holatda qoldiring

4. Maxsus metodlar bilan Product recordini yarating:
   1. discountedPrice(double percentage)
   2. isFree()
   3. Statik metod: mostExpensive(Product... products)

## Chiqish namunasi:
\`\`\`
=== Rectangle Methods ===
Rectangle: Rectangle(10x20)
Area: 200
Perimeter: 60
Is square? false

Square: Rectangle(15x15)
Is square? true

Zero rectangle: Rectangle(0x0)

=== Product Methods ===
Laptop: Product[name=Laptop, price=1000.0]
Original price: $1000.0
With 20% discount: $800.0
Is free? false

Free item: Product[name=Sample, price=0.0]
Is free? true

Most expensive: Product[name=Phone, price=1500.0]
\`\`\``,
            hint1: `Recordlarga oddiy klasslar kabi metodlar qo'shing: public returnType methodName() { }. Statik metodlar va maydonlar ham xuddi shunday ishlaydi.`,
            hint2: `@Override va o'z implementatsiyangizni qo'shish orqali toString(), equals() yoki hashCode()ni bekor qilishingiz mumkin. Kompilyator avtomatik yaratilgan o'rniga sizning versiyangizni ishlatadi.`,
            whyItMatters: `Recordlarni moslash avtomatik yaratilgan kodning afzalliklarini domen-spetsifik mantiq bilan birlashtirishga imkon beradi. Recordlarning qisqaligini saqlab, biznes metodlarini, fabrika metodlarini va yordamchi funksiyalarni qo'shishingiz mumkin. Bu recordlarni faqat ma'lumot tashuvchilardan ko'ra ko'proq narsa uchun mos qiladi - ular xatti-harakatni ham inkapsulyatsiya qilishi mumkin.

**Ishlab chiqarish patterni:**
\`\`\`java
record Price(BigDecimal amount, String currency) {
    public Price withTax(BigDecimal taxRate) {
        return new Price(amount.multiply(BigDecimal.ONE.add(taxRate)), currency);
    }

    public static Price zero(String currency) {
        return new Price(BigDecimal.ZERO, currency);
    }
}
\`\`\`

**Amaliy foydalari:**
- Ma'lumotlar o'zgarmasligi bilan biznes mantiqini inkapsulyatsiya qilish
- Tez-tez ishlatiladigan qiymatlar uchun fabrika metodlari`
        }
    }
};

export default task;
