import { Task } from '../../../../types';

export const task: Task = {
    slug: 'java-variables-primitive-types',
    title: 'Variable Declaration and Primitive Types',
    difficulty: 'easy',
    tags: ['java', 'syntax', 'variables', 'types', 'primitives'],
    estimatedTime: '15m',
    isPremium: false,
    youtubeUrl: '',
    description: `Learn about Java's 8 primitive types and variable declaration.

**Requirements:**
1. Create a method \`demonstratePrimitiveTypes()\` that declares and initializes all 8 primitive types
2. Show the minimum and maximum values for numeric types using wrapper classes
3. Demonstrate type casting between compatible types
4. Print all values with descriptive labels

**Primitive Types to Use:**
- byte (8-bit)
- short (16-bit)
- int (32-bit)
- long (64-bit)
- float (32-bit)
- double (64-bit)
- boolean
- char

**Example Output:**
\`\`\`
Byte: 127 (range: -128 to 127)
Short: 32000 (range: -32768 to 32767)
Int: 2147483647 (range: -2147483648 to 2147483647)
...
\`\`\``,
    initialCode: `public class PrimitiveTypesDemo {

    public static void demonstratePrimitiveTypes() {
        // TODO: Declare and initialize all 8 primitive types



        // TODO: Show min/max values for numeric types



        // TODO: Demonstrate type casting


    }

    public static void main(String[] args) {
        demonstratePrimitiveTypes();
    }
}`,
    solutionCode: `public class PrimitiveTypesDemo {

    public static void demonstratePrimitiveTypes() {
        // Declare and initialize all 8 primitive types
        byte byteVar = 127;
        short shortVar = 32000;
        int intVar = 2147483647;
        long longVar = 9223372036854775807L;
        float floatVar = 3.14159f;
        double doubleVar = 3.141592653589793;
        boolean boolVar = true;
        char charVar = 'A';

        // Display primitive values
        System.out.println("=== Primitive Type Values ===");
        System.out.println("Byte: " + byteVar);
        System.out.println("Short: " + shortVar);
        System.out.println("Int: " + intVar);
        System.out.println("Long: " + longVar);
        System.out.println("Float: " + floatVar);
        System.out.println("Double: " + doubleVar);
        System.out.println("Boolean: " + boolVar);
        System.out.println("Char: " + charVar);

        // Show min/max values for numeric types
        System.out.println("");
        System.out.println("=== Type Ranges ===");
        System.out.println("Byte range: " + Byte.MIN_VALUE + " to " + Byte.MAX_VALUE);
        System.out.println("Short range: " + Short.MIN_VALUE + " to " + Short.MAX_VALUE);
        System.out.println("Int range: " + Integer.MIN_VALUE + " to " + Integer.MAX_VALUE);
        System.out.println("Long range: " + Long.MIN_VALUE + " to " + Long.MAX_VALUE);
        System.out.println("Float range: " + Float.MIN_VALUE + " to " + Float.MAX_VALUE);
        System.out.println("Double range: " + Double.MIN_VALUE + " to " + Double.MAX_VALUE);

        // Demonstrate type casting
        System.out.println("");
        System.out.println("=== Type Casting Examples ===");
        // Widening casting (automatic)
        int myInt = 100;
        double myDouble = myInt;
        System.out.println("Widening: int " + myInt + " -> double " + myDouble);

        // Narrowing casting (manual)
        double anotherDouble = 99.99;
        int anotherInt = (int) anotherDouble;
        System.out.println("Narrowing: double " + anotherDouble + " -> int " + anotherInt);

        // Char to int conversion
        char letter = 'Z';
        int asciiValue = letter;
        System.out.println("Char '" + letter + "' has ASCII value: " + asciiValue);
    }

    public static void main(String[] args) {
        demonstratePrimitiveTypes();
    }
}`,
    hint1: `Java has 8 primitive types: byte, short, int, long, float, double, boolean, char. Use wrapper classes like Byte.MIN_VALUE to get range information.`,
    hint2: `For type casting: widening (small to large) is automatic, narrowing (large to small) requires explicit casting with (type) syntax.`,
    whyItMatters: `Understanding primitive types is fundamental to Java programming. They are the building blocks for all data manipulation and are more memory-efficient than objects. Knowing type ranges prevents overflow errors in production applications.

**Production Pattern:**
\`\`\`java
// Correct type selection for memory optimization
public class ProductInventory {
    private int productId;      // 32-bit is sufficient for ID
    private short quantity;     // 16-bit saves memory for small values
    private long timestamp;     // 64-bit for Unix timestamp
    private double price;       // Precision for monetary operations
}
\`\`\`

**Practical Benefits:**
- Memory savings when working with large data arrays
- Prevention of overflow errors in critical calculations
- Performance optimization through proper type selection`,
    order: 0,
    testCode: `import static org.junit.Assert.*;
import org.junit.Test;
import java.io.ByteArrayOutputStream;
import java.io.PrintStream;

// Test1: Verify method runs and shows primitive types demo
class Test1 {
    @Test
    public void test() {
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        PrintStream oldOut = System.out;
        System.setOut(new PrintStream(out));
        PrimitiveTypesDemo.demonstratePrimitiveTypes();
        System.setOut(oldOut);
        String output = out.toString();
        assertTrue("Method should show primitive types demo",
            output.contains("Primitive") || output.contains("Types") ||
            output.contains("Примитивные") || output.contains("Primitiv"));
    }
}

// Test2: Output should contain byte value
class Test2 {
    @Test
    public void test() {
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        PrintStream oldOut = System.out;
        System.setOut(new PrintStream(out));
        PrimitiveTypesDemo.demonstratePrimitiveTypes();
        System.setOut(oldOut);
        String output = out.toString();
        assertTrue("Output should contain 'Byte'", output.contains("Byte"));
    }
}

// Test3: Output should contain short value
class Test3 {
    @Test
    public void test() {
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        PrintStream oldOut = System.out;
        System.setOut(new PrintStream(out));
        PrimitiveTypesDemo.demonstratePrimitiveTypes();
        System.setOut(oldOut);
        String output = out.toString();
        assertTrue("Output should contain 'Short'", output.contains("Short"));
    }
}

// Test4: Output should contain int value
class Test4 {
    @Test
    public void test() {
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        PrintStream oldOut = System.out;
        System.setOut(new PrintStream(out));
        PrimitiveTypesDemo.demonstratePrimitiveTypes();
        System.setOut(oldOut);
        String output = out.toString();
        assertTrue("Output should contain 'Int'", output.contains("Int"));
    }
}

// Test5: Output should contain long value
class Test5 {
    @Test
    public void test() {
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        PrintStream oldOut = System.out;
        System.setOut(new PrintStream(out));
        PrimitiveTypesDemo.demonstratePrimitiveTypes();
        System.setOut(oldOut);
        String output = out.toString();
        assertTrue("Output should contain 'Long'", output.contains("Long"));
    }
}

// Test6: Output should show byte range
class Test6 {
    @Test
    public void test() {
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        PrintStream oldOut = System.out;
        System.setOut(new PrintStream(out));
        PrimitiveTypesDemo.demonstratePrimitiveTypes();
        System.setOut(oldOut);
        String output = out.toString();
        assertTrue("Output should contain byte range '-128'", output.contains("-128"));
        assertTrue("Output should contain byte range '127'", output.contains("127"));
    }
}

// Test7: Output should show widening casting example
class Test7 {
    @Test
    public void test() {
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        PrintStream oldOut = System.out;
        System.setOut(new PrintStream(out));
        PrimitiveTypesDemo.demonstratePrimitiveTypes();
        System.setOut(oldOut);
        String output = out.toString().toLowerCase();
        assertTrue("Output should demonstrate widening casting",
            output.contains("widening") || output.contains("расширен") || output.contains("kengaytirish"));
    }
}

// Test8: Output should show narrowing casting example
class Test8 {
    @Test
    public void test() {
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        PrintStream oldOut = System.out;
        System.setOut(new PrintStream(out));
        PrimitiveTypesDemo.demonstratePrimitiveTypes();
        System.setOut(oldOut);
        String output = out.toString().toLowerCase();
        assertTrue("Output should demonstrate narrowing casting",
            output.contains("narrowing") || output.contains("сужен") || output.contains("toraytirish"));
    }
}

// Test9: Output should contain boolean value
class Test9 {
    @Test
    public void test() {
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        PrintStream oldOut = System.out;
        System.setOut(new PrintStream(out));
        PrimitiveTypesDemo.demonstratePrimitiveTypes();
        System.setOut(oldOut);
        String output = out.toString();
        assertTrue("Output should contain 'Boolean'", output.contains("Boolean"));
    }
}

// Test10: Output should contain char value
class Test10 {
    @Test
    public void test() {
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        PrintStream oldOut = System.out;
        System.setOut(new PrintStream(out));
        PrimitiveTypesDemo.demonstratePrimitiveTypes();
        System.setOut(oldOut);
        String output = out.toString();
        assertTrue("Output should contain 'Char'", output.contains("Char"));
    }
}
`,
    translations: {
        ru: {
            title: 'Объявление переменных и примитивные типы',
            solutionCode: `public class PrimitiveTypesDemo {

    public static void demonstratePrimitiveTypes() {
        // Объявляем и инициализируем все 8 примитивных типов
        byte byteVar = 127;
        short shortVar = 32000;
        int intVar = 2147483647;
        long longVar = 9223372036854775807L;
        float floatVar = 3.14159f;
        double doubleVar = 3.141592653589793;
        boolean boolVar = true;
        char charVar = 'A';

        // Выводим значения примитивных типов
        System.out.println("=== Значения примитивных типов ===");
        System.out.println("Byte: " + byteVar);
        System.out.println("Short: " + shortVar);
        System.out.println("Int: " + intVar);
        System.out.println("Long: " + longVar);
        System.out.println("Float: " + floatVar);
        System.out.println("Double: " + doubleVar);
        System.out.println("Boolean: " + boolVar);
        System.out.println("Char: " + charVar);

        // Показываем диапазоны значений для числовых типов
        System.out.println("");
        System.out.println("=== Диапазоны типов ===");
        System.out.println("Диапазон Byte: " + Byte.MIN_VALUE + " до " + Byte.MAX_VALUE);
        System.out.println("Диапазон Short: " + Short.MIN_VALUE + " до " + Short.MAX_VALUE);
        System.out.println("Диапазон Int: " + Integer.MIN_VALUE + " до " + Integer.MAX_VALUE);
        System.out.println("Диапазон Long: " + Long.MIN_VALUE + " до " + Long.MAX_VALUE);
        System.out.println("Диапазон Float: " + Float.MIN_VALUE + " до " + Float.MAX_VALUE);
        System.out.println("Диапазон Double: " + Double.MIN_VALUE + " до " + Double.MAX_VALUE);

        // Демонстрируем приведение типов
        System.out.println("");
        System.out.println("=== Примеры приведения типов ===");
        // Расширяющее приведение (автоматическое)
        int myInt = 100;
        double myDouble = myInt;
        System.out.println("Расширение: int " + myInt + " -> double " + myDouble);

        // Сужающее приведение (явное)
        double anotherDouble = 99.99;
        int anotherInt = (int) anotherDouble;
        System.out.println("Сужение: double " + anotherDouble + " -> int " + anotherInt);

        // Преобразование char в int
        char letter = 'Z';
        int asciiValue = letter;
        System.out.println("Символ '" + letter + "' имеет ASCII значение: " + asciiValue);
    }

    public static void main(String[] args) {
        demonstratePrimitiveTypes();
    }
}`,
            description: `Изучите 8 примитивных типов Java и объявление переменных.

**Требования:**
1. Создайте метод \`demonstratePrimitiveTypes()\`, который объявляет и инициализирует все 8 примитивных типов
2. Покажите минимальные и максимальные значения для числовых типов, используя классы-обертки
3. Продемонстрируйте приведение типов между совместимыми типами
4. Выведите все значения с описательными метками

**Примитивные типы для использования:**
- byte (8-бит)
- short (16-бит)
- int (32-бит)
- long (64-бит)
- float (32-бит)
- double (64-бит)
- boolean
- char

**Пример вывода:**
\`\`\`
Byte: 127 (диапазон: -128 до 127)
Short: 32000 (диапазон: -32768 до 32767)
Int: 2147483647 (диапазон: -2147483648 до 2147483647)
...\`\`\``,
            hint1: `В Java есть 8 примитивных типов: byte, short, int, long, float, double, boolean, char. Используйте классы-обертки, такие как Byte.MIN_VALUE, для получения информации о диапазоне.`,
            hint2: `Для приведения типов: расширение (от меньшего к большему) происходит автоматически, сужение (от большего к меньшему) требует явного приведения с синтаксисом (тип).`,
            whyItMatters: `Понимание примитивных типов является фундаментальным для программирования на Java. Они являются строительными блоками для всех манипуляций с данными и более эффективны по памяти, чем объекты. Знание диапазонов типов предотвращает ошибки переполнения в продакшн-приложениях.

**Продакшен паттерн:**
\`\`\`java
// Правильный выбор типа для оптимизации памяти
public class ProductInventory {
    private int productId;	// 32-bit достаточно для ID
    private short quantity;	// 16-bit экономит память для малых значений
    private long timestamp;	// 64-bit для Unix timestamp
    private double price;	// Точность для денежных операций
}
\`\`\`

**Практические преимущества:**
- Экономия памяти при работе с большими массивами данных
- Предотвращение overflow ошибок в критических вычислениях
- Оптимизация производительности через правильный выбор типа`
        },
        uz: {
            title: `O'zgaruvchilarni e'lon qilish va primitiv turlar`,
            solutionCode: `public class PrimitiveTypesDemo {

    public static void demonstratePrimitiveTypes() {
        // Barcha 8 primitiv turlarni e'lon qilamiz va initsializatsiya qilamiz
        byte byteVar = 127;
        short shortVar = 32000;
        int intVar = 2147483647;
        long longVar = 9223372036854775807L;
        float floatVar = 3.14159f;
        double doubleVar = 3.141592653589793;
        boolean boolVar = true;
        char charVar = 'A';

        // Primitiv turlarning qiymatlarini chiqaramiz
        System.out.println("=== Primitiv turlarning qiymatlari ===");
        System.out.println("Byte: " + byteVar);
        System.out.println("Short: " + shortVar);
        System.out.println("Int: " + intVar);
        System.out.println("Long: " + longVar);
        System.out.println("Float: " + floatVar);
        System.out.println("Double: " + doubleVar);
        System.out.println("Boolean: " + boolVar);
        System.out.println("Char: " + charVar);

        // Raqamli turlar uchun min/max qiymatlarni ko'rsatamiz
        System.out.println("");
        System.out.println("=== Turlarning diapazoni ===");
        System.out.println("Byte diapazoni: " + Byte.MIN_VALUE + " dan " + Byte.MAX_VALUE + " gacha");
        System.out.println("Short diapazoni: " + Short.MIN_VALUE + " dan " + Short.MAX_VALUE + " gacha");
        System.out.println("Int diapazoni: " + Integer.MIN_VALUE + " dan " + Integer.MAX_VALUE + " gacha");
        System.out.println("Long diapazoni: " + Long.MIN_VALUE + " dan " + Long.MAX_VALUE + " gacha");
        System.out.println("Float diapazoni: " + Float.MIN_VALUE + " dan " + Float.MAX_VALUE + " gacha");
        System.out.println("Double diapazoni: " + Double.MIN_VALUE + " dan " + Double.MAX_VALUE + " gacha");

        // Turlarni o'zgartirish namunalarini ko'rsatamiz
        System.out.println("");
        System.out.println("=== Turlarni o'zgartirish misollari ===");
        // Kengaytirish (avtomatik)
        int myInt = 100;
        double myDouble = myInt;
        System.out.println("Kengaytirish: int " + myInt + " -> double " + myDouble);

        // Toraytirish (qo'lda)
        double anotherDouble = 99.99;
        int anotherInt = (int) anotherDouble;
        System.out.println("Toraytirish: double " + anotherDouble + " -> int " + anotherInt);

        // Char dan int ga o'tkazish
        char letter = 'Z';
        int asciiValue = letter;
        System.out.println("'" + letter + "' belgisi ASCII qiymati: " + asciiValue);
    }

    public static void main(String[] args) {
        demonstratePrimitiveTypes();
    }
}`,
            description: `Java ning 8 primitiv turi va o'zgaruvchilarni e'lon qilishni o'rganing.

**Talablar:**
1. Barcha 8 primitiv turlarni e'lon qiladigan va initsializatsiya qiladigan \`demonstratePrimitiveTypes()\` metodini yarating
2. Wrapper klasslardan foydalanib, raqamli turlar uchun minimal va maksimal qiymatlarni ko'rsating
3. Mos turlar o'rtasida turlarni o'zgartirish namunasini ko'rsating
4. Barcha qiymatlarni tavsiflovchi yorliqlar bilan chiqaring

**Ishlatish uchun primitiv turlar:**
- byte (8-bit)
- short (16-bit)
- int (32-bit)
- long (64-bit)
- float (32-bit)
- double (64-bit)
- boolean
- char

**Chiqish namunasi:**
\`\`\`
Byte: 127 (diapazon: -128 dan 127 gacha)
Short: 32000 (diapazon: -32768 dan 32767 gacha)
Int: 2147483647 (diapazon: -2147483648 dan 2147483647 gacha)
...\`\`\``,
            hint1: `Java da 8 ta primitiv tur bor: byte, short, int, long, float, double, boolean, char. Diapazon ma'lumotini olish uchun Byte.MIN_VALUE kabi wrapper klasslardan foydalaning.`,
            hint2: `Turlarni o'zgartirish uchun: kengaytirish (kichikdan kattaga) avtomatik, toraytirish (kattadan kichikka) (tur) sintaksisi bilan aniq o'zgartirishni talab qiladi.`,
            whyItMatters: `Primitiv turlarni tushunish Java dasturlash uchun asosiy hisoblanadi. Ular barcha ma'lumotlarni boshqarish uchun qurilish bloklari bo'lib, obyektlarga qaraganda xotira jihatidan samaraliroqdir. Turlar diapazonini bilish ishlab chiqarish ilovalarida overflow xatolarini oldini oladi.

**Ishlab chiqarish patterni:**
\`\`\`java
// Xotirani optimallashtirish uchun to'g'ri turni tanlash
public class ProductInventory {
    private int productId;	// ID uchun 32-bit yetarli
    private short quantity;	// Kichik qiymatlar uchun 16-bit xotirani tejaydi
    private long timestamp;	// Unix timestamp uchun 64-bit
    private double price;	// Pul operatsiyalari uchun aniqlik
}
\`\`\`

**Amaliy foydalari:**
- Katta ma'lumotlar massivlari bilan ishlashda xotirani tejash
- Muhim hisob-kitoblarda overflow xatolarini oldini olish
- To'g'ri tur tanlab ishlashni optimallashtirish`
        }
    }
};

export default task;
