import { Task } from '../../../../types';

export const task: Task = {
    slug: 'java-arithmetic-logical-operators',
    title: 'Arithmetic and Logical Operators',
    difficulty: 'easy',
    tags: ['java', 'syntax', 'operators', 'arithmetic', 'logical'],
    estimatedTime: '15m',
    isPremium: false,
    youtubeUrl: '',
    description: `Master Java operators: arithmetic, logical, and bitwise operations.

**Requirements:**
1. Create a \`calculator()\` method that demonstrates arithmetic operators (+, -, *, /, %)
2. Implement a \`logicalOperations()\` method showing logical operators (&&, ||, !)
3. Create a \`bitwiseOperations()\` method demonstrating bitwise operators (&, |, ^, ~, <<, >>)
4. Show operator precedence with a complex expression

**Operators to Cover:**
- Arithmetic: +, -, *, /, %
- Logical: &&, ||, !
- Bitwise: &, |, ^, ~, <<, >>
- Comparison: ==, !=, <, >, <=, >=
- Assignment: =, +=, -=, *=, /=

**Example Output:**
\`\`\`
=== Arithmetic ===
10 + 5 = 15
10 - 5 = 5
10 * 5 = 50
...
\`\`\``,
    initialCode: `public class OperatorsDemo {

    public static void calculator(int a, int b) {
        // TODO: Demonstrate arithmetic operators


    }

    public static void logicalOperations(boolean x, boolean y) {
        // TODO: Demonstrate logical operators


    }

    public static void bitwiseOperations(int a, int b) {
        // TODO: Demonstrate bitwise operators


    }

    public static void main(String[] args) {
        calculator(10, 5);
        logicalOperations(true, false);
        bitwiseOperations(12, 5);
    }
}`,
    solutionCode: `public class OperatorsDemo {

    public static void calculator(int a, int b) {
        System.out.println("=== Arithmetic Operators ===");
        System.out.println(a + " + " + b + " = " + (a + b));
        System.out.println(a + " - " + b + " = " + (a - b));
        System.out.println(a + " * " + b + " = " + (a * b));
        System.out.println(a + " / " + b + " = " + (a / b));
        System.out.println(a + " % " + b + " = " + (a % b));

        // Increment and decrement operators
        int counter = 10;
        System.out.println("\nIncrement/Decrement:");
        System.out.println("counter = " + counter);
        System.out.println("counter++ = " + (counter++)); // Post-increment
        System.out.println("After post-increment: " + counter);
        System.out.println("++counter = " + (++counter)); // Pre-increment
        System.out.println("counter-- = " + (counter--)); // Post-decrement
        System.out.println("--counter = " + (--counter)); // Pre-decrement

        // Compound assignment operators
        int value = 100;
        System.out.println("\nCompound Assignment:");
        System.out.println("value = " + value);
        value += 20; // value = value + 20
        System.out.println("After value += 20: " + value);
        value *= 2; // value = value * 2
        System.out.println("After value *= 2: " + value);
    }

    public static void logicalOperations(boolean x, boolean y) {
        System.out.println("\n=== Logical Operators ===");
        System.out.println("x = " + x + ", y = " + y);
        System.out.println("x && y (AND) = " + (x && y));
        System.out.println("x || y (OR) = " + (x || y));
        System.out.println("!x (NOT) = " + (!x));
        System.out.println("!y (NOT) = " + (!y));

        // Short-circuit evaluation
        System.out.println("\nShort-circuit evaluation:");
        int num = 10;
        boolean result = (num > 5) || (++num > 15);
        System.out.println("(num > 5) || (++num > 15) = " + result);
        System.out.println("num after short-circuit: " + num); // num stays 10

        // Comparison operators
        System.out.println("\nComparison Operators:");
        int a = 10, b = 20;
        System.out.println(a + " == " + b + " : " + (a == b));
        System.out.println(a + " != " + b + " : " + (a != b));
        System.out.println(a + " < " + b + " : " + (a < b));
        System.out.println(a + " > " + b + " : " + (a > b));
        System.out.println(a + " <= " + b + " : " + (a <= b));
        System.out.println(a + " >= " + b + " : " + (a >= b));
    }

    public static void bitwiseOperations(int a, int b) {
        System.out.println("\n=== Bitwise Operators ===");
        System.out.println("a = " + a + " (binary: " + Integer.toBinaryString(a) + ")");
        System.out.println("b = " + b + " (binary: " + Integer.toBinaryString(b) + ")");

        System.out.println("\nBitwise AND (&): " + (a & b) +
                          " (binary: " + Integer.toBinaryString(a & b) + ")");
        System.out.println("Bitwise OR (|): " + (a | b) +
                          " (binary: " + Integer.toBinaryString(a | b) + ")");
        System.out.println("Bitwise XOR (^): " + (a ^ b) +
                          " (binary: " + Integer.toBinaryString(a ^ b) + ")");
        System.out.println("Bitwise NOT (~a): " + (~a) +
                          " (binary: " + Integer.toBinaryString(~a) + ")");

        // Bit shift operators
        System.out.println("\nBit Shift Operators:");
        System.out.println("a << 2 (left shift): " + (a << 2) +
                          " (binary: " + Integer.toBinaryString(a << 2) + ")");
        System.out.println("a >> 2 (right shift): " + (a >> 2) +
                          " (binary: " + Integer.toBinaryString(a >> 2) + ")");
        System.out.println("a >>> 2 (unsigned right shift): " + (a >>> 2) +
                          " (binary: " + Integer.toBinaryString(a >>> 2) + ")");

        // Operator precedence example
        System.out.println("\n=== Operator Precedence ===");
        int result = 10 + 5 * 2 - 8 / 4;
        System.out.println("10 + 5 * 2 - 8 / 4 = " + result);
        System.out.println("Evaluation: 10 + (5 * 2) - (8 / 4) = 10 + 10 - 2 = " + result);
    }

    public static void main(String[] args) {
        calculator(10, 5);
        logicalOperations(true, false);
        bitwiseOperations(12, 5);
    }
}`,
    hint1: `Arithmetic operators follow standard mathematical precedence: * and / before + and -. Use parentheses to override precedence.`,
    hint2: `Logical operators use short-circuit evaluation: && stops if first operand is false, || stops if first operand is true. Bitwise operators work on individual bits.`,
    whyItMatters: `Operators are the foundation of all computations in Java. Understanding operator precedence prevents bugs, bitwise operations are essential for performance optimization and low-level programming, and logical operators are crucial for control flow.

**Production Pattern:**
\`\`\`java
// Using bitwise operations for permission flags
public class PermissionManager {
    private static final int READ = 1 << 0;    // 0001
    private static final int WRITE = 1 << 1;   // 0010
    private static final int EXECUTE = 1 << 2; // 0100

    public boolean hasPermission(int userPerms, int required) {
        return (userPerms & required) == required;
    }
}
\`\`\`

**Practical Benefits:**
- Efficient storage of multiple flags in a single number
- Fast permission checking through bitmasks
- Memory optimization for systems with multiple states`,
    order: 1,
    testCode: `import static org.junit.Assert.*;
import org.junit.Test;

// Test1: Test basic arithmetic operations
class Test1 {
    @Test
    public void test() {
        assertEquals(15, 10 + 5);
        assertEquals(5, 10 - 5);
        assertEquals(50, 10 * 5);
        assertEquals(2, 10 / 5);
        assertEquals(0, 10 % 5);
    }
}

// Test2: Test increment operators
class Test2 {
    @Test
    public void test() {
        int counter = 10;
        assertEquals(10, counter++);
        assertEquals(11, counter);
        assertEquals(12, ++counter);
    }
}

// Test3: Test decrement operators
class Test3 {
    @Test
    public void test() {
        int counter = 10;
        assertEquals(10, counter--);
        assertEquals(9, counter);
        assertEquals(8, --counter);
    }
}

// Test4: Test logical AND and OR
class Test4 {
    @Test
    public void test() {
        assertTrue(true && true);
        assertFalse(true && false);
        assertTrue(true || false);
        assertFalse(false || false);
    }
}

// Test5: Test logical NOT
class Test5 {
    @Test
    public void test() {
        assertFalse(!true);
        assertTrue(!false);
    }
}

// Test6: Test comparison operators
class Test6 {
    @Test
    public void test() {
        assertTrue(10 < 20);
        assertTrue(20 > 10);
        assertTrue(10 <= 10);
        assertTrue(10 >= 10);
        assertTrue(10 == 10);
        assertTrue(10 != 20);
    }
}

// Test7: Test bitwise AND
class Test7 {
    @Test
    public void test() {
        int result = 12 & 5; // 1100 & 0101 = 0100
        assertEquals(4, result);
    }
}

// Test8: Test bitwise OR
class Test8 {
    @Test
    public void test() {
        int result = 12 | 5; // 1100 | 0101 = 1101
        assertEquals(13, result);
    }
}

// Test9: Test bitwise XOR
class Test9 {
    @Test
    public void test() {
        int result = 12 ^ 5; // 1100 ^ 0101 = 1001
        assertEquals(9, result);
    }
}

// Test10: Test bit shift operators
class Test10 {
    @Test
    public void test() {
        assertEquals(48, 12 << 2); // Left shift by 2
        assertEquals(3, 12 >> 2);  // Right shift by 2
        assertEquals(3, 12 >>> 2); // Unsigned right shift by 2
    }
}
`,
    translations: {
        ru: {
            title: 'Арифметические и логические операторы',
            solutionCode: `public class OperatorsDemo {

    public static void calculator(int a, int b) {
        System.out.println("=== Арифметические операторы ===");
        System.out.println(a + " + " + b + " = " + (a + b));
        System.out.println(a + " - " + b + " = " + (a - b));
        System.out.println(a + " * " + b + " = " + (a * b));
        System.out.println(a + " / " + b + " = " + (a / b));
        System.out.println(a + " % " + b + " = " + (a % b));

        // Операторы инкремента и декремента
        int counter = 10;
        System.out.println("\nИнкремент/Декремент:");
        System.out.println("counter = " + counter);
        System.out.println("counter++ = " + (counter++)); // Постфиксный инкремент
        System.out.println("После постфиксного инкремента: " + counter);
        System.out.println("++counter = " + (++counter)); // Префиксный инкремент
        System.out.println("counter-- = " + (counter--)); // Постфиксный декремент
        System.out.println("--counter = " + (--counter)); // Префиксный декремент

        // Составные операторы присваивания
        int value = 100;
        System.out.println("\nСоставное присваивание:");
        System.out.println("value = " + value);
        value += 20; // value = value + 20
        System.out.println("После value += 20: " + value);
        value *= 2; // value = value * 2
        System.out.println("После value *= 2: " + value);
    }

    public static void logicalOperations(boolean x, boolean y) {
        System.out.println("\n=== Логические операторы ===");
        System.out.println("x = " + x + ", y = " + y);
        System.out.println("x && y (И) = " + (x && y));
        System.out.println("x || y (ИЛИ) = " + (x || y));
        System.out.println("!x (НЕ) = " + (!x));
        System.out.println("!y (НЕ) = " + (!y));

        // Ленивые вычисления
        System.out.println("\nЛенивые вычисления:");
        int num = 10;
        boolean result = (num > 5) || (++num > 15);
        System.out.println("(num > 5) || (++num > 15) = " + result);
        System.out.println("num после ленивого вычисления: " + num); // num остается 10

        // Операторы сравнения
        System.out.println("\nОператоры сравнения:");
        int a = 10, b = 20;
        System.out.println(a + " == " + b + " : " + (a == b));
        System.out.println(a + " != " + b + " : " + (a != b));
        System.out.println(a + " < " + b + " : " + (a < b));
        System.out.println(a + " > " + b + " : " + (a > b));
        System.out.println(a + " <= " + b + " : " + (a <= b));
        System.out.println(a + " >= " + b + " : " + (a >= b));
    }

    public static void bitwiseOperations(int a, int b) {
        System.out.println("\n=== Побитовые операторы ===");
        System.out.println("a = " + a + " (двоичное: " + Integer.toBinaryString(a) + ")");
        System.out.println("b = " + b + " (двоичное: " + Integer.toBinaryString(b) + ")");

        System.out.println("\nПобитовое И (&): " + (a & b) +
                          " (двоичное: " + Integer.toBinaryString(a & b) + ")");
        System.out.println("Побитовое ИЛИ (|): " + (a | b) +
                          " (двоичное: " + Integer.toBinaryString(a | b) + ")");
        System.out.println("Побитовое исключающее ИЛИ (^): " + (a ^ b) +
                          " (двоичное: " + Integer.toBinaryString(a ^ b) + ")");
        System.out.println("Побитовое НЕ (~a): " + (~a) +
                          " (двоичное: " + Integer.toBinaryString(~a) + ")");

        // Операторы битового сдвига
        System.out.println("\nОператоры битового сдвига:");
        System.out.println("a << 2 (сдвиг влево): " + (a << 2) +
                          " (двоичное: " + Integer.toBinaryString(a << 2) + ")");
        System.out.println("a >> 2 (сдвиг вправо): " + (a >> 2) +
                          " (двоичное: " + Integer.toBinaryString(a >> 2) + ")");
        System.out.println("a >>> 2 (беззнаковый сдвиг вправо): " + (a >>> 2) +
                          " (двоичное: " + Integer.toBinaryString(a >>> 2) + ")");

        // Пример приоритета операторов
        System.out.println("\n=== Приоритет операторов ===");
        int result = 10 + 5 * 2 - 8 / 4;
        System.out.println("10 + 5 * 2 - 8 / 4 = " + result);
        System.out.println("Вычисление: 10 + (5 * 2) - (8 / 4) = 10 + 10 - 2 = " + result);
    }

    public static void main(String[] args) {
        calculator(10, 5);
        logicalOperations(true, false);
        bitwiseOperations(12, 5);
    }
}`,
            description: `Освойте операторы Java: арифметические, логические и побитовые операции.

**Требования:**
1. Создайте метод \`calculator()\`, демонстрирующий арифметические операторы (+, -, *, /, %)
2. Реализуйте метод \`logicalOperations()\`, показывающий логические операторы (&&, ||, !)
3. Создайте метод \`bitwiseOperations()\`, демонстрирующий побитовые операторы (&, |, ^, ~, <<, >>)
4. Покажите приоритет операторов со сложным выражением

**Операторы для изучения:**
- Арифметические: +, -, *, /, %
- Логические: &&, ||, !
- Побитовые: &, |, ^, ~, <<, >>
- Сравнения: ==, !=, <, >, <=, >=
- Присваивания: =, +=, -=, *=, /=

**Пример вывода:**
\`\`\`
=== Арифметические ===
10 + 5 = 15
10 - 5 = 5
10 * 5 = 50
...\`\`\``,
            hint1: `Арифметические операторы следуют стандартному математическому приоритету: * и / перед + и -. Используйте скобки для изменения приоритета.`,
            hint2: `Логические операторы используют ленивые вычисления: && останавливается, если первый операнд false, || останавливается, если первый операнд true. Побитовые операторы работают с отдельными битами.`,
            whyItMatters: `Операторы являются основой всех вычислений в Java. Понимание приоритета операторов предотвращает ошибки, побитовые операции необходимы для оптимизации производительности и низкоуровневого программирования, а логические операторы критически важны для управления потоком выполнения.

**Продакшен паттерн:**
\`\`\`java
// Использование побитовых операций для флагов разрешений
public class PermissionManager {
    private static final int READ = 1 << 0;	// 0001
    private static final int WRITE = 1 << 1;	// 0010
    private static final int EXECUTE = 1 << 2;	// 0100

    public boolean hasPermission(int userPerms, int required) {
        return (userPerms & required) == required;
    }
}
\`\`\`

**Практические преимущества:**
- Эффективное хранение множества флагов в одном числе
- Быстрая проверка разрешений через битовые маски
- Оптимизация памяти для систем с множеством состояний`
        },
        uz: {
            title: `Arifmetik va mantiqiy operatorlar`,
            solutionCode: `public class OperatorsDemo {

    public static void calculator(int a, int b) {
        System.out.println("=== Arifmetik operatorlar ===");
        System.out.println(a + " + " + b + " = " + (a + b));
        System.out.println(a + " - " + b + " = " + (a - b));
        System.out.println(a + " * " + b + " = " + (a * b));
        System.out.println(a + " / " + b + " = " + (a / b));
        System.out.println(a + " % " + b + " = " + (a % b));

        // Inkrement va dekrement operatorlari
        int counter = 10;
        System.out.println("\nInkrement/Dekrement:");
        System.out.println("counter = " + counter);
        System.out.println("counter++ = " + (counter++)); // Post-inkrement
        System.out.println("Post-inkrement dan keyin: " + counter);
        System.out.println("++counter = " + (++counter)); // Pre-inkrement
        System.out.println("counter-- = " + (counter--)); // Post-dekrement
        System.out.println("--counter = " + (--counter)); // Pre-dekrement

        // Murakkab tayinlash operatorlari
        int value = 100;
        System.out.println("\nMurakkab tayinlash:");
        System.out.println("value = " + value);
        value += 20; // value = value + 20
        System.out.println("value += 20 dan keyin: " + value);
        value *= 2; // value = value * 2
        System.out.println("value *= 2 dan keyin: " + value);
    }

    public static void logicalOperations(boolean x, boolean y) {
        System.out.println("\n=== Mantiqiy operatorlar ===");
        System.out.println("x = " + x + ", y = " + y);
        System.out.println("x && y (VA) = " + (x && y));
        System.out.println("x || y (YOKI) = " + (x || y));
        System.out.println("!x (INKOR) = " + (!x));
        System.out.println("!y (INKOR) = " + (!y));

        // Qisqa tutashuv baholash
        System.out.println("\nQisqa tutashuv baholash:");
        int num = 10;
        boolean result = (num > 5) || (++num > 15);
        System.out.println("(num > 5) || (++num > 15) = " + result);
        System.out.println("Qisqa tutashuvdan keyin num: " + num); // num 10 bo'lib qoladi

        // Taqqoslash operatorlari
        System.out.println("\nTaqqoslash operatorlari:");
        int a = 10, b = 20;
        System.out.println(a + " == " + b + " : " + (a == b));
        System.out.println(a + " != " + b + " : " + (a != b));
        System.out.println(a + " < " + b + " : " + (a < b));
        System.out.println(a + " > " + b + " : " + (a > b));
        System.out.println(a + " <= " + b + " : " + (a <= b));
        System.out.println(a + " >= " + b + " : " + (a >= b));
    }

    public static void bitwiseOperations(int a, int b) {
        System.out.println("\n=== Bitli operatorlar ===");
        System.out.println("a = " + a + " (ikkilik: " + Integer.toBinaryString(a) + ")");
        System.out.println("b = " + b + " (ikkilik: " + Integer.toBinaryString(b) + ")");

        System.out.println("\nBitli VA (&): " + (a & b) +
                          " (ikkilik: " + Integer.toBinaryString(a & b) + ")");
        System.out.println("Bitli YOKI (|): " + (a | b) +
                          " (ikkilik: " + Integer.toBinaryString(a | b) + ")");
        System.out.println("Bitli XOR (^): " + (a ^ b) +
                          " (ikkilik: " + Integer.toBinaryString(a ^ b) + ")");
        System.out.println("Bitli INKOR (~a): " + (~a) +
                          " (ikkilik: " + Integer.toBinaryString(~a) + ")");

        // Bit siljitish operatorlari
        System.out.println("\nBit siljitish operatorlari:");
        System.out.println("a << 2 (chapga siljitish): " + (a << 2) +
                          " (ikkilik: " + Integer.toBinaryString(a << 2) + ")");
        System.out.println("a >> 2 (o'ngga siljitish): " + (a >> 2) +
                          " (ikkilik: " + Integer.toBinaryString(a >> 2) + ")");
        System.out.println("a >>> 2 (belgisiz o'ngga siljitish): " + (a >>> 2) +
                          " (ikkilik: " + Integer.toBinaryString(a >>> 2) + ")");

        // Operatorlar ustunligi misoli
        System.out.println("\n=== Operatorlar ustunligi ===");
        int result = 10 + 5 * 2 - 8 / 4;
        System.out.println("10 + 5 * 2 - 8 / 4 = " + result);
        System.out.println("Baholash: 10 + (5 * 2) - (8 / 4) = 10 + 10 - 2 = " + result);
    }

    public static void main(String[] args) {
        calculator(10, 5);
        logicalOperations(true, false);
        bitwiseOperations(12, 5);
    }
}`,
            description: `Java operatorlarini o'zlashtiiring: arifmetik, mantiqiy va bitli operatsiyalar.

**Talablar:**
1. Arifmetik operatorlarni (+, -, *, /, %) ko'rsatadigan \`calculator()\` metodini yarating
2. Mantiqiy operatorlarni (&&, ||, !) ko'rsatadigan \`logicalOperations()\` metodini yarating
3. Bitli operatorlarni (&, |, ^, ~, <<, >>) ko'rsatadigan \`bitwiseOperations()\` metodini yarating
4. Murakkab ifoda bilan operatorlar ustunligini ko'rsating

**O'rganish uchun operatorlar:**
- Arifmetik: +, -, *, /, %
- Mantiqiy: &&, ||, !
- Bitli: &, |, ^, ~, <<, >>
- Taqqoslash: ==, !=, <, >, <=, >=
- Tayinlash: =, +=, -=, *=, /=

**Chiqish namunasi:**
\`\`\`
=== Arifmetik ===
10 + 5 = 15
10 - 5 = 5
10 * 5 = 50
...\`\`\``,
            hint1: `Arifmetik operatorlar standart matematik ustunlikka amal qiladi: * va / dan oldin + va -. Ustunlikni o'zgartirish uchun qavslardan foydalaning.`,
            hint2: `Mantiqiy operatorlar qisqa tutashuv baholashdan foydalanadi: && birinchi operand false bo'lsa to'xtaydi, || birinchi operand true bo'lsa to'xtaydi. Bitli operatorlar alohida bitlar bilan ishlaydi.`,
            whyItMatters: `Operatorlar Java dagi barcha hisob-kitoblarning asosi hisoblanadi. Operatorlar ustunligini tushunish xatolarni oldini oladi, bitli operatsiyalar ishlash optimallashtirish va past darajali dasturlash uchun zarur, mantiqiy operatorlar esa boshqaruv oqimi uchun juda muhimdir.

**Ishlab chiqarish patterni:**
\`\`\`java
// Ruxsatlar bayroqlari uchun bitli operatsiyalardan foydalanish
public class PermissionManager {
    private static final int READ = 1 << 0;	// 0001
    private static final int WRITE = 1 << 1;	// 0010
    private static final int EXECUTE = 1 << 2;	// 0100

    public boolean hasPermission(int userPerms, int required) {
        return (userPerms & required) == required;
    }
}
\`\`\`

**Amaliy foydalari:**
- Ko'p bayroqlarni bitta sonda samarali saqlash
- Bit maskalari orqali tez ruxsat tekshiruvi
- Ko'p holatli tizimlar uchun xotirani optimallashtirish`
        }
    }
};

export default task;
