import { Task } from '../../../../types';

export const task: Task = {
    slug: 'java-functional-interfaces',
    title: 'Functional Interfaces and @FunctionalInterface',
    difficulty: 'medium',
    tags: ['java', 'interfaces', 'functional-programming', 'java8', 'lambda'],
    estimatedTime: '25m',
    isPremium: false,
    youtubeUrl: '',
    description: `# Functional Interfaces and @FunctionalInterface

A functional interface is an interface with exactly one abstract method (SAM - Single Abstract Method). They can be instantiated using lambda expressions. Java provides built-in functional interfaces in java.util.function package.

## Requirements:
1. Create a custom \`Calculator\` functional interface:
   1.1. Annotated with \`@FunctionalInterface\`
   1.2. Single abstract method: \`int operate(int a, int b)\`
   1.3. Default method: \`String describe()\` that returns "Calculator operation"

2. Demonstrate using built-in functional interfaces:
   2.1. \`Function<T,R>\`: Transform string to uppercase
   2.2. \`Predicate<T>\`: Check if number is even
   2.3. \`Consumer<T>\`: Print formatted message
   2.4. \`Supplier<T>\`: Generate random number

3. Use custom Calculator with:
   3.1. Addition operation
   3.2. Multiplication operation
   3.3. Custom operation (power)

4. Show both anonymous class and lambda expressions (preview for next modules)

## Example Output:
\`\`\`
=== Custom Functional Interface ===
Addition: 5 + 3 = 8
Multiplication: 5 * 3 = 15
Power: 2 ^ 3 = 8

=== Built-in Functional Interfaces ===
Function - Uppercase: HELLO
Predicate - Is 4 even? true
Consumer - Message: Processing: Data
Supplier - Random number: 42
\`\`\``,
    initialCode: `import java.util.function.*;

// TODO: Create Calculator functional interface

public class FunctionalInterfaces {
    public static void main(String[] args) {
        // TODO: Test custom functional interface with different operations

        // TODO: Demonstrate built-in functional interfaces
    }
}`,
    solutionCode: `import java.util.function.*;
import java.util.Random;

// Functional interface with @FunctionalInterface annotation
@FunctionalInterface
interface Calculator {
    // Single Abstract Method (SAM)
    int operate(int a, int b);

    // Default methods are allowed
    default String describe() {
        return "Calculator operation";
    }
}

public class FunctionalInterfaces {
    public static void main(String[] args) {
        System.out.println("=== Custom Functional Interface ===");

        // Using anonymous class (traditional way)
        Calculator addition = new Calculator() {
            @Override
            public int operate(int a, int b) {
                return a + b;
            }
        };
        System.out.println("Addition: 5 + 3 = " + addition.operate(5, 3));

        // Using lambda expression (modern way - preview)
        Calculator multiplication = (a, b) -> a * b;
        System.out.println("Multiplication: 5 * 3 = " + multiplication.operate(5, 3));

        // Another lambda with block body
        Calculator power = (base, exponent) -> {
            int result = 1;
            for (int i = 0; i < exponent; i++) {
                result *= base;
            }
            return result;
        };
        System.out.println("Power: 2 ^ 3 = " + power.operate(2, 3));

        System.out.println("\\n=== Built-in Functional Interfaces ===");

        // Function<T, R> - takes one argument, returns result
        Function<String, String> toUpperCase = str -> str.toUpperCase();
        System.out.println("Function - Uppercase: " + toUpperCase.apply("hello"));

        // Predicate<T> - takes one argument, returns boolean
        Predicate<Integer> isEven = num -> num % 2 == 0;
        System.out.println("Predicate - Is 4 even? " + isEven.test(4));

        // Consumer<T> - takes one argument, returns nothing
        Consumer<String> printFormatted = msg ->
            System.out.println("Consumer - Message: Processing: " + msg);
        printFormatted.accept("Data");

        // Supplier<T> - takes no arguments, returns result
        Supplier<Integer> randomNumber = () -> new Random().nextInt(100);
        System.out.println("Supplier - Random number: " + randomNumber.get());
    }
}`,
    hint1: `A functional interface must have exactly ONE abstract method. Use @FunctionalInterface annotation to ensure this at compile time.`,
    hint2: `Built-in functional interfaces: Function (transform), Predicate (test), Consumer (accept), Supplier (provide). They work great with lambda expressions.`,
    whyItMatters: `Functional interfaces are the foundation of lambda expressions and functional programming in Java. They enable a more declarative programming style and are essential for Stream API, parallel processing, and modern Java frameworks. Understanding functional interfaces prepares you for advanced topics like method references, streams, and reactive programming.`,
    order: 5,
    testCode: `import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;
import java.util.function.*;

// Test 1: Calculator is a functional interface
class Test1 {
    @Test
    void testCalculatorIsFunctionalInterface() {
        Calculator calc = (a, b) -> a + b;
        assertNotNull(calc);
    }
}

// Test 2: Calculator addition works
class Test2 {
    @Test
    void testCalculatorAddition() {
        Calculator add = (a, b) -> a + b;
        assertEquals(8, add.operate(5, 3));
    }
}

// Test 3: Calculator multiplication works
class Test3 {
    @Test
    void testCalculatorMultiplication() {
        Calculator mult = (a, b) -> a * b;
        assertEquals(15, mult.operate(5, 3));
    }
}

// Test 4: Calculator subtraction works
class Test4 {
    @Test
    void testCalculatorSubtraction() {
        Calculator sub = (a, b) -> a - b;
        assertEquals(2, sub.operate(5, 3));
    }
}

// Test 5: Calculator default describe method
class Test5 {
    @Test
    void testCalculatorDescribe() {
        Calculator calc = (a, b) -> a + b;
        assertEquals("Calculator operation", calc.describe());
    }
}

// Test 6: Function interface works
class Test6 {
    @Test
    void testFunctionInterface() {
        Function<String, String> toUpper = str -> str.toUpperCase();
        assertEquals("HELLO", toUpper.apply("hello"));
    }
}

// Test 7: Predicate interface works
class Test7 {
    @Test
    void testPredicateInterface() {
        Predicate<Integer> isEven = num -> num % 2 == 0;
        assertTrue(isEven.test(4));
        assertFalse(isEven.test(5));
    }
}

// Test 8: Consumer interface works
class Test8 {
    @Test
    void testConsumerInterface() {
        StringBuilder sb = new StringBuilder();
        Consumer<String> append = msg -> sb.append(msg);
        append.accept("Hello");
        assertEquals("Hello", sb.toString());
    }
}

// Test 9: Supplier interface works
class Test9 {
    @Test
    void testSupplierInterface() {
        Supplier<Integer> getFortyTwo = () -> 42;
        assertEquals(42, getFortyTwo.get());
    }
}

// Test 10: Lambda with block body works
class Test10 {
    @Test
    void testLambdaWithBlockBody() {
        Calculator power = (base, exponent) -> {
            int result = 1;
            for (int i = 0; i < exponent; i++) {
                result *= base;
            }
            return result;
        };
        assertEquals(8, power.operate(2, 3));
        assertEquals(27, power.operate(3, 3));
    }
}`,
    translations: {
        ru: {
            title: 'Функциональные интерфейсы и @FunctionalInterface',
            solutionCode: `import java.util.function.*;
import java.util.Random;

// Функциональный интерфейс с аннотацией @FunctionalInterface
@FunctionalInterface
interface Calculator {
    // Единственный абстрактный метод (SAM)
    int operate(int a, int b);

    // Методы по умолчанию разрешены
    default String describe() {
        return "Calculator operation";
    }
}

public class FunctionalInterfaces {
    public static void main(String[] args) {
        System.out.println("=== Пользовательский функциональный интерфейс ===");

        // Используя анонимный класс (традиционный способ)
        Calculator addition = new Calculator() {
            @Override
            public int operate(int a, int b) {
                return a + b;
            }
        };
        System.out.println("Addition: 5 + 3 = " + addition.operate(5, 3));

        // Используя лямбда-выражение (современный способ - предпросмотр)
        Calculator multiplication = (a, b) -> a * b;
        System.out.println("Multiplication: 5 * 3 = " + multiplication.operate(5, 3));

        // Еще одна лямбда с телом блока
        Calculator power = (base, exponent) -> {
            int result = 1;
            for (int i = 0; i < exponent; i++) {
                result *= base;
            }
            return result;
        };
        System.out.println("Power: 2 ^ 3 = " + power.operate(2, 3));

        System.out.println("\\n=== Встроенные функциональные интерфейсы ===");

        // Function<T, R> - принимает один аргумент, возвращает результат
        Function<String, String> toUpperCase = str -> str.toUpperCase();
        System.out.println("Function - Uppercase: " + toUpperCase.apply("hello"));

        // Predicate<T> - принимает один аргумент, возвращает boolean
        Predicate<Integer> isEven = num -> num % 2 == 0;
        System.out.println("Predicate - Is 4 even? " + isEven.test(4));

        // Consumer<T> - принимает один аргумент, ничего не возвращает
        Consumer<String> printFormatted = msg ->
            System.out.println("Consumer - Message: Processing: " + msg);
        printFormatted.accept("Data");

        // Supplier<T> - не принимает аргументов, возвращает результат
        Supplier<Integer> randomNumber = () -> new Random().nextInt(100);
        System.out.println("Supplier - Random number: " + randomNumber.get());
    }
}`,
            description: `# Функциональные интерфейсы и @FunctionalInterface

Функциональный интерфейс - это интерфейс с ровно одним абстрактным методом (SAM - Single Abstract Method). Они могут быть созданы с использованием лямбда-выражений. Java предоставляет встроенные функциональные интерфейсы в пакете java.util.function.

## Требования:
1. Создайте пользовательский функциональный интерфейс \`Calculator\`:
   1.1. С аннотацией \`@FunctionalInterface\`
   1.2. Единственный абстрактный метод: \`int operate(int a, int b)\`
   1.3. Метод по умолчанию: \`String describe()\`, возвращающий "Calculator operation"

2. Продемонстрируйте использование встроенных функциональных интерфейсов:
   2.1. \`Function<T,R>\`: Преобразовать строку в верхний регистр
   2.2. \`Predicate<T>\`: Проверить, является ли число четным
   2.3. \`Consumer<T>\`: Напечатать форматированное сообщение
   2.4. \`Supplier<T>\`: Сгенерировать случайное число

3. Используйте пользовательский Calculator с:
   3.1. Операцией сложения
   3.2. Операцией умножения
   3.3. Пользовательской операцией (возведение в степень)

4. Покажите как анонимный класс, так и лямбда-выражения (предпросмотр для следующих модулей)

## Пример вывода:
\`\`\`
=== Custom Functional Interface ===
Addition: 5 + 3 = 8
Multiplication: 5 * 3 = 15
Power: 2 ^ 3 = 8

=== Built-in Functional Interfaces ===
Function - Uppercase: HELLO
Predicate - Is 4 even? true
Consumer - Message: Processing: Data
Supplier - Random number: 42
\`\`\``,
            hint1: `Функциональный интерфейс должен иметь ровно ОДИН абстрактный метод. Используйте аннотацию @FunctionalInterface для обеспечения этого на этапе компиляции.`,
            hint2: `Встроенные функциональные интерфейсы: Function (преобразование), Predicate (тестирование), Consumer (принятие), Supplier (предоставление). Они отлично работают с лямбда-выражениями.`,
            whyItMatters: `Функциональные интерфейсы являются основой лямбда-выражений и функционального программирования в Java. Они позволяют использовать более декларативный стиль программирования и необходимы для Stream API, параллельной обработки и современных Java-фреймворков. Понимание функциональных интерфейсов подготавливает вас к продвинутым темам, таким как ссылки на методы, потоки и реактивное программирование.`
        },
        uz: {
            title: `Funksional interfeyslar va @FunctionalInterface`,
            solutionCode: `import java.util.function.*;
import java.util.Random;

// @FunctionalInterface annotatsiyasi bilan funksional interfeys
@FunctionalInterface
interface Calculator {
    // Yagona abstrakt metod (SAM)
    int operate(int a, int b);

    // Standart metodlarga ruxsat berilgan
    default String describe() {
        return "Calculator operation";
    }
}

public class FunctionalInterfaces {
    public static void main(String[] args) {
        System.out.println("=== Maxsus funksional interfeys ===");

        // Anonim klassdan foydalanish (an'anaviy usul)
        Calculator addition = new Calculator() {
            @Override
            public int operate(int a, int b) {
                return a + b;
            }
        };
        System.out.println("Addition: 5 + 3 = " + addition.operate(5, 3));

        // Lambda ifodadan foydalanish (zamonaviy usul - oldindan ko'rish)
        Calculator multiplication = (a, b) -> a * b;
        System.out.println("Multiplication: 5 * 3 = " + multiplication.operate(5, 3));

        // Blok tanasi bilan yana bir lambda
        Calculator power = (base, exponent) -> {
            int result = 1;
            for (int i = 0; i < exponent; i++) {
                result *= base;
            }
            return result;
        };
        System.out.println("Power: 2 ^ 3 = " + power.operate(2, 3));

        System.out.println("\\n=== O'rnatilgan funksional interfeyslar ===");

        // Function<T, R> - bitta argumentni oladi, natijani qaytaradi
        Function<String, String> toUpperCase = str -> str.toUpperCase();
        System.out.println("Function - Uppercase: " + toUpperCase.apply("hello"));

        // Predicate<T> - bitta argumentni oladi, boolean qaytaradi
        Predicate<Integer> isEven = num -> num % 2 == 0;
        System.out.println("Predicate - Is 4 even? " + isEven.test(4));

        // Consumer<T> - bitta argumentni oladi, hech narsa qaytarmaydi
        Consumer<String> printFormatted = msg ->
            System.out.println("Consumer - Message: Processing: " + msg);
        printFormatted.accept("Data");

        // Supplier<T> - argumentlarni olmaydi, natijani qaytaradi
        Supplier<Integer> randomNumber = () -> new Random().nextInt(100);
        System.out.println("Supplier - Random number: " + randomNumber.get());
    }
}`,
            description: `# Funksional interfeyslar va @FunctionalInterface

Funksional interfeys aynan bitta abstrakt metodga ega interfeys hisoblanadi (SAM - Single Abstract Method). Ular lambda ifodalari yordamida yaratilishi mumkin. Java java.util.function paketida o'rnatilgan funksional interfeyslarni taqdim etadi.

## Talablar:
1. Maxsus \`Calculator\` funksional interfeysini yarating:
   1.1. \`@FunctionalInterface\` bilan izohlanган
   1.2. Yagona abstrakt metod: \`int operate(int a, int b)\`
   1.3. Standart metod: \`String describe()\` "Calculator operation" ni qaytaradi

2. O'rnatilgan funksional interfeyslardan foydalanishni namoyish eting:
   2.1. \`Function<T,R>\`: Satrni katta harflarga o'zgartirish
   2.2. \`Predicate<T>\`: Raqam juft ekanligini tekshirish
   2.3. \`Consumer<T>\`: Formatlangan xabarni chiqarish
   2.4. \`Supplier<T>\`: Tasodifiy raqam yaratish

3. Maxsus Calculator dan foydalaning:
   3.1. Qo'shish operatsiyasi
   3.2. Ko'paytirish operatsiyasi
   3.3. Maxsus operatsiya (daraja)

4. Anonim klass va lambda ifodalarini ko'rsating (keyingi modullar uchun oldindan ko'rish)

## Chiqish namunasi:
\`\`\`
=== Custom Functional Interface ===
Addition: 5 + 3 = 8
Multiplication: 5 * 3 = 15
Power: 2 ^ 3 = 8

=== Built-in Functional Interfaces ===
Function - Uppercase: HELLO
Predicate - Is 4 even? true
Consumer - Message: Processing: Data
Supplier - Random number: 42
\`\`\``,
            hint1: `Funksional interfeys aynan BITTA abstrakt metodga ega bo'lishi kerak. Buni kompilyatsiya vaqtida ta'minlash uchun @FunctionalInterface annotatsiyasidan foydalaning.`,
            hint2: `O'rnatilgan funksional interfeyslar: Function (o'zgartirish), Predicate (tekshirish), Consumer (qabul qilish), Supplier (taqdim etish). Ular lambda ifodalari bilan ajoyib ishlaydi.`,
            whyItMatters: `Funksional interfeyslar Java-da lambda ifodalari va funksional dasturlashning asosi hisoblanadi. Ular yanada deklarativ dasturlash uslubini yoqadi va Stream API, parallel qayta ishlash va zamonaviy Java frameworklari uchun zarurdir. Funksional interfeyslarni tushunish sizni metodlarga havolalar, oqimlar va reaktiv dasturlash kabi ilg'or mavzularga tayyorlaydi.`
        }
    }
};

export default task;
