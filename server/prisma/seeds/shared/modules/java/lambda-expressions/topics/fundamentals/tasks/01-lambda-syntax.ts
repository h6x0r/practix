import { Task } from '../../../../types';

export const task: Task = {
    slug: 'java-lambda-syntax',
    title: 'Lambda Expression Syntax',
    difficulty: 'easy',
    tags: ['java', 'lambda', 'functional-programming', 'java8', 'syntax'],
    estimatedTime: '20m',
    isPremium: false,
    youtubeUrl: '',
    description: `# Lambda Expression Syntax

Lambda expressions provide a concise way to represent anonymous functions. They consist of parameters, an arrow operator (->), and a body. Lambda expressions enable functional programming in Java and work with functional interfaces.

## Requirements:
1. Create different lambda expression styles:
   1.1. No parameters: \`() -> expression\`
   1.2. Single parameter: \`x -> expression\` or \`(x) -> expression\`
   1.3. Multiple parameters: \`(x, y) -> expression\`
   1.4. Block body with return: \`(x, y) -> { statements; return value; }\`

2. Demonstrate type inference:
   2.1. Explicit types: \`(String s) -> s.length()\`
   2.2. Inferred types: \`s -> s.length()\`

3. Create lambdas for various operations:
   3.1. String operations (concatenation, uppercase)
   3.2. Arithmetic operations (addition, multiplication)
   3.3. Complex operations with multiple statements

4. Show equivalent anonymous class for comparison

## Example Output:
\`\`\`
=== Lambda Syntax Examples ===
No parameters: Hello from lambda!
Single parameter: HELLO
Multiple parameters: 5 + 3 = 8
Block body: 5! = 120

=== Type Inference ===
Explicit types: Length of "Java" = 4
Inferred types: Length of "Lambda" = 6

=== Anonymous Class vs Lambda ===
Anonymous class result: 15
Lambda result: 15
\`\`\``,
    initialCode: `// TODO: Import necessary functional interfaces

public class LambdaSyntax {
    public static void main(String[] args) {
        // TODO: Create lambda with no parameters

        // TODO: Create lambda with single parameter

        // TODO: Create lambda with multiple parameters

        // TODO: Create lambda with block body

        // TODO: Demonstrate type inference

        // TODO: Compare anonymous class vs lambda
    }
}`,
    solutionCode: `import java.util.function.*;

public class LambdaSyntax {
    public static void main(String[] args) {
        System.out.println("=== Lambda Syntax Examples ===");

        // No parameters: () -> expression
        Supplier<String> greeting = () -> "Hello from lambda!";
        System.out.println("No parameters: " + greeting.get());

        // Single parameter: x -> expression (parentheses optional)
        Function<String, String> toUpperCase = s -> s.toUpperCase();
        System.out.println("Single parameter: " + toUpperCase.apply("hello"));

        // Multiple parameters: (x, y) -> expression
        BiFunction<Integer, Integer, Integer> add = (a, b) -> a + b;
        System.out.println("Multiple parameters: 5 + 3 = " + add.apply(5, 3));

        // Block body with return statement
        Function<Integer, Integer> factorial = n -> {
            int result = 1;
            for (int i = 1; i <= n; i++) {
                result *= i;
            }
            return result;
        };
        System.out.println("Block body: 5! = " + factorial.apply(5));

        System.out.println("\\n=== Type Inference ===");

        // Explicit parameter types
        Function<String, Integer> explicitTypes = (String s) -> s.length();
        System.out.println("Explicit types: Length of \\"Java\\" = " +
            explicitTypes.apply("Java"));

        // Inferred parameter types (preferred)
        Function<String, Integer> inferredTypes = s -> s.length();
        System.out.println("Inferred types: Length of \\"Lambda\\" = " +
            inferredTypes.apply("Lambda"));

        System.out.println("\\n=== Anonymous Class vs Lambda ===");

        // Traditional anonymous class
        BiFunction<Integer, Integer, Integer> multiplyAnon = new BiFunction<>() {
            @Override
            public Integer apply(Integer a, Integer b) {
                return a * b;
            }
        };
        System.out.println("Anonymous class result: " + multiplyAnon.apply(3, 5));

        // Equivalent lambda expression
        BiFunction<Integer, Integer, Integer> multiplyLambda = (a, b) -> a * b;
        System.out.println("Lambda result: " + multiplyLambda.apply(3, 5));
    }
}`,
    hint1: `Lambda syntax: (parameters) -> expression or (parameters) -> { statements; return value; }. Parentheses are optional for single parameters.`,
    hint2: `Java can infer parameter types from the context, so you don't need to specify them explicitly unless you want to.`,
    whyItMatters: `Lambda expressions make code more concise and readable. They reduce boilerplate code, enable functional programming patterns, and are essential for working with Java Streams and modern Java APIs. Understanding lambda syntax is fundamental to writing modern, clean Java code.

**Production Pattern:**

\`\`\`java
// WITHOUT lambda: Verbose anonymous class
button.addActionListener(new ActionListener() {
    @Override
    public void actionPerformed(ActionEvent e) {
        System.out.println("Button clicked");
    }
});

// WITH lambda: Concise and clear
button.addActionListener(e -> System.out.println("Button clicked"));

// Collection processing WITHOUT lambda
List<String> names = Arrays.asList("Alice", "Bob", "Charlie");
for (String name : names) {
    System.out.println(name.toUpperCase());
}

// Collection processing WITH lambda
names.forEach(name -> System.out.println(name.toUpperCase()));
\`\`\`

**Practical Benefits:**

1. **Code Reduction**: 70-80% fewer lines compared to anonymous classes
2. **Improved Readability**: Focus on logic, not boilerplate code
3. **Functional Programming**: Ability to pass behavior as parameter
4. **Stream API Compatibility**: Foundation for modern collection processing`,
    order: 1,
    testCode: `import static org.junit.Assert.*;
import org.junit.Test;
import java.util.function.*;

// Test1: Verify Supplier lambda with no parameters
class Test1 {
    @Test
    public void test() {
        Supplier<String> greeting = () -> "Hello";
        assertEquals("Hello", greeting.get());
    }
}

// Test2: Verify Function lambda with single parameter
class Test2 {
    @Test
    public void test() {
        Function<String, String> toUpper = s -> s.toUpperCase();
        assertEquals("HELLO", toUpper.apply("hello"));
    }
}

// Test3: Verify BiFunction lambda with two parameters
class Test3 {
    @Test
    public void test() {
        BiFunction<Integer, Integer, Integer> add = (a, b) -> a + b;
        assertEquals(Integer.valueOf(8), add.apply(5, 3));
    }
}

// Test4: Verify lambda with return value
class Test4 {
    @Test
    public void test() {
        Function<Integer, Integer> square = x -> x * x;
        assertEquals(Integer.valueOf(25), square.apply(5));
    }
}

// Test5: Verify lambda with block body and return
class Test5 {
    @Test
    public void test() {
        Function<Integer, Integer> factorial = n -> {
            int result = 1;
            for (int i = 1; i <= n; i++) {
                result *= i;
            }
            return result;
        };
        assertEquals(Integer.valueOf(120), factorial.apply(5));
    }
}

// Test6: Verify explicit parameter types
class Test6 {
    @Test
    public void test() {
        Function<String, Integer> len = (String s) -> s.length();
        assertEquals(Integer.valueOf(4), len.apply("Java"));
    }
}

// Test7: Verify Predicate lambda
class Test7 {
    @Test
    public void test() {
        Predicate<Integer> isEven = n -> n % 2 == 0;
        assertTrue(isEven.test(4));
        assertFalse(isEven.test(3));
    }
}

// Test8: Verify Consumer lambda outputs correctly
class Test8 {
    @Test
    public void test() {
        java.io.ByteArrayOutputStream out = new java.io.ByteArrayOutputStream();
        java.io.PrintStream originalOut = System.out;
        System.setOut(new java.io.PrintStream(out));
        try {
            Consumer<String> consumer = s -> System.out.println(s);
            consumer.accept("HelloLambda");
            String output = out.toString().trim();
            assertEquals("HelloLambda", output);
        } finally {
            System.setOut(originalOut);
        }
    }
}

// Test9: Verify BiFunction multiplication
class Test9 {
    @Test
    public void test() {
        BiFunction<Integer, Integer, Integer> mult = (a, b) -> a * b;
        assertEquals(Integer.valueOf(15), mult.apply(3, 5));
    }
}

// Test10: Verify inferred types
class Test10 {
    @Test
    public void test() {
        Function<String, Integer> len = s -> s.length();
        assertEquals(Integer.valueOf(6), len.apply("Lambda"));
    }
}`,
    translations: {
        ru: {
            title: 'Синтаксис лямбда-выражений',
            solutionCode: `import java.util.function.*;

public class LambdaSyntax {
    public static void main(String[] args) {
        System.out.println("=== Примеры синтаксиса лямбда ===");

        // Без параметров: () -> выражение
        Supplier<String> greeting = () -> "Hello from lambda!";
        System.out.println("No parameters: " + greeting.get());

        // Один параметр: x -> выражение (скобки необязательны)
        Function<String, String> toUpperCase = s -> s.toUpperCase();
        System.out.println("Single parameter: " + toUpperCase.apply("hello"));

        // Несколько параметров: (x, y) -> выражение
        BiFunction<Integer, Integer, Integer> add = (a, b) -> a + b;
        System.out.println("Multiple parameters: 5 + 3 = " + add.apply(5, 3));

        // Тело блока с оператором return
        Function<Integer, Integer> factorial = n -> {
            int result = 1;
            for (int i = 1; i <= n; i++) {
                result *= i;
            }
            return result;
        };
        System.out.println("Block body: 5! = " + factorial.apply(5));

        System.out.println("\\n=== Вывод типов ===");

        // Явные типы параметров
        Function<String, Integer> explicitTypes = (String s) -> s.length();
        System.out.println("Explicit types: Length of \\"Java\\" = " +
            explicitTypes.apply("Java"));

        // Выводимые типы параметров (предпочтительно)
        Function<String, Integer> inferredTypes = s -> s.length();
        System.out.println("Inferred types: Length of \\"Lambda\\" = " +
            inferredTypes.apply("Lambda"));

        System.out.println("\\n=== Анонимный класс vs Лямбда ===");

        // Традиционный анонимный класс
        BiFunction<Integer, Integer, Integer> multiplyAnon = new BiFunction<>() {
            @Override
            public Integer apply(Integer a, Integer b) {
                return a * b;
            }
        };
        System.out.println("Anonymous class result: " + multiplyAnon.apply(3, 5));

        // Эквивалентное лямбда-выражение
        BiFunction<Integer, Integer, Integer> multiplyLambda = (a, b) -> a * b;
        System.out.println("Lambda result: " + multiplyLambda.apply(3, 5));
    }
}`,
            description: `# Синтаксис лямбда-выражений

Лямбда-выражения предоставляют краткий способ представления анонимных функций. Они состоят из параметров, оператора стрелки (->) и тела. Лямбда-выражения обеспечивают функциональное программирование в Java и работают с функциональными интерфейсами.

## Требования:
1. Создайте различные стили лямбда-выражений:
   1.1. Без параметров: \`() -> expression\`
   1.2. Один параметр: \`x -> expression\` или \`(x) -> expression\`
   1.3. Несколько параметров: \`(x, y) -> expression\`
   1.4. Тело блока с return: \`(x, y) -> { statements; return value; }\`

2. Продемонстрируйте вывод типов:
   2.1. Явные типы: \`(String s) -> s.length()\`
   2.2. Выводимые типы: \`s -> s.length()\`

3. Создайте лямбды для различных операций:
   3.1. Операции со строками (конкатенация, верхний регистр)
   3.2. Арифметические операции (сложение, умножение)
   3.3. Сложные операции с несколькими операторами

4. Покажите эквивалентный анонимный класс для сравнения

## Пример вывода:
\`\`\`
=== Lambda Syntax Examples ===
No parameters: Hello from lambda!
Single parameter: HELLO
Multiple parameters: 5 + 3 = 8
Block body: 5! = 120

=== Type Inference ===
Explicit types: Length of "Java" = 4
Inferred types: Length of "Lambda" = 6

=== Anonymous Class vs Lambda ===
Anonymous class result: 15
Lambda result: 15
\`\`\``,
            hint1: `Синтаксис лямбды: (параметры) -> выражение или (параметры) -> { операторы; return значение; }. Скобки необязательны для одного параметра.`,
            hint2: `Java может вывести типы параметров из контекста, поэтому вам не нужно указывать их явно, если вы не хотите.`,
            whyItMatters: `Лямбда-выражения делают код более кратким и читаемым. Они уменьшают шаблонный код, обеспечивают паттерны функционального программирования и необходимы для работы с Java Streams и современными Java API. Понимание синтаксиса лямбды является основой для написания современного, чистого Java-кода.

**Продакшен паттерн:**

\`\`\`java
// БЕЗ лямбда: Многословный анонимный класс
button.addActionListener(new ActionListener() {
    @Override
    public void actionPerformed(ActionEvent e) {
        System.out.println("Button clicked");
    }
});

// С лямбда: Краткая и понятная запись
button.addActionListener(e -> System.out.println("Button clicked"));

// Обработка коллекций БЕЗ лямбда
List<String> names = Arrays.asList("Alice", "Bob", "Charlie");
for (String name : names) {
    System.out.println(name.toUpperCase());
}

// Обработка коллекций С лямбда
names.forEach(name -> System.out.println(name.toUpperCase()));
\`\`\`

**Практические преимущества:**

1. **Сокращение кода**: На 70-80% меньше строк по сравнению с анонимными классами
2. **Улучшение читаемости**: Фокус на логике, а не на шаблонном коде
3. **Функциональное программирование**: Возможность передавать поведение как параметр
4. **Совместимость со Stream API**: Основа для современной обработки коллекций`
        },
        uz: {
            title: `Lambda ifodalar sintaksisi`,
            solutionCode: `import java.util.function.*;

public class LambdaSyntax {
    public static void main(String[] args) {
        System.out.println("=== Lambda sintaksisi namunalari ===");

        // Parametrlarsiz: () -> ifoda
        Supplier<String> greeting = () -> "Hello from lambda!";
        System.out.println("No parameters: " + greeting.get());

        // Bitta parametr: x -> ifoda (qavslar ixtiyoriy)
        Function<String, String> toUpperCase = s -> s.toUpperCase();
        System.out.println("Single parameter: " + toUpperCase.apply("hello"));

        // Bir nechta parametr: (x, y) -> ifoda
        BiFunction<Integer, Integer, Integer> add = (a, b) -> a + b;
        System.out.println("Multiple parameters: 5 + 3 = " + add.apply(5, 3));

        // Return operatori bilan blok tanasi
        Function<Integer, Integer> factorial = n -> {
            int result = 1;
            for (int i = 1; i <= n; i++) {
                result *= i;
            }
            return result;
        };
        System.out.println("Block body: 5! = " + factorial.apply(5));

        System.out.println("\\n=== Turni aniqlash ===");

        // Aniq parametr turlari
        Function<String, Integer> explicitTypes = (String s) -> s.length();
        System.out.println("Explicit types: Length of \\"Java\\" = " +
            explicitTypes.apply("Java"));

        // Aniqlanadigan parametr turlari (afzal)
        Function<String, Integer> inferredTypes = s -> s.length();
        System.out.println("Inferred types: Length of \\"Lambda\\" = " +
            inferredTypes.apply("Lambda"));

        System.out.println("\\n=== Anonim klass vs Lambda ===");

        // An'anaviy anonim klass
        BiFunction<Integer, Integer, Integer> multiplyAnon = new BiFunction<>() {
            @Override
            public Integer apply(Integer a, Integer b) {
                return a * b;
            }
        };
        System.out.println("Anonymous class result: " + multiplyAnon.apply(3, 5));

        // Ekvivalent lambda ifoda
        BiFunction<Integer, Integer, Integer> multiplyLambda = (a, b) -> a * b;
        System.out.println("Lambda result: " + multiplyLambda.apply(3, 5));
    }
}`,
            description: `# Lambda ifodalar sintaksisi

Lambda ifodalari anonim funksiyalarni ifodalashning qisqa usulini taqdim etadi. Ular parametrlar, o'q operatori (->) va tanadan iborat. Lambda ifodalari Java-da funksional dasturlashni yoqadi va funksional interfeyslar bilan ishlaydi.

## Talablar:
1. Turli lambda ifoda uslublarini yarating:
   1.1. Parametrlarsiz: \`() -> expression\`
   1.2. Bitta parametr: \`x -> expression\` yoki \`(x) -> expression\`
   1.3. Bir nechta parametr: \`(x, y) -> expression\`
   1.4. Return bilan blok tanasi: \`(x, y) -> { statements; return value; }\`

2. Turni aniqlashni namoyish eting:
   2.1. Aniq turlar: \`(String s) -> s.length()\`
   2.2. Aniqlanadigan turlar: \`s -> s.length()\`

3. Turli operatsiyalar uchun lambdalar yarating:
   3.1. Satr operatsiyalari (birikma, katta harf)
   3.2. Arifmetik operatsiyalar (qo'shish, ko'paytirish)
   3.3. Bir nechta operatorlar bilan murakkab operatsiyalar

4. Taqqoslash uchun ekvivalent anonim klassni ko'rsating

## Chiqish namunasi:
\`\`\`
=== Lambda Syntax Examples ===
No parameters: Hello from lambda!
Single parameter: HELLO
Multiple parameters: 5 + 3 = 8
Block body: 5! = 120

=== Type Inference ===
Explicit types: Length of "Java" = 4
Inferred types: Length of "Lambda" = 6

=== Anonymous Class vs Lambda ===
Anonymous class result: 15
Lambda result: 15
\`\`\``,
            hint1: `Lambda sintaksisi: (parametrlar) -> ifoda yoki (parametrlar) -> { operatorlar; return qiymat; }. Qavslar bitta parametr uchun ixtiyoriy.`,
            hint2: `Java kontekstdan parametr turlarini aniqlay oladi, shuning uchun siz ularni aniq ko'rsatishingiz shart emas.`,
            whyItMatters: `Lambda ifodalari kodni qisqaroq va o'qilishi osonroq qiladi. Ular shablon kodini kamaytiradi, funksional dasturlash naqshlarini yoqadi va Java Streams va zamonaviy Java API-lari bilan ishlash uchun zarurdir. Lambda sintaksisini tushunish zamonaviy, toza Java kodi yozishning asosi hisoblanadi.

**Ishlab chiqarish patterni:**

\`\`\`java
// Lambdasiz: Ko'p so'zli anonim klass
button.addActionListener(new ActionListener() {
    @Override
    public void actionPerformed(ActionEvent e) {
        System.out.println("Button clicked");
    }
});

// Lambda bilan: Qisqa va tushunarli yozuv
button.addActionListener(e -> System.out.println("Button clicked"));

// Kolleksiyalarni qayta ishlash lambdasiz
List<String> names = Arrays.asList("Alice", "Bob", "Charlie");
for (String name : names) {
    System.out.println(name.toUpperCase());
}

// Kolleksiyalarni qayta ishlash lambda bilan
names.forEach(name -> System.out.println(name.toUpperCase()));
\`\`\`

**Amaliy foydalari:**

1. **Kodni qisqartirish**: Anonim klasslarga nisbatan 70-80% kam qatorlar
2. **O'qilishini yaxshilash**: Shablon kodi emas, mantiqqa e'tibor
3. **Funksional dasturlash**: Xatti-harakatni parametr sifatida uzatish imkoniyati
4. **Stream API bilan moslashuvchanlik**: Zamonaviy kolleksiyalarni qayta ishlash uchun asos`
        }
    }
};

export default task;
