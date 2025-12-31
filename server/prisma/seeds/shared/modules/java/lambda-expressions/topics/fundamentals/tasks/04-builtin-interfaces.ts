import { Task } from '../../../../types';

export const task: Task = {
    slug: 'java-builtin-functional-interfaces',
    title: 'Built-in Functional Interfaces',
    difficulty: 'medium',
    tags: ['java', 'functional-interface', 'lambda', 'java8', 'java-util-function'],
    estimatedTime: '35m',
    isPremium: false,
    youtubeUrl: '',
    description: `# Built-in Functional Interfaces

Java provides a rich set of functional interfaces in java.util.function package. These interfaces cover common use cases and can be used with lambda expressions and method references. Understanding these interfaces is essential for working with streams and functional programming.

## Requirements:
1. Core functional interfaces:
   1.1. \`Function<T, R>\`: Takes T, returns R - transform data
   1.2. \`Predicate<T>\`: Takes T, returns boolean - test condition
   1.3. \`Consumer<T>\`: Takes T, returns void - consume/process
   1.4. \`Supplier<T>\`: Takes nothing, returns T - supply/generate

2. Binary variants (two parameters):
   2.1. \`BiFunction<T, U, R>\`: Takes T and U, returns R
   2.2. \`BiPredicate<T, U>\`: Takes T and U, returns boolean
   2.3. \`BiConsumer<T, U>\`: Takes T and U, returns void

3. Specialized primitive interfaces:
   3.1. \`IntFunction<R>\`, \`ToIntFunction<T>\`: Work with int
   3.2. \`IntPredicate\`, \`IntConsumer\`, \`IntSupplier\`
   3.3. Similar for Long and Double

4. Demonstrate chaining and composition:
   4.1. Function composition: \`andThen()\`, \`compose()\`
   4.2. Predicate logic: \`and()\`, \`or()\`, \`negate()\`

## Example Output:
\`\`\`
=== Core Functional Interfaces ===
Function: 5 -> 25
Predicate: Is 8 even? true
Consumer: Processing: Test Data
Supplier: Generated UUID: 123e4567-e89b

=== Binary Functional Interfaces ===
BiFunction: max(10, 20) = 20
BiPredicate: Are "hello" and "hello" equal? true
BiConsumer: Key=name, Value=John

=== Specialized Interfaces ===
IntFunction: Square of 7 = 49
ToIntFunction: Length of "Java" = 4
IntPredicate: Is 15 positive? true

=== Composition and Chaining ===
Chained function: 3 -> 9 -> 18
Combined predicate: Is 10 even and > 5? true
Negated predicate: Is 3 not even? true
\`\`\``,
    initialCode: `import java.util.function.*;

public class BuiltInInterfaces {
    public static void main(String[] args) {
        // TODO: Demonstrate Function, Predicate, Consumer, Supplier

        // TODO: Demonstrate BiFunction, BiPredicate, BiConsumer

        // TODO: Demonstrate specialized primitive interfaces

        // TODO: Demonstrate composition and chaining
    }
}`,
    solutionCode: `import java.util.function.*;
import java.util.UUID;

public class BuiltInInterfaces {
    public static void main(String[] args) {
        System.out.println("=== Core Functional Interfaces ===");

        // Function<T, R> - transform data
        Function<Integer, Integer> square = x -> x * x;
        System.out.println("Function: 5 -> " + square.apply(5));

        // Predicate<T> - test condition
        Predicate<Integer> isEven = n -> n % 2 == 0;
        System.out.println("Predicate: Is 8 even? " + isEven.test(8));

        // Consumer<T> - consume/process data
        Consumer<String> printWithPrefix = s ->
            System.out.println("Consumer: Processing: " + s);
        printWithPrefix.accept("Test Data");

        // Supplier<T> - supply/generate data
        Supplier<String> uuidSupplier = () -> UUID.randomUUID().toString().substring(0, 13);
        System.out.println("Supplier: Generated UUID: " + uuidSupplier.get());

        System.out.println("\\n=== Binary Functional Interfaces ===");

        // BiFunction<T, U, R> - two inputs, one output
        BiFunction<Integer, Integer, Integer> max = (a, b) -> Math.max(a, b);
        System.out.println("BiFunction: max(10, 20) = " + max.apply(10, 20));

        // BiPredicate<T, U> - two inputs, boolean output
        BiPredicate<String, String> areEqual = (s1, s2) -> s1.equals(s2);
        System.out.println("BiPredicate: Are \\"hello\\" and \\"hello\\" equal? " +
            areEqual.test("hello", "hello"));

        // BiConsumer<T, U> - two inputs, void output
        BiConsumer<String, String> printKeyValue = (key, value) ->
            System.out.println("BiConsumer: Key=" + key + ", Value=" + value);
        printKeyValue.accept("name", "John");

        System.out.println("\\n=== Specialized Interfaces ===");

        // IntFunction<R> - int input, R output
        IntFunction<String> squareAsString = n -> "Square of " + n + " = " + (n * n);
        System.out.println("IntFunction: " + squareAsString.apply(7));

        // ToIntFunction<T> - T input, int output
        ToIntFunction<String> stringLength = s -> s.length();
        System.out.println("ToIntFunction: Length of \\"Java\\" = " +
            stringLength.applyAsInt("Java"));

        // IntPredicate - int input, boolean output
        IntPredicate isPositive = n -> n > 0;
        System.out.println("IntPredicate: Is 15 positive? " + isPositive.test(15));

        System.out.println("\\n=== Composition and Chaining ===");

        // Function composition with andThen
        Function<Integer, Integer> multiplyBy3 = x -> x * 3;
        Function<Integer, Integer> multiplyBy2 = x -> x * 2;
        Function<Integer, Integer> chained = square.andThen(multiplyBy2);
        System.out.println("Chained function: 3 -> " + chained.apply(3));

        // Predicate combination with and/or
        Predicate<Integer> isGreaterThan5 = n -> n > 5;
        Predicate<Integer> isEvenAndGreaterThan5 = isEven.and(isGreaterThan5);
        System.out.println("Combined predicate: Is 10 even and > 5? " +
            isEvenAndGreaterThan5.test(10));

        // Predicate negation
        Predicate<Integer> isOdd = isEven.negate();
        System.out.println("Negated predicate: Is 3 not even? " + isOdd.test(3));
    }
}`,
    hint1: `The main functional interfaces follow a pattern: Function (transform), Predicate (test), Consumer (process), Supplier (generate). Each has a specific purpose.`,
    hint2: `Use specialized primitive interfaces (IntFunction, LongPredicate, etc.) to avoid boxing/unboxing overhead when working with primitives.`,
    whyItMatters: `Built-in functional interfaces are the building blocks of functional programming in Java. They're used extensively in Stream API, Optional, CompletableFuture, and modern Java frameworks. Mastering these interfaces enables you to write expressive, functional code and understand how Java's functional APIs work. They're essential for modern Java development.

**Production Pattern:**

\`\`\`java
// Data validation pipeline with Predicate
Predicate<User> isActive = user -> user.getStatus() == Status.ACTIVE;
Predicate<User> hasEmail = user -> user.getEmail() != null;
Predicate<User> isVerified = user -> user.isEmailVerified();

List<User> validUsers = users.stream()
    .filter(isActive.and(hasEmail).and(isVerified))
    .collect(Collectors.toList());

// Data transformation with Function
Function<Order, OrderDTO> toDTO = order -> new OrderDTO(order.getId(), order.getTotal());
Function<OrderDTO, String> toJson = dto -> objectMapper.writeValueAsString(dto);
Function<Order, String> orderToJson = toDTO.andThen(toJson);

// Event handling with Consumer
Consumer<Event> logger = event -> log.info("Event: {}", event);
Consumer<Event> metrics = event -> metricsService.recordEvent(event);
Consumer<Event> eventHandler = logger.andThen(metrics);
events.forEach(eventHandler);

// Lazy initialization with Supplier
Supplier<DatabaseConnection> dbSupplier = () -> createConnection();
DatabaseConnection db = dbSupplier.get(); // Created only when needed
\`\`\`

**Practical Benefits:**

1. **Standardization**: Everyone uses same interfaces - better team collaboration
2. **Composability**: Combine simple functions into complex operations
3. **Framework Integration**: Seamlessly works with Spring, Stream API, Optional
4. **No Reinventing**: Rich set of interfaces covers most use cases`,
    order: 4,
    testCode: `import static org.junit.Assert.*;
import org.junit.Test;
import java.util.function.*;

// Test1: Verify Function interface
class Test1 {
    @Test
    public void test() {
        Function<Integer, Integer> square = x -> x * x;
        assertEquals(Integer.valueOf(25), square.apply(5));
    }
}

// Test2: Verify Predicate interface
class Test2 {
    @Test
    public void test() {
        Predicate<Integer> isEven = n -> n % 2 == 0;
        assertTrue(isEven.test(8));
        assertFalse(isEven.test(7));
    }
}

// Test3: Verify Consumer interface outputs correctly
class Test3 {
    @Test
    public void test() {
        java.io.ByteArrayOutputStream out = new java.io.ByteArrayOutputStream();
        java.io.PrintStream originalOut = System.out;
        System.setOut(new java.io.PrintStream(out));
        try {
            Consumer<String> printer = s -> System.out.println(s);
            printer.accept("ConsumerTest");
            String output = out.toString().trim();
            assertEquals("ConsumerTest", output);
        } finally {
            System.setOut(originalOut);
        }
    }
}

// Test4: Verify Supplier interface
class Test4 {
    @Test
    public void test() {
        Supplier<String> uuidSupplier = () -> "UUID-" + System.currentTimeMillis();
        assertNotNull(uuidSupplier.get());
    }
}

// Test5: Verify BiFunction interface
class Test5 {
    @Test
    public void test() {
        BiFunction<Integer, Integer, Integer> max = (a, b) -> Math.max(a, b);
        assertEquals(Integer.valueOf(20), max.apply(10, 20));
    }
}

// Test6: Verify BiPredicate interface
class Test6 {
    @Test
    public void test() {
        BiPredicate<String, String> areEqual = (s1, s2) -> s1.equals(s2);
        assertTrue(areEqual.test("hello", "hello"));
        assertFalse(areEqual.test("hello", "world"));
    }
}

// Test7: Verify BiConsumer interface outputs correctly
class Test7 {
    @Test
    public void test() {
        java.io.ByteArrayOutputStream out = new java.io.ByteArrayOutputStream();
        java.io.PrintStream originalOut = System.out;
        System.setOut(new java.io.PrintStream(out));
        try {
            BiConsumer<String, String> printer = (k, v) -> System.out.println(k + "=" + v);
            printer.accept("name", "John");
            String output = out.toString().trim();
            assertEquals("name=John", output);
        } finally {
            System.setOut(originalOut);
        }
    }
}

// Test8: Verify Function andThen composition
class Test8 {
    @Test
    public void test() {
        Function<Integer, Integer> square = x -> x * x;
        Function<Integer, Integer> addTwo = x -> x + 2;
        Function<Integer, Integer> composed = square.andThen(addTwo);
        assertEquals(Integer.valueOf(11), composed.apply(3)); // 3*3=9, 9+2=11
    }
}

// Test9: Verify Predicate and() combination
class Test9 {
    @Test
    public void test() {
        Predicate<Integer> isEven = n -> n % 2 == 0;
        Predicate<Integer> isGreaterThan5 = n -> n > 5;
        Predicate<Integer> combined = isEven.and(isGreaterThan5);
        assertTrue(combined.test(10));
        assertFalse(combined.test(4));
    }
}

// Test10: Verify Predicate negate()
class Test10 {
    @Test
    public void test() {
        Predicate<Integer> isEven = n -> n % 2 == 0;
        Predicate<Integer> isOdd = isEven.negate();
        assertTrue(isOdd.test(3));
        assertFalse(isOdd.test(4));
    }
}`,
    translations: {
        ru: {
            title: 'Встроенные функциональные интерфейсы',
            solutionCode: `import java.util.function.*;
import java.util.UUID;

public class BuiltInInterfaces {
    public static void main(String[] args) {
        System.out.println("=== Основные функциональные интерфейсы ===");

        // Function<T, R> - преобразование данных
        Function<Integer, Integer> square = x -> x * x;
        System.out.println("Function: 5 -> " + square.apply(5));

        // Predicate<T> - проверка условия
        Predicate<Integer> isEven = n -> n % 2 == 0;
        System.out.println("Predicate: Is 8 even? " + isEven.test(8));

        // Consumer<T> - потребление/обработка данных
        Consumer<String> printWithPrefix = s ->
            System.out.println("Consumer: Processing: " + s);
        printWithPrefix.accept("Test Data");

        // Supplier<T> - предоставление/генерация данных
        Supplier<String> uuidSupplier = () -> UUID.randomUUID().toString().substring(0, 13);
        System.out.println("Supplier: Generated UUID: " + uuidSupplier.get());

        System.out.println("\\n=== Бинарные функциональные интерфейсы ===");

        // BiFunction<T, U, R> - два входа, один выход
        BiFunction<Integer, Integer, Integer> max = (a, b) -> Math.max(a, b);
        System.out.println("BiFunction: max(10, 20) = " + max.apply(10, 20));

        // BiPredicate<T, U> - два входа, boolean выход
        BiPredicate<String, String> areEqual = (s1, s2) -> s1.equals(s2);
        System.out.println("BiPredicate: Are \\"hello\\" and \\"hello\\" equal? " +
            areEqual.test("hello", "hello"));

        // BiConsumer<T, U> - два входа, void выход
        BiConsumer<String, String> printKeyValue = (key, value) ->
            System.out.println("BiConsumer: Key=" + key + ", Value=" + value);
        printKeyValue.accept("name", "John");

        System.out.println("\\n=== Специализированные интерфейсы ===");

        // IntFunction<R> - int вход, R выход
        IntFunction<String> squareAsString = n -> "Square of " + n + " = " + (n * n);
        System.out.println("IntFunction: " + squareAsString.apply(7));

        // ToIntFunction<T> - T вход, int выход
        ToIntFunction<String> stringLength = s -> s.length();
        System.out.println("ToIntFunction: Length of \\"Java\\" = " +
            stringLength.applyAsInt("Java"));

        // IntPredicate - int вход, boolean выход
        IntPredicate isPositive = n -> n > 0;
        System.out.println("IntPredicate: Is 15 positive? " + isPositive.test(15));

        System.out.println("\\n=== Композиция и цепочки ===");

        // Композиция функций с andThen
        Function<Integer, Integer> multiplyBy3 = x -> x * 3;
        Function<Integer, Integer> multiplyBy2 = x -> x * 2;
        Function<Integer, Integer> chained = square.andThen(multiplyBy2);
        System.out.println("Chained function: 3 -> " + chained.apply(3));

        // Комбинирование предикатов с and/or
        Predicate<Integer> isGreaterThan5 = n -> n > 5;
        Predicate<Integer> isEvenAndGreaterThan5 = isEven.and(isGreaterThan5);
        System.out.println("Combined predicate: Is 10 even and > 5? " +
            isEvenAndGreaterThan5.test(10));

        // Отрицание предиката
        Predicate<Integer> isOdd = isEven.negate();
        System.out.println("Negated predicate: Is 3 not even? " + isOdd.test(3));
    }
}`,
            description: `# Встроенные функциональные интерфейсы

Java предоставляет богатый набор функциональных интерфейсов в пакете java.util.function. Эти интерфейсы охватывают распространенные случаи использования и могут использоваться с лямбда-выражениями и ссылками на методы. Понимание этих интерфейсов необходимо для работы с потоками и функциональным программированием.

## Требования:
1. Основные функциональные интерфейсы:
   1.1. \`Function<T, R>\`: Принимает T, возвращает R - преобразовать данные
   1.2. \`Predicate<T>\`: Принимает T, возвращает boolean - проверить условие
   1.3. \`Consumer<T>\`: Принимает T, возвращает void - потребить/обработать
   1.4. \`Supplier<T>\`: Ничего не принимает, возвращает T - предоставить/сгенерировать

2. Бинарные варианты (два параметра):
   2.1. \`BiFunction<T, U, R>\`: Принимает T и U, возвращает R
   2.2. \`BiPredicate<T, U>\`: Принимает T и U, возвращает boolean
   2.3. \`BiConsumer<T, U>\`: Принимает T и U, возвращает void

3. Специализированные примитивные интерфейсы:
   3.1. \`IntFunction<R>\`, \`ToIntFunction<T>\`: Работают с int
   3.2. \`IntPredicate\`, \`IntConsumer\`, \`IntSupplier\`
   3.3. Аналогично для Long и Double

4. Продемонстрируйте цепочки и композицию:
   4.1. Композиция функций: \`andThen()\`, \`compose()\`
   4.2. Логика предикатов: \`and()\`, \`or()\`, \`negate()\`

## Пример вывода:
\`\`\`
=== Core Functional Interfaces ===
Function: 5 -> 25
Predicate: Is 8 even? true
Consumer: Processing: Test Data
Supplier: Generated UUID: 123e4567-e89b

=== Binary Functional Interfaces ===
BiFunction: max(10, 20) = 20
BiPredicate: Are "hello" and "hello" equal? true
BiConsumer: Key=name, Value=John

=== Specialized Interfaces ===
IntFunction: Square of 7 = 49
ToIntFunction: Length of "Java" = 4
IntPredicate: Is 15 positive? true

=== Composition and Chaining ===
Chained function: 3 -> 9 -> 18
Combined predicate: Is 10 even and > 5? true
Negated predicate: Is 3 not even? true
\`\`\``,
            hint1: `Основные функциональные интерфейсы следуют паттерну: Function (преобразование), Predicate (тестирование), Consumer (обработка), Supplier (генерация). Каждый имеет конкретную цель.`,
            hint2: `Используйте специализированные примитивные интерфейсы (IntFunction, LongPredicate и т.д.) для избежания накладных расходов на упаковку/распаковку при работе с примитивами.`,
            whyItMatters: `Встроенные функциональные интерфейсы являются строительными блоками функционального программирования в Java. Они широко используются в Stream API, Optional, CompletableFuture и современных Java-фреймворках. Освоение этих интерфейсов позволяет вам писать выразительный функциональный код и понимать, как работают функциональные API Java. Они необходимы для современной разработки на Java.

**Продакшен паттерн:**

\`\`\`java
// Конвейер валидации данных с Predicate
Predicate<User> isActive = user -> user.getStatus() == Status.ACTIVE;
Predicate<User> hasEmail = user -> user.getEmail() != null;
Predicate<User> isVerified = user -> user.isEmailVerified();

List<User> validUsers = users.stream()
    .filter(isActive.and(hasEmail).and(isVerified))
    .collect(Collectors.toList());

// Трансформация данных с Function
Function<Order, OrderDTO> toDTO = order -> new OrderDTO(order.getId(), order.getTotal());
Function<OrderDTO, String> toJson = dto -> objectMapper.writeValueAsString(dto);
Function<Order, String> orderToJson = toDTO.andThen(toJson);

// Обработка событий с Consumer
Consumer<Event> logger = event -> log.info("Event: {}", event);
Consumer<Event> emailSender = event -> sendNotification(event);
Consumer<Event> pipeline = logger.andThen(emailSender);
\`\`\`

**Практические преимущества:**
- Стандартные интерфейсы обеспечивают совместимость между библиотеками
- Композиция (and, andThen) позволяет создавать сложную логику из простых частей
- Примитивные специализации избегают накладных расходов на boxing/unboxing`
        },
        uz: {
            title: `O'rnatilgan funksional interfeyslar`,
            solutionCode: `import java.util.function.*;
import java.util.UUID;

public class BuiltInInterfaces {
    public static void main(String[] args) {
        System.out.println("=== Asosiy funksional interfeyslar ===");

        // Function<T, R> - ma'lumotlarni o'zgartirish
        Function<Integer, Integer> square = x -> x * x;
        System.out.println("Function: 5 -> " + square.apply(5));

        // Predicate<T> - shartni tekshirish
        Predicate<Integer> isEven = n -> n % 2 == 0;
        System.out.println("Predicate: Is 8 even? " + isEven.test(8));

        // Consumer<T> - ma'lumotlarni iste'mol qilish/qayta ishlash
        Consumer<String> printWithPrefix = s ->
            System.out.println("Consumer: Processing: " + s);
        printWithPrefix.accept("Test Data");

        // Supplier<T> - ma'lumotlarni taqdim etish/yaratish
        Supplier<String> uuidSupplier = () -> UUID.randomUUID().toString().substring(0, 13);
        System.out.println("Supplier: Generated UUID: " + uuidSupplier.get());

        System.out.println("\\n=== Ikkilik funksional interfeyslar ===");

        // BiFunction<T, U, R> - ikki kirish, bitta chiqish
        BiFunction<Integer, Integer, Integer> max = (a, b) -> Math.max(a, b);
        System.out.println("BiFunction: max(10, 20) = " + max.apply(10, 20));

        // BiPredicate<T, U> - ikki kirish, boolean chiqish
        BiPredicate<String, String> areEqual = (s1, s2) -> s1.equals(s2);
        System.out.println("BiPredicate: Are \\"hello\\" and \\"hello\\" equal? " +
            areEqual.test("hello", "hello"));

        // BiConsumer<T, U> - ikki kirish, void chiqish
        BiConsumer<String, String> printKeyValue = (key, value) ->
            System.out.println("BiConsumer: Key=" + key + ", Value=" + value);
        printKeyValue.accept("name", "John");

        System.out.println("\\n=== Maxsus interfeyslar ===");

        // IntFunction<R> - int kirish, R chiqish
        IntFunction<String> squareAsString = n -> "Square of " + n + " = " + (n * n);
        System.out.println("IntFunction: " + squareAsString.apply(7));

        // ToIntFunction<T> - T kirish, int chiqish
        ToIntFunction<String> stringLength = s -> s.length();
        System.out.println("ToIntFunction: Length of \\"Java\\" = " +
            stringLength.applyAsInt("Java"));

        // IntPredicate - int kirish, boolean chiqish
        IntPredicate isPositive = n -> n > 0;
        System.out.println("IntPredicate: Is 15 positive? " + isPositive.test(15));

        System.out.println("\\n=== Kompozitsiya va zanjirlar ===");

        // andThen bilan funksiyalar kompozitsiyasi
        Function<Integer, Integer> multiplyBy3 = x -> x * 3;
        Function<Integer, Integer> multiplyBy2 = x -> x * 2;
        Function<Integer, Integer> chained = square.andThen(multiplyBy2);
        System.out.println("Chained function: 3 -> " + chained.apply(3));

        // and/or bilan predikatlarni birlashtirish
        Predicate<Integer> isGreaterThan5 = n -> n > 5;
        Predicate<Integer> isEvenAndGreaterThan5 = isEven.and(isGreaterThan5);
        System.out.println("Combined predicate: Is 10 even and > 5? " +
            isEvenAndGreaterThan5.test(10));

        // Predikat inkor
        Predicate<Integer> isOdd = isEven.negate();
        System.out.println("Negated predicate: Is 3 not even? " + isOdd.test(3));
    }
}`,
            description: `# O'rnatilgan funksional interfeyslar

Java java.util.function paketida boy funksional interfeyslar to'plamini taqdim etadi. Bu interfeyslar umumiy foydalanish holatlarini qamrab oladi va lambda ifodalari va metod havolalari bilan ishlatilishi mumkin. Bu interfeyslarni tushunish oqimlar va funksional dasturlash bilan ishlash uchun zarurdir.

## Talablar:
1. Asosiy funksional interfeyslar:
   1.1. \`Function<T, R>\`: T oladi, R qaytaradi - ma'lumotlarni o'zgartirish
   1.2. \`Predicate<T>\`: T oladi, boolean qaytaradi - shartni tekshirish
   1.3. \`Consumer<T>\`: T oladi, void qaytaradi - iste'mol qilish/qayta ishlash
   1.4. \`Supplier<T>\`: Hech narsa olmaydi, T qaytaradi - taqdim etish/yaratish

2. Ikkilik variantlar (ikki parametr):
   2.1. \`BiFunction<T, U, R>\`: T va U oladi, R qaytaradi
   2.2. \`BiPredicate<T, U>\`: T va U oladi, boolean qaytaradi
   2.3. \`BiConsumer<T, U>\`: T va U oladi, void qaytaradi

3. Maxsus primitiv interfeyslar:
   3.1. \`IntFunction<R>\`, \`ToIntFunction<T>\`: int bilan ishlaydi
   3.2. \`IntPredicate\`, \`IntConsumer\`, \`IntSupplier\`
   3.3. Long va Double uchun o'xshash

4. Zanjirlar va kompozitsiyani namoyish eting:
   4.1. Funksiyalar kompozitsiyasi: \`andThen()\`, \`compose()\`
   4.2. Predikat mantigi: \`and()\`, \`or()\`, \`negate()\`

## Chiqish namunasi:
\`\`\`
=== Core Functional Interfaces ===
Function: 5 -> 25
Predicate: Is 8 even? true
Consumer: Processing: Test Data
Supplier: Generated UUID: 123e4567-e89b

=== Binary Functional Interfaces ===
BiFunction: max(10, 20) = 20
BiPredicate: Are "hello" and "hello" equal? true
BiConsumer: Key=name, Value=John

=== Specialized Interfaces ===
IntFunction: Square of 7 = 49
ToIntFunction: Length of "Java" = 4
IntPredicate: Is 15 positive? true

=== Composition and Chaining ===
Chained function: 3 -> 9 -> 18
Combined predicate: Is 10 even and > 5? true
Negated predicate: Is 3 not even? true
\`\`\``,
            hint1: `Asosiy funksional interfeyslar naqshga amal qiladi: Function (o'zgartirish), Predicate (tekshirish), Consumer (qayta ishlash), Supplier (yaratish). Har biri ma'lum maqsadga ega.`,
            hint2: `Primitivlar bilan ishlashda qadoqlash/ochish xarajatlaridan qochish uchun maxsus primitiv interfeyslardan (IntFunction, LongPredicate va boshqalar) foydalaning.`,
            whyItMatters: `O'rnatilgan funksional interfeyslar Java-da funksional dasturlashning qurilish bloklari hisoblanadi. Ular Stream API, Optional, CompletableFuture va zamonaviy Java frameworklarida keng qo'llaniladi. Bu interfeyslarni o'zlashtirish sizga ifodali funksional kod yozish va Java-ning funksional API-lari qanday ishlashini tushunish imkonini beradi. Ular zamonaviy Java ishlab chiqish uchun zarurdir.

**Ishlab chiqarish patterni:**

\`\`\`java
// Predicate bilan ma'lumotlarni tekshirish konveyeri
Predicate<User> isActive = user -> user.getStatus() == Status.ACTIVE;
Predicate<User> hasEmail = user -> user.getEmail() != null;
Predicate<User> isVerified = user -> user.isEmailVerified();

List<User> validUsers = users.stream()
    .filter(isActive.and(hasEmail).and(isVerified))
    .collect(Collectors.toList());

// Function bilan ma'lumotlarni transformatsiya qilish
Function<Order, OrderDTO> toDTO = order -> new OrderDTO(order.getId(), order.getTotal());
Function<OrderDTO, String> toJson = dto -> objectMapper.writeValueAsString(dto);
Function<Order, String> orderToJson = toDTO.andThen(toJson);

// Consumer bilan hodisalarni qayta ishlash
Consumer<Event> logger = event -> log.info("Event: {}", event);
Consumer<Event> emailSender = event -> sendNotification(event);
Consumer<Event> pipeline = logger.andThen(emailSender);
\`\`\`

**Amaliy foydalari:**
- Standart interfeyslar kutubxonalar o'rtasida moslashuvchanlikni ta'minlaydi
- Kompozitsiya (and, andThen) oddiy qismlardan murakkab mantiq yaratish imkonini beradi
- Primitiv spetsializatsiyalar boxing/unboxing xarajatlaridan qochadi`
        }
    }
};

export default task;
