import { Task } from '../../../../types';

export const task: Task = {
    slug: 'java-optional-transformation',
    title: 'Optional Transformation',
    difficulty: 'medium',
    tags: ['java', 'optional', 'java8', 'transformation', 'functional'],
    estimatedTime: '30m',
    isPremium: false,
    youtubeUrl: '',
    description: `# Optional Transformation

Optional provides powerful transformation methods that enable functional-style programming. These methods allow you to transform, filter, and chain operations on Optional values without explicit null checks. Understanding these transformations is key to writing clean, expressive code.

## Requirements:
1. Demonstrate \`map()\` for transformations:
   1.1. Transform value if present
   1.2. Chain multiple transformations
   1.3. Handle type conversions

2. Use \`flatMap()\` for nested Optionals:
   2.1. Flatten Optional<Optional<T>> to Optional<T>
   2.2. Chain operations that return Optional
   2.3. Avoid nested Optional structures

3. Apply \`filter()\` for conditional selection:
   3.1. Keep value only if it matches predicate
   3.2. Return empty Optional if predicate fails
   3.3. Combine with other transformations

4. Show practical examples:
   4.1. String transformations (uppercase, length, parsing)
   4.2. Object property extraction
   4.3. Complex transformation chains

## Example Output:
\`\`\`
=== Map Transformations ===
Original: Java
Uppercase: JAVA
Length: 4

=== FlatMap for Nested Optionals ===
Nested without flatMap: Optional[Optional[JAVA]]
Flattened with flatMap: Optional[JAVA]

=== Filter Operations ===
Long name (>5 chars): Optional[Python]
Short name (>5 chars): Optional.empty

=== Chaining Transformations ===
Original: "  hello world  "
Result: HELLO WORLD
Steps: trim -> uppercase -> check length
\`\`\``,
    initialCode: `// TODO: Import Optional

public class OptionalTransformation {
    public static void main(String[] args) {
        // TODO: Demonstrate map() transformations

        // TODO: Show flatMap() for nested Optionals

        // TODO: Use filter() for conditional selection

        // TODO: Chain multiple transformations
    }
}`,
    solutionCode: `import java.util.Optional;

public class OptionalTransformation {
    public static void main(String[] args) {
        System.out.println("=== Map Transformations ===");

        Optional<String> language = Optional.of("Java");

        // Transform value with map
        Optional<String> uppercase = language.map(String::toUpperCase);
        Optional<Integer> length = language.map(String::length);

        System.out.println("Original: " + language.orElse(""));
        System.out.println("Uppercase: " + uppercase.orElse(""));
        System.out.println("Length: " + length.orElse(0));

        // Map on empty Optional returns empty
        Optional<String> empty = Optional.empty();
        Optional<String> emptyResult = empty.map(String::toUpperCase);
        System.out.println("Map on empty: " + emptyResult);

        System.out.println("\\n=== FlatMap for Nested Optionals ===");

        // Problem with map: creates nested Optional
        Optional<Optional<String>> nested = language.map(s ->
            Optional.of(s.toUpperCase())
        );
        System.out.println("Nested without flatMap: " + nested);

        // Solution: use flatMap to flatten
        Optional<String> flattened = language.flatMap(s ->
            Optional.of(s.toUpperCase())
        );
        System.out.println("Flattened with flatMap: " + flattened);

        // Practical flatMap example
        Optional<String> result = findUser("john")
            .flatMap(user -> findEmail(user))
            .map(String::toLowerCase);
        System.out.println("User email: " + result.orElse("not found"));

        System.out.println("\\n=== Filter Operations ===");

        // Filter keeps value only if predicate is true
        Optional<String> longName = Optional.of("Python")
            .filter(s -> s.length() > 5);
        System.out.println("Long name (>5 chars): " + longName);

        Optional<String> shortName = Optional.of("Java")
            .filter(s -> s.length() > 5);
        System.out.println("Short name (>5 chars): " + shortName);

        // Filter with complex predicates
        Optional<Integer> evenNumber = Optional.of(42)
            .filter(n -> n % 2 == 0);
        System.out.println("Even number: " + evenNumber);

        System.out.println("\\n=== Chaining Transformations ===");

        // Complex transformation chain
        String input = "  hello world  ";
        Optional<String> processed = Optional.of(input)
            .map(String::trim)
            .filter(s -> !s.isEmpty())
            .map(String::toUpperCase)
            .filter(s -> s.length() > 5);

        System.out.println("Original: \\"" + input + "\\"");
        System.out.println("Result: " + processed.orElse("filtered out"));
        System.out.println("Steps: trim -> uppercase -> check length");

        System.out.println("\\n=== Practical Examples ===");

        // Parse string to integer safely
        Optional<Integer> number = parseInteger("123")
            .filter(n -> n > 0)
            .map(n -> n * 2);
        System.out.println("Parsed and doubled: " + number.orElse(0));

        // Extract and transform object property
        Optional<Person> person = Optional.of(new Person("Alice", 30));
        Optional<String> nameLength = person
            .map(Person::getName)
            .filter(name -> name.length() > 3)
            .map(name -> name + " (" + name.length() + " chars)");
        System.out.println("Person info: " + nameLength.orElse("N/A"));

        // Nested object navigation
        Optional<String> city = Optional.of(new Person("Bob", 25))
            .flatMap(Person::getAddress)
            .map(Address::getCity)
            .map(String::toUpperCase);
        System.out.println("City: " + city.orElse("Unknown"));
    }

    private static Optional<String> findUser(String username) {
        return Optional.of("john@example.com");
    }

    private static Optional<String> findEmail(String user) {
        return Optional.of(user);
    }

    private static Optional<Integer> parseInteger(String str) {
        try {
            return Optional.of(Integer.parseInt(str));
        } catch (NumberFormatException e) {
            return Optional.empty();
        }
    }

    static class Person {
        private String name;
        private int age;

        public Person(String name, int age) {
            this.name = name;
            this.age = age;
        }

        public String getName() {
            return name;
        }

        public Optional<Address> getAddress() {
            return Optional.of(new Address("New York"));
        }
    }

    static class Address {
        private String city;

        public Address(String city) {
            this.city = city;
        }

        public String getCity() {
            return city;
        }
    }
}`,
    hint1: `Use map() for simple transformations that return a value. Use flatMap() when your transformation returns an Optional, to avoid Optional<Optional<T>>.`,
    hint2: `filter() returns the same Optional if the predicate is true, or empty Optional if false. It's perfect for conditional checks in transformation chains.`,
    whyItMatters: `Transformation methods enable functional programming patterns and eliminate nested null checks. They make code more readable, composable, and less error-prone. Understanding map, flatMap, and filter is essential for working with Optional effectively and writing modern Java code.

**Production Pattern:**
\`\`\`java
// Safe string parsing
Optional<Integer> port = config.get("port")
    .map(String::trim)
    .filter(s -> !s.isEmpty())
    .flatMap(this::parseInt)
    .filter(p -> p > 0 && p < 65536);

// Extract nested property with transformation
String department = employee
    .flatMap(Employee::getManager)
    .map(Manager::getDepartment)
    .map(Department::getName)
    .map(String::toUpperCase)
    .orElse("UNKNOWN");

// Validation chain
Optional<Email> validEmail = parseEmail(input)
    .filter(Email::isValid)
    .filter(e -> !e.isDuplicate())
    .map(Email::normalize);
\`\`\`

**Practical Benefits:**
- Eliminates nested null checks and if-statements
- Composable transformations create readable pipelines
- Type-safe navigation through object graphs`,
    order: 3,
    testCode: `import static org.junit.Assert.*;
import org.junit.Test;
import java.io.ByteArrayOutputStream;
import java.io.PrintStream;

// Test1: main method should produce output
class Test1 {
    @Test
    public void test() {
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        PrintStream oldOut = System.out;
        System.setOut(new PrintStream(out));
        OptionalTransformation.main(new String[]{});
        System.setOut(oldOut);
        String output = out.toString();
        assertTrue("Should show Optional transformation demo",
            output.contains("Optional") || output.contains("map") ||
            output.contains("Transformation") || output.contains("Преобразование"));
    }
}

// Test2: should demonstrate map transformation
class Test2 {
    @Test
    public void test() {
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        PrintStream oldOut = System.out;
        System.setOut(new PrintStream(out));
        OptionalTransformation.main(new String[]{});
        System.setOut(oldOut);
        String output = out.toString();
        assertTrue("Should show map transformation", output.contains("Uppercase") || output.contains("JAVA"));
    }
}

// Test3: should show length transformation
class Test3 {
    @Test
    public void test() {
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        PrintStream oldOut = System.out;
        System.setOut(new PrintStream(out));
        OptionalTransformation.main(new String[]{});
        System.setOut(oldOut);
        String output = out.toString();
        assertTrue("Should show length transformation", output.contains("Length") || output.contains("4"));
    }
}

// Test4: should demonstrate flatMap
class Test4 {
    @Test
    public void test() {
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        PrintStream oldOut = System.out;
        System.setOut(new PrintStream(out));
        OptionalTransformation.main(new String[]{});
        System.setOut(oldOut);
        String output = out.toString();
        assertTrue("Should show flatMap", output.contains("flatMap") || output.contains("Flatten"));
    }
}

// Test5: should show nested Optional problem
class Test5 {
    @Test
    public void test() {
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        PrintStream oldOut = System.out;
        System.setOut(new PrintStream(out));
        OptionalTransformation.main(new String[]{});
        System.setOut(oldOut);
        String output = out.toString();
        assertTrue("Should show nested Optional", output.contains("Nested") || output.contains("Optional[Optional"));
    }
}

// Test6: should demonstrate filter operations
class Test6 {
    @Test
    public void test() {
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        PrintStream oldOut = System.out;
        System.setOut(new PrintStream(out));
        OptionalTransformation.main(new String[]{});
        System.setOut(oldOut);
        String output = out.toString();
        assertTrue("Should show filter", output.contains("Filter") || output.contains("Long name") || output.contains("chars"));
    }
}

// Test7: should show filter with empty result
class Test7 {
    @Test
    public void test() {
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        PrintStream oldOut = System.out;
        System.setOut(new PrintStream(out));
        OptionalTransformation.main(new String[]{});
        System.setOut(oldOut);
        String output = out.toString();
        assertTrue("Should show empty filter result", output.contains("Optional.empty") || output.contains("Short name"));
    }
}

// Test8: should demonstrate chaining
class Test8 {
    @Test
    public void test() {
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        PrintStream oldOut = System.out;
        System.setOut(new PrintStream(out));
        OptionalTransformation.main(new String[]{});
        System.setOut(oldOut);
        String output = out.toString();
        assertTrue("Should show chaining", output.contains("Chaining") || output.contains("Steps") || output.contains("trim"));
    }
}

// Test9: should show practical examples
class Test9 {
    @Test
    public void test() {
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        PrintStream oldOut = System.out;
        System.setOut(new PrintStream(out));
        OptionalTransformation.main(new String[]{});
        System.setOut(oldOut);
        String output = out.toString();
        assertTrue("Should show practical examples", output.contains("Practical") || output.contains("Person") || output.contains("City"));
    }
}

// Test10: should have section headers
class Test10 {
    @Test
    public void test() {
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        PrintStream oldOut = System.out;
        System.setOut(new PrintStream(out));
        OptionalTransformation.main(new String[]{});
        System.setOut(oldOut);
        String output = out.toString();
        boolean hasHeaders = output.contains("===") ||
                             output.contains("Map") || output.contains("Трансформаци") || output.contains("transformatsiya");
        assertTrue("Should have section headers", hasHeaders);
    }
}`,
    translations: {
        ru: {
            title: 'Трансформация Optional',
            solutionCode: `import java.util.Optional;

public class OptionalTransformation {
    public static void main(String[] args) {
        System.out.println("=== Трансформации с map ===");

        Optional<String> language = Optional.of("Java");

        // Трансформация значения с помощью map
        Optional<String> uppercase = language.map(String::toUpperCase);
        Optional<Integer> length = language.map(String::length);

        System.out.println("Original: " + language.orElse(""));
        System.out.println("Uppercase: " + uppercase.orElse(""));
        System.out.println("Length: " + length.orElse(0));

        // Map на пустом Optional возвращает пустой
        Optional<String> empty = Optional.empty();
        Optional<String> emptyResult = empty.map(String::toUpperCase);
        System.out.println("Map on empty: " + emptyResult);

        System.out.println("\\n=== FlatMap для вложенных Optional ===");

        // Проблема с map: создает вложенный Optional
        Optional<Optional<String>> nested = language.map(s ->
            Optional.of(s.toUpperCase())
        );
        System.out.println("Nested without flatMap: " + nested);

        // Решение: использовать flatMap для выравнивания
        Optional<String> flattened = language.flatMap(s ->
            Optional.of(s.toUpperCase())
        );
        System.out.println("Flattened with flatMap: " + flattened);

        // Практический пример flatMap
        Optional<String> result = findUser("john")
            .flatMap(user -> findEmail(user))
            .map(String::toLowerCase);
        System.out.println("User email: " + result.orElse("not found"));

        System.out.println("\\n=== Операции фильтрации ===");

        // Filter сохраняет значение только если предикат истинен
        Optional<String> longName = Optional.of("Python")
            .filter(s -> s.length() > 5);
        System.out.println("Long name (>5 chars): " + longName);

        Optional<String> shortName = Optional.of("Java")
            .filter(s -> s.length() > 5);
        System.out.println("Short name (>5 chars): " + shortName);

        // Filter со сложными предикатами
        Optional<Integer> evenNumber = Optional.of(42)
            .filter(n -> n % 2 == 0);
        System.out.println("Even number: " + evenNumber);

        System.out.println("\\n=== Цепочки трансформаций ===");

        // Сложная цепочка трансформаций
        String input = "  hello world  ";
        Optional<String> processed = Optional.of(input)
            .map(String::trim)
            .filter(s -> !s.isEmpty())
            .map(String::toUpperCase)
            .filter(s -> s.length() > 5);

        System.out.println("Original: \\"" + input + "\\"");
        System.out.println("Result: " + processed.orElse("filtered out"));
        System.out.println("Steps: trim -> uppercase -> check length");

        System.out.println("\\n=== Практические примеры ===");

        // Безопасный парсинг строки в целое число
        Optional<Integer> number = parseInteger("123")
            .filter(n -> n > 0)
            .map(n -> n * 2);
        System.out.println("Parsed and doubled: " + number.orElse(0));

        // Извлечение и трансформация свойства объекта
        Optional<Person> person = Optional.of(new Person("Alice", 30));
        Optional<String> nameLength = person
            .map(Person::getName)
            .filter(name -> name.length() > 3)
            .map(name -> name + " (" + name.length() + " chars)");
        System.out.println("Person info: " + nameLength.orElse("N/A"));

        // Навигация по вложенным объектам
        Optional<String> city = Optional.of(new Person("Bob", 25))
            .flatMap(Person::getAddress)
            .map(Address::getCity)
            .map(String::toUpperCase);
        System.out.println("City: " + city.orElse("Unknown"));
    }

    private static Optional<String> findUser(String username) {
        return Optional.of("john@example.com");
    }

    private static Optional<String> findEmail(String user) {
        return Optional.of(user);
    }

    private static Optional<Integer> parseInteger(String str) {
        try {
            return Optional.of(Integer.parseInt(str));
        } catch (NumberFormatException e) {
            return Optional.empty();
        }
    }

    static class Person {
        private String name;
        private int age;

        public Person(String name, int age) {
            this.name = name;
            this.age = age;
        }

        public String getName() {
            return name;
        }

        public Optional<Address> getAddress() {
            return Optional.of(new Address("New York"));
        }
    }

    static class Address {
        private String city;

        public Address(String city) {
            this.city = city;
        }

        public String getCity() {
            return city;
        }
    }
}`,
            description: `# Трансформация Optional

Optional предоставляет мощные методы трансформации, которые обеспечивают функциональный стиль программирования. Эти методы позволяют трансформировать, фильтровать и объединять операции над значениями Optional без явных проверок на null. Понимание этих трансформаций является ключом к написанию чистого, выразительного кода.

## Требования:
1. Продемонстрируйте \`map()\` для трансформаций:
   1.1. Трансформируйте значение если присутствует
   1.2. Объедините несколько трансформаций
   1.3. Обработайте преобразования типов

2. Используйте \`flatMap()\` для вложенных Optional:
   2.1. Выровняйте Optional<Optional<T>> в Optional<T>
   2.2. Объедините операции, возвращающие Optional
   2.3. Избегайте вложенных структур Optional

3. Примените \`filter()\` для условного выбора:
   3.1. Сохраните значение только если оно соответствует предикату
   3.2. Верните пустой Optional если предикат не выполнен
   3.3. Комбинируйте с другими трансформациями

4. Покажите практические примеры:
   4.1. Трансформации строк (верхний регистр, длина, парсинг)
   4.2. Извлечение свойств объекта
   4.3. Сложные цепочки трансформаций

## Пример вывода:
\`\`\`
=== Map Transformations ===
Original: Java
Uppercase: JAVA
Length: 4

=== FlatMap for Nested Optionals ===
Nested without flatMap: Optional[Optional[JAVA]]
Flattened with flatMap: Optional[JAVA]

=== Filter Operations ===
Long name (>5 chars): Optional[Python]
Short name (>5 chars): Optional.empty

=== Chaining Transformations ===
Original: "  hello world  "
Result: HELLO WORLD
Steps: trim -> uppercase -> check length
\`\`\``,
            hint1: `Используйте map() для простых трансформаций, возвращающих значение. Используйте flatMap(), когда ваша трансформация возвращает Optional, чтобы избежать Optional<Optional<T>>.`,
            hint2: `filter() возвращает тот же Optional если предикат истинен, или пустой Optional если ложен. Он идеален для условных проверок в цепочках трансформаций.`,
            whyItMatters: `Методы трансформации обеспечивают паттерны функционального программирования и устраняют вложенные проверки на null. Они делают код более читаемым, композируемым и менее подверженным ошибкам. Понимание map, flatMap и filter необходимо для эффективной работы с Optional и написания современного Java-кода.

**Продакшен паттерн:**
\`\`\`java
// Безопасный парсинг строк
Optional<Integer> port = config.get("port")
    .map(String::trim)
    .filter(s -> !s.isEmpty())
    .flatMap(this::parseInt)
    .filter(p -> p > 0 && p < 65536);

// Извлечение вложенных свойств с трансформацией
String department = employee
    .flatMap(Employee::getManager)
    .map(Manager::getDepartment)
    .map(Department::getName)
    .map(String::toUpperCase)
    .orElse("UNKNOWN");

// Цепочка валидации
Optional<Email> validEmail = parseEmail(input)
    .filter(Email::isValid)
    .filter(e -> !e.isDuplicate())
    .map(Email::normalize);
\`\`\`

**Практические преимущества:**
- Устраняет вложенные проверки на null и if-операторы
- Композируемые трансформации создают читаемые конвейеры
- Типобезопасная навигация по графам объектов`
        },
        uz: {
            title: `Optional transformatsiyasi`,
            solutionCode: `import java.util.Optional;

public class OptionalTransformation {
    public static void main(String[] args) {
        System.out.println("=== Map transformatsiyalari ===");

        Optional<String> language = Optional.of("Java");

        // map yordamida qiymatni transformatsiya qilish
        Optional<String> uppercase = language.map(String::toUpperCase);
        Optional<Integer> length = language.map(String::length);

        System.out.println("Original: " + language.orElse(""));
        System.out.println("Uppercase: " + uppercase.orElse(""));
        System.out.println("Length: " + length.orElse(0));

        // Bo'sh Optional da map bo'sh qaytaradi
        Optional<String> empty = Optional.empty();
        Optional<String> emptyResult = empty.map(String::toUpperCase);
        System.out.println("Map on empty: " + emptyResult);

        System.out.println("\\n=== Ichki Optional lar uchun FlatMap ===");

        // map bilan muammo: ichki Optional yaratadi
        Optional<Optional<String>> nested = language.map(s ->
            Optional.of(s.toUpperCase())
        );
        System.out.println("Nested without flatMap: " + nested);

        // Yechim: tekislash uchun flatMap dan foydalanish
        Optional<String> flattened = language.flatMap(s ->
            Optional.of(s.toUpperCase())
        );
        System.out.println("Flattened with flatMap: " + flattened);

        // Amaliy flatMap misoli
        Optional<String> result = findUser("john")
            .flatMap(user -> findEmail(user))
            .map(String::toLowerCase);
        System.out.println("User email: " + result.orElse("not found"));

        System.out.println("\\n=== Filtrlash operatsiyalari ===");

        // Filter faqat predikat to'g'ri bo'lsa qiymatni saqlaydi
        Optional<String> longName = Optional.of("Python")
            .filter(s -> s.length() > 5);
        System.out.println("Long name (>5 chars): " + longName);

        Optional<String> shortName = Optional.of("Java")
            .filter(s -> s.length() > 5);
        System.out.println("Short name (>5 chars): " + shortName);

        // Murakkab predikatlar bilan filter
        Optional<Integer> evenNumber = Optional.of(42)
            .filter(n -> n % 2 == 0);
        System.out.println("Even number: " + evenNumber);

        System.out.println("\\n=== Transformatsiyalar zanjiri ===");

        // Murakkab transformatsiya zanjiri
        String input = "  hello world  ";
        Optional<String> processed = Optional.of(input)
            .map(String::trim)
            .filter(s -> !s.isEmpty())
            .map(String::toUpperCase)
            .filter(s -> s.length() > 5);

        System.out.println("Original: \\"" + input + "\\"");
        System.out.println("Result: " + processed.orElse("filtered out"));
        System.out.println("Steps: trim -> uppercase -> check length");

        System.out.println("\\n=== Amaliy misollar ===");

        // Satrni xavfsiz tarzda butun songa o'girish
        Optional<Integer> number = parseInteger("123")
            .filter(n -> n > 0)
            .map(n -> n * 2);
        System.out.println("Parsed and doubled: " + number.orElse(0));

        // Obyekt xususiyatini ajratib olish va transformatsiya qilish
        Optional<Person> person = Optional.of(new Person("Alice", 30));
        Optional<String> nameLength = person
            .map(Person::getName)
            .filter(name -> name.length() > 3)
            .map(name -> name + " (" + name.length() + " chars)");
        System.out.println("Person info: " + nameLength.orElse("N/A"));

        // Ichki obyektlar bo'ylab navigatsiya
        Optional<String> city = Optional.of(new Person("Bob", 25))
            .flatMap(Person::getAddress)
            .map(Address::getCity)
            .map(String::toUpperCase);
        System.out.println("City: " + city.orElse("Unknown"));
    }

    private static Optional<String> findUser(String username) {
        return Optional.of("john@example.com");
    }

    private static Optional<String> findEmail(String user) {
        return Optional.of(user);
    }

    private static Optional<Integer> parseInteger(String str) {
        try {
            return Optional.of(Integer.parseInt(str));
        } catch (NumberFormatException e) {
            return Optional.empty();
        }
    }

    static class Person {
        private String name;
        private int age;

        public Person(String name, int age) {
            this.name = name;
            this.age = age;
        }

        public String getName() {
            return name;
        }

        public Optional<Address> getAddress() {
            return Optional.of(new Address("New York"));
        }
    }

    static class Address {
        private String city;

        public Address(String city) {
            this.city = city;
        }

        public String getCity() {
            return city;
        }
    }
}`,
            description: `# Optional transformatsiyasi

Optional funksional dasturlash uslubini ta'minlaydigan kuchli transformatsiya metodlarini taqdim etadi. Bu metodlar aniq null tekshiruvisiz Optional qiymatlari ustida transformatsiya, filtrlash va zanjir operatsiyalarini amalga oshirishga imkon beradi. Bu transformatsiyalarni tushunish toza, ifodali kod yozish kaliti hisoblanadi.

## Talablar:
1. Transformatsiyalar uchun \`map()\` ni namoyish eting:
   1.1. Mavjud bo'lsa qiymatni transformatsiya qiling
   1.2. Bir nechta transformatsiyalarni zanjirlang
   1.3. Tur o'zgartirishlarini boshqaring

2. Ichki Optionallar uchun \`flatMap()\` dan foydalaning:
   2.1. Optional<Optional<T>> ni Optional<T> ga tekislang
   2.2. Optional qaytaradigan operatsiyalarni zanjirlang
   2.3. Ichki Optional strukturalaridan qoching

3. Shartli tanlash uchun \`filter()\` ni qo'llang:
   3.1. Faqat predikatga mos kelsa qiymatni saqlang
   3.2. Predikat muvaffaqiyatsiz bo'lsa bo'sh Optional qaytaring
   3.3. Boshqa transformatsiyalar bilan birlashtiring

4. Amaliy misollarni ko'rsating:
   4.1. Satr transformatsiyalari (katta harf, uzunlik, parsing)
   4.2. Obyekt xususiyatini ajratib olish
   4.3. Murakkab transformatsiya zanjirlari

## Chiqish namunasi:
\`\`\`
=== Map Transformations ===
Original: Java
Uppercase: JAVA
Length: 4

=== FlatMap for Nested Optionals ===
Nested without flatMap: Optional[Optional[JAVA]]
Flattened with flatMap: Optional[JAVA]

=== Filter Operations ===
Long name (>5 chars): Optional[Python]
Short name (>5 chars): Optional.empty

=== Chaining Transformations ===
Original: "  hello world  "
Result: HELLO WORLD
Steps: trim -> uppercase -> check length
\`\`\``,
            hint1: `Qiymat qaytaradigan oddiy transformatsiyalar uchun map() dan foydalaning. Transformatsiyangiz Optional qaytarganda Optional<Optional<T>> dan qochish uchun flatMap() dan foydalaning.`,
            hint2: `filter() predikat to'g'ri bo'lsa bir xil Optional ni qaytaradi, yolg'on bo'lsa bo'sh Optional ni. Bu transformatsiya zanjirlaridagi shartli tekshiruvlar uchun ideal.`,
            whyItMatters: `Transformatsiya metodlari funksional dasturlash naqshlarini ta'minlaydi va ichki null tekshiruvlarini yo'q qiladi. Ular kodni yanada o'qilishi oson, kompozitsion va xatolarga kam moyil qiladi. map, flatMap va filter ni tushunish Optional bilan samarali ishlash va zamonaviy Java kodi yozish uchun zarurdir.

**Ishlab chiqarish patterni:**
\`\`\`java
// Xavfsiz satr parsing
Optional<Integer> port = config.get("port")
    .map(String::trim)
    .filter(s -> !s.isEmpty())
    .flatMap(this::parseInt)
    .filter(p -> p > 0 && p < 65536);

// Transformatsiya bilan ichki xususiyatlarni ajratib olish
String department = employee
    .flatMap(Employee::getManager)
    .map(Manager::getDepartment)
    .map(Department::getName)
    .map(String::toUpperCase)
    .orElse("UNKNOWN");

// Validatsiya zanjiri
Optional<Email> validEmail = parseEmail(input)
    .filter(Email::isValid)
    .filter(e -> !e.isDuplicate())
    .map(Email::normalize);
\`\`\`

**Amaliy foydalari:**
- Ichki null tekshiruvlari va if-operatorlarni yo'q qiladi
- Kompozitsion transformatsiyalar o'qilishi oson konveyerlar yaratadi
- Obyektlar grafigi bo'ylab tur-xavfsiz navigatsiya`
        }
    }
};

export default task;
