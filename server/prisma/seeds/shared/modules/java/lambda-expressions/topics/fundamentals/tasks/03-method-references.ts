import { Task } from '../../../../types';

export const task: Task = {
    slug: 'java-method-references',
    title: 'Method References',
    difficulty: 'medium',
    tags: ['java', 'method-reference', 'lambda', 'java8', 'functional-programming'],
    estimatedTime: '30m',
    isPremium: false,
    youtubeUrl: '',
    description: `# Method References

Method references are shorthand notation for lambda expressions that call a single method. They make code more concise and readable when the lambda simply calls an existing method. There are four types of method references in Java.

## Requirements:
1. Demonstrate four types of method references:
   1.1. Static method reference: \`ClassName::staticMethod\`
   1.2. Instance method of particular object: \`object::instanceMethod\`
   1.3. Instance method of arbitrary object: \`ClassName::instanceMethod\`
   1.4. Constructor reference: \`ClassName::new\`

2. Compare lambda expressions with equivalent method references:
   2.1. Show both forms side by side
   2.2. Demonstrate when method references can be used

3. Create practical examples:
   3.1. String operations (toUpperCase, length)
   3.2. Math operations (max, abs)
   3.3. Object creation (constructor references)
   3.4. Collection operations

4. Show array sorting using method references

## Example Output:
\`\`\`
=== Static Method Reference ===
Lambda: 25
Method Reference: 25

=== Instance Method Reference (Particular Object) ===
Lambda: HELLO
Method Reference: HELLO

=== Instance Method Reference (Arbitrary Object) ===
Lambda: [APPLE, BANANA, CHERRY]
Method Reference: [APPLE, BANANA, CHERRY]

=== Constructor Reference ===
Lambda: Person(John)
Method Reference: Person(Alice)

=== Sorting with Method References ===
Natural order: [1, 2, 3, 4, 5]
Reverse order: [5, 4, 3, 2, 1]
\`\`\``,
    initialCode: `import java.util.*;
import java.util.function.*;

public class MethodReferences {
    // TODO: Create helper methods for demonstrations

    public static void main(String[] args) {
        // TODO: Demonstrate static method reference

        // TODO: Demonstrate instance method reference (particular object)

        // TODO: Demonstrate instance method reference (arbitrary object)

        // TODO: Demonstrate constructor reference

        // TODO: Sort collection using method references
    }
}`,
    solutionCode: `import java.util.*;
import java.util.function.*;

class Person {
    private String name;

    public Person(String name) {
        this.name = name;
    }

    @Override
    public String toString() {
        return "Person(" + name + ")";
    }
}

public class MethodReferences {
    // Static method for demonstration
    public static int square(int n) {
        return n * n;
    }

    // Instance method for demonstration
    public String convertToUpperCase(String s) {
        return s.toUpperCase();
    }

    public static void main(String[] args) {
        System.out.println("=== Static Method Reference ===");
        // Lambda expression
        Function<Integer, Integer> squareLambda = n -> square(n);
        System.out.println("Lambda: " + squareLambda.apply(5));

        // Method reference (equivalent)
        Function<Integer, Integer> squareMethodRef = MethodReferences::square;
        System.out.println("Method Reference: " + squareMethodRef.apply(5));

        System.out.println("\\n=== Instance Method Reference (Particular Object) ===");
        MethodReferences obj = new MethodReferences();

        // Lambda expression
        Function<String, String> upperLambda = s -> obj.convertToUpperCase(s);
        System.out.println("Lambda: " + upperLambda.apply("hello"));

        // Method reference (particular object)
        Function<String, String> upperMethodRef = obj::convertToUpperCase;
        System.out.println("Method Reference: " + upperMethodRef.apply("hello"));

        System.out.println("\\n=== Instance Method Reference (Arbitrary Object) ===");
        List<String> words = Arrays.asList("apple", "banana", "cherry");

        // Lambda expression
        List<String> upperWordsLambda = new ArrayList<>();
        words.forEach(w -> upperWordsLambda.add(w.toUpperCase()));
        System.out.println("Lambda: " + upperWordsLambda);

        // Method reference (arbitrary object of type String)
        List<String> upperWordsMethodRef = new ArrayList<>();
        words.stream()
            .map(String::toUpperCase)
            .forEach(upperWordsMethodRef::add);
        System.out.println("Method Reference: " + upperWordsMethodRef);

        System.out.println("\\n=== Constructor Reference ===");
        // Lambda expression
        Function<String, Person> personLambda = name -> new Person(name);
        System.out.println("Lambda: " + personLambda.apply("John"));

        // Constructor reference
        Function<String, Person> personMethodRef = Person::new;
        System.out.println("Method Reference: " + personMethodRef.apply("Alice"));

        System.out.println("\\n=== Sorting with Method References ===");
        List<Integer> numbers = Arrays.asList(3, 1, 4, 1, 5, 9, 2, 6);

        // Natural order using method reference
        List<Integer> sorted = new ArrayList<>(numbers);
        sorted.sort(Integer::compareTo);
        System.out.println("Natural order: " + sorted);

        // Reverse order using method reference
        List<Integer> reversed = new ArrayList<>(numbers);
        reversed.sort(Comparator.reverseOrder());
        System.out.println("Reverse order: " + reversed);
    }
}`,
    hint1: `Method references have four forms: ClassName::staticMethod, object::instanceMethod, ClassName::instanceMethod, ClassName::new. Choose based on what the lambda does.`,
    hint2: `Instance method reference of arbitrary object (String::toUpperCase) works when the lambda passes the parameter to an instance method of that parameter's type.`,
    whyItMatters: `Method references make functional code more concise and readable. They're especially useful with Stream API and collection operations. Understanding when to use method references versus lambda expressions helps write cleaner, more maintainable code. They're a key feature of modern Java programming style.

**Production Pattern:**

\`\`\`java
// Processing user list
List<User> users = userRepository.findAll();

// WITHOUT method references: Verbose
users.stream()
    .map(user -> user.getEmail())
    .filter(email -> email.contains("@company.com"))
    .forEach(email -> System.out.println(email));

// WITH method references: Readable and concise
users.stream()
    .map(User::getEmail)
    .filter(this::isCompanyEmail)
    .forEach(System.out::println);

// Sorting by multiple criteria
products.sort(Comparator.comparing(Product::getCategory)
    .thenComparing(Product::getPrice)
    .thenComparing(Product::getName));

// Creating objects through constructor
List<String> names = Arrays.asList("Alice", "Bob", "Charlie");
List<User> newUsers = names.stream()
    .map(User::new)  // Constructor reference
    .collect(Collectors.toList());
\`\`\`

**Practical Benefits:**

1. **Improved Readability**: Code looks like natural language (map User to Email)
2. **Less Boilerplate**: No need for lambda parameters and bodies
3. **Better IDE Support**: Refactoring automatically updates method references
4. **Reusability**: Method references encourage extracting logic to methods`,
    order: 3,
    testCode: `import static org.junit.Assert.*;
import org.junit.Test;
import java.util.function.*;

// Test1: Verify static method reference
class Test1 {
    @Test
    public void test() {
        Function<Integer, Integer> squareRef = MethodReferences::square;
        assertEquals(Integer.valueOf(25), squareRef.apply(5));
    }
}

// Test2: Verify instance method reference (particular object)
class Test2 {
    @Test
    public void test() {
        MethodReferences obj = new MethodReferences();
        Function<String, String> upperRef = obj::convertToUpperCase;
        assertEquals("HELLO", upperRef.apply("hello"));
    }
}

// Test3: Verify instance method reference (arbitrary object)
class Test3 {
    @Test
    public void test() {
        Function<String, String> upperRef = String::toUpperCase;
        assertEquals("JAVA", upperRef.apply("java"));
    }
}

// Test4: Verify constructor reference
class Test4 {
    @Test
    public void test() {
        Function<String, Person> personRef = Person::new;
        Person p = personRef.apply("Alice");
        assertNotNull(p);
    }
}

// Test5: Verify String::length method reference
class Test5 {
    @Test
    public void test() {
        Function<String, Integer> lenRef = String::length;
        assertEquals(Integer.valueOf(4), lenRef.apply("Java"));
    }
}

// Test6: Verify Integer::compareTo for sorting
class Test6 {
    @Test
    public void test() {
        java.util.Comparator<Integer> comp = Integer::compareTo;
        assertTrue(comp.compare(1, 2) < 0);
        assertTrue(comp.compare(2, 1) > 0);
    }
}

// Test7: Verify System.out::println method reference outputs correctly
class Test7 {
    @Test
    public void test() {
        java.io.ByteArrayOutputStream out = new java.io.ByteArrayOutputStream();
        java.io.PrintStream originalOut = System.out;
        System.setOut(new java.io.PrintStream(out));
        try {
            Consumer<String> printer = System.out::println;
            printer.accept("MethodRef");
            String output = out.toString().trim();
            assertEquals("MethodRef", output);
        } finally {
            System.setOut(originalOut);
        }
    }
}

// Test8: Verify Math::abs static method reference
class Test8 {
    @Test
    public void test() {
        Function<Integer, Integer> absRef = Math::abs;
        assertEquals(Integer.valueOf(5), absRef.apply(-5));
    }
}

// Test9: Verify String::concat method reference
class Test9 {
    @Test
    public void test() {
        BiFunction<String, String, String> concatRef = String::concat;
        assertEquals("HelloWorld", concatRef.apply("Hello", "World"));
    }
}

// Test10: Verify constructor reference with Person
class Test10 {
    @Test
    public void test() {
        Supplier<Person> personSupplier = () -> new Person("Default");
        Person p = personSupplier.get();
        assertNotNull(p);
    }
}`,
    translations: {
        ru: {
            title: 'Ссылки на методы',
            solutionCode: `import java.util.*;
import java.util.function.*;

class Person {
    private String name;

    public Person(String name) {
        this.name = name;
    }

    @Override
    public String toString() {
        return "Person(" + name + ")";
    }
}

public class MethodReferences {
    // Статический метод для демонстрации
    public static int square(int n) {
        return n * n;
    }

    // Метод экземпляра для демонстрации
    public String convertToUpperCase(String s) {
        return s.toUpperCase();
    }

    public static void main(String[] args) {
        System.out.println("=== Ссылка на статический метод ===");
        // Лямбда-выражение
        Function<Integer, Integer> squareLambda = n -> square(n);
        System.out.println("Lambda: " + squareLambda.apply(5));

        // Ссылка на метод (эквивалентна)
        Function<Integer, Integer> squareMethodRef = MethodReferences::square;
        System.out.println("Method Reference: " + squareMethodRef.apply(5));

        System.out.println("\\n=== Ссылка на метод экземпляра (конкретный объект) ===");
        MethodReferences obj = new MethodReferences();

        // Лямбда-выражение
        Function<String, String> upperLambda = s -> obj.convertToUpperCase(s);
        System.out.println("Lambda: " + upperLambda.apply("hello"));

        // Ссылка на метод (конкретный объект)
        Function<String, String> upperMethodRef = obj::convertToUpperCase;
        System.out.println("Method Reference: " + upperMethodRef.apply("hello"));

        System.out.println("\\n=== Ссылка на метод экземпляра (произвольный объект) ===");
        List<String> words = Arrays.asList("apple", "banana", "cherry");

        // Лямбда-выражение
        List<String> upperWordsLambda = new ArrayList<>();
        words.forEach(w -> upperWordsLambda.add(w.toUpperCase()));
        System.out.println("Lambda: " + upperWordsLambda);

        // Ссылка на метод (произвольный объект типа String)
        List<String> upperWordsMethodRef = new ArrayList<>();
        words.stream()
            .map(String::toUpperCase)
            .forEach(upperWordsMethodRef::add);
        System.out.println("Method Reference: " + upperWordsMethodRef);

        System.out.println("\\n=== Ссылка на конструктор ===");
        // Лямбда-выражение
        Function<String, Person> personLambda = name -> new Person(name);
        System.out.println("Lambda: " + personLambda.apply("John"));

        // Ссылка на конструктор
        Function<String, Person> personMethodRef = Person::new;
        System.out.println("Method Reference: " + personMethodRef.apply("Alice"));

        System.out.println("\\n=== Сортировка со ссылками на методы ===");
        List<Integer> numbers = Arrays.asList(3, 1, 4, 1, 5, 9, 2, 6);

        // Естественный порядок используя ссылку на метод
        List<Integer> sorted = new ArrayList<>(numbers);
        sorted.sort(Integer::compareTo);
        System.out.println("Natural order: " + sorted);

        // Обратный порядок используя ссылку на метод
        List<Integer> reversed = new ArrayList<>(numbers);
        reversed.sort(Comparator.reverseOrder());
        System.out.println("Reverse order: " + reversed);
    }
}`,
            description: `# Ссылки на методы

Ссылки на методы - это сокращенная запись для лямбда-выражений, которые вызывают один метод. Они делают код более кратким и читаемым, когда лямбда просто вызывает существующий метод. В Java существует четыре типа ссылок на методы.

## Требования:
1. Продемонстрируйте четыре типа ссылок на методы:
   1.1. Ссылка на статический метод: \`ClassName::staticMethod\`
   1.2. Ссылка на метод экземпляра конкретного объекта: \`object::instanceMethod\`
   1.3. Ссылка на метод экземпляра произвольного объекта: \`ClassName::instanceMethod\`
   1.4. Ссылка на конструктор: \`ClassName::new\`

2. Сравните лямбда-выражения с эквивалентными ссылками на методы:
   2.1. Покажите обе формы рядом
   2.2. Продемонстрируйте, когда можно использовать ссылки на методы

3. Создайте практические примеры:
   3.1. Операции со строками (toUpperCase, length)
   3.2. Математические операции (max, abs)
   3.3. Создание объектов (ссылки на конструкторы)
   3.4. Операции с коллекциями

4. Покажите сортировку массива с использованием ссылок на методы

## Пример вывода:
\`\`\`
=== Static Method Reference ===
Lambda: 25
Method Reference: 25

=== Instance Method Reference (Particular Object) ===
Lambda: HELLO
Method Reference: HELLO

=== Instance Method Reference (Arbitrary Object) ===
Lambda: [APPLE, BANANA, CHERRY]
Method Reference: [APPLE, BANANA, CHERRY]

=== Constructor Reference ===
Lambda: Person(John)
Method Reference: Person(Alice)

=== Sorting with Method References ===
Natural order: [1, 2, 3, 4, 5]
Reverse order: [5, 4, 3, 2, 1]
\`\`\``,
            hint1: `Ссылки на методы имеют четыре формы: ClassName::staticMethod, object::instanceMethod, ClassName::instanceMethod, ClassName::new. Выбирайте на основе того, что делает лямбда.`,
            hint2: `Ссылка на метод экземпляра произвольного объекта (String::toUpperCase) работает, когда лямбда передает параметр в метод экземпляра этого типа параметра.`,
            whyItMatters: `Ссылки на методы делают функциональный код более кратким и читаемым. Они особенно полезны с Stream API и операциями коллекций. Понимание того, когда использовать ссылки на методы вместо лямбда-выражений, помогает писать более чистый и поддерживаемый код. Это ключевая особенность современного стиля программирования на Java.

**Продакшен паттерн:**

\`\`\`java
// Обработка списка пользователей
List<User> users = userRepository.findAll();

// БЕЗ ссылок на методы: Многословно
users.stream()
    .map(user -> user.getEmail())
    .filter(email -> email.contains("@company.com"))
    .forEach(email -> System.out.println(email));

// СО ссылками на методы: Читаемо и кратко
users.stream()
    .map(User::getEmail)
    .filter(this::isCompanyEmail)
    .forEach(System.out::println);

// Сортировка по нескольким критериям
products.sort(Comparator.comparing(Product::getCategory)
    .thenComparing(Product::getPrice)
    .thenComparing(Product::getName));

// Создание объектов через конструктор
List<String> names = Arrays.asList("Alice", "Bob", "Charlie");
List<User> newUsers = names.stream()
    .map(User::new)  // Конструктор reference
    .collect(Collectors.toList());
\`\`\`

**Практические преимущества:**

1. **Улучшенная читаемость**: Код выглядит как естественный язык (map User to Email)
2. **Меньше шаблонного кода**: Нет необходимости в параметрах и телах лямбд
3. **Лучшая поддержка IDE**: Рефакторинг автоматически обновляет ссылки на методы
4. **Повторное использование**: Ссылки на методы поощряют извлечение логики в методы`
        },
        uz: {
            title: `Metodlarga havolalar`,
            solutionCode: `import java.util.*;
import java.util.function.*;

class Person {
    private String name;

    public Person(String name) {
        this.name = name;
    }

    @Override
    public String toString() {
        return "Person(" + name + ")";
    }
}

public class MethodReferences {
    // Namoyish uchun statik metod
    public static int square(int n) {
        return n * n;
    }

    // Namoyish uchun instansiya metodi
    public String convertToUpperCase(String s) {
        return s.toUpperCase();
    }

    public static void main(String[] args) {
        System.out.println("=== Statik metod havolasi ===");
        // Lambda ifoda
        Function<Integer, Integer> squareLambda = n -> square(n);
        System.out.println("Lambda: " + squareLambda.apply(5));

        // Metod havolasi (ekvivalent)
        Function<Integer, Integer> squareMethodRef = MethodReferences::square;
        System.out.println("Method Reference: " + squareMethodRef.apply(5));

        System.out.println("\\n=== Instansiya metodi havolasi (muayyan ob'ekt) ===");
        MethodReferences obj = new MethodReferences();

        // Lambda ifoda
        Function<String, String> upperLambda = s -> obj.convertToUpperCase(s);
        System.out.println("Lambda: " + upperLambda.apply("hello"));

        // Metod havolasi (muayyan ob'ekt)
        Function<String, String> upperMethodRef = obj::convertToUpperCase;
        System.out.println("Method Reference: " + upperMethodRef.apply("hello"));

        System.out.println("\\n=== Instansiya metodi havolasi (ixtiyoriy ob'ekt) ===");
        List<String> words = Arrays.asList("apple", "banana", "cherry");

        // Lambda ifoda
        List<String> upperWordsLambda = new ArrayList<>();
        words.forEach(w -> upperWordsLambda.add(w.toUpperCase()));
        System.out.println("Lambda: " + upperWordsLambda);

        // Metod havolasi (String turining ixtiyoriy ob'ekti)
        List<String> upperWordsMethodRef = new ArrayList<>();
        words.stream()
            .map(String::toUpperCase)
            .forEach(upperWordsMethodRef::add);
        System.out.println("Method Reference: " + upperWordsMethodRef);

        System.out.println("\\n=== Konstruktor havolasi ===");
        // Lambda ifoda
        Function<String, Person> personLambda = name -> new Person(name);
        System.out.println("Lambda: " + personLambda.apply("John"));

        // Konstruktor havolasi
        Function<String, Person> personMethodRef = Person::new;
        System.out.println("Method Reference: " + personMethodRef.apply("Alice"));

        System.out.println("\\n=== Metod havolalari bilan saralash ===");
        List<Integer> numbers = Arrays.asList(3, 1, 4, 1, 5, 9, 2, 6);

        // Metod havolasidan foydalanib tabiiy tartib
        List<Integer> sorted = new ArrayList<>(numbers);
        sorted.sort(Integer::compareTo);
        System.out.println("Natural order: " + sorted);

        // Metod havolasidan foydalanib teskari tartib
        List<Integer> reversed = new ArrayList<>(numbers);
        reversed.sort(Comparator.reverseOrder());
        System.out.println("Reverse order: " + reversed);
    }
}`,
            description: `# Metodlarga havolalar

Metod havolalari bitta metodga chaqiruv qiluvchi lambda ifodalari uchun qisqa yozuvdir. Ular kodni qisqaroq va o'qishga osonroq qiladi, qachonki lambda shunchaki mavjud metodga chaqiruv qiladi. Java-da metod havolalarining to'rt turi mavjud.

## Talablar:
1. To'rt turdagi metod havolalarini namoyish eting:
   1.1. Statik metod havolasi: \`ClassName::staticMethod\`
   1.2. Muayyan ob'ekt instansiya metodi havolasi: \`object::instanceMethod\`
   1.3. Ixtiyoriy ob'ekt instansiya metodi havolasi: \`ClassName::instanceMethod\`
   1.4. Konstruktor havolasi: \`ClassName::new\`

2. Lambda ifodalarni ekvivalent metod havolalari bilan solishtiring:
   2.1. Ikkala shaklni yonma-yon ko'rsating
   2.2. Metod havolalari qachon ishlatilishini namoyish eting

3. Amaliy misollar yarating:
   3.1. Satr operatsiyalari (toUpperCase, length)
   3.2. Matematik operatsiyalar (max, abs)
   3.3. Ob'ekt yaratish (konstruktor havolalari)
   3.4. Kolleksiya operatsiyalari

4. Metod havolalaridan foydalanib massiv saralashni ko'rsating

## Chiqish namunasi:
\`\`\`
=== Static Method Reference ===
Lambda: 25
Method Reference: 25

=== Instance Method Reference (Particular Object) ===
Lambda: HELLO
Method Reference: HELLO

=== Instance Method Reference (Arbitrary Object) ===
Lambda: [APPLE, BANANA, CHERRY]
Method Reference: [APPLE, BANANA, CHERRY]

=== Constructor Reference ===
Lambda: Person(John)
Method Reference: Person(Alice)

=== Sorting with Method References ===
Natural order: [1, 2, 3, 4, 5]
Reverse order: [5, 4, 3, 2, 1]
\`\`\``,
            hint1: `Metod havolalari to'rtta shaklga ega: ClassName::staticMethod, object::instanceMethod, ClassName::instanceMethod, ClassName::new. Lambda nima qilishiga qarab tanlang.`,
            hint2: `Ixtiyoriy ob'ekt instansiya metodi havolasi (String::toUpperCase) lambda parametrni o'sha parametr turining instansiya metodiga o'tkazganda ishlaydi.`,
            whyItMatters: `Metod havolalari funksional kodni qisqaroq va o'qishga osonroq qiladi. Ular ayniqsa Stream API va kolleksiya operatsiyalari bilan foydalidir. Metod havolalari va lambda ifodalarini qachon ishlatishni tushunish yanada toza va boshqariladigan kod yozishga yordam beradi. Bu zamonaviy Java dasturlash uslubining asosiy xususiyati hisoblanadi.

**Ishlab chiqarish patterni:**

\`\`\`java
// Foydalanuvchilar ro'yxatini qayta ishlash
List<User> users = userRepository.findAll();

// Metod havolalarisiz: Ko'p so'zli
users.stream()
    .map(user -> user.getEmail())
    .filter(email -> email.contains("@company.com"))
    .forEach(email -> System.out.println(email));

// Metod havolalari bilan: O'qiladigan va qisqa
users.stream()
    .map(User::getEmail)
    .filter(this::isCompanyEmail)
    .forEach(System.out::println);

// Bir nechta mezonlar bo'yicha saralash
products.sort(Comparator.comparing(Product::getCategory)
    .thenComparing(Product::getPrice)
    .thenComparing(Product::getName));

// Konstruktor orqali ob'ektlar yaratish
List<String> names = Arrays.asList("Alice", "Bob", "Charlie");
List<User> newUsers = names.stream()
    .map(User::new)  // Konstruktor havolasi
    .collect(Collectors.toList());
\`\`\`

**Amaliy foydalari:**

1. **Yaxshilangan o'qiluvchanlik**: Kod tabiiy til kabi ko'rinadi (map User to Email)
2. **Kamroq shablon kodi**: Lambda parametrlari va tanalari kerak emas
3. **Yaxshi IDE qo'llab-quvvatlash**: Refaktoring metod havolalarini avtomatik yangilaydi
4. **Qayta foydalanish**: Metod havolalari mantiqni metodlarga ajratishni rag'batlantiradi`
        }
    }
};

export default task;
