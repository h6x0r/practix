import { Task } from '../../../../types';

export const task: Task = {
    slug: 'java-intermediate-operations',
    title: 'Intermediate Operations',
    difficulty: 'easy',
    tags: ['java', 'stream-api', 'filter', 'map', 'flatmap', 'functional-programming'],
    estimatedTime: '30m',
    isPremium: false,
    youtubeUrl: '',
    description: `# Intermediate Operations

Intermediate operations transform a stream into another stream. They are lazy - they don't execute until a terminal operation is called. Common intermediate operations include filter, map, flatMap, distinct, sorted, peek, limit, and skip.

## Requirements:
1. Demonstrate filter() - select elements based on a condition:
   1.1. Filter even numbers
   1.2. Filter strings by length
   1.3. Filter objects by property

2. Demonstrate map() - transform each element:
   2.1. Convert strings to uppercase
   2.2. Extract object properties
   2.3. Perform calculations

3. Demonstrate flatMap() - flatten nested structures:
   3.1. Flatten list of lists
   3.2. Split strings into words
   3.3. Transform and flatten

4. Demonstrate distinct(), sorted(), limit(), skip():
   4.1. Remove duplicates
   4.2. Sort elements
   4.3. Take first N elements
   4.4. Skip first N elements

## Example Output:
\`\`\`
=== filter() Operation ===
Original: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
Even numbers: [2, 4, 6, 8, 10]
Long names: [Alexander, Elizabeth, Christopher]

=== map() Operation ===
Original: [hello, world, java, stream]
Uppercase: [HELLO, WORLD, JAVA, STREAM]
Lengths: [5, 5, 4, 6]
Squared: [1, 4, 9, 16, 25]

=== flatMap() Operation ===
Nested lists: [[1, 2], [3, 4], [5, 6]]
Flattened: [1, 2, 3, 4, 5, 6]
Sentences to words: [Java, Stream, API, is, powerful, and, elegant]

=== distinct(), sorted(), limit(), skip() ===
Original: [5, 2, 8, 2, 9, 1, 5, 3]
Distinct: [5, 2, 8, 9, 1, 3]
Sorted: [1, 2, 3, 5, 8, 9]
First 3: [1, 2, 3]
Skip 2: [3, 5, 8, 9]
\`\`\``,
    initialCode: `import java.util.*;
import java.util.stream.*;

public class IntermediateOperations {
    public static void main(String[] args) {
        // TODO: Demonstrate filter() operation

        // TODO: Demonstrate map() operation

        // TODO: Demonstrate flatMap() operation

        // TODO: Demonstrate distinct(), sorted(), limit(), skip()
    }
}`,
    solutionCode: `import java.util.*;
import java.util.stream.*;

public class IntermediateOperations {
    public static void main(String[] args) {
        System.out.println("=== filter() Operation ===");

        // Filter even numbers
        List<Integer> numbers = Arrays.asList(1, 2, 3, 4, 5, 6, 7, 8, 9, 10);
        System.out.println("Original: " + numbers);

        List<Integer> evenNumbers = numbers.stream()
            .filter(n -> n % 2 == 0)
            .collect(Collectors.toList());
        System.out.println("Even numbers: " + evenNumbers);

        // Filter strings by length
        List<String> names = Arrays.asList("John", "Alexander", "Bob", "Elizabeth", "Christopher", "Ann");
        List<String> longNames = names.stream()
            .filter(name -> name.length() > 5)
            .collect(Collectors.toList());
        System.out.println("Long names: " + longNames);

        System.out.println("\\n=== map() Operation ===");

        // Transform to uppercase
        List<String> words = Arrays.asList("hello", "world", "java", "stream");
        System.out.println("Original: " + words);

        List<String> uppercase = words.stream()
            .map(String::toUpperCase)
            .collect(Collectors.toList());
        System.out.println("Uppercase: " + uppercase);

        // Map to lengths
        List<Integer> lengths = words.stream()
            .map(String::length)
            .collect(Collectors.toList());
        System.out.println("Lengths: " + lengths);

        // Map with calculation
        List<Integer> squares = Arrays.asList(1, 2, 3, 4, 5).stream()
            .map(n -> n * n)
            .collect(Collectors.toList());
        System.out.println("Squared: " + squares);

        System.out.println("\\n=== flatMap() Operation ===");

        // Flatten nested lists
        List<List<Integer>> nestedLists = Arrays.asList(
            Arrays.asList(1, 2),
            Arrays.asList(3, 4),
            Arrays.asList(5, 6)
        );
        System.out.println("Nested lists: " + nestedLists);

        List<Integer> flattened = nestedLists.stream()
            .flatMap(List::stream)
            .collect(Collectors.toList());
        System.out.println("Flattened: " + flattened);

        // Split sentences into words
        List<String> sentences = Arrays.asList(
            "Java Stream API",
            "is powerful and elegant"
        );

        List<String> allWords = sentences.stream()
            .flatMap(sentence -> Arrays.stream(sentence.split(" ")))
            .collect(Collectors.toList());
        System.out.println("Sentences to words: " + allWords);

        System.out.println("\\n=== distinct(), sorted(), limit(), skip() ===");

        List<Integer> mixedNumbers = Arrays.asList(5, 2, 8, 2, 9, 1, 5, 3);
        System.out.println("Original: " + mixedNumbers);

        // Remove duplicates
        List<Integer> distinct = mixedNumbers.stream()
            .distinct()
            .collect(Collectors.toList());
        System.out.println("Distinct: " + distinct);

        // Sort elements
        List<Integer> sorted = mixedNumbers.stream()
            .distinct()
            .sorted()
            .collect(Collectors.toList());
        System.out.println("Sorted: " + sorted);

        // Take first 3
        List<Integer> first3 = mixedNumbers.stream()
            .distinct()
            .sorted()
            .limit(3)
            .collect(Collectors.toList());
        System.out.println("First 3: " + first3);

        // Skip first 2
        List<Integer> skip2 = mixedNumbers.stream()
            .distinct()
            .sorted()
            .skip(2)
            .collect(Collectors.toList());
        System.out.println("Skip 2: " + skip2);
    }
}`,
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
        IntermediateOperations.main(new String[]{});
        System.setOut(oldOut);
        String output = out.toString();
        assertTrue("Should show intermediate operations demo",
            output.contains("Stream") || output.contains("filter") ||
            output.contains("Intermediate") || output.contains("Промежуточные"));
    }
}

// Test2: should demonstrate filter operation
class Test2 {
    @Test
    public void test() {
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        PrintStream oldOut = System.out;
        System.setOut(new PrintStream(out));
        IntermediateOperations.main(new String[]{});
        System.setOut(oldOut);
        String output = out.toString();
        assertTrue("Should show filter operation", output.contains("filter") || output.contains("Even") || output.contains("juft"));
    }
}

// Test3: should show filtered even numbers
class Test3 {
    @Test
    public void test() {
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        PrintStream oldOut = System.out;
        System.setOut(new PrintStream(out));
        IntermediateOperations.main(new String[]{});
        System.setOut(oldOut);
        String output = out.toString();
        assertTrue("Should show even numbers", output.contains("[2, 4, 6") || output.contains("Even numbers") || output.contains("2, 4"));
    }
}

// Test4: should demonstrate map operation
class Test4 {
    @Test
    public void test() {
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        PrintStream oldOut = System.out;
        System.setOut(new PrintStream(out));
        IntermediateOperations.main(new String[]{});
        System.setOut(oldOut);
        String output = out.toString();
        assertTrue("Should show map operation", output.contains("map") || output.contains("Uppercase") || output.contains("katta harf"));
    }
}

// Test5: should show uppercase transformation
class Test5 {
    @Test
    public void test() {
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        PrintStream oldOut = System.out;
        System.setOut(new PrintStream(out));
        IntermediateOperations.main(new String[]{});
        System.setOut(oldOut);
        String output = out.toString();
        assertTrue("Should show uppercase", output.contains("HELLO") || output.contains("WORLD") || output.contains("JAVA"));
    }
}

// Test6: should demonstrate flatMap operation
class Test6 {
    @Test
    public void test() {
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        PrintStream oldOut = System.out;
        System.setOut(new PrintStream(out));
        IntermediateOperations.main(new String[]{});
        System.setOut(oldOut);
        String output = out.toString();
        assertTrue("Should show flatMap operation", output.contains("flatMap") || output.contains("Flatten") || output.contains("tekis"));
    }
}

// Test7: should show distinct operation
class Test7 {
    @Test
    public void test() {
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        PrintStream oldOut = System.out;
        System.setOut(new PrintStream(out));
        IntermediateOperations.main(new String[]{});
        System.setOut(oldOut);
        String output = out.toString();
        assertTrue("Should show distinct", output.contains("distinct") || output.contains("Distinct") || output.contains("Dublikat"));
    }
}

// Test8: should show sorted operation
class Test8 {
    @Test
    public void test() {
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        PrintStream oldOut = System.out;
        System.setOut(new PrintStream(out));
        IntermediateOperations.main(new String[]{});
        System.setOut(oldOut);
        String output = out.toString();
        assertTrue("Should show sorted", output.contains("sorted") || output.contains("Sorted") || output.contains("saral"));
    }
}

// Test9: should show limit or skip operation
class Test9 {
    @Test
    public void test() {
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        PrintStream oldOut = System.out;
        System.setOut(new PrintStream(out));
        IntermediateOperations.main(new String[]{});
        System.setOut(oldOut);
        String output = out.toString();
        assertTrue("Should show limit or skip", output.contains("limit") || output.contains("skip") || output.contains("First") || output.contains("Skip"));
    }
}

// Test10: should have section headers
class Test10 {
    @Test
    public void test() {
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        PrintStream oldOut = System.out;
        System.setOut(new PrintStream(out));
        IntermediateOperations.main(new String[]{});
        System.setOut(oldOut);
        String output = out.toString();
        boolean hasHeaders = output.contains("===") ||
                             output.contains("filter") || output.contains("map") || output.contains("operatsiya");
        assertTrue("Should have section headers", hasHeaders);
    }
}
`,
    hint1: `filter() keeps elements that match a condition. map() transforms each element. flatMap() transforms and flattens nested structures.`,
    hint2: `Intermediate operations can be chained: stream.filter(...).map(...).sorted().collect(...). Each operation returns a new stream.`,
    whyItMatters: `Intermediate operations are the building blocks of stream processing. They allow you to transform, filter, and manipulate data in a declarative way. Mastering these operations is essential for writing expressive and efficient data processing code in Java.

**Production Pattern:**
\`\`\`java
// Processing user data with validation and transformation
List<UserDTO> validAdultUsers = users.stream()
    .filter(user -> user.getAge() >= 18)
    .filter(user -> user.isEmailVerified())
    .map(user -> new UserDTO(user.getId(), user.getName()))
    .distinct()
    .collect(Collectors.toList());
\`\`\`

**Practical Benefits:**
- Chaining transformations for complex business logic
- No intermediate collections saves memory
- Code is easy to test and maintain`,
    order: 2,
    translations: {
        ru: {
            title: 'Промежуточные операции',
            solutionCode: `import java.util.*;
import java.util.stream.*;

public class IntermediateOperations {
    public static void main(String[] args) {
        System.out.println("=== Операция filter() ===");

        // Фильтр четных чисел
        List<Integer> numbers = Arrays.asList(1, 2, 3, 4, 5, 6, 7, 8, 9, 10);
        System.out.println("Original: " + numbers);

        List<Integer> evenNumbers = numbers.stream()
            .filter(n -> n % 2 == 0)
            .collect(Collectors.toList());
        System.out.println("Even numbers: " + evenNumbers);

        // Фильтр строк по длине
        List<String> names = Arrays.asList("John", "Alexander", "Bob", "Elizabeth", "Christopher", "Ann");
        List<String> longNames = names.stream()
            .filter(name -> name.length() > 5)
            .collect(Collectors.toList());
        System.out.println("Long names: " + longNames);

        System.out.println("\\n=== Операция map() ===");

        // Преобразование в верхний регистр
        List<String> words = Arrays.asList("hello", "world", "java", "stream");
        System.out.println("Original: " + words);

        List<String> uppercase = words.stream()
            .map(String::toUpperCase)
            .collect(Collectors.toList());
        System.out.println("Uppercase: " + uppercase);

        // Отображение длин
        List<Integer> lengths = words.stream()
            .map(String::length)
            .collect(Collectors.toList());
        System.out.println("Lengths: " + lengths);

        // Отображение с вычислением
        List<Integer> squares = Arrays.asList(1, 2, 3, 4, 5).stream()
            .map(n -> n * n)
            .collect(Collectors.toList());
        System.out.println("Squared: " + squares);

        System.out.println("\\n=== Операция flatMap() ===");

        // Выравнивание вложенных списков
        List<List<Integer>> nestedLists = Arrays.asList(
            Arrays.asList(1, 2),
            Arrays.asList(3, 4),
            Arrays.asList(5, 6)
        );
        System.out.println("Nested lists: " + nestedLists);

        List<Integer> flattened = nestedLists.stream()
            .flatMap(List::stream)
            .collect(Collectors.toList());
        System.out.println("Flattened: " + flattened);

        // Разделение предложений на слова
        List<String> sentences = Arrays.asList(
            "Java Stream API",
            "is powerful and elegant"
        );

        List<String> allWords = sentences.stream()
            .flatMap(sentence -> Arrays.stream(sentence.split(" ")))
            .collect(Collectors.toList());
        System.out.println("Sentences to words: " + allWords);

        System.out.println("\\n=== distinct(), sorted(), limit(), skip() ===");

        List<Integer> mixedNumbers = Arrays.asList(5, 2, 8, 2, 9, 1, 5, 3);
        System.out.println("Original: " + mixedNumbers);

        // Удаление дубликатов
        List<Integer> distinct = mixedNumbers.stream()
            .distinct()
            .collect(Collectors.toList());
        System.out.println("Distinct: " + distinct);

        // Сортировка элементов
        List<Integer> sorted = mixedNumbers.stream()
            .distinct()
            .sorted()
            .collect(Collectors.toList());
        System.out.println("Sorted: " + sorted);

        // Взять первые 3
        List<Integer> first3 = mixedNumbers.stream()
            .distinct()
            .sorted()
            .limit(3)
            .collect(Collectors.toList());
        System.out.println("First 3: " + first3);

        // Пропустить первые 2
        List<Integer> skip2 = mixedNumbers.stream()
            .distinct()
            .sorted()
            .skip(2)
            .collect(Collectors.toList());
        System.out.println("Skip 2: " + skip2);
    }
}`,
            description: `# Промежуточные операции

Промежуточные операции преобразуют один поток в другой поток. Они ленивые - они не выполняются до вызова терминальной операции. Распространенные промежуточные операции включают filter, map, flatMap, distinct, sorted, peek, limit и skip.

## Требования:
1. Продемонстрируйте filter() - выбор элементов по условию:
   1.1. Фильтр четных чисел
   1.2. Фильтр строк по длине
   1.3. Фильтр объектов по свойству

2. Продемонстрируйте map() - преобразование каждого элемента:
   2.1. Преобразование строк в верхний регистр
   2.2. Извлечение свойств объекта
   2.3. Выполнение вычислений

3. Продемонстрируйте flatMap() - выравнивание вложенных структур:
   3.1. Выравнивание списка списков
   3.2. Разделение строк на слова
   3.3. Преобразование и выравнивание

4. Продемонстрируйте distinct(), sorted(), limit(), skip():
   4.1. Удаление дубликатов
   4.2. Сортировка элементов
   4.3. Взятие первых N элементов
   4.4. Пропуск первых N элементов

## Пример вывода:
\`\`\`
=== filter() Operation ===
Original: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
Even numbers: [2, 4, 6, 8, 10]
Long names: [Alexander, Elizabeth, Christopher]

=== map() Operation ===
Original: [hello, world, java, stream]
Uppercase: [HELLO, WORLD, JAVA, STREAM]
Lengths: [5, 5, 4, 6]
Squared: [1, 4, 9, 16, 25]

=== flatMap() Operation ===
Nested lists: [[1, 2], [3, 4], [5, 6]]
Flattened: [1, 2, 3, 4, 5, 6]
Sentences to words: [Java, Stream, API, is, powerful, and, elegant]

=== distinct(), sorted(), limit(), skip() ===
Original: [5, 2, 8, 2, 9, 1, 5, 3]
Distinct: [5, 2, 8, 9, 1, 3]
Sorted: [1, 2, 3, 5, 8, 9]
First 3: [1, 2, 3]
Skip 2: [3, 5, 8, 9]
\`\`\``,
            hint1: `filter() сохраняет элементы, соответствующие условию. map() преобразует каждый элемент. flatMap() преобразует и выравнивает вложенные структуры.`,
            hint2: `Промежуточные операции можно связывать в цепочку: stream.filter(...).map(...).sorted().collect(...). Каждая операция возвращает новый поток.`,
            whyItMatters: `Промежуточные операции являются строительными блоками обработки потоков. Они позволяют преобразовывать, фильтровать и манипулировать данными декларативным способом. Освоение этих операций необходимо для написания выразительного и эффективного кода обработки данных в Java.

**Продакшен паттерн:**
\`\`\`java
// Обработка пользовательских данных с валидацией и трансформацией
List<UserDTO> validAdultUsers = users.stream()
    .filter(user -> user.getAge() >= 18)
    .filter(user -> user.isEmailVerified())
    .map(user -> new UserDTO(user.getId(), user.getName()))
    .distinct()
    .collect(Collectors.toList());
\`\`\`

**Практические преимущества:**
- Цепочка преобразований для сложной бизнес-логики
- Отсутствие промежуточных коллекций экономит память
- Код легко тестировать и поддерживать`
        },
        uz: {
            title: 'Oraliq operatsiyalar',
            solutionCode: `import java.util.*;
import java.util.stream.*;

public class IntermediateOperations {
    public static void main(String[] args) {
        System.out.println("=== filter() operatsiyasi ===");

        // Juft sonlarni filtrlash
        List<Integer> numbers = Arrays.asList(1, 2, 3, 4, 5, 6, 7, 8, 9, 10);
        System.out.println("Original: " + numbers);

        List<Integer> evenNumbers = numbers.stream()
            .filter(n -> n % 2 == 0)
            .collect(Collectors.toList());
        System.out.println("Even numbers: " + evenNumbers);

        // Satrlarni uzunlik bo'yicha filtrlash
        List<String> names = Arrays.asList("John", "Alexander", "Bob", "Elizabeth", "Christopher", "Ann");
        List<String> longNames = names.stream()
            .filter(name -> name.length() > 5)
            .collect(Collectors.toList());
        System.out.println("Long names: " + longNames);

        System.out.println("\\n=== map() operatsiyasi ===");

        // Katta harflarga o'zgartirish
        List<String> words = Arrays.asList("hello", "world", "java", "stream");
        System.out.println("Original: " + words);

        List<String> uppercase = words.stream()
            .map(String::toUpperCase)
            .collect(Collectors.toList());
        System.out.println("Uppercase: " + uppercase);

        // Uzunliklarni aks ettirish
        List<Integer> lengths = words.stream()
            .map(String::length)
            .collect(Collectors.toList());
        System.out.println("Lengths: " + lengths);

        // Hisoblash bilan aks ettirish
        List<Integer> squares = Arrays.asList(1, 2, 3, 4, 5).stream()
            .map(n -> n * n)
            .collect(Collectors.toList());
        System.out.println("Squared: " + squares);

        System.out.println("\\n=== flatMap() operatsiyasi ===");

        // Ichki ro'yxatlarni tekislash
        List<List<Integer>> nestedLists = Arrays.asList(
            Arrays.asList(1, 2),
            Arrays.asList(3, 4),
            Arrays.asList(5, 6)
        );
        System.out.println("Nested lists: " + nestedLists);

        List<Integer> flattened = nestedLists.stream()
            .flatMap(List::stream)
            .collect(Collectors.toList());
        System.out.println("Flattened: " + flattened);

        // Jumlalarni so'zlarga ajratish
        List<String> sentences = Arrays.asList(
            "Java Stream API",
            "is powerful and elegant"
        );

        List<String> allWords = sentences.stream()
            .flatMap(sentence -> Arrays.stream(sentence.split(" ")))
            .collect(Collectors.toList());
        System.out.println("Sentences to words: " + allWords);

        System.out.println("\\n=== distinct(), sorted(), limit(), skip() ===");

        List<Integer> mixedNumbers = Arrays.asList(5, 2, 8, 2, 9, 1, 5, 3);
        System.out.println("Original: " + mixedNumbers);

        // Dublikatlarni olib tashlash
        List<Integer> distinct = mixedNumbers.stream()
            .distinct()
            .collect(Collectors.toList());
        System.out.println("Distinct: " + distinct);

        // Elementlarni saralash
        List<Integer> sorted = mixedNumbers.stream()
            .distinct()
            .sorted()
            .collect(Collectors.toList());
        System.out.println("Sorted: " + sorted);

        // Birinchi 3 tani olish
        List<Integer> first3 = mixedNumbers.stream()
            .distinct()
            .sorted()
            .limit(3)
            .collect(Collectors.toList());
        System.out.println("First 3: " + first3);

        // Birinchi 2 tani o'tkazib yuborish
        List<Integer> skip2 = mixedNumbers.stream()
            .distinct()
            .sorted()
            .skip(2)
            .collect(Collectors.toList());
        System.out.println("Skip 2: " + skip2);
    }
}`,
            description: `# Oraliq operatsiyalar

Oraliq operatsiyalar bir streamni boshqa streamga o'zgartiradi. Ular lazy - yakuniy operatsiya chaqirilgunga qadar bajarilmaydi. Keng tarqalgan oraliq operatsiyalar filter, map, flatMap, distinct, sorted, peek, limit va skip kiradi.

## Talablar:
1. filter() ni namoyish eting - shart bo'yicha elementlarni tanlash:
   1.1. Juft sonlarni filtrlash
   1.2. Satrlarni uzunlik bo'yicha filtrlash
   1.3. Obyektlarni xususiyat bo'yicha filtrlash

2. map() ni namoyish eting - har bir elementni o'zgartirish:
   2.1. Satrlarni katta harflarga o'zgartirish
   2.2. Obyekt xususiyatlarini olish
   2.3. Hisoblashlarni bajarish

3. flatMap() ni namoyish eting - ichki strukturalarni tekislash:
   3.1. Ro'yxatlar ro'yxatini tekislash
   3.2. Satrlarni so'zlarga ajratish
   3.3. O'zgartirish va tekislash

4. distinct(), sorted(), limit(), skip() ni namoyish eting:
   4.1. Dublikatlarni olib tashlash
   4.2. Elementlarni saralash
   4.3. Birinchi N elementni olish
   4.4. Birinchi N elementni o'tkazib yuborish

## Chiqish namunasi:
\`\`\`
=== filter() Operation ===
Original: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
Even numbers: [2, 4, 6, 8, 10]
Long names: [Alexander, Elizabeth, Christopher]

=== map() Operation ===
Original: [hello, world, java, stream]
Uppercase: [HELLO, WORLD, JAVA, STREAM]
Lengths: [5, 5, 4, 6]
Squared: [1, 4, 9, 16, 25]

=== flatMap() Operation ===
Nested lists: [[1, 2], [3, 4], [5, 6]]
Flattened: [1, 2, 3, 4, 5, 6]
Sentences to words: [Java, Stream, API, is, powerful, and, elegant]

=== distinct(), sorted(), limit(), skip() ===
Original: [5, 2, 8, 2, 9, 1, 5, 3]
Distinct: [5, 2, 8, 9, 1, 3]
Sorted: [1, 2, 3, 5, 8, 9]
First 3: [1, 2, 3]
Skip 2: [3, 5, 8, 9]
\`\`\``,
            hint1: `filter() shartga mos keladigan elementlarni saqlaydi. map() har bir elementni o'zgartiradi. flatMap() ichki strukturalarni o'zgartiradi va tekislaydi.`,
            hint2: `Oraliq operatsiyalarni zanjirga bog'lash mumkin: stream.filter(...).map(...).sorted().collect(...). Har bir operatsiya yangi stream qaytaradi.`,
            whyItMatters: `Oraliq operatsiyalar stream qayta ishlashning qurilish bloklari hisoblanadi. Ular ma'lumotlarni deklarativ usulda o'zgartirish, filtrlash va boshqarishga imkon beradi. Bu operatsiyalarni o'zlashtirish Java-da ifodali va samarali ma'lumotlarni qayta ishlash kodini yozish uchun zarur.

**Ishlab chiqarish patterni:**
\`\`\`java
// Foydalanuvchi ma'lumotlarini validatsiya va transformatsiya qilish
List<UserDTO> validAdultUsers = users.stream()
    .filter(user -> user.getAge() >= 18)
    .filter(user -> user.isEmailVerified())
    .map(user -> new UserDTO(user.getId(), user.getName()))
    .distinct()
    .collect(Collectors.toList());
\`\`\`

**Amaliy foydalari:**
- Murakkab biznes mantiq uchun transformatsiyalar zanjiri
- Oraliq kolleksiyalar yo'qligi xotirani tejaydi
- Kodni test qilish va qo'llab-quvvatlash oson`
        }
    }
};

export default task;
