import { Task } from '../../../../types';

export const task: Task = {
    slug: 'java-lambda-best-practices',
    title: 'Lambda Best Practices',
    difficulty: 'medium',
    tags: ['java', 'lambda', 'best-practices', 'clean-code', 'functional-programming'],
    estimatedTime: '30m',
    isPremium: false,
    youtubeUrl: '',
    description: `# Lambda Best Practices

Writing effective lambda expressions requires following best practices. This includes keeping lambdas short and focused, avoiding side effects, handling exceptions properly, and knowing when to extract lambdas to named methods. These practices lead to maintainable and bug-free code.

## Requirements:
1. Keep lambdas short and readable:
   1.1. Single responsibility principle
   1.2. Extract complex logic to methods
   1.3. Use meaningful variable names
   1.4. When to use method references instead

2. Avoid side effects:
   2.1. Don't modify external state
   2.2. Prefer pure functions
   2.3. Show problematic examples
   2.4. Demonstrate correct alternatives

3. Exception handling in lambdas:
   3.1. Checked exceptions problem
   3.2. Wrapper methods for exception handling
   3.3. Custom functional interfaces with throws
   3.4. Try-catch within lambda body

4. Performance considerations:
   4.1. Avoid creating unnecessary objects
   4.2. Use primitive specialized interfaces
   4.3. Lambda vs method reference performance
   4.4. When lambdas are compiled to methods

## Example Output:
\`\`\`
=== Keep Lambdas Short ===
Bad: Complex lambda with multiple operations
Good: Extracted to named method - processData()
Good: Method reference - String::toUpperCase

=== Avoid Side Effects ===
Bad: Lambda modifies external list
Good: Returns new list without side effects
Result: [2, 4, 6, 8]

=== Exception Handling ===
Problem: Checked exceptions in lambda
Solution: Wrapper method handles exceptions
Result: Successfully processed 3 items

=== Performance Tips ===
Using primitive specialized: IntFunction
Avoiding boxing: Direct int operations
Method reference vs lambda: Equivalent performance
\`\`\``,
    initialCode: `import java.util.*;
import java.util.function.*;

public class LambdaBestPractices {
    public static void main(String[] args) {
        // TODO: Demonstrate keeping lambdas short and readable

        // TODO: Show side effects problems and solutions

        // TODO: Demonstrate exception handling techniques

        // TODO: Show performance considerations
    }

    // TODO: Add helper methods for demonstrations
}`,
    solutionCode: `import java.util.*;
import java.util.function.*;
import java.util.stream.*;
import java.io.*;

public class LambdaBestPractices {

    public static void main(String[] args) {
        System.out.println("=== Keep Lambdas Short ===");

        List<String> data = Arrays.asList("hello", "world", "java", "lambda");

        // BAD: Complex lambda with multiple operations
        List<String> badResult = data.stream()
            .map(s -> {
                String trimmed = s.trim();
                String upper = trimmed.toUpperCase();
                String prefix = "ITEM: " + upper;
                return prefix;
            })
            .collect(Collectors.toList());
        System.out.println("Bad: Complex lambda with multiple operations");

        // GOOD: Extract to named method
        List<String> goodResult = data.stream()
            .map(LambdaBestPractices::processData)
            .collect(Collectors.toList());
        System.out.println("Good: Extracted to named method - processData()");

        // GOOD: Use method reference when possible
        List<String> bestResult = data.stream()
            .map(String::toUpperCase)
            .collect(Collectors.toList());
        System.out.println("Good: Method reference - String::toUpperCase");

        System.out.println("\\n=== Avoid Side Effects ===");

        List<Integer> numbers = new ArrayList<>(Arrays.asList(1, 2, 3, 4));
        List<Integer> results = new ArrayList<>();

        // BAD: Lambda has side effects (modifies external state)
        numbers.stream()
            .map(n -> n * 2)
            .forEach(results::add); // Side effect: modifying external list
        System.out.println("Bad: Lambda modifies external list");

        // GOOD: Return new collection without side effects
        List<Integer> goodResults = numbers.stream()
            .map(n -> n * 2)
            .collect(Collectors.toList());
        System.out.println("Good: Returns new list without side effects");
        System.out.println("Result: " + goodResults);

        System.out.println("\\n=== Exception Handling ===");

        List<String> filenames = Arrays.asList("file1.txt", "file2.txt", "file3.txt");

        // PROBLEM: Can't use methods that throw checked exceptions directly
        // filenames.forEach(f -> readFile(f)); // Won't compile

        // SOLUTION 1: Wrapper method that handles exceptions
        filenames.forEach(f -> readFileSafe(f));
        System.out.println("Solution: Wrapper method handles exceptions");

        // SOLUTION 2: Try-catch within lambda
        long processedCount = filenames.stream()
            .filter(f -> {
                try {
                    return new File(f).exists();
                } catch (Exception e) {
                    return false;
                }
            })
            .count();
        System.out.println("Result: Successfully processed " + processedCount + " items");

        System.out.println("\\n=== Performance Tips ===");

        // Use primitive specialized interfaces to avoid boxing
        IntStream.range(1, 5)
            .map(n -> n * n)
            .forEach(n -> {}); // No boxing overhead
        System.out.println("Using primitive specialized: IntFunction");

        // Prefer primitive operations
        int sum = IntStream.range(1, 1000)
            .map(n -> n * 2)
            .sum(); // Efficient, no boxing
        System.out.println("Avoiding boxing: Direct int operations");

        // Method reference vs lambda - both are efficient
        List<String> words = Arrays.asList("one", "two", "three");
        words.stream().map(String::toUpperCase).count(); // Method reference
        words.stream().map(s -> s.toUpperCase()).count(); // Lambda
        System.out.println("Method reference vs lambda: Equivalent performance");
    }

    // Helper method: Extract complex logic from lambda
    private static String processData(String input) {
        return "ITEM: " + input.trim().toUpperCase();
    }

    // Helper method: Would throw checked exception
    private static void readFile(String filename) throws IOException {
        // Simulated file reading
        throw new IOException("File not found");
    }

    // Helper method: Wrapper that handles exceptions
    private static void readFileSafe(String filename) {
        try {
            readFile(filename);
        } catch (IOException e) {
            // Handle exception appropriately
            // System.err.println("Error reading file: " + filename);
        }
    }

    // Custom functional interface that allows checked exceptions
    @FunctionalInterface
    interface CheckedFunction<T, R> {
        R apply(T t) throws Exception;
    }

    // Wrapper to convert checked function to standard Function
    private static <T, R> Function<T, R> wrap(CheckedFunction<T, R> checked) {
        return t -> {
            try {
                return checked.apply(t);
            } catch (Exception e) {
                throw new RuntimeException(e);
            }
        };
    }
}`,
    hint1: `Keep lambdas short - ideally one line. If a lambda needs multiple statements or complex logic, extract it to a named method. This improves readability and testability.`,
    hint2: `Avoid side effects in lambdas. Lambdas should be pure functions that don't modify external state. This makes code predictable and enables optimizations like parallel processing.`,
    whyItMatters: `Following lambda best practices leads to clean, maintainable code. Short lambdas are easier to understand and debug. Avoiding side effects makes code thread-safe and enables parallel processing. Proper exception handling prevents runtime errors. Performance considerations matter in production code, especially with large data sets. These practices are essential for professional Java development.`,
    order: 7,
    testCode: `import static org.junit.Assert.*;
import org.junit.Test;
import java.util.*;
import java.util.stream.*;

// Test1: Verify short lambda is preferred
class Test1 {
    @Test
    public void test() {
        List<String> data = Arrays.asList("hello", "world");
        List<String> result = data.stream()
            .map(String::toUpperCase)
            .collect(Collectors.toList());
        assertEquals("HELLO", result.get(0));
    }
}

// Test2: Verify lambda without side effects
class Test2 {
    @Test
    public void test() {
        List<Integer> numbers = Arrays.asList(1, 2, 3, 4);
        List<Integer> doubled = numbers.stream()
            .map(n -> n * 2)
            .collect(Collectors.toList());
        assertEquals(4, doubled.size());
        assertEquals(Integer.valueOf(2), doubled.get(0));
    }
}

// Test3: Verify extracted method for complex logic
class Test3 {
    @Test
    public void test() {
        List<String> data = Arrays.asList("hello", "world");
        List<String> result = data.stream()
            .map(LambdaBestPractices::processData)
            .collect(Collectors.toList());
        assertTrue(result.get(0).contains("ITEM:"));
    }
}

// Test4: Verify method reference usage
class Test4 {
    @Test
    public void test() {
        List<String> data = Arrays.asList("one", "two", "three");
        List<String> result = data.stream()
            .map(String::toUpperCase)
            .collect(Collectors.toList());
        assertEquals("ONE", result.get(0));
    }
}

// Test5: Verify exception handling with try-catch in lambda
class Test5 {
    @Test
    public void test() {
        List<String> filenames = Arrays.asList("file1.txt", "file2.txt");
        long count = filenames.stream()
            .filter(f -> {
                try {
                    return new java.io.File(f).exists();
                } catch (Exception e) {
                    return false;
                }
            })
            .count();
        assertTrue(count >= 0);
    }
}

// Test6: Verify primitive specialized interface usage
class Test6 {
    @Test
    public void test() {
        int sum = java.util.stream.IntStream.range(1, 5)
            .map(n -> n * 2)
            .sum();
        assertEquals(20, sum);
    }
}

// Test7: Verify avoiding side effects in stream
class Test7 {
    @Test
    public void test() {
        List<Integer> numbers = Arrays.asList(1, 2, 3);
        List<Integer> result = numbers.stream()
            .map(n -> n * 3)
            .collect(Collectors.toList());
        assertEquals(3, result.size());
    }
}

// Test8: Verify lambda with single responsibility
class Test8 {
    @Test
    public void test() {
        List<String> words = Arrays.asList("apple", "banana");
        List<String> result = words.stream()
            .map(String::toUpperCase)
            .collect(Collectors.toList());
        assertEquals("APPLE", result.get(0));
    }
}

// Test9: Verify using Predicate composition
class Test9 {
    @Test
    public void test() {
        java.util.function.Predicate<Integer> isEven = n -> n % 2 == 0;
        java.util.function.Predicate<Integer> isPositive = n -> n > 0;
        java.util.function.Predicate<Integer> combined = isEven.and(isPositive);
        assertTrue(combined.test(4));
        assertFalse(combined.test(-2));
    }
}

// Test10: Verify performance with method reference
class Test10 {
    @Test
    public void test() {
        List<String> words = Arrays.asList("one", "two", "three");
        long count1 = words.stream().map(String::toUpperCase).count();
        long count2 = words.stream().map(s -> s.toUpperCase()).count();
        assertEquals(count1, count2);
    }
}`,
    translations: {
        ru: {
            title: 'Лучшие практики лямбд',
            solutionCode: `import java.util.*;
import java.util.function.*;
import java.util.stream.*;
import java.io.*;

public class LambdaBestPractices {

    public static void main(String[] args) {
        System.out.println("=== Держите лямбды короткими ===");

        List<String> data = Arrays.asList("hello", "world", "java", "lambda");

        // ПЛОХО: Сложная лямбда с несколькими операциями
        List<String> badResult = data.stream()
            .map(s -> {
                String trimmed = s.trim();
                String upper = trimmed.toUpperCase();
                String prefix = "ITEM: " + upper;
                return prefix;
            })
            .collect(Collectors.toList());
        System.out.println("Bad: Complex lambda with multiple operations");

        // ХОРОШО: Извлечь в именованный метод
        List<String> goodResult = data.stream()
            .map(LambdaBestPractices::processData)
            .collect(Collectors.toList());
        System.out.println("Good: Extracted to named method - processData()");

        // ХОРОШО: Использовать ссылку на метод, когда возможно
        List<String> bestResult = data.stream()
            .map(String::toUpperCase)
            .collect(Collectors.toList());
        System.out.println("Good: Method reference - String::toUpperCase");

        System.out.println("\\n=== Избегайте побочных эффектов ===");

        List<Integer> numbers = new ArrayList<>(Arrays.asList(1, 2, 3, 4));
        List<Integer> results = new ArrayList<>();

        // ПЛОХО: Лямбда имеет побочные эффекты (изменяет внешнее состояние)
        numbers.stream()
            .map(n -> n * 2)
            .forEach(results::add); // Побочный эффект: изменение внешнего списка
        System.out.println("Bad: Lambda modifies external list");

        // ХОРОШО: Вернуть новую коллекцию без побочных эффектов
        List<Integer> goodResults = numbers.stream()
            .map(n -> n * 2)
            .collect(Collectors.toList());
        System.out.println("Good: Returns new list without side effects");
        System.out.println("Result: " + goodResults);

        System.out.println("\\n=== Обработка исключений ===");

        List<String> filenames = Arrays.asList("file1.txt", "file2.txt", "file3.txt");

        // ПРОБЛЕМА: Нельзя напрямую использовать методы, которые выбрасывают проверяемые исключения
        // filenames.forEach(f -> readFile(f)); // Не скомпилируется

        // РЕШЕНИЕ 1: Метод-обертка, который обрабатывает исключения
        filenames.forEach(f -> readFileSafe(f));
        System.out.println("Solution: Wrapper method handles exceptions");

        // РЕШЕНИЕ 2: Try-catch внутри лямбды
        long processedCount = filenames.stream()
            .filter(f -> {
                try {
                    return new File(f).exists();
                } catch (Exception e) {
                    return false;
                }
            })
            .count();
        System.out.println("Result: Successfully processed " + processedCount + " items");

        System.out.println("\\n=== Советы по производительности ===");

        // Используйте специализированные примитивные интерфейсы для избежания упаковки
        IntStream.range(1, 5)
            .map(n -> n * n)
            .forEach(n -> {}); // Нет накладных расходов на упаковку
        System.out.println("Using primitive specialized: IntFunction");

        // Предпочитайте примитивные операции
        int sum = IntStream.range(1, 1000)
            .map(n -> n * 2)
            .sum(); // Эффективно, без упаковки
        System.out.println("Avoiding boxing: Direct int operations");

        // Ссылка на метод vs лямбда - оба эффективны
        List<String> words = Arrays.asList("one", "two", "three");
        words.stream().map(String::toUpperCase).count(); // Ссылка на метод
        words.stream().map(s -> s.toUpperCase()).count(); // Лямбда
        System.out.println("Method reference vs lambda: Equivalent performance");
    }

    // Вспомогательный метод: Извлечь сложную логику из лямбды
    private static String processData(String input) {
        return "ITEM: " + input.trim().toUpperCase();
    }

    // Вспомогательный метод: Выбрасывает проверяемое исключение
    private static void readFile(String filename) throws IOException {
        // Симуляция чтения файла
        throw new IOException("File not found");
    }

    // Вспомогательный метод: Обертка, которая обрабатывает исключения
    private static void readFileSafe(String filename) {
        try {
            readFile(filename);
        } catch (IOException e) {
            // Обработать исключение соответствующим образом
            // System.err.println("Error reading file: " + filename);
        }
    }

    // Пользовательский функциональный интерфейс, который разрешает проверяемые исключения
    @FunctionalInterface
    interface CheckedFunction<T, R> {
        R apply(T t) throws Exception;
    }

    // Обертка для преобразования проверяемой функции в стандартную Function
    private static <T, R> Function<T, R> wrap(CheckedFunction<T, R> checked) {
        return t -> {
            try {
                return checked.apply(t);
            } catch (Exception e) {
                throw new RuntimeException(e);
            }
        };
    }
}`,
            description: `# Лучшие практики лямбд

Написание эффективных лямбда-выражений требует следования лучшим практикам. Это включает в себя поддержание лямбд короткими и сфокусированными, избегание побочных эффектов, правильную обработку исключений и знание того, когда извлекать лямбды в именованные методы. Эти практики приводят к поддерживаемому и безошибочному коду.

## Требования:
1. Держите лямбды короткими и читаемыми:
   1.1. Принцип единственной ответственности
   1.2. Извлечение сложной логики в методы
   1.3. Использование значащих имен переменных
   1.4. Когда использовать ссылки на методы вместо лямбд

2. Избегайте побочных эффектов:
   2.1. Не изменяйте внешнее состояние
   2.2. Предпочитайте чистые функции
   2.3. Покажите проблемные примеры
   2.4. Продемонстрируйте правильные альтернативы

3. Обработка исключений в лямбдах:
   3.1. Проблема проверяемых исключений
   3.2. Методы-обертки для обработки исключений
   3.3. Пользовательские функциональные интерфейсы с throws
   3.4. Try-catch внутри тела лямбды

4. Соображения производительности:
   4.1. Избегайте создания ненужных объектов
   4.2. Используйте специализированные примитивные интерфейсы
   4.3. Производительность лямбды vs ссылки на метод
   4.4. Когда лямбды компилируются в методы

## Пример вывода:
\`\`\`
=== Keep Lambdas Short ===
Bad: Complex lambda with multiple operations
Good: Extracted to named method - processData()
Good: Method reference - String::toUpperCase

=== Avoid Side Effects ===
Bad: Lambda modifies external list
Good: Returns new list without side effects
Result: [2, 4, 6, 8]

=== Exception Handling ===
Problem: Checked exceptions in lambda
Solution: Wrapper method handles exceptions
Result: Successfully processed 3 items

=== Performance Tips ===
Using primitive specialized: IntFunction
Avoiding boxing: Direct int operations
Method reference vs lambda: Equivalent performance
\`\`\``,
            hint1: `Держите лямбды короткими - в идеале одна строка. Если лямбде нужно несколько операторов или сложная логика, извлеките её в именованный метод. Это улучшает читаемость и тестируемость.`,
            hint2: `Избегайте побочных эффектов в лямбдах. Лямбды должны быть чистыми функциями, которые не изменяют внешнее состояние. Это делает код предсказуемым и позволяет оптимизации, такие как параллельная обработка.`,
            whyItMatters: `Следование лучшим практикам лямбд приводит к чистому, поддерживаемому коду. Короткие лямбды легче понять и отладить. Избегание побочных эффектов делает код потокобезопасным и позволяет параллельную обработку. Правильная обработка исключений предотвращает ошибки во время выполнения. Соображения производительности важны в продакшн-коде, особенно с большими наборами данных. Эти практики необходимы для профессиональной разработки на Java.`
        },
        uz: {
            title: `Lambda eng yaxshi amaliyotlar`,
            solutionCode: `import java.util.*;
import java.util.function.*;
import java.util.stream.*;
import java.io.*;

public class LambdaBestPractices {

    public static void main(String[] args) {
        System.out.println("=== Lambdalarni qisqa saqlang ===");

        List<String> data = Arrays.asList("hello", "world", "java", "lambda");

        // YOMON: Bir nechta operatsiyalar bilan murakkab lambda
        List<String> badResult = data.stream()
            .map(s -> {
                String trimmed = s.trim();
                String upper = trimmed.toUpperCase();
                String prefix = "ITEM: " + upper;
                return prefix;
            })
            .collect(Collectors.toList());
        System.out.println("Bad: Complex lambda with multiple operations");

        // YAXSHI: Nomlangan metodga ajratish
        List<String> goodResult = data.stream()
            .map(LambdaBestPractices::processData)
            .collect(Collectors.toList());
        System.out.println("Good: Extracted to named method - processData()");

        // YAXSHI: Iloji boricha metod havolasidan foydalaning
        List<String> bestResult = data.stream()
            .map(String::toUpperCase)
            .collect(Collectors.toList());
        System.out.println("Good: Method reference - String::toUpperCase");

        System.out.println("\\n=== Yon ta'sirlardan qoching ===");

        List<Integer> numbers = new ArrayList<>(Arrays.asList(1, 2, 3, 4));
        List<Integer> results = new ArrayList<>();

        // YOMON: Lambda yon ta'sirlarga ega (tashqi holatni o'zgartiradi)
        numbers.stream()
            .map(n -> n * 2)
            .forEach(results::add); // Yon ta'sir: tashqi ro'yxatni o'zgartirish
        System.out.println("Bad: Lambda modifies external list");

        // YAXSHI: Yon ta'sirlarsiz yangi kolleksiyani qaytarish
        List<Integer> goodResults = numbers.stream()
            .map(n -> n * 2)
            .collect(Collectors.toList());
        System.out.println("Good: Returns new list without side effects");
        System.out.println("Result: " + goodResults);

        System.out.println("\\n=== Istisnolarni boshqarish ===");

        List<String> filenames = Arrays.asList("file1.txt", "file2.txt", "file3.txt");

        // MUAMMO: Tekshirilgan istisnolarni chiqaradigan metodlarni to'g'ridan-to'g'ri ishlatib bo'lmaydi
        // filenames.forEach(f -> readFile(f)); // Kompilyatsiya qilinmaydi

        // YECHIM 1: Istisnolarni boshqaradigan o'rovchi metod
        filenames.forEach(f -> readFileSafe(f));
        System.out.println("Solution: Wrapper method handles exceptions");

        // YECHIM 2: Lambda ichida try-catch
        long processedCount = filenames.stream()
            .filter(f -> {
                try {
                    return new File(f).exists();
                } catch (Exception e) {
                    return false;
                }
            })
            .count();
        System.out.println("Result: Successfully processed " + processedCount + " items");

        System.out.println("\\n=== Ishlash maslahatlar ===");

        // Qadoqlashdan qochish uchun primitiv maxsus interfeyslardan foydalaning
        IntStream.range(1, 5)
            .map(n -> n * n)
            .forEach(n -> {}); // Qadoqlash xarajatlari yo'q
        System.out.println("Using primitive specialized: IntFunction");

        // Primitiv operatsiyalarni afzal ko'ring
        int sum = IntStream.range(1, 1000)
            .map(n -> n * 2)
            .sum(); // Samarali, qadoqlashsiz
        System.out.println("Avoiding boxing: Direct int operations");

        // Metod havolasi vs lambda - ikkalasi ham samarali
        List<String> words = Arrays.asList("one", "two", "three");
        words.stream().map(String::toUpperCase).count(); // Metod havolasi
        words.stream().map(s -> s.toUpperCase()).count(); // Lambda
        System.out.println("Method reference vs lambda: Equivalent performance");
    }

    // Yordamchi metod: Lambdadan murakkab mantiqni ajratish
    private static String processData(String input) {
        return "ITEM: " + input.trim().toUpperCase();
    }

    // Yordamchi metod: Tekshirilgan istisnoni chiqaradi
    private static void readFile(String filename) throws IOException {
        // Fayl o'qish simulyatsiyasi
        throw new IOException("File not found");
    }

    // Yordamchi metod: Istisnolarni boshqaradigan o'rovchi
    private static void readFileSafe(String filename) {
        try {
            readFile(filename);
        } catch (IOException e) {
            // Istisnoni mos tarzda boshqarish
            // System.err.println("Error reading file: " + filename);
        }
    }

    // Tekshirilgan istisnolarga ruxsat beradigan maxsus funksional interfeys
    @FunctionalInterface
    interface CheckedFunction<T, R> {
        R apply(T t) throws Exception;
    }

    // Tekshirilgan funksiyani standart Function ga aylantirish uchun o'rovchi
    private static <T, R> Function<T, R> wrap(CheckedFunction<T, R> checked) {
        return t -> {
            try {
                return checked.apply(t);
            } catch (Exception e) {
                throw new RuntimeException(e);
            }
        };
    }
}`,
            description: `# Lambda eng yaxshi amaliyotlar

Samarali lambda ifodalarini yozish eng yaxshi amaliyotlarga rioya qilishni talab qiladi. Bu lambdalarni qisqa va diqqat markazida saqlash, yon ta'sirlardan qochish, istisnolarni to'g'ri boshqarish va lambdalarni nomlangan metodlarga qachon ajratishni bilishni o'z ichiga oladi. Bu amaliyotlar boshqariladigan va xatosiz kodga olib keladi.

## Talablar:
1. Lambdalarni qisqa va o'qilishi oson saqlang:
   1.1. Yagona mas'uliyat printsipi
   1.2. Murakkab mantiqni metodlarga ajratish
   1.3. Ma'noli o'zgaruvchi nomlaridan foydalanish
   1.4. Lambdalar o'rniga metod havolalarini qachon ishlatish

2. Yon ta'sirlardan qoching:
   2.1. Tashqi holatni o'zgartirmang
   2.2. Toza funksiyalarni afzal ko'ring
   2.3. Muammoli misollarni ko'rsating
   2.4. To'g'ri alternativalarni namoyish eting

3. Lambdalarda istisnolarni boshqarish:
   3.1. Tekshirilgan istisnolar muammosi
   3.2. Istisnolarni boshqarish uchun o'rovchi metodlar
   3.3. throws bilan maxsus funksional interfeyslar
   3.4. Lambda tanasi ichida try-catch

4. Ishlash ko'rsatkichlari:
   4.1. Keraksiz ob'ektlar yaratishdan qoching
   4.2. Primitiv maxsus interfeyslardan foydalaning
   4.3. Lambda vs metod havolasi ishlashi
   4.4. Lambdalar qachon metodlarga kompilyatsiya qilinadi

## Chiqish namunasi:
\`\`\`
=== Keep Lambdas Short ===
Bad: Complex lambda with multiple operations
Good: Extracted to named method - processData()
Good: Method reference - String::toUpperCase

=== Avoid Side Effects ===
Bad: Lambda modifies external list
Good: Returns new list without side effects
Result: [2, 4, 6, 8]

=== Exception Handling ===
Problem: Checked exceptions in lambda
Solution: Wrapper method handles exceptions
Result: Successfully processed 3 items

=== Performance Tips ===
Using primitive specialized: IntFunction
Avoiding boxing: Direct int operations
Method reference vs lambda: Equivalent performance
\`\`\``,
            hint1: `Lambdalarni qisqa saqlang - ideal holda bitta qator. Agar lambdaga bir nechta operator yoki murakkab mantiq kerak bo'lsa, uni nomlangan metodga ajrating. Bu o'qilishni va testlanishni yaxshilaydi.`,
            hint2: `Lambdalarda yon ta'sirlardan qoching. Lambdalar tashqi holatni o'zgartirmaydigan toza funksiyalar bo'lishi kerak. Bu kodni bashorat qilinadigan qiladi va parallel qayta ishlash kabi optimizatsiyalarni yoqadi.`,
            whyItMatters: `Lambda eng yaxshi amaliyotlariga rioya qilish toza, boshqariladigan kodga olib keladi. Qisqa lambdalar tushunish va disk raskadka qilish osonroq. Yon ta'sirlardan qochish kodni thread-safe qiladi va parallel qayta ishlashni yoqadi. To'g'ri istisnolarni boshqarish runtime xatolarining oldini oladi. Ishlash ko'rsatkichlari ishlab chiqarish kodida, ayniqsa katta ma'lumotlar to'plamlari bilan muhimdir. Bu amaliyotlar professional Java ishlab chiqish uchun zarurdir.`
        }
    }
};

export default task;
