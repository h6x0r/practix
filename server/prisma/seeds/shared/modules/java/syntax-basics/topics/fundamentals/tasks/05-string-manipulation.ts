import { Task } from '../../../../types';

export const task: Task = {
    slug: 'java-string-operations',
    title: 'String Operations and StringBuilder',
    difficulty: 'easy',
    tags: ['java', 'syntax', 'strings', 'stringbuilder', 'text-processing'],
    estimatedTime: '20m',
    isPremium: false,
    youtubeUrl: '',
    description: `Master Java string operations, manipulation, and efficient string building.

**Requirements:**
1. Create a \`stringBasics()\` method demonstrating string creation, comparison, and immutability
2. Implement a \`stringMethods(String text)\` method showing common string operations
3. Create a \`stringBuilderDemo()\` method demonstrating StringBuilder for efficient concatenation
4. Implement a \`stringFormatting()\` method showing various formatting techniques

**String Operations to Cover:**
- String creation and literals
- String comparison (==, equals, equalsIgnoreCase)
- Common methods: length, charAt, substring, indexOf, contains
- String splitting and joining
- String formatting (String.format, printf)
- StringBuilder for efficient concatenation
- Text blocks (Java 15+)

**Example Output:**
\`\`\`
Length: 13
Substring: World
Contains "Java": true
Split words: [Hello, Java, World]
\`\`\``,
    initialCode: `public class StringManipulationDemo {

    public static void stringBasics() {
        // TODO: Demonstrate string creation and comparison


    }

    public static void stringMethods(String text) {
        // TODO: Show common string operations


    }

    public static void stringBuilderDemo() {
        // TODO: Demonstrate StringBuilder usage


    }

    public static void stringFormatting() {
        // TODO: Show string formatting techniques


    }

    public static void main(String[] args) {
        stringBasics();
        stringMethods("Hello Java World");
        stringBuilderDemo();
        stringFormatting();
    }
}`,
    solutionCode: `public class StringManipulationDemo {

    public static void stringBasics() {
        System.out.println("=== String Basics ===");

        // String creation
        String str1 = "Hello"; // String literal (stored in String pool)
        String str2 = "Hello"; // References same object in pool
        String str3 = new String("Hello"); // New object on heap

        // String comparison
        System.out.println("str1 == str2: " + (str1 == str2)); // true (same reference)
        System.out.println("str1 == str3: " + (str1 == str3)); // false (different references)
        System.out.println("str1.equals(str3): " + str1.equals(str3)); // true (same content)

        // String immutability
        String original = "Java";
        String modified = original.concat(" Programming");
        System.out.println("");
        System.out.println("Immutability example:");
        System.out.println("Original: " + original); // Still "Java"
        System.out.println("Modified: " + modified); // "Java Programming"

        // Case-insensitive comparison
        String lower = "hello";
        String upper = "HELLO";
        System.out.println("");
        System.out.println("Case comparison:");
        System.out.println("equals: " + lower.equals(upper)); // false
        System.out.println("equalsIgnoreCase: " + lower.equalsIgnoreCase(upper)); // true
    }

    public static void stringMethods(String text) {
        System.out.println("");
        System.out.println("=== String Methods ===");
        System.out.println("Original text: \"" + text + "\"");

        // Length and character access
        System.out.println("");
        System.out.println("Length: " + text.length());
        System.out.println("First character: " + text.charAt(0));
        System.out.println("Last character: " + text.charAt(text.length() - 1));

        // Substring operations
        System.out.println("");
        System.out.println("Substring operations:");
        System.out.println("substring(0, 5): \"" + text.substring(0, 5) + "\"");
        System.out.println("substring(6): \"" + text.substring(6) + "\"");

        // Searching
        System.out.println("");
        System.out.println("Searching:");
        System.out.println("indexOf('Java'): " + text.indexOf("Java"));
        System.out.println("lastIndexOf('o'): " + text.lastIndexOf('o'));
        System.out.println("contains('Java'): " + text.contains("Java"));
        System.out.println("startsWith('Hello'): " + text.startsWith("Hello"));
        System.out.println("endsWith('World'): " + text.endsWith("World"));

        // Splitting and joining
        System.out.println("");
        System.out.println("Splitting:");
        String[] words = text.split(" ");
        System.out.print("Split by space: [");
        for (int i = 0; i < words.length; i++) {
            System.out.print("\"" + words[i] + "\"");
            if (i < words.length - 1) System.out.print(", ");
        }
        System.out.println("]");

        String joined = String.join("-", words);
        System.out.println("Joined with '-': " + joined);

        // Trimming and replacing
        String padded = "  Java  ";
        System.out.println("");
        System.out.println("Trimming:");
        System.out.println("Before trim: \"" + padded + "\" (length: " + padded.length() + ")");
        System.out.println("After trim: \"" + padded.trim() + "\" (length: " + padded.trim().length() + ")");

        System.out.println("");
        System.out.println("Replacing:");
        System.out.println("replace('Java', 'Python'): " + text.replace("Java", "Python"));
        System.out.println("replaceAll('\\\\s+', '-'): " + text.replaceAll("\\s+", "-"));

        // Case conversion
        System.out.println("");
        System.out.println("Case conversion:");
        System.out.println("toUpperCase(): " + text.toUpperCase());
        System.out.println("toLowerCase(): " + text.toLowerCase());
    }

    public static void stringBuilderDemo() {
        System.out.println("");
        System.out.println("=== StringBuilder Demo ===");

        // StringBuilder for efficient string building
        StringBuilder sb = new StringBuilder();
        sb.append("Java");
        sb.append(" is");
        sb.append(" awesome");
        System.out.println("Built string: " + sb.toString());

        // StringBuilder methods
        System.out.println("");
        System.out.println("StringBuilder operations:");
        StringBuilder builder = new StringBuilder("Hello");
        System.out.println("Initial: " + builder);

        builder.append(" World");
        System.out.println("After append: " + builder);

        builder.insert(5, " Java");
        System.out.println("After insert: " + builder);

        builder.delete(5, 10);
        System.out.println("After delete: " + builder);

        builder.reverse();
        System.out.println("After reverse: " + builder);

        builder.reverse(); // Reverse back
        builder.replace(0, 5, "Hi");
        System.out.println("After replace: " + builder);

        // Performance comparison
        System.out.println("");
        System.out.println("Performance example:");
        long startTime, endTime;

        // Using String concatenation (slower for many operations)
        startTime = System.nanoTime();
        String result1 = "";
        for (int i = 0; i < 1000; i++) {
            result1 += "a";
        }
        endTime = System.nanoTime();
        System.out.println("String concatenation time: " + (endTime - startTime) + " ns");

        // Using StringBuilder (much faster)
        startTime = System.nanoTime();
        StringBuilder sb2 = new StringBuilder();
        for (int i = 0; i < 1000; i++) {
            sb2.append("a");
        }
        String result2 = sb2.toString();
        endTime = System.nanoTime();
        System.out.println("StringBuilder time: " + (endTime - startTime) + " ns");
    }

    public static void stringFormatting() {
        System.out.println("");
        System.out.println("=== String Formatting ===");

        String name = "Alice";
        int age = 25;
        double salary = 75000.50;

        // String.format()
        String formatted1 = String.format("Name: %s, Age: %d, Salary: $%.2f", name, age, salary);
        System.out.println("String.format(): " + formatted1);

        // printf (prints directly)
        System.out.print("printf(): ");
        System.out.printf("Name: %s, Age: %d, Salary: $%.2f%n", name, age, salary);

        // Formatting numbers
        System.out.println("");
        System.out.println("Number formatting:");
        System.out.printf("Integer with leading zeros: %05d%n", 42);
        System.out.printf("Float with 3 decimals: %.3f%n", 3.14159);
        System.out.printf("Scientific notation: %e%n", 123456.789);
        System.out.printf("Percentage: %.1f%%%n", 85.5);

        // Alignment
        System.out.println("");
        System.out.println("Alignment:");
        System.out.printf("Left-aligned: %-10s|%n", "Java");
        System.out.printf("Right-aligned: %10s|%n", "Java");

        // Text blocks (Java 15+)
        System.out.println("");
        System.out.println("Text blocks:");
        String json = """
                {
                    "name": "Alice",
                    "age": 25,
                    "city": "New York"
                }
                """;
        System.out.println("JSON text block:");
        System.out.println(json);

        // formatted() method on text blocks
        String template = """
                Name: %s
                Age: %d
                Salary: $%.2f
                """;
        String result = template.formatted(name, age, salary);
        System.out.println("Formatted text block:");
        System.out.println(result);
    }

    public static void main(String[] args) {
        stringBasics();
        stringMethods("Hello Java World");
        stringBuilderDemo();
        stringFormatting();
    }
}`,
    hint1: `Strings are immutable in Java - every modification creates a new String object. Use StringBuilder when building strings in loops for better performance.`,
    hint2: `Always use .equals() to compare string content, not ==. The == operator compares references. For case-insensitive comparison, use equalsIgnoreCase().`,
    whyItMatters: `String manipulation is one of the most common operations in programming. Understanding string immutability prevents memory issues, using StringBuilder improves performance in text-heavy applications, and mastering string methods is essential for data processing, parsing, and user interface development.

**Production Pattern:**
\`\`\`java
// Building SQL queries or JSON with StringBuilder
public String buildUserQuery(List<String> userIds) {
    if (userIds.isEmpty()) return "";

    StringBuilder query = new StringBuilder("SELECT * FROM users WHERE id IN (");
    for (int i = 0; i < userIds.size(); i++) {
        query.append("'").append(userIds.get(i)).append("'");
        if (i < userIds.size() - 1) query.append(", ");
    }
    query.append(")");
    return query.toString();
}
\`\`\`

**Practical Benefits:**
- Significant performance improvement when concatenating in loops
- Efficient construction of complex text structures (XML, JSON, SQL)
- Reduced load on garbage collector`,
    order: 4,
    testCode: `import static org.junit.Assert.*;
import org.junit.Test;
import java.io.ByteArrayOutputStream;
import java.io.PrintStream;

// Test1: stringBasics should show string comparison
class Test1 {
    @Test
    public void test() {
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        PrintStream oldOut = System.out;
        System.setOut(new PrintStream(out));
        StringManipulationDemo.stringBasics();
        System.setOut(oldOut);
        String output = out.toString();
        assertTrue("Should mention equals", output.contains("equals"));
    }
}

// Test2: stringBasics should show immutability
class Test2 {
    @Test
    public void test() {
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        PrintStream oldOut = System.out;
        System.setOut(new PrintStream(out));
        StringManipulationDemo.stringBasics();
        System.setOut(oldOut);
        String output = out.toString().toLowerCase();
        assertTrue("Should mention immutability or original/modified",
            output.contains("immutab") || output.contains("original") ||
            output.contains("неизменяем") || output.contains("o'zgarmas") ||
            output.contains("оригинал") || output.contains("asl"));
    }
}

// Test3: stringMethods should show length
class Test3 {
    @Test
    public void test() {
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        PrintStream oldOut = System.out;
        System.setOut(new PrintStream(out));
        StringManipulationDemo.stringMethods("Hello Java World");
        System.setOut(oldOut);
        String output = out.toString().toLowerCase();
        assertTrue("Should show length", output.contains("length") || output.contains("длина") || output.contains("uzunlik"));
    }
}

// Test4: stringMethods should show substring
class Test4 {
    @Test
    public void test() {
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        PrintStream oldOut = System.out;
        System.setOut(new PrintStream(out));
        StringManipulationDemo.stringMethods("Hello Java World");
        System.setOut(oldOut);
        String output = out.toString().toLowerCase();
        assertTrue("Should show substring", output.contains("substring") || output.contains("подстрока") || output.contains("hello"));
    }
}

// Test5: stringMethods should show indexOf or contains
class Test5 {
    @Test
    public void test() {
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        PrintStream oldOut = System.out;
        System.setOut(new PrintStream(out));
        StringManipulationDemo.stringMethods("Hello Java World");
        System.setOut(oldOut);
        String output = out.toString().toLowerCase();
        assertTrue("Should show search functionality",
            output.contains("indexof") || output.contains("contains") ||
            output.contains("search") || output.contains("поиск") || output.contains("qidirish"));
    }
}

// Test6: stringMethods should process the input text
class Test6 {
    @Test
    public void test() {
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        PrintStream oldOut = System.out;
        System.setOut(new PrintStream(out));
        StringManipulationDemo.stringMethods("TestInput");
        System.setOut(oldOut);
        String output = out.toString();
        assertTrue("Should contain or process the input text", output.contains("Test") || output.contains("9"));
    }
}

// Test7: stringBuilderDemo should demonstrate StringBuilder
class Test7 {
    @Test
    public void test() {
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        PrintStream oldOut = System.out;
        System.setOut(new PrintStream(out));
        StringManipulationDemo.stringBuilderDemo();
        System.setOut(oldOut);
        String output = out.toString();
        assertTrue("Should mention StringBuilder", output.contains("StringBuilder") || output.contains("append"));
    }
}

// Test8: stringBuilderDemo should show StringBuilder usage
class Test8 {
    @Test
    public void test() {
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        PrintStream oldOut = System.out;
        System.setOut(new PrintStream(out));
        StringManipulationDemo.stringBuilderDemo();
        System.setOut(oldOut);
        String output = out.toString();
        assertTrue("Should show StringBuilder usage",
            output.contains("StringBuilder") || output.contains("append") ||
            output.contains("Build") || output.contains("Строитель"));
    }
}

// Test9: stringFormatting should show formatted output
class Test9 {
    @Test
    public void test() {
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        PrintStream oldOut = System.out;
        System.setOut(new PrintStream(out));
        StringManipulationDemo.stringFormatting();
        System.setOut(oldOut);
        String output = out.toString().toLowerCase();
        assertTrue("Should demonstrate formatting",
            output.contains("format") || output.contains("%") ||
            output.contains("форматиров") || output.contains("formatlash"));
    }
}

// Test10: All methods should produce output
class Test10 {
    @Test
    public void test() {
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        PrintStream oldOut = System.out;
        System.setOut(new PrintStream(out));
        StringManipulationDemo.stringBasics();
        StringManipulationDemo.stringMethods("Test");
        StringManipulationDemo.stringBuilderDemo();
        StringManipulationDemo.stringFormatting();
        System.setOut(oldOut);
        String output = out.toString();
        assertTrue("All methods should produce substantial output", output.length() > 100);
    }
}
`,
    translations: {
        ru: {
            title: 'Операции со строками и StringBuilder',
            solutionCode: `public class StringManipulationDemo {

    public static void stringBasics() {
        System.out.println("=== Основы строк ===");

        // Создание строк
        String str1 = "Hello"; // Строковый литерал (хранится в пуле строк)
        String str2 = "Hello"; // Ссылается на тот же объект в пуле
        String str3 = new String("Hello"); // Новый объект в куче

        // Сравнение строк
        System.out.println("str1 == str2: " + (str1 == str2)); // true (та же ссылка)
        System.out.println("str1 == str3: " + (str1 == str3)); // false (разные ссылки)
        System.out.println("str1.equals(str3): " + str1.equals(str3)); // true (одинаковое содержимое)

        // Неизменяемость строк
        String original = "Java";
        String modified = original.concat(" Programming");
        System.out.println("");
        System.out.println("Пример неизменяемости:");
        System.out.println("Оригинал: " + original); // Все еще "Java"
        System.out.println("Измененная: " + modified); // "Java Programming"

        // Сравнение без учета регистра
        String lower = "hello";
        String upper = "HELLO";
        System.out.println("");
        System.out.println("Сравнение регистра:");
        System.out.println("equals: " + lower.equals(upper)); // false
        System.out.println("equalsIgnoreCase: " + lower.equalsIgnoreCase(upper)); // true
    }

    public static void stringMethods(String text) {
        System.out.println("");
        System.out.println("=== Методы строк ===");
        System.out.println("Исходный текст: \"" + text + "\"");

        // Длина и доступ к символам
        System.out.println("");
        System.out.println("Длина: " + text.length());
        System.out.println("Первый символ: " + text.charAt(0));
        System.out.println("Последний символ: " + text.charAt(text.length() - 1));

        // Операции с подстроками
        System.out.println("");
        System.out.println("Операции с подстроками:");
        System.out.println("substring(0, 5): \"" + text.substring(0, 5) + "\"");
        System.out.println("substring(6): \"" + text.substring(6) + "\"");

        // Поиск
        System.out.println("");
        System.out.println("Поиск:");
        System.out.println("indexOf('Java'): " + text.indexOf("Java"));
        System.out.println("lastIndexOf('o'): " + text.lastIndexOf('o'));
        System.out.println("contains('Java'): " + text.contains("Java"));
        System.out.println("startsWith('Hello'): " + text.startsWith("Hello"));
        System.out.println("endsWith('World'): " + text.endsWith("World"));

        // Разделение и объединение
        System.out.println("");
        System.out.println("Разделение:");
        String[] words = text.split(" ");
        System.out.print("Разделено по пробелу: [");
        for (int i = 0; i < words.length; i++) {
            System.out.print("\"" + words[i] + "\"");
            if (i < words.length - 1) System.out.print(", ");
        }
        System.out.println("]");

        String joined = String.join("-", words);
        System.out.println("Объединено с '-': " + joined);

        // Обрезка и замена
        String padded = "  Java  ";
        System.out.println("");
        System.out.println("Обрезка:");
        System.out.println("До trim: \"" + padded + "\" (длина: " + padded.length() + ")");
        System.out.println("После trim: \"" + padded.trim() + "\" (длина: " + padded.trim().length() + ")");

        System.out.println("");
        System.out.println("Замена:");
        System.out.println("replace('Java', 'Python'): " + text.replace("Java", "Python"));
        System.out.println("replaceAll('\\\\s+', '-'): " + text.replaceAll("\\s+", "-"));

        // Преобразование регистра
        System.out.println("");
        System.out.println("Преобразование регистра:");
        System.out.println("toUpperCase(): " + text.toUpperCase());
        System.out.println("toLowerCase(): " + text.toLowerCase());
    }

    public static void stringBuilderDemo() {
        System.out.println("");
        System.out.println("=== Демонстрация StringBuilder ===");

        // StringBuilder для эффективного построения строк
        StringBuilder sb = new StringBuilder();
        sb.append("Java");
        sb.append(" это");
        sb.append(" круто");
        System.out.println("Построенная строка: " + sb.toString());

        // Методы StringBuilder
        System.out.println("");
        System.out.println("Операции StringBuilder:");
        StringBuilder builder = new StringBuilder("Hello");
        System.out.println("Начальное: " + builder);

        builder.append(" World");
        System.out.println("После append: " + builder);

        builder.insert(5, " Java");
        System.out.println("После insert: " + builder);

        builder.delete(5, 10);
        System.out.println("После delete: " + builder);

        builder.reverse();
        System.out.println("После reverse: " + builder);

        builder.reverse(); // Обратно
        builder.replace(0, 5, "Привет");
        System.out.println("После replace: " + builder);

        // Сравнение производительности
        System.out.println("");
        System.out.println("Пример производительности:");
        long startTime, endTime;

        // Использование конкатенации String (медленнее для многих операций)
        startTime = System.nanoTime();
        String result1 = "";
        for (int i = 0; i < 1000; i++) {
            result1 += "a";
        }
        endTime = System.nanoTime();
        System.out.println("Время конкатенации String: " + (endTime - startTime) + " нс");

        // Использование StringBuilder (намного быстрее)
        startTime = System.nanoTime();
        StringBuilder sb2 = new StringBuilder();
        for (int i = 0; i < 1000; i++) {
            sb2.append("a");
        }
        String result2 = sb2.toString();
        endTime = System.nanoTime();
        System.out.println("Время StringBuilder: " + (endTime - startTime) + " нс");
    }

    public static void stringFormatting() {
        System.out.println("");
        System.out.println("=== Форматирование строк ===");

        String name = "Алиса";
        int age = 25;
        double salary = 75000.50;

        // String.format()
        String formatted1 = String.format("Имя: %s, Возраст: %d, Зарплата: $%.2f", name, age, salary);
        System.out.println("String.format(): " + formatted1);

        // printf (печатает напрямую)
        System.out.print("printf(): ");
        System.out.printf("Имя: %s, Возраст: %d, Зарплата: $%.2f%n", name, age, salary);

        // Форматирование чисел
        System.out.println("");
        System.out.println("Форматирование чисел:");
        System.out.printf("Целое с ведущими нулями: %05d%n", 42);
        System.out.printf("Число с 3 десятичными знаками: %.3f%n", 3.14159);
        System.out.printf("Научная нотация: %e%n", 123456.789);
        System.out.printf("Процент: %.1f%%%n", 85.5);

        // Выравнивание
        System.out.println("");
        System.out.println("Выравнивание:");
        System.out.printf("По левому краю: %-10s|%n", "Java");
        System.out.printf("По правому краю: %10s|%n", "Java");

        // Текстовые блоки (Java 15+)
        System.out.println("");
        System.out.println("Текстовые блоки:");
        String json = """
                {
                    "name": "Алиса",
                    "age": 25,
                    "city": "Москва"
                }
                """;
        System.out.println("JSON текстовый блок:");
        System.out.println(json);

        // Метод formatted() для текстовых блоков
        String template = """
                Имя: %s
                Возраст: %d
                Зарплата: $%.2f
                """;
        String result = template.formatted(name, age, salary);
        System.out.println("Отформатированный текстовый блок:");
        System.out.println(result);
    }

    public static void main(String[] args) {
        stringBasics();
        stringMethods("Hello Java World");
        stringBuilderDemo();
        stringFormatting();
    }
}`,
            description: `Освойте операции со строками Java, манипуляции и эффективное построение строк.

**Требования:**
1. Создайте метод \`stringBasics()\`, демонстрирующий создание строк, сравнение и неизменяемость
2. Реализуйте метод \`stringMethods(String text)\`, показывающий общие операции со строками
3. Создайте метод \`stringBuilderDemo()\`, демонстрирующий StringBuilder для эффективной конкатенации
4. Реализуйте метод \`stringFormatting()\`, показывающий различные техники форматирования

**Операции со строками для изучения:**
- Создание строк и литералы
- Сравнение строк (==, equals, equalsIgnoreCase)
- Общие методы: length, charAt, substring, indexOf, contains
- Разделение и объединение строк
- Форматирование строк (String.format, printf)
- StringBuilder для эффективной конкатенации
- Текстовые блоки (Java 15+)

**Пример вывода:**
\`\`\`
Длина: 13
Подстрока: World
Содержит "Java": true
Разделенные слова: [Hello, Java, World]
\`\`\``,
            hint1: `Строки в Java неизменяемы - каждая модификация создает новый объект String. Используйте StringBuilder при построении строк в циклах для лучшей производительности.`,
            hint2: `Всегда используйте .equals() для сравнения содержимого строк, а не ==. Оператор == сравнивает ссылки. Для сравнения без учета регистра используйте equalsIgnoreCase().`,
            whyItMatters: `Манипуляция со строками - одна из самых распространенных операций в программировании. Понимание неизменяемости строк предотвращает проблемы с памятью, использование StringBuilder улучшает производительность в текстоемких приложениях, а освоение методов строк необходимо для обработки данных, парсинга и разработки пользовательских интерфейсов.

**Продакшен паттерн:**
\`\`\`java
// Построение SQL запросов или JSON с StringBuilder
public String buildUserQuery(List<String> userIds) {
    if (userIds.isEmpty()) return "";

    StringBuilder query = new StringBuilder("SELECT * FROM users WHERE id IN (");
    for (int i = 0; i < userIds.size(); i++) {
        query.append("'").append(userIds.get(i)).append("'");
        if (i < userIds.size() - 1) query.append(", ");
    }
    query.append(")");
    return query.toString();
}
\`\`\`

**Практические преимущества:**
- Значительное улучшение производительности при конкатенации в циклах
- Эффективное построение сложных текстовых структур (XML, JSON, SQL)
- Снижение нагрузки на garbage collector`
        },
        uz: {
            title: `String operatsiyalari va StringBuilder`,
            solutionCode: `public class StringManipulationDemo {

    public static void stringBasics() {
        System.out.println("=== String asoslari ===");

        // String yaratish
        String str1 = "Hello"; // String literal (String poolda saqlanadi)
        String str2 = "Hello"; // Pooldagi bir xil obyektga havola
        String str3 = new String("Hello"); // Heap da yangi obyekt

        // String taqqoslash
        System.out.println("str1 == str2: " + (str1 == str2)); // true (bir xil havola)
        System.out.println("str1 == str3: " + (str1 == str3)); // false (turli havolalar)
        System.out.println("str1.equals(str3): " + str1.equals(str3)); // true (bir xil tarkib)

        // String o'zgarmasligi
        String original = "Java";
        String modified = original.concat(" Programming");
        System.out.println("");
        System.out.println("O'zgarmaslik misoli:");
        System.out.println("Asl: " + original); // Hali ham "Java"
        System.out.println("O'zgartirilgan: " + modified); // "Java Programming"

        // Katta-kichik harfga sezgir bo'lmagan taqqoslash
        String lower = "hello";
        String upper = "HELLO";
        System.out.println("");
        System.out.println("Registr taqqoslash:");
        System.out.println("equals: " + lower.equals(upper)); // false
        System.out.println("equalsIgnoreCase: " + lower.equalsIgnoreCase(upper)); // true
    }

    public static void stringMethods(String text) {
        System.out.println("");
        System.out.println("=== String metodlari ===");
        System.out.println("Asl matn: \"" + text + "\"");

        // Uzunlik va belgiga kirish
        System.out.println("");
        System.out.println("Uzunlik: " + text.length());
        System.out.println("Birinchi belgi: " + text.charAt(0));
        System.out.println("Oxirgi belgi: " + text.charAt(text.length() - 1));

        // Substring operatsiyalari
        System.out.println("");
        System.out.println("Substring operatsiyalari:");
        System.out.println("substring(0, 5): \"" + text.substring(0, 5) + "\"");
        System.out.println("substring(6): \"" + text.substring(6) + "\"");

        // Qidirish
        System.out.println("");
        System.out.println("Qidirish:");
        System.out.println("indexOf('Java'): " + text.indexOf("Java"));
        System.out.println("lastIndexOf('o'): " + text.lastIndexOf('o'));
        System.out.println("contains('Java'): " + text.contains("Java"));
        System.out.println("startsWith('Hello'): " + text.startsWith("Hello"));
        System.out.println("endsWith('World'): " + text.endsWith("World"));

        // Bo'lish va birlashtirish
        System.out.println("");
        System.out.println("Bo'lish:");
        String[] words = text.split(" ");
        System.out.print("Bo'sh joy bo'yicha bo'lish: [");
        for (int i = 0; i < words.length; i++) {
            System.out.print("\"" + words[i] + "\"");
            if (i < words.length - 1) System.out.print(", ");
        }
        System.out.println("]");

        String joined = String.join("-", words);
        System.out.println("'-' bilan birlashtirilgan: " + joined);

        // Kesish va almashtirish
        String padded = "  Java  ";
        System.out.println("");
        System.out.println("Kesish:");
        System.out.println("trim dan oldin: \"" + padded + "\" (uzunlik: " + padded.length() + ")");
        System.out.println("trim dan keyin: \"" + padded.trim() + "\" (uzunlik: " + padded.trim().length() + ")");

        System.out.println("");
        System.out.println("Almashtirish:");
        System.out.println("replace('Java', 'Python'): " + text.replace("Java", "Python"));
        System.out.println("replaceAll('\\\\s+', '-'): " + text.replaceAll("\\s+", "-"));

        // Registrni o'zgartirish
        System.out.println("");
        System.out.println("Registrni o'zgartirish:");
        System.out.println("toUpperCase(): " + text.toUpperCase());
        System.out.println("toLowerCase(): " + text.toLowerCase());
    }

    public static void stringBuilderDemo() {
        System.out.println("");
        System.out.println("=== StringBuilder namoyishi ===");

        // Samarali string qurish uchun StringBuilder
        StringBuilder sb = new StringBuilder();
        sb.append("Java");
        sb.append(" juda");
        sb.append(" zo'r");
        System.out.println("Qurilgan string: " + sb.toString());

        // StringBuilder metodlari
        System.out.println("");
        System.out.println("StringBuilder operatsiyalari:");
        StringBuilder builder = new StringBuilder("Hello");
        System.out.println("Boshlang'ich: " + builder);

        builder.append(" World");
        System.out.println("append dan keyin: " + builder);

        builder.insert(5, " Java");
        System.out.println("insert dan keyin: " + builder);

        builder.delete(5, 10);
        System.out.println("delete dan keyin: " + builder);

        builder.reverse();
        System.out.println("reverse dan keyin: " + builder);

        builder.reverse(); // Orqaga
        builder.replace(0, 5, "Salom");
        System.out.println("replace dan keyin: " + builder);

        // Ishlash taqqoslash
        System.out.println("");
        System.out.println("Ishlash misoli:");
        long startTime, endTime;

        // String biriktirishdan foydalanish (ko'p operatsiyalar uchun sekinroq)
        startTime = System.nanoTime();
        String result1 = "";
        for (int i = 0; i < 1000; i++) {
            result1 += "a";
        }
        endTime = System.nanoTime();
        System.out.println("String biriktirish vaqti: " + (endTime - startTime) + " ns");

        // StringBuilder dan foydalanish (ancha tezroq)
        startTime = System.nanoTime();
        StringBuilder sb2 = new StringBuilder();
        for (int i = 0; i < 1000; i++) {
            sb2.append("a");
        }
        String result2 = sb2.toString();
        endTime = System.nanoTime();
        System.out.println("StringBuilder vaqti: " + (endTime - startTime) + " ns");
    }

    public static void stringFormatting() {
        System.out.println("");
        System.out.println("=== String formatlash ===");

        String name = "Olim";
        int age = 25;
        double salary = 75000.50;

        // String.format()
        String formatted1 = String.format("Ism: %s, Yosh: %d, Maosh: $%.2f", name, age, salary);
        System.out.println("String.format(): " + formatted1);

        // printf (to'g'ridan-to'g'ri chop etadi)
        System.out.print("printf(): ");
        System.out.printf("Ism: %s, Yosh: %d, Maosh: $%.2f%n", name, age, salary);

        // Raqamlarni formatlash
        System.out.println("");
        System.out.println("Raqamlarni formatlash:");
        System.out.printf("Boshidagi nollar bilan butun son: %05d%n", 42);
        System.out.printf("3 o'nlik bilan float: %.3f%n", 3.14159);
        System.out.printf("Ilmiy notatsiya: %e%n", 123456.789);
        System.out.printf("Foiz: %.1f%%%n", 85.5);

        // Tekislash
        System.out.println("");
        System.out.println("Tekislash:");
        System.out.printf("Chapga tekislangan: %-10s|%n", "Java");
        System.out.printf("O'ngga tekislangan: %10s|%n", "Java");

        // Matn bloklari (Java 15+)
        System.out.println("");
        System.out.println("Matn bloklari:");
        String json = """
                {
                    "name": "Olim",
                    "age": 25,
                    "city": "Toshkent"
                }
                """;
        System.out.println("JSON matn bloki:");
        System.out.println(json);

        // Matn bloklarida formatted() metodi
        String template = """
                Ism: %s
                Yosh: %d
                Maosh: $%.2f
                """;
        String result = template.formatted(name, age, salary);
        System.out.println("Formatlangan matn bloki:");
        System.out.println(result);
    }

    public static void main(String[] args) {
        stringBasics();
        stringMethods("Hello Java World");
        stringBuilderDemo();
        stringFormatting();
    }
}`,
            description: `Java string operatsiyalari, boshqarish va samarali string qurish.

**Talablar:**
1. String yaratish, taqqoslash va o'zgarmaslikni ko'rsatadigan \`stringBasics()\` metodini yarating
2. Umumiy string operatsiyalarini ko'rsatadigan \`stringMethods(String text)\` metodini yarating
3. Samarali biriktirish uchun StringBuilder ni ko'rsatadigan \`stringBuilderDemo()\` metodini yarating
4. Turli formatlash usullarini ko'rsatadigan \`stringFormatting()\` metodini yarating

**O'rganish uchun string operatsiyalari:**
- String yaratish va literallar
- String taqqoslash (==, equals, equalsIgnoreCase)
- Umumiy metodlar: length, charAt, substring, indexOf, contains
- Stringni bo'lish va birlashtirish
- String formatlash (String.format, printf)
- Samarali biriktirish uchun StringBuilder
- Matn bloklari (Java 15+)

**Chiqish namunasi:**
\`\`\`
Uzunlik: 13
Substring: World
"Java" ni o'z ichiga oladi: true
Bo'lingan so'zlar: [Hello, Java, World]
\`\`\``,
            hint1: `Java da stringlar o'zgarmasdir - har bir o'zgartirish yangi String obyektini yaratadi. Sikllarda string qurish uchun yaxshi ishlash uchun StringBuilder dan foydalaning.`,
            hint2: `String tarkibini taqqoslash uchun doim .equals() dan foydalaning, == emas. == operatori havolalarni taqqoslaydi. Katta-kichik harfga sezgir bo'lmagan taqqoslash uchun equalsIgnoreCase() dan foydalaning.`,
            whyItMatters: `String boshqarish dasturchilikning eng keng tarqalgan operatsiyalaridan biridir. String o'zgarmasligini tushunish xotira muammolarini oldini oladi, StringBuilder dan foydalanish matn ko'p dasturlarda ishlashni yaxshilaydi va string metodlarini o'zlashtirish ma'lumotlarni qayta ishlash, parsing va foydalanuvchi interfeysi ishlab chiqish uchun zarur.

**Ishlab chiqarish patterni:**
\`\`\`java
// StringBuilder yordamida SQL so'rovlari yoki JSON qurish
public String buildUserQuery(List<String> userIds) {
    if (userIds.isEmpty()) return "";

    StringBuilder query = new StringBuilder("SELECT * FROM users WHERE id IN (");
    for (int i = 0; i < userIds.size(); i++) {
        query.append("'").append(userIds.get(i)).append("'");
        if (i < userIds.size() - 1) query.append(", ");
    }
    query.append(")");
    return query.toString();
}
\`\`\`

**Amaliy foydalari:**
- Sikllarda birlashtirish vaqtida ishlashni sezilarli yaxshilash
- Murakkab matn strukturalarini samarali qurish (XML, JSON, SQL)
- Garbage collector yukini kamaytirish`
        }
    }
};

export default task;
