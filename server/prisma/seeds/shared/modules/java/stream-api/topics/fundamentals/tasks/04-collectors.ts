import { Task } from '../../../../types';

export const task: Task = {
    slug: 'java-collectors',
    title: 'Collectors',
    difficulty: 'medium',
    tags: ['java', 'stream-api', 'collectors', 'grouping', 'partitioning', 'java8'],
    estimatedTime: '35m',
    isPremium: false,
    youtubeUrl: '',
    description: `# Collectors

Collectors are powerful utilities for accumulating stream elements into collections and performing complex operations. The Collectors class provides many predefined collectors for common operations like toList, toSet, toMap, groupingBy, partitioningBy, joining, and statistical operations.

## Requirements:
1. Demonstrate basic collectors:
   1.1. toList() - collect to List
   1.2. toSet() - collect to Set
   1.3. toMap() - collect to Map with key and value mappers
   1.4. toCollection() - collect to specific collection type

2. Demonstrate groupingBy():
   2.1. Group by single property
   2.2. Group with downstream collector
   2.3. Multi-level grouping

3. Demonstrate partitioningBy():
   3.1. Partition into two groups based on predicate
   3.2. Partition with downstream collector

4. Demonstrate string collectors:
   4.1. joining() - concatenate strings with delimiter
   4.2. joining() with prefix and suffix

5. Demonstrate statistical collectors:
   5.1. counting()
   5.2. summingInt()
   5.3. averagingInt()
   5.4. summarizingInt()

## Example Output:
\`\`\`
=== Basic Collectors ===
To List: [Alice, Bob, Charlie, David]
To Set: [Alice, Bob, Charlie, David]
To Map: {Alice=5, Bob=3, Charlie=7, David=5}

=== groupingBy() ===
Group by length: {3=[Bob], 5=[Alice, David], 7=[Charlie]}
Group by length with count: {3=1, 5=2, 7=1}
Group by first letter: {A=[Alice], B=[Bob], C=[Charlie], D=[David]}

=== partitioningBy() ===
Partition by length > 4: {false=[Bob], true=[Alice, Charlie, David]}
Partition with count: {false=1, true=3}

=== String Collectors ===
Joining: Alice, Bob, Charlie, David
With brackets: [Alice, Bob, Charlie, David]

=== Statistical Collectors ===
Count: 5
Sum: 150
Average: 30.0
Statistics: IntSummaryStatistics{count=5, sum=150, min=10, average=30.0, max=50}
\`\`\``,
    initialCode: `import java.util.*;
import java.util.stream.*;

public class CollectorsDemo {
    public static void main(String[] args) {
        // TODO: Demonstrate basic collectors

        // TODO: Demonstrate groupingBy()

        // TODO: Demonstrate partitioningBy()

        // TODO: Demonstrate string collectors

        // TODO: Demonstrate statistical collectors
    }
}`,
    solutionCode: `import java.util.*;
import java.util.stream.*;

public class CollectorsDemo {
    public static void main(String[] args) {
        System.out.println("=== Basic Collectors ===");

        List<String> names = Arrays.asList("Alice", "Bob", "Charlie", "David");

        // Collect to List
        List<String> list = names.stream().collect(Collectors.toList());
        System.out.println("To List: " + list);

        // Collect to Set
        Set<String> set = names.stream().collect(Collectors.toSet());
        System.out.println("To Set: " + set);

        // Collect to Map (name -> length)
        Map<String, Integer> nameToLength = names.stream()
            .collect(Collectors.toMap(
                name -> name,	// key mapper
                name -> name.length()	// value mapper
            ));
        System.out.println("To Map: " + nameToLength);

        System.out.println("\\n=== groupingBy() ===");

        // Group by string length
        Map<Integer, List<String>> groupedByLength = names.stream()
            .collect(Collectors.groupingBy(String::length));
        System.out.println("Group by length: " + groupedByLength);

        // Group by length with counting
        Map<Integer, Long> lengthCounts = names.stream()
            .collect(Collectors.groupingBy(
                String::length,
                Collectors.counting()
            ));
        System.out.println("Group by length with count: " + lengthCounts);

        // Group by first letter
        Map<Character, List<String>> groupedByFirstLetter = names.stream()
            .collect(Collectors.groupingBy(name -> name.charAt(0)));
        System.out.println("Group by first letter: " + groupedByFirstLetter);

        System.out.println("\\n=== partitioningBy() ===");

        // Partition by length > 4
        Map<Boolean, List<String>> partitioned = names.stream()
            .collect(Collectors.partitioningBy(name -> name.length() > 4));
        System.out.println("Partition by length > 4: " + partitioned);

        // Partition with counting
        Map<Boolean, Long> partitionCounts = names.stream()
            .collect(Collectors.partitioningBy(
                name -> name.length() > 4,
                Collectors.counting()
            ));
        System.out.println("Partition with count: " + partitionCounts);

        System.out.println("\\n=== String Collectors ===");

        // Joining with delimiter
        String joined = names.stream()
            .collect(Collectors.joining(", "));
        System.out.println("Joining: " + joined);

        // Joining with delimiter, prefix, and suffix
        String joinedWithBrackets = names.stream()
            .collect(Collectors.joining(", ", "[", "]"));
        System.out.println("With brackets: " + joinedWithBrackets);

        System.out.println("\\n=== Statistical Collectors ===");

        List<Integer> numbers = Arrays.asList(10, 20, 30, 40, 50);

        // Count
        Long count = numbers.stream()
            .collect(Collectors.counting());
        System.out.println("Count: " + count);

        // Sum
        Integer sum = numbers.stream()
            .collect(Collectors.summingInt(Integer::intValue));
        System.out.println("Sum: " + sum);

        // Average
        Double average = numbers.stream()
            .collect(Collectors.averagingInt(Integer::intValue));
        System.out.println("Average: " + average);

        // Summary statistics (count, sum, min, max, average)
        IntSummaryStatistics stats = numbers.stream()
            .collect(Collectors.summarizingInt(Integer::intValue));
        System.out.println("Statistics: " + stats);
    }
}`,
    testCode: `import static org.junit.Assert.*;
import org.junit.Test;
import java.util.*;
import java.util.stream.*;

// Test1: Verify toList() collector
class Test1 {
    @Test
    public void test() {
        List<String> names = Arrays.asList("Alice", "Bob", "Charlie");
        List<String> list = names.stream().collect(Collectors.toList());
        assertEquals(3, list.size());
        assertTrue(list.contains("Alice"));
    }
}

// Test2: Verify toSet() removes duplicates
class Test2 {
    @Test
    public void test() {
        List<Integer> numbers = Arrays.asList(1, 2, 2, 3, 3, 3);
        Set<Integer> set = numbers.stream().collect(Collectors.toSet());
        assertEquals(3, set.size());
    }
}

// Test3: Verify toMap() creates map from stream
class Test3 {
    @Test
    public void test() {
        List<String> words = Arrays.asList("a", "bb", "ccc");
        Map<String, Integer> map = words.stream()
            .collect(Collectors.toMap(s -> s, String::length));
        assertEquals(Integer.valueOf(3), map.get("ccc"));
    }
}

// Test4: Verify groupingBy() groups elements
class Test4 {
    @Test
    public void test() {
        List<String> words = Arrays.asList("a", "bb", "c", "dd");
        Map<Integer, List<String>> grouped = words.stream()
            .collect(Collectors.groupingBy(String::length));
        assertEquals(2, grouped.get(2).size());
    }
}

// Test5: Verify partitioningBy() creates boolean map
class Test5 {
    @Test
    public void test() {
        List<Integer> numbers = Arrays.asList(1, 2, 3, 4, 5);
        Map<Boolean, List<Integer>> partitioned = numbers.stream()
            .collect(Collectors.partitioningBy(n -> n % 2 == 0));
        assertEquals(2, partitioned.get(true).size());
        assertEquals(3, partitioned.get(false).size());
    }
}

// Test6: Verify joining() concatenates strings
class Test6 {
    @Test
    public void test() {
        List<String> words = Arrays.asList("Hello", "World");
        String joined = words.stream().collect(Collectors.joining(" "));
        assertEquals("Hello World", joined);
    }
}

// Test7: Verify joining() with delimiter and affixes
class Test7 {
    @Test
    public void test() {
        List<String> items = Arrays.asList("a", "b", "c");
        String result = items.stream().collect(Collectors.joining(", ", "[", "]"));
        assertEquals("[a, b, c]", result);
    }
}

// Test8: Verify counting() collector
class Test8 {
    @Test
    public void test() {
        List<Integer> numbers = Arrays.asList(1, 2, 3, 4, 5);
        Long count = numbers.stream().collect(Collectors.counting());
        assertEquals(Long.valueOf(5), count);
    }
}

// Test9: Verify summingInt() collector
class Test9 {
    @Test
    public void test() {
        List<Integer> numbers = Arrays.asList(1, 2, 3, 4, 5);
        Integer sum = numbers.stream().collect(Collectors.summingInt(Integer::intValue));
        assertEquals(Integer.valueOf(15), sum);
    }
}

// Test10: Verify averagingInt() collector
class Test10 {
    @Test
    public void test() {
        List<Integer> numbers = Arrays.asList(2, 4, 6);
        Double avg = numbers.stream().collect(Collectors.averagingInt(Integer::intValue));
        assertEquals(4.0, avg, 0.001);
    }
}
`,
    hint1: `Collectors.toList() and toSet() are simple collectors. Collectors.toMap() needs key and value mappers. Use groupingBy() to group elements by a classifier function.`,
    hint2: `groupingBy() can take a downstream collector as second parameter. partitioningBy() is a special case of grouping that splits into exactly two groups based on a predicate.`,
    whyItMatters: `Collectors are essential for transforming stream results into useful data structures. They enable powerful operations like grouping, partitioning, and statistical analysis. Mastering collectors allows you to write concise code for complex data aggregation tasks that would otherwise require verbose loops.

**Production Pattern:**
\`\`\`java
// Grouping transactions by status with sum calculation
Map<TransactionStatus, BigDecimal> totalByStatus = transactions.stream()
    .collect(Collectors.groupingBy(
        Transaction::getStatus,
        Collectors.reducing(
            BigDecimal.ZERO,
            Transaction::getAmount,
            BigDecimal::add
        )
    ));
\`\`\`

**Practical Benefits:**
- Complex data analysis without SQL queries
- Multi-level grouping for sophisticated reporting
- Efficient processing of large datasets in memory`,
    order: 4,
    translations: {
        ru: {
            title: 'Коллекторы',
            solutionCode: `import java.util.*;
import java.util.stream.*;

public class CollectorsDemo {
    public static void main(String[] args) {
        System.out.println("=== Основные коллекторы ===");

        List<String> names = Arrays.asList("Alice", "Bob", "Charlie", "David");

        // Собрать в List
        List<String> list = names.stream().collect(Collectors.toList());
        System.out.println("To List: " + list);

        // Собрать в Set
        Set<String> set = names.stream().collect(Collectors.toSet());
        System.out.println("To Set: " + set);

        // Собрать в Map (имя -> длина)
        Map<String, Integer> nameToLength = names.stream()
            .collect(Collectors.toMap(
                name -> name,	// отображение ключа
                name -> name.length()	// отображение значения
            ));
        System.out.println("To Map: " + nameToLength);

        System.out.println("\\n=== groupingBy() ===");

        // Группировка по длине строки
        Map<Integer, List<String>> groupedByLength = names.stream()
            .collect(Collectors.groupingBy(String::length));
        System.out.println("Group by length: " + groupedByLength);

        // Группировка по длине с подсчетом
        Map<Integer, Long> lengthCounts = names.stream()
            .collect(Collectors.groupingBy(
                String::length,
                Collectors.counting()
            ));
        System.out.println("Group by length with count: " + lengthCounts);

        // Группировка по первой букве
        Map<Character, List<String>> groupedByFirstLetter = names.stream()
            .collect(Collectors.groupingBy(name -> name.charAt(0)));
        System.out.println("Group by first letter: " + groupedByFirstLetter);

        System.out.println("\\n=== partitioningBy() ===");

        // Разделение по длине > 4
        Map<Boolean, List<String>> partitioned = names.stream()
            .collect(Collectors.partitioningBy(name -> name.length() > 4));
        System.out.println("Partition by length > 4: " + partitioned);

        // Разделение с подсчетом
        Map<Boolean, Long> partitionCounts = names.stream()
            .collect(Collectors.partitioningBy(
                name -> name.length() > 4,
                Collectors.counting()
            ));
        System.out.println("Partition with count: " + partitionCounts);

        System.out.println("\\n=== Строковые коллекторы ===");

        // Объединение с разделителем
        String joined = names.stream()
            .collect(Collectors.joining(", "));
        System.out.println("Joining: " + joined);

        // Объединение с разделителем, префиксом и суффиксом
        String joinedWithBrackets = names.stream()
            .collect(Collectors.joining(", ", "[", "]"));
        System.out.println("With brackets: " + joinedWithBrackets);

        System.out.println("\\n=== Статистические коллекторы ===");

        List<Integer> numbers = Arrays.asList(10, 20, 30, 40, 50);

        // Подсчет
        Long count = numbers.stream()
            .collect(Collectors.counting());
        System.out.println("Count: " + count);

        // Сумма
        Integer sum = numbers.stream()
            .collect(Collectors.summingInt(Integer::intValue));
        System.out.println("Sum: " + sum);

        // Среднее
        Double average = numbers.stream()
            .collect(Collectors.averagingInt(Integer::intValue));
        System.out.println("Average: " + average);

        // Суммарная статистика (количество, сумма, мин, макс, среднее)
        IntSummaryStatistics stats = numbers.stream()
            .collect(Collectors.summarizingInt(Integer::intValue));
        System.out.println("Statistics: " + stats);
    }
}`,
            description: `# Коллекторы

Коллекторы - это мощные утилиты для накопления элементов потока в коллекции и выполнения сложных операций. Класс Collectors предоставляет множество предопределенных коллекторов для общих операций, таких как toList, toSet, toMap, groupingBy, partitioningBy, joining и статистических операций.

## Требования:
1. Продемонстрируйте базовые коллекторы:
   1.1. toList() - сбор в List
   1.2. toSet() - сбор в Set
   1.3. toMap() - сбор в Map с отображением ключей и значений
   1.4. toCollection() - сбор в конкретный тип коллекции

2. Продемонстрируйте groupingBy():
   2.1. Группировка по одному свойству
   2.2. Группировка с downstream коллектором
   2.3. Многоуровневая группировка

3. Продемонстрируйте partitioningBy():
   3.1. Разделение на две группы на основе предиката
   3.2. Разделение с downstream коллектором

4. Продемонстрируйте строковые коллекторы:
   4.1. joining() - конкатенация строк с разделителем
   4.2. joining() с префиксом и суффиксом

5. Продемонстрируйте статистические коллекторы:
   5.1. counting()
   5.2. summingInt()
   5.3. averagingInt()
   5.4. summarizingInt()

## Пример вывода:
\`\`\`
=== Basic Collectors ===
To List: [Alice, Bob, Charlie, David]
To Set: [Alice, Bob, Charlie, David]
To Map: {Alice=5, Bob=3, Charlie=7, David=5}

=== groupingBy() ===
Group by length: {3=[Bob], 5=[Alice, David], 7=[Charlie]}
Group by length with count: {3=1, 5=2, 7=1}
Group by first letter: {A=[Alice], B=[Bob], C=[Charlie], D=[David]}

=== partitioningBy() ===
Partition by length > 4: {false=[Bob], true=[Alice, Charlie, David]}
Partition with count: {false=1, true=3}

=== String Collectors ===
Joining: Alice, Bob, Charlie, David
With brackets: [Alice, Bob, Charlie, David]

=== Statistical Collectors ===
Count: 5
Sum: 150
Average: 30.0
Statistics: IntSummaryStatistics{count=5, sum=150, min=10, average=30.0, max=50}
\`\`\``,
            hint1: `Collectors.toList() и toSet() - это простые коллекторы. Collectors.toMap() нуждается в отображении ключей и значений. Используйте groupingBy() для группировки элементов по функции классификации.`,
            hint2: `groupingBy() может принимать downstream коллектор в качестве второго параметра. partitioningBy() - это особый случай группировки, который разделяет ровно на две группы на основе предиката.`,
            whyItMatters: `Коллекторы необходимы для преобразования результатов потока в полезные структуры данных. Они обеспечивают мощные операции, такие как группировка, разделение и статистический анализ. Освоение коллекторов позволяет писать краткий код для сложных задач агрегации данных, которые в противном случае потребовали бы многословных циклов.

**Продакшен паттерн:**
\`\`\`java
// Группировка транзакций по статусу с подсчётом суммы
Map<TransactionStatus, BigDecimal> totalByStatus = transactions.stream()
    .collect(Collectors.groupingBy(
        Transaction::getStatus,
        Collectors.reducing(
            BigDecimal.ZERO,
            Transaction::getAmount,
            BigDecimal::add
        )
    ));
\`\`\`

**Практические преимущества:**
- Комплексная аналитика данных без SQL запросов
- Многоуровневая группировка для сложной отчётности
- Эффективная обработка больших наборов данных в памяти`
        },
        uz: {
            title: 'Kollektorlar',
            solutionCode: `import java.util.*;
import java.util.stream.*;

public class CollectorsDemo {
    public static void main(String[] args) {
        System.out.println("=== Asosiy kollektorlar ===");

        List<String> names = Arrays.asList("Alice", "Bob", "Charlie", "David");

        // List ga yig'ish
        List<String> list = names.stream().collect(Collectors.toList());
        System.out.println("To List: " + list);

        // Set ga yig'ish
        Set<String> set = names.stream().collect(Collectors.toSet());
        System.out.println("To Set: " + set);

        // Map ga yig'ish (ism -> uzunlik)
        Map<String, Integer> nameToLength = names.stream()
            .collect(Collectors.toMap(
                name -> name,	// kalit aks ettirish
                name -> name.length()	// qiymat aks ettirish
            ));
        System.out.println("To Map: " + nameToLength);

        System.out.println("\\n=== groupingBy() ===");

        // Satr uzunligi bo'yicha guruhlash
        Map<Integer, List<String>> groupedByLength = names.stream()
            .collect(Collectors.groupingBy(String::length));
        System.out.println("Group by length: " + groupedByLength);

        // Uzunlik bo'yicha guruhlash va hisoblash
        Map<Integer, Long> lengthCounts = names.stream()
            .collect(Collectors.groupingBy(
                String::length,
                Collectors.counting()
            ));
        System.out.println("Group by length with count: " + lengthCounts);

        // Birinchi harf bo'yicha guruhlash
        Map<Character, List<String>> groupedByFirstLetter = names.stream()
            .collect(Collectors.groupingBy(name -> name.charAt(0)));
        System.out.println("Group by first letter: " + groupedByFirstLetter);

        System.out.println("\\n=== partitioningBy() ===");

        // Uzunlik > 4 bo'yicha bo'lish
        Map<Boolean, List<String>> partitioned = names.stream()
            .collect(Collectors.partitioningBy(name -> name.length() > 4));
        System.out.println("Partition by length > 4: " + partitioned);

        // Hisoblash bilan bo'lish
        Map<Boolean, Long> partitionCounts = names.stream()
            .collect(Collectors.partitioningBy(
                name -> name.length() > 4,
                Collectors.counting()
            ));
        System.out.println("Partition with count: " + partitionCounts);

        System.out.println("\\n=== Satr kollektorlari ===");

        // Ajratuvchi bilan birlashtirish
        String joined = names.stream()
            .collect(Collectors.joining(", "));
        System.out.println("Joining: " + joined);

        // Ajratuvchi, prefiks va suffiks bilan birlashtirish
        String joinedWithBrackets = names.stream()
            .collect(Collectors.joining(", ", "[", "]"));
        System.out.println("With brackets: " + joinedWithBrackets);

        System.out.println("\\n=== Statistik kollektorlar ===");

        List<Integer> numbers = Arrays.asList(10, 20, 30, 40, 50);

        // Hisoblash
        Long count = numbers.stream()
            .collect(Collectors.counting());
        System.out.println("Count: " + count);

        // Yig'indi
        Integer sum = numbers.stream()
            .collect(Collectors.summingInt(Integer::intValue));
        System.out.println("Sum: " + sum);

        // O'rtacha
        Double average = numbers.stream()
            .collect(Collectors.averagingInt(Integer::intValue));
        System.out.println("Average: " + average);

        // Umumiy statistika (soni, yig'indi, min, maks, o'rtacha)
        IntSummaryStatistics stats = numbers.stream()
            .collect(Collectors.summarizingInt(Integer::intValue));
        System.out.println("Statistics: " + stats);
    }
}`,
            description: `# Kollektorlar

Kollektorlar stream elementlarini kolleksiyalarga yig'ish va murakkab operatsiyalarni bajarish uchun kuchli vositalardir. Collectors klassi toList, toSet, toMap, groupingBy, partitioningBy, joining va statistik operatsiyalar kabi umumiy operatsiyalar uchun ko'plab oldindan belgilangan kollektorlarni taqdim etadi.

## Talablar:
1. Asosiy kollektorlarni namoyish eting:
   1.1. toList() - List ga yig'ish
   1.2. toSet() - Set ga yig'ish
   1.3. toMap() - kalit va qiymat aks ettirichlari bilan Map ga yig'ish
   1.4. toCollection() - maxsus kolleksiya turiga yig'ish

2. groupingBy() ni namoyish eting:
   2.1. Bitta xususiyat bo'yicha guruhlash
   2.2. Downstream kollektor bilan guruhlash
   2.3. Ko'p darajali guruhlash

3. partitioningBy() ni namoyish eting:
   3.1. Predikat asosida ikki guruhga bo'lish
   3.2. Downstream kollektor bilan bo'lish

4. Satr kollektorlarini namoyish eting:
   4.1. joining() - ajratuvchi bilan satrlarni birlashtirish
   4.2. joining() prefiks va suffiks bilan

5. Statistik kollektorlarni namoyish eting:
   5.1. counting()
   5.2. summingInt()
   5.3. averagingInt()
   5.4. summarizingInt()

## Chiqish namunasi:
\`\`\`
=== Basic Collectors ===
To List: [Alice, Bob, Charlie, David]
To Set: [Alice, Bob, Charlie, David]
To Map: {Alice=5, Bob=3, Charlie=7, David=5}

=== groupingBy() ===
Group by length: {3=[Bob], 5=[Alice, David], 7=[Charlie]}
Group by length with count: {3=1, 5=2, 7=1}
Group by first letter: {A=[Alice], B=[Bob], C=[Charlie], D=[David]}

=== partitioningBy() ===
Partition by length > 4: {false=[Bob], true=[Alice, Charlie, David]}
Partition with count: {false=1, true=3}

=== String Collectors ===
Joining: Alice, Bob, Charlie, David
With brackets: [Alice, Bob, Charlie, David]

=== Statistical Collectors ===
Count: 5
Sum: 150
Average: 30.0
Statistics: IntSummaryStatistics{count=5, sum=150, min=10, average=30.0, max=50}
\`\`\``,
            hint1: `Collectors.toList() va toSet() oddiy kollektorlardir. Collectors.toMap() kalit va qiymat aks ettirishni talab qiladi. Elementlarni klassifikatsiya funksiyasi bo'yicha guruhlash uchun groupingBy() ishlatiladi.`,
            hint2: `groupingBy() ikkinchi parametr sifatida downstream kollektorni qabul qilishi mumkin. partitioningBy() predikat asosida aynan ikki guruhga bo'ladigan guruhlashtirish maxsus holati hisoblanadi.`,
            whyItMatters: `Kollektorlar stream natijalarini foydali ma'lumotlar strukturalariga o'zgartirish uchun zarur. Ular guruhlash, bo'lish va statistik tahlil kabi kuchli operatsiyalarni ta'minlaydi. Kollektorlarni o'zlashtirish, aks holda ko'p so'zli halqalar talab qiladigan murakkab ma'lumotlarni yig'ish vazifalariga qisqa kod yozishga imkon beradi.

**Ishlab chiqarish patterni:**
\`\`\`java
// Tranzaksiyalarni holat bo'yicha guruhlash va summani hisoblash
Map<TransactionStatus, BigDecimal> totalByStatus = transactions.stream()
    .collect(Collectors.groupingBy(
        Transaction::getStatus,
        Collectors.reducing(
            BigDecimal.ZERO,
            Transaction::getAmount,
            BigDecimal::add
        )
    ));
\`\`\`

**Amaliy foydalari:**
- SQL so'rovlarsiz kompleks ma'lumotlar tahlili
- Murakkab hisobotlar uchun ko'p darajali guruhlash
- Xotirada katta ma'lumotlar to'plamlarini samarali qayta ishlash`
        }
    }
};

export default task;
