import { Task } from '../../../../types';

export const task: Task = {
    slug: 'java-terminal-operations',
    title: 'Terminal Operations',
    difficulty: 'easy',
    tags: ['java', 'stream-api', 'collect', 'reduce', 'foreach', 'functional-programming'],
    estimatedTime: '30m',
    isPremium: false,
    youtubeUrl: '',
    description: `# Terminal Operations

Terminal operations produce a result or side-effect and terminate the stream. Once a terminal operation is called, the stream is consumed and cannot be reused. Common terminal operations include forEach, collect, reduce, count, findFirst, findAny, anyMatch, allMatch, and noneMatch.

## Requirements:
1. Demonstrate forEach() and forEachOrdered():
   1.1. Iterate over elements
   1.2. Perform side effects
   1.3. Show ordering differences

2. Demonstrate collect():
   2.1. Collect to List
   2.2. Collect to Set
   2.3. Collect to custom collection

3. Demonstrate reduce():
   3.1. Sum numbers
   3.2. Find maximum/minimum
   3.3. String concatenation

4. Demonstrate count(), findFirst(), findAny():
   4.1. Count elements
   4.2. Find first element
   4.3. Find any element

5. Demonstrate matching operations:
   5.1. anyMatch() - check if any element matches
   5.2. allMatch() - check if all elements match
   5.3. noneMatch() - check if no elements match

## Example Output:
\`\`\`
=== forEach() and forEachOrdered() ===
forEach: 1 2 3 4 5
forEachOrdered: 1 2 3 4 5

=== collect() Operations ===
Collect to List: [1, 2, 3, 4, 5]
Collect to Set: [1, 2, 3, 4, 5]
Even numbers: [2, 4, 6, 8, 10]

=== reduce() Operations ===
Sum: 55
Product: 3628800
Max: 10
Concatenation: Hello World from Java

=== count(), findFirst(), findAny() ===
Count: 10
First element: 1
Any element: 1

=== Matching Operations ===
Any even? true
All positive? true
None negative? true
Any > 100? false
\`\`\``,
    initialCode: `import java.util.*;
import java.util.stream.*;

public class TerminalOperations {
    public static void main(String[] args) {
        // TODO: Demonstrate forEach() and forEachOrdered()

        // TODO: Demonstrate collect()

        // TODO: Demonstrate reduce()

        // TODO: Demonstrate count(), findFirst(), findAny()

        // TODO: Demonstrate matching operations
    }
}`,
    solutionCode: `import java.util.*;
import java.util.stream.*;

public class TerminalOperations {
    public static void main(String[] args) {
        System.out.println("=== forEach() and forEachOrdered() ===");

        List<Integer> numbers = Arrays.asList(1, 2, 3, 4, 5);

        // forEach - may not preserve order in parallel streams
        System.out.print("forEach: ");
        numbers.stream().forEach(n -> System.out.print(n + " "));
        System.out.println();

        // forEachOrdered - preserves order even in parallel streams
        System.out.print("forEachOrdered: ");
        numbers.stream().forEachOrdered(n -> System.out.print(n + " "));
        System.out.println();

        System.out.println("\\n=== collect() Operations ===");

        List<Integer> range = Arrays.asList(1, 2, 3, 4, 5);

        // Collect to List
        List<Integer> list = range.stream().collect(Collectors.toList());
        System.out.println("Collect to List: " + list);

        // Collect to Set
        Set<Integer> set = range.stream().collect(Collectors.toSet());
        System.out.println("Collect to Set: " + set);

        // Filter and collect
        List<Integer> evenNumbers = Arrays.asList(1, 2, 3, 4, 5, 6, 7, 8, 9, 10).stream()
            .filter(n -> n % 2 == 0)
            .collect(Collectors.toList());
        System.out.println("Even numbers: " + evenNumbers);

        System.out.println("\\n=== reduce() Operations ===");

        List<Integer> nums = Arrays.asList(1, 2, 3, 4, 5, 6, 7, 8, 9, 10);

        // Sum using reduce
        int sum = nums.stream()
            .reduce(0, (a, b) -> a + b);
        System.out.println("Sum: " + sum);

        // Product using reduce
        int product = nums.stream()
            .reduce(1, (a, b) -> a * b);
        System.out.println("Product: " + product);

        // Find maximum
        Optional<Integer> max = nums.stream()
            .reduce((a, b) -> a > b ? a : b);
        max.ifPresent(m -> System.out.println("Max: " + m));

        // String concatenation
        List<String> words = Arrays.asList("Hello", "World", "from", "Java");
        String concatenated = words.stream()
            .reduce("", (a, b) -> a.isEmpty() ? b : a + " " + b);
        System.out.println("Concatenation: " + concatenated);

        System.out.println("\\n=== count(), findFirst(), findAny() ===");

        // Count elements
        long count = nums.stream().count();
        System.out.println("Count: " + count);

        // Find first element
        Optional<Integer> first = nums.stream().findFirst();
        first.ifPresent(f -> System.out.println("First element: " + f));

        // Find any element
        Optional<Integer> any = nums.stream().findAny();
        any.ifPresent(a -> System.out.println("Any element: " + a));

        System.out.println("\\n=== Matching Operations ===");

        // anyMatch - returns true if any element matches
        boolean anyEven = nums.stream().anyMatch(n -> n % 2 == 0);
        System.out.println("Any even? " + anyEven);

        // allMatch - returns true if all elements match
        boolean allPositive = nums.stream().allMatch(n -> n > 0);
        System.out.println("All positive? " + allPositive);

        // noneMatch - returns true if no elements match
        boolean noneNegative = nums.stream().noneMatch(n -> n < 0);
        System.out.println("None negative? " + noneNegative);

        // More complex matching
        boolean anyLarge = nums.stream().anyMatch(n -> n > 100);
        System.out.println("Any > 100? " + anyLarge);
    }
}`,
    testCode: `import static org.junit.Assert.*;
import org.junit.Test;
import java.util.*;
import java.util.stream.*;

// Test1: Verify forEach processes all elements
class Test1 {
    @Test
    public void test() {
        List<Integer> numbers = Arrays.asList(1, 2, 3);
        List<Integer> result = new ArrayList<>();
        numbers.stream().forEach(result::add);
        assertEquals(3, result.size());
    }
}

// Test2: Verify collect() to List
class Test2 {
    @Test
    public void test() {
        List<Integer> numbers = Arrays.asList(1, 2, 3, 4, 5);
        List<Integer> collected = numbers.stream().collect(Collectors.toList());
        assertEquals(5, collected.size());
    }
}

// Test3: Verify collect() to Set removes duplicates
class Test3 {
    @Test
    public void test() {
        List<Integer> numbers = Arrays.asList(1, 2, 2, 3, 3, 3);
        Set<Integer> collected = numbers.stream().collect(Collectors.toSet());
        assertEquals(3, collected.size());
    }
}

// Test4: Verify reduce() sum operation
class Test4 {
    @Test
    public void test() {
        List<Integer> numbers = Arrays.asList(1, 2, 3, 4, 5);
        int sum = numbers.stream().reduce(0, (a, b) -> a + b);
        assertEquals(15, sum);
    }
}

// Test5: Verify reduce() product operation
class Test5 {
    @Test
    public void test() {
        List<Integer> numbers = Arrays.asList(2, 3, 4);
        int product = numbers.stream().reduce(1, (a, b) -> a * b);
        assertEquals(24, product);
    }
}

// Test6: Verify count() terminal operation
class Test6 {
    @Test
    public void test() {
        List<Integer> numbers = Arrays.asList(1, 2, 3, 4, 5);
        long count = numbers.stream().filter(n -> n % 2 == 0).count();
        assertEquals(2, count);
    }
}

// Test7: Verify findFirst() returns Optional
class Test7 {
    @Test
    public void test() {
        List<Integer> numbers = Arrays.asList(1, 2, 3, 4, 5);
        Optional<Integer> first = numbers.stream().findFirst();
        assertTrue(first.isPresent());
        assertEquals(Integer.valueOf(1), first.get());
    }
}

// Test8: Verify anyMatch() returns true when match found
class Test8 {
    @Test
    public void test() {
        List<Integer> numbers = Arrays.asList(1, 2, 3, 4, 5);
        boolean anyEven = numbers.stream().anyMatch(n -> n % 2 == 0);
        assertTrue(anyEven);
    }
}

// Test9: Verify allMatch() checks all elements
class Test9 {
    @Test
    public void test() {
        List<Integer> numbers = Arrays.asList(2, 4, 6, 8);
        boolean allEven = numbers.stream().allMatch(n -> n % 2 == 0);
        assertTrue(allEven);
    }
}

// Test10: Verify noneMatch() when no matches exist
class Test10 {
    @Test
    public void test() {
        List<Integer> numbers = Arrays.asList(1, 2, 3, 4, 5);
        boolean noneNegative = numbers.stream().noneMatch(n -> n < 0);
        assertTrue(noneNegative);
    }
}
`,
    hint1: `Terminal operations trigger the execution of the stream pipeline. They produce a result (like collect, reduce) or side-effect (like forEach).`,
    hint2: `reduce() takes an identity value and a combining function. collect() gathers elements into a collection. Matching operations return boolean values.`,
    whyItMatters: `Terminal operations complete the stream pipeline and produce the final result. Understanding when and how to use each terminal operation is crucial for effective stream processing. They determine what you get from your stream pipeline and how the data is consumed.

**Production Pattern:**
\`\`\`java
// Aggregating order data with various metrics
Map<String, DoubleSummaryStatistics> salesByCategory = orders.stream()
    .collect(Collectors.groupingBy(
        Order::getCategory,
        Collectors.summarizingDouble(Order::getAmount)
    ));
boolean hasHighValueOrder = orders.stream()
    .anyMatch(order -> order.getAmount() > 10000);
\`\`\`

**Practical Benefits:**
- Powerful aggregation operations for analytics
- Efficient search with short-circuit evaluation (anyMatch, findFirst)
- Flexible ways to collect results`,
    order: 3,
    translations: {
        ru: {
            title: 'Терминальные операции',
            solutionCode: `import java.util.*;
import java.util.stream.*;

public class TerminalOperations {
    public static void main(String[] args) {
        System.out.println("=== forEach() и forEachOrdered() ===");

        List<Integer> numbers = Arrays.asList(1, 2, 3, 4, 5);

        // forEach - может не сохранять порядок в параллельных потоках
        System.out.print("forEach: ");
        numbers.stream().forEach(n -> System.out.print(n + " "));
        System.out.println();

        // forEachOrdered - сохраняет порядок даже в параллельных потоках
        System.out.print("forEachOrdered: ");
        numbers.stream().forEachOrdered(n -> System.out.print(n + " "));
        System.out.println();

        System.out.println("\\n=== Операции collect() ===");

        List<Integer> range = Arrays.asList(1, 2, 3, 4, 5);

        // Собрать в List
        List<Integer> list = range.stream().collect(Collectors.toList());
        System.out.println("Collect to List: " + list);

        // Собрать в Set
        Set<Integer> set = range.stream().collect(Collectors.toSet());
        System.out.println("Collect to Set: " + set);

        // Фильтр и сбор
        List<Integer> evenNumbers = Arrays.asList(1, 2, 3, 4, 5, 6, 7, 8, 9, 10).stream()
            .filter(n -> n % 2 == 0)
            .collect(Collectors.toList());
        System.out.println("Even numbers: " + evenNumbers);

        System.out.println("\\n=== Операции reduce() ===");

        List<Integer> nums = Arrays.asList(1, 2, 3, 4, 5, 6, 7, 8, 9, 10);

        // Сумма используя reduce
        int sum = nums.stream()
            .reduce(0, (a, b) -> a + b);
        System.out.println("Sum: " + sum);

        // Произведение используя reduce
        int product = nums.stream()
            .reduce(1, (a, b) -> a * b);
        System.out.println("Product: " + product);

        // Найти максимум
        Optional<Integer> max = nums.stream()
            .reduce((a, b) -> a > b ? a : b);
        max.ifPresent(m -> System.out.println("Max: " + m));

        // Конкатенация строк
        List<String> words = Arrays.asList("Hello", "World", "from", "Java");
        String concatenated = words.stream()
            .reduce("", (a, b) -> a.isEmpty() ? b : a + " " + b);
        System.out.println("Concatenation: " + concatenated);

        System.out.println("\\n=== count(), findFirst(), findAny() ===");

        // Подсчет элементов
        long count = nums.stream().count();
        System.out.println("Count: " + count);

        // Найти первый элемент
        Optional<Integer> first = nums.stream().findFirst();
        first.ifPresent(f -> System.out.println("First element: " + f));

        // Найти любой элемент
        Optional<Integer> any = nums.stream().findAny();
        any.ifPresent(a -> System.out.println("Any element: " + a));

        System.out.println("\\n=== Операции сопоставления ===");

        // anyMatch - возвращает true, если любой элемент соответствует
        boolean anyEven = nums.stream().anyMatch(n -> n % 2 == 0);
        System.out.println("Any even? " + anyEven);

        // allMatch - возвращает true, если все элементы соответствуют
        boolean allPositive = nums.stream().allMatch(n -> n > 0);
        System.out.println("All positive? " + allPositive);

        // noneMatch - возвращает true, если ни один элемент не соответствует
        boolean noneNegative = nums.stream().noneMatch(n -> n < 0);
        System.out.println("None negative? " + noneNegative);

        // Более сложное сопоставление
        boolean anyLarge = nums.stream().anyMatch(n -> n > 100);
        System.out.println("Any > 100? " + anyLarge);
    }
}`,
            description: `# Терминальные операции

Терминальные операции производят результат или побочный эффект и завершают поток. После вызова терминальной операции поток потребляется и не может быть повторно использован. Распространенные терминальные операции включают forEach, collect, reduce, count, findFirst, findAny, anyMatch, allMatch и noneMatch.

## Требования:
1. Продемонстрируйте forEach() и forEachOrdered():
   1.1. Итерация по элементам
   1.2. Выполнение побочных эффектов
   1.3. Показать различия в порядке

2. Продемонстрируйте collect():
   2.1. Сбор в List
   2.2. Сбор в Set
   2.3. Сбор в пользовательскую коллекцию

3. Продемонстрируйте reduce():
   3.1. Сумма чисел
   3.2. Поиск максимума/минимума
   3.3. Конкатенация строк

4. Продемонстрируйте count(), findFirst(), findAny():
   4.1. Подсчет элементов
   4.2. Поиск первого элемента
   4.3. Поиск любого элемента

5. Продемонстрируйте операции сопоставления:
   5.1. anyMatch() - проверить, соответствует ли любой элемент
   5.2. allMatch() - проверить, соответствуют ли все элементы
   5.3. noneMatch() - проверить, не соответствует ли ни один элемент

## Пример вывода:
\`\`\`
=== forEach() and forEachOrdered() ===
forEach: 1 2 3 4 5
forEachOrdered: 1 2 3 4 5

=== collect() Operations ===
Collect to List: [1, 2, 3, 4, 5]
Collect to Set: [1, 2, 3, 4, 5]
Even numbers: [2, 4, 6, 8, 10]

=== reduce() Operations ===
Sum: 55
Product: 3628800
Max: 10
Concatenation: Hello World from Java

=== count(), findFirst(), findAny() ===
Count: 10
First element: 1
Any element: 1

=== Matching Operations ===
Any even? true
All positive? true
None negative? true
Any > 100? false
\`\`\``,
            hint1: `Терминальные операции запускают выполнение конвейера потока. Они производят результат (как collect, reduce) или побочный эффект (как forEach).`,
            hint2: `reduce() принимает значение идентичности и комбинирующую функцию. collect() собирает элементы в коллекцию. Операции сопоставления возвращают булевы значения.`,
            whyItMatters: `Терминальные операции завершают конвейер потока и производят конечный результат. Понимание, когда и как использовать каждую терминальную операцию, имеет решающее значение для эффективной обработки потоков. Они определяют, что вы получаете от конвейера потока и как потребляются данные.

**Продакшен паттерн:**
\`\`\`java
// Агрегация данных заказов с различными метриками
Map<String, DoubleSummaryStatistics> salesByCategory = orders.stream()
    .collect(Collectors.groupingBy(
        Order::getCategory,
        Collectors.summarizingDouble(Order::getAmount)
    ));
boolean hasHighValueOrder = orders.stream()
    .anyMatch(order -> order.getAmount() > 10000);
\`\`\`

**Практические преимущества:**
- Мощные операции агрегации для аналитики
- Эффективный поиск с короткой оценкой (anyMatch, findFirst)
- Гибкие способы сбора результатов`
        },
        uz: {
            title: 'Yakuniy operatsiyalar',
            solutionCode: `import java.util.*;
import java.util.stream.*;

public class TerminalOperations {
    public static void main(String[] args) {
        System.out.println("=== forEach() va forEachOrdered() ===");

        List<Integer> numbers = Arrays.asList(1, 2, 3, 4, 5);

        // forEach - parallel streamlarda tartibni saqlamasligi mumkin
        System.out.print("forEach: ");
        numbers.stream().forEach(n -> System.out.print(n + " "));
        System.out.println();

        // forEachOrdered - parallel streamlarda ham tartibni saqlaydi
        System.out.print("forEachOrdered: ");
        numbers.stream().forEachOrdered(n -> System.out.print(n + " "));
        System.out.println();

        System.out.println("\\n=== collect() operatsiyalari ===");

        List<Integer> range = Arrays.asList(1, 2, 3, 4, 5);

        // List ga yig'ish
        List<Integer> list = range.stream().collect(Collectors.toList());
        System.out.println("Collect to List: " + list);

        // Set ga yig'ish
        Set<Integer> set = range.stream().collect(Collectors.toSet());
        System.out.println("Collect to Set: " + set);

        // Filtrlash va yig'ish
        List<Integer> evenNumbers = Arrays.asList(1, 2, 3, 4, 5, 6, 7, 8, 9, 10).stream()
            .filter(n -> n % 2 == 0)
            .collect(Collectors.toList());
        System.out.println("Even numbers: " + evenNumbers);

        System.out.println("\\n=== reduce() operatsiyalari ===");

        List<Integer> nums = Arrays.asList(1, 2, 3, 4, 5, 6, 7, 8, 9, 10);

        // reduce yordamida yig'indi
        int sum = nums.stream()
            .reduce(0, (a, b) -> a + b);
        System.out.println("Sum: " + sum);

        // reduce yordamida ko'paytma
        int product = nums.stream()
            .reduce(1, (a, b) -> a * b);
        System.out.println("Product: " + product);

        // Maksimumni topish
        Optional<Integer> max = nums.stream()
            .reduce((a, b) -> a > b ? a : b);
        max.ifPresent(m -> System.out.println("Max: " + m));

        // Satrlarni birlashtirish
        List<String> words = Arrays.asList("Hello", "World", "from", "Java");
        String concatenated = words.stream()
            .reduce("", (a, b) -> a.isEmpty() ? b : a + " " + b);
        System.out.println("Concatenation: " + concatenated);

        System.out.println("\\n=== count(), findFirst(), findAny() ===");

        // Elementlarni hisoblash
        long count = nums.stream().count();
        System.out.println("Count: " + count);

        // Birinchi elementni topish
        Optional<Integer> first = nums.stream().findFirst();
        first.ifPresent(f -> System.out.println("First element: " + f));

        // Har qanday elementni topish
        Optional<Integer> any = nums.stream().findAny();
        any.ifPresent(a -> System.out.println("Any element: " + a));

        System.out.println("\\n=== Moslashtirish operatsiyalari ===");

        // anyMatch - biron element mos kelsa true qaytaradi
        boolean anyEven = nums.stream().anyMatch(n -> n % 2 == 0);
        System.out.println("Any even? " + anyEven);

        // allMatch - barcha elementlar mos kelsa true qaytaradi
        boolean allPositive = nums.stream().allMatch(n -> n > 0);
        System.out.println("All positive? " + allPositive);

        // noneMatch - hech bir element mos kelmasa true qaytaradi
        boolean noneNegative = nums.stream().noneMatch(n -> n < 0);
        System.out.println("None negative? " + noneNegative);

        // Murakkabroq moslashtirish
        boolean anyLarge = nums.stream().anyMatch(n -> n > 100);
        System.out.println("Any > 100? " + anyLarge);
    }
}`,
            description: `# Yakuniy operatsiyalar

Yakuniy operatsiyalar natija yoki yon ta'sir ishlab chiqaradi va streamni tugatadi. Yakuniy operatsiya chaqirilgandan keyin stream iste'mol qilinadi va qayta ishlatilishi mumkin emas. Keng tarqalgan yakuniy operatsiyalar forEach, collect, reduce, count, findFirst, findAny, anyMatch, allMatch va noneMatch kiradi.

## Talablar:
1. forEach() va forEachOrdered() ni namoyish eting:
   1.1. Elementlar bo'ylab iteratsiya
   1.2. Yon ta'sirlarni bajarish
   1.3. Tartib farqlarini ko'rsatish

2. collect() ni namoyish eting:
   2.1. List ga yig'ish
   2.2. Set ga yig'ish
   2.3. Maxsus kolleksiyaga yig'ish

3. reduce() ni namoyish eting:
   3.1. Sonlar yig'indisi
   3.2. Maksimum/minimumni topish
   3.3. Satrlarni birlashtirish

4. count(), findFirst(), findAny() ni namoyish eting:
   4.1. Elementlarni hisoblash
   4.2. Birinchi elementni topish
   4.3. Har qanday elementni topish

5. Moslashtirish operatsiyalarini namoyish eting:
   5.1. anyMatch() - biron element mos kelishini tekshirish
   5.2. allMatch() - barcha elementlar mos kelishini tekshirish
   5.3. noneMatch() - hech bir element mos kelmasligini tekshirish

## Chiqish namunasi:
\`\`\`
=== forEach() and forEachOrdered() ===
forEach: 1 2 3 4 5
forEachOrdered: 1 2 3 4 5

=== collect() Operations ===
Collect to List: [1, 2, 3, 4, 5]
Collect to Set: [1, 2, 3, 4, 5]
Even numbers: [2, 4, 6, 8, 10]

=== reduce() Operations ===
Sum: 55
Product: 3628800
Max: 10
Concatenation: Hello World from Java

=== count(), findFirst(), findAny() ===
Count: 10
First element: 1
Any element: 1

=== Matching Operations ===
Any even? true
All positive? true
None negative? true
Any > 100? false
\`\`\``,
            hint1: `Yakuniy operatsiyalar stream konveyerining bajarilishini boshlaydi. Ular natija (collect, reduce kabi) yoki yon ta'sir (forEach kabi) ishlab chiqaradi.`,
            hint2: `reduce() identifikatsiya qiymatini va birlashtiradigan funksiyani qabul qiladi. collect() elementlarni kolleksiyaga yig'adi. Moslashtirish operatsiyalari mantiqiy qiymatlar qaytaradi.`,
            whyItMatters: `Yakuniy operatsiyalar stream konveyerini yakunlaydi va yakuniy natijani ishlab chiqaradi. Har bir yakuniy operatsiyani qachon va qanday ishlatishni tushunish samarali stream qayta ishlash uchun muhimdir. Ular stream konveyeridan nima olishingiz va ma'lumotlar qanday iste'mol qilinishini belgilaydi.

**Ishlab chiqarish patterni:**
\`\`\`java
// Buyurtmalar ma'lumotlarini turli metrikalar bilan agregatsiya qilish
Map<String, DoubleSummaryStatistics> salesByCategory = orders.stream()
    .collect(Collectors.groupingBy(
        Order::getCategory,
        Collectors.summarizingDouble(Order::getAmount)
    ));
boolean hasHighValueOrder = orders.stream()
    .anyMatch(order -> order.getAmount() > 10000);
\`\`\`

**Amaliy foydalari:**
- Analitika uchun kuchli agregatsiya operatsiyalari
- Qisqa baholash bilan samarali qidiruv (anyMatch, findFirst)
- Natijalarni yig'ishning moslashuvchan usullari`
        }
    }
};

export default task;
